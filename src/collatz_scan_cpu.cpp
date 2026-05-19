#include "collatz/core.hpp"
#include "collatz/feature_io.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

enum class OutputFormat {
    Csv,
    Binary,
};

struct Options {
    std::uint64_t start = 1;
    std::uint64_t end = 0;
    std::uint64_t chunk_size = 100000;
    std::uint32_t max_steps = 10000000;
    std::uint32_t threads = 0;
    std::string output = "data/generated/features.csv";
    std::string progress = "logs/progress.jsonl";
    std::string metadata;
    std::string mode_label = "cpu";
    OutputFormat format = OutputFormat::Csv;
    bool resume = false;
};

void usage(std::ostream &out) {
    out << "usage: collatz_scan_cpu --start N --end N [options]\n\n"
        << "options:\n"
        << "  --output FILE       CSV feature output (default data/generated/features.csv)\n"
        << "  --progress FILE     JSONL progress output (default logs/progress.jsonl)\n"
        << "  --metadata FILE     dataset metadata sidecar (default OUTPUT.metadata.json)\n"
        << "  --chunk-size N      progress interval (default 100000)\n"
        << "  --max-steps N       per-start safety cap (default 10000000)\n"
        << "  --threads N         worker threads, default hardware concurrency\n"
        << "  --format csv|bin    output format, default csv\n"
        << "  --resume            append after the last n already present in output\n"
        << "  --mode-label TEXT   label written to progress records (default cpu)\n";
}

Options parse_args(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--start") {
            auto value = collatz::parse_u64(need_value("--start"));
            if (!value || *value == 0) {
                throw std::runtime_error("--start must be a positive integer");
            }
            options.start = *value;
        } else if (arg == "--end") {
            auto value = collatz::parse_u64(need_value("--end"));
            if (!value || *value == 0) {
                throw std::runtime_error("--end must be a positive integer");
            }
            options.end = *value;
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--progress") {
            options.progress = need_value("--progress");
        } else if (arg == "--metadata") {
            options.metadata = need_value("--metadata");
        } else if (arg == "--chunk-size") {
            auto value = collatz::parse_u64(need_value("--chunk-size"));
            if (!value || *value == 0) {
                throw std::runtime_error("--chunk-size must be a positive integer");
            }
            options.chunk_size = *value;
        } else if (arg == "--max-steps") {
            auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--threads") {
            auto value = collatz::parse_u32(need_value("--threads"));
            if (!value || *value == 0) {
                throw std::runtime_error("--threads must be a positive integer");
            }
            options.threads = *value;
        } else if (arg == "--format") {
            const std::string value = need_value("--format");
            if (value == "csv") {
                options.format = OutputFormat::Csv;
            } else if (value == "bin") {
                options.format = OutputFormat::Binary;
            } else {
                throw std::runtime_error("--format must be csv or bin");
            }
        } else if (arg == "--mode-label") {
            options.mode_label = need_value("--mode-label");
        } else if (arg == "--resume") {
            options.resume = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (options.end == 0) {
        throw std::runtime_error("--end is required");
    }
    if (options.start > options.end) {
        throw std::runtime_error("--start cannot be greater than --end");
    }
    if (options.threads == 0) {
        options.threads = std::max(1u, std::thread::hardware_concurrency());
    }
    if (options.metadata.empty()) {
        options.metadata = options.output + ".metadata.json";
    }
    return options;
}

struct ChunkResult {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t processed = 0;
    std::uint64_t max_steps_n = 0;
    std::uint32_t max_steps_value = 0;
    std::uint64_t max_peak_n = 0;
    long double max_peak_log2 = 0.0L;
    std::string rows;
    std::vector<collatz::BinaryFeatureRecord> records;
    std::uint64_t checksum = 14695981039346656037ull;
};

const char *format_name(OutputFormat format) {
    return format == OutputFormat::Csv ? "csv" : "bin";
}

void mix_checksum(std::uint64_t &hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        hash ^= static_cast<unsigned char>((value >> (i * 8u)) & 0xffu);
        hash *= 1099511628211ull;
    }
}

void sync_binary_header_if_needed(const Options &options, std::uint64_t requested_start) {
    if (options.format != OutputFormat::Binary || !std::filesystem::exists(options.output)) {
        return;
    }
    const auto header = collatz::read_binary_header(options.output);
    if (!collatz::valid_binary_header(header)) {
        throw std::runtime_error("binary feature file has an incompatible header: " + options.output);
    }
    if (header.range_start != requested_start) {
        throw std::runtime_error("binary output range_start does not match requested --start");
    }
    if (header.max_steps != options.max_steps) {
        throw std::runtime_error("binary resume requires the same --max-steps used to create the file");
    }
    if (header.range_end != options.end) {
        collatz::update_binary_header(options.output, options.end, options.max_steps);
    }
}

std::uint64_t last_output_n_from_csv(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        return 0;
    }

    std::string line;
    std::string last;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            last = line;
        }
    }
    if (last.empty() || last.rfind("n,", 0) == 0) {
        return 0;
    }
    const auto comma = last.find(',');
    if (comma == std::string::npos) {
        return 0;
    }
    return collatz::parse_u64(std::string_view(last).substr(0, comma)).value_or(0);
}

void append_progress(
    const Options &options,
    std::uint64_t current,
    std::uint64_t processed,
    double throughput,
    std::uint64_t max_steps_n,
    std::uint32_t max_steps_value,
    std::uint64_t max_peak_n,
    long double max_peak_log2) {
    collatz::ensure_parent_dir(options.progress);
    std::ofstream out(options.progress, std::ios::app);
    out << "{\"type\":\"scan_progress\""
        << ",\"timestamp\":\"" << collatz::now_iso8601() << "\""
        << ",\"mode\":\"" << collatz::json_escape(options.mode_label) << "\""
        << ",\"threads\":" << options.threads
        << ",\"format\":\"" << (options.format == OutputFormat::Csv ? "csv" : "bin") << "\""
        << ",\"range_start\":" << options.start
        << ",\"range_end\":" << options.end
        << ",\"current\":" << current
        << ",\"processed\":" << processed
        << ",\"throughput_per_sec\":" << throughput
        << ",\"max_total_steps_n\":" << max_steps_n
        << ",\"max_total_steps\":" << max_steps_value
        << ",\"max_peak_n\":" << max_peak_n
        << ",\"max_peak_log2\":" << static_cast<double>(max_peak_log2)
        << ",\"output\":\"" << collatz::json_escape(options.output) << "\""
        << "}\n";
}

ChunkResult scan_chunk(const Options &options, std::uint64_t begin, std::uint64_t end) {
    ChunkResult result;
    result.begin = begin;
    result.end = end;

    std::ostringstream rows;
    if (options.format == OutputFormat::Binary) {
        result.records.reserve(static_cast<std::size_t>(end - begin + 1));
    }
    for (std::uint64_t n = begin; n <= end; ++n) {
        const auto feature = collatz::compute_feature(n, options.max_steps);
        if (options.format == OutputFormat::Csv) {
            rows << collatz::feature_to_csv(feature) << '\n';
        } else {
            result.records.push_back(collatz::to_binary_record(feature));
        }

        if (feature.total_steps > result.max_steps_value) {
            result.max_steps_value = feature.total_steps;
            result.max_steps_n = n;
        }
        if (feature.peak_log2 > result.max_peak_log2) {
            result.max_peak_log2 = feature.peak_log2;
            result.max_peak_n = n;
        }

        mix_checksum(result.checksum, feature.checksum);
        ++result.processed;
        if (n == std::numeric_limits<std::uint64_t>::max()) {
            break;
        }
    }

    if (options.format == OutputFormat::Csv) {
        result.rows = std::move(rows).str();
    }
    return result;
}

void write_metadata(
    const Options &options,
    std::uint64_t requested_start,
    std::uint64_t effective_start,
    std::uint64_t processed_total,
    std::uint64_t max_steps_n,
    std::uint32_t max_steps_value,
    std::uint64_t max_peak_n,
    long double max_peak_log2,
    double throughput,
    const std::string &started_utc,
    std::uint64_t checksum) {
    collatz::ensure_parent_dir(options.metadata);
    std::ofstream out(options.metadata);
    if (!out) {
        throw std::runtime_error("failed to open metadata sidecar: " + options.metadata);
    }

    std::uint64_t observed_records = processed_total;
    if (options.format == OutputFormat::Binary && std::filesystem::exists(options.output)) {
        observed_records = collatz::binary_record_count(options.output);
    }

    std::uint64_t output_bytes = 0;
    if (std::filesystem::exists(options.output)) {
        output_bytes = std::filesystem::file_size(options.output);
    }

    out << "{\n"
        << "  \"dataset_type\": \"collatz_features\",\n"
        << "  \"scanner\": \"collatz_scan_cpu\",\n"
        << "  \"scanner_version\": \"0.1.0\",\n"
        << "  \"created_utc\": \"" << collatz::json_escape(started_utc) << "\",\n"
        << "  \"completed_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"requested_range_start\": " << requested_start << ",\n"
        << "  \"effective_range_start\": " << effective_start << ",\n"
        << "  \"range_end\": " << options.end << ",\n"
        << "  \"chunk_size\": " << options.chunk_size << ",\n"
        << "  \"max_steps\": " << options.max_steps << ",\n"
        << "  \"threads\": " << options.threads << ",\n"
        << "  \"mode\": \"" << collatz::json_escape(options.mode_label) << "\",\n"
        << "  \"format\": \"" << format_name(options.format) << "\",\n"
        << "  \"feature_schema_version\": " << collatz::kFeatureVersion << ",\n"
        << "  \"binary_feature_version\": " << collatz::kBinaryFeatureVersion << ",\n"
        << "  \"output\": \"" << collatz::json_escape(options.output) << "\",\n"
        << "  \"output_bytes\": " << output_bytes << ",\n"
        << "  \"run_records_written\": " << processed_total << ",\n"
        << "  \"dataset_records_observed\": " << observed_records << ",\n"
        << "  \"checksum_fnv1a64\": " << checksum << ",\n"
        << "  \"max_total_steps_n\": " << max_steps_n << ",\n"
        << "  \"max_total_steps\": " << max_steps_value << ",\n"
        << "  \"max_peak_n\": " << max_peak_n << ",\n"
        << "  \"max_peak_log2\": " << static_cast<double>(max_peak_log2) << ",\n"
        << "  \"throughput_per_sec\": " << throughput << ",\n"
        << "  \"validation_authorities\": [\n"
        << "    \"OEIS A006577\",\n"
        << "    \"OEIS A006884\",\n"
        << "    \"Roosendaal delay/path records\",\n"
        << "    \"Oliveira e Silva verification records\",\n"
        << "    \"Barina path records\"\n"
        << "  ]\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse_args(argc, argv);
        const std::uint64_t requested_start = options.start;
        const std::string started_utc = collatz::now_iso8601();

        if (options.resume) {
            std::uint64_t last = 0;
            if (options.format == OutputFormat::Csv) {
                last = last_output_n_from_csv(options.output);
            } else {
                last = collatz::binary_completed_through(options.output, options.start).value_or(0);
            }
            if (last >= options.start && last < options.end) {
                options.start = last + 1;
            } else if (last >= options.end) {
                sync_binary_header_if_needed(options, requested_start);
                std::cout << "range already complete through n=" << last << "\n";
                return 0;
            }
        }
        sync_binary_header_if_needed(options, requested_start);
        const std::uint64_t effective_start = options.start;

        collatz::ensure_parent_dir(options.output);
        const bool output_exists = std::filesystem::exists(options.output) && std::filesystem::file_size(options.output) > 0;
        std::ofstream output(
            options.output,
            std::ios::out | std::ios::app | (options.format == OutputFormat::Binary ? std::ios::binary : std::ios::openmode{}));
        if (!output) {
            throw std::runtime_error("failed to open output: " + options.output);
        }
        if (!output_exists && options.format == OutputFormat::Csv) {
            output << collatz::feature_csv_header() << '\n';
        } else if (!output_exists && options.format == OutputFormat::Binary) {
            collatz::write_binary_header(output, options.start, options.end, options.max_steps);
        }

        std::uint64_t processed_total = 0;
        std::uint64_t max_steps_n = 0;
        std::uint32_t max_steps_value = 0;
        std::uint64_t max_peak_n = 0;
        long double max_peak_log2 = 0.0L;
        std::uint64_t scan_checksum = 14695981039346656037ull;

        std::atomic<std::uint64_t> next_start{options.start};
        std::mutex mutex;
        std::condition_variable cv;
        std::map<std::uint64_t, ChunkResult> completed;
        std::exception_ptr first_error;

        const auto started = std::chrono::steady_clock::now();
        auto worker = [&]() {
            try {
                while (true) {
                    const std::uint64_t begin = next_start.fetch_add(options.chunk_size);
                    if (begin > options.end) {
                        break;
                    }
                    const std::uint64_t span = std::min(options.chunk_size - 1, options.end - begin);
                    const std::uint64_t end = begin + span;
                    auto result = scan_chunk(options, begin, end);
                    {
                        std::lock_guard lock(mutex);
                        completed.emplace(result.begin, std::move(result));
                    }
                    cv.notify_one();
                }
            } catch (...) {
                {
                    std::lock_guard lock(mutex);
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                }
                cv.notify_one();
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(options.threads);
        for (std::uint32_t i = 0; i < options.threads; ++i) {
            workers.emplace_back(worker);
        }

        std::uint64_t expected = options.start;
        while (expected <= options.end) {
            ChunkResult result;
            bool has_error = false;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&]() {
                    return first_error != nullptr || completed.find(expected) != completed.end();
                });
                if (first_error) {
                    has_error = true;
                } else {
                    auto it = completed.find(expected);
                    result = std::move(it->second);
                    completed.erase(it);
                }
            }
            if (has_error) {
                break;
            }

            if (options.format == OutputFormat::Csv) {
                output << result.rows;
            } else if (!result.records.empty()) {
                output.write(
                    reinterpret_cast<const char *>(result.records.data()),
                    static_cast<std::streamsize>(result.records.size() * sizeof(collatz::BinaryFeatureRecord)));
            }
            output.flush();
            processed_total += result.processed;
            mix_checksum(scan_checksum, result.checksum);
            if (result.max_steps_value > max_steps_value) {
                max_steps_value = result.max_steps_value;
                max_steps_n = result.max_steps_n;
            }
            if (result.max_peak_log2 > max_peak_log2) {
                max_peak_log2 = result.max_peak_log2;
                max_peak_n = result.max_peak_n;
            }

            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const double throughput = elapsed > 0.0 ? static_cast<double>(processed_total) / elapsed : 0.0;
            append_progress(options, result.end, processed_total, throughput, max_steps_n, max_steps_value, max_peak_n, max_peak_log2);
            std::cout << "processed=" << processed_total
                      << " current=" << result.end
                      << " threads=" << options.threads
                      << " throughput=" << throughput << "/s"
                      << " max_steps=" << max_steps_value << "@" << max_steps_n
                      << "\n";

            if (result.end == std::numeric_limits<std::uint64_t>::max()) {
                break;
            }
            expected = result.end + 1;
        }

        for (auto &thread : workers) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        if (first_error) {
            std::rethrow_exception(first_error);
        }

        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const double throughput = elapsed > 0.0 ? static_cast<double>(processed_total) / elapsed : 0.0;
        write_metadata(
            options,
            requested_start,
            effective_start,
            processed_total,
            max_steps_n,
            max_steps_value,
            max_peak_n,
            max_peak_log2,
            throughput,
            started_utc,
            scan_checksum);

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
