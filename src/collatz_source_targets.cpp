#include "collatz/core.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::string oeis_stopping;
    std::string oeis_path_records;
    std::string output = "data/source_validation/public_source_targets.csv";
    std::string metadata;
    std::uint64_t max_n = 100000;
    std::uint32_t max_steps = 10000000;
    std::size_t stopping_limit = 5000;
    std::size_t path_record_limit = 100;
};

struct Counters {
    std::size_t stopping_rows_read = 0;
    std::size_t stopping_rows_written = 0;
    std::size_t stopping_rows_skipped_above_max_n = 0;
    std::size_t stopping_rows_bad = 0;
    std::size_t stopping_rows_mismatch = 0;
    std::size_t path_record_rows_read = 0;
    std::size_t path_record_rows_written = 0;
    std::size_t path_record_rows_skipped_above_max_n = 0;
    std::size_t path_record_rows_bad = 0;
    std::size_t skipped_overflow_or_unfinished = 0;
};

void usage(std::ostream &out) {
    out << "usage: collatz_source_targets [options]\n\n"
        << "Build a public source-validation target CSV from OEIS b-files.\n\n"
        << "options:\n"
        << "  --oeis-stopping FILE       OEIS A006577 b-file (n total_steps)\n"
        << "  --oeis-path-records FILE   OEIS A006884 b-file (index n)\n"
        << "  --output FILE              output CSV (default data/source_validation/public_source_targets.csv)\n"
        << "  --metadata FILE            metadata JSON (default OUTPUT.metadata.json)\n"
        << "  --max-n N                  keep only starts <= N (default 100000)\n"
        << "  --max-steps N              Collatz compute guard (default 10000000)\n"
        << "  --stopping-limit N         max A006577 rows to write (default 5000)\n"
        << "  --path-record-limit N      max A006884 rows to write (default 100)\n";
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
        if (arg == "--oeis-stopping") {
            options.oeis_stopping = need_value("--oeis-stopping");
        } else if (arg == "--oeis-path-records") {
            options.oeis_path_records = need_value("--oeis-path-records");
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--metadata") {
            options.metadata = need_value("--metadata");
        } else if (arg == "--max-n") {
            const auto value = collatz::parse_u64(need_value("--max-n"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-n must be a positive integer");
            }
            options.max_n = *value;
        } else if (arg == "--max-steps") {
            const auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--stopping-limit") {
            const auto value = collatz::parse_u64(need_value("--stopping-limit"));
            if (!value) {
                throw std::runtime_error("--stopping-limit must be a non-negative integer");
            }
            options.stopping_limit = static_cast<std::size_t>(std::min<std::uint64_t>(*value, std::numeric_limits<std::size_t>::max()));
        } else if (arg == "--path-record-limit") {
            const auto value = collatz::parse_u64(need_value("--path-record-limit"));
            if (!value) {
                throw std::runtime_error("--path-record-limit must be a non-negative integer");
            }
            options.path_record_limit = static_cast<std::size_t>(std::min<std::uint64_t>(*value, std::numeric_limits<std::size_t>::max()));
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.oeis_stopping.empty() && options.oeis_path_records.empty()) {
        throw std::runtime_error("at least one OEIS input file is required");
    }
    if (options.metadata.empty()) {
        options.metadata = options.output + ".metadata.json";
    }
    return options;
}

bool comment_or_blank(const std::string &line) {
    for (const char ch : line) {
        if (ch == '#') {
            return true;
        }
        if (ch != ' ' && ch != '\t' && ch != '\r') {
            return false;
        }
    }
    return true;
}

bool read_two_u64(const std::string &line, std::uint64_t &left, std::uint64_t &right) {
    std::string left_text;
    std::string right_text;
    std::stringstream stream(line);
    if (!(stream >> left_text >> right_text)) {
        return false;
    }
    const auto parsed_left = collatz::parse_u64(left_text);
    const auto parsed_right = collatz::parse_u64(right_text);
    if (!parsed_left || !parsed_right) {
        return false;
    }
    left = *parsed_left;
    right = *parsed_right;
    return true;
}

bool feature_is_usable(const collatz::FeatureRow &feature) {
    return (feature.flags & collatz::FeatureReachedOne) != 0 && (feature.flags & collatz::FeatureOverflow) == 0 &&
           (feature.flags & collatz::FeatureMaxSteps) == 0 && feature.peak.high == 0;
}

void write_target(std::ofstream &out, std::string_view source, const collatz::FeatureRow &feature) {
    out << source << ',' << feature.n << ',' << feature.total_steps << ',' << feature.peak.low << '\n';
}

void process_stopping_file(const Options &options, std::ofstream &out, Counters &counters) {
    if (options.oeis_stopping.empty() || options.stopping_limit == 0) {
        return;
    }
    std::ifstream in(options.oeis_stopping);
    if (!in) {
        throw std::runtime_error("failed to open OEIS stopping file: " + options.oeis_stopping);
    }

    std::string line;
    while (std::getline(in, line) && counters.stopping_rows_written < options.stopping_limit) {
        if (comment_or_blank(line)) {
            continue;
        }
        ++counters.stopping_rows_read;
        std::uint64_t n = 0;
        std::uint64_t steps = 0;
        if (!read_two_u64(line, n, steps) || n == 0 || steps > std::numeric_limits<std::uint32_t>::max()) {
            ++counters.stopping_rows_bad;
            continue;
        }
        if (n > options.max_n) {
            ++counters.stopping_rows_skipped_above_max_n;
            continue;
        }

        const auto feature = collatz::compute_feature(n, options.max_steps);
        if (!feature_is_usable(feature)) {
            ++counters.skipped_overflow_or_unfinished;
            continue;
        }
        if (feature.total_steps != static_cast<std::uint32_t>(steps)) {
            ++counters.stopping_rows_mismatch;
            continue;
        }

        write_target(out, "OEIS_A006577_total_stopping_time", feature);
        ++counters.stopping_rows_written;
    }
}

void process_path_record_file(const Options &options, std::ofstream &out, Counters &counters) {
    if (options.oeis_path_records.empty() || options.path_record_limit == 0) {
        return;
    }
    std::ifstream in(options.oeis_path_records);
    if (!in) {
        throw std::runtime_error("failed to open OEIS path-record file: " + options.oeis_path_records);
    }

    std::string line;
    while (std::getline(in, line) && counters.path_record_rows_written < options.path_record_limit) {
        if (comment_or_blank(line)) {
            continue;
        }
        ++counters.path_record_rows_read;
        std::uint64_t index = 0;
        std::uint64_t n = 0;
        if (!read_two_u64(line, index, n) || index == 0 || n == 0) {
            ++counters.path_record_rows_bad;
            continue;
        }
        if (n > options.max_n) {
            ++counters.path_record_rows_skipped_above_max_n;
            continue;
        }

        const auto feature = collatz::compute_feature(n, options.max_steps);
        if (!feature_is_usable(feature)) {
            ++counters.skipped_overflow_or_unfinished;
            continue;
        }

        write_target(out, "OEIS_A006884_path_record", feature);
        ++counters.path_record_rows_written;
    }
}

void write_metadata(const Options &options, const Counters &counters) {
    collatz::ensure_parent_dir(options.metadata);
    std::ofstream out(options.metadata);
    if (!out) {
        throw std::runtime_error("failed to open metadata output: " + options.metadata);
    }
    out << "{\n"
        << "  \"dataset_type\": \"collatz_public_source_targets\",\n"
        << "  \"tool\": \"collatz_source_targets\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"output\": \"" << collatz::json_escape(options.output) << "\",\n"
        << "  \"max_n\": " << options.max_n << ",\n"
        << "  \"max_steps\": " << options.max_steps << ",\n"
        << "  \"sources\": [\n"
        << "    {\"id\":\"OEIS_A006577\",\"url\":\"https://oeis.org/A006577/b006577.txt\",\"role\":\"total stopping-time validation\"},\n"
        << "    {\"id\":\"OEIS_A006884\",\"url\":\"https://oeis.org/A006884/b006884.txt\",\"role\":\"path-record validation\"}\n"
        << "  ],\n"
        << "  \"rows_written\": " << (counters.stopping_rows_written + counters.path_record_rows_written) << ",\n"
        << "  \"stopping_rows_read\": " << counters.stopping_rows_read << ",\n"
        << "  \"stopping_rows_written\": " << counters.stopping_rows_written << ",\n"
        << "  \"stopping_rows_skipped_above_max_n\": " << counters.stopping_rows_skipped_above_max_n << ",\n"
        << "  \"stopping_rows_bad\": " << counters.stopping_rows_bad << ",\n"
        << "  \"stopping_rows_mismatch\": " << counters.stopping_rows_mismatch << ",\n"
        << "  \"path_record_rows_read\": " << counters.path_record_rows_read << ",\n"
        << "  \"path_record_rows_written\": " << counters.path_record_rows_written << ",\n"
        << "  \"path_record_rows_skipped_above_max_n\": " << counters.path_record_rows_skipped_above_max_n << ",\n"
        << "  \"path_record_rows_bad\": " << counters.path_record_rows_bad << ",\n"
        << "  \"skipped_overflow_or_unfinished\": " << counters.skipped_overflow_or_unfinished << "\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        collatz::ensure_parent_dir(options.output);
        std::ofstream out(options.output);
        if (!out) {
            throw std::runtime_error("failed to open output: " + options.output);
        }
        out << "source,n,total_steps,peak_low\n";

        Counters counters;
        process_stopping_file(options, out, counters);
        process_path_record_file(options, out, counters);
        out.close();
        write_metadata(options, counters);

        const auto rows_written = counters.stopping_rows_written + counters.path_record_rows_written;
        std::cout << "source_targets=" << rows_written
                  << " stopping=" << counters.stopping_rows_written
                  << " path_records=" << counters.path_record_rows_written
                  << " output=" << options.output
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
