#include "collatz/feature_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string output = "data/generated/representatives.csv";
    std::size_t top = 64;
    std::size_t residue_top = 4;
    std::uint64_t stride = 1000000;
    std::uint64_t limit = 0;
};

struct ScoredRecord {
    double score = 0.0;
    collatz::BinaryFeatureRecord record = {};
};

struct SelectedRecord {
    std::string category;
    collatz::BinaryFeatureRecord record = {};
};

void usage(std::ostream &out) {
    out << "usage: collatz_select_representatives --input FILE [options]\n\n"
        << "options:\n"
        << "  --output FILE       representatives CSV (default data/generated/representatives.csv)\n"
        << "  --top N             global top count per score family (default 64)\n"
        << "  --residue-top N     per residue_mod32 top total-step count (default 4)\n"
        << "  --stride N          deterministic sample stride by n, 0 disables (default 1000000)\n"
        << "  --limit N           max binary records to read, 0 means all records\n";
}

std::size_t parse_size_arg(const std::string &value, const char *name) {
    const auto parsed = collatz::parse_u64(value);
    if (!parsed) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    return static_cast<std::size_t>(*parsed);
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

        if (arg == "--input") {
            options.input = need_value("--input");
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--top") {
            options.top = parse_size_arg(need_value("--top"), "--top");
        } else if (arg == "--residue-top") {
            options.residue_top = parse_size_arg(need_value("--residue-top"), "--residue-top");
        } else if (arg == "--stride") {
            const auto value = collatz::parse_u64(need_value("--stride"));
            if (!value) {
                throw std::runtime_error("--stride must be an integer");
            }
            options.stride = *value;
        } else if (arg == "--limit") {
            const auto value = collatz::parse_u64(need_value("--limit"));
            if (!value) {
                throw std::runtime_error("--limit must be an integer");
            }
            options.limit = *value;
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    return options;
}

void consider_top(std::vector<ScoredRecord> &records, const collatz::BinaryFeatureRecord &record, double score, std::size_t keep) {
    if (keep == 0) {
        return;
    }
    if (records.size() < keep) {
        records.push_back({score, record});
        return;
    }
    auto min_it = std::min_element(records.begin(), records.end(), [](const auto &left, const auto &right) {
        return left.score < right.score;
    });
    if (min_it != records.end() && score > min_it->score) {
        *min_it = {score, record};
    }
}

void append_category(std::vector<SelectedRecord> &out, const std::string &category, std::vector<ScoredRecord> records) {
    std::sort(records.begin(), records.end(), [](const auto &left, const auto &right) {
        if (left.score == right.score) {
            return left.record.n < right.record.n;
        }
        return left.score > right.score;
    });
    for (const auto &record : records) {
        out.push_back({category, record.record});
    }
}

void write_metadata(
    const Options &options,
    const collatz::BinaryFeatureHeader &header,
    std::uint64_t records_read,
    std::uint64_t selected_rows) {
    const std::string metadata = options.output + ".metadata.json";
    collatz::ensure_parent_dir(metadata);
    std::ofstream out(metadata);
    if (!out) {
        throw std::runtime_error("failed to open metadata output: " + metadata);
    }
    out << "{\n"
        << "  \"dataset_type\": \"collatz_representatives\",\n"
        << "  \"tool\": \"collatz_select_representatives\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"input\": \"" << collatz::json_escape(options.input) << "\",\n"
        << "  \"input_range_start\": " << header.range_start << ",\n"
        << "  \"input_range_end\": " << header.range_end << ",\n"
        << "  \"input_records_read\": " << records_read << ",\n"
        << "  \"selected_rows\": " << selected_rows << ",\n"
        << "  \"top\": " << options.top << ",\n"
        << "  \"residue_top\": " << options.residue_top << ",\n"
        << "  \"stride\": " << options.stride << ",\n"
        << "  \"feature_schema_version\": " << collatz::kFeatureVersion << ",\n"
        << "  \"binary_feature_version\": " << collatz::kBinaryFeatureVersion << "\n"
        << "}\n";
}

void write_csv(const Options &options, const std::vector<SelectedRecord> &selected) {
    collatz::ensure_parent_dir(options.output);
    std::ofstream out(options.output);
    if (!out) {
        throw std::runtime_error("failed to open output: " + options.output);
    }
    out << std::setprecision(10);
    out << "category,n,total_steps,first_drop_time,odd_steps,even_steps,peak_step,peak_log2,peak_ratio_log2,residue_mod32,flags,checksum\n";
    for (const auto &item : selected) {
        const auto &record = item.record;
        out << item.category << ','
            << record.n << ','
            << record.total_steps << ','
            << record.first_drop_time << ','
            << record.odd_steps << ','
            << record.even_steps << ','
            << record.peak_step << ','
            << record.peak_log2 << ','
            << record.peak_ratio_log2 << ','
            << static_cast<unsigned>(record.residue_mod32) << ','
            << record.flags << ','
            << record.checksum << '\n';
    }
}

std::vector<SelectedRecord> dedupe_selected(const std::vector<SelectedRecord> &input) {
    std::unordered_set<std::uint64_t> seen;
    std::vector<SelectedRecord> out;
    out.reserve(input.size());
    for (const auto &item : input) {
        if (seen.insert(item.record.n).second) {
            out.push_back(item);
        }
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto header = collatz::read_binary_header(options.input);
        if (!collatz::valid_binary_header(header)) {
            throw std::runtime_error("input has an invalid binary feature header");
        }

        std::ifstream in(options.input, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open input: " + options.input);
        }
        in.seekg(static_cast<std::streamoff>(sizeof(collatz::BinaryFeatureHeader)));

        const auto available = collatz::binary_record_count(options.input);
        const auto wanted = options.limit == 0 ? available : std::min<std::uint64_t>(available, options.limit);
        std::vector<ScoredRecord> top_steps;
        std::vector<ScoredRecord> top_peak_log2;
        std::vector<ScoredRecord> top_peak_ratio;
        std::vector<ScoredRecord> deterministic_samples;
        std::array<std::vector<ScoredRecord>, 32> residue_steps;

        std::uint64_t records_read = 0;
        for (; records_read < wanted; ++records_read) {
            collatz::BinaryFeatureRecord record{};
            in.read(reinterpret_cast<char *>(&record), sizeof(record));
            if (!in) {
                throw std::runtime_error("failed while reading binary feature record");
            }

            consider_top(top_steps, record, static_cast<double>(record.total_steps), options.top);
            consider_top(top_peak_log2, record, record.peak_log2, options.top);
            consider_top(top_peak_ratio, record, record.peak_ratio_log2, options.top);
            consider_top(residue_steps[record.residue_mod32], record, static_cast<double>(record.total_steps), options.residue_top);
            if (options.stride != 0 && record.n % options.stride == 0) {
                deterministic_samples.push_back({static_cast<double>(record.n), record});
            }
        }

        std::vector<SelectedRecord> selected;
        append_category(selected, "top_total_steps", top_steps);
        append_category(selected, "top_peak_log2", top_peak_log2);
        append_category(selected, "top_peak_ratio_log2", top_peak_ratio);
        for (std::size_t residue = 0; residue < residue_steps.size(); ++residue) {
            append_category(selected, "residue_mod32_" + std::to_string(residue) + "_top_steps", residue_steps[residue]);
        }
        append_category(selected, "stride_sample", deterministic_samples);
        selected = dedupe_selected(selected);

        write_csv(options, selected);
        write_metadata(options, header, records_read, selected.size());
        std::cout << "records_read=" << records_read
                  << " selected_rows=" << selected.size()
                  << " output=" << options.output
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
