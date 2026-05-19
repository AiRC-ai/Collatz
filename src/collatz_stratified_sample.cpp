#include "collatz/feature_io.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string output_dir = "data/generated/stratified";
    std::string clusters;
    std::string representatives;
    std::string starts;
    std::size_t random_count = 4096;
    std::size_t global_top = 256;
    std::size_t residue_top = 16;
    std::size_t range_bands = 16;
    std::size_t range_top = 24;
    std::uint64_t limit = 0;
    std::uint64_t seed = 0x3a5f2d91c7b04913ull;
};

struct ScoredRecord {
    double score = 0.0;
    collatz::BinaryFeatureRecord record = {};
};

struct TopRecords {
    std::size_t keep = 0;
    std::vector<ScoredRecord> records;
};

struct SelectedRecord {
    collatz::BinaryFeatureRecord record = {};
    std::set<std::string> reasons;
};

void usage(std::ostream &out) {
    out << "usage: collatz_stratified_sample --input FILE [options]\n\n"
        << "options:\n"
        << "  --output-dir DIR          output directory (default data/generated/stratified)\n"
        << "  --clusters FILE          optional topology clusters.csv for loose/tight reps\n"
        << "  --representatives FILE   optional representatives.csv to preserve prior reps\n"
        << "  --starts FILE            optional graph starts.csv for GNN starts\n"
        << "  --random-count N         deterministic random baseline rows (default 4096)\n"
        << "  --global-top N           global top rows per score family (default 256)\n"
        << "  --residue-top N          top total-step rows per residue_mod32 (default 16)\n"
        << "  --range-bands N          numeric range bands (default 16)\n"
        << "  --range-top N            top rows per range band (default 24)\n"
        << "  --limit N                max binary records to read, 0 means all records\n"
        << "  --seed N                 deterministic sampling seed\n";
}

std::size_t parse_size_arg(const std::string &value, const char *name, bool allow_zero = false) {
    const auto parsed = collatz::parse_u64(value);
    if (!parsed || (!allow_zero && *parsed == 0)) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
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
        } else if (arg == "--output-dir") {
            options.output_dir = need_value("--output-dir");
        } else if (arg == "--clusters") {
            options.clusters = need_value("--clusters");
        } else if (arg == "--representatives") {
            options.representatives = need_value("--representatives");
        } else if (arg == "--starts") {
            options.starts = need_value("--starts");
        } else if (arg == "--random-count") {
            options.random_count = parse_size_arg(need_value("--random-count"), "--random-count", true);
        } else if (arg == "--global-top") {
            options.global_top = parse_size_arg(need_value("--global-top"), "--global-top", true);
        } else if (arg == "--residue-top") {
            options.residue_top = parse_size_arg(need_value("--residue-top"), "--residue-top", true);
        } else if (arg == "--range-bands") {
            options.range_bands = parse_size_arg(need_value("--range-bands"), "--range-bands", true);
        } else if (arg == "--range-top") {
            options.range_top = parse_size_arg(need_value("--range-top"), "--range-top", true);
        } else if (arg == "--limit") {
            const auto value = collatz::parse_u64(need_value("--limit"));
            if (!value) {
                throw std::runtime_error("--limit must be an integer");
            }
            options.limit = *value;
        } else if (arg == "--seed") {
            const auto value = collatz::parse_u64(need_value("--seed"));
            if (!value) {
                throw std::runtime_error("--seed must be an integer");
            }
            options.seed = *value;
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

std::string path_join(const std::string &dir, const std::string &file) {
    return (std::filesystem::path(dir) / file).string();
}

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> parts;
    std::string part;
    std::stringstream stream(line);
    while (std::getline(stream, part, ',')) {
        parts.push_back(part);
    }
    return parts;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

void mix_checksum(std::uint64_t &hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        hash ^= static_cast<unsigned char>((value >> (i * 8u)) & 0xffu);
        hash *= 1099511628211ull;
    }
}

struct MinScoreFirst {
    bool operator()(const ScoredRecord &left, const ScoredRecord &right) const {
        if (left.score == right.score) {
            return left.record.n > right.record.n;
        }
        return left.score > right.score;
    }
};

void consider_top(TopRecords &top, const collatz::BinaryFeatureRecord &record, double score) {
    if (top.keep == 0) {
        return;
    }
    if (top.records.size() < top.keep) {
        top.records.push_back({score, record});
        std::push_heap(top.records.begin(), top.records.end(), MinScoreFirst{});
        return;
    }
    if (!top.records.empty() && score > top.records.front().score) {
        std::pop_heap(top.records.begin(), top.records.end(), MinScoreFirst{});
        top.records.back() = {score, record};
        std::push_heap(top.records.begin(), top.records.end(), MinScoreFirst{});
    }
}

void append_reason(
    std::unordered_map<std::uint64_t, SelectedRecord> &selected,
    std::map<std::string, std::size_t> &reason_counts,
    const collatz::BinaryFeatureRecord &record,
    const std::string &reason) {
    auto &entry = selected[record.n];
    if (entry.record.n == 0) {
        entry.record = record;
    }
    if (entry.reasons.insert(reason).second) {
        ++reason_counts[reason];
    }
}

void append_scored(
    std::unordered_map<std::uint64_t, SelectedRecord> &selected,
    std::map<std::string, std::size_t> &reason_counts,
    std::vector<ScoredRecord> records,
    const std::string &reason) {
    std::sort(records.begin(), records.end(), [](const auto &left, const auto &right) {
        if (left.score == right.score) {
            return left.record.n < right.record.n;
        }
        return left.score > right.score;
    });
    for (const auto &item : records) {
        append_reason(selected, reason_counts, item.record, reason);
    }
}

std::string join_reasons(const std::set<std::string> &reasons) {
    std::ostringstream out;
    bool first = true;
    for (const auto &reason : reasons) {
        if (!first) {
            out << '|';
        }
        out << reason;
        first = false;
    }
    return out.str();
}

void read_n_category_file(const std::string &path, const std::string &default_reason, std::unordered_map<std::uint64_t, std::set<std::string>> &wanted) {
    if (path.empty()) {
        return;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open optional sample input: " + path);
    }
    std::string header;
    if (!std::getline(in, header)) {
        return;
    }
    const auto columns = split_csv_line(header);
    auto find_column = [&](const std::string &name) -> int {
        for (std::size_t i = 0; i < columns.size(); ++i) {
            if (columns[i] == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    const int n_index = find_column("n");
    const int category_index = find_column("category");
    const int representative_index = find_column("representative_n");
    if (n_index < 0 && representative_index < 0) {
        throw std::runtime_error("optional sample input needs n or representative_n column: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        const int value_index = n_index >= 0 ? n_index : representative_index;
        if (value_index >= static_cast<int>(parts.size())) {
            continue;
        }
        const auto n = collatz::parse_u64(parts[static_cast<std::size_t>(value_index)]);
        if (!n || *n == 0) {
            continue;
        }
        std::string reason = default_reason;
        if (category_index >= 0 && category_index < static_cast<int>(parts.size()) && !parts[static_cast<std::size_t>(category_index)].empty()) {
            reason += "_" + parts[static_cast<std::size_t>(category_index)];
        }
        wanted[*n].insert(reason);
    }
}

void read_clusters_file(const std::string &path, std::unordered_map<std::uint64_t, std::set<std::string>> &wanted) {
    if (path.empty()) {
        return;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open clusters input: " + path);
    }
    std::string line;
    if (!std::getline(in, line)) {
        return;
    }
    if (line != "cluster,count,cx,cy,representative_n,representative_distance") {
        throw std::runtime_error("clusters input has unexpected header: " + path);
    }
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() < 5) {
            continue;
        }
        const auto cluster = collatz::parse_u64(parts[0]);
        const auto n = collatz::parse_u64(parts[4]);
        if (!cluster || !n || *n == 0) {
            continue;
        }
        wanted[*n].insert("topology_cluster_" + std::to_string(*cluster) + "_representative");
    }
}

std::size_t range_band_for(std::uint64_t range_start, std::uint64_t range_end, const collatz::BinaryFeatureRecord &record, std::size_t bands) {
    if (bands == 0 || range_end <= range_start || record.n < range_start) {
        return 0;
    }
    const long double offset = static_cast<long double>(record.n - range_start);
    const long double span = static_cast<long double>(range_end - range_start + 1);
    std::size_t band = static_cast<std::size_t>((offset / span) * static_cast<long double>(bands));
    if (band >= bands) {
        band = bands - 1;
    }
    return band;
}

void write_csv(const std::string &path, const std::unordered_map<std::uint64_t, SelectedRecord> &selected) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open sample output: " + path);
    }
    std::vector<SelectedRecord> rows;
    rows.reserve(selected.size());
    for (const auto &item : selected) {
        rows.push_back(item.second);
    }
    std::sort(rows.begin(), rows.end(), [](const auto &left, const auto &right) {
        return left.record.n < right.record.n;
    });

    out << std::setprecision(10);
    out << "n,reasons,total_steps,first_drop_time,odd_steps,even_steps,accelerated_steps,peak_step,peak_log2,peak_ratio_log2,residue_mod3,residue_mod4,residue_mod8,residue_mod16,residue_mod32,flags,checksum\n";
    for (const auto &row : rows) {
        const auto &record = row.record;
        out << record.n << ','
            << join_reasons(row.reasons) << ','
            << record.total_steps << ','
            << record.first_drop_time << ','
            << record.odd_steps << ','
            << record.even_steps << ','
            << record.accelerated_steps << ','
            << record.peak_step << ','
            << record.peak_log2 << ','
            << record.peak_ratio_log2 << ','
            << static_cast<unsigned>(record.residue_mod3) << ','
            << static_cast<unsigned>(record.residue_mod4) << ','
            << static_cast<unsigned>(record.residue_mod8) << ','
            << static_cast<unsigned>(record.residue_mod16) << ','
            << static_cast<unsigned>(record.residue_mod32) << ','
            << record.flags << ','
            << record.checksum << '\n';
    }
}

void write_metadata(
    const Options &options,
    const collatz::BinaryFeatureHeader &header,
    std::uint64_t available_records,
    std::uint64_t records_read,
    std::uint64_t effective_range_end,
    const std::unordered_map<std::uint64_t, SelectedRecord> &selected,
    const std::map<std::string, std::size_t> &reason_counts,
    std::uint64_t checksum) {
    const std::string path = path_join(options.output_dir, "metadata.json");
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open sample metadata: " + path);
    }

    std::size_t assignments = 0;
    for (const auto &item : reason_counts) {
        assignments += item.second;
    }

    out << "{\n"
        << "  \"dataset_type\": \"collatz_stratified_sample\",\n"
        << "  \"tool\": \"collatz_stratified_sample\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"input\": \"" << collatz::json_escape(options.input) << "\",\n"
        << "  \"input_range_start\": " << header.range_start << ",\n"
        << "  \"input_range_end\": " << header.range_end << ",\n"
        << "  \"effective_range_end\": " << effective_range_end << ",\n"
        << "  \"input_records_available\": " << available_records << ",\n"
        << "  \"input_records_read\": " << records_read << ",\n"
        << "  \"selected_rows\": " << selected.size() << ",\n"
        << "  \"reason_assignments\": " << assignments << ",\n"
        << "  \"reason_count\": " << reason_counts.size() << ",\n"
        << "  \"random_count\": " << options.random_count << ",\n"
        << "  \"global_top\": " << options.global_top << ",\n"
        << "  \"residue_top\": " << options.residue_top << ",\n"
        << "  \"range_bands\": " << options.range_bands << ",\n"
        << "  \"range_top\": " << options.range_top << ",\n"
        << "  \"seed\": " << options.seed << ",\n"
        << "  \"feature_schema_version\": " << collatz::kFeatureVersion << ",\n"
        << "  \"binary_feature_version\": " << collatz::kBinaryFeatureVersion << ",\n"
        << "  \"checksum_fnv1a64\": " << checksum << ",\n"
        << "  \"files\": {\"samples\": \"samples.csv\"},\n"
        << "  \"reasons\": {\n";
    std::size_t index = 0;
    for (const auto &item : reason_counts) {
        out << "    \"" << collatz::json_escape(item.first) << "\": " << item.second;
        if (++index != reason_counts.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  }\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto header = collatz::read_binary_header(options.input);
        if (!collatz::valid_binary_header(header)) {
            throw std::runtime_error("input has an invalid binary feature header");
        }

        std::unordered_map<std::uint64_t, std::set<std::string>> wanted;
        read_clusters_file(options.clusters, wanted);
        read_n_category_file(options.representatives, "representative", wanted);
        read_n_category_file(options.starts, "gnn_start", wanted);

        const auto available = collatz::binary_record_count(options.input);
        const auto wanted_records = options.limit == 0 ? available : std::min<std::uint64_t>(available, options.limit);
        const std::uint64_t sequential_range_end =
            wanted_records == 0 ? header.range_end : header.range_start + wanted_records - 1;
        const std::uint64_t effective_range_end = std::max(header.range_end, sequential_range_end);

        std::ifstream in(options.input, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open input: " + options.input);
        }
        in.seekg(static_cast<std::streamoff>(sizeof(collatz::BinaryFeatureHeader)));

        TopRecords random_baseline{options.random_count};
        TopRecords top_total_steps{options.global_top};
        TopRecords top_first_drop{options.global_top};
        TopRecords top_peak_ratio{options.global_top};
        TopRecords top_peak_log2{options.global_top};
        std::vector<ScoredRecord> record_total_steps;
        std::vector<ScoredRecord> record_peak_ratio;
        std::array<TopRecords, 32> residue_total_steps;
        for (auto &top : residue_total_steps) {
            top.keep = options.residue_top;
        }
        std::vector<TopRecords> range_total_steps(options.range_bands);
        std::vector<TopRecords> range_peak_ratio(options.range_bands);
        for (auto &top : range_total_steps) {
            top.keep = options.range_top;
        }
        for (auto &top : range_peak_ratio) {
            top.keep = options.range_top;
        }
        std::unordered_map<std::uint64_t, SelectedRecord> selected;
        std::map<std::string, std::size_t> reason_counts;

        std::uint32_t best_total_steps = 0;
        double best_peak_ratio = -1.0;
        std::uint64_t checksum = 14695981039346656037ull;
        std::uint64_t records_read = 0;

        for (; records_read < wanted_records; ++records_read) {
            collatz::BinaryFeatureRecord record{};
            in.read(reinterpret_cast<char *>(&record), sizeof(record));
            if (!in) {
                throw std::runtime_error("failed while reading binary feature record");
            }
            mix_checksum(checksum, record.n);
            mix_checksum(checksum, record.checksum);

            const auto target = wanted.find(record.n);
            if (target != wanted.end()) {
                for (const auto &reason : target->second) {
                    append_reason(selected, reason_counts, record, reason);
                }
            }

            const double random_score = -static_cast<double>(splitmix64(record.n ^ options.seed) >> 11u);
            consider_top(random_baseline, record, random_score);
            consider_top(top_total_steps, record, static_cast<double>(record.total_steps));
            consider_top(top_first_drop, record, static_cast<double>(record.first_drop_time));
            consider_top(top_peak_ratio, record, record.peak_ratio_log2);
            consider_top(top_peak_log2, record, record.peak_log2);
            consider_top(residue_total_steps[record.residue_mod32], record, static_cast<double>(record.total_steps));

            if (options.range_bands != 0 && options.range_top != 0) {
                const auto band = range_band_for(header.range_start, effective_range_end, record, options.range_bands);
                consider_top(range_total_steps[band], record, static_cast<double>(record.total_steps));
                consider_top(range_peak_ratio[band], record, record.peak_ratio_log2);
            }

            if (record.total_steps > best_total_steps) {
                best_total_steps = record.total_steps;
                record_total_steps.push_back({static_cast<double>(record.total_steps), record});
            }
            if (record.peak_ratio_log2 > best_peak_ratio) {
                best_peak_ratio = record.peak_ratio_log2;
                record_peak_ratio.push_back({record.peak_ratio_log2, record});
            }
        }

        append_scored(selected, reason_counts, random_baseline.records, "random_baseline");
        append_scored(selected, reason_counts, top_total_steps.records, "global_top_total_steps");
        append_scored(selected, reason_counts, top_first_drop.records, "global_top_first_drop_time");
        append_scored(selected, reason_counts, top_peak_ratio.records, "global_top_peak_ratio");
        append_scored(selected, reason_counts, top_peak_log2.records, "global_top_peak_log2");
        append_scored(selected, reason_counts, record_total_steps, "record_ladder_total_steps");
        append_scored(selected, reason_counts, record_peak_ratio, "record_ladder_peak_ratio");
        for (std::size_t residue = 0; residue < residue_total_steps.size(); ++residue) {
            append_scored(selected, reason_counts, residue_total_steps[residue].records, "residue_mod32_" + std::to_string(residue) + "_top_total_steps");
        }
        for (std::size_t band = 0; band < range_total_steps.size(); ++band) {
            append_scored(selected, reason_counts, range_total_steps[band].records, "range_band_" + std::to_string(band) + "_top_total_steps");
            append_scored(selected, reason_counts, range_peak_ratio[band].records, "range_band_" + std::to_string(band) + "_top_peak_ratio");
        }

        std::filesystem::create_directories(options.output_dir);
        write_csv(path_join(options.output_dir, "samples.csv"), selected);
        write_metadata(options, header, available, records_read, effective_range_end, selected, reason_counts, checksum);

        std::cout << "records_read=" << records_read
                  << " selected_rows=" << selected.size()
                  << " reason_count=" << reason_counts.size()
                  << " output_dir=" << options.output_dir
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
