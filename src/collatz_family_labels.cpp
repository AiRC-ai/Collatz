#include "collatz/feature_io.hpp"
#include "collatz/ml.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string sample_file;
    std::string source_targets;
    std::string output = "data/generated/ml_labels/families.csv";
    std::string metadata = "data/generated/ml_labels/metadata.json";
    std::uint32_t max_steps = 0;
    std::size_t tail_cap = 256;
    std::size_t range_bands = 16;
};

struct LabelRow {
    std::uint64_t n = 0;
    std::string tail_entry_value;
    std::uint64_t tail_hash = 0;
    std::uint64_t coalescence_family_id = 0;
    std::uint32_t first_drop_bucket = 0;
    std::uint32_t total_steps_bucket = 0;
    std::uint32_t peak_ratio_bucket = 0;
    std::uint64_t parity_motif_hash = 0;
    std::uint64_t residue_motif_hash = 0;
    std::size_t range_band = 0;
    std::uint32_t bit_length = 0;
    std::string source_family = "none";
};

void usage(std::ostream &out) {
    out << "usage: collatz_family_labels --input FILE --sample-file FILE [options]\n\n"
        << "options:\n"
        << "  --source-targets FILE  optional public source target CSV\n"
        << "  --output FILE          families.csv output path\n"
        << "  --metadata FILE        metadata JSON output path\n"
        << "  --max-steps N          max path steps, default binary header max_steps\n"
        << "  --tail-cap N           values to hash from shared tail suffix (default 256)\n"
        << "  --range-bands N        number of range bands (default 16)\n";
}

std::uint64_t parse_required_u64(const std::string &text, const char *name) {
    const auto value = collatz::parse_u64(text);
    if (!value) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    return *value;
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
        } else if (arg == "--sample-file") {
            options.sample_file = need_value("--sample-file");
        } else if (arg == "--source-targets") {
            options.source_targets = need_value("--source-targets");
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--metadata") {
            options.metadata = need_value("--metadata");
        } else if (arg == "--max-steps") {
            const auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--tail-cap") {
            options.tail_cap = static_cast<std::size_t>(parse_required_u64(need_value("--tail-cap"), "--tail-cap"));
            if (options.tail_cap == 0) {
                throw std::runtime_error("--tail-cap must be positive");
            }
        } else if (arg == "--range-bands") {
            options.range_bands = static_cast<std::size_t>(parse_required_u64(need_value("--range-bands"), "--range-bands"));
            if (options.range_bands == 0) {
                throw std::runtime_error("--range-bands must be positive");
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.input.empty() || options.sample_file.empty()) {
        throw std::runtime_error("--input and --sample-file are required");
    }
    return options;
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

std::unordered_set<std::uint64_t> read_sample_filter(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open sample file: " + path);
    }
    std::string header;
    if (!std::getline(in, header)) {
        return {};
    }
    const auto columns = split_csv_line(header);
    const auto found = std::find(columns.begin(), columns.end(), "n");
    if (found == columns.end()) {
        throw std::runtime_error("sample file must include n column: " + path);
    }
    const auto n_column = static_cast<std::size_t>(std::distance(columns.begin(), found));
    std::unordered_set<std::uint64_t> selected;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (n_column >= parts.size()) {
            continue;
        }
        const auto n = collatz::parse_u64(parts[n_column]);
        if (n && *n != 0) {
            selected.insert(*n);
        }
    }
    return selected;
}

std::string source_family(std::string source) {
    std::string lower;
    lower.reserve(source.size());
    for (const char ch : source) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lower.find("oeis") != std::string::npos || lower.find("a006") != std::string::npos) {
        return "oeis";
    }
    if (lower.find("roosendaal") != std::string::npos) {
        return "roosendaal";
    }
    if (lower.find("oliveira") != std::string::npos || lower.find("silva") != std::string::npos) {
        return "oliveira_e_silva";
    }
    if (lower.find("barina") != std::string::npos) {
        return "barina";
    }
    return source.empty() ? "none" : source;
}

std::unordered_map<std::uint64_t, std::string> read_source_families(const std::string &path) {
    std::unordered_map<std::uint64_t, std::string> families;
    if (path.empty()) {
        return families;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open source targets: " + path);
    }
    std::string header;
    if (!std::getline(in, header)) {
        return families;
    }
    const auto columns = split_csv_line(header);
    const auto source_it = std::find(columns.begin(), columns.end(), "source");
    const auto n_it = std::find(columns.begin(), columns.end(), "n");
    if (source_it == columns.end() || n_it == columns.end()) {
        throw std::runtime_error("source targets must contain source and n columns: " + path);
    }
    const auto source_column = static_cast<std::size_t>(std::distance(columns.begin(), source_it));
    const auto n_column = static_cast<std::size_t>(std::distance(columns.begin(), n_it));
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (source_column >= parts.size() || n_column >= parts.size()) {
            continue;
        }
        const auto n = collatz::parse_u64(parts[n_column]);
        if (n && *n != 0) {
            families.emplace(*n, source_family(parts[source_column]));
        }
    }
    return families;
}

void hash_u64(std::uint64_t &hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        hash ^= static_cast<unsigned char>((value >> (i * 8u)) & 0xffu);
        hash *= 1099511628211ull;
    }
}

void hash_u128(std::uint64_t &hash, collatz::UInt128 value) {
    const auto parts = collatz::split_uint128(value);
    hash_u64(hash, parts.low);
    hash_u64(hash, parts.high);
}

std::uint64_t hash_tail(const std::vector<collatz::PathPoint> &path, std::size_t start_index, std::size_t cap) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto end = std::min(path.size(), start_index + cap);
    for (std::size_t i = start_index; i < end; ++i) {
        hash_u128(hash, path[i].value);
    }
    return hash;
}

std::uint64_t hash_u16_sequence(const std::vector<std::uint16_t> &tokens) {
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto token : tokens) {
        hash_u64(hash, token);
    }
    return hash;
}

std::uint64_t hash_u8_sequence(const std::vector<std::uint8_t> &tokens) {
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto token : tokens) {
        hash_u64(hash, token);
    }
    return hash;
}

std::uint32_t bucket_u32(std::uint32_t value) {
    if (value == 0) {
        return 0;
    }
    std::uint32_t bucket = 0;
    while (value > 1) {
        value >>= 1u;
        ++bucket;
    }
    return bucket;
}

std::uint32_t bucket_double(double value) {
    if (value <= 0.0) {
        return 0;
    }
    return static_cast<std::uint32_t>(std::min(63.0, std::floor(value * 2.0)));
}

std::uint32_t bit_length(std::uint64_t value) {
    std::uint32_t bits = 0;
    while (value != 0) {
        ++bits;
        value >>= 1u;
    }
    return std::max<std::uint32_t>(1, bits);
}

std::size_t range_band(std::uint64_t n, std::uint64_t start, std::uint64_t end, std::size_t bands) {
    const auto span = end > start ? end - start + 1 : 1;
    return std::min<std::size_t>(bands - 1, static_cast<std::size_t>(((n - start) * bands) / span));
}

std::vector<std::uint16_t> residue_transition_tokens(std::uint64_t n, std::uint32_t max_steps, std::size_t dims, std::uint8_t modulus) {
    const auto residues = collatz::residue_sequence(n, max_steps, dims, modulus);
    std::vector<std::uint16_t> tokens;
    if (residues.size() < 2) {
        return tokens;
    }
    tokens.reserve(residues.size() - 1);
    for (std::size_t i = 1; i < residues.size(); ++i) {
        tokens.push_back(static_cast<std::uint16_t>(residues[i - 1] * modulus + residues[i]));
    }
    return tokens;
}

LabelRow make_label(
    const collatz::BinaryFeatureHeader &header,
    const collatz::BinaryFeatureRecord &record,
    const Options &options,
    const std::unordered_map<std::uint64_t, std::string> &source_by_n,
    std::uint32_t max_steps) {
    bool overflow = false;
    const auto path = collatz::generate_path(record.n, max_steps, &overflow);
    std::size_t tail_index = 0;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (path[i].value < record.n) {
            tail_index = i;
            break;
        }
    }
    if (path.empty()) {
        throw std::runtime_error("empty path while labeling n=" + std::to_string(record.n));
    }

    LabelRow row;
    row.n = record.n;
    row.tail_entry_value = collatz::uint128_to_decimal(path[tail_index].value);
    row.tail_hash = hash_tail(path, tail_index, options.tail_cap);
    row.first_drop_bucket = bucket_u32(record.first_drop_time);
    row.total_steps_bucket = bucket_u32(record.total_steps);
    row.peak_ratio_bucket = bucket_double(record.peak_ratio_log2);
    row.parity_motif_hash = hash_u16_sequence(collatz::parity_run_tokens(record));
    row.residue_motif_hash = hash_u16_sequence(residue_transition_tokens(record.n, max_steps, 128, 32));
    row.range_band = range_band(record.n, header.range_start, header.range_end, options.range_bands);
    row.bit_length = bit_length(record.n);
    const auto found_source = source_by_n.find(record.n);
    if (found_source != source_by_n.end()) {
        row.source_family = found_source->second;
    }
    return row;
}

void assign_family_ids(std::vector<LabelRow> &rows) {
    std::map<std::pair<std::string, std::uint64_t>, std::uint64_t> ids;
    for (const auto &row : rows) {
        ids.emplace(std::make_pair(row.tail_entry_value, row.tail_hash), 0);
    }
    std::uint64_t next = 0;
    for (auto &item : ids) {
        item.second = next++;
    }
    for (auto &row : rows) {
        row.coalescence_family_id = ids[std::make_pair(row.tail_entry_value, row.tail_hash)];
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto header = collatz::read_binary_header(options.input);
        if (!collatz::valid_binary_header(header)) {
            throw std::runtime_error("input has an invalid binary feature header");
        }
        const auto selected = read_sample_filter(options.sample_file);
        const auto source_by_n = read_source_families(options.source_targets);
        const auto available = collatz::binary_record_count(options.input);
        const std::uint32_t max_steps = options.max_steps == 0 ? header.max_steps : options.max_steps;

        std::ifstream in(options.input, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open input: " + options.input);
        }
        in.seekg(static_cast<std::streamoff>(sizeof(collatz::BinaryFeatureHeader)));

        std::vector<LabelRow> rows;
        rows.reserve(selected.size());
        for (std::uint64_t i = 0; i < available; ++i) {
            collatz::BinaryFeatureRecord record{};
            in.read(reinterpret_cast<char *>(&record), sizeof(record));
            if (!in) {
                throw std::runtime_error("failed while reading binary feature record");
            }
            if (!selected.empty() && selected.find(record.n) == selected.end()) {
                continue;
            }
            rows.push_back(make_label(header, record, options, source_by_n, max_steps));
            if (!selected.empty() && rows.size() >= selected.size()) {
                break;
            }
        }
        assign_family_ids(rows);
        std::sort(rows.begin(), rows.end(), [](const LabelRow &a, const LabelRow &b) { return a.n < b.n; });

        collatz::ensure_parent_dir(options.output);
        std::ofstream out(options.output);
        if (!out) {
            throw std::runtime_error("failed to open output: " + options.output);
        }
        out << "n,tail_entry_value,tail_hash,coalescence_family_id,first_drop_bucket,total_steps_bucket,"
            << "peak_ratio_bucket,parity_motif_hash,residue_motif_hash,range_band,bit_length,source_family\n";
        for (const auto &row : rows) {
            out << row.n << ','
                << row.tail_entry_value << ','
                << row.tail_hash << ','
                << row.coalescence_family_id << ','
                << row.first_drop_bucket << ','
                << row.total_steps_bucket << ','
                << row.peak_ratio_bucket << ','
                << row.parity_motif_hash << ','
                << row.residue_motif_hash << ','
                << row.range_band << ','
                << row.bit_length << ','
                << row.source_family << '\n';
        }

        collatz::ensure_parent_dir(options.metadata);
        std::ofstream meta(options.metadata);
        if (!meta) {
            throw std::runtime_error("failed to open metadata: " + options.metadata);
        }
        meta << "{\n"
             << "  \"dataset_type\": \"collatz_family_labels\",\n"
             << "  \"tool\": \"collatz_family_labels\",\n"
             << "  \"schema_version\": \"family_labels_v1\",\n"
             << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
             << "  \"input\": \"" << collatz::json_escape(options.input) << "\",\n"
             << "  \"sample_file\": \"" << collatz::json_escape(options.sample_file) << "\",\n"
             << "  \"source_targets\": \"" << collatz::json_escape(options.source_targets) << "\",\n"
             << "  \"audit_range_start\": " << header.range_start << ",\n"
             << "  \"audit_range_end\": " << header.range_end << ",\n"
             << "  \"sample_count\": " << rows.size() << ",\n"
             << "  \"tail_cap\": " << options.tail_cap << ",\n"
             << "  \"range_bands\": " << options.range_bands << ",\n"
             << "  \"outputs\": {\"families\": \"" << std::filesystem::path(options.output).filename().string() << "\"}\n"
             << "}\n";

        std::cout << "family_labels=" << rows.size() << " output=" << options.output << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
