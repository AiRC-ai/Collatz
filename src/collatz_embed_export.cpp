#include "collatz/feature_io.hpp"
#include "collatz/ml.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
    std::string input;
    std::string output_dir = "data/generated/ml";
    std::string sample_file;
    std::uint64_t limit = 0;
    std::uint32_t max_steps = 0;
    std::size_t sketch_dims = collatz::kDefaultSketchLength;
    std::size_t residue_dims = collatz::kDefaultSketchLength;
    std::uint8_t residue_modulus = 32;
    std::string metric_mode = "full";
};

void usage(std::ostream &out) {
    out << "usage: collatz_embed_export --input FILE [options]\n\n"
        << "options:\n"
        << "  --output-dir DIR       output directory (default data/generated/ml)\n"
        << "  --sample-file FILE     optional samples.csv with an n column to export\n"
        << "  --limit N              max records to export, 0 means all records\n"
        << "  --max-steps N          max path steps for derived sketches, default input header max_steps\n"
        << "  --sketch-dims N        fixed log-path sketch length (default 128)\n"
        << "  --residue-dims N       residue sequence length (default 128)\n"
        << "  --residue-modulus N    residue modulus for transition stream (default 32)\n"
        << "  --metric-mode MODE     full or safe; safe also writes metrics_safe.csv (default full)\n";
}

std::size_t parse_size_arg(const std::string &value, const char *name) {
    const auto parsed = collatz::parse_u64(value);
    if (!parsed || *parsed == 0) {
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
        } else if (arg == "--sample-file") {
            options.sample_file = need_value("--sample-file");
        } else if (arg == "--limit") {
            const auto value = collatz::parse_u64(need_value("--limit"));
            if (!value) {
                throw std::runtime_error("--limit must be an integer");
            }
            options.limit = *value;
        } else if (arg == "--max-steps") {
            const auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--sketch-dims") {
            options.sketch_dims = parse_size_arg(need_value("--sketch-dims"), "--sketch-dims");
        } else if (arg == "--residue-dims") {
            options.residue_dims = parse_size_arg(need_value("--residue-dims"), "--residue-dims");
        } else if (arg == "--residue-modulus") {
            const auto value = collatz::parse_u32(need_value("--residue-modulus"));
            if (!value || *value < 2 || *value > 255) {
                throw std::runtime_error("--residue-modulus must be in [2,255]");
            }
            options.residue_modulus = static_cast<std::uint8_t>(*value);
        } else if (arg == "--metric-mode") {
            options.metric_mode = need_value("--metric-mode");
            if (options.metric_mode != "full" && options.metric_mode != "safe") {
                throw std::runtime_error("--metric-mode must be full or safe");
            }
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

std::unordered_set<std::uint64_t> read_sample_filter(const std::string &path) {
    std::unordered_set<std::uint64_t> selected;
    if (path.empty()) {
        return selected;
    }
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open sample file: " + path);
    }
    std::string header;
    if (!std::getline(in, header)) {
        return selected;
    }
    const auto columns = split_csv_line(header);
    int n_index = -1;
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "n") {
            n_index = static_cast<int>(i);
            break;
        }
    }
    if (n_index < 0) {
        throw std::runtime_error("sample file must contain an n column: " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (n_index >= static_cast<int>(parts.size())) {
            continue;
        }
        const auto n = collatz::parse_u64(parts[static_cast<std::size_t>(n_index)]);
        if (n && *n != 0) {
            selected.insert(*n);
        }
    }
    return selected;
}

std::string fixed_c_string(const char *data, std::size_t size) {
    std::size_t len = 0;
    while (len < size && data[len] != '\0') {
        ++len;
    }
    return std::string(data, len);
}

void open_output(std::ofstream &out, const std::string &path) {
    collatz::ensure_parent_dir(path);
    out.open(path);
    if (!out) {
        throw std::runtime_error("failed to open output: " + path);
    }
    out << std::setprecision(10);
}

template <typename T>
void write_integral_sequence(std::ostream &out, const std::vector<T> &values, char delimiter) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << delimiter;
        }
        out << static_cast<unsigned long long>(values[i]);
    }
}

void write_double_sequence(std::ostream &out, const std::vector<double> &values, char delimiter) {
    out << std::setprecision(10);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << delimiter;
        }
        out << values[i];
    }
}

std::vector<std::uint16_t> residue_transition_tokens(
    std::uint64_t n,
    std::uint32_t max_steps,
    std::size_t dims,
    std::uint8_t modulus) {
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

void write_metrics_header(std::ostream &out) {
    out << "n";
    for (std::size_t i = 0; i < collatz::kMetricVectorDims; ++i) {
        out << ",m" << i;
    }
    out << '\n';
}

void write_metrics_header(std::ostream &out, std::size_t dims) {
    out << "n";
    for (std::size_t i = 0; i < dims; ++i) {
        out << ",m" << i;
    }
    out << '\n';
}

void write_sketch_header(std::ostream &out, std::size_t dims, const char *prefix) {
    out << "n";
    for (std::size_t i = 0; i < dims; ++i) {
        out << ',' << prefix << i;
    }
    out << '\n';
}

void write_metadata(
    const Options &options,
    const collatz::BinaryFeatureHeader &header,
    std::uint64_t available_records,
    std::uint64_t exported_rows,
    std::uint64_t records_scanned,
    std::uint64_t sample_filter_rows,
    std::uint64_t checksum) {
    const std::string path = path_join(options.output_dir, "metadata.json");
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open metadata output: " + path);
    }

    out << "{\n"
        << "  \"dataset_type\": \"collatz_embedding_inputs\",\n"
        << "  \"tool\": \"collatz_embed_export\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"input\": \"" << collatz::json_escape(options.input) << "\",\n"
        << "  \"sample_file\": \"" << collatz::json_escape(options.sample_file) << "\",\n"
        << "  \"input_created_utc\": \"" << collatz::json_escape(fixed_c_string(header.created_utc, sizeof(header.created_utc))) << "\",\n"
        << "  \"input_range_start\": " << header.range_start << ",\n"
        << "  \"input_range_end\": " << header.range_end << ",\n"
        << "  \"input_max_steps\": " << header.max_steps << ",\n"
        << "  \"input_records_available\": " << available_records << ",\n"
        << "  \"records_scanned\": " << records_scanned << ",\n"
        << "  \"rows_exported\": " << exported_rows << ",\n"
        << "  \"sample_filter_rows\": " << sample_filter_rows << ",\n"
        << "  \"metric_vector_dims\": " << collatz::kMetricVectorDims << ",\n"
        << "  \"metric_mode\": \"" << options.metric_mode << "\",\n"
        << "  \"metric_schema\": \"" << (options.metric_mode == "safe" ? "safe_v1" : "full_v1") << "\",\n"
        << "  \"safe_metric_vector_dims\": " << collatz::kSafeMetricVectorDims << ",\n"
        << "  \"excluded_metric_indices\": [";
    const auto &excluded = collatz::unsafe_metric_indices();
    for (std::size_t i = 0; i < excluded.size(); ++i) {
        out << excluded[i];
        if (i + 1 != excluded.size()) {
            out << ',';
        }
    }
    out << "],\n"
        << "  \"sketch_dims\": " << options.sketch_dims << ",\n"
        << "  \"residue_dims\": " << options.residue_dims << ",\n"
        << "  \"residue_modulus\": " << static_cast<unsigned>(options.residue_modulus) << ",\n"
        << "  \"max_steps_used\": " << (options.max_steps == 0 ? header.max_steps : options.max_steps) << ",\n"
        << "  \"feature_schema_version\": " << collatz::kFeatureVersion << ",\n"
        << "  \"binary_feature_version\": " << collatz::kBinaryFeatureVersion << ",\n"
        << "  \"checksum_fnv1a64\": " << checksum << ",\n"
        << "  \"outputs\": {\n"
        << "    \"metrics\": \"metrics.csv\",\n"
        << "    \"metrics_safe\": " << (options.metric_mode == "safe" ? "\"metrics_safe.csv\"" : "null") << ",\n"
        << "    \"parity_runs\": \"parity_runs.csv\",\n"
        << "    \"log_sketch\": \"log_sketch.csv\",\n"
        << "    \"residue_sequence\": \"residue_mod" << static_cast<unsigned>(options.residue_modulus) << ".csv\",\n"
        << "    \"residue_transitions\": \"residue_transitions_mod" << static_cast<unsigned>(options.residue_modulus) << ".csv\"\n"
        << "  }\n"
        << "}\n";
}

void mix_checksum(std::uint64_t &hash, std::uint64_t value) {
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        hash ^= static_cast<unsigned char>((value >> (i * 8u)) & 0xffu);
        hash *= 1099511628211ull;
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto header = collatz::read_binary_header(options.input);
        if (!collatz::valid_binary_header(header)) {
            throw std::runtime_error("input has an invalid binary feature header");
        }

        const auto available = collatz::binary_record_count(options.input);
        const auto sample_filter = read_sample_filter(options.sample_file);
        const bool has_filter = !sample_filter.empty();
        const auto wanted = options.limit == 0 ? available : std::min<std::uint64_t>(available, options.limit);
        const std::uint32_t max_steps = options.max_steps == 0 ? header.max_steps : options.max_steps;

        std::filesystem::create_directories(options.output_dir);
        std::ofstream metrics;
        std::ofstream metrics_safe;
        std::ofstream parity_runs;
        std::ofstream log_sketch;
        std::ofstream residues;
        std::ofstream transitions;
        open_output(metrics, path_join(options.output_dir, "metrics.csv"));
        if (options.metric_mode == "safe") {
            open_output(metrics_safe, path_join(options.output_dir, "metrics_safe.csv"));
        }
        open_output(parity_runs, path_join(options.output_dir, "parity_runs.csv"));
        open_output(log_sketch, path_join(options.output_dir, "log_sketch.csv"));
        open_output(residues, path_join(options.output_dir, "residue_mod" + std::to_string(options.residue_modulus) + ".csv"));
        open_output(transitions, path_join(options.output_dir, "residue_transitions_mod" + std::to_string(options.residue_modulus) + ".csv"));

        write_metrics_header(metrics);
        if (options.metric_mode == "safe") {
            write_metrics_header(metrics_safe, collatz::kSafeMetricVectorDims);
        }
        parity_runs << "n,tokens\n";
        write_sketch_header(log_sketch, options.sketch_dims, "s");
        write_sketch_header(residues, options.residue_dims, "r");
        transitions << "n,tokens\n";

        std::ifstream in(options.input, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open input: " + options.input);
        }
        in.seekg(static_cast<std::streamoff>(sizeof(collatz::BinaryFeatureHeader)));

        std::uint64_t exported = 0;
        std::uint64_t scanned = 0;
        std::uint64_t checksum = 14695981039346656037ull;
        for (; scanned < wanted;) {
            collatz::BinaryFeatureRecord record{};
            in.read(reinterpret_cast<char *>(&record), sizeof(record));
            if (!in) {
                throw std::runtime_error("failed while reading binary feature record");
            }
            ++scanned;
            if (has_filter && sample_filter.find(record.n) == sample_filter.end()) {
                continue;
            }
            mix_checksum(checksum, record.checksum);

            const auto metric = collatz::metric_vector(record);
            if (metric.size() != collatz::kMetricVectorDims) {
                throw std::runtime_error("metric vector dimension mismatch");
            }
            metrics << record.n;
            for (const auto value : metric) {
                metrics << ',' << value;
            }
            metrics << '\n';

            if (options.metric_mode == "safe") {
                const auto safe_metric = collatz::safe_metric_vector(record);
                if (safe_metric.size() != collatz::kSafeMetricVectorDims) {
                    throw std::runtime_error("safe metric vector dimension mismatch");
                }
                metrics_safe << record.n;
                for (const auto value : safe_metric) {
                    metrics_safe << ',' << value;
                }
                metrics_safe << '\n';
            }

            parity_runs << record.n << ',';
            write_integral_sequence(parity_runs, collatz::parity_run_tokens(record), ';');
            parity_runs << '\n';

            log_sketch << record.n << ',';
            write_double_sequence(log_sketch, collatz::log_path_sketch(record.n, max_steps, options.sketch_dims), ',');
            log_sketch << '\n';

            residues << record.n << ',';
            write_integral_sequence(residues, collatz::residue_sequence(record.n, max_steps, options.residue_dims, options.residue_modulus), ',');
            residues << '\n';

            transitions << record.n << ',';
            write_integral_sequence(transitions, residue_transition_tokens(record.n, max_steps, options.residue_dims, options.residue_modulus), ';');
            transitions << '\n';

            ++exported;
            if (has_filter && exported >= sample_filter.size()) {
                break;
            }
        }

        write_metadata(options, header, available, exported, scanned, sample_filter.size(), checksum);
        std::cout << "exported_rows=" << exported
                  << " output_dir=" << options.output_dir
                  << " metric_dims=" << collatz::kMetricVectorDims
                  << " sketch_dims=" << options.sketch_dims
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
