#include "collatz/feature_io.hpp"
#include "collatz/ml.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::uint64_t n = 0;
    std::string input;
    std::string starts_file;
    std::string format = "pgm";
    std::string output_dir = "data/generated/images";
    std::uint64_t count = 8;
    std::uint32_t max_steps = 0;
    std::size_t size = collatz::kDefaultImageSize;
    std::size_t mtf_bins = 16;
    std::uint8_t residue_modulus = 32;
};

struct ArtifactSet {
    std::uint64_t n = 0;
    std::string recurrence;
    std::string gaf;
    std::string mtf;
    std::string parity;
    std::string residue;
};

void usage(std::ostream &out) {
    out << "usage: collatz_path_image (--n N | --input FILE | --starts-file FILE) [options]\n\n"
        << "options:\n"
        << "  --output-dir DIR       output directory (default data/generated/images)\n"
        << "  --format MODE          pgm or tensor (default pgm)\n"
        << "  --count N              records to render from --input (default 8)\n"
        << "  --max-steps N          max path steps, default input header max_steps or 1000000\n"
        << "  --size N               square image size in pixels (default 64)\n"
        << "  --mtf-bins N           Markov transition bins (default 16)\n"
        << "  --residue-modulus N    residue raster modulus (default 32)\n";
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

        if (arg == "--n") {
            const auto value = collatz::parse_u64(need_value("--n"));
            if (!value || *value == 0) {
                throw std::runtime_error("--n must be a positive uint64 integer");
            }
            options.n = *value;
        } else if (arg == "--input") {
            options.input = need_value("--input");
        } else if (arg == "--starts-file") {
            options.starts_file = need_value("--starts-file");
        } else if (arg == "--sample-file") {
            options.starts_file = need_value("--sample-file");
        } else if (arg == "--format") {
            options.format = need_value("--format");
            if (options.format != "pgm" && options.format != "tensor") {
                throw std::runtime_error("--format must be pgm or tensor");
            }
        } else if (arg == "--output-dir") {
            options.output_dir = need_value("--output-dir");
        } else if (arg == "--count") {
            const auto value = collatz::parse_u64(need_value("--count"));
            if (!value) {
                throw std::runtime_error("--count must be an integer");
            }
            options.count = *value;
        } else if (arg == "--max-steps") {
            const auto value = collatz::parse_u32(need_value("--max-steps"));
            if (!value || *value == 0) {
                throw std::runtime_error("--max-steps must be a positive integer");
            }
            options.max_steps = *value;
        } else if (arg == "--size") {
            options.size = parse_size_arg(need_value("--size"), "--size");
        } else if (arg == "--mtf-bins") {
            options.mtf_bins = parse_size_arg(need_value("--mtf-bins"), "--mtf-bins");
        } else if (arg == "--residue-modulus") {
            const auto value = collatz::parse_u32(need_value("--residue-modulus"));
            if (!value || *value < 2 || *value > 255) {
                throw std::runtime_error("--residue-modulus must be in [2,255]");
            }
            options.residue_modulus = static_cast<std::uint8_t>(*value);
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    const int source_count = (options.n != 0 ? 1 : 0) + (!options.input.empty() ? 1 : 0) + (!options.starts_file.empty() ? 1 : 0);
    if (source_count != 1) {
        throw std::runtime_error("exactly one of --n, --input, or --starts-file is required");
    }
    return options;
}

std::string path_join(const std::string &dir, const std::string &file) {
    return (std::filesystem::path(dir) / file).string();
}

std::string image_name(std::uint64_t n, const std::string &kind) {
    return "n_" + std::to_string(n) + "_" + kind + ".pgm";
}

std::vector<std::uint64_t> starts_from_input(const std::string &path, std::uint64_t count, std::uint32_t &max_steps) {
    const auto header = collatz::read_binary_header(path);
    if (!collatz::valid_binary_header(header)) {
        throw std::runtime_error("input has an invalid binary feature header");
    }
    if (max_steps == 0) {
        max_steps = header.max_steps;
    }

    const auto available = collatz::binary_record_count(path);
    const auto wanted = std::min<std::uint64_t>(available, count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open input: " + path);
    }
    in.seekg(static_cast<std::streamoff>(sizeof(collatz::BinaryFeatureHeader)));

    std::vector<std::uint64_t> starts;
    starts.reserve(static_cast<std::size_t>(wanted));
    for (std::uint64_t i = 0; i < wanted; ++i) {
        collatz::BinaryFeatureRecord record{};
        in.read(reinterpret_cast<char *>(&record), sizeof(record));
        if (!in) {
            throw std::runtime_error("failed while reading binary feature record");
        }
        starts.push_back(record.n);
    }
    return starts;
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

std::vector<std::uint64_t> starts_from_file(const std::string &path, std::uint64_t count) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open starts file: " + path);
    }

    std::vector<std::uint64_t> starts;
    std::string line;
    std::size_t n_column = 0;
    bool header_checked = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.empty()) {
            continue;
        }
        if (!header_checked) {
            header_checked = true;
            auto header_it = std::find(parts.begin(), parts.end(), "n");
            if (header_it != parts.end()) {
                n_column = static_cast<std::size_t>(std::distance(parts.begin(), header_it));
                continue;
            }
        }
        if (n_column >= parts.size()) {
            throw std::runtime_error("starts file row does not contain n column");
        }
        const auto n = collatz::parse_u64(parts[n_column]);
        if (!n || *n == 0) {
            continue;
        }
        starts.push_back(*n);
        if (count != 0 && starts.size() >= count) {
            break;
        }
    }
    return starts;
}

ArtifactSet render_start(const Options &options, std::uint64_t n, std::uint32_t max_steps) {
    const auto feature = collatz::compute_feature(n, max_steps);
    const auto record = collatz::to_binary_record(feature);
    const auto sketch = collatz::log_path_sketch(n, max_steps, options.size);
    const auto parity_bits = collatz::parity_bits_from_record(record);
    const auto residues = collatz::residue_sequence(n, max_steps, options.size, options.residue_modulus);

    ArtifactSet artifacts;
    artifacts.n = n;
    artifacts.recurrence = image_name(n, "recurrence");
    artifacts.gaf = image_name(n, "gaf");
    artifacts.mtf = image_name(n, "mtf");
    artifacts.parity = image_name(n, "parity");
    artifacts.residue = image_name(n, "residue_mod" + std::to_string(static_cast<unsigned>(options.residue_modulus)));

    collatz::write_pgm(
        path_join(options.output_dir, artifacts.recurrence),
        collatz::recurrence_image(sketch, options.size),
        options.size,
        options.size);
    collatz::write_pgm(
        path_join(options.output_dir, artifacts.gaf),
        collatz::gramian_angular_field(sketch, options.size),
        options.size,
        options.size);
    collatz::write_pgm(
        path_join(options.output_dir, artifacts.mtf),
        collatz::markov_transition_field(sketch, options.size, options.mtf_bins),
        options.size,
        options.size);
    collatz::write_pgm(
        path_join(options.output_dir, artifacts.parity),
        collatz::parity_raster(parity_bits, options.size),
        options.size,
        options.size);
    collatz::write_pgm(
        path_join(options.output_dir, artifacts.residue),
        collatz::residue_raster(residues, options.size, options.residue_modulus),
        options.size,
        options.size);
    return artifacts;
}

std::vector<std::uint8_t> tensor_channels(const Options &options, std::uint64_t n, std::uint32_t max_steps) {
    const auto feature = collatz::compute_feature(n, max_steps);
    const auto record = collatz::to_binary_record(feature);
    const auto sketch = collatz::log_path_sketch(n, max_steps, options.size);
    const auto parity_bits = collatz::parity_bits_from_record(record);
    const auto residues = collatz::residue_sequence(n, max_steps, options.size, options.residue_modulus);

    std::vector<std::uint8_t> tensor;
    tensor.reserve(options.size * options.size * 5);
    auto append = [&](const std::vector<std::uint8_t> &pixels) {
        tensor.insert(tensor.end(), pixels.begin(), pixels.end());
    };
    append(collatz::recurrence_image(sketch, options.size));
    append(collatz::gramian_angular_field(sketch, options.size));
    append(collatz::markov_transition_field(sketch, options.size, options.mtf_bins));
    append(collatz::parity_raster(parity_bits, options.size));
    append(collatz::residue_raster(residues, options.size, options.residue_modulus));
    return tensor;
}

void write_manifest(
    const Options &options,
    const std::vector<ArtifactSet> &artifacts,
    std::uint32_t max_steps) {
    const std::string path = path_join(options.output_dir, "manifest.json");
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open manifest: " + path);
    }

    out << "{\n"
        << "  \"dataset_type\": \"collatz_path_images\",\n"
        << "  \"tool\": \"collatz_path_image\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"input\": \"" << collatz::json_escape(options.input) << "\",\n"
        << "  \"image_size\": " << options.size << ",\n"
        << "  \"max_steps\": " << max_steps << ",\n"
        << "  \"mtf_bins\": " << options.mtf_bins << ",\n"
        << "  \"residue_modulus\": " << static_cast<unsigned>(options.residue_modulus) << ",\n"
        << "  \"artifacts\": [\n";
    for (std::size_t i = 0; i < artifacts.size(); ++i) {
        const auto &item = artifacts[i];
        out << "    {\"n\": " << item.n
            << ", \"recurrence\": \"" << item.recurrence
            << "\", \"gaf\": \"" << item.gaf
            << "\", \"mtf\": \"" << item.mtf
            << "\", \"parity\": \"" << item.parity
            << "\", \"residue\": \"" << item.residue << "\"}";
        if (i + 1 != artifacts.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n"
        << "}\n";
}

void write_tensor_outputs(
    const Options &options,
    const std::vector<std::uint64_t> &starts,
    std::uint32_t max_steps) {
    const auto tensor_path = path_join(options.output_dir, "image_tensors.bin");
    const auto index_path = path_join(options.output_dir, "image_tensors_index.csv");
    const auto metadata_path = path_join(options.output_dir, "metadata.json");
    collatz::ensure_parent_dir(tensor_path);

    std::ofstream tensor(tensor_path, std::ios::binary);
    if (!tensor) {
        throw std::runtime_error("failed to open tensor output: " + tensor_path);
    }
    std::ofstream index(index_path);
    if (!index) {
        throw std::runtime_error("failed to open tensor index output: " + index_path);
    }
    index << "row,n,offset,channels,height,width\n";
    const std::size_t row_bytes = options.size * options.size * 5;
    std::uint64_t row = 0;
    for (const auto n : starts) {
        const auto bytes = tensor_channels(options, n, max_steps);
        if (bytes.size() != row_bytes) {
            throw std::runtime_error("tensor row byte count mismatch");
        }
        const auto offset = row * row_bytes;
        tensor.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!tensor) {
            throw std::runtime_error("failed while writing tensor output");
        }
        index << row << ',' << n << ',' << offset << ",5," << options.size << ',' << options.size << '\n';
        ++row;
    }

    std::ofstream metadata(metadata_path);
    if (!metadata) {
        throw std::runtime_error("failed to open tensor metadata: " + metadata_path);
    }
    metadata << "{\n"
             << "  \"dataset_type\": \"collatz_path_image_tensors\",\n"
             << "  \"tool\": \"collatz_path_image\",\n"
             << "  \"schema_version\": \"image_tensor_v1\",\n"
             << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
             << "  \"format\": \"tensor\",\n"
             << "  \"dtype\": \"uint8\",\n"
             << "  \"layout\": \"rows,channels,height,width\",\n"
             << "  \"rows\": " << starts.size() << ",\n"
             << "  \"channels\": [\"recurrence\", \"gaf\", \"mtf\", \"parity\", \"residue\"],\n"
             << "  \"height\": " << options.size << ",\n"
             << "  \"width\": " << options.size << ",\n"
             << "  \"row_bytes\": " << row_bytes << ",\n"
             << "  \"max_steps\": " << max_steps << ",\n"
             << "  \"outputs\": {\"tensor\": \"image_tensors.bin\", \"index\": \"image_tensors_index.csv\"}\n"
             << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse_args(argc, argv);
        std::uint32_t max_steps = options.max_steps;
        std::vector<std::uint64_t> starts;

        if (options.n != 0) {
            starts.push_back(options.n);
            if (max_steps == 0) {
                max_steps = 1000000;
            }
        } else if (!options.input.empty()) {
            starts = starts_from_input(options.input, options.count, max_steps);
        } else {
            starts = starts_from_file(options.starts_file, options.count);
            if (max_steps == 0) {
                max_steps = 1000000;
            }
        }
        if (max_steps == 0) {
            max_steps = 1000000;
        }

        std::filesystem::create_directories(options.output_dir);
        if (options.format == "tensor") {
            write_tensor_outputs(options, starts, max_steps);
        } else {
            std::vector<ArtifactSet> artifacts;
            artifacts.reserve(starts.size());
            for (const auto n : starts) {
                artifacts.push_back(render_start(options, n, max_steps));
            }
            write_manifest(options, artifacts, max_steps);
        }

        std::cout << "rendered_starts=" << starts.size()
                  << " output_dir=" << options.output_dir
                  << " format=" << options.format
                  << " image_size=" << options.size
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
