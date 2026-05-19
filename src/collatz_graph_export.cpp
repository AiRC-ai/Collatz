#include "collatz/core.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Options {
    std::uint64_t n = 0;
    std::string starts_file;
    std::string output_dir = "data/generated/graphs";
    std::uint64_t count = 64;
    std::uint32_t max_steps = 1000000;
    std::size_t preview_nodes = 512;
    std::size_t preview_edges = 1024;
};

struct StartSpec {
    std::uint64_t n = 0;
    std::string category = "manual";
};

struct Node {
    std::uint32_t id = 0;
    collatz::UInt128 value = 0;
    long double log2_value = 0.0L;
    bool is_start = false;
    bool is_terminal = false;
    std::uint32_t in_degree = 0;
    std::uint32_t out_degree = 0;
};

struct Edge {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
};

struct StartPath {
    std::uint64_t n = 0;
    std::string category;
    std::uint32_t start_node_id = 0;
    std::uint32_t steps = 0;
    bool overflow = false;
};

void usage(std::ostream &out) {
    out << "usage: collatz_graph_export (--n N | --starts-file FILE) [options]\n\n"
        << "options:\n"
        << "  --output-dir DIR       graph artifact directory (default data/generated/graphs)\n"
        << "  --count N              starts to read from CSV, 0 means all (default 64)\n"
        << "  --max-steps N          per-start path cap (default 1000000)\n"
        << "  --preview-nodes N      nodes embedded in dashboard JSON (default 512)\n"
        << "  --preview-edges N      edges embedded in dashboard JSON (default 1024)\n";
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
        } else if (arg == "--starts-file") {
            options.starts_file = need_value("--starts-file");
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
        } else if (arg == "--preview-nodes") {
            const auto value = collatz::parse_u64(need_value("--preview-nodes"));
            if (!value) {
                throw std::runtime_error("--preview-nodes must be an integer");
            }
            options.preview_nodes = static_cast<std::size_t>(*value);
        } else if (arg == "--preview-edges") {
            const auto value = collatz::parse_u64(need_value("--preview-edges"));
            if (!value) {
                throw std::runtime_error("--preview-edges must be an integer");
            }
            options.preview_edges = static_cast<std::size_t>(*value);
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    const int source_count = (options.n != 0 ? 1 : 0) + (!options.starts_file.empty() ? 1 : 0);
    if (source_count != 1) {
        throw std::runtime_error("exactly one of --n or --starts-file is required");
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

std::vector<StartSpec> starts_from_file(const std::string &path, std::uint64_t count) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open starts file: " + path);
    }

    std::vector<StartSpec> starts;
    std::unordered_set<std::uint64_t> seen;
    std::string line;
    std::size_t n_column = 0;
    std::size_t category_column = std::numeric_limits<std::size_t>::max();
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
            auto n_it = std::find(parts.begin(), parts.end(), "n");
            if (n_it != parts.end()) {
                n_column = static_cast<std::size_t>(std::distance(parts.begin(), n_it));
                auto category_it = std::find(parts.begin(), parts.end(), "category");
                if (category_it != parts.end()) {
                    category_column = static_cast<std::size_t>(std::distance(parts.begin(), category_it));
                }
                continue;
            }
        }
        if (n_column >= parts.size()) {
            throw std::runtime_error("starts file row does not contain n column");
        }
        const auto n = collatz::parse_u64(parts[n_column]);
        if (!n || *n == 0 || !seen.insert(*n).second) {
            continue;
        }
        std::string category = "selected";
        if (category_column < parts.size() && !parts[category_column].empty()) {
            category = parts[category_column];
        }
        starts.push_back({*n, category});
        if (count != 0 && starts.size() >= count) {
            break;
        }
    }
    return starts;
}

std::uint8_t residue(collatz::UInt128 value, std::uint8_t modulus) {
    return static_cast<std::uint8_t>(value % modulus);
}

std::uint32_t node_id_for(
    std::vector<Node> &nodes,
    std::unordered_map<std::string, std::uint32_t> &node_ids,
    collatz::UInt128 value) {
    const std::string key = collatz::uint128_to_decimal(value);
    auto found = node_ids.find(key);
    if (found != node_ids.end()) {
        return found->second;
    }
    const auto id = static_cast<std::uint32_t>(nodes.size());
    node_ids.emplace(key, id);
    nodes.push_back({id, value, collatz::log2_uint128(value), false, value == 1, 0, 0});
    return id;
}

void write_nodes(const std::string &path, const std::vector<Node> &nodes) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open nodes output: " + path);
    }
    out << std::setprecision(10);
    out << "id,value,log2_value,parity,residue_mod3,residue_mod4,residue_mod8,residue_mod16,residue_mod32,is_start,is_terminal,in_degree,out_degree\n";
    for (const auto &node : nodes) {
        out << node.id << ','
            << collatz::uint128_to_decimal(node.value) << ','
            << static_cast<double>(node.log2_value) << ','
            << static_cast<unsigned>(node.value & 1u) << ','
            << static_cast<unsigned>(residue(node.value, 3)) << ','
            << static_cast<unsigned>(residue(node.value, 4)) << ','
            << static_cast<unsigned>(residue(node.value, 8)) << ','
            << static_cast<unsigned>(residue(node.value, 16)) << ','
            << static_cast<unsigned>(residue(node.value, 32)) << ','
            << (node.is_start ? 1 : 0) << ','
            << (node.is_terminal ? 1 : 0) << ','
            << node.in_degree << ','
            << node.out_degree << '\n';
    }
}

void write_edges(const std::string &path, const std::vector<Edge> &edges) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open edges output: " + path);
    }
    out << "source,target\n";
    for (const auto &edge : edges) {
        out << edge.source << ',' << edge.target << '\n';
    }
}

void write_starts(const std::string &path, const std::vector<StartPath> &starts) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open starts output: " + path);
    }
    out << "n,category,start_node_id,steps,overflow\n";
    for (const auto &start : starts) {
        out << start.n << ','
            << start.category << ','
            << start.start_node_id << ','
            << start.steps << ','
            << (start.overflow ? 1 : 0) << '\n';
    }
}

void write_manifest(
    const Options &options,
    const std::vector<Node> &nodes,
    const std::vector<Edge> &edges,
    const std::vector<StartPath> &starts) {
    const std::string path = path_join(options.output_dir, "trajectory_graph.json");
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open graph manifest: " + path);
    }

    out << std::setprecision(10);
    out << "{\n"
        << "  \"dataset_type\": \"collatz_gnn_graph\",\n"
        << "  \"tool\": \"collatz_graph_export\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"gnn_ready\": true,\n"
        << "  \"graph_kind\": \"directed_collatz_trajectory_dag_with_shared_tails\",\n"
        << "  \"start_count\": " << starts.size() << ",\n"
        << "  \"node_count\": " << nodes.size() << ",\n"
        << "  \"edge_count\": " << edges.size() << ",\n"
        << "  \"max_steps\": " << options.max_steps << ",\n"
        << "  \"files\": {\"nodes\": \"nodes.csv\", \"edges\": \"edges.csv\", \"starts\": \"starts.csv\"},\n"
        << "  \"node_feature_schema\": [\"log2_value\", \"parity\", \"residue_mod3\", \"residue_mod4\", \"residue_mod8\", \"residue_mod16\", \"residue_mod32\", \"is_start\", \"is_terminal\", \"in_degree\", \"out_degree\"],\n"
        << "  \"preview\": {\n"
        << "    \"nodes\": [\n";
    const auto node_limit = std::min(options.preview_nodes, nodes.size());
    for (std::size_t i = 0; i < node_limit; ++i) {
        const auto &node = nodes[i];
        out << "      {\"id\": " << node.id
            << ", \"label\": \"" << collatz::uint128_to_decimal(node.value)
            << "\", \"log2\": " << static_cast<double>(node.log2_value)
            << ", \"residue_mod32\": " << static_cast<unsigned>(residue(node.value, 32))
            << ", \"is_start\": " << (node.is_start ? "true" : "false")
            << ", \"is_terminal\": " << (node.is_terminal ? "true" : "false") << "}";
        if (i + 1 != node_limit) {
            out << ',';
        }
        out << '\n';
    }
    out << "    ],\n"
        << "    \"edges\": [\n";
    const auto edge_limit = std::min(options.preview_edges, edges.size());
    for (std::size_t i = 0; i < edge_limit; ++i) {
        const auto &edge = edges[i];
        out << "      {\"source\": " << edge.source << ", \"target\": " << edge.target << "}";
        if (i + 1 != edge_limit) {
            out << ',';
        }
        out << '\n';
    }
    out << "    ]\n"
        << "  }\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::vector<StartSpec> start_specs;
        if (options.n != 0) {
            start_specs.push_back({options.n, "manual"});
        } else {
            start_specs = starts_from_file(options.starts_file, options.count);
        }
        if (start_specs.empty()) {
            throw std::runtime_error("no starts selected");
        }

        std::filesystem::create_directories(options.output_dir);
        std::vector<Node> nodes;
        std::vector<Edge> edges;
        std::vector<StartPath> starts;
        std::unordered_map<std::string, std::uint32_t> node_ids;
        std::unordered_set<std::uint64_t> edge_ids;

        for (const auto &start : start_specs) {
            bool overflow = false;
            const auto path = collatz::generate_path(start.n, options.max_steps, &overflow);
            if (path.empty()) {
                continue;
            }
            const auto start_node_id = node_id_for(nodes, node_ids, path.front().value);
            nodes[start_node_id].is_start = true;
            starts.push_back({start.n, start.category, start_node_id, static_cast<std::uint32_t>(path.size() - 1), overflow});

            for (std::size_t i = 1; i < path.size(); ++i) {
                const auto source = node_id_for(nodes, node_ids, path[i - 1].value);
                const auto target = node_id_for(nodes, node_ids, path[i].value);
                const std::uint64_t edge_key = (static_cast<std::uint64_t>(source) << 32u) | target;
                if (edge_ids.insert(edge_key).second) {
                    edges.push_back({source, target});
                    nodes[source].out_degree += 1;
                    nodes[target].in_degree += 1;
                }
            }
        }

        write_nodes(path_join(options.output_dir, "nodes.csv"), nodes);
        write_edges(path_join(options.output_dir, "edges.csv"), edges);
        write_starts(path_join(options.output_dir, "starts.csv"), starts);
        write_manifest(options, nodes, edges, starts);

        std::cout << "starts=" << starts.size()
                  << " nodes=" << nodes.size()
                  << " edges=" << edges.size()
                  << " output_dir=" << options.output_dir
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
