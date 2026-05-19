#include "collatz/core.hpp"

#include <algorithm>
#include <cmath>
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
    std::string projection = "data/generated/topology/projection.csv";
    std::string source_samples = "data/source_validation/reference_samples.csv";
    std::string output_dir = "data/generated/source_alignment";
    std::size_t neighbors = 8;
};

struct Point {
    std::uint64_t n = 0;
    double x = 0.0;
    double y = 0.0;
    std::size_t cluster = 0;
};

struct SourceTarget {
    std::string source;
    std::uint64_t n = 0;
    std::uint32_t total_steps = 0;
    std::uint64_t peak_low = 0;
};

struct Neighbor {
    const Point *point = nullptr;
    double distance = 0.0;
};

void usage(std::ostream &out) {
    out << "usage: collatz_source_align [options]\n\n"
        << "options:\n"
        << "  --projection FILE       projection.csv from collatz_embedding_analyze\n"
        << "  --source-samples FILE   source validation CSV\n"
        << "  --output-dir DIR        output directory (default data/generated/source_alignment)\n"
        << "  --neighbors N           nearest neighbors per matched source target (default 8)\n";
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
        if (arg == "--projection") {
            options.projection = need_value("--projection");
        } else if (arg == "--source-samples") {
            options.source_samples = need_value("--source-samples");
        } else if (arg == "--output-dir") {
            options.output_dir = need_value("--output-dir");
        } else if (arg == "--neighbors") {
            const auto value = collatz::parse_u64(need_value("--neighbors"));
            if (!value || *value == 0) {
                throw std::runtime_error("--neighbors must be a positive integer");
            }
            options.neighbors = static_cast<std::size_t>(*value);
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
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

std::vector<Point> read_projection(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open projection input: " + path);
    }
    std::string line;
    if (!std::getline(in, line) || line != "n,x,y,cluster") {
        throw std::runtime_error("projection input must start with n,x,y,cluster");
    }
    std::vector<Point> points;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() != 4) {
            throw std::runtime_error("projection row has wrong column count");
        }
        const auto n = collatz::parse_u64(parts[0]);
        const auto cluster = collatz::parse_u64(parts[3]);
        if (!n || *n == 0 || !cluster) {
            throw std::runtime_error("projection row has invalid n or cluster");
        }
        points.push_back({*n, std::stod(parts[1]), std::stod(parts[2]), static_cast<std::size_t>(*cluster)});
    }
    if (points.empty()) {
        throw std::runtime_error("no projection points read");
    }
    return points;
}

std::vector<SourceTarget> read_source_targets(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open source samples: " + path);
    }
    std::string line;
    if (!std::getline(in, line) || line != "source,n,total_steps,peak_low") {
        throw std::runtime_error("source sample input must start with source,n,total_steps,peak_low");
    }
    std::vector<SourceTarget> targets;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() < 4) {
            throw std::runtime_error("bad source target row: " + line);
        }
        const auto n = collatz::parse_u64(parts[1]);
        const auto steps = collatz::parse_u32(parts[2]);
        const auto peak = collatz::parse_u64(parts[3]);
        if (!n || *n == 0 || !steps || !peak) {
            throw std::runtime_error("bad numeric source target row: " + line);
        }
        targets.push_back({parts[0], *n, *steps, *peak});
    }
    if (targets.empty()) {
        throw std::runtime_error("no source targets read");
    }
    return targets;
}

double distance(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<Neighbor> nearest_neighbors(const std::vector<Point> &points, const Point &center, std::size_t limit) {
    std::vector<Neighbor> neighbors;
    neighbors.reserve(points.size());
    for (const auto &point : points) {
        if (point.n == center.n) {
            continue;
        }
        neighbors.push_back({&point, distance(center.x, center.y, point.x, point.y)});
    }
    const auto keep = std::min(limit, neighbors.size());
    std::partial_sort(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(keep), neighbors.end(),
                      [](const Neighbor &left, const Neighbor &right) {
                          return left.distance < right.distance;
                      });
    neighbors.resize(keep);
    return neighbors;
}

void write_alignment(const Options &options, const std::vector<Point> &points, const std::vector<SourceTarget> &targets) {
    const std::string summary_path = path_join(options.output_dir, "source_alignment.json");
    const std::string detail_path = path_join(options.output_dir, "source_targets.csv");
    collatz::ensure_parent_dir(summary_path);
    std::filesystem::create_directories(options.output_dir);

    std::unordered_map<std::uint64_t, const Point *> point_by_n;
    point_by_n.reserve(points.size());
    for (const auto &point : points) {
        point_by_n[point.n] = &point;
    }

    std::ofstream detail(detail_path);
    if (!detail) {
        throw std::runtime_error("failed to open source alignment detail output: " + detail_path);
    }
    detail << "source,n,status,total_steps,peak_low,cluster,neighbor_count,mean_neighbor_distance,nearest_n,nearest_distance\n";

    std::size_t matched = 0;
    std::size_t missing = 0;
    std::set<std::string> sources;
    std::set<std::size_t> matched_clusters;
    std::map<std::size_t, std::size_t> matched_by_cluster;
    std::uint64_t max_source_n = 0;
    double all_mean_distance = 0.0;
    std::vector<std::string> target_json;

    for (const auto &target : targets) {
        sources.insert(target.source);
        max_source_n = std::max(max_source_n, target.n);
        const auto found = point_by_n.find(target.n);
        if (found == point_by_n.end()) {
            ++missing;
            detail << target.source << ',' << target.n << ",missing," << target.total_steps << ',' << target.peak_low
                   << ",,,,,\n";
            std::ostringstream item;
            item << "{\"source\":\"" << collatz::json_escape(target.source) << "\",\"n\":" << target.n
                 << ",\"status\":\"missing\"}";
            target_json.push_back(item.str());
            continue;
        }

        ++matched;
        const Point &point = *found->second;
        matched_clusters.insert(point.cluster);
        ++matched_by_cluster[point.cluster];
        auto neighbors = nearest_neighbors(points, point, options.neighbors);
        double mean_distance = 0.0;
        for (const auto &neighbor : neighbors) {
            mean_distance += neighbor.distance;
        }
        if (!neighbors.empty()) {
            mean_distance /= static_cast<double>(neighbors.size());
        }
        all_mean_distance += mean_distance;
        const std::uint64_t nearest_n = neighbors.empty() ? 0 : neighbors.front().point->n;
        const double nearest_distance = neighbors.empty() ? 0.0 : neighbors.front().distance;
        detail << target.source << ',' << target.n << ",matched," << target.total_steps << ',' << target.peak_low
               << ',' << point.cluster << ',' << neighbors.size() << ',' << mean_distance
               << ',' << nearest_n << ',' << nearest_distance << '\n';

        std::ostringstream item;
        item << "{\"source\":\"" << collatz::json_escape(target.source)
             << "\",\"n\":" << target.n
             << ",\"status\":\"matched\""
             << ",\"total_steps\":" << target.total_steps
             << ",\"peak_low\":" << target.peak_low
             << ",\"cluster\":" << point.cluster
             << ",\"neighbor_count\":" << neighbors.size()
             << ",\"mean_neighbor_distance\":" << mean_distance
             << ",\"nearest_n\":" << nearest_n
             << ",\"nearest_distance\":" << nearest_distance
             << "}";
        target_json.push_back(item.str());
    }

    if (matched != 0) {
        all_mean_distance /= static_cast<double>(matched);
    }
    const bool source_smoke_only = max_source_n <= 1000 || targets.size() < 25;
    const std::string alignment_status = matched == targets.size()
                                             ? (source_smoke_only ? "source-smoke-aligned" : "public-source-aligned")
                                             : matched > 0 ? "partial-source-match"
                                                           : "missing-source-targets";
    const std::string limit = source_smoke_only
                                  ? "Current source target set is a smoke check, not a large public import."
                                  : matched == targets.size()
                                        ? "Current source alignment uses public imported targets; Roosendaal, Oliveira e Silva, and Barina imports remain the next confidence gate."
                                        : "Some public source targets were not present in the current topology sample; either expand topology coverage or use a source target file scoped to embedded starts.";

    std::ofstream out(summary_path);
    if (!out) {
        throw std::runtime_error("failed to open source alignment summary output: " + summary_path);
    }
    out << std::setprecision(10);
    out << "{\n"
        << "  \"dataset_type\": \"collatz_source_alignment\",\n"
        << "  \"tool\": \"collatz_source_align\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"projection\": \"" << collatz::json_escape(options.projection) << "\",\n"
        << "  \"source_samples\": \"" << collatz::json_escape(options.source_samples) << "\",\n"
        << "  \"alignment_status\": \"" << alignment_status << "\",\n"
        << "  \"target_count\": " << targets.size() << ",\n"
        << "  \"matched_targets\": " << matched << ",\n"
        << "  \"missing_targets\": " << missing << ",\n"
        << "  \"source_count\": " << sources.size() << ",\n"
        << "  \"matched_cluster_count\": " << matched_clusters.size() << ",\n"
        << "  \"source_smoke_only\": " << (source_smoke_only ? "true" : "false") << ",\n"
        << "  \"mean_neighbor_distance\": " << all_mean_distance << ",\n"
        << "  \"limit\": \"" << collatz::json_escape(limit) << "\",\n"
        << "  \"next_action\": \"" << (alignment_status == "public-source-aligned"
                                           ? "Add Roosendaal, Oliveira e Silva, and Barina record imports, then test whether source neighborhoods stay stable."
                                           : "Import larger dated source-record tables and compare their neighborhoods before promoting to source-aligned candidate.")
        << "\",\n"
        << "  \"files\": {\"source_targets\": \"source_targets.csv\"},\n"
        << "  \"targets\": [\n";
    for (std::size_t i = 0; i < target_json.size(); ++i) {
        out << "    " << target_json[i];
        if (i + 1 != target_json.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto points = read_projection(options.projection);
        const auto targets = read_source_targets(options.source_samples);
        write_alignment(options, points, targets);
        std::cout << "source_targets=" << targets.size()
                  << " projection_points=" << points.size()
                  << " output_dir=" << options.output_dir
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
