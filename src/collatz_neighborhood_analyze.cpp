#include "collatz/core.hpp"

#include <algorithm>
#include <cmath>
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
#include <vector>

namespace {

struct Options {
    std::string projection = "data/generated/topology/projection.csv";
    std::string clusters = "data/generated/topology/clusters.csv";
    std::string output = "data/generated/topology/neighborhoods.json";
    std::size_t neighbors = 12;
};

struct Point {
    std::uint64_t n = 0;
    double x = 0.0;
    double y = 0.0;
    std::size_t cluster = 0;
};

struct Cluster {
    std::size_t id = 0;
    std::size_t count = 0;
    double cx = 0.0;
    double cy = 0.0;
    std::uint64_t representative_n = 0;
};

struct Neighbor {
    const Point *point = nullptr;
    double distance = 0.0;
};

void usage(std::ostream &out) {
    out << "usage: collatz_neighborhood_analyze [options]\n\n"
        << "options:\n"
        << "  --projection FILE  projection.csv from collatz_embedding_analyze\n"
        << "  --clusters FILE    clusters.csv from collatz_embedding_analyze\n"
        << "  --output FILE      JSON output path\n"
        << "  --neighbors N      nearest neighbors per cluster representative (default 12)\n";
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
        } else if (arg == "--clusters") {
            options.clusters = need_value("--clusters");
        } else if (arg == "--output") {
            options.output = need_value("--output");
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

std::vector<Cluster> read_clusters(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open clusters input: " + path);
    }
    std::string line;
    if (!std::getline(in, line) || line != "cluster,count,cx,cy,representative_n,representative_distance") {
        throw std::runtime_error("clusters input has unexpected header");
    }
    std::vector<Cluster> clusters;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() != 6) {
            throw std::runtime_error("cluster row has wrong column count");
        }
        const auto id = collatz::parse_u64(parts[0]);
        const auto count = collatz::parse_u64(parts[1]);
        const auto representative_n = collatz::parse_u64(parts[4]);
        if (!id || !count || !representative_n || *representative_n == 0) {
            throw std::runtime_error("cluster row has invalid numeric fields");
        }
        clusters.push_back({
            static_cast<std::size_t>(*id),
            static_cast<std::size_t>(*count),
            std::stod(parts[2]),
            std::stod(parts[3]),
            *representative_n,
        });
    }
    if (clusters.empty()) {
        throw std::runtime_error("no clusters read");
    }
    return clusters;
}

double distance(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

const Point *find_point(const std::vector<Point> &points, std::uint64_t n) {
    const auto it = std::find_if(points.begin(), points.end(), [&](const Point &point) {
        return point.n == n;
    });
    return it == points.end() ? nullptr : &*it;
}

std::vector<Neighbor> nearest_neighbors(const std::vector<Point> &points, const Cluster &cluster, std::size_t limit) {
    const Point *center = find_point(points, cluster.representative_n);
    const double center_x = center ? center->x : cluster.cx;
    const double center_y = center ? center->y : cluster.cy;

    std::vector<Neighbor> neighbors;
    neighbors.reserve(points.size());
    for (const auto &point : points) {
        if (point.n == cluster.representative_n) {
            continue;
        }
        neighbors.push_back({&point, distance(center_x, center_y, point.x, point.y)});
    }
    const auto keep = std::min(limit, neighbors.size());
    std::partial_sort(neighbors.begin(), neighbors.begin() + static_cast<std::ptrdiff_t>(keep), neighbors.end(),
                      [](const Neighbor &left, const Neighbor &right) {
                          return left.distance < right.distance;
                      });
    neighbors.resize(keep);
    return neighbors;
}

void write_json(const Options &options, const std::vector<Point> &points, const std::vector<Cluster> &clusters) {
    collatz::ensure_parent_dir(options.output);
    std::ofstream out(options.output);
    if (!out) {
        throw std::runtime_error("failed to open output: " + options.output);
    }

    out << std::setprecision(10);
    out << "{\n"
        << "  \"dataset_type\": \"collatz_embedding_neighborhoods\",\n"
        << "  \"tool\": \"collatz_neighborhood_analyze\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"projection\": \"" << collatz::json_escape(options.projection) << "\",\n"
        << "  \"clusters\": \"" << collatz::json_escape(options.clusters) << "\",\n"
        << "  \"point_count\": " << points.size() << ",\n"
        << "  \"cluster_count\": " << clusters.size() << ",\n"
        << "  \"neighbors_per_center\": " << options.neighbors << ",\n"
        << "  \"neighborhood_count\": " << clusters.size() << ",\n"
        << "  \"neighborhoods\": [\n";

    for (std::size_t i = 0; i < clusters.size(); ++i) {
        const auto &cluster = clusters[i];
        const Point *center = find_point(points, cluster.representative_n);
        const double center_x = center ? center->x : cluster.cx;
        const double center_y = center ? center->y : cluster.cy;
        auto neighbors = nearest_neighbors(points, cluster, options.neighbors);
        double mean_distance = 0.0;
        double max_distance = 0.0;
        for (const auto &neighbor : neighbors) {
            mean_distance += neighbor.distance;
            max_distance = std::max(max_distance, neighbor.distance);
        }
        if (!neighbors.empty()) {
            mean_distance /= static_cast<double>(neighbors.size());
        }

        out << "    {\"kind\": \"cluster_representative\""
            << ", \"cluster\": " << cluster.id
            << ", \"cluster_size\": " << cluster.count
            << ", \"center_n\": " << cluster.representative_n
            << ", \"center_x\": " << center_x
            << ", \"center_y\": " << center_y
            << ", \"neighbor_count\": " << neighbors.size()
            << ", \"mean_distance\": " << mean_distance
            << ", \"max_distance\": " << max_distance
            << ", \"neighbors\": [";
        for (std::size_t n = 0; n < neighbors.size(); ++n) {
            const auto &neighbor = neighbors[n];
            out << "{\"rank\": " << (n + 1)
                << ", \"n\": " << neighbor.point->n
                << ", \"distance\": " << neighbor.distance
                << ", \"cluster\": " << neighbor.point->cluster << "}";
            if (n + 1 != neighbors.size()) {
                out << ", ";
            }
        }
        out << "]}";
        if (i + 1 != clusters.size()) {
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
        const auto clusters = read_clusters(options.clusters);
        write_json(options, points, clusters);
        std::cout << "points=" << points.size()
                  << " clusters=" << clusters.size()
                  << " neighbors=" << options.neighbors
                  << " output=" << options.output
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
