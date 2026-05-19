#include "collatz/core.hpp"
#include "collatz/ml.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string input = "data/generated/ml/metrics.csv";
    std::string output_dir = "data/generated/topology";
    std::uint64_t limit = 0;
    std::size_t clusters = 16;
    std::size_t iterations = 30;
    std::size_t preview_points = 5000;
};

struct Point {
    std::uint64_t n = 0;
    std::vector<double> values;
    double x = 0.0;
    double y = 0.0;
    std::size_t cluster = 0;
};

struct ClusterSummary {
    std::size_t id = 0;
    std::size_t count = 0;
    double cx = 0.0;
    double cy = 0.0;
    std::uint64_t representative_n = 0;
    double representative_distance = 0.0;
};

void usage(std::ostream &out) {
    out << "usage: collatz_embedding_analyze [options]\n\n"
        << "options:\n"
        << "  --input FILE       metrics.csv from collatz_embed_export\n"
        << "  --output-dir DIR   output directory (default data/generated/topology)\n"
        << "  --limit N          max rows to read, 0 means all\n"
        << "  --clusters N       k-means clusters (default 16)\n"
        << "  --iterations N     k-means iterations (default 30)\n"
        << "  --preview-points N preview points embedded in JSON (default 5000)\n";
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
        } else if (arg == "--limit") {
            const auto value = collatz::parse_u64(need_value("--limit"));
            if (!value) {
                throw std::runtime_error("--limit must be an integer");
            }
            options.limit = *value;
        } else if (arg == "--clusters") {
            const auto value = collatz::parse_u64(need_value("--clusters"));
            if (!value || *value == 0) {
                throw std::runtime_error("--clusters must be a positive integer");
            }
            options.clusters = static_cast<std::size_t>(*value);
        } else if (arg == "--iterations") {
            const auto value = collatz::parse_u64(need_value("--iterations"));
            if (!value || *value == 0) {
                throw std::runtime_error("--iterations must be a positive integer");
            }
            options.iterations = static_cast<std::size_t>(*value);
        } else if (arg == "--preview-points") {
            const auto value = collatz::parse_u64(need_value("--preview-points"));
            if (!value) {
                throw std::runtime_error("--preview-points must be an integer");
            }
            options.preview_points = static_cast<std::size_t>(*value);
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

std::vector<Point> read_points(const Options &options, std::size_t &dims) {
    std::ifstream in(options.input);
    if (!in) {
        throw std::runtime_error("failed to open input: " + options.input);
    }
    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("metrics input is empty");
    }
    const auto header = split_csv_line(line);
    if (header.size() < 2 || header.front() != "n") {
        throw std::runtime_error("metrics input must start with n,m0,... header");
    }
    dims = header.size() - 1;

    std::vector<Point> points;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() != dims + 1) {
            throw std::runtime_error("metrics row has wrong column count");
        }
        const auto n = collatz::parse_u64(parts[0]);
        if (!n || *n == 0) {
            throw std::runtime_error("metrics row has invalid n");
        }
        Point point;
        point.n = *n;
        point.values.resize(dims);
        for (std::size_t i = 0; i < dims; ++i) {
            point.values[i] = std::stod(parts[i + 1]);
        }
        points.push_back(std::move(point));
        if (options.limit != 0 && points.size() >= options.limit) {
            break;
        }
    }
    if (points.empty()) {
        throw std::runtime_error("no metrics rows read");
    }
    return points;
}

std::vector<double> compute_mean(const std::vector<Point> &points, std::size_t dims) {
    std::vector<double> mean(dims, 0.0);
    for (const auto &point : points) {
        for (std::size_t i = 0; i < dims; ++i) {
            mean[i] += point.values[i];
        }
    }
    for (auto &value : mean) {
        value /= static_cast<double>(points.size());
    }
    return mean;
}

std::vector<double> covariance(const std::vector<Point> &points, const std::vector<double> &mean, std::size_t dims) {
    std::vector<double> cov(dims * dims, 0.0);
    const double denom = std::max<std::size_t>(1, points.size() - 1);
    for (const auto &point : points) {
        for (std::size_t r = 0; r < dims; ++r) {
            const double rv = point.values[r] - mean[r];
            for (std::size_t c = 0; c < dims; ++c) {
                cov[r * dims + c] += rv * (point.values[c] - mean[c]) / denom;
            }
        }
    }
    return cov;
}

std::vector<double> mat_vec(const std::vector<double> &matrix, const std::vector<double> &vector, std::size_t dims) {
    std::vector<double> out(dims, 0.0);
    for (std::size_t r = 0; r < dims; ++r) {
        for (std::size_t c = 0; c < dims; ++c) {
            out[r] += matrix[r * dims + c] * vector[c];
        }
    }
    return out;
}

double dot(const std::vector<double> &left, const std::vector<double> &right) {
    double value = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        value += left[i] * right[i];
    }
    return value;
}

void normalize(std::vector<double> &vector) {
    const double norm = std::sqrt(std::max(0.0, dot(vector, vector)));
    if (norm <= std::numeric_limits<double>::epsilon()) {
        return;
    }
    for (auto &value : vector) {
        value /= norm;
    }
}

std::vector<double> power_vector(const std::vector<double> &matrix, std::size_t dims, std::size_t seed_offset) {
    std::vector<double> vector(dims, 0.0);
    for (std::size_t i = 0; i < dims; ++i) {
        vector[i] = static_cast<double>(((i + 1 + seed_offset) % 7) + 1);
    }
    normalize(vector);
    for (std::size_t iter = 0; iter < 80; ++iter) {
        vector = mat_vec(matrix, vector, dims);
        normalize(vector);
    }
    return vector;
}

void project_points(std::vector<Point> &points, const std::vector<double> &mean, std::size_t dims) {
    auto cov = covariance(points, mean, dims);
    auto pc1 = power_vector(cov, dims, 0);
    const double lambda1 = dot(pc1, mat_vec(cov, pc1, dims));
    for (std::size_t r = 0; r < dims; ++r) {
        for (std::size_t c = 0; c < dims; ++c) {
            cov[r * dims + c] -= lambda1 * pc1[r] * pc1[c];
        }
    }
    auto pc2 = power_vector(cov, dims, 3);

    for (auto &point : points) {
        std::vector<double> centered(dims, 0.0);
        for (std::size_t i = 0; i < dims; ++i) {
            centered[i] = point.values[i] - mean[i];
        }
        point.x = dot(centered, pc1);
        point.y = dot(centered, pc2);
    }
}

double distance2(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

std::vector<ClusterSummary> assign_clusters(std::vector<Point> &points, std::size_t k, std::size_t iterations) {
    k = std::min(k, points.size());
    std::vector<ClusterSummary> clusters(k);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t index = (i * points.size()) / k;
        clusters[i].id = i;
        clusters[i].cx = points[index].x;
        clusters[i].cy = points[index].y;
    }

    for (std::size_t iter = 0; iter < iterations; ++iter) {
        for (auto &point : points) {
            double best = std::numeric_limits<double>::max();
            std::size_t best_id = 0;
            for (const auto &cluster : clusters) {
                const double dist = distance2(point.x, point.y, cluster.cx, cluster.cy);
                if (dist < best) {
                    best = dist;
                    best_id = cluster.id;
                }
            }
            point.cluster = best_id;
        }

        std::vector<double> sx(k, 0.0);
        std::vector<double> sy(k, 0.0);
        std::vector<std::size_t> counts(k, 0);
        for (const auto &point : points) {
            sx[point.cluster] += point.x;
            sy[point.cluster] += point.y;
            counts[point.cluster] += 1;
        }
        for (auto &cluster : clusters) {
            if (counts[cluster.id] == 0) {
                continue;
            }
            cluster.cx = sx[cluster.id] / static_cast<double>(counts[cluster.id]);
            cluster.cy = sy[cluster.id] / static_cast<double>(counts[cluster.id]);
        }
    }

    for (auto &cluster : clusters) {
        cluster.count = 0;
        cluster.representative_distance = std::numeric_limits<double>::max();
    }
    for (const auto &point : points) {
        auto &cluster = clusters[point.cluster];
        cluster.count += 1;
        const double dist = std::sqrt(distance2(point.x, point.y, cluster.cx, cluster.cy));
        if (dist < cluster.representative_distance) {
            cluster.representative_distance = dist;
            cluster.representative_n = point.n;
        }
    }
    return clusters;
}

void write_projection(const std::string &path, const std::vector<Point> &points) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open projection output: " + path);
    }
    out << std::setprecision(10);
    out << "n,x,y,cluster\n";
    for (const auto &point : points) {
        out << point.n << ',' << point.x << ',' << point.y << ',' << point.cluster << '\n';
    }
}

void write_clusters(const std::string &path, const std::vector<ClusterSummary> &clusters) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open clusters output: " + path);
    }
    out << std::setprecision(10);
    out << "cluster,count,cx,cy,representative_n,representative_distance\n";
    for (const auto &cluster : clusters) {
        out << cluster.id << ','
            << cluster.count << ','
            << cluster.cx << ','
            << cluster.cy << ','
            << cluster.representative_n << ','
            << cluster.representative_distance << '\n';
    }
}

void write_manifest(
    const Options &options,
    const std::vector<Point> &points,
    const std::vector<ClusterSummary> &clusters,
    std::size_t dims) {
    const std::string path = path_join(options.output_dir, "embedding_topology.json");
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open topology manifest: " + path);
    }
    out << std::setprecision(10);
    out << "{\n"
        << "  \"dataset_type\": \"collatz_embedding_topology\",\n"
        << "  \"tool\": \"collatz_embedding_analyze\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"input\": \"" << collatz::json_escape(options.input) << "\",\n"
        << "  \"projection\": \"pca2_metric_baseline\",\n"
        << "  \"point_count\": " << points.size() << ",\n"
        << "  \"metric_dims\": " << dims << ",\n"
        << "  \"cluster_count\": " << clusters.size() << ",\n"
        << "  \"files\": {\"projection\": \"projection.csv\", \"clusters\": \"clusters.csv\"},\n"
        << "  \"clusters\": [\n";
    for (std::size_t i = 0; i < clusters.size(); ++i) {
        const auto &cluster = clusters[i];
        out << "    {\"id\": " << cluster.id
            << ", \"count\": " << cluster.count
            << ", \"cx\": " << cluster.cx
            << ", \"cy\": " << cluster.cy
            << ", \"representative_n\": " << cluster.representative_n << "}";
        if (i + 1 != clusters.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ],\n"
        << "  \"preview_points\": [\n";
    const auto preview_count = std::min(options.preview_points, points.size());
    for (std::size_t i = 0; i < preview_count; ++i) {
        const auto &point = points[i];
        out << "    {\"n\": " << point.n
            << ", \"x\": " << point.x
            << ", \"y\": " << point.y
            << ", \"cluster\": " << point.cluster << "}";
        if (i + 1 != preview_count) {
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
        const Options options = parse_args(argc, argv);
        std::size_t dims = 0;
        auto points = read_points(options, dims);
        const auto mean = compute_mean(points, dims);
        project_points(points, mean, dims);
        auto clusters = assign_clusters(points, options.clusters, options.iterations);

        std::filesystem::create_directories(options.output_dir);
        write_projection(path_join(options.output_dir, "projection.csv"), points);
        write_clusters(path_join(options.output_dir, "clusters.csv"), clusters);
        write_manifest(options, points, clusters, dims);

        std::cout << "points=" << points.size()
                  << " dims=" << dims
                  << " clusters=" << clusters.size()
                  << " output_dir=" << options.output_dir
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
