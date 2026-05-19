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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string projection = "data/generated/topology/projection.csv";
    std::string clusters = "data/generated/topology/clusters.csv";
    std::string scan_metadata = "data/generated/features.bin.metadata.json";
    std::string output_dir = "data/generated/insights";
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

struct ClusterInsight {
    std::size_t cluster = 0;
    std::uint64_t center_n = 0;
    double mean_distance = 0.0;
    double max_distance = 0.0;
    double purity = 0.0;
};

struct Finding {
    std::string severity;
    std::string title;
    std::string message;
};

std::string fixed_percent(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << (value * 100.0) << "%";
    return out.str();
}

void usage(std::ostream &out) {
    out << "usage: collatz_insight_analyze [options]\n\n"
        << "options:\n"
        << "  --projection FILE     projection.csv from collatz_embedding_analyze\n"
        << "  --clusters FILE       clusters.csv from collatz_embedding_analyze\n"
        << "  --scan-metadata FILE  scan metadata sidecar\n"
        << "  --output-dir DIR      output directory (default data/generated/insights)\n"
        << "  --neighbors N         local neighbors per cluster center (default 12)\n";
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
        } else if (arg == "--scan-metadata") {
            options.scan_metadata = need_value("--scan-metadata");
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

std::string read_file_or_empty(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

std::uint64_t json_u64_or_zero(const std::string &json, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return 0;
    }
    const auto colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return 0;
    }
    std::size_t start = colon + 1;
    while (start < json.size() && (json[start] == ' ' || json[start] == '\n' || json[start] == '\t')) {
        ++start;
    }
    std::size_t end = start;
    while (end < json.size() && json[end] >= '0' && json[end] <= '9') {
        ++end;
    }
    if (end <= start) {
        return 0;
    }
    const auto value = collatz::parse_u64(std::string_view(json).substr(start, end - start));
    return value.value_or(0);
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
        if (!id || !count || !representative_n) {
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

ClusterInsight analyze_cluster(const std::vector<Point> &points, const Cluster &cluster, std::size_t neighbors) {
    const Point *center = find_point(points, cluster.representative_n);
    const double cx = center ? center->x : cluster.cx;
    const double cy = center ? center->y : cluster.cy;

    struct Candidate {
        const Point *point = nullptr;
        double distance = 0.0;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(points.size());
    for (const auto &point : points) {
        if (point.n == cluster.representative_n) {
            continue;
        }
        candidates.push_back({&point, distance(cx, cy, point.x, point.y)});
    }
    const auto keep = std::min(neighbors, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep), candidates.end(),
                      [](const Candidate &left, const Candidate &right) {
                          return left.distance < right.distance;
                      });
    candidates.resize(keep);

    ClusterInsight insight;
    insight.cluster = cluster.id;
    insight.center_n = cluster.representative_n;
    std::size_t same_cluster = 0;
    for (const auto &candidate : candidates) {
        insight.mean_distance += candidate.distance;
        insight.max_distance = std::max(insight.max_distance, candidate.distance);
        if (candidate.point->cluster == cluster.id) {
            ++same_cluster;
        }
    }
    if (!candidates.empty()) {
        insight.mean_distance /= static_cast<double>(candidates.size());
        insight.purity = static_cast<double>(same_cluster) / static_cast<double>(candidates.size());
    }
    return insight;
}

void write_outputs(
    const Options &options,
    const std::vector<Point> &points,
    const std::vector<Cluster> &clusters,
    const std::vector<ClusterInsight> &cluster_insights,
    const std::vector<Finding> &findings,
    std::uint64_t scan_records) {
    std::filesystem::create_directories(options.output_dir);
    const std::string json_path = path_join(options.output_dir, "insights.json");
    const std::string md_path = path_join(options.output_dir, "insights.md");

    const bool sample_limited = scan_records > points.size();
    const double coverage = scan_records == 0 ? 0.0 : static_cast<double>(points.size()) / static_cast<double>(scan_records);
    const double mean_purity = std::accumulate(cluster_insights.begin(), cluster_insights.end(), 0.0, [](double acc, const ClusterInsight &insight) {
        return acc + insight.purity;
    }) / static_cast<double>(std::max<std::size_t>(1, cluster_insights.size()));
    const std::string conclusion =
        sample_limited
            ? "Early signal: the sampled Collatz paths are forming stable behavior families, but this is not a full-range conclusion yet."
            : "Signal: the embedded Collatz paths are forming stable behavior families across the analyzed range.";
    const std::string meaning =
        "Each colored topology band is a group of starting numbers whose path metrics look alike. " +
        fixed_percent(mean_purity) +
        " local-neighborhood purity means the nearest examples around cluster centers stayed in the same family.";
    const std::string limit =
        sample_limited
            ? "The topology currently covers " + std::to_string(points.size()) + " embedded starts out of " +
                  std::to_string(scan_records) + " scanned starts (" + fixed_percent(coverage) +
                  "), so it proves the pipeline and sample structure, not a Collatz law."
            : "The topology covers the analyzed scan range, but it is still an empirical pattern map, not a mathematical proof.";
    const std::string next_step =
        "Next test: build a stratified embedding sample from the full scan: record setters, high peaks, long stopping times, residue classes, random starts, and loose-family boundary candidates.";
    const std::string confidence =
        sample_limited ? "Promising, sample-limited" : "Promising, range-limited";

    {
        std::ofstream out(json_path);
        if (!out) {
            throw std::runtime_error("failed to open insights output: " + json_path);
        }
        out << std::setprecision(10);
        out << "{\n"
            << "  \"dataset_type\": \"collatz_ai_insights\",\n"
            << "  \"tool\": \"collatz_insight_analyze\",\n"
            << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
            << "  \"projection\": \"" << collatz::json_escape(options.projection) << "\",\n"
            << "  \"clusters\": \"" << collatz::json_escape(options.clusters) << "\",\n"
            << "  \"scan_records\": " << scan_records << ",\n"
            << "  \"topology_points\": " << points.size() << ",\n"
            << "  \"cluster_count\": " << clusters.size() << ",\n"
            << "  \"neighbors_per_cluster\": " << options.neighbors << ",\n"
            << "  \"insight_count\": " << findings.size() << ",\n"
            << "  \"confidence\": \"" << collatz::json_escape(confidence) << "\",\n"
            << "  \"conclusion\": \"" << collatz::json_escape(conclusion) << "\",\n"
            << "  \"meaning\": \"" << collatz::json_escape(meaning) << "\",\n"
            << "  \"limit\": \"" << collatz::json_escape(limit) << "\",\n"
            << "  \"next_step\": \"" << collatz::json_escape(next_step) << "\",\n"
            << "  \"findings\": [\n";
        for (std::size_t i = 0; i < findings.size(); ++i) {
            const auto &finding = findings[i];
            out << "    {\"severity\": \"" << collatz::json_escape(finding.severity)
                << "\", \"title\": \"" << collatz::json_escape(finding.title)
                << "\", \"message\": \"" << collatz::json_escape(finding.message) << "\"}";
            if (i + 1 != findings.size()) {
                out << ',';
            }
            out << '\n';
        }
        out << "  ],\n"
            << "  \"cluster_diagnostics\": [\n";
        for (std::size_t i = 0; i < cluster_insights.size(); ++i) {
            const auto &insight = cluster_insights[i];
            out << "    {\"cluster\": " << insight.cluster
                << ", \"center_n\": " << insight.center_n
                << ", \"mean_neighbor_distance\": " << insight.mean_distance
                << ", \"max_neighbor_distance\": " << insight.max_distance
                << ", \"neighbor_purity\": " << insight.purity << "}";
            if (i + 1 != cluster_insights.size()) {
                out << ',';
            }
            out << '\n';
        }
        out << "  ],\n"
            << "  \"next_experiments\": [\n"
            << "    \"Export a stratified topology sample from the full 100M scan: record setters, high peaks, residue strata, and random rows.\",\n"
            << "    \"Use same-neighborhood pairs as positives and cross-neighborhood pairs as negatives for contrastive training.\",\n"
            << "    \"Compare neighborhood purity against parity-run, residue-transition, and path-image embeddings.\"\n"
            << "  ]\n"
            << "}\n";
    }

    {
        std::ofstream out(md_path);
        if (!out) {
            throw std::runtime_error("failed to open insights markdown: " + md_path);
        }
        out << "# Collatz AI Insights\n\n";
        out << "## Current Conclusion\n\n" << conclusion << "\n\n";
        out << "## What It Means\n\n" << meaning << "\n\n";
        out << "## Current Limit\n\n" << limit << "\n\n";
        out << "## Next Test\n\n" << next_step << "\n\n";
        out << "## Supporting Findings\n\n";
        for (const auto &finding : findings) {
            out << "- **" << finding.title << "**: " << finding.message << "\n";
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const auto points = read_projection(options.projection);
        const auto clusters = read_clusters(options.clusters);
        const std::uint64_t scan_records =
            json_u64_or_zero(read_file_or_empty(options.scan_metadata), "dataset_records_observed");

        std::vector<ClusterInsight> cluster_insights;
        cluster_insights.reserve(clusters.size());
        for (const auto &cluster : clusters) {
            cluster_insights.push_back(analyze_cluster(points, cluster, options.neighbors));
        }

        const auto minmax_cluster = std::minmax_element(clusters.begin(), clusters.end(), [](const Cluster &left, const Cluster &right) {
            return left.count < right.count;
        });
        const auto loose = std::max_element(cluster_insights.begin(), cluster_insights.end(), [](const ClusterInsight &left, const ClusterInsight &right) {
            return left.mean_distance < right.mean_distance;
        });
        const auto tight = std::min_element(cluster_insights.begin(), cluster_insights.end(), [](const ClusterInsight &left, const ClusterInsight &right) {
            return left.mean_distance < right.mean_distance;
        });
        const double mean_purity = std::accumulate(cluster_insights.begin(), cluster_insights.end(), 0.0, [](double acc, const ClusterInsight &insight) {
            return acc + insight.purity;
        }) / static_cast<double>(std::max<std::size_t>(1, cluster_insights.size()));

        std::vector<Finding> findings;
        if (scan_records > points.size()) {
            findings.push_back({
                "high",
                "Topology Sample Scope",
                "The current topology map covers " + std::to_string(points.size()) +
                    " embedded rows while the scan has " + std::to_string(scan_records) +
                    " rows, so conclusions are about the embedding sample, not the full scan yet.",
            });
        }
        findings.push_back({
            "medium",
            "Neighborhood Stability",
            "Cluster-representative nearest-neighbor purity is " + fixed_percent(mean_purity) +
                " across " + std::to_string(cluster_insights.size()) +
                " neighborhoods, which suggests the baseline features are separating repeatable path families.",
        });
        if (loose != cluster_insights.end()) {
            findings.push_back({
                "medium",
                "Loosest Family",
                "Cluster " + std::to_string(loose->cluster) + " around n=" + std::to_string(loose->center_n) +
                    " has the widest local neighborhood, making it a useful anomaly or boundary candidate.",
            });
        }
        if (tight != cluster_insights.end()) {
            findings.push_back({
                "low",
                "Tightest Family",
                "Cluster " + std::to_string(tight->cluster) + " around n=" + std::to_string(tight->center_n) +
                    " has the tightest local neighborhood and is a good positive-pair source for contrastive training.",
            });
        }
        if (minmax_cluster.first != clusters.end() && minmax_cluster.second != clusters.end()) {
            const double imbalance =
                static_cast<double>(minmax_cluster.second->count) / static_cast<double>(std::max<std::size_t>(1, minmax_cluster.first->count));
            findings.push_back({
                "low",
                "Cluster Imbalance",
                "The largest cluster has " + std::to_string(minmax_cluster.second->count) +
                    " rows and the smallest has " + std::to_string(minmax_cluster.first->count) +
                    ", a " + std::to_string(imbalance).substr(0, 4) + "x spread worth tracking across larger samples.",
            });
        }

        write_outputs(options, points, clusters, cluster_insights, findings, scan_records);
        std::cout << "points=" << points.size()
                  << " clusters=" << clusters.size()
                  << " findings=" << findings.size()
                  << " output_dir=" << options.output_dir
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
