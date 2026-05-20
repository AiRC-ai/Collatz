#include "collatz/core.hpp"
#include "collatz/feature_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
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
    std::string feature_bin;
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
    std::string source_kind;
    std::uint64_t source_rank = 0;
    std::string source_url;
    std::string retrieved_utc;
    std::string parser;
    bool parse_error = false;
    std::string parse_error_detail;
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
        << "  --feature-bin FILE      optional binary feature file for row-level mismatch taxonomy\n"
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
        } else if (arg == "--feature-bin") {
            options.feature_bin = need_value("--feature-bin");
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
    if (!std::getline(in, line)) {
        throw std::runtime_error("source sample input is empty");
    }
    const auto header = split_csv_line(line);
    if (header.size() < 4 || header[0] != "source" || header[1] != "n" ||
        header[2] != "total_steps" || header[3] != "peak_low") {
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
        SourceTarget target;
        target.source = parts[0];
        target.source_kind = parts.size() > 4 ? parts[4] : "";
        if (parts.size() > 5) {
            target.source_rank = collatz::parse_u64(parts[5]).value_or(0);
        }
        target.source_url = parts.size() > 6 ? parts[6] : "";
        target.retrieved_utc = parts.size() > 7 ? parts[7] : "";
        target.parser = parts.size() > 8 ? parts[8] : "";
        const auto n = collatz::parse_u64(parts[1]);
        const auto steps = collatz::parse_u32(parts[2]);
        const auto peak = collatz::parse_u64(parts[3]);
        if (!n || *n == 0 || !steps || !peak) {
            target.parse_error = true;
            target.parse_error_detail = "bad numeric source target row";
        } else {
            target.n = *n;
            target.total_steps = *steps;
            target.peak_low = *peak;
        }
        targets.push_back(target);
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

std::string source_family_for(const std::string &source) {
    if (source.rfind("OEIS_", 0) == 0) {
        return "OEIS";
    }
    if (source.rfind("Roosendaal_", 0) == 0) {
        return "Roosendaal";
    }
    if (source.rfind("Oliveira_e_Silva_", 0) == 0) {
        return "Oliveira_e_Silva";
    }
    if (source.rfind("Barina_", 0) == 0) {
        return "Barina";
    }
    return source;
}

std::string source_family_key(const std::string &source) {
    const auto family = source_family_for(source);
    if (family == "OEIS") {
        return "oeis";
    }
    if (family == "Roosendaal") {
        return "roosendaal";
    }
    if (family == "Oliveira_e_Silva") {
        return "oliveira_e_silva";
    }
    if (family == "Barina") {
        return "barina";
    }
    return family;
}

struct FeatureLookup {
    std::string path;
    collatz::BinaryFeatureHeader header{};
    std::uint64_t records = 0;
    bool enabled = false;
};

FeatureLookup open_feature_lookup(const std::string &path) {
    FeatureLookup lookup;
    lookup.path = path;
    if (path.empty()) {
        return lookup;
    }
    lookup.header = collatz::read_binary_header(path);
    if (!collatz::valid_binary_header(lookup.header)) {
        throw std::runtime_error("feature binary has an incompatible header: " + path);
    }
    lookup.records = collatz::binary_record_count(path);
    lookup.enabled = true;
    return lookup;
}

std::optional<collatz::BinaryFeatureRecord> read_feature_for_n(const FeatureLookup &lookup, std::uint64_t n) {
    if (!lookup.enabled || n < lookup.header.range_start) {
        return std::nullopt;
    }
    const std::uint64_t index = n - lookup.header.range_start;
    if (index >= lookup.records) {
        return std::nullopt;
    }
    std::ifstream in(lookup.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open binary feature file: " + lookup.path);
    }
    const auto offset = static_cast<std::streamoff>(lookup.header.header_size + index * lookup.header.record_size);
    in.seekg(offset);
    collatz::BinaryFeatureRecord record{};
    in.read(reinterpret_cast<char *>(&record), sizeof(record));
    if (!in || record.n != n) {
        return std::nullopt;
    }
    return record;
}

const std::vector<std::string> &reason_buckets() {
    static const std::vector<std::string> buckets = {
        "above_active_scan_range",
        "missing_from_topology_sample",
        "missing_feature_row",
        "parser_error",
        "step_convention_mismatch",
        "peak_convention_mismatch",
        "true_mismatch",
        "missing_topology_projection_node",
        "duplicated_source_row",
        "future_source_target",
        "unknown",
    };
    return buckets;
}

struct FamilyCoverage {
    bool present = false;
    std::size_t targets = 0;
    std::size_t matched = 0;
    std::size_t unmatched = 0;
};

void write_unmatched_row(
    std::ofstream &out,
    const SourceTarget &target,
    std::string_view bucket,
    std::string_view detail,
    std::uint32_t observed_steps,
    std::uint64_t observed_peak,
    std::uint64_t active_start,
    std::uint64_t active_end,
    bool topology_present,
    bool feature_row_present) {
    out << target.source << ','
        << source_family_for(target.source) << ','
        << target.source_kind << ','
        << target.n << ','
        << target.total_steps << ','
        << target.peak_low << ','
        << observed_steps << ','
        << observed_peak << ','
        << bucket << ','
        << collatz::json_escape(detail) << ','
        << target.parser << ','
        << target.source_url << ','
        << target.retrieved_utc << ','
        << active_start << ','
        << active_end << ','
        << (topology_present ? "true" : "false") << ','
        << (feature_row_present ? "true" : "false") << '\n';
}

void write_alignment(const Options &options, const std::vector<Point> &points, const std::vector<SourceTarget> &targets) {
    const std::string summary_path = path_join(options.output_dir, "source_alignment.json");
    const std::string detail_path = path_join(options.output_dir, "source_targets.csv");
    const std::string unmatched_path = path_join(options.output_dir, "unmatched_rows.csv");
    collatz::ensure_parent_dir(summary_path);
    std::filesystem::create_directories(options.output_dir);
    const FeatureLookup feature_lookup = open_feature_lookup(options.feature_bin);
    const std::uint64_t active_start = feature_lookup.enabled ? feature_lookup.header.range_start : 0;
    const std::uint64_t active_end =
        feature_lookup.enabled && feature_lookup.records > 0 ? feature_lookup.header.range_start + feature_lookup.records - 1 : 0;

    std::unordered_map<std::uint64_t, const Point *> point_by_n;
    point_by_n.reserve(points.size());
    for (const auto &point : points) {
        point_by_n[point.n] = &point;
    }

    std::ofstream detail_stream(detail_path);
    if (!detail_stream) {
        throw std::runtime_error("failed to open source alignment detail output: " + detail_path);
    }
    detail_stream << "source,n,status,total_steps,peak_low,cluster,neighbor_count,mean_neighbor_distance,nearest_n,nearest_distance\n";
    std::ofstream unmatched(unmatched_path);
    if (!unmatched) {
        throw std::runtime_error("failed to open source alignment unmatched output: " + unmatched_path);
    }
    unmatched << "source,source_family,source_kind,n,expected_total_steps,expected_peak_low,observed_total_steps,"
              << "observed_peak_low,reason_bucket,reason_detail,parser,source_url,retrieved_at_utc,"
              << "active_range_start,active_range_end,topology_present,feature_row_present\n";

    std::size_t matched = 0;
    std::size_t missing = 0;
    std::set<std::string> sources;
    std::set<std::string> source_families;
    std::set<std::size_t> matched_clusters;
    std::map<std::size_t, std::size_t> matched_by_cluster;
    std::uint64_t max_source_n = 0;
    double all_mean_distance = 0.0;
    std::vector<std::string> target_json;
    std::map<std::string, std::size_t> unmatched_breakdown;
    std::map<std::string, FamilyCoverage> family_coverage;
    for (const auto &bucket : reason_buckets()) {
        unmatched_breakdown[bucket] = 0;
    }
    std::set<std::string> seen_source_rows;

    for (const auto &target : targets) {
        sources.insert(target.source);
        const auto family_name = source_family_for(target.source);
        const auto family_key = source_family_key(target.source);
        source_families.insert(family_name);
        auto &coverage = family_coverage[family_key];
        coverage.present = true;
        ++coverage.targets;
        max_source_n = std::max(max_source_n, target.n);
        const std::string duplicate_key = target.source + ":" + std::to_string(target.n);
        const bool duplicate = seen_source_rows.count(duplicate_key) != 0;
        seen_source_rows.insert(duplicate_key);
        const auto found = point_by_n.find(target.n);
        const bool topology_present = found != point_by_n.end();

        std::optional<collatz::BinaryFeatureRecord> feature;
        bool feature_row_present = false;
        std::uint32_t observed_steps = 0;
        std::uint64_t observed_peak = 0;
        if (feature_lookup.enabled && !target.parse_error && target.n >= active_start && target.n <= active_end) {
            feature = read_feature_for_n(feature_lookup, target.n);
            if (feature) {
                feature_row_present = true;
                observed_steps = feature->total_steps;
                observed_peak = feature->peak_low;
            }
        }
        auto record_unmatched = [&](std::string_view bucket, std::string_view reason_detail) {
            ++missing;
            ++coverage.unmatched;
            ++unmatched_breakdown[std::string(bucket)];
            detail_stream << target.source << ',' << target.n << ",missing," << target.total_steps << ',' << target.peak_low
                          << ",,,,,\n";
            write_unmatched_row(unmatched, target, bucket, reason_detail, observed_steps, observed_peak,
                                active_start, active_end, topology_present, feature_row_present);
            std::ostringstream item;
            item << "{\"source\":\"" << collatz::json_escape(target.source) << "\",\"n\":" << target.n
                 << ",\"status\":\"missing\""
                 << ",\"reason_bucket\":\"" << bucket << "\"}";
            target_json.push_back(item.str());
        };
        if (target.parse_error) {
            record_unmatched("parser_error", target.parse_error_detail);
            continue;
        }
        if (duplicate) {
            record_unmatched("duplicated_source_row", "duplicate source/n row");
            continue;
        }
        if (feature_lookup.enabled && (target.n < active_start || target.n > active_end)) {
            record_unmatched("above_active_scan_range", "source target outside the active binary feature range");
            continue;
        }
        if (feature_lookup.enabled && !feature_row_present) {
            record_unmatched("missing_feature_row", "source target is inside range but no binary feature row was found");
            continue;
        }
        if (feature_row_present && observed_steps != target.total_steps && observed_peak != target.peak_low) {
            record_unmatched("true_mismatch", "binary feature row disagrees with both expected total steps and expected peak");
            continue;
        }
        if (feature_row_present && observed_steps != target.total_steps) {
            record_unmatched("step_convention_mismatch", "binary feature row total steps differs from source expected total steps");
            continue;
        }
        if (feature_row_present && observed_peak != target.peak_low) {
            record_unmatched("peak_convention_mismatch", "binary feature row peak differs from source expected peak");
            continue;
        }
        if (found == point_by_n.end()) {
            record_unmatched(feature_lookup.enabled ? "missing_from_topology_sample" : "missing_topology_projection_node",
                             feature_lookup.enabled
                                 ? "source target exists in the feature file but is absent from the topology sample"
                                 : "source target is absent from the topology projection");
            continue;
        }

        ++matched;
        ++coverage.matched;
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
        detail_stream << target.source << ',' << target.n << ",matched," << target.total_steps << ',' << target.peak_low
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
    const bool all_matched = matched == targets.size();
    const std::size_t supplemental_families =
        (source_families.count("Roosendaal") != 0 ? 1 : 0) +
        (source_families.count("Oliveira_e_Silva") != 0 ? 1 : 0) +
        (source_families.count("Barina") != 0 ? 1 : 0);
    const bool multi_source = source_families.count("OEIS") != 0 && supplemental_families >= 2;
    const std::string alignment_status = all_matched
                                             ? (source_smoke_only ? "source-smoke-aligned"
                                                                  : multi_source ? "multi-source-aligned"
                                                                                 : "public-source-aligned")
                                             : matched > 0 ? "partial-source-match"
                                                           : "missing-source-targets";
    const std::string limit = source_smoke_only
                                  ? "Current source target set is a smoke check, not a large public import."
                                  : all_matched && multi_source
                                        ? "Multiple independent public source families agree with the current topology sample; this remains evidence, not proof."
                                        : all_matched
                                              ? "Current source alignment uses public imported targets from fewer than three source families; add at least two of Roosendaal, Oliveira e Silva, and Barina before stronger promotion."
                                              : "Some public source targets were not present in the current topology sample; the missing count is the next falsification target.";

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
        << "  \"matched_fraction\": " << (targets.empty() ? 0.0 : static_cast<double>(matched) / static_cast<double>(targets.size())) << ",\n"
        << "  \"source_count\": " << sources.size() << ",\n"
        << "  \"source_family_count\": " << source_families.size() << ",\n"
        << "  \"matched_cluster_count\": " << matched_clusters.size() << ",\n"
        << "  \"source_smoke_only\": " << (source_smoke_only ? "true" : "false") << ",\n"
        << "  \"mean_neighbor_distance\": " << all_mean_distance << ",\n"
        << "  \"active_range_start\": " << active_start << ",\n"
        << "  \"active_range_end\": " << active_end << ",\n"
        << "  \"unmatched_breakdown\": {";
    for (std::size_t i = 0; i < reason_buckets().size(); ++i) {
        const auto &bucket = reason_buckets()[i];
        out << "\"" << bucket << "\":" << unmatched_breakdown[bucket];
        if (i + 1 != reason_buckets().size()) {
            out << ',';
        }
    }
    const double unknown_unmatched_percent =
        missing == 0 ? 0.0 : static_cast<double>(unmatched_breakdown["unknown"]) * 100.0 / static_cast<double>(missing);
    out << "},\n"
        << "  \"unknown_unmatched_percent\": " << unknown_unmatched_percent << ",\n"
        << "  \"source_family_coverage\": {\n";
    const std::vector<std::string> public_families = {"oeis", "roosendaal", "oliveira_e_silva", "barina"};
    for (std::size_t i = 0; i < public_families.size(); ++i) {
        const auto &key = public_families[i];
        const auto found_family = family_coverage.find(key);
        const FamilyCoverage empty;
        const auto &family = found_family == family_coverage.end() ? empty : found_family->second;
        out << "    \"" << key << "\": {\"present\":" << (family.present ? "true" : "false")
            << ",\"complete\":" << (family.present && family.unmatched == 0 ? "true" : "false")
            << ",\"targets\":" << family.targets
            << ",\"matched\":" << family.matched
            << ",\"unmatched\":" << family.unmatched
            << "}";
        if (i + 1 != public_families.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  },\n"
        << "  \"limit\": \"" << collatz::json_escape(limit) << "\",\n"
        << "  \"next_action\": \"" << (alignment_status == "multi-source-aligned"
                                           ? "Run source-anchored topology, path-image, and GNN ablations before promoting any stronger research candidate."
                                           : alignment_status == "public-source-aligned"
                                                 ? "Add at least two independent source families beyond OEIS, then test whether source neighborhoods stay stable."
                                                 : alignment_status == "partial-source-match"
                                                       ? "Expand topology coverage or scope source targets to embedded starts, then rerun alignment."
                                                       : "Import larger dated source-record tables and compare their neighborhoods before promoting confidence.")
        << "\",\n"
        << "  \"files\": {\"source_targets\": \"source_targets.csv\", \"unmatched_rows\": \"unmatched_rows.csv\"},\n"
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
