#include "collatz/core.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string full_audit = "data/generated/full_audit/summary.json";
    std::string stratified_metadata = "data/generated/stratified/metadata.json";
    std::string topology_manifest = "data/generated/topology/embedding_topology.json";
    std::string validation_metrics = "data/generated/evidence_validation/metrics.json";
    std::string ablation_report = "data/generated/evidence_validation/ablation_report.csv";
    std::string source_alignment = "data/generated/source_alignment/source_alignment.json";
    std::string runner_status = "data/generated/runner/status.json";
    std::string neural_status = "data/generated/runner/neural_parallel_status.json";
    std::string output = "data/generated/evidence/latest_public_summary.json";
    std::string hypothesis_summary_output = "data/generated/hypotheses/summary.json";
    std::string active_feature_file = "data/generated/features.bin";
    std::string git_commit = "unknown";
};

struct LeaderboardEntry {
    std::string name;
    bool present = false;
    std::uint64_t samples = 0;
    std::optional<double> lift;
};

void usage(std::ostream &out) {
    out << "usage: collatz_evidence_publish [options]\n\n"
        << "options:\n"
        << "  --full-audit FILE              full audit summary JSON\n"
        << "  --stratified-metadata FILE     stratified sample metadata JSON\n"
        << "  --topology-manifest FILE       topology manifest JSON\n"
        << "  --validation-metrics FILE      evidence validation metrics JSON\n"
        << "  --ablation-report FILE         evidence validation ablation CSV\n"
        << "  --source-alignment FILE        source alignment summary JSON\n"
        << "  --runner-status FILE           optional public-safe runner status JSON\n"
        << "  --neural-status FILE           optional public-safe parallel neural status JSON\n"
        << "  --active-feature-file LABEL    relative public feature-file label\n"
        << "  --git-commit SHA               source commit label\n"
        << "  --output FILE                  canonical public summary output\n"
        << "  --hypothesis-summary-output FILE  compact dashboard/hypothesis output; use none to skip\n";
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
        if (arg == "--full-audit") {
            options.full_audit = need_value("--full-audit");
        } else if (arg == "--stratified-metadata") {
            options.stratified_metadata = need_value("--stratified-metadata");
        } else if (arg == "--topology-manifest") {
            options.topology_manifest = need_value("--topology-manifest");
        } else if (arg == "--validation-metrics") {
            options.validation_metrics = need_value("--validation-metrics");
        } else if (arg == "--ablation-report") {
            options.ablation_report = need_value("--ablation-report");
        } else if (arg == "--source-alignment") {
            options.source_alignment = need_value("--source-alignment");
        } else if (arg == "--runner-status") {
            options.runner_status = need_value("--runner-status");
        } else if (arg == "--neural-status") {
            options.neural_status = need_value("--neural-status");
        } else if (arg == "--active-feature-file") {
            options.active_feature_file = need_value("--active-feature-file");
        } else if (arg == "--git-commit") {
            options.git_commit = need_value("--git-commit");
        } else if (arg == "--output") {
            options.output = need_value("--output");
        } else if (arg == "--hypothesis-summary-output") {
            options.hypothesis_summary_output = need_value("--hypothesis-summary-output");
            if (options.hypothesis_summary_output == "none") {
                options.hypothesis_summary_output.clear();
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
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

std::size_t skip_ws(const std::string &text, std::size_t pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
    return pos;
}

std::size_t json_value_end(const std::string &text, std::size_t pos) {
    pos = skip_ws(text, pos);
    if (pos >= text.size()) {
        return pos;
    }
    if (text[pos] == '"') {
        ++pos;
        bool escaped = false;
        while (pos < text.size()) {
            const char c = text[pos++];
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                break;
            }
        }
        return pos;
    }
    if (text[pos] == '{' || text[pos] == '[') {
        const char open = text[pos];
        const char close = open == '{' ? '}' : ']';
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        while (pos < text.size()) {
            const char c = text[pos++];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }
            if (c == '"') {
                in_string = true;
            } else if (c == open || (open == '{' && c == '[') || (open == '[' && c == '{')) {
                ++depth;
            } else if (c == close || (open == '{' && c == ']') || (open == '[' && c == '}')) {
                --depth;
                if (depth == 0) {
                    break;
                }
            }
        }
        return pos;
    }
    while (pos < text.size() && text[pos] != ',' && text[pos] != '}' && text[pos] != ']' &&
           text[pos] != '\n' && text[pos] != '\r') {
        ++pos;
    }
    return pos;
}

std::string json_value_for_key(const std::string &json, const std::string &key) {
    if (json.empty()) {
        return {};
    }
    const std::string needle = "\"" + key + "\"";
    const auto key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return {};
    }
    const auto colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return {};
    }
    const auto start = skip_ws(json, colon + 1);
    const auto end = json_value_end(json, start);
    if (end <= start) {
        return {};
    }
    return json.substr(start, end - start);
}

std::string json_string_or(const std::string &json, const std::string &key, const std::string &fallback = "") {
    std::string value = json_value_for_key(json, key);
    if (value.size() < 2 || value.front() != '"' || value.back() != '"') {
        return fallback;
    }
    value = value.substr(1, value.size() - 2);
    std::string out;
    out.reserve(value.size());
    bool escaped = false;
    for (const char c : value) {
        if (escaped) {
            out.push_back(c);
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::uint64_t json_u64_or(const std::string &json, const std::string &key, std::uint64_t fallback = 0) {
    const auto value = json_value_for_key(json, key);
    if (value.empty() || value == "null") {
        return fallback;
    }
    const auto parsed = collatz::parse_u64(value);
    return parsed.value_or(fallback);
}

double json_double_or(const std::string &json, const std::string &key, double fallback = 0.0) {
    const auto value = json_value_for_key(json, key);
    if (value.empty() || value == "null") {
        return fallback;
    }
    try {
        return std::stod(value);
    } catch (...) {
        return fallback;
    }
}

bool json_bool_or(const std::string &json, const std::string &key, bool fallback = false) {
    const auto value = json_value_for_key(json, key);
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    return fallback;
}

std::string json_object_for_key(const std::string &json, const std::string &key) {
    const auto value = json_value_for_key(json, key);
    return value.size() >= 2 && value.front() == '{' ? value : std::string();
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

std::string percent_text(double fraction) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << fraction * 100.0 << "%";
    return out.str();
}

double percent_value(double fraction) {
    return fraction * 100.0;
}

std::string number_or_null(std::optional<double> value, int precision = 6) {
    if (!value) {
        return "null";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << *value;
    return out.str();
}

bool has_blocker(const std::vector<std::string> &blockers, const std::string &target) {
    return std::find(blockers.begin(), blockers.end(), target) != blockers.end();
}

void write_string_array(std::ostream &out, const std::vector<std::string> &items, const std::string &indent) {
    out << '[';
    if (!items.empty()) {
        out << '\n';
        for (std::size_t i = 0; i < items.size(); ++i) {
            out << indent << '"' << collatz::json_escape(items[i]) << '"';
            if (i + 1 != items.size()) {
                out << ',';
            }
            out << '\n';
        }
    }
    out << (items.empty() ? "" : "    ") << ']';
}

std::string lower_feature_name(std::string name) {
    for (char &ch : name) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (name == "metrics") {
        return "metrics-only";
    }
    if (name == "parity") {
        return "parity-sequence-only";
    }
    if (name == "residue") {
        return "residue-sequence-only";
    }
    if (name == "shape") {
        return "shape-only";
    }
    if (name == "tokens" || name == "token") {
        return "token-only";
    }
    if (name == "gnn" || name == "gnn-only") {
        return "GNN-only";
    }
    if (name == "image" || name == "image-only") {
        return "image-only";
    }
    return name;
}

std::map<std::string, LeaderboardEntry> read_leaderboard(const std::string &ablation_report) {
    std::map<std::string, LeaderboardEntry> entries;
    const std::vector<std::string> names = {
        "metrics-only", "shape-only", "parity-sequence-only", "residue-sequence-only",
        "token-only", "image-only", "GNN-only", "hybrid",
    };
    for (const auto &name : names) {
        entries[name] = LeaderboardEntry{name};
    }
    std::ifstream in(ablation_report);
    if (!in) {
        return entries;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return entries;
    }
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const auto parts = split_csv_line(line);
        if (parts.size() < 6 || (parts[0] != "learned" && parts[0] != "raw")) {
            continue;
        }
        const auto name = lower_feature_name(parts[1]);
        if (entries.find(name) == entries.end()) {
            continue;
        }
        auto &entry = entries[name];
        // Prefer learned entries if both raw and learned rows are present for the same model.
        if (entry.present && parts[0] != "learned" && entry.lift.has_value()) {
            continue;
        }
        entry.name = name;
        entry.present = true;
        entry.samples = collatz::parse_u64(parts[2]).value_or(0);
        try {
            entry.lift = std::stod(parts[5]);
        } catch (...) {
            entry.lift = std::nullopt;
        }
    }
    return entries;
}

std::string family_object(const std::string &source_alignment, const std::string &family_key) {
    const auto families = json_object_for_key(source_alignment, "source_family_coverage");
    return json_object_for_key(families, family_key);
}

bool family_complete(const std::string &source_alignment, const std::string &family_key) {
    return json_bool_or(family_object(source_alignment, family_key), "complete", false);
}

bool family_present(const std::string &source_alignment, const std::string &family_key) {
    return json_bool_or(family_object(source_alignment, family_key), "present", false);
}

std::uint64_t breakdown_count(const std::string &source_alignment, const std::string &bucket) {
    return json_u64_or(json_object_for_key(source_alignment, "unmatched_breakdown"), bucket, 0);
}

std::string canonical_unmatched_breakdown_json(const std::string &source_alignment) {
    const auto count = [&](const std::string &bucket) { return breakdown_count(source_alignment, bucket); };
    const std::uint64_t missing_topology_projection =
        count("missing_topology_projection_node") + count("missing_topology_node");
    std::ostringstream out;
    out << "{\"above_active_scan_range\":" << count("above_active_scan_range")
        << ",\"missing_from_topology_sample\":" << count("missing_from_topology_sample")
        << ",\"missing_feature_row\":" << count("missing_feature_row")
        << ",\"parser_error\":" << count("parser_error")
        << ",\"step_convention_mismatch\":" << count("step_convention_mismatch")
        << ",\"peak_convention_mismatch\":" << count("peak_convention_mismatch")
        << ",\"true_mismatch\":" << count("true_mismatch")
        << ",\"missing_topology_projection_node\":" << missing_topology_projection
        << ",\"duplicated_source_row\":" << count("duplicated_source_row")
        << ",\"future_source_target\":" << count("future_source_target")
        << ",\"unknown\":" << count("unknown") << "}";
    return out.str();
}

std::string confidence_interpretation(const std::string &label) {
    if (label == "candidate pattern") {
        return "The learned structure survives the configured source, holdout, seed, and ablation gates, but it is still not proof.";
    }
    if (label == "source-neighborhood-supported") {
        return "Public validation starts agree with the learned topology-neighborhood gate, but this is evidence rather than proof.";
    }
    if (label == "range-stable signal") {
        return "The learned neighborhood signal survives current range and holdout checks, but it is not proof and is not yet source-neighborhood-supported.";
    }
    if (label == "sample-local signal") {
        return "The learned signal beats a random baseline on the current sample, but range/source gates have not promoted it.";
    }
    return "No empirical evidence label is allowed to become a proof without a formal independently checkable proof artifact.";
}

bool unsafe_public_text(const std::string &text) {
    static const std::regex private_ipv4(
        R"((^|[^0-9])(10\.[0-9]{1,3}\.[0-9]{1,3}\.|192\.168\.[0-9]{1,3}\.|172\.(1[6-9]|2[0-9]|3[0-1])\.[0-9]{1,3}\.))");
    return text.find("/Users/") != std::string::npos || text.find("/home/") != std::string::npos ||
           std::regex_search(text, private_ipv4) || text.find("ssh ") != std::string::npos ||
           text.find("nvidia-smi") != std::string::npos || text.find("bash -lc") != std::string::npos;
}

void write_matched_controls(std::ostream &out, bool enabled) {
    out << "{\"bit_length\":" << (enabled ? "true" : "false")
        << ",\"range_band\":" << (enabled ? "true" : "false")
        << ",\"residue_class\":" << (enabled ? "true" : "false")
        << ",\"stopping_time_bucket\":" << (enabled ? "true" : "false")
        << ",\"peak_ratio_bucket\":" << (enabled ? "true" : "false")
        << ",\"first_drop_bucket\":" << (enabled ? "true" : "false") << "}";
}

std::string matched_controls_json(const std::string &validation) {
    const auto controls = json_object_for_key(validation, "matched_controls");
    if (!controls.empty()) {
        return controls;
    }
    std::ostringstream out;
    write_matched_controls(out, false);
    return out.str();
}

bool matched_controls_complete_from_validation(const std::string &validation) {
    const auto controls = json_object_for_key(validation, "matched_controls");
    if (controls.empty()) {
        return false;
    }
    return json_bool_or(controls, "bit_length") &&
           json_bool_or(controls, "range_band") &&
           json_bool_or(controls, "residue_class") &&
           json_bool_or(controls, "stopping_time_bucket") &&
           json_bool_or(controls, "peak_ratio_bucket") &&
           json_bool_or(controls, "first_drop_bucket");
}

bool lift_statistics_complete_from_validation(const std::string &validation) {
    const auto stats = json_object_for_key(validation, "lift_statistics");
    const auto ci = json_value_for_key(stats, "ci_95");
    return json_u64_or(validation, "n_seeds") >= 5 &&
           json_u64_or(validation, "n_folds") >= 5 &&
           !ci.empty() && ci != "null";
}

void write_summary_json(
    const Options &options,
    const std::string &label,
    const std::string &conclusion,
    const std::string &coverage,
    const std::string &strongest_evidence,
    const std::string &weakest_limit,
    const std::string &latest_neural_result,
    const std::string &next_experiment,
    const std::string &source_alignment_text) {
    if (options.hypothesis_summary_output.empty()) {
        return;
    }
    collatz::ensure_parent_dir(options.hypothesis_summary_output);
    std::ofstream out(options.hypothesis_summary_output);
    if (!out) {
        throw std::runtime_error("failed to open hypothesis summary output: " + options.hypothesis_summary_output);
    }
    out << "{\n"
        << "  \"dataset_type\": \"collatz_hypothesis_summary\",\n"
        << "  \"tool\": \"collatz_evidence_publish\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"canonical_evidence\": \"" << collatz::json_escape(options.output) << "\",\n"
        << "  \"hypothesis_count\": 1,\n"
        << "  \"confidence_level\": \"" << collatz::json_escape(label) << "\",\n"
        << "  \"conclusion\": \"" << collatz::json_escape(conclusion) << "\",\n"
        << "  \"coverage\": \"" << collatz::json_escape(coverage) << "\",\n"
        << "  \"strongest_evidence\": \"" << collatz::json_escape(strongest_evidence) << "\",\n"
        << "  \"weakest_limit\": \"" << collatz::json_escape(weakest_limit) << "\",\n"
        << "  \"latest_neural_result\": \"" << collatz::json_escape(latest_neural_result) << "\",\n"
        << "  \"next_experiment\": \"" << collatz::json_escape(next_experiment) << "\",\n"
        << "  \"source_alignment\": \"" << collatz::json_escape(source_alignment_text) << "\"\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const std::string full_audit = read_file_or_empty(options.full_audit);
        const std::string stratified = read_file_or_empty(options.stratified_metadata);
        const std::string topology = read_file_or_empty(options.topology_manifest);
        const std::string validation = read_file_or_empty(options.validation_metrics);
        const std::string source_alignment = read_file_or_empty(options.source_alignment);
        const std::string runner = read_file_or_empty(options.runner_status);
        const std::string neural_status = read_file_or_empty(options.neural_status);
        const auto leaderboard = read_leaderboard(options.ablation_report);

        const std::uint64_t audit_rows = json_u64_or(full_audit, "records_read");
        const std::uint64_t range_start = json_u64_or(full_audit, "input_range_start", 1);
        const std::uint64_t range_end = std::max(json_u64_or(full_audit, "effective_range_end"),
                                                 json_u64_or(full_audit, "input_range_end"));
        const bool full_audit_completed = audit_rows > 0 && json_double_or(full_audit, "coverage_ratio") >= 0.999;
        const std::uint64_t topology_rows = json_u64_or(topology, "point_count");
        const std::uint64_t stratified_rows = json_u64_or(stratified, "selected_rows");
        const std::uint64_t sample_rows = json_u64_or(validation, "sample_rows", stratified_rows);
        const double topology_percent = audit_rows == 0 ? 0.0 : percent_value(static_cast<double>(topology_rows) / audit_rows);
        const double stratified_percent = audit_rows == 0 ? 0.0 : percent_value(static_cast<double>(stratified_rows) / audit_rows);

        const double contrastive_lift = json_double_or(validation, "contrastive_lift");
        const double contrastive_minus_numeric = json_double_or(validation, "contrastive_minus_numeric");
        const double range_min_lift = json_double_or(validation, "range_min_lift");
        const double fold_min_lift = json_double_or(validation, "fold_min_lift");
        const double residue_mean_lift = json_double_or(validation, "residue_mean_lift");
        const std::uint64_t parallel_jobs_completed =
            std::max(json_u64_or(runner, "complete_jobs", 0), json_u64_or(neural_status, "complete_jobs", 0));
        const bool gpu_used = json_string_or(validation, "evaluation_device") == "cuda";

        const std::uint64_t targets_total = json_u64_or(source_alignment, "target_count");
        const std::uint64_t matched = json_u64_or(source_alignment, "matched_targets");
        const std::uint64_t unmatched = json_u64_or(source_alignment, "missing_targets", targets_total > matched ? targets_total - matched : 0);
        const double matched_fraction = targets_total == 0 ? 0.0 : static_cast<double>(matched) / static_cast<double>(targets_total);
        const std::uint64_t unknown = breakdown_count(source_alignment, "unknown");
        const std::uint64_t true_mismatch = breakdown_count(source_alignment, "true_mismatch");
        const bool oeis_complete = family_complete(source_alignment, "oeis");
        const bool roosendaal_complete = family_complete(source_alignment, "roosendaal");
        const bool oliveira_complete = family_complete(source_alignment, "oliveira_e_silva");
        const bool barina_complete = family_complete(source_alignment, "barina");
        const int supplemental_complete = (roosendaal_complete ? 1 : 0) + (oliveira_complete ? 1 : 0) + (barina_complete ? 1 : 0);

        const auto metrics = leaderboard.at("metrics-only");
        const auto hybrid = leaderboard.at("hybrid");
        const double metrics_lift = metrics.lift.value_or(-1.0);
        const double hybrid_lift = hybrid.lift.value_or(contrastive_lift);
        const bool metric_dominant = metrics.lift && hybrid.lift && metrics_lift > hybrid_lift;
        const bool matched_controls_complete = matched_controls_complete_from_validation(validation);
        const bool lift_statistics_complete = lift_statistics_complete_from_validation(validation);
        const std::uint64_t n_folds = json_u64_or(validation, "n_folds", 0);
        const std::uint64_t n_seeds = json_u64_or(validation, "n_seeds", 0);
        const auto lift_stats = json_object_for_key(validation, "lift_statistics");
        const std::string ci_95 = json_value_for_key(lift_stats, "ci_95");
        const std::string lift_std = json_value_for_key(lift_stats, "std");
        const std::string controls = matched_controls_json(validation);

        const bool gate1 = full_audit_completed && contrastive_lift > 0.0 && contrastive_minus_numeric > 0.0 &&
                           range_min_lift > 0.0 && fold_min_lift > 0.0;
        const bool gate2 = gate1 && oeis_complete && supplemental_complete >= 2 && true_mismatch == 0 && unknown == 0 &&
                           targets_total > 0;
        const bool richer_beats_metrics = hybrid.lift && metrics.lift && hybrid_lift > metrics_lift;
        const bool gate3 = gate2 && richer_beats_metrics && matched_controls_complete && lift_statistics_complete;

        std::vector<std::string> promotion_blockers;
        if (!gate1) {
            promotion_blockers.push_back("range_stable_gate_incomplete");
        }
        if (!oeis_complete) {
            promotion_blockers.push_back("oeis_source_family_incomplete");
        }
        if (supplemental_complete < 2) {
            promotion_blockers.push_back("non_oeis_source_families_incomplete");
        }
        if (unknown > 0) {
            promotion_blockers.push_back("source_alignment_unknown_rows_present");
        }
        if (true_mismatch > 0) {
            promotion_blockers.push_back("source_alignment_true_mismatch_present");
        }
        if (targets_total == 0 || unknown > 0 || true_mismatch > 0) {
            promotion_blockers.push_back("source_alignment_unmatched_rows_present");
        }
        if (metric_dominant) {
            promotion_blockers.push_back("metrics_only_exceeds_hybrid");
        }
        if (!richer_beats_metrics) {
            promotion_blockers.push_back("richer_representation_not_stronger");
        }
        if (!matched_controls_complete) {
            promotion_blockers.push_back("matched_controls_incomplete");
        }
        if (!lift_statistics_complete) {
            promotion_blockers.push_back("missing_lift_statistics");
        }

        std::string label = "sample-local signal";
        int rank = 0;
        if (gate1) {
            label = "range-stable signal";
            rank = 1;
        }
        if (gate2) {
            label = "source-neighborhood-supported";
            rank = 2;
        }
        if (gate3) {
            label = "candidate pattern";
            rank = 3;
        }

        const std::string signal_type = metric_dominant ? "metric-dominant signal" : "hybrid-or-nonmetric signal";
        const std::string signal_reason =
            metric_dominant
                ? "metrics-only lift exceeds hybrid lift under the current evidence run (healthy negative control)"
                : richer_beats_metrics
                      ? "hybrid lift exceeds metrics-only under matched controls and holdouts"
                      : "hybrid lift has not yet exceeded metrics-only under the current evidence run";
        std::string next_summary;
        if (targets_total == 0 || unknown > 0 || true_mismatch > 0 ||
            !oeis_complete || supplemental_complete < 2) {
            next_summary =
                "Classify unmatched source targets, expand non-OEIS source imports, then rerun source-neighborhood, path-image, GNN, and matched-control ablations.";
        } else if (!matched_controls_complete || !lift_statistics_complete) {
            next_summary =
                "Run matched hard-negative family-pair training with at least 5 seeds by 5 folds, then republish lift confidence intervals and retrieval metrics.";
        } else if (metric_dominant) {
            next_summary =
                "Strengthen ordered parity, residue, and log-shape branches with representation dropout until hybrid beats metrics-only under matched controls.";
        } else {
            next_summary =
                "Validate source-neighborhood, path-image, GNN, and matched-control ablations against retrieval and falsification targets.";
        }
        const std::string falsification =
            "Any true mismatch in public validation targets or degradation below matched-control baselines blocks confidence promotion.";
        const std::string conclusion = confidence_interpretation(label);
        const std::string coverage_text =
            "Full audit covers " + std::to_string(audit_rows) + " binary rows; topology covers " +
            percent_text(audit_rows == 0 ? 0.0 : static_cast<double>(topology_rows) / audit_rows) +
            "; stratified evidence sample covers " +
            percent_text(audit_rows == 0 ? 0.0 : static_cast<double>(stratified_rows) / audit_rows) +
            " directly while intentionally oversampling rare behaviors.";
        const std::string evidence_text =
            "Learned embeddings beat random by " + percent_text(contrastive_lift) +
            " and numeric adjacency by " + percent_text(contrastive_minus_numeric) +
            "; range min lift is " + percent_text(range_min_lift) + ".";
        std::string limit_text;
        if (promotion_blockers.empty()) {
            limit_text = "No explicit promotion blockers are active in this canonical evidence cycle.";
        } else if (has_blocker(promotion_blockers, "metrics_only_exceeds_hybrid")) {
            limit_text = "Metrics-only currently beats hybrid; richer representation lift must exceed this baseline under matched controls for neural promotion.";
        } else if (has_blocker(promotion_blockers, "richer_representation_not_stronger")) {
            limit_text = "Richer representation families have not yet exceeded metrics-only in this cycle under matched controls.";
        } else if (has_blocker(promotion_blockers, "matched_controls_incomplete")) {
            limit_text = "Matched hard-negative controls remain incomplete, so hybrid claims are not yet robustly comparable.";
        } else if (has_blocker(promotion_blockers, "missing_lift_statistics")) {
            limit_text = "Lift statistics are incomplete (seeds/folds/CI), so no reliable trend comparison is available.";
        } else if (has_blocker(promotion_blockers, "source_alignment_unmatched_rows_present") ||
                   has_blocker(promotion_blockers, "source_alignment_unknown_rows_present") ||
                   has_blocker(promotion_blockers, "source_alignment_true_mismatch_present")) {
            limit_text = "Source alignment quality is not yet fully clean; unmatched/unknown/true-mismatch rows block promotion.";
        } else if (has_blocker(promotion_blockers, "oeis_source_family_incomplete") ||
                   has_blocker(promotion_blockers, "non_oeis_source_families_incomplete")) {
            limit_text = "Source-family completeness is still missing; complete source imports are required before additional promotion.";
        } else {
            limit_text = "One or more confidence gate blockers are active; rerun with the blocked controls resolved.";
        }
        const std::string neural_text =
            "Hybrid contrastive lift " + percent_text(contrastive_lift) + "; best current learned ablation " +
            (metrics.lift && metrics_lift >= hybrid_lift ? "metrics-only" : "hybrid") +
            " with lift " + percent_text(std::max(metrics_lift, hybrid_lift)) + ".";
        const std::string source_text =
            "Source alignment matched " + std::to_string(matched) + " of " + std::to_string(targets_total) +
            " public targets; unknown unmatched rows: " + std::to_string(unknown) + ".";

        std::ostringstream json;
        json << std::setprecision(10);
        json << "{\n"
             << "  \"schema_version\": \"evidence_public_summary_v1\",\n"
             << "  \"run_id\": \"" << collatz::now_iso8601() << "\",\n"
             << "  \"git_commit\": \"" << collatz::json_escape(options.git_commit) << "\",\n"
             << "  \"generated_at_utc\": \"" << collatz::now_iso8601() << "\",\n"
             << "  \"confidence\": {\n"
             << "    \"label\": \"" << label << "\",\n"
             << "    \"rank\": " << rank << ",\n"
             << "    \"claim_is_proof\": false,\n"
             << "    \"claim_is_source_neighborhood_supported\": " << (rank >= 2 ? "true" : "false") << ",\n"
             << "    \"claim_is_candidate_pattern\": " << (rank >= 3 ? "true" : "false") << ",\n"
             << "    \"interpretation\": \"" << collatz::json_escape(conclusion) << "\",\n"
             << "    \"promotion_blockers\": ";
        write_string_array(json, promotion_blockers, "      ");
        json << "\n"
             << "  },\n"
             << "  \"audit\": {\n"
             << "    \"active_feature_file\": \"" << collatz::json_escape(options.active_feature_file) << "\",\n"
             << "    \"rows\": " << audit_rows << ",\n"
             << "    \"range_start\": " << range_start << ",\n"
             << "    \"range_end\": " << range_end << ",\n"
             << "    \"feature_schema_version\": \"feature_v" << json_u64_or(full_audit, "feature_schema_version", 1) << "\",\n"
             << "    \"binary_schema_version\": \"bin_v" << json_u64_or(full_audit, "binary_feature_version", 1) << "\",\n"
             << "    \"full_audit_completed\": " << (full_audit_completed ? "true" : "false") << ",\n"
             << "    \"feature_file_sha256\": null,\n"
             << "    \"metadata_sha256\": null\n"
             << "  },\n"
             << "  \"coverage\": {\n"
             << "    \"topology\": {\"rows\": " << topology_rows << ", \"percent_of_audit\": " << topology_percent << "},\n"
             << "    \"stratified_evidence_sample\": {\"rows\": " << stratified_rows
             << ", \"percent_of_audit\": " << stratified_percent
             << ", \"sampling_note\": \"Stratified sample intentionally oversamples rare behavior.\"}\n"
             << "  },\n"
             << "  \"neural\": {\n"
             << "    \"latest_run\": {\"sample_rows\": " << sample_rows
             << ", \"gpu_used\": " << (gpu_used ? "true" : "false")
             << ", \"parallel_jobs_completed\": " << parallel_jobs_completed << "},\n"
             << "    \"leaderboard\": [\n";
        std::size_t entry_index = 0;
        for (const auto &[_, entry] : leaderboard) {
            json << "      {\"name\": \"" << collatz::json_escape(entry.name)
                 << "\", \"lift_percent\": {\"mean\": " << number_or_null(entry.lift ? std::optional<double>(percent_value(*entry.lift)) : std::nullopt, 3)
                 << ", \"std\": " << (lift_std.empty() ? "null" : lift_std)
                 << ", \"ci_95\": " << (ci_95.empty() ? "null" : ci_95)
                 << "}, \"n_folds\": " << (n_folds == 0 ? "null" : std::to_string(n_folds))
                 << ", \"n_seeds\": " << (n_seeds == 0 ? "null" : std::to_string(n_seeds))
                 << ", \"n_samples\": "
                 << (entry.present ? entry.samples : sample_rows)
                 << ", \"matched_controls\": " << controls << "}";
            if (++entry_index != leaderboard.size()) {
                json << ',';
            }
            json << '\n';
        }
        json << "    ],\n"
             << "    \"interpretation\": {\"signal_type\": \"" << signal_type
             << "\", \"reason\": \"" << collatz::json_escape(signal_reason)
             << "\", \"may_promote_confidence\": false},\n"
             << "    \"holdouts\": {\"weakest_range_lift_percent\": " << percent_value(range_min_lift)
             << ", \"fold_min_lift_percent\": " << percent_value(fold_min_lift)
             << ", \"numeric_adjacency_lift_percent\": " << percent_value(contrastive_minus_numeric) << "},\n"
             << "    \"retrieval\": " << (json_object_for_key(validation, "retrieval").empty()
                                           ? "{}"
                                           : json_object_for_key(validation, "retrieval")) << "\n"
             << "  },\n"
             << "  \"source_alignment\": {\n"
             << "    \"targets_total\": " << targets_total << ",\n"
             << "    \"matched\": " << matched << ",\n"
             << "    \"unmatched\": " << unmatched << ",\n"
             << "    \"matched_fraction\": " << matched_fraction << ",\n"
             << "    \"source_family_coverage\": {\n"
             << "      \"oeis\": {\"present\": " << (family_present(source_alignment, "oeis") ? "true" : "false")
             << ", \"complete\": " << (oeis_complete ? "true" : "false") << "},\n"
             << "      \"roosendaal\": {\"present\": " << (family_present(source_alignment, "roosendaal") ? "true" : "false")
             << ", \"complete\": " << (roosendaal_complete ? "true" : "false") << "},\n"
             << "      \"oliveira_e_silva\": {\"present\": " << (family_present(source_alignment, "oliveira_e_silva") ? "true" : "false")
             << ", \"complete\": " << (oliveira_complete ? "true" : "false") << "},\n"
             << "      \"barina\": {\"present\": " << (family_present(source_alignment, "barina") ? "true" : "false")
             << ", \"complete\": " << (barina_complete ? "true" : "false") << "}\n"
             << "    },\n"
             << "    \"unmatched_breakdown\": " << canonical_unmatched_breakdown_json(source_alignment)
             << ",\n"
             << "    \"unknown_unmatched_percent\": " << json_double_or(source_alignment, "unknown_unmatched_percent") << "\n"
             << "  },\n"
             << "  \"next_experiment\": {\"summary\": \"" << collatz::json_escape(next_summary)
             << "\", \"falsification_target\": \"" << collatz::json_escape(falsification) << "\"},\n"
             << "  \"public_safety\": {\n"
             << "    \"sanitized\": " << (unsafe_public_text(json.str()) ? "false" : "true") << ",\n"
             << "    \"contains_hostnames\": false,\n"
             << "    \"contains_internal_ips\": false,\n"
             << "    \"contains_usernames\": false,\n"
             << "    \"contains_absolute_local_paths\": false,\n"
             << "    \"contains_raw_internal_command_traces\": false\n"
             << "  }\n"
             << "}\n";

        collatz::ensure_parent_dir(options.output);
        std::ofstream out(options.output);
        if (!out) {
            throw std::runtime_error("failed to open canonical evidence output: " + options.output);
        }
        out << json.str();

        write_summary_json(options, label, conclusion, coverage_text, evidence_text, limit_text, neural_text, next_summary, source_text);

        std::cout << "public_evidence=" << options.output
                  << " confidence=" << label
                  << " audit_rows=" << audit_rows
                  << " source_matched=" << matched << "/" << targets_total
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
