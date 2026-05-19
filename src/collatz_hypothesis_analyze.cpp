#include "collatz/core.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string insights = "data/generated/insights/insights.json";
    std::string stratified_metadata = "data/generated/stratified/metadata.json";
    std::string contrastive_metrics = "data/generated/contrastive/metrics.json";
    std::string autoencoder_metrics = "data/generated/anomalies/metrics.json";
    std::string gnn_metrics = "data/generated/gnn/metrics.json";
    std::string validation_metrics = "data/generated/evidence_validation/metrics.json";
    std::string source_alignment = "data/generated/source_alignment/source_alignment.json";
    std::string output_dir = "data/generated/hypotheses";
};

struct Hypothesis {
    std::string id;
    std::string claim;
    std::string confidence_level;
    std::string evidence;
    std::string limit;
    std::string falsification_test;
    std::string next_action;
};

void usage(std::ostream &out) {
    out << "usage: collatz_hypothesis_analyze [options]\n\n"
        << "options:\n"
        << "  --insights FILE              topology insights JSON\n"
        << "  --stratified-metadata FILE   stratified sample metadata JSON\n"
        << "  --contrastive-metrics FILE   contrastive model metrics JSON\n"
        << "  --autoencoder-metrics FILE   autoencoder anomaly metrics JSON\n"
        << "  --gnn-metrics FILE           GNN metrics JSON\n"
        << "  --validation-metrics FILE    evidence validation metrics JSON\n"
        << "  --source-alignment FILE      source neighborhood alignment JSON\n"
        << "  --output-dir DIR             output directory (default data/generated/hypotheses)\n";
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
        if (arg == "--insights") {
            options.insights = need_value("--insights");
        } else if (arg == "--stratified-metadata") {
            options.stratified_metadata = need_value("--stratified-metadata");
        } else if (arg == "--contrastive-metrics") {
            options.contrastive_metrics = need_value("--contrastive-metrics");
        } else if (arg == "--autoencoder-metrics") {
            options.autoencoder_metrics = need_value("--autoencoder-metrics");
        } else if (arg == "--gnn-metrics") {
            options.gnn_metrics = need_value("--gnn-metrics");
        } else if (arg == "--validation-metrics") {
            options.validation_metrics = need_value("--validation-metrics");
        } else if (arg == "--source-alignment") {
            options.source_alignment = need_value("--source-alignment");
        } else if (arg == "--output-dir") {
            options.output_dir = need_value("--output-dir");
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

std::uint64_t json_u64_or(const std::string &json, const std::string &key, std::uint64_t fallback = 0) {
    const auto value = json_value_for_key(json, key);
    if (value.empty()) {
        return fallback;
    }
    const auto parsed = collatz::parse_u64(value);
    return parsed.value_or(fallback);
}

double json_double_or(const std::string &json, const std::string &key, double fallback = 0.0) {
    const auto value = json_value_for_key(json, key);
    if (value.empty()) {
        return fallback;
    }
    try {
        return std::stod(value);
    } catch (...) {
        return fallback;
    }
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

std::string percent(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << (value * 100.0) << "%";
    return out.str();
}

std::string fixed(double value, int digits = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(digits) << value;
    return out.str();
}

void write_hypotheses_jsonl(const std::string &path, const std::vector<Hypothesis> &hypotheses) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open hypothesis ledger: " + path);
    }
    for (const auto &hypothesis : hypotheses) {
        out << "{\"timestamp\":\"" << collatz::now_iso8601()
            << "\",\"id\":\"" << collatz::json_escape(hypothesis.id)
            << "\",\"claim\":\"" << collatz::json_escape(hypothesis.claim)
            << "\",\"confidence_level\":\"" << collatz::json_escape(hypothesis.confidence_level)
            << "\",\"evidence\":\"" << collatz::json_escape(hypothesis.evidence)
            << "\",\"limit\":\"" << collatz::json_escape(hypothesis.limit)
            << "\",\"falsification_test\":\"" << collatz::json_escape(hypothesis.falsification_test)
            << "\",\"next_action\":\"" << collatz::json_escape(hypothesis.next_action)
            << "\"}\n";
    }
}

void write_summary(
    const std::string &path,
    const std::vector<Hypothesis> &hypotheses,
    const std::string &confidence_level,
    const std::string &conclusion,
    const std::string &coverage,
    const std::string &strongest_evidence,
    const std::string &weakest_limit,
    const std::string &latest_neural_result,
    const std::string &next_experiment,
    const std::string &source_alignment) {
    collatz::ensure_parent_dir(path);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open hypothesis summary: " + path);
    }
    out << "{\n"
        << "  \"dataset_type\": \"collatz_hypothesis_summary\",\n"
        << "  \"tool\": \"collatz_hypothesis_analyze\",\n"
        << "  \"created_utc\": \"" << collatz::now_iso8601() << "\",\n"
        << "  \"hypothesis_count\": " << hypotheses.size() << ",\n"
        << "  \"confidence_level\": \"" << collatz::json_escape(confidence_level) << "\",\n"
        << "  \"conclusion\": \"" << collatz::json_escape(conclusion) << "\",\n"
        << "  \"coverage\": \"" << collatz::json_escape(coverage) << "\",\n"
        << "  \"strongest_evidence\": \"" << collatz::json_escape(strongest_evidence) << "\",\n"
        << "  \"weakest_limit\": \"" << collatz::json_escape(weakest_limit) << "\",\n"
        << "  \"latest_neural_result\": \"" << collatz::json_escape(latest_neural_result) << "\",\n"
        << "  \"next_experiment\": \"" << collatz::json_escape(next_experiment) << "\",\n"
        << "  \"source_alignment\": \"" << collatz::json_escape(source_alignment) << "\",\n"
        << "  \"files\": {\"hypotheses\": \"hypotheses.jsonl\"}\n"
        << "}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = parse_args(argc, argv);
        const std::string insights = read_file_or_empty(options.insights);
        const std::string stratified = read_file_or_empty(options.stratified_metadata);
        const std::string contrastive = read_file_or_empty(options.contrastive_metrics);
        const std::string autoencoder = read_file_or_empty(options.autoencoder_metrics);
        const std::string gnn = read_file_or_empty(options.gnn_metrics);
        const std::string validation = read_file_or_empty(options.validation_metrics);
        const std::string source_alignment = read_file_or_empty(options.source_alignment);

        const std::uint64_t scanned_rows = std::max(json_u64_or(stratified, "input_records_read"), json_u64_or(insights, "scan_records"));
        const std::uint64_t selected_rows = json_u64_or(stratified, "selected_rows");
        const std::uint64_t topology_points = json_u64_or(insights, "topology_points");
        const std::uint64_t reason_count = json_u64_or(stratified, "reason_count");
        const double topology_coverage = scanned_rows == 0 ? 0.0 : static_cast<double>(topology_points) / static_cast<double>(scanned_rows);
        const double stratified_coverage = scanned_rows == 0 ? 0.0 : static_cast<double>(selected_rows) / static_cast<double>(scanned_rows);

        const std::string contrastive_status = json_string_or(contrastive, "status", "missing");
        const double neighbor_purity = json_double_or(contrastive, "neighbor_purity", 0.0);
        const double baseline_purity = json_double_or(contrastive, "random_baseline_purity", 0.0);
        const double purity_lift = json_double_or(contrastive, "purity_lift", neighbor_purity - baseline_purity);
        const std::uint64_t embedding_count = json_u64_or(contrastive, "embedding_count");

        const std::string autoencoder_status = json_string_or(autoencoder, "status", "missing");
        const std::uint64_t anomaly_count = json_u64_or(autoencoder, "anomaly_count");
        const double mean_error = json_double_or(autoencoder, "mean_reconstruction_error", 0.0);

        const std::string gnn_status = json_string_or(gnn, "status", "missing");
        const double gnn_loss = json_double_or(gnn, "loss_final", 0.0);
        const std::uint64_t gnn_nodes = json_u64_or(gnn, "node_count");

        const std::string validation_status = json_string_or(validation, "status", "missing");
        const std::string validation_confidence = json_string_or(validation, "confidence_level", "pipeline-check");
        const double validation_lift = json_double_or(validation, "contrastive_lift", 0.0);
        const double contrastive_minus_numeric = json_double_or(validation, "contrastive_minus_numeric", 0.0);
        const std::uint64_t range_holdout_count = json_u64_or(validation, "range_holdout_count");
        const double range_min_lift = json_double_or(validation, "range_min_lift", 0.0);
        const std::uint64_t residue_holdout_count = json_u64_or(validation, "residue_holdout_count");
        const double residue_mean_lift = json_double_or(validation, "residue_mean_lift", 0.0);
        const std::uint64_t fold_count = json_u64_or(validation, "fold_count");
        const double fold_min_lift = json_double_or(validation, "fold_min_lift", 0.0);
        const std::uint64_t ablation_count = json_u64_or(validation, "ablation_count");
        const std::string best_ablation = json_string_or(validation, "best_ablation", "pending");
        const double best_lift = json_double_or(validation, "best_lift", 0.0);

        const std::string alignment_status = json_string_or(source_alignment, "alignment_status", "missing");
        const std::uint64_t source_target_count = json_u64_or(source_alignment, "target_count");
        const std::uint64_t matched_source_targets = json_u64_or(source_alignment, "matched_targets");
        const std::uint64_t matched_source_clusters = json_u64_or(source_alignment, "matched_cluster_count");
        const std::uint64_t source_family_count = std::max(json_u64_or(source_alignment, "source_family_count"),
                                                           json_u64_or(source_alignment, "source_count"));
        const std::string source_limit = json_string_or(
            source_alignment,
            "limit",
            "No source alignment artifact was found.");

        std::vector<Hypothesis> hypotheses;
        hypotheses.push_back({
            "stratified-evidence-coverage",
            selected_rows > 0
                ? "The analysis now has an evidence-first stratified sample that pulls from multiple Collatz behavior buckets."
                : "The system still needs a stratified sample before it can test full-scan pattern claims.",
            selected_rows > 0 ? "sample-local signal" : "pipeline-check",
            selected_rows > 0
                ? std::to_string(selected_rows) + " selected rows across " + std::to_string(reason_count) +
                      " selection reasons from " + std::to_string(scanned_rows) + " scanned rows."
                : "No stratified sample metadata was found.",
            "A stratified sample is evidence coverage, not a proof; it must be compared across independent samples.",
            "Rebuild the sample with different seeds and range holdouts; reject claims that do not survive.",
            selected_rows > 0 ? "Train contrastive and anomaly models on this sample." : "Run collatz_stratified_sample on features.bin.",
        });

        hypotheses.push_back({
            "baseline-topology-families",
            topology_points > 0
                ? "The baseline metric topology separates Collatz starts into repeatable local neighborhoods."
                : "No topology artifact is available yet.",
            topology_points > 0 ? "sample-local signal" : "pipeline-check",
            topology_points > 0
                ? std::to_string(topology_points) + " projected points cover " + percent(topology_coverage) + " of the scanned rows."
                : "The topology projection and cluster files were not found.",
            "Current topology is a deterministic baseline and can overstate structure if features are too correlated.",
            "Run ablations for metric-only, parity-only, residue-only, image-only, GNN-only, and hybrid features.",
            "Compare learned contrastive embeddings against this baseline.",
        });

        hypotheses.push_back({
            "contrastive-neural-neighborhoods",
            contrastive_status == "complete"
                ? "The contrastive encoder found learned neighborhoods that can be compared against random adjacency."
                : "The contrastive encoder has not produced a complete metrics artifact yet.",
            contrastive_status == "complete" && purity_lift > 0.05 ? "candidate pattern" : "pipeline-check",
            contrastive_status == "complete"
                ? std::to_string(embedding_count) + " embeddings, nearest-neighbor purity " + percent(neighbor_purity) +
                      ", random baseline " + percent(baseline_purity) + ", lift " + percent(purity_lift) + "."
                : "No complete contrastive metrics were found.",
            "A learned neighborhood only matters if it survives holdouts and feature ablation.",
            "Train on one range/residue split and verify nearest-neighbor purity on held-out splits.",
            contrastive_status == "complete" ? "Run cross-sample stability tests." : "Run research/contrastive_train.py.",
        });

        hypotheses.push_back({
            "autoencoder-anomaly-candidates",
            autoencoder_status == "complete"
                ? "The autoencoder produced reconstruction-error anomaly candidates for manual path review."
                : "The autoencoder anomaly model has not produced a complete artifact yet.",
            autoencoder_status == "complete" && anomaly_count > 0 ? "sample-local signal" : "pipeline-check",
            autoencoder_status == "complete"
                ? std::to_string(anomaly_count) + " anomalies with mean reconstruction error " + fixed(mean_error, 6) + "."
                : "No complete autoencoder metrics were found.",
            "High reconstruction error can mean rare-but-valid feature combinations, not a mathematically special path.",
            "Rerun with different seeds and verify whether the same starts remain anomalous.",
            autoencoder_status == "complete" ? "Generate full paths and path-image atlases for stable anomalies." : "Run research/autoencoder_anomaly.py.",
        });

        hypotheses.push_back({
            "gnn-shared-tail-topology",
            gnn_status == "complete"
                ? "The GNN trainer can embed the shared-tail trajectory graph on a CUDA-capable GPU."
                : "The GNN graph path is not complete yet.",
            gnn_status == "complete" ? "pipeline-check" : "pipeline-check",
            gnn_status == "complete"
                ? std::to_string(gnn_nodes) + " graph nodes trained with final link-reconstruction loss " + fixed(gnn_loss, 6) + "."
                : "No complete GNN metrics were found.",
            "Current GNN loss proves the pipeline trains, not that it has discovered a durable family law.",
            "Compare GNN nearest neighbors against metric, parity, and residue embeddings.",
            "Use GNN embeddings as one ablation input in the stability suite.",
        });

        hypotheses.push_back({
            "holdout-ablation-validation",
            validation_status == "complete"
                ? "The evidence validator tested the learned neighborhoods against range, residue, fold, numeric-adjacency, and ablation baselines."
                : "The evidence validator has not run yet.",
            validation_status == "complete" ? validation_confidence : "pipeline-check",
            validation_status == "complete"
                ? std::to_string(range_holdout_count) + " range holdouts, " + std::to_string(residue_holdout_count) +
                      " residue holdouts, " + std::to_string(fold_count) + " folds, " +
                      std::to_string(ablation_count) + " learned ablations; best ablation " + best_ablation +
                      " with lift " + percent(best_lift) + "."
                : "No evidence validation metrics were found.",
            "Passing holdouts shows empirical stability, not proof; the signal can still be an artifact of the selected feature labels.",
            "Reject or demote the claim if new seeded samples, source-aligned records, or feature ablations lose lift.",
            validation_status == "complete"
                ? "Run independent seeded stratified samples and compare source-record neighborhoods."
                : "Run research/evidence_validate.py after contrastive and ablation runs.",
        });

        hypotheses.push_back({
            "source-record-neighborhoods",
            alignment_status != "missing"
                ? "Known source validation targets are now checked against embedding neighborhoods."
                : "Source-record neighborhood comparison has not run yet.",
            alignment_status == "multi-source-aligned"
                ? "multi-source-aligned candidate"
                : alignment_status == "public-source-aligned"
                ? "source-aligned candidate"
                : alignment_status == "source-smoke-aligned" ? "sample-local signal" : "pipeline-check",
            alignment_status != "missing"
                ? std::to_string(matched_source_targets) + " of " + std::to_string(source_target_count) +
                      " source targets matched the topology sample across " + std::to_string(matched_source_clusters) +
                      " clusters from " + std::to_string(source_family_count) + " source families."
                : "No source alignment metrics were found.",
            source_limit,
            "Promote only if independent source families, holdouts, and ablations keep agreeing; this remains evidence, not proof.",
            alignment_status == "multi-source-aligned"
                ? "Run source-anchored path-image and GNN ablations against these matched targets."
                : "Import larger source-record tables and compare their learned and path-image neighborhoods.",
        });

        std::string confidence_level = "pipeline-check";
        if (selected_rows > 0 || topology_points > 0) {
            confidence_level = "sample-local signal";
        }
        if (validation_status == "complete" && validation_confidence == "range-stable signal") {
            confidence_level = "range-stable signal";
        } else if (contrastive_status == "complete" && purity_lift > 0.05 && selected_rows > 0) {
            confidence_level = "sample-local signal";
        }
        if (confidence_level == "range-stable signal" && alignment_status == "multi-source-aligned" &&
            source_target_count >= 25 && source_family_count >= 3) {
            confidence_level = "multi-source-aligned candidate";
        } else if (confidence_level == "range-stable signal" && alignment_status == "public-source-aligned" &&
                   source_target_count >= 25) {
            confidence_level = "source-aligned candidate";
        }

        std::string latest_neural_result = "No learned contrastive or anomaly model has completed yet.";
        if (validation_status == "complete") {
            latest_neural_result = "Evidence validation complete: contrastive lift " + percent(validation_lift) +
                                   ", range min lift " + percent(range_min_lift) +
                                   ", fold min lift " + percent(fold_min_lift) + ".";
        } else if (contrastive_status == "complete") {
            latest_neural_result = "Contrastive encoder complete: purity lift " + percent(purity_lift) +
                                   " over random baseline across " + std::to_string(embedding_count) + " embeddings.";
        } else if (autoencoder_status == "complete") {
            latest_neural_result = "Autoencoder complete: " + std::to_string(anomaly_count) + " anomaly candidates.";
        } else if (gnn_status == "complete") {
            latest_neural_result = "GNN complete: " + std::to_string(gnn_nodes) + " nodes, final loss " + fixed(gnn_loss, 6) + ".";
        }

        const std::string coverage =
            "Topology covers " + percent(topology_coverage) + " of scanned rows; stratified evidence sample covers " +
            percent(stratified_coverage) + " directly while intentionally oversampling rare behaviors.";
        const std::string strongest_evidence =
            validation_status == "complete"
                ? "Learned embeddings beat random by " + percent(validation_lift) + " and numeric adjacency by " +
                      percent(contrastive_minus_numeric) + "; range min lift is " + percent(range_min_lift) + "."
                : contrastive_status == "complete" && purity_lift > 0.05
                      ? "Learned nearest-neighbor purity beats the random baseline by " + percent(purity_lift) + "."
                      : selected_rows > 0
                            ? "The sample now includes " + std::to_string(selected_rows) + " starts across " +
                                  std::to_string(reason_count) + " evidence buckets from the full scan."
                            : json_string_or(insights, "meaning", "The existing topology and GNN artifacts are available, but the evidence sample is not built.");
        const std::string weakest_limit =
            validation_status == "complete"
                ? alignment_status == "source-smoke-aligned"
                      ? "Source alignment is currently only a smoke check; larger dated record imports are still required."
                      : alignment_status == "multi-source-aligned"
                            ? "Multiple source families match, but the stronger claim still needs source-anchored path-image and GNN ablations."
                      : alignment_status == "public-source-aligned"
                            ? "Source alignment covers fewer than three independent source families; Roosendaal, Oliveira e Silva, and Barina imports still need to agree."
                      : "This is still empirical structure: source-record alignment and independent new seeded samples are the next falsification gates."
                : "No claim is promoted beyond empirical pattern evidence until independent range, residue, and feature-ablation holdouts agree.";
        const std::string conclusion =
            confidence_level == "multi-source-aligned candidate"
                ? "The AI evidence engine has a multi-source-aligned candidate: the learned path-family signal survives current holdouts and agrees with multiple public source families, but it is still not a proof."
                : confidence_level == "source-aligned candidate"
                ? "The AI evidence engine has a source-aligned candidate: the learned path-family signal survives current holdouts and matches public source targets, but it is still not a proof."
                : confidence_level == "range-stable signal"
                ? "The AI evidence engine has a range-stable learned path-family signal across current holdouts, but it is still not a proof or source-aligned candidate."
                : contrastive_status == "complete" && purity_lift > 0.05
                      ? "The AI evidence engine has a sample-local learned path-family signal; it needs holdout and ablation validation before promotion."
                      : selected_rows > 0
                            ? "The system has moved from a narrow topology sample to evidence-first pattern testing; conclusions remain sample-local."
                            : "The current dashboard is still a pipeline/topology check until the stratified evidence sample is generated.";
        const std::string next_experiment =
            confidence_level == "multi-source-aligned candidate"
                ? "Run source-anchored path-image and GNN ablations, then test whether independent seeded samples keep the same source neighborhoods."
                : confidence_level == "source-aligned candidate"
                ? "Add Roosendaal, Oliveira e Silva, and Barina record imports, then rerun source-neighborhood, path-image, and GNN ablations."
                : confidence_level == "range-stable signal"
                ? alignment_status == "source-smoke-aligned"
                      ? "Import larger dated Roosendaal/Oliveira/Barina/OEIS source records, then rerun source-neighborhood alignment."
                      : "Run independent seeded stratified samples, source-record neighborhood comparisons, and path-image/GNN ablations."
                : contrastive_status == "complete" && purity_lift > 0.05
                      ? "Run independent range/residue holdouts and feature ablations against the learned embeddings."
                      : contrastive_status == "complete"
                            ? "Retune contrastive labels/features and compare against autoencoder anomalies before promoting a learned signal."
                            : selected_rows > 0 ? "Train the contrastive encoder and autoencoder on the stratified sample." : "Generate the stratified full-scan evidence sample.";
        const std::string source_alignment_summary =
            alignment_status == "multi-source-aligned"
                ? "Multi-source check matched " + std::to_string(matched_source_targets) + " of " +
                      std::to_string(source_target_count) + " validation starts from " +
                      std::to_string(source_family_count) + " source families across " +
                      std::to_string(matched_source_clusters) + " topology clusters."
                : alignment_status == "public-source-aligned"
                ? "Public source target check matched " + std::to_string(matched_source_targets) + " of " +
                      std::to_string(source_target_count) + " validation starts across " +
                      std::to_string(matched_source_clusters) + " topology clusters."
                : alignment_status == "source-smoke-aligned"
                ? "Source smoke check matched " + std::to_string(matched_source_targets) + " of " +
                      std::to_string(source_target_count) + " known validation starts across " +
                      std::to_string(matched_source_clusters) + " topology clusters."
                : alignment_status == "missing"
                      ? "Source-neighborhood alignment has not run yet."
                      : "Source-neighborhood alignment is partial: " + std::to_string(matched_source_targets) +
                            " of " + std::to_string(source_target_count) + " targets matched.";

        std::filesystem::create_directories(options.output_dir);
        write_hypotheses_jsonl(path_join(options.output_dir, "hypotheses.jsonl"), hypotheses);
        write_summary(
            path_join(options.output_dir, "summary.json"),
            hypotheses,
            confidence_level,
            conclusion,
            coverage,
            strongest_evidence,
            weakest_limit,
            latest_neural_result,
            next_experiment,
            source_alignment_summary);

        std::cout << "hypotheses=" << hypotheses.size()
                  << " confidence_level=" << confidence_level
                  << " output_dir=" << options.output_dir
                  << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
