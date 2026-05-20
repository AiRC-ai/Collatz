#include "collatz/core.hpp"

#include <arpa/inet.h>
#include <csignal>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>

namespace {

volatile std::sig_atomic_t g_stop = 0;

struct Options {
    std::string host = "0.0.0.0";
    std::uint16_t port = 8080;
    std::string progress = "logs/progress.jsonl";
    std::string ledger = "logs/iteration-ledger.jsonl";
    std::string scan_metadata = "data/generated/features.bin.metadata.json";
    std::string embedding_metadata = "data/generated/ml/metadata.json";
    std::string image_manifest = "data/generated/images/manifest.json";
    std::string graph_manifest = "data/generated/graphs/trajectory_graph.json";
    std::string topology_manifest = "data/generated/topology/embedding_topology.json";
    std::string neighborhood_manifest = "data/generated/topology/neighborhoods.json";
    std::string insights_manifest = "data/generated/insights/insights.json";
    std::string gnn_metadata = "data/generated/gnn/metrics.json";
    std::string hypothesis_summary = "data/generated/hypotheses/summary.json";
    std::string runner_status = "data/generated/runner/status.json";
    std::string neural_status = "data/generated/runner/neural_parallel_status.json";
    std::string full_audit = "data/generated/full_audit/summary.json";
    std::string evidence_summary = "data/generated/evidence/latest_public_summary.json";
};

void usage(std::ostream &out) {
    out << "usage: collatz_web [--host IP] [--port N] [--progress FILE] [--ledger FILE]\n"
        << "                   [--scan-metadata FILE] [--embedding-metadata FILE]\n"
        << "                   [--image-manifest FILE] [--graph-manifest FILE]\n"
        << "                   [--topology-manifest FILE] [--neighborhood-manifest FILE]\n"
        << "                   [--insights-manifest FILE] [--gnn-metadata FILE]\n"
        << "                   [--hypothesis-summary FILE] [--runner-status FILE]\n"
        << "                   [--neural-status FILE] [--full-audit FILE]\n"
        << "                   [--evidence-summary FILE]\n";
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
        if (arg == "--host") {
            options.host = need_value("--host");
        } else if (arg == "--port") {
            auto value = collatz::parse_u64(need_value("--port"));
            if (!value || *value == 0 || *value > 65535) {
                throw std::runtime_error("--port must be 1..65535");
            }
            options.port = static_cast<std::uint16_t>(*value);
        } else if (arg == "--progress") {
            options.progress = need_value("--progress");
        } else if (arg == "--ledger") {
            options.ledger = need_value("--ledger");
        } else if (arg == "--scan-metadata") {
            options.scan_metadata = need_value("--scan-metadata");
        } else if (arg == "--embedding-metadata") {
            options.embedding_metadata = need_value("--embedding-metadata");
        } else if (arg == "--image-manifest") {
            options.image_manifest = need_value("--image-manifest");
        } else if (arg == "--graph-manifest") {
            options.graph_manifest = need_value("--graph-manifest");
        } else if (arg == "--topology-manifest") {
            options.topology_manifest = need_value("--topology-manifest");
        } else if (arg == "--neighborhood-manifest") {
            options.neighborhood_manifest = need_value("--neighborhood-manifest");
        } else if (arg == "--insights-manifest") {
            options.insights_manifest = need_value("--insights-manifest");
        } else if (arg == "--gnn-metadata") {
            options.gnn_metadata = need_value("--gnn-metadata");
        } else if (arg == "--hypothesis-summary") {
            options.hypothesis_summary = need_value("--hypothesis-summary");
        } else if (arg == "--runner-status") {
            options.runner_status = need_value("--runner-status");
        } else if (arg == "--neural-status") {
            options.neural_status = need_value("--neural-status");
        } else if (arg == "--full-audit") {
            options.full_audit = need_value("--full-audit");
        } else if (arg == "--evidence-summary") {
            options.evidence_summary = need_value("--evidence-summary");
        } else if (arg == "--help" || arg == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

void on_signal(int) {
    g_stop = 1;
}

std::string compact_object_from_json(const std::string &json, const std::vector<std::string> &keys);

std::string latest_progress_json(const Options &options) {
    const auto line = collatz::read_last_nonempty_line(options.progress);
    if (line.empty()) {
        return "{\"status\":\"waiting\",\"message\":\"no scanner progress yet\"}";
    }
    const std::string compact = compact_object_from_json(
        line,
        {"type", "timestamp", "mode", "threads", "format", "range_start", "range_end", "current", "processed",
         "throughput_per_sec", "max_total_steps_n", "max_total_steps", "max_peak_n", "max_peak_log2"});
    return compact == "null" ? "{\"status\":\"waiting\"}" : compact;
}

std::string read_file(const std::string &path) {
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
            } else if (c == open) {
                ++depth;
            } else if (c == close) {
                --depth;
                if (depth == 0) {
                    break;
                }
            } else if (open == '{' && c == '[') {
                ++depth;
            } else if (open == '{' && c == ']') {
                --depth;
            } else if (open == '[' && c == '{') {
                ++depth;
            } else if (open == '[' && c == '}') {
                --depth;
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
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return {};
    }
    const std::size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return {};
    }
    const std::size_t start = skip_ws(json, colon + 1);
    const std::size_t end = json_value_end(json, start);
    if (end <= start) {
        return {};
    }
    return json.substr(start, end - start);
}

std::string json_value_or(const std::string &json, const std::string &key, const std::string &fallback) {
    const std::string value = json_value_for_key(json, key);
    return value.empty() ? fallback : value;
}

std::string compact_object_from_json(const std::string &json, const std::vector<std::string> &keys) {
    if (json.empty()) {
        return "null";
    }
    std::ostringstream out;
    out << '{';
    bool wrote = false;
    for (const auto &key : keys) {
        const std::string value = json_value_for_key(json, key);
        if (value.empty()) {
            continue;
        }
        if (wrote) {
            out << ',';
        }
        out << '"' << key << "\":" << value;
        wrote = true;
    }
    out << '}';
    return wrote ? out.str() : "null";
}

std::string limited_json_array(const std::string &array_json, std::size_t limit) {
    std::size_t pos = skip_ws(array_json, 0);
    if (pos >= array_json.size() || array_json[pos] != '[') {
        return "[]";
    }
    ++pos;
    std::ostringstream out;
    out << '[';
    std::size_t count = 0;
    bool wrote = false;
    while (pos < array_json.size()) {
        pos = skip_ws(array_json, pos);
        if (pos >= array_json.size() || array_json[pos] == ']') {
            break;
        }
        if (count >= limit) {
            break;
        }
        const std::size_t end = json_value_end(array_json, pos);
        if (end <= pos) {
            break;
        }
        if (wrote) {
            out << ',';
        }
        out << array_json.substr(pos, end - pos);
        wrote = true;
        ++count;
        pos = skip_ws(array_json, end);
        if (pos < array_json.size() && array_json[pos] == ',') {
            ++pos;
        }
    }
    out << ']';
    return out.str();
}

std::string compact_graph_manifest_json(const Options &options) {
    const std::string json = read_file(options.graph_manifest);
    if (json.empty()) {
        return "null";
    }
    const std::string preview = json_value_for_key(json, "preview");
    const std::string nodes = limited_json_array(json_value_for_key(preview, "nodes"), 64);
    const std::string edges = limited_json_array(json_value_for_key(preview, "edges"), 96);
    std::ostringstream out;
    out << "{\"node_count\":" << json_value_or(json, "node_count", "0")
        << ",\"edge_count\":" << json_value_or(json, "edge_count", "0")
        << ",\"preview\":{\"nodes\":" << nodes << ",\"edges\":" << edges << "}}";
    return out.str();
}

std::string compact_topology_manifest_json(const Options &options) {
    const std::string json = read_file(options.topology_manifest);
    if (json.empty()) {
        return "null";
    }
    std::ostringstream out;
    out << "{\"point_count\":" << json_value_or(json, "point_count", "0")
        << ",\"cluster_count\":" << json_value_or(json, "cluster_count", "0")
        << ",\"preview_points\":" << limited_json_array(json_value_for_key(json, "preview_points"), 64)
        << "}";
    return out.str();
}

std::string compact_insights_manifest_json(const Options &options) {
    const std::string json = read_file(options.insights_manifest);
    if (json.empty()) {
        return "null";
    }
    std::ostringstream out;
    out << "{\"insight_count\":" << json_value_or(json, "insight_count", "0")
        << ",\"confidence\":" << json_value_or(json, "confidence", "\"pending\"")
        << ",\"conclusion\":" << json_value_or(json, "conclusion", "\"pending\"")
        << ",\"meaning\":" << json_value_or(json, "meaning", "\"pending\"")
        << ",\"limit\":" << json_value_or(json, "limit", "\"pending\"")
        << ",\"next_step\":" << json_value_or(json, "next_step", "\"pending\"")
        << ",\"findings\":" << limited_json_array(json_value_for_key(json, "findings"), 3)
        << "}";
    return out.str();
}

std::string compact_hypothesis_summary_json(const Options &options) {
    const std::string json = read_file(options.hypothesis_summary);
    if (json.empty()) {
        return "null";
    }
    std::ostringstream out;
    out << "{\"hypothesis_count\":" << json_value_or(json, "hypothesis_count", "0")
        << ",\"confidence_level\":" << json_value_or(json, "confidence_level", "\"pipeline-check\"")
        << ",\"conclusion\":" << json_value_or(json, "conclusion", "\"pending\"")
        << ",\"coverage\":" << json_value_or(json, "coverage", "\"pending\"")
        << ",\"strongest_evidence\":" << json_value_or(json, "strongest_evidence", "\"pending\"")
        << ",\"weakest_limit\":" << json_value_or(json, "weakest_limit", "\"pending\"")
        << ",\"latest_neural_result\":" << json_value_or(json, "latest_neural_result", "\"pending\"")
        << ",\"next_experiment\":" << json_value_or(json, "next_experiment", "\"pending\"")
        << ",\"source_alignment\":" << json_value_or(json, "source_alignment", "\"pending\"")
        << "}";
    return out.str();
}

std::string compact_full_audit_json(const Options &options) {
    const std::string json = read_file(options.full_audit);
    if (json.empty()) {
        return "null";
    }
    std::ostringstream out;
    out << "{\"records_read\":" << json_value_or(json, "records_read", "0")
        << ",\"coverage_ratio\":" << json_value_or(json, "coverage_ratio", "0")
        << ",\"max_total_steps\":" << json_value_or(json, "max_total_steps", "0")
        << ",\"max_total_steps_n\":" << json_value_or(json, "max_total_steps_n", "0")
        << ",\"total_steps_mean\":" << json_value_or(json, "total_steps_mean", "0")
        << ",\"total_steps_quantiles\":" << json_value_or(json, "total_steps_quantiles", "{}")
        << "}";
    return out.str();
}

bool unsafe_public_value(const std::string &value) {
    return value.find("/Users/") != std::string::npos || value.find("/home/") != std::string::npos ||
           value.find("10.") != std::string::npos || value.find("192.168.") != std::string::npos ||
           value.find("172.16.") != std::string::npos || value.find("172.17.") != std::string::npos ||
           value.find("172.18.") != std::string::npos || value.find("http://") != std::string::npos ||
           value.find("https://") != std::string::npos || value.find("ssh ") != std::string::npos;
}

std::string safe_runner_value_or(const std::string &json, const std::string &key, const std::string &fallback) {
    const std::string value = json_value_for_key(json, key);
    if (value.empty()) {
        return fallback;
    }
    return unsafe_public_value(value) ? fallback : value;
}

std::string compact_runner_status_json(const Options &options) {
    const std::string json = read_file(options.runner_status);
    if (json.empty()) {
        return "{\"state\":\"idle\",\"current_stage\":\"not configured\",\"last_started_utc\":\"\",\"last_finished_utc\":\"\",\"last_success\":false,\"last_error_summary\":\"\",\"active_experiment\":\"source alignment plus optional CPU/GPU crunch\",\"source_target_count\":0,\"matched_source_targets\":0,\"source_match_rate\":0,\"evidence_score\":0,\"evidence_delta\":0,\"cycle_count\":0,\"cpu_crunch_state\":\"disabled\",\"cpu_threads\":0,\"cpu_target_end\":0,\"gpu_state\":\"unknown\",\"gpu_free_memory_mb\":0,\"gpu_compute_process_count\":0,\"neural_stage\":\"disabled\",\"full_audit_state\":\"missing\",\"full_audit_records\":0,\"next_stage\":\"run evidence cycle\"}";
    }
    std::ostringstream out;
    out << "{\"state\":" << safe_runner_value_or(json, "state", "\"idle\"")
        << ",\"current_stage\":" << safe_runner_value_or(json, "current_stage", "\"idle\"")
        << ",\"last_started_utc\":" << safe_runner_value_or(json, "last_started_utc", "\"\"")
        << ",\"last_finished_utc\":" << safe_runner_value_or(json, "last_finished_utc", "\"\"")
        << ",\"last_success\":" << safe_runner_value_or(json, "last_success", "false")
        << ",\"last_error_summary\":" << safe_runner_value_or(json, "last_error_summary", "\"\"")
        << ",\"active_experiment\":" << safe_runner_value_or(json, "active_experiment", "\"source alignment plus optional CPU/GPU crunch\"")
        << ",\"source_target_count\":" << safe_runner_value_or(json, "source_target_count", "0")
        << ",\"matched_source_targets\":" << safe_runner_value_or(json, "matched_source_targets", "0")
        << ",\"source_match_rate\":" << safe_runner_value_or(json, "source_match_rate", "0")
        << ",\"evidence_score\":" << safe_runner_value_or(json, "evidence_score", "0")
        << ",\"evidence_delta\":" << safe_runner_value_or(json, "evidence_delta", "0")
        << ",\"cycle_count\":" << safe_runner_value_or(json, "cycle_count", "0")
        << ",\"cpu_crunch_state\":" << safe_runner_value_or(json, "cpu_crunch_state", "\"disabled\"")
        << ",\"cpu_threads\":" << safe_runner_value_or(json, "cpu_threads", "0")
        << ",\"cpu_target_end\":" << safe_runner_value_or(json, "cpu_target_end", "0")
        << ",\"gpu_state\":" << safe_runner_value_or(json, "gpu_state", "\"unknown\"")
        << ",\"gpu_free_memory_mb\":" << safe_runner_value_or(json, "gpu_free_memory_mb", "0")
        << ",\"gpu_compute_process_count\":" << safe_runner_value_or(json, "gpu_compute_process_count", "0")
        << ",\"neural_stage\":" << safe_runner_value_or(json, "neural_stage", "\"disabled\"")
        << ",\"full_audit_state\":" << safe_runner_value_or(json, "full_audit_state", "\"missing\"")
        << ",\"full_audit_records\":" << safe_runner_value_or(json, "full_audit_records", "0")
        << ",\"next_stage\":" << safe_runner_value_or(json, "next_stage", "\"run evidence cycle\"")
        << "}";
    return out.str();
}

std::string compact_neural_status_json(const Options &options) {
    const std::string json = read_file(options.neural_status);
    if (json.empty()) {
        return "{\"state\":\"idle\",\"current_stage\":\"not configured\",\"updated_utc\":\"\",\"concurrency_limit\":0,\"active_jobs\":0,\"complete_jobs\":0,\"failed_jobs\":0,\"gpu_utilization_percent\":0,\"gpu_memory_used_mb\":0,\"gpu_memory_free_mb\":0,\"gpu_power_watts\":0,\"jobs\":[]}";
    }
    std::ostringstream out;
    out << "{\"state\":" << safe_runner_value_or(json, "state", "\"idle\"")
        << ",\"current_stage\":" << safe_runner_value_or(json, "current_stage", "\"idle\"")
        << ",\"updated_utc\":" << safe_runner_value_or(json, "updated_utc", "\"\"")
        << ",\"concurrency_limit\":" << safe_runner_value_or(json, "concurrency_limit", "0")
        << ",\"active_jobs\":" << safe_runner_value_or(json, "active_jobs", "0")
        << ",\"complete_jobs\":" << safe_runner_value_or(json, "complete_jobs", "0")
        << ",\"failed_jobs\":" << safe_runner_value_or(json, "failed_jobs", "0")
        << ",\"gpu_utilization_percent\":" << safe_runner_value_or(json, "gpu_utilization_percent", "0")
        << ",\"gpu_memory_used_mb\":" << safe_runner_value_or(json, "gpu_memory_used_mb", "0")
        << ",\"gpu_memory_free_mb\":" << safe_runner_value_or(json, "gpu_memory_free_mb", "0")
        << ",\"gpu_power_watts\":" << safe_runner_value_or(json, "gpu_power_watts", "0")
        << ",\"jobs\":" << limited_json_array(json_value_for_key(json, "jobs"), 8)
        << "}";
    return out.str();
}

std::string dashboard_progress_json(const Options &options) {
    const std::string scan = read_file(options.scan_metadata);
    const std::string gnn = read_file(options.gnn_metadata);
    std::string evidence = read_file(options.evidence_summary);
    if (evidence.empty()) {
        evidence = "{\"schema_version\":\"evidence_public_summary_v1\",\"confidence\":{\"label\":\"sample-local signal\",\"interpretation\":\"Canonical evidence summary is not available yet.\"}}";
    }
    return "{\"evidence\":" + evidence +
           ",\"operations\":{\"progress\":" + latest_progress_json(options) +
           ",\"scan_metadata\":" + compact_object_from_json(scan, {"dataset_records_observed", "completed_utc"}) +
           ",\"graph_manifest\":" + compact_graph_manifest_json(options) +
           ",\"topology_manifest\":" + compact_topology_manifest_json(options) +
           ",\"neighborhood_manifest\":" +
           compact_object_from_json(read_file(options.neighborhood_manifest), {"neighborhood_count", "neighbors_per_center"}) +
           ",\"insights_manifest\":" + compact_insights_manifest_json(options) +
           ",\"hypothesis_summary\":" + compact_hypothesis_summary_json(options) +
           ",\"full_audit\":" + compact_full_audit_json(options) +
           ",\"gnn_metadata\":" + compact_object_from_json(gnn, {"status", "device", "epochs", "loss_final"}) +
           ",\"runner_status\":" + compact_runner_status_json(options) +
           ",\"neural_status\":" + compact_neural_status_json(options) +
           "}}";
}

std::string html_page() {
    return R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3xN1 Progress</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #0a0f1d; color: #f5f7fb; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    header { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    h1 { margin: 0; font-size: 25px; font-weight: 750; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 15px; font-weight: 700; color: #d8def1; }
    h3 { margin: 0 0 6px; font-size: 12px; color: #9aa7c7; text-transform: uppercase; letter-spacing: .08em; }
    .subtle { margin: 6px 0 0; color: #9aa7c7; font-size: 14px; }
    .status { color: #83e6a5; font-size: 13px; white-space: nowrap; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .panel { border: 1px solid #27314f; border-radius: 8px; padding: 13px; background: #111832; min-width: 0; }
    .label { color: #9aa7c7; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 6px; font-size: 21px; font-weight: 760; overflow-wrap: anywhere; line-height: 1.05; }
    .canvas-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
    .canvas-panel { border: 1px solid #27314f; border-radius: 8px; padding: 12px; background: #0d1427; min-width: 0; }
    canvas { width: 100%; height: 300px; display: block; background: #070b16; border: 1px solid #27314f; border-radius: 8px; }
    .readout { border: 1px solid #27314f; border-radius: 8px; padding: 16px; background: #111832; margin-bottom: 12px; }
    .readout-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; }
    .readout-box { border: 1px solid #27314f; border-radius: 8px; padding: 12px; background: #0d1427; }
    .readout-text { color: #eef3ff; font-size: 14px; line-height: 1.38; }
    .readout-note { color: #b8c3df; font-size: 13px; line-height: 1.38; }
    @media (max-width: 980px) {
      .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .canvas-grid { grid-template-columns: 1fr; }
      .readout-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      main { padding: 16px; }
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .value { font-size: 18px; }
      canvas { height: 260px; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>3xN1 Collatz Research</h1>
      <p class="subtle">Scan, embedding topology, and Graph Neural Network progress.</p>
    </div>
    <div id="status" class="status">waiting</div>
  </header>
  <section class="grid">
    <div class="panel"><div class="label">Dataset</div><div id="records" class="value">0</div></div>
    <div class="panel"><div class="label">Full Audit</div><div id="fullAudit" class="value">missing</div></div>
    <div class="panel"><div class="label">Throughput</div><div id="throughput" class="value">0/s</div></div>
    <div class="panel"><div class="label">Evidence Score</div><div id="evidenceScore" class="value">0</div></div>
    <div class="panel"><div class="label">Topology</div><div id="topology" class="value">0 / 0</div></div>
    <div class="panel"><div class="label">Automation</div><div id="automation" class="value">idle</div></div>
    <div class="panel"><div class="label">CPU Crunch</div><div id="cpuCrunch" class="value">disabled</div></div>
    <div class="panel"><div class="label">GPU / Neural</div><div id="gpuStage" class="value">disabled</div></div>
    <div class="panel"><div class="label">Source Gate</div><div id="sourceGate" class="value">0 / 0</div></div>
    <div class="panel"><div class="label">GNN Graph</div><div id="graph" class="value">0 / 0</div></div>
    <div class="panel"><div class="label">Confidence</div><div id="aiConfidence" class="value">pending</div></div>
  </section>
  <section class="canvas-grid">
    <div class="canvas-panel">
      <h2>Embedding Topology</h2>
      <canvas id="topologyCanvas" width="720" height="320"></canvas>
    </div>
    <div class="canvas-panel">
      <h2>GNN Trajectory Graph</h2>
      <canvas id="graphCanvas" width="720" height="320"></canvas>
    </div>
  </section>
  <section class="readout">
    <h2>AI Readout</h2>
    <div class="readout-grid">
      <div class="readout-box">
        <h3>Current Conclusion</h3>
        <div id="aiConclusion" class="readout-text">waiting for insights</div>
      </div>
      <div class="readout-box">
        <h3>Coverage</h3>
        <div id="aiCoverage" class="readout-note">waiting for hypotheses</div>
      </div>
      <div class="readout-box">
        <h3>Strongest Evidence</h3>
        <div id="aiEvidence" class="readout-note">waiting for hypotheses</div>
      </div>
      <div class="readout-box">
        <h3>Weakest Limit</h3>
        <div id="aiLimit" class="readout-note">waiting for hypotheses</div>
      </div>
      <div class="readout-box">
        <h3>Latest Neural Result</h3>
        <div id="aiNeural" class="readout-note">waiting for neural metrics</div>
      </div>
      <div class="readout-box">
        <h3>Source Check</h3>
        <div id="aiSource" class="readout-note">waiting for source alignment</div>
      </div>
      <div class="readout-box">
        <h3>Next Test</h3>
        <div id="aiNextStep" class="readout-note">waiting for hypotheses</div>
      </div>
    </div>
  </section>
</main>
<script>
function compact(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return '0';
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 2 }).format(number);
}

function fixed(value, digits) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'pending';
}

async function refresh() {
  const response = await fetch('/api/progress');
  const data = await response.json();
  const evidence = data.evidence || {};
  const operations = data.operations || {};
  const progress = operations.progress || {};
  const scan = operations.scan_metadata || {};
  const graph = operations.graph_manifest || {};
  const topology = operations.topology_manifest || {};
  const fullAudit = operations.full_audit || {};
  const gnn = operations.gnn_metadata || {};
  const runner = operations.runner_status || {};
  const neuralOps = operations.neural_status || {};
  const confidence = evidence.confidence || {};
  const audit = evidence.audit || {};
  const coverage = evidence.coverage || {};
  const neuralEvidence = evidence.neural || {};
  const sourceAlignment = evidence.source_alignment || {};
  const mode = progress.mode || progress.status || gnn.status || 'waiting';
  const processed = progress.processed ? `, ${compact(progress.processed)} processed` : '';
  const throughput = progress.throughput_per_sec ? ` at ${compact(progress.throughput_per_sec)}/s` : '';
  document.getElementById('status').textContent = `${mode}${processed}${throughput}`;
  document.getElementById('records').textContent = compact(audit.rows || scan.dataset_records_observed);
  const auditRows = Number(fullAudit.records_read || runner.full_audit_records || 0);
  const auditState = runner.full_audit_state || (auditRows ? 'complete' : 'missing');
  document.getElementById('fullAudit').textContent = audit.rows ? `${compact(audit.rows)} canonical` : (auditRows ? `${compact(auditRows)} ${auditState}` : auditState);
  document.getElementById('throughput').textContent = progress.throughput_per_sec ? `${compact(progress.throughput_per_sec)}/s` : 'idle';
  const score = Number(runner.evidence_score || 0);
  const delta = Number(runner.evidence_delta || 0);
  const deltaText = delta > 0 ? ` +${delta.toFixed(2)}` : delta < 0 ? ` ${delta.toFixed(2)}` : ' +0.00';
  document.getElementById('evidenceScore').textContent = score ? `${score.toFixed(2)}${deltaText}` : 'pending';
  document.getElementById('topology').textContent = `${compact(topology.point_count)} / ${compact(topology.cluster_count)} clusters`;
  const stage = runner.current_stage || runner.next_stage || 'idle';
  document.getElementById('automation').textContent = `${runner.state || 'idle'}: ${stage}`;
  const cpuState = runner.cpu_crunch_state || 'disabled';
  const cpuThreads = Number(runner.cpu_threads || 0);
  const cpuTarget = Number(runner.cpu_target_end || 0);
  document.getElementById('cpuCrunch').textContent = cpuState === 'running'
    ? `${cpuThreads || '?'} threads -> ${compact(cpuTarget)}`
    : cpuState;
  const gpuState = runner.gpu_state || 'unknown';
  const neuralStage = runner.neural_stage || 'disabled';
  const gpuFree = Number(runner.gpu_free_memory_mb || 0);
  const neuralActive = Number(neuralOps.active_jobs || 0);
  const neuralComplete = Number(neuralOps.complete_jobs || 0);
  const neuralFailed = Number(neuralOps.failed_jobs || 0);
  const gpuUtil = Number(neuralOps.gpu_utilization_percent || 0);
  const gpuUsed = Number(neuralOps.gpu_memory_used_mb || 0);
  document.getElementById('gpuStage').textContent = neuralActive || neuralComplete || neuralFailed
    ? `${neuralActive} active / ${neuralComplete} done, ${gpuUtil.toFixed(0)}% GPU, ${compact(gpuUsed)} MB`
    : gpuFree
    ? `${neuralStage} / ${gpuState} (${compact(gpuFree)} MB free)`
    : `${neuralStage} / ${gpuState}`;
  document.getElementById('sourceGate').textContent = sourceAlignment.targets_total ? `${compact(sourceAlignment.matched)} / ${compact(sourceAlignment.targets_total)}` : 'pending';
  document.getElementById('graph').textContent = `${compact(graph.node_count)} / ${compact(graph.edge_count)} edges`;
  document.getElementById('aiConfidence').textContent = confidence.label || 'pending';
  document.getElementById('aiConclusion').textContent = confidence.interpretation || 'waiting for canonical evidence';
  const topologyCoverage = coverage.topology || {};
  const stratifiedCoverage = coverage.stratified_evidence_sample || {};
  document.getElementById('aiCoverage').textContent = audit.rows
    ? `${compact(audit.rows)} audited rows; topology ${fixed(topologyCoverage.percent_of_audit, 3)}%; stratified sample ${fixed(stratifiedCoverage.percent_of_audit, 3)}%.`
    : 'waiting for canonical audit coverage';
  const holdouts = neuralEvidence.holdouts || {};
  document.getElementById('aiEvidence').textContent = holdouts.weakest_range_lift_percent !== undefined
    ? `Lift over random is represented in the leaderboard; range minimum ${fixed(holdouts.weakest_range_lift_percent, 3)}%, fold minimum ${fixed(holdouts.fold_min_lift_percent, 3)}%, numeric-adjacency lift ${fixed(holdouts.numeric_adjacency_lift_percent, 3)}%.`
    : 'waiting for neural holdout evidence';
  const neuralInterpretation = neuralEvidence.interpretation || {};
  const blockers = Array.isArray(confidence.promotion_blockers) ? confidence.promotion_blockers : [];
  document.getElementById('aiLimit').textContent = blockers.length
    ? `${neuralInterpretation.reason || 'current gates are incomplete'} Blockers: ${blockers.slice(0, 4).join(', ')}${blockers.length > 4 ? '...' : ''}.`
    : neuralInterpretation.reason || 'waiting for canonical limitation';
  const latestRun = neuralEvidence.latest_run || {};
  document.getElementById('aiNeural').textContent = latestRun.sample_rows
    ? `${compact(latestRun.sample_rows)} sample rows, ${latestRun.parallel_jobs_completed || 0} parallel jobs, GPU used: ${latestRun.gpu_used ? 'yes' : 'no'}.`
    : 'waiting for neural metrics';
  const taxonomy = sourceAlignment.unmatched_breakdown || {};
  const unknown = taxonomy.unknown || 0;
  const trueMismatch = taxonomy.true_mismatch || 0;
  const missingFeature = taxonomy.missing_feature_row || 0;
  const missingTopology = (taxonomy.missing_from_topology_sample || 0) + (taxonomy.missing_topology_projection_node || 0);
  document.getElementById('aiSource').textContent = sourceAlignment.targets_total
    ? `${compact(sourceAlignment.matched)} of ${compact(sourceAlignment.targets_total)} public source targets matched; taxonomy unknown ${compact(unknown)}, true mismatch ${compact(trueMismatch)}, missing feature ${compact(missingFeature)}, missing topology ${compact(missingTopology)}.`
    : 'waiting for source alignment';
  document.getElementById('aiNextStep').textContent = evidence.next_experiment?.summary || 'waiting for canonical next experiment';
  drawTopology(topology);
  drawGraph(graph);
}

function drawTopology(topology) {
  const canvas = document.getElementById('topologyCanvas');
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#070b16';
  ctx.fillRect(0, 0, width, height);
  const points = topology.preview_points || [];
  if (!points.length) {
    ctx.fillStyle = '#9aa7c7';
    ctx.font = '16px system-ui';
    ctx.fillText('No embedding topology artifact yet', 24, 40);
    return;
  }
  const xs = points.map(p => p.x || 0);
  const ys = points.map(p => p.y || 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(1e-9, maxX - minX);
  const spanY = Math.max(1e-9, maxY - minY);
  points.forEach(point => {
    const x = 24 + ((point.x || 0) - minX) / spanX * (width - 48);
    const y = height - 24 - ((point.y || 0) - minY) / spanY * (height - 48);
    ctx.beginPath();
    ctx.fillStyle = `hsl(${((point.cluster || 0) * 47) % 360}, 72%, 62%)`;
    ctx.arc(x, y, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawGraph(graph) {
  const canvas = document.getElementById('graphCanvas');
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#070b16';
  ctx.fillRect(0, 0, width, height);
  const nodes = graph.preview?.nodes || [];
  const edges = graph.preview?.edges || [];
  if (!nodes.length) {
    ctx.fillStyle = '#9aa7c7';
    ctx.font = '16px system-ui';
    ctx.fillText('No GNN graph artifact yet', 24, 40);
    return;
  }
  const maxLog = Math.max(...nodes.map(n => n.log2 || 0), 1);
  const positions = new Map();
  nodes.forEach((node, index) => {
    const x = 24 + (index % 64) * ((width - 48) / 63);
    const band = Math.floor(index / 64);
    const yBase = height - 28 - ((node.log2 || 0) / maxLog) * (height - 64);
    const y = Math.max(18, Math.min(height - 18, yBase + (band % 5) * 5));
    positions.set(node.id, {x, y});
  });
  ctx.strokeStyle = 'rgba(116, 162, 255, 0.24)';
  ctx.lineWidth = 1;
  edges.slice(0, 1200).forEach(edge => {
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) return;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  nodes.forEach(node => {
    const p = positions.get(node.id);
    if (!p) return;
    ctx.beginPath();
    ctx.fillStyle = node.is_terminal ? '#83e6a5' : node.is_start ? '#ffce6b' : `hsl(${(node.residue_mod32 || 0) * 11}, 72%, 62%)`;
    ctx.arc(p.x, p.y, node.is_start || node.is_terminal ? 4 : 2.4, 0, Math.PI * 2);
    ctx.fill();
  });
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
)HTML";
}

std::string http_response(std::string_view content_type, std::string_view body) {
    std::ostringstream out;
    out << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return out.str();
}

std::string not_found_response() {
    const std::string body = "not found\n";
    std::ostringstream out;
    out << "HTTP/1.1 404 Not Found\r\n"
        << "Content-Type: text/plain\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return out.str();
}

std::string handle_request(const Options &options, const std::string &request) {
    const bool wants_api = request.rfind("GET /api/progress", 0) == 0;
    const bool wants_health = request.rfind("GET /health", 0) == 0;
    const bool wants_home = request.rfind("GET / ", 0) == 0 || request.rfind("GET /HTTP", 0) == 0;

    if (wants_api) {
        return http_response("application/json", dashboard_progress_json(options));
    }
    if (wants_health) {
        return http_response("application/json", "{\"ok\":true}");
    }
    if (wants_home) {
        return http_response("text/html; charset=utf-8", html_page());
    }
    return not_found_response();
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_args(argc, argv);
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) {
            throw std::runtime_error("socket failed");
        }
        int yes = 1;
        setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(options.port);
        if (inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
            throw std::runtime_error("invalid IPv4 host: " + options.host);
        }
        if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
            throw std::runtime_error(std::string("bind failed: ") + std::strerror(errno));
        }
        if (listen(server, 16) != 0) {
            throw std::runtime_error("listen failed");
        }

        std::cout << "collatz_web listening on http://" << options.host << ':' << options.port << "\n";
        while (!g_stop) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(server, &read_fds);
            timeval timeout{};
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;
            const int ready = select(server + 1, &read_fds, nullptr, nullptr, &timeout);
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("select failed");
            }
            if (ready == 0) {
                continue;
            }

            sockaddr_in client_address{};
            socklen_t client_len = sizeof(client_address);
            const int client = accept(server, reinterpret_cast<sockaddr *>(&client_address), &client_len);
            if (client < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("accept failed");
            }

            char buffer[4096] = {};
            const ssize_t bytes = read(client, buffer, sizeof(buffer) - 1);
            const std::string request = bytes > 0 ? std::string(buffer, static_cast<std::size_t>(bytes)) : std::string();
            const std::string response = handle_request(options, request);
            const char *data = response.data();
            std::size_t remaining = response.size();
            while (remaining > 0) {
                const ssize_t written = write(client, data, remaining);
                if (written <= 0) {
                    break;
                }
                data += written;
                remaining -= static_cast<std::size_t>(written);
            }
            close(client);
        }
        close(server);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        usage(std::cerr);
        return 1;
    }
}
