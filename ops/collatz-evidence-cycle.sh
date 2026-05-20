#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --once) ;;
    --help|-h)
      echo "usage: ops/collatz-evidence-cycle.sh [--dry-run] [--once]"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="${COLLATZ_REPO_DIR:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry-run stage: build tools if needed"
  echo "dry-run stage: fetch public source records"
  echo "dry-run stage: generate expanded source targets"
  echo "dry-run stage: align source targets to current topology"
  echo "dry-run stage: regenerate hypothesis summary"
  echo "dry-run stage: optionally run one CPU crunch scan batch if configured"
  echo "dry-run stage: run deterministic full dataset audit"
  echo "dry-run stage: optionally run one neural stage if GPU is available and configured"
  echo "dry-run stage: neural stage prepares safe metrics, family labels, and matched pairs when enabled"
  echo "dry-run stage: refresh hypothesis summary after crunch stages"
  echo "dry-run stage: publish canonical evidence summary"
  echo "dry-run stage: regenerate public dashboard and historical evidence images"
  echo "dry-run stage: run public privacy scan"
  echo "dry-run stage: append sanitized iteration ledger"
  echo "dry-run stage: write public-safe runner status"
  exit 0
fi

cd "$REPO_DIR"

RUNNER_DIR="${COLLATZ_RUNNER_DIR:-data/generated/runner}"
STATUS_FILE="${COLLATZ_RUNNER_STATUS:-$RUNNER_DIR/status.json}"
NEURAL_STATUS_FILE="${COLLATZ_NEURAL_STATUS:-$RUNNER_DIR/neural_parallel_status.json}"
HISTORY_FILE="${COLLATZ_RUNNER_HISTORY:-$RUNNER_DIR/history.jsonl}"
LOCK_DIR="${COLLATZ_RUNNER_LOCK:-$RUNNER_DIR/lock}"
BUILD_DIR="${COLLATZ_BUILD_DIR:-build}"
IMPORT_DIR="${COLLATZ_IMPORT_DIR:-data/imported}"
FEATURE_FILE="${COLLATZ_FEATURE_FILE:-data/generated/features.bin}"
PROGRESS_FILE="${COLLATZ_PROGRESS_FILE:-logs/progress.jsonl}"
SCAN_METADATA="${COLLATZ_SCAN_METADATA:-data/generated/features.bin.metadata.json}"
SOURCE_OUTPUT="${COLLATZ_SOURCE_OUTPUT:-data/generated/source_validation/public_source_targets.csv}"
SOURCE_METADATA="${COLLATZ_SOURCE_METADATA:-$SOURCE_OUTPUT.metadata.json}"
SOURCE_ALIGNMENT_DIR="${COLLATZ_SOURCE_ALIGNMENT_DIR:-data/generated/source_alignment}"
PROJECTION_FILE="${COLLATZ_PROJECTION_FILE:-data/generated/topology/projection.csv}"
HYPOTHESES_DIR="${COLLATZ_HYPOTHESES_DIR:-data/generated/hypotheses}"
INSIGHTS_FILE="${COLLATZ_INSIGHTS_FILE:-data/generated/insights/insights.json}"
STRATIFIED_METADATA="${COLLATZ_STRATIFIED_METADATA:-data/generated/stratified/metadata.json}"
CONTRASTIVE_METRICS="${COLLATZ_CONTRASTIVE_METRICS:-data/generated/contrastive/metrics.json}"
AUTOENCODER_METRICS="${COLLATZ_AUTOENCODER_METRICS:-data/generated/anomalies/metrics.json}"
GNN_METRICS="${COLLATZ_GNN_METRICS:-data/generated/gnn/metrics.json}"
VALIDATION_METRICS="${COLLATZ_VALIDATION_METRICS:-data/generated/evidence_validation/metrics.json}"
FULL_AUDIT_FILE="${COLLATZ_FULL_AUDIT_FILE:-data/generated/full_audit/summary.json}"
PUBLIC_EVIDENCE_FILE="${COLLATZ_PUBLIC_EVIDENCE_FILE:-data/generated/evidence/latest_public_summary.json}"
LEDGER_FILE="${COLLATZ_LEDGER_FILE:-logs/iteration-ledger.jsonl}"
MAX_N="${COLLATZ_SOURCE_MAX_N:-100000000}"
GPU_MIN_FREE_MB="${COLLATZ_GPU_MIN_FREE_MB:-4096}"
GPU_ALLOW_SHARED="${COLLATZ_GPU_ALLOW_SHARED:-0}"
RUN_CPU_CRUNCH="${COLLATZ_RUN_CPU_CRUNCH:-0}"
CPU_CRUNCH_END="${COLLATZ_CPU_CRUNCH_END:-}"
CPU_CRUNCH_STEP="${COLLATZ_CPU_CRUNCH_STEP:-10000000}"
CPU_CRUNCH_THREADS="${COLLATZ_CPU_CRUNCH_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
CPU_CRUNCH_CHUNK_SIZE="${COLLATZ_CPU_CRUNCH_CHUNK_SIZE:-250000}"
CPU_CRUNCH_MAX_STEPS="${COLLATZ_CPU_CRUNCH_MAX_STEPS:-10000000}"
CPU_CRUNCH_COMMAND="${COLLATZ_CPU_CRUNCH_COMMAND:-}"
RUN_NEURAL="${COLLATZ_RUN_NEURAL:-0}"
NEURAL_COMMAND="${COLLATZ_NEURAL_COMMAND:-}"
EVIDENCE_SCORE=0
EVIDENCE_DELTA=0
SOURCE_MATCH_RATE=0
CYCLE_COUNT=0
CPU_CRUNCH_STATE="disabled"
CPU_THREADS="$CPU_CRUNCH_THREADS"
CPU_TARGET_END=0
GPU_STATE="unknown"
GPU_FREE_MEMORY_MB=0
GPU_COMPUTE_PROCESS_COUNT=0
NEURAL_STAGE="disabled"
FULL_AUDIT_STATE="pending"
FULL_AUDIT_RECORDS=0

mkdir -p "$RUNNER_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "evidence cycle already running"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\r//g; s/\n/ /g'
}

write_status() {
  local state="$1"
  local current_stage="$2"
  local last_success="$3"
  local error_summary="$4"
  local next_stage="$5"
  local source_target_count="${6:-0}"
  local matched_source_targets="${7:-0}"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$(dirname "$STATUS_FILE")"
  {
    printf '{\n'
    printf '  "state": "%s",\n' "$(json_escape "$state")"
    printf '  "current_stage": "%s",\n' "$(json_escape "$current_stage")"
    printf '  "last_started_utc": "%s",\n' "$(json_escape "${LAST_STARTED_UTC:-$now}")"
    printf '  "last_finished_utc": "%s",\n' "$(json_escape "$now")"
    printf '  "last_success": %s,\n' "$last_success"
    printf '  "last_error_summary": "%s",\n' "$(json_escape "$error_summary")"
    printf '  "active_experiment": "source alignment plus optional CPU/GPU crunch",\n'
    printf '  "source_target_count": %s,\n' "$source_target_count"
    printf '  "matched_source_targets": %s,\n' "$matched_source_targets"
    printf '  "source_match_rate": %s,\n' "$SOURCE_MATCH_RATE"
    printf '  "evidence_score": %s,\n' "$EVIDENCE_SCORE"
    printf '  "evidence_delta": %s,\n' "$EVIDENCE_DELTA"
    printf '  "cycle_count": %s,\n' "$CYCLE_COUNT"
    printf '  "cpu_crunch_state": "%s",\n' "$(json_escape "$CPU_CRUNCH_STATE")"
    printf '  "cpu_threads": %s,\n' "$CPU_THREADS"
    printf '  "cpu_target_end": %s,\n' "$CPU_TARGET_END"
    printf '  "gpu_state": "%s",\n' "$(json_escape "$GPU_STATE")"
    printf '  "gpu_free_memory_mb": %s,\n' "$GPU_FREE_MEMORY_MB"
    printf '  "gpu_compute_process_count": %s,\n' "$GPU_COMPUTE_PROCESS_COUNT"
    printf '  "neural_stage": "%s",\n' "$(json_escape "$NEURAL_STAGE")"
    printf '  "full_audit_state": "%s",\n' "$(json_escape "$FULL_AUDIT_STATE")"
    printf '  "full_audit_records": %s,\n' "$FULL_AUDIT_RECORDS"
    printf '  "next_stage": "%s"\n' "$(json_escape "$next_stage")"
    printf '}\n'
  } > "$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

fail_cycle() {
  local stage="$1"
  local code="${2:-1}"
  write_status "error" "$stage" false "stage failed with exit code $code" "$stage"
  exit "$code"
}

stage() {
  CURRENT_STAGE="$1"
  write_status "running" "$CURRENT_STAGE" false "" "$CURRENT_STAGE"
  echo "stage: $CURRENT_STAGE"
}

read_json_number() {
  local key="$1"
  local file="$2"
  if [ ! -f "$file" ]; then
    echo 0
    return
  fi
  awk -v key="\"$key\"" '
    BEGIN { found=0 }
    index($0, key) {
      value=$0
      sub(/^.*: */, "", value)
      gsub(/[^0-9]/, "", value)
      if (value == "") value=0
      print value
      found=1
      exit
    }
    END { if (found == 0) print 0 }
  ' "$file"
}

read_json_string() {
  local key="$1"
  local file="$2"
  if [ ! -f "$file" ]; then
    echo ""
    return
  fi
  awk -v key="\"$key\"" -F'"' '
    index($0, key) {
      print $4
      found=1
      exit
    }
    END { if (found != 1) print "" }
  ' "$file"
}

read_json_decimal() {
  local key="$1"
  local file="$2"
  if [ ! -f "$file" ]; then
    echo 0
    return
  fi
  awk -v key="\"$key\"" '
    BEGIN { found=0 }
    index($0, key) {
      value=$0
      sub(/^.*: */, "", value)
      gsub(/[^0-9.eE+-]/, "", value)
      if (value == "") value=0
      print value
      found=1
      exit
    }
    END { if (found == 0) print 0 }
  ' "$file"
}

evidence_base_for_confidence() {
  case "$1" in
    "candidate pattern") echo 85 ;;
    "source-neighborhood-supported") echo 75 ;;
    "range-stable signal") echo 62 ;;
    "sample-local signal") echo 35 ;;
    *) echo 10 ;;
  esac
}

last_history_value() {
  local key="$1"
  if [ ! -f "$HISTORY_FILE" ]; then
    echo 0
    return
  fi
  tail -n 1 "$HISTORY_FILE" | awk -v key="\"$key\"" '
    {
      value=$0
      sub("^.*" key "[ ]*:[ ]*", "", value)
      sub(",.*$", "", value)
      gsub(/[^0-9.eE+-]/, "", value)
      if (value == "") value=0
      print value
    }
  '
}

update_evidence_score() {
  local alignment_json="$SOURCE_ALIGNMENT_DIR/source_alignment.json"
  local summary_json="$HYPOTHESES_DIR/summary.json"
  local target_count matched confidence base previous previous_cycle validation_lift neural_bonus audit_records audit_range_end
  target_count="$(read_json_number target_count "$alignment_json")"
  matched="$(read_json_number matched_targets "$alignment_json")"
  confidence="$(read_json_string confidence_level "$summary_json")"
  base="$(evidence_base_for_confidence "$confidence")"
  previous="$(last_history_value evidence_score)"
  previous_cycle="$(last_history_value cycle_count)"
  validation_lift="$(read_json_decimal contrastive_lift "$VALIDATION_METRICS")"
  audit_records="$(read_json_number records_read "$FULL_AUDIT_FILE")"
  audit_range_end="$(read_json_number effective_range_end "$FULL_AUDIT_FILE")"
  neural_bonus="$(awk -v lift="$validation_lift" 'BEGIN { if (lift < 0) lift = 0; if (lift > 0.30) lift = 0.30; printf "%.4f", lift * 20.0 }')"

  SOURCE_MATCH_RATE="$(awk -v matched="$matched" -v total="$target_count" 'BEGIN { if (total > 0) printf "%.4f", matched / total; else printf "0.0000" }')"
  EVIDENCE_SCORE="$(awk -v base="$base" -v source="$SOURCE_MATCH_RATE" -v neural="$neural_bonus" 'BEGIN { score = base + source * 10.0 + neural; if (score > 99) score = 99; printf "%.2f", score }')"
  EVIDENCE_DELTA="$(awk -v current="$EVIDENCE_SCORE" -v previous="$previous" 'BEGIN { printf "%.2f", current - previous }')"
  CYCLE_COUNT="$(awk -v previous="$previous_cycle" 'BEGIN { printf "%d", previous + 1 }')"
  mkdir -p "$(dirname "$HISTORY_FILE")"
  printf '{"timestamp":"%s","cycle_count":%s,"evidence_score":%s,"evidence_delta":%s,"confidence_level":"%s","full_audit_records":%s,"audit_range_end":%s,"source_match_rate":%s,"source_target_count":%s,"matched_source_targets":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CYCLE_COUNT" "$EVIDENCE_SCORE" "$EVIDENCE_DELTA" \
    "$(json_escape "${confidence:-pipeline-check}")" "${audit_records:-0}" "${audit_range_end:-0}" "$SOURCE_MATCH_RATE" "$target_count" "$matched" >> "$HISTORY_FILE"
}

download_required() {
  local url="$1"
  local dest="$2"
  curl -L -f -sS "$url" -o "$dest"
}

download_optional() {
  local url="$1"
  local dest="$2"
  curl -L -f -sS "$url" -o "$dest" || return 0
}

decompress_optional() {
  local url="$1"
  local dest="$2"
  local tmp="$dest.gz"
  if curl -L -f -sS "$url" -o "$tmp"; then
    gzip -dc "$tmp" > "$dest" || return 0
  fi
}

build_if_needed() {
  if [ -x "$BUILD_DIR/collatz_source_targets" ] &&
     [ -x "$BUILD_DIR/collatz_source_align" ] &&
     [ -x "$BUILD_DIR/collatz_hypothesis_analyze" ] &&
     [ -x "$BUILD_DIR/collatz_evidence_publish" ] &&
     [ -x "$BUILD_DIR/collatz_full_audit" ]; then
    return
  fi
  cmake -S . -B "$BUILD_DIR"
  if [ -n "${COLLATZ_BUILD_JOBS:-}" ]; then
    cmake --build "$BUILD_DIR" --parallel "$COLLATZ_BUILD_JOBS"
  else
    cmake --build "$BUILD_DIR" --parallel
  fi
}

regenerate_hypothesis_summary() {
  "$BUILD_DIR/collatz_hypothesis_analyze" \
    --insights "$INSIGHTS_FILE" \
    --stratified-metadata "$STRATIFIED_METADATA" \
    --contrastive-metrics "$CONTRASTIVE_METRICS" \
    --autoencoder-metrics "$AUTOENCODER_METRICS" \
    --gnn-metrics "$GNN_METRICS" \
    --validation-metrics "$VALIDATION_METRICS" \
    --full-audit "$FULL_AUDIT_FILE" \
    --source-alignment "$SOURCE_ALIGNMENT_DIR/source_alignment.json" \
    --output-dir "$HYPOTHESES_DIR"
}

publish_canonical_evidence() {
  local git_commit
  git_commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  "$BUILD_DIR/collatz_evidence_publish" \
    --full-audit "$FULL_AUDIT_FILE" \
    --stratified-metadata "$STRATIFIED_METADATA" \
    --topology-manifest "${COLLATZ_TOPOLOGY_MANIFEST:-data/generated/topology/embedding_topology.json}" \
    --validation-metrics "$VALIDATION_METRICS" \
    --ablation-report "${COLLATZ_ABLATION_REPORT:-data/generated/evidence_validation/ablation_report.csv}" \
    --source-alignment "$SOURCE_ALIGNMENT_DIR/source_alignment.json" \
    --runner-status "$STATUS_FILE" \
    --neural-status "$NEURAL_STATUS_FILE" \
    --active-feature-file "${COLLATZ_PUBLIC_FEATURE_LABEL:-data/generated/features.bin}" \
    --git-commit "$git_commit" \
    --output "$PUBLIC_EVIDENCE_FILE" \
    --hypothesis-summary-output "$HYPOTHESES_DIR/summary.json"
}

run_full_audit() {
  FULL_AUDIT_STATE="running"
  write_status "running" "full dataset audit" false "" "optional neural stage"
  "$BUILD_DIR/collatz_full_audit" \
    --input "$FEATURE_FILE" \
    --output "$FULL_AUDIT_FILE" \
    --range-bands "${COLLATZ_FULL_AUDIT_RANGE_BANDS:-16}" \
    --top-count "${COLLATZ_FULL_AUDIT_TOP_COUNT:-16}" || fail_cycle "full dataset audit" "$?"
  FULL_AUDIT_RECORDS="$(read_json_number records_read "$FULL_AUDIT_FILE")"
  FULL_AUDIT_STATE="complete"
}

gpu_available() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    GPU_STATE="not_available"
    GPU_FREE_MEMORY_MB=0
    GPU_COMPUTE_PROCESS_COUNT=0
    return 1
  fi
  local busy
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF { c++ } END { print c+0 }')"
  local free_mb
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 { print int($1) }')"
  GPU_COMPUTE_PROCESS_COUNT="${busy:-0}"
  GPU_FREE_MEMORY_MB="${free_mb:-0}"
  if [ "${free_mb:-0}" -lt "$GPU_MIN_FREE_MB" ]; then
    GPU_STATE="low_memory"
    return 1
  fi
  if [ "${busy:-0}" -gt 0 ] && [ "$GPU_ALLOW_SHARED" != "1" ]; then
    GPU_STATE="busy"
    return 1
  fi
  if [ "${busy:-0}" -gt 0 ]; then
    GPU_STATE="shared"
  else
    GPU_STATE="available"
  fi
  return 0
}

current_feature_records() {
  if [ -x "$BUILD_DIR/collatzctl" ] && [ -f "$FEATURE_FILE" ]; then
    "$BUILD_DIR/collatzctl" inspect-bin "$FEATURE_FILE" 2>/dev/null | awk '
      /"records"/ {
        value=$0
        sub(/^.*"records":/, "", value)
        sub(/[^0-9].*$/, "", value)
        print value
        found=1
      }
      END { if (found != 1) print 0 }
    '
    return
  fi
  read_json_number dataset_records_observed "$SCAN_METADATA"
}

run_cpu_crunch_if_configured() {
  if [ "$RUN_CPU_CRUNCH" != "1" ]; then
    CPU_CRUNCH_STATE="disabled"
    return
  fi
  local current_records target_end
  current_records="$(current_feature_records)"
  if [ -n "$CPU_CRUNCH_END" ]; then
    target_end="$CPU_CRUNCH_END"
  else
    target_end="$(awk -v current="${current_records:-0}" -v step="$CPU_CRUNCH_STEP" 'BEGIN { printf "%d", current + step }')"
  fi
  if [ "${target_end:-0}" -le "${current_records:-0}" ]; then
    CPU_CRUNCH_STATE="complete"
    CPU_TARGET_END="${target_end:-0}"
    return
  fi
  CPU_CRUNCH_STATE="running"
  CPU_THREADS="$CPU_CRUNCH_THREADS"
  CPU_TARGET_END="$target_end"
  write_status "running" "cpu crunch scan" false "" "optional neural stage"
  if [ -n "$CPU_CRUNCH_COMMAND" ]; then
    SCAN_START=1 \
    SCAN_END="$target_end" \
    SCAN_THREADS="$CPU_CRUNCH_THREADS" \
    SCAN_CHUNK_SIZE="$CPU_CRUNCH_CHUNK_SIZE" \
    bash -lc "$CPU_CRUNCH_COMMAND" || fail_cycle "cpu crunch scan" "$?"
  else
    "$BUILD_DIR/collatz_scan_cpu" \
      --start 1 \
      --end "$target_end" \
      --output "$FEATURE_FILE" \
      --progress "$PROGRESS_FILE" \
      --metadata "$SCAN_METADATA" \
      --chunk-size "$CPU_CRUNCH_CHUNK_SIZE" \
      --max-steps "$CPU_CRUNCH_MAX_STEPS" \
      --threads "$CPU_CRUNCH_THREADS" \
      --format bin \
      --resume \
      --mode-label cpu-crunch || fail_cycle "cpu crunch scan" "$?"
  fi
  CPU_CRUNCH_STATE="complete"
}

append_ledger() {
  local alignment_json="$SOURCE_ALIGNMENT_DIR/source_alignment.json"
  local target_count matched source_count status
  target_count="$(read_json_number target_count "$alignment_json")"
  matched="$(read_json_number matched_targets "$alignment_json")"
  source_count="$(read_json_number source_family_count "$alignment_json")"
  status="$(awk -F'"' '/"alignment_status"/ { print $4; exit }' "$alignment_json" 2>/dev/null || true)"
  mkdir -p "$(dirname "$LEDGER_FILE")"
  printf '{"timestamp":"%s","event":"evidence_cycle","alignment_status":"%s","source_target_count":%s,"matched_source_targets":%s,"source_family_count":%s,"public_safe":true}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(json_escape "${status:-missing}")" "$target_count" "$matched" "$source_count" >> "$LEDGER_FILE"
}

LAST_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EVIDENCE_SCORE="$(last_history_value evidence_score)"
SOURCE_MATCH_RATE="$(last_history_value source_match_rate)"
CYCLE_COUNT="$(last_history_value cycle_count)"
write_status "running" "starting" false "" "build tools"

stage "build tools"
build_if_needed || fail_cycle "build tools" "$?"

stage "fetch public source records"
mkdir -p "$IMPORT_DIR"
download_required "https://oeis.org/A006577/b006577.txt" "$IMPORT_DIR/b006577.txt" || fail_cycle "fetch public source records" "$?"
download_required "https://oeis.org/A006884/b006884.txt" "$IMPORT_DIR/b006884.txt" || fail_cycle "fetch public source records" "$?"
download_optional "https://www.ericr.nl/wondrous/pathrecs.html" "$IMPORT_DIR/roosendaal_pathrecs.html"
download_optional "https://www.ericr.nl/wondrous/delrecs.html" "$IMPORT_DIR/roosendaal_delrecs.html"
download_optional "https://pcbarina.fit.vutbr.cz/path-records.htm" "$IMPORT_DIR/barina_path_records.html"
decompress_optional "https://sweet.ua.pt/tos/3x%2B1/t0.txt.gz" "$IMPORT_DIR/oliveira_max_excursion.txt"
decompress_optional "https://sweet.ua.pt/tos/3x%2B1/t1.txt.gz" "$IMPORT_DIR/oliveira_stopping.txt"

stage "generate expanded source targets"
"$BUILD_DIR/collatz_source_targets" \
  --oeis-stopping "$IMPORT_DIR/b006577.txt" \
  --oeis-path-records "$IMPORT_DIR/b006884.txt" \
  --roosendaal-path-records "$IMPORT_DIR/roosendaal_pathrecs.html" \
  --roosendaal-delay-records "$IMPORT_DIR/roosendaal_delrecs.html" \
  --barina-path-records "$IMPORT_DIR/barina_path_records.html" \
  --oliveira-max-excursion-records "$IMPORT_DIR/oliveira_max_excursion.txt" \
  --oliveira-stopping-records "$IMPORT_DIR/oliveira_stopping.txt" \
  --output "$SOURCE_OUTPUT" \
  --metadata "$SOURCE_METADATA" \
  --max-n "$MAX_N" \
  --stopping-limit "${COLLATZ_SOURCE_STOPPING_LIMIT:-5000}" \
  --path-record-limit "${COLLATZ_SOURCE_PATH_RECORD_LIMIT:-100}" \
  --generic-record-limit "${COLLATZ_SOURCE_GENERIC_LIMIT:-250}" || fail_cycle "generate expanded source targets" "$?"

stage "align source targets"
"$BUILD_DIR/collatz_source_align" \
  --projection "$PROJECTION_FILE" \
  --source-samples "$SOURCE_OUTPUT" \
  --feature-bin "$FEATURE_FILE" \
  --output-dir "$SOURCE_ALIGNMENT_DIR" \
  --neighbors "${COLLATZ_SOURCE_ALIGNMENT_NEIGHBORS:-8}" || fail_cycle "align source targets" "$?"

stage "regenerate hypothesis summary"
regenerate_hypothesis_summary || fail_cycle "regenerate hypothesis summary" "$?"

stage "optional cpu crunch scan"
run_cpu_crunch_if_configured

stage "full dataset audit"
run_full_audit

stage "optional neural stage"
if [ "$RUN_NEURAL" = "1" ] && [ -n "$NEURAL_COMMAND" ]; then
  if gpu_available; then
    NEURAL_STAGE="running"
    write_status "running" "optional neural stage" false "" "run public privacy scan"
    bash -lc "$NEURAL_COMMAND" || fail_cycle "optional neural stage" "$?"
    NEURAL_STAGE="complete"
  else
    NEURAL_STAGE="skipped_${GPU_STATE}"
    echo "neural stage skipped: gpu unavailable, low memory, or busy"
  fi
else
  NEURAL_STAGE="disabled"
  echo "neural stage skipped: not configured"
fi

stage "refresh post-crunch hypothesis summary"
regenerate_hypothesis_summary || fail_cycle "refresh post-crunch hypothesis summary" "$?"

stage "publish canonical evidence summary"
publish_canonical_evidence || fail_cycle "publish canonical evidence summary" "$?"

stage "regenerate public visual artifacts"
if [ -x tools/render_dashboard_summary.py ]; then
  tools/render_dashboard_summary.py \
    --input "$PUBLIC_EVIDENCE_FILE" \
    --output docs/media/dashboard-summary.svg || fail_cycle "regenerate public visual artifacts" "$?"
fi
if [ -x tools/render_evidence_history.py ]; then
  echo "historical evidence graph waits until the current cycle is appended"
fi

stage "run public privacy scan"
if [ -x ops/privacy-scan.sh ]; then
  ops/privacy-scan.sh || fail_cycle "run public privacy scan" "$?"
else
  echo "privacy scan skipped: script missing"
fi

stage "append sanitized ledger"
update_evidence_score || fail_cycle "append sanitized ledger" "$?"
append_ledger || fail_cycle "append sanitized ledger" "$?"

stage "regenerate historical evidence graph"
if [ -x tools/render_evidence_history.py ]; then
  tools/render_evidence_history.py \
    --history "$HISTORY_FILE" \
    --evidence "$PUBLIC_EVIDENCE_FILE" \
    --output docs/media/evidence-history.svg || fail_cycle "regenerate historical evidence graph" "$?"
fi

TARGET_COUNT="$(read_json_number target_count "$SOURCE_ALIGNMENT_DIR/source_alignment.json")"
MATCHED_COUNT="$(read_json_number matched_targets "$SOURCE_ALIGNMENT_DIR/source_alignment.json")"
write_status "idle" "complete" true "" "next scheduled evidence cycle" "$TARGET_COUNT" "$MATCHED_COUNT"
echo "evidence cycle complete: matched $MATCHED_COUNT of $TARGET_COUNT source targets"
