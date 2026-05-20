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
  echo "dry-run stage: optionally run one neural stage if GPU is available and configured"
  echo "dry-run stage: run public privacy scan"
  echo "dry-run stage: append sanitized iteration ledger"
  echo "dry-run stage: write public-safe runner status"
  exit 0
fi

cd "$REPO_DIR"

RUNNER_DIR="${COLLATZ_RUNNER_DIR:-data/generated/runner}"
STATUS_FILE="${COLLATZ_RUNNER_STATUS:-$RUNNER_DIR/status.json}"
LOCK_DIR="${COLLATZ_RUNNER_LOCK:-$RUNNER_DIR/lock}"
BUILD_DIR="${COLLATZ_BUILD_DIR:-build}"
IMPORT_DIR="${COLLATZ_IMPORT_DIR:-data/imported}"
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
LEDGER_FILE="${COLLATZ_LEDGER_FILE:-logs/iteration-ledger.jsonl}"
MAX_N="${COLLATZ_SOURCE_MAX_N:-100000000}"
GPU_MIN_FREE_MB="${COLLATZ_GPU_MIN_FREE_MB:-4096}"
RUN_NEURAL="${COLLATZ_RUN_NEURAL:-0}"
NEURAL_COMMAND="${COLLATZ_NEURAL_COMMAND:-}"

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
    printf '  "active_experiment": "source-record neighborhood alignment",\n'
    printf '  "source_target_count": %s,\n' "$source_target_count"
    printf '  "matched_source_targets": %s,\n' "$matched_source_targets"
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
     [ -x "$BUILD_DIR/collatz_hypothesis_analyze" ]; then
    return
  fi
  cmake -S . -B "$BUILD_DIR"
  if [ -n "${COLLATZ_BUILD_JOBS:-}" ]; then
    cmake --build "$BUILD_DIR" --parallel "$COLLATZ_BUILD_JOBS"
  else
    cmake --build "$BUILD_DIR" --parallel
  fi
}

gpu_available() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local busy
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF { c++ } END { print c+0 }')"
  local free_mb
  free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 { print int($1) }')"
  [ "${busy:-1}" -eq 0 ] && [ "${free_mb:-0}" -ge "$GPU_MIN_FREE_MB" ]
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
  --output-dir "$SOURCE_ALIGNMENT_DIR" \
  --neighbors "${COLLATZ_SOURCE_ALIGNMENT_NEIGHBORS:-8}" || fail_cycle "align source targets" "$?"

stage "regenerate hypothesis summary"
"$BUILD_DIR/collatz_hypothesis_analyze" \
  --insights "$INSIGHTS_FILE" \
  --stratified-metadata "$STRATIFIED_METADATA" \
  --contrastive-metrics "$CONTRASTIVE_METRICS" \
  --autoencoder-metrics "$AUTOENCODER_METRICS" \
  --gnn-metrics "$GNN_METRICS" \
  --validation-metrics "$VALIDATION_METRICS" \
  --source-alignment "$SOURCE_ALIGNMENT_DIR/source_alignment.json" \
  --output-dir "$HYPOTHESES_DIR" || fail_cycle "regenerate hypothesis summary" "$?"

stage "optional neural stage"
if [ "$RUN_NEURAL" = "1" ] && [ -n "$NEURAL_COMMAND" ]; then
  if gpu_available; then
    bash -lc "$NEURAL_COMMAND" || fail_cycle "optional neural stage" "$?"
  else
    echo "neural stage skipped: gpu unavailable or busy"
  fi
else
  echo "neural stage skipped: not configured"
fi

stage "run public privacy scan"
if [ -x ops/privacy-scan.sh ]; then
  ops/privacy-scan.sh || fail_cycle "run public privacy scan" "$?"
else
  echo "privacy scan skipped: script missing"
fi

stage "append sanitized ledger"
append_ledger || fail_cycle "append sanitized ledger" "$?"

TARGET_COUNT="$(read_json_number target_count "$SOURCE_ALIGNMENT_DIR/source_alignment.json")"
MATCHED_COUNT="$(read_json_number matched_targets "$SOURCE_ALIGNMENT_DIR/source_alignment.json")"
write_status "idle" "complete" true "" "next scheduled evidence cycle" "$TARGET_COUNT" "$MATCHED_COUNT"
echo "evidence cycle complete: matched $MATCHED_COUNT of $TARGET_COUNT source targets"
