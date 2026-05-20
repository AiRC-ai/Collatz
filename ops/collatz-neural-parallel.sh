#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      echo "usage: ops/collatz-neural-parallel.sh [--dry-run]"
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
cd "$REPO_DIR"

STATUS_FILE="${COLLATZ_NEURAL_STATUS:-data/generated/runner/neural_parallel_status.json}"
LOG_DIR="${COLLATZ_NEURAL_LOG_DIR:-logs/neural-parallel}"
COMPOSE_OVERRIDE="${COLLATZ_COMPOSE_OVERRIDE:-}"
MAX_PARALLEL="${COLLATZ_PARALLEL_NEURAL_JOBS:-6}"
FEATURE_SETS="${COLLATZ_PARALLEL_FEATURE_SETS:-hybrid metrics shape parity-sequence residue-sequence}"
RUN_AUTOENCODER="${COLLATZ_PARALLEL_AUTOENCODER:-1}"
RUN_IMAGE_CONTRASTIVE="${COLLATZ_PARALLEL_IMAGE_CONTRASTIVE:-0}"
RUN_EVIDENCE="${COLLATZ_PARALLEL_EVIDENCE_VALIDATE:-1}"
RUN_STRATIFIED="${COLLATZ_PARALLEL_PREPARE_SAMPLE:-1}"
CONTRASTIVE_SERVICE="${COLLATZ_CONTRASTIVE_SERVICE:-contrastive-v2}"

mkdir -p "$(dirname "$STATUS_FILE")" "$LOG_DIR"

COMPOSE=(docker compose --project-directory "$REPO_DIR" -f "$REPO_DIR/compose.yaml")
if [ -n "$COMPOSE_OVERRIDE" ]; then
  COMPOSE+=(-f "$COMPOSE_OVERRIDE")
fi

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\r//g; s/\n/ /g'
}

gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,power.draw \
      --format=csv,noheader,nounits 2>/dev/null | awk -F, 'NR==1 {
        gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4);
        printf "%s %s %s %s", $1, $2, $3, $4
      }'
  else
    printf '0 0 0 0'
  fi
}

declare -a JOB_NAMES=()
declare -a JOB_TYPES=()
declare -a JOB_FEATURES=()
declare -a JOB_PIDS=()
declare -a JOB_STATES=()
declare -a JOB_CODES=()

write_status() {
  local state="$1"
  local stage="$2"
  local started="${STARTED_UTC:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
  local updated
  updated="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local active=0 complete=0 failed=0
  for item in "${JOB_STATES[@]:-}"; do
    case "$item" in
      running) active=$((active + 1)) ;;
      complete) complete=$((complete + 1)) ;;
      failed) failed=$((failed + 1)) ;;
    esac
  done
  read -r gpu_util gpu_used gpu_free gpu_power <<<"$(gpu_snapshot)"
  {
    printf '{\n'
    printf '  "state": "%s",\n' "$(json_escape "$state")"
    printf '  "current_stage": "%s",\n' "$(json_escape "$stage")"
    printf '  "started_utc": "%s",\n' "$(json_escape "$started")"
    printf '  "updated_utc": "%s",\n' "$(json_escape "$updated")"
    printf '  "concurrency_limit": %s,\n' "$MAX_PARALLEL"
    printf '  "active_jobs": %s,\n' "$active"
    printf '  "complete_jobs": %s,\n' "$complete"
    printf '  "failed_jobs": %s,\n' "$failed"
    printf '  "gpu_utilization_percent": %s,\n' "${gpu_util:-0}"
    printf '  "gpu_memory_used_mb": %s,\n' "${gpu_used:-0}"
    printf '  "gpu_memory_free_mb": %s,\n' "${gpu_free:-0}"
    printf '  "gpu_power_watts": %s,\n' "${gpu_power:-0}"
    printf '  "jobs": [\n'
    for index in "${!JOB_NAMES[@]}"; do
      if [ "$index" -gt 0 ]; then
        printf ',\n'
      fi
      printf '    {"name":"%s","kind":"%s","feature_set":"%s","state":"%s","exit_code":%s}' \
        "$(json_escape "${JOB_NAMES[$index]}")" \
        "$(json_escape "${JOB_TYPES[$index]}")" \
        "$(json_escape "${JOB_FEATURES[$index]}")" \
        "$(json_escape "${JOB_STATES[$index]}")" \
        "${JOB_CODES[$index]}"
    done
    printf '\n  ]\n'
    printf '}\n'
  } > "$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

mark_finished_jobs() {
  local running=0
  local live_pids
  live_pids=" $(jobs -pr) "
  for index in "${!JOB_PIDS[@]}"; do
    if [ "${JOB_STATES[$index]}" != "running" ]; then
      continue
    fi
    if [[ "$live_pids" == *" ${JOB_PIDS[$index]} "* ]]; then
      running=$((running + 1))
      continue
    fi
    if wait "${JOB_PIDS[$index]}"; then
      JOB_STATES[$index]="complete"
      JOB_CODES[$index]=0
    else
      local code=$?
      JOB_STATES[$index]="failed"
      JOB_CODES[$index]=$code
    fi
  done
  echo "$running"
}

running_count() {
  local count=0
  for state in "${JOB_STATES[@]:-}"; do
    if [ "$state" = "running" ]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

wait_for_slot() {
  while [ "$(running_count)" -ge "$MAX_PARALLEL" ]; do
    mark_finished_jobs >/dev/null
    write_status "running" "training parallel neural jobs"
    sleep 5
  done
}

launch_contrastive() {
  local feature="$1"
  local name="contrastive_${feature}"
  local output="/work/data/generated/contrastive_${feature}"
  if [ "$feature" = "hybrid" ]; then
    output="/work/data/generated/contrastive"
  fi
  wait_for_slot
  (
    export CONTRASTIVE_FEATURE_SET="$feature"
    export CONTRASTIVE_OUTPUT_DIR="$output"
    export CONTRASTIVE_LIMIT="${CONTRASTIVE_LIMIT:-0}"
    export CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-160}"
    export CONTRASTIVE_BATCH_SIZE="${CONTRASTIVE_BATCH_SIZE:-4096}"
    export CONTRASTIVE_HIDDEN_DIMS="${CONTRASTIVE_HIDDEN_DIMS:-512}"
    export CONTRASTIVE_EMBEDDING_DIMS="${CONTRASTIVE_EMBEDDING_DIMS:-128}"
    export CONTRASTIVE_PAIR_MODE="${CONTRASTIVE_PAIR_MODE:-family_pairs}"
    export CONTRASTIVE_EVAL_CHUNK="${CONTRASTIVE_EVAL_CHUNK:-2048}"
    export CONTRASTIVE_EVAL_DEVICE="${CONTRASTIVE_EVAL_DEVICE:-auto}"
    "${COMPOSE[@]}" --profile neural run --rm "$CONTRASTIVE_SERVICE"
  ) > "$LOG_DIR/$name.log" 2>&1 &
  JOB_NAMES+=("$name")
  JOB_TYPES+=("contrastive")
  JOB_FEATURES+=("$feature")
  JOB_PIDS+=("$!")
  JOB_STATES+=("running")
  JOB_CODES+=("null")
  write_status "running" "training parallel neural jobs"
}

launch_image_contrastive() {
  wait_for_slot
  (
    export IMAGE_CONTRASTIVE_EPOCHS="${IMAGE_CONTRASTIVE_EPOCHS:-80}"
    export IMAGE_CONTRASTIVE_BATCH_SIZE="${IMAGE_CONTRASTIVE_BATCH_SIZE:-1024}"
    export IMAGE_CONTRASTIVE_EMBEDDING_DIMS="${IMAGE_CONTRASTIVE_EMBEDDING_DIMS:-128}"
    "${COMPOSE[@]}" --profile neural run --rm image-contrastive
  ) > "$LOG_DIR/image-contrastive.log" 2>&1 &
  JOB_NAMES+=("image_contrastive")
  JOB_TYPES+=("image-contrastive")
  JOB_FEATURES+=("image-only")
  JOB_PIDS+=("$!")
  JOB_STATES+=("running")
  JOB_CODES+=("null")
  write_status "running" "training parallel neural jobs"
}

launch_autoencoder() {
  wait_for_slot
  (
    export AUTOENCODER_OUTPUT_DIR="${AUTOENCODER_OUTPUT_DIR:-/work/data/generated/anomalies}"
    export AUTOENCODER_LIMIT="${AUTOENCODER_LIMIT:-100000}"
    export AUTOENCODER_EPOCHS="${AUTOENCODER_EPOCHS:-160}"
    export AUTOENCODER_BATCH_SIZE="${AUTOENCODER_BATCH_SIZE:-4096}"
    export AUTOENCODER_HIDDEN_DIMS="${AUTOENCODER_HIDDEN_DIMS:-384}"
    export AUTOENCODER_LATENT_DIMS="${AUTOENCODER_LATENT_DIMS:-48}"
    export AUTOENCODER_TOP_ANOMALIES="${AUTOENCODER_TOP_ANOMALIES:-512}"
    "${COMPOSE[@]}" --profile neural run --rm autoencoder
  ) > "$LOG_DIR/autoencoder.log" 2>&1 &
  JOB_NAMES+=("autoencoder")
  JOB_TYPES+=("autoencoder")
  JOB_FEATURES+=("metrics")
  JOB_PIDS+=("$!")
  JOB_STATES+=("running")
  JOB_CODES+=("null")
  write_status "running" "training parallel neural jobs"
}

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
write_status "running" "preparing stratified tensors"

if [ "$DRY_RUN" -eq 1 ]; then
  for feature in $FEATURE_SETS; do
    JOB_NAMES+=("contrastive_$feature")
    JOB_TYPES+=("contrastive")
    JOB_FEATURES+=("$feature")
    JOB_PIDS+=(0)
    JOB_STATES+=("complete")
    JOB_CODES+=(0)
  done
  if [ "$RUN_AUTOENCODER" = "1" ]; then
    JOB_NAMES+=("autoencoder")
    JOB_TYPES+=("autoencoder")
    JOB_FEATURES+=("metrics")
    JOB_PIDS+=(0)
    JOB_STATES+=("complete")
    JOB_CODES+=(0)
  fi
  if [ "$RUN_IMAGE_CONTRASTIVE" = "1" ]; then
    JOB_NAMES+=("image_contrastive")
    JOB_TYPES+=("image-contrastive")
    JOB_FEATURES+=("image-only")
    JOB_PIDS+=(0)
    JOB_STATES+=("complete")
    JOB_CODES+=(0)
  fi
  write_status "complete" "dry-run parallel neural evidence"
  echo "dry-run parallel neural evidence complete"
  exit 0
fi

if [ "$RUN_STRATIFIED" = "1" ]; then
  export STRATIFIED_RANDOM_COUNT="${STRATIFIED_RANDOM_COUNT:-100000}"
  export STRATIFIED_GLOBAL_TOP="${STRATIFIED_GLOBAL_TOP:-2048}"
  export STRATIFIED_RESIDUE_TOP="${STRATIFIED_RESIDUE_TOP:-128}"
  export STRATIFIED_RANGE_TOP="${STRATIFIED_RANGE_TOP:-128}"
  export STRATIFIED_RANGE_BANDS="${STRATIFIED_RANGE_BANDS:-16}"
  export EMBED_SKETCH_DIMS="${EMBED_SKETCH_DIMS:-128}"
  "${COMPOSE[@]}" --profile research run --rm stratified > "$LOG_DIR/stratified.log" 2>&1
  "${COMPOSE[@]}" --profile research run --rm stratified-embedder > "$LOG_DIR/stratified-embedder.log" 2>&1
  "${COMPOSE[@]}" --profile research run --rm family-labels > "$LOG_DIR/family-labels.log" 2>&1
  "${COMPOSE[@]}" --profile research run --rm pair-sampler > "$LOG_DIR/pair-sampler.log" 2>&1
  if [ "$RUN_IMAGE_CONTRASTIVE" = "1" ]; then
    "${COMPOSE[@]}" --profile research run --rm image-tensors > "$LOG_DIR/image-tensors.log" 2>&1
  fi
fi

for feature in $FEATURE_SETS; do
  launch_contrastive "$feature"
done
if [ "$RUN_AUTOENCODER" = "1" ]; then
  launch_autoencoder
fi
if [ "$RUN_IMAGE_CONTRASTIVE" = "1" ]; then
  launch_image_contrastive
fi

while [ "$(running_count)" -gt 0 ]; do
  mark_finished_jobs >/dev/null
  write_status "running" "training parallel neural jobs"
  sleep 5
done
mark_finished_jobs >/dev/null

failures=0
for state in "${JOB_STATES[@]:-}"; do
  if [ "$state" = "failed" ]; then
    failures=$((failures + 1))
  fi
done
if [ "$failures" -gt 0 ]; then
  write_status "error" "parallel neural jobs failed"
  exit 1
fi

if [ "$RUN_EVIDENCE" = "1" ]; then
  write_status "running" "validating neural evidence"
  export EVIDENCE_EVAL_CHUNK="${EVIDENCE_EVAL_CHUNK:-2048}"
  export EVIDENCE_EVAL_DEVICE="${EVIDENCE_EVAL_DEVICE:-auto}"
  "${COMPOSE[@]}" --profile research run --rm evidence-validation > "$LOG_DIR/evidence-validation.log" 2>&1
fi

write_status "complete" "parallel neural evidence complete"
echo "parallel neural evidence complete"
