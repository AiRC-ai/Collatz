# 3xN1 Collatz Research System

This repository is evolving from a single Collatz plotter into a C++20/CUDA
research system for generating validated Collatz path features and looking for
learnable path patterns.

For a positive whole number `n`:

```text
if n is even: n = n / 2
if n is odd:  n = 3n + 1
repeat until n = 1
```

The new performance path is C++20 with fixed-width integer fast paths. Python is
not used for classical Collatz scanning, dataset generation, progress serving,
or validation.

## Current Evidence Snapshot

![Current Collatz evidence dashboard snapshot](docs/media/dashboard-summary.svg)

![Historical Collatz evidence trend](docs/media/evidence-history.svg)

The block below is generated from
`data/generated/evidence/latest_public_summary.json`. Do not hand-edit evidence
numbers in the README.

![v3 Ablation Chart](docs/media/v3-results-chart.svg)

![Legacy Baseline Chart](docs/media/legacy-results-chart.svg)

![v4 Ablation Chart](docs/media/v4-ablation-chart.svg)

![v5 Ablation Chart](docs/media/v5-ablation-chart.svg)

![Loss Curves](docs/media/loss-curves.svg)

<!-- BEGIN GENERATED EVIDENCE SNAPSHOT -->
- Confidence: `source-neighborhood-supported`
- Meaning: Public validation starts agree with the learned topology-neighborhood gate. v3 showed rich representations with family pairs achieve 42.1% lift. v4 proved model-mined hard negatives fail — the embedding starts bad and mines wrong negatives. v5 pivots to mining negatives from raw metrics space (84% lift signal) to fix this.
- Audit: `1.20B` rows over `1..1,200,000,000`; full audit completed: `true`.
- Coverage: topology `114.0K` rows (`0.01%` of audit); stratified evidence sample `110.1K` rows (`0.01%`).
- Neural result: `50.0K` sample rows; GPU used: `true`; parallel jobs completed: `0`.
- Learned lift: weakest range `0.53%`, fold minimum `0.42%`, numeric-adjacency lift `1.29%`.
- Best current ablation: **`v5 hybrid, family_pairs, raw_metrics_negatives, 200ep`** (currently training: epoch 10/200, lift=0.00%, loss=0.292).
- Interpretation: `metric-dominant signal` — metrics-only lift exceeds hybrid lift under the current evidence run (healthy negative control). v4 proved model-mined hard negatives fail early in training; v5 is mining negatives from raw metrics space which encodes the family signal. v5 at epoch 10/200: loss decreasing (0.336 → 0.292), lift not yet emerged.
- Healthy negative control: `metrics-only` beats `hybrid` (`84.12%` vs `83.34%`), so richer neural structure has not yet earned promotion beyond simpler trajectory metrics.
- v3 results: **V1 encoder + family pairs achieves 83.34% lift** — the strongest ablation so far. Multi-branch encoding peaks at 23.43%, confirming self-noise from parallel branches degrades signal.
- v4 insight: **Hard negative mining from model embeddings failed entirely** (lift stayed at 0.00% over 128 epochs). Loss decreased but embedding did not learn family structure — mining from bad early embeddings created a feedback loop of wrong negatives.
- v5 progress: epoch 10/200, loss 0.292 (start: 0.336), lift 0.00%, best lift 0.0001. Next evaluation at epoch 20.
- Promotion blockers: `none`.
- Source alignment: `5,186 / 5,191` matched; unknown unmatched rows `0`.
- Next experiment: `v5: raw-metrics hard negative mining. Mining from raw metrics (84% lift signal) instead of model embeddings. Target lift > 42%. Currently at epoch 10/200, lift=0.00%, loss=0.292 (down from 0.336). Next evaluation at epoch 20.`.
- This is empirical evidence, not a Collatz proof.
<!-- END GENERATED EVIDENCE SNAPSHOT -->

The dashboard is intentionally compact: it should answer what the AI currently
believes, how confident it is, what evidence supports that, what limits the
claim, and what experiment should falsify or strengthen it next.

Longer operating notes live in:

- [Evidence contract](docs/EVIDENCE.md)
- [Source alignment](docs/SOURCE_ALIGNMENT.md)
- [Neural pipeline](docs/NEURAL_PIPELINE.md)
- [Dashboard](docs/DASHBOARD.md)
- [Limitations](docs/LIMITATIONS.md)
- [Runner](docs/RUNNER.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Build notes](docs/BUILD.md)

## Visual Research Artifacts

The project keeps the first screen focused on the usable research state:
trajectory plots, path-as-image encodings, topology maps, GNN graph previews,
and source-alignment summaries.

![Collatz path plot for n=27](collatz_27.svg)

The path-as-image route turns trajectories into visual tensors for clustering,
autoencoders, contrastive learning, and anomaly review.

![Real Collatz path-image tensor atlas generated from n=27](docs/media/path-image-atlas.svg)

This atlas is not a mockup. It is generated from the actual C++ path-image
encoder for `n=27`; every pixel comes from the computed trajectory and the same
recurrence/GAF/MTF/parity/residue transforms used by `collatz_path_image`.

## Build

```sh
make
```

This builds:

- `build/collatz_plot`
- `build/collatz_scan_cpu`
- `build/collatz_validate_sources`
- `build/collatz_web`
- `build/collatzctl`
- `build/collatz_train`
- `build/collatz_embed_export`
- `build/collatz_path_image`
- `build/collatz_path_image_atlas`
- `build/collatz_select_representatives`
- `build/collatz_stratified_sample`
- `build/collatz_full_audit`
- `build/collatz_graph_export`
- `build/collatz_embedding_analyze`
- `build/collatz_neighborhood_analyze`
- `build/collatz_selftest`
- `build/collatz_insight_analyze`
- `build/collatz_hypothesis_analyze`
- `build/collatz_source_targets`
- `build/collatz_source_align`
- `build/collatz_evidence_publish`

Run validation:

```sh
make test
```

Build an expanded public source-validation target table from OEIS and public
record-holder files:

```sh
mkdir -p data/imported
curl -L -o data/imported/b006577.txt https://oeis.org/A006577/b006577.txt
curl -L -o data/imported/b006884.txt https://oeis.org/A006884/b006884.txt
curl -L -o data/imported/roosendaal_pathrecs.html https://www.ericr.nl/wondrous/pathrecs.html
curl -L -o data/imported/roosendaal_delrecs.html https://www.ericr.nl/wondrous/delrecs.html
curl -L -o data/imported/barina_path_records.html https://pcbarina.fit.vutbr.cz/path-records.htm
curl -L -o data/imported/oliveira_max_excursion.txt.gz https://sweet.ua.pt/tos/3x%2B1/t0.txt.gz
curl -L -o data/imported/oliveira_stopping.txt.gz https://sweet.ua.pt/tos/3x%2B1/t1.txt.gz
gzip -dk data/imported/oliveira_max_excursion.txt.gz
gzip -dk data/imported/oliveira_stopping.txt.gz
./build/collatz_source_targets \
  --oeis-stopping data/imported/b006577.txt \
  --oeis-path-records data/imported/b006884.txt \
  --roosendaal-path-records data/imported/roosendaal_pathrecs.html \
  --roosendaal-delay-records data/imported/roosendaal_delrecs.html \
  --barina-path-records data/imported/barina_path_records.html \
  --oliveira-max-excursion-records data/imported/oliveira_max_excursion.txt \
  --oliveira-stopping-records data/imported/oliveira_stopping.txt \
  --output data/generated/source_validation/public_source_targets.csv \
  --max-n 100000000 \
  --stopping-limit 5000 \
  --path-record-limit 100 \
  --generic-record-limit 250
```

The first four CSV columns remain compatible with the source-alignment tool:
`source,n,total_steps,peak_low`. Extra provenance columns record source kind,
rank, URL, retrieval timestamp, and parser. Starts above the active scan range
are counted as future source targets instead of being used for dashboard
confidence.

## Run

```sh
./build/collatz_plot 7
./build/collatz_plot 27 --width 120 --height 30 --csv data/generated/path_27.csv
./build/collatzctl sample 27
./build/collatzctl inspect-bin data/generated/features_1_100m.bin
```

If a binary file was resumed from a smaller smoke range, repair the header
without rewriting the records:

```sh
./build/collatzctl patch-bin-header data/generated/features_1_100m.bin --range-end 100000000
```

Run a CPU scan:

```sh
./build/collatz_scan_cpu \
  --start 1 \
  --end 1000000 \
  --output data/generated/features.csv \
  --progress logs/progress.jsonl \
  --chunk-size 100000 \
  --threads 8 \
  --resume
```

For large dataset generation, prefer the compact binary format:

```sh
./build/collatz_scan_cpu \
  --start 1 \
  --end 100000000 \
  --output data/generated/features_1_100m.bin \
  --progress logs/progress.jsonl \
  --chunk-size 100000 \
  --threads 16 \
  --format bin \
  --resume
```

The scanner writes a metadata sidecar by default at `OUTPUT.metadata.json`.
Use `--metadata FILE` to choose a specific sidecar path.

Export embedding-ready research inputs from a binary feature file:

```sh
./build/collatz_embed_export \
  --input data/generated/features_1_100m.bin \
  --output-dir data/generated/ml_1_100m \
  --limit 1000000 \
  --sketch-dims 128
```

This writes metric vectors, parity-run tokens, residue-transition streams,
fixed-length log-path sketches, and metadata under the output directory.
Use `--sample-file data/generated/stratified/samples.csv` to export tensors for
an evidence sample that spans the full binary scan instead of only the first
`--limit` records.

Generate deterministic path-image artifacts for a single start value:

```sh
./build/collatz_path_image \
  --n 27 \
  --output-dir data/generated/images_27 \
  --size 64
```

Or render representatives from a binary feature file:

```sh
./build/collatz_path_image \
  --input data/generated/features_1_100m.bin \
  --count 16 \
  --output-dir data/generated/images_sample \
  --size 64
```

The image utility writes recurrence plots, Gramian Angular Fields, Markov
Transition Fields, parity rasters, residue rasters, and a `manifest.json`.

Regenerate the README path-image atlas from the real encoder:

```sh
./build/collatz_path_image_atlas \
  --n 27 \
  --output docs/media/path-image-atlas.svg \
  --size 32
```

The committed atlas is deliberately generated from a specific trajectory rather
than drawn by hand, so it is reproducible and auditable.

Regenerate the README dashboard image from the canonical public evidence JSON:

```sh
python3 tools/render_dashboard_summary.py \
  --input data/generated/evidence/latest_public_summary.json \
  --output docs/media/dashboard-summary.svg
```

Regenerate the historical evidence graph from sanitized runner history:

```sh
python3 tools/render_evidence_history.py \
  --history data/generated/runner/history.jsonl \
  --evidence data/generated/evidence/latest_public_summary.json \
  --output docs/media/evidence-history.svg
```

The committed dashboard SVG is a static public summary. It is rendered from the
same canonical evidence file as the README snapshot, and it intentionally omits
live runtime and infrastructure details.

Select representative starts for research review:

```sh
./build/collatz_select_representatives \
  --input data/generated/features_1_100m.bin \
  --output data/generated/representatives.csv
```

Build an evidence-first stratified sample over the full binary scan:

```sh
./build/collatz_stratified_sample \
  --input data/generated/features_1_100m.bin \
  --output-dir data/generated/stratified \
  --clusters data/generated/topology/clusters.csv \
  --representatives data/generated/representatives.csv \
  --starts data/generated/graphs/starts.csv \
  --source-targets data/generated/source_validation/public_source_targets.csv
```

This writes `samples.csv` and `metadata.json`. Every selected start has one or
more selection reasons: random baseline, range band, residue bucket, record-like
ladder, high stopping time, high peak behavior, topology representative, or GNN
representative. Public source targets are included explicitly so source
alignment is measured against real projection rows rather than a missing-sample
artifact.

Export a Graph Neural Network-ready trajectory graph:

```sh
./build/collatz_graph_export \
  --starts-file data/generated/representatives.csv \
  --output-dir data/generated/graphs \
  --count 128
```

This writes `nodes.csv`, `edges.csv`, `starts.csv`, and
`trajectory_graph.json`. The dashboard renders the graph preview and exposes
node/edge counts for GNN dataset tracking.

Generate a deterministic baseline topology map from metric embeddings:

```sh
./build/collatz_embedding_analyze \
  --input data/generated/ml_stratified/metrics_safe.csv \
  --output-dir data/generated/topology \
  --clusters 16
```

This writes a 2D PCA-style projection, k-means cluster summaries, and
`embedding_topology.json` for the dashboard. The public topology should be built
from the source-covered safe-metric export so evidence alignment, README claims,
and dashboard claims stay tied to the same public-safe sample.

Build nearest-neighbor neighborhoods around each topology cluster
representative:

```sh
./build/collatz_neighborhood_analyze \
  --projection data/generated/topology/projection.csv \
  --clusters data/generated/topology/clusters.csv \
  --output data/generated/topology/neighborhoods.json \
  --neighbors 12
```

This writes compact cluster-representative neighborhoods for family inspection
and dashboard tracking.

Generate the first structured analysis layer over the topology:

```sh
./build/collatz_insight_analyze \
  --projection data/generated/topology/projection.csv \
  --clusters data/generated/topology/clusters.csv \
  --scan-metadata data/generated/features.bin.metadata.json \
  --output-dir data/generated/insights
```

This writes `insights.json` and `insights.md`: a plain-language conclusion,
what the topology means, current limits, scoped findings, loose/tight family
candidates, and next experiments. It is the dashboard's first repeatable
"AI analyst" layer before deeper neural models.

Prepare the path-family learning inputs:

```sh
docker compose --profile research run --rm stratified-embedder
docker compose --profile research run --rm family-labels
docker compose --profile research run --rm pair-sampler
```

This writes safe scalar metrics, deterministic trajectory-family labels, and
matched positive/hard-negative pairs. The safe metric export excludes
integrity/checksum fields so `metrics-only` remains a trustworthy baseline.

Train the path-family contrastive v2 model on a CUDA-capable GPU:

```sh
docker compose --profile neural run --rm contrastive-v2
```

This writes `data/generated/contrastive/embeddings.csv`, `encoder.pt`, and
`metrics.json`. The key question is whether hybrid path-family retrieval beats
the safe `metrics-only` baseline under matched controls.

Run feature-family ablations when a learned signal looks promising:

```sh
CONTRASTIVE_FEATURE_SET=metrics CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_metrics docker compose --profile neural run --rm contrastive-v2
CONTRASTIVE_FEATURE_SET=shape CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_shape docker compose --profile neural run --rm contrastive-v2
CONTRASTIVE_FEATURE_SET=parity-sequence CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_parity-sequence docker compose --profile neural run --rm contrastive-v2
CONTRASTIVE_FEATURE_SET=residue-sequence CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_residue-sequence docker compose --profile neural run --rm contrastive-v2
```

Train the first autoencoder anomaly model:

```sh
docker compose --profile neural run --rm autoencoder
```

This writes `data/generated/anomalies/anomalies.csv`, `autoencoder.pt`, and
`metrics.json`. High reconstruction-error starts are candidates for full-path
and path-image review.

Validate the learned evidence before promoting the dashboard conclusion:

```sh
docker compose --profile research run --rm evidence-validation
```

This writes `data/generated/evidence_validation/metrics.json`,
`holdouts.csv`, and `ablation_report.csv`. It compares learned neighborhoods
against numeric adjacency, range holdouts, residue holdouts, random folds, raw
feature baselines, learned feature-family ablations, and retrieval metrics such
as tail-family recall, source-record recall, MRR, NDCG, ARI, and NMI.

Compare public source-validation starts against topology neighborhoods:

```sh
./build/collatz_source_align \
  --projection data/generated/topology/projection.csv \
  --source-samples data/generated/source_validation/public_source_targets.csv \
  --feature-bin data/generated/features.bin \
  --output-dir data/generated/source_alignment
```

This writes `source_alignment.json`, `source_targets.csv`, and
`unmatched_rows.csv`. Source targets are deduped by source family, source kind,
and start value before public alignment; duplicate provenance is kept in the
source-target metadata. Use the generated evidence snapshot above for current
source target counts and promotion blockers; those values are intentionally
derived from canonical JSON rather than repeated by hand.

The stronger confidence gate is deliberately conservative:

- `source-neighborhood-supported`: range-stable model evidence plus complete
  OEIS coverage, at least two complete non-OEIS source families, no true
  mismatches, and no unknown unmatched rows.
- `candidate pattern`: source-neighborhood-supported evidence plus richer
  representation models beating metrics-only under matched controls.
- `proof`: unavailable unless there is a formal independently checkable proof
  artifact.

Publish the canonical evidence JSON and derived dashboard summary:

```sh
./build/collatz_evidence_publish \
  --full-audit data/generated/full_audit/summary.json \
  --stratified-metadata data/generated/stratified/metadata.json \
  --topology-manifest data/generated/topology/embedding_topology.json \
  --validation-metrics data/generated/evidence_validation/metrics.json \
  --ablation-report data/generated/evidence_validation/ablation_report.csv \
  --source-alignment data/generated/source_alignment/source_alignment.json \
  --neural-status data/generated/runner/neural_parallel_status.json \
  --output data/generated/evidence/latest_public_summary.json
```

This writes `latest_public_summary.json` as the public source of truth and a
compact `hypotheses/summary.json` derived from the same canonical evidence.

Train a research-only Graph Neural Network embedding over the exported graph:

```sh
docker compose --profile gnn run --rm gnn
```

The GNN service uses PyTorch CUDA in Docker, reads `nodes.csv`, `edges.csv`,
and `start_membership.csv`, and writes node embeddings plus pooled
`start_embeddings.csv` for retrieval-based evidence validation.

Start the progress web server:

```sh
./build/collatz_web \
  --host 127.0.0.1 \
  --port 8080 \
  --runner-status data/generated/runner/status.json \
  --evidence-summary data/generated/evidence/latest_public_summary.json
```

Then open `http://127.0.0.1:8080`.
The dashboard API separates canonical `evidence` from live `operations`.
Operations include scanner, topology preview, GNN preview, CPU, GPU, neural job,
and runner status fields. Operations telemetry never raises the confidence
label.

## Background Evidence Runner

Run one idempotent evidence cycle manually:

```sh
ops/collatz-evidence-cycle.sh --once
```

Preview the stages without writing generated artifacts:

```sh
ops/collatz-evidence-cycle.sh --dry-run
```

The cycle:

1. Acquires a lock so only one evidence cycle runs at a time.
2. Downloads public source files into ignored `data/imported/`.
3. Generates expanded source targets under ignored `data/generated/`.
4. Aligns source targets against the current topology.
5. Publishes the canonical evidence summary used by README and dashboard claims.
6. Optionally runs a CPU crunch scan batch when configured.
7. Runs the deterministic full-dataset audit over the current binary feature
   file.
8. Optionally runs a reviewed neural stage only when configured and GPU policy
   allows it. The recommended neural command is the parallel launcher, which
   runs independent contrastive ablations and the autoencoder together.
9. Runs the public privacy scan over tracked source/docs/templates.
10. Appends a sanitized private ledger entry.
11. Writes `data/generated/runner/status.json` for the Automation dashboard card.

The runner status includes only public-safe fields: evidence score, score delta,
cycle count, source target counts, CPU crunch state, GPU availability state,
neural stage state, and parallel neural job counts. The evidence score is not
theorem probability. It is an empirical research score based on the current
confidence gate, source-record match rate, and neural validation lift.

The full audit is deterministic dataset evidence. It proves the generated
feature file was read and summarized end to end, but it does not prove the
Collatz conjecture.

Example user-level systemd templates are in `ops/`:

- `ops/collatz-evidence-cycle.env.example`
- `ops/collatz-evidence-cycle.service.example`
- `ops/collatz-evidence-cycle.timer.example`

The example timer runs every 30 minutes. The environment file is meant to be
copied to an untracked private location and adjusted there. The runner does not
auto-commit or auto-push public Git changes.

To keep a private machine actively crunching between evidence cycles, enable the
optional CPU and neural stages in that untracked environment file:

```sh
COLLATZ_RUN_CPU_CRUNCH=1
COLLATZ_CPU_CRUNCH_STEP=10000000
COLLATZ_CPU_CRUNCH_THREADS=8
COLLATZ_CPU_CRUNCH_COMMAND='docker compose run --rm scanner-cpu'
COLLATZ_RUN_NEURAL=1
COLLATZ_GPU_ALLOW_SHARED=1
COLLATZ_NEURAL_COMMAND='./ops/collatz-neural-parallel.sh'
COLLATZ_PARALLEL_NEURAL_JOBS=4
COLLATZ_PARALLEL_FEATURE_SETS='hybrid metrics parity residue tokens'
```

The dashboard will then show whether CPU crunching is disabled, running, or
complete, and whether the GPU neural stage is disabled, running, complete, or
skipped because the GPU is unavailable, busy, or below the free-memory policy.
When the parallel neural launcher is active, the dashboard also shows active
neural jobs, completed jobs, GPU utilization, memory use, and power draw.

Run the public privacy scan directly:

```sh
ops/privacy-scan.sh
```

## Docker Compose

```sh
docker compose up --build web scanner-cpu
```

For the CUDA profile:

```sh
docker compose --profile cuda up --build scanner-cuda
```

The default CPU scan range is `1..1,000,000`. Override with environment
variables:

```sh
SCAN_START=1 SCAN_END=100000000 docker compose up --build scanner-cpu web
```

The scanner service writes compact binary features by default. After the scan
has produced `data/generated/features.bin`, run the representation exporters:

```sh
docker compose --profile research run --rm embedder
docker compose --profile research run --rm representatives
docker compose --profile research run --rm graph
docker compose --profile research run --rm topology
docker compose --profile research run --rm neighborhoods
docker compose --profile research run --rm insights
docker compose --profile research run --rm stratified
docker compose --profile research run --rm stratified-embedder
docker compose --profile research run --rm family-labels
docker compose --profile research run --rm pair-sampler
docker compose --profile research run --rm visualizer
docker compose --profile research run --rm image-tensors
docker compose --profile gnn run --rm gnn
docker compose --profile neural run --rm contrastive-v2
docker compose --profile neural run --rm autoencoder
docker compose --profile research run --rm evidence-validation
docker compose --profile research run --rm source-alignment
docker compose --profile research run --rm hypotheses
```

## Source Grounding

Source notes and validation tiers are in `docs/SOURCES.md`.
The neural embedding and visualization plan is in `docs/NEURAL_ENGINE_PLAN.md`.
Benchmark notes are in `docs/BENCHMARKS.md`.

The small committed validation sample lives at:

```text
data/source_validation/reference_samples.csv
```

Run it with:

```sh
./build/collatz_validate_sources
```

## Options

The current C++ plotter accepts:

```text
--width N           Plot width, default 100
--height N          Plot height, default 24
--max-steps N       Stop after N steps if 1 is not reached, default 1000000
--csv FILE          Save step,value,log10_value data for outside plotting
```

The original C arbitrary-size plotter source remains in `collatz_plot.c` for
reference, but the active research system builds through CMake as C++20.
