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

## Current Dashboard Snapshot

![Current Collatz evidence dashboard snapshot](docs/media/dashboard-summary.svg)

Latest checked state:

- Confidence: `range-stable signal`
- Conclusion: the learned path-family signal survives current holdouts, but it
  is still not a proof or source-aligned candidate.
- Coverage: topology covers `0.067%` of scanned rows; the stratified evidence
  sample covers `0.036%` directly while intentionally oversampling rare
  behaviors.
- Full audit: the deterministic C++ audit reads every binary feature row and
  supplies the denominator for neural coverage and holdout claims.
- Strongest evidence: learned embeddings beat random by `9.557%`, numeric
  adjacency by `5.168%`, and the weakest range holdout still has `6.428%`
  lift.
- Latest neural result: contrastive lift `9.557%`, range minimum lift `6.428%`,
  fold minimum lift `8.070%`.
- Source check: `5,091 / 5,191` public validation starts currently align with
  the topology source-neighborhood gate.
- Current limitation: source-record alignment is still partial. Roosendaal,
  Oliveira e Silva, and Barina imports need to agree before the claim gets
  promoted beyond a range-stable signal.
- Next experiment: expand source-record imports beyond OEIS, then rerun
  source-neighborhood, path-image, and GNN ablations.
- Automation: a private-safe evidence runner can refresh generated source
  targets, source alignment, hypotheses, and dashboard status on a timer. The
  runner writes public-safe state only; it does not auto-commit or auto-push.

The dashboard is intentionally compact: it should answer what the AI currently
believes, how confident it is, what evidence supports that, what limits the
claim, and what experiment should falsify or strengthen it next.

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
  --starts data/generated/graphs/starts.csv
```

This writes `samples.csv` and `metadata.json`. Every selected start has one or
more selection reasons: random baseline, range band, residue bucket, record-like
ladder, high stopping time, high peak behavior, topology representative, or GNN
representative.

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
  --input data/generated/ml_1_100m/metrics.csv \
  --output-dir data/generated/topology \
  --clusters 16
```

This writes a 2D PCA-style projection, k-means cluster summaries, and
`embedding_topology.json` for the dashboard. It is a baseline sanity map before
UMAP/RAPIDS/FAISS/GNN training.

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

Train the first research contrastive embedding model on a CUDA-capable GPU:

```sh
docker compose --profile neural run --rm contrastive
```

This writes `data/generated/contrastive/embeddings.csv`, `encoder.pt`, and
`metrics.json`. The key metric is whether nearest-neighbor purity beats the
random label baseline.

Run feature-family ablations when a learned signal looks promising:

```sh
CONTRASTIVE_FEATURE_SET=metrics CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_metrics docker compose --profile neural run --rm contrastive
CONTRASTIVE_FEATURE_SET=parity CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_parity docker compose --profile neural run --rm contrastive
CONTRASTIVE_FEATURE_SET=residue CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_residue docker compose --profile neural run --rm contrastive
CONTRASTIVE_FEATURE_SET=tokens CONTRASTIVE_OUTPUT_DIR=/work/data/generated/contrastive_tokens docker compose --profile neural run --rm contrastive
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
feature baselines, and learned feature-family ablations.

Compare public source-validation starts against topology neighborhoods:

```sh
./build/collatz_source_align \
  --projection data/generated/topology/projection.csv \
  --source-samples data/generated/source_validation/public_source_targets.csv \
  --output-dir data/generated/source_alignment
```

This writes `source_alignment.json` and `source_targets.csv`. The current
public source target table expands the source check from a smoke test to 5,019
OEIS-derived validation starts. It can support a `source-aligned candidate`,
but it still needs larger Roosendaal, Oliveira e Silva, and Barina imports
before the claim gets stronger.

The stronger confidence gate is deliberately conservative:

- `source-aligned candidate`: range-stable model evidence plus public source
  alignment from fewer than three independent source families.
- `multi-source-aligned candidate`: range-stable model evidence plus full
  agreement from OEIS and at least two of Roosendaal, Oliveira e Silva, and
  Barina.
- Any missing imported source target keeps the conclusion at the lower current
  confidence and reports the missing count as the next falsification target.

Generate the AI hypothesis ledger and dashboard summary:

```sh
./build/collatz_hypothesis_analyze \
  --insights data/generated/insights/insights.json \
  --stratified-metadata data/generated/stratified/metadata.json \
  --contrastive-metrics data/generated/contrastive/metrics.json \
  --autoencoder-metrics data/generated/anomalies/metrics.json \
  --gnn-metrics data/generated/gnn/metrics.json \
  --validation-metrics data/generated/evidence_validation/metrics.json \
  --source-alignment data/generated/source_alignment/source_alignment.json \
  --output-dir data/generated/hypotheses
```

This writes `hypotheses.jsonl` and `summary.json`. The dashboard uses this as
the compact AI conclusion layer: confidence level, coverage, strongest evidence,
weakest limitation, latest neural result, and next experiment.

Train a research-only Graph Neural Network embedding over the exported graph:

```sh
docker compose --profile gnn run --rm gnn
```

The GNN service uses PyTorch CUDA in Docker, reads `nodes.csv` and `edges.csv`,
and writes `data/generated/gnn/embeddings.csv` plus `metrics.json` for the
dashboard.

Start the progress web server:

```sh
./build/collatz_web \
  --host 127.0.0.1 \
  --port 8080 \
  --runner-status data/generated/runner/status.json
```

Then open `http://127.0.0.1:8080`.
The dashboard reads compact scanner, topology, graph, GNN, hypothesis, and
automation status fields when those files exist. The API intentionally returns
only public-safe runner fields such as state, stage, source target count,
matched source count, and next stage.

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
5. Regenerates the hypothesis summary used by the dashboard.
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
docker compose --profile research run --rm visualizer
docker compose --profile gnn run --rm gnn
docker compose --profile neural run --rm contrastive
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
