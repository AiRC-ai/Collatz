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
  is not a proof and is not yet a source-aligned candidate.
- Coverage: topology covers `0.100%` of scanned rows; the stratified evidence
  sample covers `0.006%` directly while intentionally oversampling rare
  behaviors.
- Strongest evidence: learned embeddings beat random by `20.141%`, numeric
  adjacency by `7.738%`, and the weakest range holdout still has `14.537%`
  lift.
- Latest neural result: contrastive lift `20.141%`, range minimum lift
  `14.537%`, fold minimum lift `14.021%`.
- Source check: `11 / 11` known validation starts matched across `9` topology
  clusters.
- Current limitation: source alignment is still a smoke check; larger dated
  Roosendaal, Oliveira e Silva, Barina, and OEIS record imports are required
  before promoting this to `source-aligned candidate`.
- Next experiment: import larger source-record tables and rerun
  source-neighborhood alignment.

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

![Collatz path image atlas preview](docs/media/path-image-atlas.svg)

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
- `build/collatz_select_representatives`
- `build/collatz_stratified_sample`
- `build/collatz_graph_export`
- `build/collatz_embedding_analyze`
- `build/collatz_neighborhood_analyze`
- `build/collatz_selftest`
- `build/collatz_insight_analyze`
- `build/collatz_hypothesis_analyze`

Run validation:

```sh
make test
```

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

Compare known source-validation starts against topology neighborhoods:

```sh
./build/collatz_source_align \
  --projection data/generated/topology/projection.csv \
  --source-samples data/source_validation/reference_samples.csv \
  --output-dir data/generated/source_alignment
```

This writes `source_alignment.json` and `source_targets.csv`. The current
committed source file is a smoke check, so this can confirm source wiring but
cannot promote a claim to `source-aligned candidate` until larger dated
Roosendaal, Oliveira e Silva, Barina, and OEIS imports are added.

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
./build/collatz_web --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080`.
The dashboard reads compact scanner, topology, neighborhood, graph, and GNN
status fields when those files exist.

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
