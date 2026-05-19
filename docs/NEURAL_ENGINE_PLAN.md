# Neural Engine Plan: Latent Collatz Topology

The neural layer should not start as a simple stopping-time predictor. The main
research target is the latent organization of Collatz trajectory space: path
families, attractor-like structures, anomaly islands, recurrence texture, and
residue-driven topology.

This is closer to modern NDR, IDS, malware-family clustering, and behavioral
embedding systems than traditional brute-force number theory tooling.

## Priority Shift

Primary:

- learn path embeddings
- discover clusters and outliers
- render latent maps
- compare path-as-image encodings
- identify residue/path families

Secondary:

- predict stopping time
- predict first-drop time
- predict peak behavior
- classify trajectory families

The scanner remains C++/CUDA only. The ML research layer may use PyTorch CUDA,
RAPIDS, FAISS, UMAP, and related GPU tools because the useful work runs in GPU
kernels and the ecosystem matters more than Python dispatch overhead at this
stage. If a model becomes production-critical, export or port that model later
to LibTorch C++, ONNX Runtime, or TensorRT.

## Data Products From Scanner

The scanner should produce these data families:

- scalar metrics: total steps, first-drop time, peak ratio, odd/even counts
- parity streams: bit-packed odd/even prefixes and run-length sequences
- residue trajectories: mod 3, 4, 8, 16, 32, and later larger residue ladders
- compressed path sketches: fixed-length log-value curves and slope-change bins
- transition summaries: halving-run histograms and residue transition counts
- selected full paths: record setters, anomalies, cluster representatives

Compact binary feature files are the default for high-volume generation.

Current concrete artifact path:

- `collatz_scan_cpu --format bin` writes compact feature rows and a metadata
  sidecar.
- `collatz_embed_export` derives normalized metric vectors, parity-run token
  streams, residue-transition token streams, and fixed-length log-path sketches.
  It can export either the first N rows or a stratified evidence sample via
  `--sample-file`.
- `collatz_path_image` renders recurrence plots, Gramian Angular Fields, Markov
  Transition Fields, parity rasters, and residue rasters.
- `collatz_select_representatives` picks record-like, residue-family, and
  deterministic sample starts for deeper review.
- `collatz_stratified_sample` builds evidence-first samples from the full
  binary scan: random baselines, range bands, residue buckets, high-step paths,
  high-peak paths, record ladders, topology reps, and GNN starts.
- `collatz_graph_export` converts selected trajectories into GNN-ready
  `nodes.csv`, `edges.csv`, `starts.csv`, and dashboard preview JSON.
- `collatz_embedding_analyze` builds the first deterministic baseline topology
  map from exported metric vectors using 2D PCA-style projection and k-means.
- `collatz_hypothesis_analyze` turns scanner, topology, neural, anomaly, and
  GNN artifacts into a compact falsifiable hypothesis ledger for the dashboard.

## Embedding Targets

Each start value should eventually map to one or more learned vectors:

```text
n -> feature vector -> encoder -> embedding[128 or 256]
```

Recommended first embeddings:

- Metric embedding: scalar and histogram features.
- Parity embedding: bitstream or tokenized parity-run sequence.
- Shape embedding: fixed-length log-value trajectory sketch.
- Residue embedding: residue-transition token stream.
- Hybrid embedding: concatenated or cross-attended representation.

## Model Families To Test

1. Contrastive encoder
   - Best first serious model.
   - Learns that similar trajectory dynamics should be close and different
     dynamics should be far apart.
   - Good for unknown-family discovery.

2. Autoencoder
   - Best for anomaly embeddings.
   - Train reconstruction on common behavior; surface high-error paths.

3. Transformer encoder
   - Best for long-range parity and residue sequence structure.
   - Treat parity runs or residue states as tokens.

4. CNN over path images
   - Best for recurrence plots, Gramian Angular Fields, and transition fields.
   - Useful when visual texture becomes more informative than scalar metrics.

5. Graph neural network
   - Represent trajectories and coalescing paths as graph structures.
   - Promising for equivalence classes, shared tails, and convergence topology.
   - First concrete artifact: directed trajectory graph with shared-tail nodes,
     scalar node features, and edge lists ready for PyTorch Geometric, DGL, or
     another GNN stack.
   - Current research trainer: `research/gnn_train.py`, a lightweight PyTorch
     CUDA GraphSAGE-style link-reconstruction encoder that writes node
     embeddings and dashboard metrics.

## Path-As-Image Encodings

Generate image-like tensors from paths and train image encoders against them:

- Recurrence plots:
  - `pixel(i,j) = similarity(path_i, path_j)`
  - Target: self-similarity, repeated motifs, resonance bands.

- Gramian Angular Fields:
  - Convert normalized sequence values into angular image space.
  - Target: global shape and phase relationships.

- Markov Transition Fields:
  - Encode transition probabilities over quantized states.
  - Target: residue corridors and convergence-state patterns.

- Parity raster:
  - Render parity/run tokens into fixed-width binary images.
  - Target: parity motifs and recursive structure.

## Visualization Layer

The visualizer should produce:

- UMAP maps
- t-SNE maps for smaller samples
- PCA baselines
- deterministic baseline topology maps from `collatz_embedding_analyze`
- FAISS nearest-neighbor neighborhoods
- recurrence plot atlases
- GAF and transition-field atlases
- cluster labels and anomaly maps
- GNN graph previews with node/edge counts, selected start families, and
  coalescing shared-tail structure

Expected visual outputs:

- families
- attractors
- ridges
- transition bands
- outlier islands
- near-record neighborhoods

## Service Additions

Add these services after scanner output is stable:

```yaml
embedder:
  purpose: Train and export trajectory embeddings.
  stack: PyTorch CUDA, FAISS, optional RAPIDS.

visualizer:
  purpose: Generate latent maps and path-image atlases.
  stack: UMAP, t-SNE, PCA, cuML/RAPIDS where practical.

notebook:
  purpose: Research-only exploration surface.
  stack: Jupyter or equivalent, never part of the core scanner.
```

The repository currently exposes `embedder` and `visualizer` Compose services
under the `research` profile. They are intentionally simple artifact exporters
first; learned embeddings, UMAP, FAISS, and model checkpoints layer on top after
the binary dataset path is validated.

The core dashboard should show embedding progress later:

- dataset version
- encoder type
- embedding dimension
- cluster count
- nearest-neighbor sample
- anomaly candidates
- latest UMAP image/map path
- current hypothesis confidence level
- evidence coverage
- strongest evidence and weakest limitation
- latest neural result
- next falsification experiment

## First Experiments

1. Generate `1..100M` compact binary features.
2. Build an evidence-first stratified sample by range, stopping-time extremes,
   peak-ratio extremes, first-drop outliers, residue class, record ladders,
   topology representatives, GNN starts, and deterministic random baselines.
3. Train a metric + parity-run contrastive encoder.
4. Run feature ablations: metrics-only, parity-only, residue-only,
   parity+residue tokens, and hybrid.
5. Validate learned neighborhoods against numeric adjacency, range holdouts,
   residue holdouts, random folds, and raw-feature baselines.
6. Compare known source-validation starts against topology neighborhoods.
7. Project embeddings with UMAP.
8. Inspect cluster neighborhoods against known records.
9. Train an autoencoder and collect high-reconstruction-error anomalies.
10. Generate recurrence/GAF/transition images for cluster representatives.
11. Write a hypothesis ledger entry that states the claim, confidence level,
   evidence, limit, falsification test, and next action.

## Success Criteria

The neural engine is useful when it can:

- produce stable clusters across independent numeric ranges
- place known record setters near meaningful neighborhoods or outlier bands
- identify repeated path families not obvious from raw stopping time alone
- reveal visual structures that survive resampling and feature ablation
- produce anomaly candidates worth saving as full paths for inspection

## Confidence Levels

The system must avoid proof language. Hypotheses move through these empirical
levels only:

- `pipeline-check`: the artifact path works, but the result is not yet evidence.
- `sample-local signal`: a pattern appears in one sample or baseline view.
- `range-stable signal`: the pattern survives independent numeric ranges.
- `source-aligned candidate`: the pattern is consistent with trusted external
  records or known record setters.
- `candidate pattern`: the pattern survives holdouts and feature ablations and
  is worth deeper mathematical inspection.
