# Neural Pipeline

The neural layer is an evidence engine, not a proof engine. It compares learned
neighborhoods against random, numeric-adjacency, holdout, source, and ablation
baselines.

The public leaderboard tracks:

- `metrics-only`
- `shape-only`
- `parity-sequence-only`
- `residue-sequence-only`
- `image-only`
- `GNN-only`
- `hybrid`

Missing fold, seed, or confidence-interval statistics are represented as
explicit `null` values. If `metrics-only` beats `hybrid`, the summary marks the
result as a `metric-dominant signal` and blocks candidate-pattern promotion.

The v2 path-family flow is:

1. Export `metrics_safe.csv` with `collatz_embed_export --metric-mode safe`.
2. Generate deterministic trajectory labels with `collatz_family_labels`.
3. Build positive family pairs and matched hard negatives with
   `research/pair_sampler.py`. The canonical default requests four exact
   hard negatives per anchor so rare family/control buckets do not dominate the
   match-rate gate.
4. Train `research/contrastive_train_v2.py` using family pairs, ordered parity
   sequences, ordered residue sequences, log-path sketches, and representation
   dropout. Neighbor-purity reporting defaults to `tail_hash`, so the primary
   target asks whether nearby embeddings share a true trajectory tail rather
   than a coarse magnitude bucket.
5. Validate with retrieval metrics and at least five seeds by five folds before
   any confidence promotion.

Matched controls are false unless the hard-negative sampler produced and
reported valid matches above its match-rate threshold. The hard-negative control
key is bit length, range band, residue class, stopping-time bucket,
peak-ratio bucket, and first-drop bucket; source family is a positive-pair
signal, not a matched-control substitute. Operational GPU telemetry can show
whether training is running, but it never changes the evidence confidence label.
