# Neural Pipeline

The neural layer is an evidence engine, not a proof engine. It compares learned
neighborhoods against random, numeric-adjacency, holdout, source, and ablation
baselines.

The public leaderboard tracks:

- `metrics-only`
- `parity-only`
- `residue-only`
- `token-only`
- `image-only`
- `GNN-only`
- `hybrid`

Missing fold, seed, or confidence-interval statistics are represented as
explicit `null` values. If `metrics-only` beats `hybrid`, the summary marks the
result as a `metric-dominant signal` and blocks candidate-pattern promotion.
