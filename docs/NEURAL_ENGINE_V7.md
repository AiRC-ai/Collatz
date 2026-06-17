# Neural Engine v7: Raw-Metrics Hard Negatives + Coarse Labels

## What v7 is

Contrastive embedding model for Collatz trajectory space using:

- **Single V1 encoder** — one MLP that concatenates all feature branches (metrics-only).
- **Range-band positive pairs** — 16 coarse-grained classes (~6,250 each) from `range_band`.
- **Hard negatives from raw metrics** — mined once at the start from normalized 32-dim metrics space.
- **Auxiliary metrics preservation loss** — keeps the embedding directionally aligned with raw features.
- **No self-noise pairs**, no per-epoch negative mining.
- **Evaluation every 10 epochs**.

## Why v7 exists: lessons from v4, v5, v6

### v4 — mined from model embeddings (failed)
- Mined hard negatives from the model's own embeddings.
- Embeddings start random → mine wrong negatives → model optimizes away wrong signal.
- Result: lift ~0.0001, essentially no learning.

### v5 — mined from raw metrics but used fine-grained labels (failed)
- Correctly mined from raw metrics (bypasses embedding feedback loop).
- But used `coalescence_family_id` — 61K classes with ~1.6 examples each.
- Contrastive learning needs multiple examples per class to learn discriminative features.
- Result: lift ~0.0000, no signal.

### v6 — coarse labels (right) but complex training loop (incomplete)
- Switched to `range_band` (16 classes, 6,250 each) — correct direction.
- Raw metrics mining — correct.
- But: all-to-all positive pairs + random downsample was wasteful.
- Precomputed embeddings before training, then tried to re-mine them — redundant.
- Training loop used batch-relative indexing for negatives, but negatives were global indices — mismatch.

### v7 — clean, working implementation
- Positive pairs: sample one positive per anchor from the same label group each epoch.
- Hard negatives: mined once from raw metrics, never re-mined.
- Triplet loss uses a precomputed `z_all` reference tensor for correct negative indexing.

## Architecture

```
Input: metrics (32-dim)
    → Linear(32→192) → GELU → Dropout(0.15)
    → Linear(192→192) → GELU → Dropout(0.1)
    → Linear(192→64) → embedding z[64]

Auxiliary head:
    z[64] → Linear(64→512) → GELU → Dropout(0.1) → Linear(512→32) → projected z
```

## Loss

```
triplet = max(0, d(anchor, positive) - min(d(anchor, neg_i)) + margin)
preservation = 1 - cosine(projected_z, raw_metrics)
total = 0.8 * triplet + 0.2 * preservation
```

## Results

| Metric | Value |
|--------|-------|
| Epochs | 200/200 |
| Start loss | 0.095 |
| Final loss | 0.0016 |
| Neighbor purity (k=2) | 6.48% |
| Random baseline | 6.25% (1/16) |
| **Purity lift** | **0.23%** |
| Best lift | 0.23% |

**Verdict: v7 failed.** Loss dropped smoothly but the embedding did not learn the range_band family structure. The lift of 0.23% is effectively noise.

## Root cause analysis

The loss minimized but lift did not improve. This means:

1. The model optimized the triplet loss (pulling positives closer, pushing negatives away in raw metrics space)
2. But this optimization did not translate to better family clustering in the embedding
3. The metrics preservation loss (20% weight) likely anchored the embedding to raw metric space, preventing it from discovering latent family structure

**Conclusion**: Raw metrics + triplet loss is not a sufficient signal for learning family structure. The embedding needs either richer features (hybrid) or a different loss formulation (InfoNCE/NT-Xent).

## Usage

```bash
# Metrics-only, range_band labels
python3 research/contrastive_train_v7.py \
     --limit 0 \
     --feature-set metrics \
     --primary-label range_band

# No auxiliary loss, higher temperature
python3 research/contrastive_train_v7.py \
     --no-metrics-loss \
     --temperature 0.1 \
     --epochs 300
```

## Next: v8

- **Hybrid features** (all branches: metrics, shape, parity, residue)
- **InfoNCE/NT-Xent loss** instead of triplet loss
- **Higher temperature** (0.15) for more diverse negatives
- **No auxiliary loss** (remove the metrics preservation anchor)
- **Evaluate every 10 epochs**
