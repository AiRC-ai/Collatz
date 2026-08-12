# Neural Engine v9/v10: Supervised Embedding (retires self-supervised contrastive)

## What changed and why

v9 and v10 stop doing self-supervised contrastive learning and use the labels
directly. The whole v4-v8 line tried to rediscover label structure *without*
using the labels, and it kept collapsing. v9/v10 use supervised metric learning
because the labels are available and that is the reliable tool for this case.

## The diagnostic that ended the contrastive line

Before writing another loss function, three zero/low-training checks were run on
the real 100k dataset (RTX 3090 reference workstation):

1. **Raw-metrics k-NN (zero training).** Take the 32 standardized `m0..m31`
   metrics, L2-normalize, find the 2 nearest neighbors, measure same-label
   purity.
2. **Supervised MLP.** A 2-layer MLP, 40 epochs, classify the label.
3. **Supervised embedding.** The 64-d penultimate layer of that MLP, same k-NN
   purity metric used by v7.

| Label | classes | random | raw-kNN lift | supervised MLP acc | 64-d embedding lift |
|---|---:|---:|---:|---:|---:|
| range_band | 16 | 0.0625 | +0.1346 | 95.8% | +0.9205 |
| bit_length | 17 | 0.2619 | +0.2984 | 98.6% | +0.7304 |
| peak_ratio_bucket | 28 | 0.1289 | +0.5599 | 99.1% | +0.8666 |

The decisive number: **v7 self-supervised contrastive got +0.23% lift on
range_band, while the raw metrics already gave +13.5% with no training at all.**
v7's training destroyed signal that was trivially present. The features were
never the problem; the self-supervised contrastive objective was.

Two side findings:

- **`log_sketch` (s0..s127) is dead weight** for these labels (raw-kNN lift
  0.00 for range_band, anti-correlated for bit_length). v8's planned
  "hybrid = metrics + log_sketch" would have diluted the 32 good dims with 128
  noise dims. v9/v10 use metrics only.
- **v8 was broken in six ways** (read nonexistent named CSV columns -> all-zero
  features; InfoNCE passed embedding vectors as label indices; tuple crash in
  the hybrid branch; parity dimension mismatch; positional label alignment;
  inconsistent metrics-anchor default). It was rewritten to be correct, but
  the supervised route made it moot.

## v9 - single-task supervised embedding

`research/contrastive_train_v9.py`. Shared `V1Encoder` (metrics -> 64-d
embedding) plus a linear classification head; cross-entropy on one label;
cosine LR; 60 epochs; metrics-only features.

Result on range_band:

- Test accuracy **98.5%** (random 6.25%).
- Embedding neighbor purity **98.99%** -> lift **+92.7%**.
- vs v7 self-supervised **+0.23%**, vs raw-metrics k-NN **+13.5%**.

### v9 generalization check (does the embedding beat raw metrics everywhere?)

| Label | raw-metrics lift | v9 embedding lift | Winner |
|---|---:|---:|---|
| range_band | +0.1346 | +0.9274 | v9 |
| bit_length | +0.2984 | +0.7340 | v9 |
| peak_ratio_bucket | +0.5599 | +0.2373 | raw metrics |
| coalescence_family_id | +0.0085 | +0.0003 | raw metrics |

v9 is a **magnitude specialist**: it wins on range_band and the correlated
bit_length, but loses on peak_ratio_bucket and family_id, because supervising
one label sacrifices the others.

## v10 - multi-task supervised embedding

`research/contrastive_train_v10.py`. One shared 64-d embedding, three linear
heads (range_band, bit_length, peak_ratio_bucket), summed cross-entropy. This
forces the embedding to preserve all three kinds of structure.

Result (best epoch 60, summed lift +2.4513):

| Label | classes | random | raw-kNN lift | v10 lift | v10 acc | Winner |
|---|---:|---:|---:|---:|---:|---|
| range_band | 16 | 0.0625 | +0.1346 | +0.8799 | 97.4% | v10 |
| bit_length | 17 | 0.2620 | +0.2984 | +0.7201 | 98.6% | v10 |
| peak_ratio_bucket | 28 | 0.1289 | +0.5599 | +0.8513 | 97.5% | v10 |

**v10 beats raw-metrics k-NN on all three labels**, including peak_ratio_bucket
(+85.1% vs raw +56.0%), the label v9 lost on. This is the first general-purpose
retrieval embedding in the project.

## v10 hybrid vs metrics-only -- the honest feature comparison

The question was whether "our" richer representation (hybrid = all branches:
metrics + shape + parity + residue) actually beats the "original" metrics-only
representation. v10 was extended to take `--feature-set hybrid|full`, and a
hybrid v10 was trained with the same multi-task supervised method, then compared
to the metrics-only v10 on the same k=2 neighbor-purity metric.

| Label | Non-AI baseline (raw) | Original (metrics-only) | Ours (hybrid) | Hybrid vs original |
|---|---:|---:|---:|---:|
| range_band | +13.5% | +88.0% | +89.3% | +1.3 (hybrid) |
| bit_length | +29.8% | +72.0% | +72.7% | +0.7 (hybrid) |
| peak_ratio_bucket | +56.0% | +85.1% | +83.8% | -1.3 (metrics-only) |
| **summed lift** | - | +2.451 | +2.459 | +0.007 |

**Result: ours (hybrid) is tied with original (metrics-only).** The extra
branches add essentially nothing for these coarse labels -- consistent with the
raw-kNN diagnostic that showed `log_sketch` (shape) has zero signal for
range_band. The decisive gain came from switching the METHOD (supervised vs
self-supervised contrastive: +0.23% -> +88%), not from switching the FEATURES
(hybrid vs metrics-only: tied). So the hybrid representation is a safe superset
(no worse) but has not earned its extra complexity for the coarse labels; the
value of the hybrid work stays an open question for the fine
`coalescence_family_id` target, where richer features might matter.

This is empirical evidence, not a Collatz proof.

## Architecture

```
metrics (m0..m31, 32-d, standardized)
  -> Linear(32, 128) -> GELU -> Dropout(0.2)
  -> Linear(128, 64)  -> GELU            # 64-d embedding (the output)
  -> head_i: Linear(64, n_classes_i)    # one per task (v9: 1 head; v10: 3 heads)
loss = sum_i w_i * CrossEntropy(head_i, label_i)
```

Outputs (in `data/generated/contrastive_v9|v10/`): `embeddings.csv`
(n, e0..e63), `encoder.pt`, `encoder_best.pt`, `metrics.json`.

## Reproduce

On a CUDA workstation with the generated dataset:

```bash
python3 research/contrastive_train_v9.py  --output-dir data/generated/contrastive_v9
python3 research/contrastive_train_v10.py --output-dir data/generated/contrastive_v10
```

Each run is ~60 epochs and finishes in roughly a minute on the 3090.

## Next

The coarse labels are solved. The open target is the fine
`coalescence_family_id` (61,754 classes, ~1.6 examples each): raw-metrics k-NN
lift is only +0.85%, and with 1-2 examples per class neither supervised
classification nor contrastive learning has enough signal. Before training,
check whether family_id is well-defined in a richer feature space; otherwise
revise the family definition rather than tuning another loss.

This is empirical evidence, not a Collatz proof.
