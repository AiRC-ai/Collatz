# Neural Engine v12: Held-out / disjoint evaluations

## Why v12

External review pointed out that the v9-v11 k-NN retrieval purity was computed
over *all* rows, which is transductive / in-sample: the same rows used to train
the embedding were also in the retrieval pool. v12 re-runs the key claims on
proper held-out, disjoint splits so the result is a generalization claim, not an
in-sample one. All lifts use the corrected k=2 metric (self excluded, topk(2)).

## What v12 does

`research/heldout_eval_v12.py` runs two held-out evaluations:

1. **Family-disjoint (fine family_id, >=5 members).** Split *families* (not
   rows) into train (80%) / test (20%). Train prototypical on train families;
   evaluate k=2 retrieval lift on the held-out test families -- families the
   model never saw. 3 seeds for mean +/- std.
2. **Row-disjoint (coarse multi-task).** Split rows 80/20; train v10-style
   multi-task supervised on the train rows; evaluate k=2 retrieval lift on the
   held-out test rows per coarse label.

Queries and the retrieval pool come only from the held-out split in both cases.

## Results

### Family-disjoint -- fine family_id (>=5 members)

| metric | value |
|---|---:|
| train families / test families | 1,016 / 253 |
| raw-metrics lift (test pool) | +25.1% |
| held-out prototypical lift (mean over 3 seeds) | **+82.3%** |
| std across 3 seeds | +/-1.2% (CI95 +79.9% .. +84.6%) |
| per-seed held-out | +0.840 / +0.812 / +0.816 |

The fine-family result generalizes to *unseen* families and is stable across
seeds. It is not transductive overfitting. (Note: the held-out pool has fewer
families than the v11 in-sample pool, so the raw baseline here (+25.1%) differs
from v11's (+12.5%); the fair comparison is held-out prototypical vs raw *on the
same held-out pool*, where the learned metric is ~3.3x raw.)

### Row-disjoint -- coarse multi-task (80/20 rows)

| Label | raw (test) | held-out (test) | beats raw |
|---|---:|---:|---|
| range_band | +12.0% | +84.8% | yes |
| bit_length | +25.1% | +70.9% | yes |
| peak_ratio_bucket | +50.4% | +83.3% | yes |

The coarse embedding generalizes to held-out rows; the held-out lifts (+85/+71/+83)
match the in-sample v10 numbers (+88.6/+72.2/+85.4), so the coarse result is not
row-memorization.

## Verdict

Both the fine (family-disjoint) and coarse (row-disjoint) results hold on
held-out splits. The v9-v11 findings are real generalization, not transductive
artifacts.

## Reproduce

```bash
python3 research/heldout_eval_v12.py --output-dir data/generated/contrastive_v12
```

~15-20 minutes on the RTX 3090 (3 seeds x 3000 prototypical episodes + a 60-epoch
coarse retrain + held-out evals).

This is empirical evidence, not a Collatz proof.
