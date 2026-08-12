# Neural Engine v11: Few-shot prototypical + re-clustered families

## Why v11

- Coarse labels (`range_band`, `bit_length`, `peak_ratio_bucket`) are saturated
  at 88-99% under supervised v10; pushing them is noise.
- The real target is the fine `coalescence_family_id`, but it is 58% singletons
  (35,856 of 61,754 families have one member) and 0 families have >=20 members.
  You cannot learn a class from one example, so the original fine label is
  unlearnable for most rows.
- Diagnostics showed the signal *does* exist and scales with family size:
  raw-metrics k-NN lift on `family_id` is +1.9% for >=2-member families and
  **+15.0% (k=1) / +11.1% (k=2)** for >=5-member families. The barrier is the
  family-size distribution, not the features or the model.
- Hybrid features HURT the fine target (full hybrid +5.6% vs metrics +15.0%),
  so v11 uses metrics-only (`m0`-`m31`).

## What v11 does

`research/prototypical_train_v11.py` does two things:

1. **Re-cluster.** k-means in metrics space into ~2,000 families (~50 members
   each) so the target covers the whole dataset. Diagnose whether the clusters
   are meaningful (non-magnitude) and whether they recover the original
   `family_id` structure (NMI).

2. **Prototypical few-shot.** Episodic prototypical networks (Snell et al.) on
   metrics-only, on (a) the >=5-member original families and (b) the re-clustered
   families. Evaluate k=2 retrieval lift vs the raw-metrics baseline.

## Results

### Re-cluster (structural win)

| quantity | value |
|---|---|
| clusters | 1,999 |
| size: min / median / max / avg | 1 / 47 / 192 / 50.0 |
| modal `range_band` purity per cluster | 0.223 (only 22% magnitude) |
| NMI(cluster, range_band) | 0.134 |
| NMI(cluster, family_id) | **0.768** |
| NMI(family_id, range_band) | 0.415 |

The re-clustered families cover all 100,000 rows at a learnable granularity
(~50 members each, vs 58% singletons), are only 22% magnitude (finer structure,
not just `range_band`), and recover 77% of the original fine family structure.
The original `family_id` is itself 42% magnitude-aligned, so the clusters are
*less* magnitude-dominated than the raw label while still capturing most of it.
**Conclusion: the fine family structure is real and learnable at a coarser
granularity; the singleton problem was a granularity artifact, not a "families
do not exist" problem.**

### Few-shot prototypical (method win) -- >=5-member original families

| representation | k=2 retrieval lift |
|---|---:|
| raw metrics (no learning) | +11.1% |
| prototypical few-shot (learned) | **+58.1%** |

1,269 families, 7,101 rows, metrics-only features, 3,000 episodes (30-way,
3-shot, 2-query). The learned metric is 5.2x the raw-metrics baseline. The fine
target is improvable, and the lever is the few-shot method -- exactly the v9/v10
supervised idea adapted to many-class-few-shot.

### Re-cluster prototypical (circular, for completeness)

| representation | k=2 lift by cluster |
|---|---:|
| raw metrics | +64.6% |
| prototypical learned | +67.1% |

This is **circular** -- the cluster label is defined by k-means *in metrics
space*, so raw-metrics k-NN purity by cluster is high by construction. It is
reported only for completeness; the meaningful re-cluster signal is the
diagnostics above.

## Architecture

```
metrics (m0..m31, 32-d, standardized)
  -> Linear(32, 128) -> GELU -> Dropout(0.2)
  -> Linear(128, 64)  -> GELU            # 64-d embedding
prototypical episode: sample N_way classes, K_shot support + Q_query per class;
  prototype = mean support embedding; loss = CE(-dist(query, prototypes))
```

k-means is a from-scratch GPU implementation (no sklearn on the reference CUDA
workstation); NMI is computed from joint histograms.

## Reproduce

```bash
python3 research/prototypical_train_v11.py --output-dir data/generated/contrastive_v11
```

~10-15 minutes on the RTX 3090 (k-means + two 3,000-episode prototypical
trainings + evals).

## Next

The re-clustered families are a validated, learnable target covering the whole
dataset. The natural next step is supervised/prototypical training *on the
re-clustered families as the label* (non-circularly: train on a train split,
evaluate retrieval lift on a held-out split) to get a single general-purpose
fine-structure embedding, and to study what the ~2,000 clusters actually
correspond to in Collatz-trajectory terms.

This is empirical evidence, not a Collatz proof.
