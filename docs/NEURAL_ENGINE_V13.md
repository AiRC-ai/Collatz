# Neural Engine v13: Non-circular re-cluster evaluation + cluster characterization

## Why v13

v11 re-clustered trajectories into ~2,000 families and reported a retrieval lift,
but that number was **circular**: the clusters were defined by k-means *in metrics
space*, so raw-metrics k-NN purity by cluster is high by construction. v13 closes
that open item with (a) a characterization of what the clusters actually are and
(b) non-circular evaluations: a cluster-disjoint split and a cross-target test.

## What v13 does

`research/recluster_eval_v13.py`:

- **A.** Re-cluster: k-means in metrics space, K=2000 (~50 members each), all 100k.
- **B.** Characterize the clusters: NMI(cluster, each known label) and size-weighted
  modal purity per label.
- **C.** Cluster-disjoint held-out: split clusters 80/20; train prototypical on
  train clusters; retrieve on held-out test clusters. 3 seeds.
- **D.** Cross-target: train prototypical on ALL cluster-proxy labels (never seeing
  `family_id`); emit the 64-d embedding for all 100k; evaluate family-disjoint on
  the original fine `family_id` (>=5 members). Tests whether cluster-supervision
  produces a general-purpose fine-family embedding.

k=2 corrected metric throughout.

## Results

### B -- what the clusters are

| Label | NMI(cluster, label) | modal purity |
|---|---:|---:|
| range_band | 0.134 | 0.223 |
| bit_length | 0.155 | 0.495 |
| peak_ratio_bucket | 0.310 | 0.603 |
| coalescence_family_id | 0.768 | 0.030 |

The clusters recover fine family structure (NMI 0.768) and are NOT magnitude
(range_band NMI 0.134, only 22% modal purity). Their strongest single-label
alignment is peak_ratio (0.31). So the clusters are fine-family + peak-ratio
structure, not magnitude buckets in disguise.

### C -- cluster target is circular

| metric | value |
|---|---:|
| raw-metrics lift (test clusters) | +88.6% |
| held-out prototypical lift | +89.4% (std +/-0.2) |

Raw is near-ceiling because k-means clusters are Voronoi cells in metrics space,
so retrieval-by-cluster is circular by construction. This confirms the reviewer's
point that the v11 re-cluster retrieval number was not meaningful as a
generalization claim.

### D -- cluster-proxy does not beat raw metrics (honest negative)

Train prototypical on the cluster labels (never seeing `family_id`), then evaluate
family-disjoint on the original fine `family_id` (>=5 members):

| metric | value |
|---|---:|
| raw-metrics lift (test families) | +24.7% |
| cluster-trained held-out lift | +21.0% (std +/-0.9) |
| beats raw | no |

An embedding trained on cluster-proxy labels does NOT beat raw metrics on the
original fine families. The cluster-proxy is a weaker signal than direct
fine-family supervision.

## Verdict

- Re-clustering is a useful **diagnostic**: it proves fine structure exists at a
  learnable, non-magnitude granularity (NMI 0.768 with `family_id`, only 22%
  magnitude) and that the singleton problem was a granularity artifact.
- It is **not** a training target that beats raw metrics for fine-family retrieval:
  as a target it is circular (C), and as proxy training labels it underperforms
  raw metrics on the real fine families (D, +21.0% vs +24.7%).
- The winning fine-family approach remains **direct** few-shot prototypical
  supervision on the fine families themselves (v12: +82.3% held-out, family-
  disjoint), which far exceeds both raw (+24.7%) and the cluster-proxy (+21.0%).

## Reproduce

```bash
python3 research/recluster_eval_v13.py --output-dir data/generated/contrastive_v13
```

~20-25 minutes on the RTX 3090 (k-means K=2000 + 3 seeds x 3000 cluster-disjoint
episodes + one 3000-episode all-cluster train + cross-target evals).

This is empirical evidence, not a Collatz proof.
