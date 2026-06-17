#!/usr/bin/env python3
"""v11: few-shot prototypical metric learning + re-clustered learnable families.

Why v11 (the fine-structure frontier):
- Coarse labels (range_band/bit_length/peak_ratio) are saturated at 88-99% under
  supervised v10; pushing them is noise.
- Fine coalescence_family_id is 58% singletons (unlearnable as classes) and only
  1,269 families have >=5 members. But on those >=5-member families, raw-metrics
  k-NN already gives +15.0% lift -- the signal exists, it is just few-shot.
- Hybrid features HURT the fine target (full hybrid +5.6% vs metrics +15.0%), so
  v11 uses metrics-only (m0-m31).

v11 does two things:
1. RE-CLUSTER: k-means in metrics space into ~2000 families (~50 members each)
   so the target covers the whole dataset. Diagnose whether the clusters are
   meaningful (non-magnitude) and whether they recover the original family_id
   structure (NMI).
2. PROTOTYPICAL FEW-SHOT: train a metric/embedding with episodic prototypical
   networks (Snell et al.) on metrics-only, on (a) the >=5-member original
   families and (b) the re-clustered families. Evaluate k-NN retrieval lift vs
   the raw-metrics baseline.

All numbers on the same metric: k=2 nearest-neighbor same-label purity minus
the random baseline.
"""

from __future__ import annotations
import argparse, csv, json, math, random
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="v11: prototypical few-shot + re-clustered families")
    p.add_argument("--metrics-safe", default="data/generated/ml_stratified/metrics_safe.csv")
    p.add_argument("--families", default="data/generated/ml_labels/families.csv")
    p.add_argument("--output-dir", default="data/generated/contrastive_v11")
    p.add_argument("--n-clusters", type=int, default=2000, help="k-means clusters for re-cluster target")
    p.add_argument("--min-family", type=int, default=5, help="min members to keep an original family")
    p.add_argument("--n-way", type=int, default=30)
    p.add_argument("--k-shot", type=int, default=3)
    p.add_argument("--q-query", type=int, default=2)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--eval-every", type=int, default=500)
    return p.parse_args()


# ---- data ----
def standardize(x):
    m, s = x.mean(0), x.std(0)
    return (x - m) / s.clamp(min=1e-6)

def read_matrix(path, prefix, limit=0):
    starts, rows = [], []
    with open(path, newline="") as h:
        r = csv.DictReader(h)
        cols = [c for c in r.fieldnames or [] if c.startswith(prefix)]
        for row in r:
            starts.append(int(row["n"]))
            rows.append([float(row[c]) for c in cols])
            if limit and len(starts) >= limit:
                break
    return starts, standardize(torch.tensor(rows, dtype=torch.float32))

def read_labels(path, starts, col):
    idx = {n: i for i, n in enumerate(starts)}
    lab = ["?"] * len(starts)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if n in idx:
                lab[idx[n]] = row.get(col, "unknown")
    return lab


# ---- k-means (GPU) ----
def kmeans(X, k, iters=40, seed=0, chunk=8000):
    torch.manual_seed(seed)
    dev = X.device
    centroids = X[torch.randperm(X.shape[0], device=dev)[:k]].clone()
    labels = torch.zeros(X.shape[0], dtype=torch.long, device=dev)
    for _ in range(iters):
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            labels[s:e] = torch.cdist(X[s:e], centroids).argmin(1)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(0)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        labels[s:e] = torch.cdist(X[s:e], centroids).argmin(1)
    return labels.cpu()


# ---- NMI (manual) ----
def nmi(a, b):
    a = a.tolist() if torch.is_tensor(a) else list(a)
    b = b.tolist() if torch.is_tensor(b) else list(b)
    n = len(a)
    ua, ub = sorted(set(a)), sorted(set(b))
    ca = {v: i for i, v in enumerate(ua)}; cb = {v: i for i, v in enumerate(ub)}
    joint = torch.zeros(len(ua), len(ub))
    for x, y in zip(a, b):
        joint[ca[x], cb[y]] += 1
    joint = joint / n
    pa = joint.sum(1)   # [A]
    pb = joint.sum(0)   # [B]
    denom = pa[:, None] * pb[None, :]    # [A, B]
    mask = (joint > 0) & (denom > 0)
    mi = (joint[mask] * torch.log(joint[mask] / denom[mask])).sum().item()
    ha = -(pa[pa > 0] * torch.log(pa[pa > 0])).sum().item()
    hb = -(pb[pb > 0] * torch.log(pb[pb > 0])).sum().item()
    if ha <= 0 or hb <= 0:
        return 0.0
    return mi / math.sqrt(ha * hb)


# ---- model ----
class Encoder(nn.Module):
    def __init__(self, ind, hid, emb):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ind, hid), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hid, emb), nn.GELU())
    def forward(self, x):
        return self.net(x)


# ---- k-NN retrieval lift (chunked) ----
def knn_lift(feat, labels, k=2, chunk=8000):
    feat = F.normalize(feat, dim=1).to(feat.device)
    N = feat.shape[0]; dev = feat.device
    lint = {n: i for i, n in enumerate(sorted(set(labels)))}
    lt = torch.tensor([lint[l] for l in labels], device=dev)
    match = tot = 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = feat[s:e] @ feat.T
        for j in range(e - s):
            sim[j, s + j] = -1e9
        top = sim.topk(k, dim=1).indices
        nl = lt[top]
        sm = top == torch.arange(s, e, device=dev).unsqueeze(1)
        nl[sm] = lt[N - 1] + 1
        ns = ~sm
        match += int((nl == lt[s:e].unsqueeze(1).expand_as(nl))[ns].sum().item())
        tot += int(ns.sum().item())
    pur = match / max(tot, 1)
    c = Counter(labels); base = sum((v / N) ** 2 for v in c.values())
    return round(pur, 5), round(base, 5), round(pur - base, 5)


def embed_all(model, X, batch=4096):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            out.append(model(X[s:s + batch]))
    return torch.cat(out, 0)


# ---- prototypical episode training ----
def run_episode(model, l2i, n_way, k_shot, q_query, device):
    classes = list(l2i.keys())
    cls = random.sample(classes, n_way)
    support_idx, query_idx, qy = [], [], []
    for ci, c in enumerate(cls):
        members = l2i[c]
        if len(members) < k_shot + q_query:
            members = members + random.choices(members, k=k_shot + q_query - len(members))
        perm = random.sample(range(len(members)), k_shot + q_query)
        sup = [members[i] for i in perm[:k_shot]]
        qry = [members[i] for i in perm[k_shot:]]
        support_idx += sup; query_idx += qry; qy += [ci] * len(qry)
    sx = X_g[support_idx]; qx = X_g[query_idx]
    se = model(sx); qe = model(qx)
    protos = se.view(n_way, k_shot, -1).mean(1)  # [n_way, emb]
    logits = -torch.cdist(qe, protos)  # euclidean neg -> similarity
    loss = F.cross_entropy(logits, torch.tensor(qy, device=device))
    return loss


def train_prototypical(model, l2i, args, device, eval_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for ep in range(1, args.episodes + 1):
        model.train()
        loss = run_episode(model, l2i, args.n_way, args.k_shot, args.q_query, device)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0:
            history.append(float(loss.detach()))
        if eval_fn and (ep % args.eval_every == 0 or ep == 1):
            lift = eval_fn(model)
            print(f"    episode {ep}/{args.episodes} loss={float(loss.detach()):.4f} {eval_fn.label}={lift:+.4f}")
    return history


X_g = None  # set in main; global for episode indexing


def main():
    global X_g
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"v11 PROTOTYPICAL | device={device} | metrics-only features")

    starts, met = read_matrix(args.metrics_safe, "m", args.limit)
    N = len(starts)
    print(f"  metrics {met.shape} | {N} items")
    X_g = met.to(device)
    range_band = read_labels(args.families, starts, "range_band")
    family_id = read_labels(args.families, starts, "coalescence_family_id")

    # ---- 1. RE-CLUSTER ----
    print(f"\nRe-clustering: k-means K={args.n_clusters} in metrics space...")
    cluster_labels = kmeans(X_g, args.n_clusters, seed=args.seed)
    csize = Counter(cluster_labels.tolist())
    print(f"  {len(csize)} clusters | sizes min={min(csize.values())} med={sorted(csize.values())[len(csize)//2]} max={max(csize.values())} avg={N/len(csize):.1f}")
    # magnitude check: modal range_band purity per cluster
    rb_int = {v: i for i, v in enumerate(sorted(set(range_band)))}
    rb_t = torch.tensor([rb_int[r] for r in range_band])
    cl_t = torch.tensor(cluster_labels.tolist())
    modal = 0.0
    for c in csize:
        m = cl_t == c
        modal += Counter(rb_t[m].tolist()).most_common(1)[0][1]
    modal_purity = modal / N
    nmi_clu_rb = nmi(cluster_labels, range_band)
    nmi_clu_fam = nmi(cluster_labels, family_id)
    nmi_fam_rb = nmi(family_id, range_band)
    print(f"  magnitude check: modal range_band purity/cluster = {modal_purity:.3f} (1.0=clusters ARE magnitude, lower=finer structure)")
    print(f"  NMI(cluster, range_band)={nmi_clu_rb:.3f}  NMI(cluster, family_id)={nmi_clu_fam:.3f}  NMI(family_id, range_band)={nmi_fam_rb:.3f}")

    results = {"recluster": {"n_clusters": len(csize), "sizes": {"min": min(csize.values()), "median": sorted(csize.values())[len(csize)//2], "max": max(csize.values()), "avg": round(N/len(csize),2)},
                             "modal_range_band_purity": round(modal_purity, 4),
                             "nmi_cluster_range_band": round(nmi_clu_rb, 4),
                             "nmi_cluster_family_id": round(nmi_clu_fam, 4),
                             "nmi_family_id_range_band": round(nmi_fam_rb, 4)}}

    # build label_to_indices for a target
    def l2i_from(labels, min_members=1, keep_idx=None):
        d = {}
        for i, l in enumerate(labels):
            if keep_idx is not None and i not in keep_idx:
                continue
            d.setdefault(l, []).append(i)
        if min_members > 1:
            d = {k: v for k, v in d.items() if len(v) >= min_members}
        return d

    # ---- 2a. PROTOTYPICAL on >=5-member original families ----
    print(f"\nPrototypical training on original families with >= {args.min_family} members...")
    fam_cnt = Counter(family_id)
    keep5 = sorted(i for i, f in enumerate(family_id) if fam_cnt[f] >= args.min_family)
    keep5_set = set(keep5)
    sub5 = [family_id[i] for i in keep5]   # aligned with keep5 (sorted)
    print(f"  {len(set(sub5))} families, {len(keep5)} rows")
    l2i_fam5 = l2i_from(family_id, args.min_family, keep5_set)
    raw5 = knn_lift(met[keep5].clone(), sub5, k=2)
    print(f"  baseline raw-metrics k-NN lift (k=2): {raw5[2]:+.4f}")

    def eval_fam5(model):
        emb = embed_all(model, X_g)
        return knn_lift(emb[keep5].cpu(), sub5, k=2)[2]
    eval_fam5.label = "fam5_lift"

    m1 = Encoder(met.shape[1], args.hidden_dims, args.embedding_dim).to(device)
    h1 = train_prototypical(m1, l2i_fam5, args, device, eval_fam5)
    emb_all1 = embed_all(m1, X_g)
    fam5_lift = knn_lift(emb_all1[keep5].cpu(), sub5, k=2)
    print(f"  -> prototypical fam5 embedding lift={fam5_lift[2]:+.4f} vs raw {raw5[2]:+.4f} (beats raw: {fam5_lift[2] > raw5[2]})")
    results["family5_prototypical"] = {"n_families": len(set(sub5)), "n_rows": len(keep5),
        "raw_metrics_lift": raw5[2], "prototypical_lift": fam5_lift[2],
        "beats_raw": bool(fam5_lift[2] > raw5[2]), "loss_history": h1}

    # ---- 2b. PROTOTYPICAL on re-clustered families ----
    print(f"\nPrototypical training on re-clustered families (K={args.n_clusters})...")
    l2i_clu = l2i_from(cluster_labels.tolist(), args.k_shot + args.q_query)
    sub_c = cluster_labels.tolist()
    raw_c = knn_lift(met.clone(), sub_c, k=2)
    print(f"  baseline raw-metrics k-NN lift by cluster (k=2): {raw_c[2]:+.4f} (circular: clusters defined in this space)")

    def eval_clu(model):
        emb = embed_all(model, X_g).cpu()
        return knn_lift(emb, sub_c, k=2)[2]
    eval_clu.label = "recluster_lift"

    m2 = Encoder(met.shape[1], args.hidden_dims, args.embedding_dim).to(device)
    h2 = train_prototypical(m2, l2i_clu, args, device, eval_clu)
    emb_all2 = embed_all(m2, X_g).cpu()
    clu_lift = knn_lift(emb_all2, sub_c, k=2)
    print(f"  -> prototypical recluster embedding lift={clu_lift[2]:+.4f} vs raw {raw_c[2]:+.4f}")
    results["recluster_prototypical"] = {"n_clusters": len(csize), "n_rows": N,
        "raw_metrics_lift": raw_c[2], "prototypical_lift": clu_lift[2],
        "beats_raw": bool(clu_lift[2] > raw_c[2]), "loss_history": h2}

    # ---- save ----
    metrics = {"dataset_type": "collatz_contrastive_embeddings_v11", "tool": "research/prototypical_train_v11.py",
               "status": "complete", "method": "prototypical_few_shot_plus_recluster", "device": str(device),
               "features": "metrics-only (m0-m31)", "embedding_count": N, "embedding_dims": args.embedding_dim,
               "episodes": args.episodes, "n_way": args.n_way, "k_shot": args.k_shot, "q_query": args.q_query,
               "results": results,
               "note": "family5: prototypical learned metric vs raw-metrics on >=5-member original families (non-circular). recluster: k-means target (raw lift is circular since clusters defined in metrics space; the meaningful signal is the diagnostics: modal_range_band_purity and NMI vs original family_id).",
               "outputs": {"family5_embeddings": "family5_embeddings.csv", "recluster_embeddings": "recluster_embeddings.csv"}}
    # save embeddings (full) for both
    for name, emb in [("family5_embeddings", emb_all1), ("recluster_embeddings", emb_all2)]:
        with (out / f"{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["n"] + [f"e{i}" for i in range(emb.shape[1])])
            for n, vec in zip(starts, emb.tolist()):
                w.writerow([n] + [f"{v:.9g}" for v in vec])
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(f"\nv11 done | outputs: {args.output_dir}")
    print(json.dumps({k: v for k, v in results.items()}, indent=2)[:1500])


if __name__ == "__main__":
    main()
