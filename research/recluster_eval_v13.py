#!/usr/bin/env python3
"""v13: non-circular re-cluster evaluation + cluster characterization.

Closes the open item from v11: the v11 re-cluster retrieval number was circular
(clusters were defined in metrics space, so raw-metrics k-NN purity by cluster is
high by construction). v13 does the non-circular versions and characterizes what
the clusters actually are.

A. Re-cluster: k-means in metrics space, K=2000 (~50 members each), all 100k rows.
B. Characterize the clusters: NMI(cluster, each known label) and average modal
   purity per label -- are the clusters magnitude, shape, or fine structure?
C. Cluster-disjoint held-out: split CLUSTERS into train/test (80/20); train
   prototypical on train clusters; evaluate k=2 retrieval on held-out test
   clusters (non-circular). 3 seeds.
D. General-purpose embedding: train prototypical on ALL clusters, emit the 64-d
   embedding for all 100k, then evaluate it cross-target on the ORIGINAL fine
   family_id (>=5 members) with a family-disjoint held-out split -- does an
   embedding trained only on cluster-proxy labels recover the original fine
   families? This is the "single general-purpose fine-structure embedding" test.

k=2 corrected metric throughout.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

def parse_args():
    p = argparse.ArgumentParser(description="v13: non-circular re-cluster eval + characterization")
    p.add_argument("--metrics-safe", default="data/generated/ml_stratified/metrics_safe.csv")
    p.add_argument("--families", default="data/generated/ml_labels/families.csv")
    p.add_argument("--output-dir", default="data/generated/contrastive_v13")
    p.add_argument("--n-clusters", type=int, default=2000)
    p.add_argument("--min-family", type=int, default=5)
    p.add_argument("--n-way", type=int, default=50)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=5)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--limit", type=int, default=100000)
    return p.parse_args()

def standardize(x):
    m, s = x.mean(0), x.std(0); return (x - m) / s.clamp(min=1e-6)
def read_matrix(path, prefix, limit=0):
    starts, rows = [], []
    with open(path, newline="") as h:
        r = csv.DictReader(h); cols = [c for c in r.fieldnames or [] if c.startswith(prefix)]
        for row in r:
            starts.append(int(row["n"])); rows.append([float(row[c]) for c in cols])
            if limit and len(starts) >= limit: break
    return starts, standardize(torch.tensor(rows, dtype=torch.float32))
def read_labels(path, starts, col):
    idx = {n: i for i, n in enumerate(starts)}; lab = ["?"] * len(starts)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if n in idx: lab[idx[n]] = row.get(col, "unknown")
    return lab

def kmeans(X, k, iters=40, seed=0, chunk=8000):
    torch.manual_seed(seed); dev = X.device
    centroids = X[torch.randperm(X.shape[0], device=dev)[:k]].clone()
    labels = torch.zeros(X.shape[0], dtype=torch.long, device=dev)
    for _ in range(iters):
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0]); labels[s:e] = torch.cdist(X[s:e], centroids).argmin(1)
        for c in range(k):
            m = labels == c
            if m.any(): centroids[c] = X[m].mean(0)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0]); labels[s:e] = torch.cdist(X[s:e], centroids).argmin(1)
    return labels.cpu()

def nmi(a, b):
    a = a.tolist() if torch.is_tensor(a) else list(a); b = b.tolist() if torch.is_tensor(b) else list(b)
    n = len(a); ua, ub = sorted(set(a)), sorted(set(b))
    ca = {v: i for i, v in enumerate(ua)}; cb = {v: i for i, v in enumerate(ub)}
    joint = torch.zeros(len(ua), len(ub))
    for x, y in zip(a, b): joint[ca[x], cb[y]] += 1
    joint = joint / n; pa = joint.sum(1); pb = joint.sum(0); denom = pa[:, None] * pb[None, :]
    mask = (joint > 0) & (denom > 0)
    mi = (joint[mask] * torch.log(joint[mask] / denom[mask])).sum().item()
    ha = -(pa[pa > 0] * torch.log(pa[pa > 0])).sum().item(); hb = -(pb[pb > 0] * torch.log(pb[pb > 0])).sum().item()
    return 0.0 if (ha <= 0 or hb <= 0) else mi / math.sqrt(ha * hb)

def knn(feat, labels, k=2, chunk=8000):
    feat = F.normalize(feat, dim=1).to(feat.device); N = feat.shape[0]; dev = feat.device
    lint = {n: i for i, n in enumerate(sorted(set(labels)))}
    lt = torch.tensor([lint[l] for l in labels], device=dev); match = tot = 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N); sim = feat[s:e] @ feat.T
        for j in range(e - s): sim[j, s + j] = -1e9
        top = sim.topk(k, dim=1).indices; nl = lt[top]
        ns = ~(top == torch.arange(s, e, device=dev).unsqueeze(1))
        match += int((nl == lt[s:e].unsqueeze(1).expand_as(nl))[ns].sum()); tot += int(ns.sum())
    pur = match / max(tot, 1); c = Counter(labels); base = sum((v / N) ** 2 for v in c.values())
    return round(pur, 5), round(base, 5), round(pur - base, 5)

class Encoder(nn.Module):
    def __init__(s, ind, hid, emb):
        super().__init__(); s.net = nn.Sequential(nn.Linear(ind, hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, emb), nn.GELU())
    def embed(s, x): return s.net(x)
    def forward(s, x): return s.net(x)

def embed_all(model, X, batch=4096):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch): out.append(model.embed(X[s:s + batch]))
    return torch.cat(out, 0)

def run_episode(model, l2i, n_way, k_shot, q_query, X):
    cls = random.sample(list(l2i.keys()), n_way); si, qi, qy = [], [], []
    for ci, c in enumerate(cls):
        m = l2i[c]
        if len(m) < k_shot + q_query: m = m + random.choices(m, k=k_shot + q_query - len(m))
        perm = random.sample(range(len(m)), k_shot + q_query)
        si += [m[i] for i in perm[:k_shot]]; qi += [m[i] for i in perm[k_shot:]]; qy += [ci] * q_query
    se = model(X[si]); qe = model(X[qi]); protos = se.view(n_way, k_shot, -1).mean(1)
    return F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(qy, device=X.device))

def train_proto(model, l2i, args, X):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(1, args.episodes + 1):
        model.train(); loss = run_episode(model, l2i, args.n_way, args.k_shot, args.q_query, X)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"v13 RECLUSTER EVAL | device={device} | k=2 corrected")
    starts, met = read_matrix(args.metrics_safe, "m", args.limit); N = len(starts)
    X = met.to(device)
    range_band = read_labels(args.families, starts, "range_band")
    bit_length = read_labels(args.families, starts, "bit_length")
    peak_ratio = read_labels(args.families, starts, "peak_ratio_bucket")
    family_id = read_labels(args.families, starts, "coalescence_family_id")
    print(f"  metrics {met.shape} | {N} items")

    # A. re-cluster
    print(f"\n[A] Re-cluster k-means K={args.n_clusters}...")
    clu = kmeans(X, args.n_clusters, seed=args.seed)
    clu_list = clu.tolist()
    csz = Counter(clu_list)
    print(f"  {len(csz)} clusters | min={min(csz.values())} med={sorted(csz.values())[len(csz)//2]} max={max(csz.values())} avg={N/len(csz):.1f}")

    # B. characterize clusters vs known labels
    print("\n[B] Cluster characterization (what do the clusters capture?):")
    char = {}
    for name, lab in [("range_band", range_band), ("bit_length", bit_length), ("peak_ratio_bucket", peak_ratio), ("coalescence_family_id", family_id)]:
        nm = nmi(clu, lab)
        # average modal purity: for each cluster, fraction in its modal label; average across clusters (size-weighted)
        cl_t = torch.tensor(clu_list); lab_int = {v: i for i, v in enumerate(sorted(set(lab)))}; lab_t = torch.tensor([lab_int[l] for l in lab])
        modal = 0
        for c in csz:
            m = cl_t == c; modal += Counter(lab_t[m].tolist()).most_common(1)[0][1]
        mp = modal / N
        char[name] = {"nmi_cluster_label": round(nm, 4), "size_weighted_modal_purity": round(mp, 4)}
        print(f"  {name:<22} NMI={nm:.3f}  modal_purity={mp:.3f}")

    # C. cluster-disjoint held-out (non-circular)
    print(f"\n[C] Cluster-disjoint held-out (train on 80% clusters, retrieve on held-out 20% clusters), {args.seeds} seeds:")
    cl_to_idx = {c: [i for i, x in enumerate(clu_list) if x == c] for c in csz}
    Cres = []
    for seed in range(args.seeds):
        random.seed(args.seed + seed); torch.manual_seed(args.seed + seed)
        fams = sorted(csz); rng = random.Random(args.seed + seed); rng.shuffle(fams)
        nte = int(len(fams) * 0.2); test_c, train_c = fams[:nte], fams[nte:]
        train_idx = [i for c in train_c for i in cl_to_idx[c]]; test_idx = [i for c in test_c for i in cl_to_idx[c]]
        tl = [clu_list[i] for i in test_idx]
        raw = knn(met[test_idx].clone(), tl, k=2)[2]
        l2i = {c: cl_to_idx[c] for c in train_c}
        model = Encoder(met.shape[1], args.hidden_dims, args.embedding_dim).to(device)
        train_proto(model, l2i, args, X)
        emb = embed_all(model, X)
        held = knn(emb[test_idx].cpu(), tl, k=2)[2]
        Cres.append({"seed": args.seed + seed, "n_train_clusters": len(train_c), "n_test_clusters": len(test_c), "raw": raw, "held_out": held})
        print(f"  seed={args.seed+seed}: test_clusters={len(test_c)} rows={len(test_idx)} raw={raw:+.4f} held_out={held:+.4f}")
    lifts = [r["held_out"] for r in Cres]; mean = sum(lifts)/len(lifts); std = (sum((l-mean)**2 for l in lifts)/len(lifts))**0.5
    Csummary = {"raw_mean": round(sum(r["raw"] for r in Cres)/len(Cres),5), "held_out_mean": round(mean,5),
                "held_out_std": round(std,5), "held_out_ci95": [round(mean-1.96*std,5), round(mean+1.96*std,5)], "seeds": Cres}

    # D. general-purpose embedding: train on ALL clusters, eval cross-target on held-out fine family_id>=5
    print(f"\n[D] General-purpose embedding: train prototypical on ALL clusters, eval cross-target on held-out fine family_id (>=5)...")
    random.seed(args.seed); torch.manual_seed(args.seed)
    l2i_all = cl_to_idx
    model = Encoder(met.shape[1], args.hidden_dims, args.embedding_dim).to(device)
    train_proto(model, l2i_all, args, X)
    emb_all = embed_all(model, X).cpu()
    # save the general-purpose embedding
    with (out / "general_purpose_embeddings.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["n"] + [f"e{i}" for i in range(emb_all.shape[1])])
        for n_, vec in zip(starts, emb_all.tolist()): w.writerow([n_] + [f"{v:.9g}" for v in vec])
    # family-disjoint held-out eval using the cluster-trained embedding
    fc = Counter(family_id); fams5 = sorted([f for f, c in fc.items() if c >= args.min_family])
    f5_to_idx = {f: [i for i, x in enumerate(family_id) if x == f] for f in fams5}
    Dseeds = []
    for seed in range(args.seeds):
        rng = random.Random(args.seed + seed); ff = fams5[:]; rng.shuffle(ff)
        nte = int(len(ff) * 0.2); test_f = ff[:nte]
        test_idx = [i for f in test_f for i in f5_to_idx[f]]; tl = [family_id[i] for i in test_idx]
        raw = knn(met[test_idx].clone(), tl, k=2)[2]
        held = knn(emb_all[test_idx].clone(), tl, k=2)[2]
        Dseeds.append({"seed": args.seed + seed, "n_test_families": len(test_f), "raw": raw, "held_out": held})
        print(f"  seed={args.seed+seed}: test_families={len(test_f)} raw={raw:+.4f} cluster-trained_held_out={held:+.4f}")
    dlifts = [r["held_out"] for r in Dseeds]; dmean = sum(dlifts)/len(dlifts); dstd = (sum((l-dmean)**2 for l in dlifts)/len(dlifts))**0.5
    Dsummary = {"setup": "embedding trained ONLY on cluster-proxy labels (never saw family_id); evaluated family-disjoint on original fine family_id >=5",
                "raw_mean": round(sum(r["raw"] for r in Dseeds)/len(Dseeds),5), "held_out_mean": round(dmean,5),
                "held_out_std": round(dstd,5), "held_out_ci95": [round(dmean-1.96*dstd,5), round(dmean+1.96*dstd,5)], "seeds": Dseeds}

    metrics = {"dataset_type":"collatz_contrastive_embeddings_v13","tool":"research/recluster_eval_v13.py","status":"complete",
               "method":"non_circular_recluster_eval_plus_characterization","device":str(device),"metric":"k=2 neighbor-purity lift (corrected)",
               "recluster":{"n_clusters":len(csz),"sizes":{"min":min(csz.values()),"median":sorted(csz.values())[len(csz)//2],"max":max(csz.values()),"avg":round(N/len(csz),2)}},
               "characterization":char,
               "cluster_disjoint_heldout":Csummary,
               "general_purpose_cross_target":Dsummary,
               "note":"C is non-circular (train/test clusters disjoint). D trains only on cluster-proxy labels and tests cross-target on the original fine family_id (family-disjoint). The v11 recluster number was circular; these are not."}
    json.dump(metrics, open(out/"metrics.json","w"), indent=2)
    print("\nv13 done | outputs:", args.output_dir)
    print(json.dumps({"characterization":char,"cluster_disjoint_heldout":Csummary["held_out_mean"],"cross_target_heldout":Dsummary["held_out_mean"]},indent=2)[:900])

if __name__ == "__main__":
    main()
