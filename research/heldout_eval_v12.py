#!/usr/bin/env python3
"""v12: held-out / disjoint evaluations (review fix #5).

The v9-v11 k-NN purity was computed over ALL rows (transductive / in-sample).
v12 adds the rigorous held-out versions:

A. FAMILY-DISJOINT (fine family_id >=5 members): split FAMILIES (not rows) into
   train / test; train prototypical on train families; evaluate k=2 retrieval
   lift on the held-out test families (families the model never saw). Run 3
   seeds for mean +/- std. This tests generalization to NEW families.

B. ROW-DISJOINT (coarse multi-task): split rows 80/20; train v10-style
   multi-task supervised on the train rows; evaluate k=2 retrieval lift on the
   held-out test rows per coarse label. Tests the coarse embedding is not just
   memorizing specific rows.

All lifts use the corrected k=2 metric (self excluded, then topk(2)).
"""
from __future__ import annotations
import argparse, csv, json, math, random
from collections import Counter
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

TASKS = ("range_band", "bit_length", "peak_ratio_bucket")

def parse_args():
    p = argparse.ArgumentParser(description="v12: held-out disjoint evaluations")
    p.add_argument("--metrics-safe", default="data/generated/ml_stratified/metrics_safe.csv")
    p.add_argument("--families", default="data/generated/ml_labels/families.csv")
    p.add_argument("--output-dir", default="data/generated/contrastive_v12")
    p.add_argument("--min-family", type=int, default=5)
    p.add_argument("--family-test-frac", type=float, default=0.2)
    p.add_argument("--n-way", type=int, default=30)
    p.add_argument("--k-shot", type=int, default=3)
    p.add_argument("--q-query", type=int, default=2)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--coarse-epochs", type=int, default=60)
    p.add_argument("--coarse-batch", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
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

class MultiTask(nn.Module):
    def __init__(s, ind, hid, emb, ncls):
        super().__init__(); s.enc = nn.Sequential(nn.Linear(ind, hid), nn.GELU(), nn.Dropout(0.2), nn.Linear(hid, emb), nn.GELU())
        s.heads = nn.ModuleList([nn.Linear(emb, c) for c in ncls])
    def embed(s, x): return s.enc(x)
    def forward(s, x):
        z = s.enc(x); return [h(z) for h in s.heads], z

def embed_all(model, X, batch=4096):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch): out.append(model.embed(X[s:s + batch]))
    return torch.cat(out, 0)

# ---- family-disjoint prototypical ----
def run_episode(model, l2i, n_way, k_shot, q_query, X):
    cls = random.sample(list(l2i.keys()), n_way)
    si, qi, qy = [], [], []
    for ci, c in enumerate(cls):
        m = l2i[c]
        if len(m) < k_shot + q_query: m = m + random.choices(m, k=k_shot + q_query - len(m))
        perm = random.sample(range(len(m)), k_shot + q_query)
        si += [m[i] for i in perm[:k_shot]]; qi += [m[i] for i in perm[k_shot:]]; qy += [ci] * q_query
    se = model(X[si]); qe = model(X[qi]); protos = se.view(n_way, k_shot, -1).mean(1)
    return F.cross_entropy(-torch.cdist(qe, protos), torch.tensor(qy, device=X.device))

def family_disjoint(args, met, family_id, device):
    fc = Counter(family_id)
    fams = sorted([f for f, c in fc.items() if c >= args.min_family])
    fam_to_idx = {f: [i for i, x in enumerate(family_id) if x == f] for f in fams}
    print(f"\n[A] FAMILY-DISJOINT fine family_id (>= {args.min_family} members): {len(fams)} families")
    results = []
    Xfull = met.to(device)
    for seed in range(args.seeds):
        random.seed(args.seed + seed); torch.manual_seed(args.seed + seed)
        rng = random.Random(args.seed + seed)
        rng.shuffle(fams)
        nte = int(len(fams) * args.family_test_frac)
        test_fams, train_fams = fams[:nte], fams[nte:]
        train_idx = [i for f in train_fams for i in fam_to_idx[f]]
        test_idx = [i for f in test_fams for i in fam_to_idx[f]]
        test_labels = [family_id[i] for i in test_idx]
        l2i = {f: fam_to_idx[f] for f in train_fams}
        # raw baseline on test (held-out) families
        raw = knn(met[test_idx].clone(), test_labels, k=2)[2]
        model = Encoder(met.shape[1], args.hidden_dims, args.embedding_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for ep in range(1, args.episodes + 1):
            model.train()
            loss = run_episode(model, l2i, args.n_way, args.k_shot, args.q_query, Xfull)
            opt.zero_grad(); loss.backward(); opt.step()
        emb = embed_all(model, Xfull)
        held = knn(emb[test_idx].cpu(), test_labels, k=2)[2]
        results.append({"seed": args.seed + seed, "n_train_families": len(train_fams),
                        "n_test_families": len(test_fams), "n_test_rows": len(test_idx),
                        "raw_metrics_lift": raw, "held_out_lift": held,
                        "in_sample_reference": 0.6399})
        print(f"  seed={args.seed+seed}: test_families={len(test_fams)} rows={len(test_idx)} | raw={raw:+.4f} held_out={held:+.4f} (in-sample was +0.6399)")
    lifts = [r["held_out_lift"] for r in results]
    mean = sum(lifts) / len(lifts); std = (sum((l - mean) ** 2 for l in lifts) / len(lifts)) ** 0.5
    summary = {"n_train_families": results[0]["n_train_families"], "n_test_families": results[0]["n_test_families"],
               "raw_metrics_lift_mean": results[0]["raw_metrics_lift"], "held_out_lift_mean": round(mean, 5),
               "held_out_lift_std": round(std, 5), "held_out_lift_ci95": [round(mean - 1.96 * std, 5), round(mean + 1.96 * std, 5)] if len(lifts) > 1 else [round(mean, 5), round(mean, 5)],
               "in_sample_reference_lift": 0.6399, "seeds": results}
    return summary

# ---- row-disjoint coarse multi-task ----
def row_disjoint_coarse(args, met, starts, labels_by_task, device):
    N = len(starts)
    print(f"\n[B] ROW-DISJOINT coarse multi-task: {N} rows, 80/20 split")
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g); ntr = int(N * 0.8)
    tr = perm[:ntr].to(device); te = perm[ntr:].cpu()   # tr on device (index X), te on cpu (index met/emb)
    cls_per = []
    Y = {}
    for t in TASKS:
        cs = sorted(set(labels_by_task[t])); cmap = {c: i for i, c in enumerate(cs)}
        Y[t] = torch.tensor([cmap[l] for l in labels_by_task[t]], dtype=torch.long); cls_per.append(len(cs))
    X = met.to(device); Ydev = {t: Y[t].to(device) for t in TASKS}
    model = MultiTask(met.shape[1], args.hidden_dims, args.embedding_dim, cls_per).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def lr_at(e): return args.lr * 0.5 * (1 + math.cos(math.pi * e / args.coarse_epochs))
    for ep in range(args.coarse_epochs):
        model.train(); p = tr[torch.randperm(len(tr), device=device)]; bs = args.coarse_batch
        for s in range(0, len(tr), bs):
            idx = p[s:s + bs]; logits, _ = model(X[idx])
            loss = sum(F.cross_entropy(logits[i], Ydev[TASKS[i]][idx]) for i in range(len(TASKS)))
            opt.zero_grad(); loss.backward(); opt.step()
        for gg in opt.param_groups: gg["lr"] = lr_at(ep + 1)
    emb = embed_all(model, X).cpu()
    out = {"n_train_rows": int(len(tr)), "n_test_rows": int(len(te)), "per_label": []}
    print("  label              raw(test)   held_out(test)  beats_raw")
    for t in TASKS:
        tl = [labels_by_task[t][i] for i in te.tolist()]
        raw = knn(met[te].clone(), tl, k=2)[2]
        held = knn(emb[te].clone(), tl, k=2)[2]
        out["per_label"].append({"label": t, "raw_metrics_lift_test": raw, "held_out_lift_test": held, "beats_raw": bool(held > raw)})
        print(f"  {t:<18}{raw:+.4f}    {held:+.4f}       {held > raw}")
    return out

def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"v12 HELD-OUT EVAL | device={device} | k=2 corrected metric")
    starts, met = read_matrix(args.metrics_safe, "m", args.limit)
    family_id = read_labels(args.families, starts, "coalescence_family_id")
    labels_by_task = {t: read_labels(args.families, starts, t) for t in TASKS}
    print(f"  metrics {met.shape} | {len(starts)} items")
    fam5 = family_disjoint(args, met, family_id, device)
    coarse = row_disjoint_coarse(args, met, starts, labels_by_task, device)
    metrics = {"dataset_type": "collatz_contrastive_embeddings_v12", "tool": "research/heldout_eval_v12.py",
               "status": "complete", "method": "held_out_disjoint_evaluations", "device": str(device),
               "metric": "k=2 neighbor-purity lift (corrected), held-out (queries/pool from the held-out split only)",
               "family_disjoint_fine": fam5, "row_disjoint_coarse": coarse,
               "note": "Family-disjoint trains prototypical on 80% of >=5-member families and evaluates retrieval on the held-out 20% of families (unseen in training), 3 seeds -> mean+/-std. Row-disjoint trains multi-task supervised on 80% of rows and evaluates retrieval on held-out 20% of rows per coarse label. These are true held-out (non-transductive) generalization tests, unlike the all-row k-NN in v9-v11."}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print("\nv12 done | outputs:", args.output_dir)
    print(json.dumps({"family_disjoint_held_out_lift_mean": fam5["held_out_lift_mean"], "std": fam5["held_out_lift_std"], "raw": fam5["raw_metrics_lift_mean"], "coarse": coarse}, indent=2)[:1200])

if __name__ == "__main__":
    main()
