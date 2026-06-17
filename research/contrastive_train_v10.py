#!/usr/bin/env python3
"""Train Collatz embeddings v10: MULTI-TASK supervised embedding.

Why v10 (the v9 follow-up):
- v9 (single-label supervised on range_band) beat raw metrics on range_band
  (+92.7% vs +13.5%) and the correlated bit_length (+73.4% vs +29.8%), but it
  LOST on peak_ratio_bucket (+23.7% vs raw +56.0%) and family_id. Supervising
  one label specializes the embedding to that label and sacrifices the others.
- v10 trains ONE shared encoder to predict THREE coarse labels at once
  (range_band, bit_length, peak_ratio_bucket) with summed cross-entropy. This
  forces the 64-d embedding to preserve all three kinds of structure, so it
  should beat raw metrics across the board -- a general-purpose retrieval
  embedding rather than a magnitude specialist.
- Method is still supervised (labels are available); this is the reliable path
  proven by v9, extended to multi-task.

v10:
- Shared V1Encoder -> 64-d embedding -> three linear classification heads.
- Loss = w_range*CE(range_band) + w_bit*CE(bit_length) + w_peak*CE(peak_ratio).
- Eval per epoch: per-label held-out accuracy + per-label embedding k-NN lift,
  reported against the raw-metrics k-NN baseline for each label.
- Cosine LR, evaluate every N epochs, same output contract as v7/v8/v9.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# The three coarse labels learned jointly. Order is fixed for stable indexing.
TASKS = ("range_band", "bit_length", "peak_ratio_bucket")


# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="v10: multi-task supervised embedding")
    p.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    p.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    p.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v10")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--evaluate-every", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.2)
    # Per-task loss weights (peak_ratio has more classes; give it a touch more)
    p.add_argument("--w-range", type=float, default=1.0)
    p.add_argument("--w-bit", type=float, default=1.0)
    p.add_argument("--w-peak", type=float, default=1.0)
    return p.parse_args()


# ------------------------------------------------------------------
# Data loading (proven-correct loaders reused from v7/v9)
# ------------------------------------------------------------------
def standardize(x):
    m, s = x.mean(dim=0), x.std(dim=0)
    return (x - m) / s.clamp(min=1e-6)


def read_matrix(path: Path, prefix: str, limit: int = 0):
    starts, rows = [], []
    with path.open(newline="") as h:
        r = csv.DictReader(h)
        fields = [f for f in r.fieldnames or [] if f.startswith(prefix)]
        for row in r:
            starts.append(int(row["n"]))
            rows.append([float(row[f]) for f in fields])
            if limit and len(starts) >= limit:
                break
    if not rows:
        raise RuntimeError(f"no rows loaded from {path}")
    return starts, standardize(torch.tensor(rows, dtype=torch.float32))


def read_labels(path, starts, col):
    idx = {n: i for i, n in enumerate(starts)}
    labels = ["unknown"] * len(starts)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if n in idx:
                labels[idx[n]] = row.get(col, "unknown")
    return labels


# ------------------------------------------------------------------
# Model: shared encoder + one head per task
# ------------------------------------------------------------------
class MultiTaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, n_classes_per_task):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim), nn.GELU(),
        )
        self.heads = nn.ModuleList([nn.Linear(embedding_dim, nc)
                                    for nc in n_classes_per_task])

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return [h(z) for h in self.heads], z


# ------------------------------------------------------------------
# Chunked k-NN purity (GPU-safe, never materializes [N,N])
# ------------------------------------------------------------------
def knn_lift(feat, labels, k=2, chunk=8000):
    feat = F.normalize(feat, dim=1)
    N = feat.shape[0]
    dev = feat.device
    lint = {n: i for i, n in enumerate(sorted(set(labels)))}
    lt = torch.tensor([lint[l] for l in labels], device=dev)
    match, tot = 0, 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = feat[s:e] @ feat.T
        sim[torch.arange(e - s), torch.arange(s, e, device=dev)] = float("-inf")
        top = sim.topk(k + 1, dim=1).indices
        nl = lt[top]
        selfm = top == torch.arange(s, e, device=dev).unsqueeze(1)
        nl[selfm] = lt[N - 1] + 1
        ns = ~selfm
        match += int((nl == lt[s:e].unsqueeze(1).expand_as(nl))[ns].sum().item())
        tot += int(ns.sum().item())
    purity = match / max(tot, 1)
    counts = Counter(labels)
    base = sum((c / N) ** 2 for c in counts.values())
    return round(purity, 5), round(base, 5), round(purity - base, 5)


def embed_all(model, X, N, batch_size, device):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            if e - s < 2:
                continue
            out.append(model.embed(X[s:e]))
    return torch.cat(out, dim=0)


def cosine_lr(epoch, total, base):
    return base * 0.5 * (1 + math.cos(math.pi * epoch / total))


# ------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"v10 MULTI-TASK | tasks={TASKS} device={device}")

    # ---- Load features + all three label sets ----
    print("\nLoading data...")
    starts, m_data = read_matrix(Path(args.metrics_safe), "m", args.limit)
    N = len(starts)
    print(f"  metrics {m_data.shape} | {N} items")

    label_sets = {}
    cls_per_task = []
    label_ids = {}
    for task in TASKS:
        labels = read_labels(Path(args.families), starts, task)
        cls = sorted(set(labels))
        cmap = {c: i for i, c in enumerate(cls)}
        label_ids[task] = torch.tensor([cmap[l] for l in labels], dtype=torch.long)
        label_sets[task] = labels
        cls_per_task.append(len(cls))
        counts = Counter(labels)
        print(f"  {task}: {len(cls)} classes (min={min(counts.values())}, max={max(counts.values())})")

    X = m_data.to(device)
    Y = {t: label_ids[t].to(device) for t in TASKS}
    total_input = m_data.shape[1]

    # ---- Train/test split ----
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(N * (1 - args.test_frac))
    tr, te = perm[:n_tr].to(device), perm[n_tr:].to(device)
    print(f"  split: train={len(tr)} test={len(te)} | input_dim={total_input}")

    # ---- Model ----
    model = MultiTaskEncoder(total_input, args.hidden_dims, args.embedding_dim, cls_per_task).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = [args.w_range, args.w_bit, args.w_peak]

    # ---- Train ----
    print(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}")
    losses = []
    best_sum_lift = -999.0
    best_snapshot = None

    for epoch in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        tr_perm = tr[torch.randperm(len(tr), device=device)]
        for s in range(0, len(tr), args.batch_size):
            e = min(s + args.batch_size, len(tr))
            idx = tr_perm[s:e]
            xb = X[idx]
            logits_list, _ = model(xb)
            loss = sum(w * F.cross_entropy(logits_list[i], Y[TASKS[i]][idx])
                       for i, w in enumerate(weights))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.detach())
            nb += 1

        new_lr = cosine_lr(epoch + 1, args.epochs, args.lr)
        for g_ in optimizer.param_groups:
            g_["lr"] = new_lr
        losses.append(ep_loss / max(nb, 1))

        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                accs = {}
                for i, t in enumerate(TASKS):
                    accs[t] = (model(X[te])[0][i].argmax(1) == Y[t][te]).float().mean().item()
            emb = embed_all(model, X, N, args.batch_size, device)
            lifts = {t: knn_lift(emb, label_sets[t]) for t in TASKS}
            sum_lift = sum(lifts[t][2] for t in TASKS)
            line = (f"  epoch {epoch+1}/{args.epochs} loss={losses[-1]:.4f} "
                    f"accs=[{', '.join(f'{t[:4]}={accs[t]*100:.1f}%' for t in TASKS)}] "
                    f"lifts=[{', '.join(f'{t[:4]}={lifts[t][2]:+.3f}' for t in TASKS)}] "
                    f"sum={sum_lift:+.4f}")
            print(line)
            if sum_lift > best_sum_lift:
                best_sum_lift = sum_lift
                best_snapshot = {"epoch": epoch + 1, "accs": accs, "lifts": lifts}
                torch.save({"model_state": model.state_dict(), "args": vars(args)},
                           out_dir / "encoder_best.pt")
        else:
            print(f"  epoch {epoch+1}/{args.epochs} loss={losses[-1]:.4f}")

    # ---- Final: load best, full per-label evaluation ----
    print("\nFinal evaluation (best checkpoint by summed lift)...")
    ckpt = torch.load(out_dir / "encoder_best.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    emb_final = embed_all(model, X, N, args.batch_size, device)

    # Raw-metrics baseline for each label (apples-to-apples k-NN on the 32-d metrics)
    raw_emb = m_data.to(device)
    raw_lifts = {t: knn_lift(raw_emb, label_sets[t]) for t in TASKS}
    v10_lifts = {t: knn_lift(emb_final, label_sets[t]) for t in TASKS}
    v10_accs = {}
    with torch.no_grad():
        for i, t in enumerate(TASKS):
            v10_accs[t] = (model(X[te])[0][i].argmax(1) == Y[t][te]).float().mean().item()

    # ---- Save embeddings ----
    emb_cpu = emb_final.cpu()
    with (out_dir / "embeddings.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["n"] + [f"e{i}" for i in range(emb_cpu.shape[1])])
        for n, vec in zip(starts, emb_cpu.tolist()):
            w.writerow([n] + [f"{v:.9g}" for v in vec])
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "encoder.pt")

    # ---- metrics.json ----
    comparison = []
    for t in TASKS:
        comparison.append({
            "label": t,
            "n_classes": len(set(label_sets[t])),
            "random_baseline_purity": v10_lifts[t][1],
            "raw_metrics_knn_lift": raw_lifts[t][2],
            "v10_embedding_lift": v10_lifts[t][2],
            "v10_test_accuracy": round(v10_accs[t], 5),
            "v10_beats_raw_metrics": bool(v10_lifts[t][2] > raw_lifts[t][2]),
        })
    metrics = {
        "dataset_type": "collatz_contrastive_embeddings_v10",
        "tool": "research/contrastive_train_v10.py",
        "status": "complete",
        "method": "multi_task_supervised_embedding",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": N,
        "embedding_dims": args.embedding_dim,
        "tasks": list(TASKS),
        "loss_weights": {"range_band": args.w_range, "bit_length": args.w_bit, "peak_ratio_bucket": args.w_peak},
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_start": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_history": losses,
        "best_epoch": best_snapshot["epoch"] if best_snapshot else None,
        "best_sum_lift": round(best_sum_lift, 5),
        "comparison": comparison,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Console summary ----
    print(f"\nv10 MULTI-TASK | embedding_dim={args.embedding_dim} best_epoch={best_snapshot['epoch']}")
    print(f"  {'label':<20}{'classes':>8}{'baseline':>10}{'raw-kNN':>12}{'v10-emb':>12}{'acc':>9}{'verdict':>12}")
    print("  " + "-" * 72)
    for t in TASKS:
        v = "v10 WINS" if v10_lifts[t][2] > raw_lifts[t][2] else "raw WINS"
        print(f"  {t:<20}{len(set(label_sets[t])):>8}{v10_lifts[t][1]:>10.4f}"
              f"{raw_lifts[t][2]:>+12.4f}{v10_lifts[t][2]:>+12.4f}{v10_accs[t]*100:>8.1f}%{v:>12}")
    print(f"\n  summed lift (best epoch {best_snapshot['epoch']}): {best_sum_lift:+.4f}")
    print(f"  outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
