#!/usr/bin/env python3
"""Train Collatz embeddings v9: SUPERVISED embedding (classification head).

Why v9 exists (the v1-v8 postmortem):
- v5-v8 were all self-supervised contrastive attempts to rediscover label
  structure WITHOUT using the labels. v7 ran 200 epochs and got +0.23% lift
  on range_band -- *worse* than the raw metrics k-NN baseline (+13.5%).
  The self-supervised contrastive setups kept collapsing.
- Diagnostics proved the features (m0-m31) contain overwhelming signal:
  a 40-epoch supervised MLP hits 95.8% accuracy on range_band, and its 64-d
  penultimate embedding reaches 98.3% neighbor purity (+92% lift).
- Conclusion: when labels are available, supervised metric learning is
  strictly more reliable than self-supervised contrastive. v9 uses the labels.

v9:
- Supervised: V1Encoder -> 64-d embedding -> linear classification head.
- Cross-entropy on the chosen label (range_band default). Optional auxiliary
  supervised-contrastive (SupCon) loss to push the embedding itself to cluster
  (off by default; CE alone already gives +92% lift).
- Metrics-only features by default (log_sketch s0-s127 has zero signal for
  range_band and would dilute). Other branches selectable via --feature-set.
- Cosine LR, evaluate every N epochs, same output contract as v7/v8
  (embeddings.csv / encoder.pt / metrics.json) so downstream tools work.
- Reports both held-out classification accuracy and embedding neighbor-purity
  lift (k=2) -- the metric directly comparable to v7's +0.23%.
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

ALL_BRANCHES = ("metrics", "shape", "parity", "residue")


# ------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="v9: supervised embedding via classification head")
    p.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    p.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    p.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    p.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    p.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    p.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v9")
    p.add_argument("--primary-label", default="range_band",
                   choices=("range_band", "bit_length", "peak_ratio_bucket"))
    p.add_argument("--feature-set", default="metrics",
                   choices=("metrics", "hybrid", "full"),
                   help="metrics=m0-m31 only (recommended); hybrid=+shape; full=+parity/residue")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--supcon-weight", type=float, default=0.0,
                   help="Auxiliary SupCon loss weight on the embedding (0 = pure CE)")
    p.add_argument("--supcon-temp", type=float, default=0.1)
    p.add_argument("--sequence-len", type=int, default=128)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--evaluate-every", type=int, default=5)
    p.add_argument("--test-frac", type=float, default=0.2)
    return p.parse_args()


# ------------------------------------------------------------------
# Data loading (proven-correct loaders reused from v7)
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


def read_token_sequence(path, starts, seq_len):
    wanted = set(starts)
    by_n = {}
    with path.open(newline="") as h:
        for row in csv.DictReader(h):
            n = int(row["n"])
            if n not in wanted:
                continue
            raw = (row.get("tokens", "") or "").split(";")
            vals = [float(int(v) & 0xffff) / 65535.0 for v in raw if v]
            arr = torch.tensor(vals if vals else [0.0], dtype=torch.float32)
            if arr.shape[0] < seq_len:
                arr = torch.cat([arr, torch.zeros(seq_len - arr.shape[0])])
            else:
                arr = arr[:seq_len]
            by_n[n] = arr
    return torch.stack([by_n[n] for n in starts])


def read_labels(path, starts, col):
    idx = {n: i for i, n in enumerate(starts)}
    labels = ["unknown"] * len(starts)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n"])
            if n in idx:
                labels[idx[n]] = row.get(col, "unknown")
    return labels


def active_branches(fs):
    return {"metrics": ("metrics",), "hybrid": ("metrics", "shape"), "full": ALL_BRANCHES}[fs]


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class SupervisedEncoder(nn.Module):
    """Encoder -> embedding -> linear classification head."""

    def __init__(self, input_dim, hidden_dim, embedding_dim, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim), nn.GELU(),
        )
        self.head = nn.Linear(embedding_dim, n_classes)

    def embed(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encoder(x))


# ------------------------------------------------------------------
# SupCon loss (optional auxiliary)
# ------------------------------------------------------------------
def supcon_loss(emb, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al.): all same-label pairs positive."""
    feat = F.normalize(emb, dim=1)
    sim = feat @ feat.T / temperature
    n = sim.shape[0]
    sim[torch.eye(n, dtype=torch.bool, device=sim.device)] = float("-inf")
    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    sim.masked_fill_(~same, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = same & ~torch.eye(n, dtype=torch.bool, device=sim.device)
    cnt = pos.sum(dim=1).clamp(min=1)
    return -(log_prob[pos] ).sum() / cnt.sum() if pos.any() else emb.sum() * 0.0


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_neighbors(embeddings, labels, k=2, chunk=8000):
    """Chunked k-NN purity so it never materializes the full [N, N] matrix."""
    feat = F.normalize(embeddings, dim=1)
    N = feat.shape[0]
    dev = feat.device
    lint = {n: i for i, n in enumerate(sorted(set(labels)))}
    lt = torch.tensor([lint[l] for l in labels], device=dev)
    match, tot = 0, 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sim = feat[s:e] @ feat.T                       # [chunk, N]
        sim[torch.arange(e - s), torch.arange(s, e, device=dev)] = float("-inf")
        top = sim.topk(k + 1, dim=1).indices           # [chunk, k+1]
        nl = lt[top]
        selfm = top == torch.arange(s, e, device=dev).unsqueeze(1)
        nl[selfm] = lt[N - 1] + 1                      # mark self as different
        ns = ~selfm
        match += int((nl == lt[s:e].unsqueeze(1).expand_as(nl))[ns].sum().item())
        tot += int(ns.sum().item())
    purity = match / max(tot, 1)
    counts = Counter(labels)
    base = sum((c / N) ** 2 for c in counts.values())
    return round(purity, 5), round(base, 5), round(purity - base, 5)


def embed_all(model, inputs, branch_names, N, batch_size, device):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            if e - s < 2:
                continue
            x = torch.cat([inputs[b][s:e] for b in branch_names], dim=1).to(device)
            out.append(model.embed(x))
    return torch.cat(out, dim=0)


# ------------------------------------------------------------------
def cosine_lr(epoch, total, base):
    return base * 0.5 * (1 + math.cos(math.pi * epoch / total))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    branch_names = active_branches(args.feature_set)

    print(f"v9 SUPERVISED | feature_set={args.feature_set} label={args.primary_label} "
          f"supcon_weight={args.supcon_weight} device={device}")

    # ---- Load data ----
    print("\nLoading data...")
    m_starts, m_data = read_matrix(Path(args.metrics_safe), "m", args.limit)
    inputs = {"metrics": m_data}
    input_dims = {"metrics": int(m_data.shape[1])}
    if "shape" in branch_names:
        s_starts, s_data = read_matrix(Path(args.log_sketch), "s", args.limit)
        assert len(s_starts) == len(m_starts), "shape row count mismatch"
        inputs["shape"] = s_data
        input_dims["shape"] = int(s_data.shape[1])
    if "parity" in branch_names:
        inputs["parity"] = read_token_sequence(Path(args.parity_runs), m_starts, args.sequence_len)
        input_dims["parity"] = int(inputs["parity"].shape[1])
    if "residue" in branch_names:
        inputs["residue"] = read_token_sequence(Path(args.transitions), m_starts, args.sequence_len)
        input_dims["residue"] = int(inputs["residue"].shape[1])

    labels = read_labels(Path(args.families), m_starts, args.primary_label)
    cls = sorted(set(labels))
    cmap = {c: i for i, c in enumerate(cls)}
    y = torch.tensor([cmap[l] for l in labels], dtype=torch.long)
    N = len(m_starts)
    counts = Counter(labels)
    print(f"  metrics {m_data.shape} | {N} items, {len(cls)} classes "
          f"(min={min(counts.values())}, max={max(counts.values())})")

    # ---- Build concatenated feature tensor on device ----
    X = torch.cat([inputs[b] for b in branch_names], dim=1).to(device)
    y_dev = y.to(device)
    total_input = sum(input_dims[b] for b in branch_names)

    # ---- Train/test split ----
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_tr = int(N * (1 - args.test_frac))
    tr, te = perm[:n_tr].to(device), perm[n_tr:].to(device)
    print(f"  split: train={len(tr)} test={len(te)} | input_dim={total_input}")

    # ---- Model ----
    model = SupervisedEncoder(total_input, args.hidden_dims, args.embedding_dim, len(cls)).to(device)
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- Train ----
    print(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}")
    losses, best_lift, best_acc = [], -999.0, 0.0

    for epoch in range(args.epochs):
        model.train()
        model.train()
        ep_loss, nb = 0.0, 0
        tr_perm = tr[torch.randperm(len(tr), device=device)]
        for s in range(0, len(tr), args.batch_size):
            e = min(s + args.batch_size, len(tr))
            idx = tr_perm[s:e]
            xb, yb = X[idx], y_dev[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if args.supcon_weight > 0 and xb.shape[0] > 1:
                loss = loss + args.supcon_weight * supcon_loss(model.embed(xb), yb, args.supcon_temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.detach()); nb += 1

        new_lr = cosine_lr(epoch + 1, args.epochs, args.lr)
        for g_ in optimizer.param_groups:
            g_["lr"] = new_lr
        losses.append(ep_loss / max(nb, 1))

        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X[te]).argmax(1) == y_dev[te]).float().mean().item()
            emb = embed_all(model, inputs, branch_names, N, args.batch_size, device)
            np_eval, rb_eval, lift = evaluate_neighbors(emb, labels, k=2)
            if lift > best_lift:
                best_lift = lift
                best_acc = acc
                torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "encoder_best.pt")
            print(f"  epoch {epoch+1}/{args.epochs} loss={losses[-1]:.4f} acc={acc*100:.1f}% "
                  f"lift={lift:+.4f} best_lift={best_lift:+.4f} lr={new_lr:.5f}")
        else:
            print(f"  epoch {epoch+1}/{args.epochs} loss={losses[-1]:.4f}")

    # ---- Final: load best, full evaluation ----
    print("\nFinal evaluation (best checkpoint)...")
    ckpt = torch.load(out_dir / "encoder_best.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        final_acc = (model(X[te]).argmax(1) == y_dev[te]).float().mean().item()
    emb_final = embed_all(model, inputs, branch_names, N, args.batch_size, device)
    np_final, rb_final, lift_final = evaluate_neighbors(emb_final, labels, k=2)

    # ---- Save embeddings ----
    emb_cpu = emb_final.cpu()
    with (out_dir / "embeddings.csv").open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["n", "label"] + [f"e{i}" for i in range(emb_cpu.shape[1])])
        for n, lab, vec in zip(m_starts, labels, emb_cpu.tolist()):
            w.writerow([n, lab] + [f"{v:.9g}" for v in vec])
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "encoder.pt")

    metrics = {
        "dataset_type": "collatz_contrastive_embeddings_v9",
        "tool": "research/contrastive_train_v9.py",
        "status": "complete",
        "method": "supervised_embedding",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": N,
        "feature_set": args.feature_set,
        "branches": list(branch_names),
        "primary_label": args.primary_label,
        "label_stats": {
            "unique_classes": len(cls),
            "min_per_class": min(counts.values()),
            "max_per_class": max(counts.values()),
            "avg_per_class": sum(counts.values()) / len(counts),
            "top5_classes": counts.most_common(5),
        },
        "embedding_dims": args.embedding_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "supcon_weight": args.supcon_weight,
        "test_accuracy": round(final_acc, 5),
        "loss_start": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_history": losses,
        "neighbor_purity": np_final,
        "random_baseline_purity": rb_final,
        "purity_lift": lift_final,
        "best_lift": best_lift,
        "comparison": {"v7_self_supervised_lift": 0.00233, "raw_metrics_knn_lift": 0.1346},
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nv9 SUPERVISED | label={args.primary_label} feature_set={args.feature_set}")
    print(f"  rows={N} test_acc={final_acc*100:.1f}% | embedding purity={np_final:.4f} "
          f"baseline={rb_final:.4f} lift={lift_final:+.4f} best_lift={best_lift:+.4f}")
    print(f"  vs v7 self-supervised lift=+0.0023 ; vs raw-knn lift=+0.1346")
    print(f"  outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
