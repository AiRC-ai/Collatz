#!/usr/bin/env python3
"""Train Collatz embeddings v8: hybrid features + InfoNCE loss.

Design from v7 lessons:
- v7 failed: metrics-only + triplet loss + 0.2 metrics-preservation anchor
  collapsed the embeddings. Loss dropped 0.095 -> 0.0016 but lift stayed at
  0.23% (neighbor purity 0.0648 vs 0.0625 random baseline).
- v8 hypothesis: switch to hybrid features + standard InfoNCE loss and drop the
  metrics-preservation anchor (which tethered the embedding to raw metrics
  instead of letting it learn latent family structure).

v8:
- Hybrid feature set by default (metrics m0-m31 + log_sketch s0-s127 = 160-dim).
  Optionally full (add parity/residue token sequences) via --feature-set.
- Standard InfoNCE / NT-Xent loss with a single label-sampled positive per
  anchor and the full dataset as the negative pool. Positives are identified
  by GLOBAL INDEX (the v8 crash bug passed embedding vectors as indices).
- No auxiliary metrics-preservation loss by default; opt in with --metrics-loss.
- The negative pool is recomputed each epoch so it stays fresh.
- Range-band labels (16 classes, 6,250 each), cosine LR, evaluate every 10
  epochs.

The previous v8 was rewritten because its data loading read nonexistent named
CSV columns (real schema: metrics_safe.csv has m0..m31, log_sketch.csv has
s0..s127, parity_runs.csv / residue_transitions_mod32.csv have a single
`tokens` column). It now reuses v7's proven read_matrix / read_token_sequence
/ read_families loaders.
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
# Argument parsing
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="v8: hybrid features + InfoNCE loss"
    )
    parser.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    parser.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v8")
    parser.add_argument("--primary-label", default="range_band",
                        choices=("range_band", "bit_length"),
                        help="Label column for positive pairs")
    parser.add_argument("--feature-set", default="hybrid",
                        choices=("hybrid", "full", "metrics", "shape",
                                 "parity-sequence", "residue-sequence"),
                        help="hybrid=metrics+shape; full=all four branches")
    parser.add_argument("--loss", default="info_nce", choices=("info_nce", "triplet"),
                        help="Contrastive loss. triplet needs hard-negative mining.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--metrics-loss-weight", type=float, default=0.2,
                        help="Weight for optional metrics-preservation anchor")
    parser.add_argument("--metrics-loss", action="store_true",
                        help="Enable auxiliary metrics-preservation loss (off by default)")
    parser.add_argument("--hard-negatives", type=int, default=50,
                        help="Hard negatives per anchor (triplet loss only)")
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=192)
    parser.add_argument("--limit", type=int, default=100000,
                        help="Limit dataset to N rows (0 = full)")
    parser.add_argument("--evaluate-every", type=int, default=10,
                        help="Evaluate every N epochs")
    return parser.parse_args()


# ------------------------------------------------------------------
# Data loading (reused from v7: proven-correct against real schema)
# ------------------------------------------------------------------
def standardize(x):
    """Column-wise standardization."""
    m, s = x.mean(dim=0), x.std(dim=0)
    s = s.clamp(min=1e-6)
    return (x - m) / s


def read_matrix(path: Path, prefix: str, limit: int = 0):
    """Read numeric columns from CSV with given prefix (e.g. 'm' -> m0..m31)."""
    starts = []
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [f for f in reader.fieldnames or [] if f.startswith(prefix)]
        for row in reader:
            starts.append(int(row["n"]))
            rows.append([float(row[f]) for f in fields])
            if limit > 0 and len(starts) >= limit:
                break
    if not rows:
        raise RuntimeError(f"no rows loaded from {path}")
    return starts, standardize(torch.tensor(rows, dtype=torch.float32))


def read_token_sequence(path, starts, seq_len):
    """Load token sequences, padding or truncating to seq_len."""
    wanted = set(starts)
    by_n = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n not in wanted:
                continue
            raw = (row.get("tokens", "") or "").split(";")
            values = [float(int(v) & 0xffff) / 65535.0 for v in raw if v]
            arr = torch.tensor(values if values else [0.0], dtype=torch.float32)
            if arr.shape[0] < seq_len:
                arr = torch.cat([arr, torch.zeros(seq_len - arr.shape[0], dtype=torch.float32)], dim=0)
            else:
                arr = arr[:seq_len]
            by_n[n] = arr
    sequences = []
    for n in starts:
        if n not in by_n:
            raise RuntimeError(f"missing sequence for n={n}")
        sequences.append(by_n[n])
    return torch.stack(sequences)


def read_families(path, starts, primary_label):
    """Load family labels aligned to `starts` by the `n` key."""
    n_to_idx = {n: i for i, n in enumerate(starts)}
    labels = ["unknown"] * len(starts)
    label_to_indices = {}
    counts = Counter()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["n"])
            if n not in n_to_idx:
                continue
            idx = n_to_idx[n]
            label = row.get(primary_label, "unknown")
            labels[idx] = label
            label_to_indices.setdefault(label, []).append(idx)
            counts[label] += 1
    return labels, label_to_indices, counts


# ------------------------------------------------------------------
# Hard negative mining (triplet loss only; InfoNCE uses the full pool)
# ------------------------------------------------------------------
def mine_hard_negatives_raw_metrics(metrics_tensor, label_to_indices, k=50, chunk_size=10000):
    n = metrics_tensor.shape[0]
    device = metrics_tensor.device

    label_names = sorted(label_to_indices.keys())
    name_to_int = {name: i for i, name in enumerate(label_names)}
    label_arr = torch.zeros(n, dtype=torch.long, device=device)
    for label, indices in label_to_indices.items():
        label_arr[torch.tensor(indices, dtype=torch.long, device=device)] = name_to_int[label]

    metrics_normed = F.normalize(metrics_tensor, dim=1)
    neg_indices = torch.full((n, k), -1, dtype=torch.long, device=device)

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_sz = chunk_end - chunk_start
        chunk_sim = metrics_normed[chunk_start:chunk_end] @ metrics_normed.T
        chunk_sim[torch.arange(chunk_sz), torch.arange(chunk_start, chunk_end, device=device)] = float("-inf")
        chunk_labels = label_arr[chunk_start:chunk_end]
        same_label_mask = chunk_labels.unsqueeze(1) == label_arr.unsqueeze(0)
        chunk_sim[same_label_mask] = float("-inf")
        k_actual = min(k, n - 1)
        _, topk_indices = chunk_sim.topk(k_actual, dim=1)
        neg_indices[chunk_start:chunk_end] = topk_indices
    return neg_indices


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class V1Encoder(nn.Module):
    """Single-branch encoder: concat selected branches, project to embedding."""

    def __init__(self, branch_names, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.branch_names = branch_names
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, branches, dropout=0.0):
        x = torch.cat([branches[name] for name in self.branch_names], dim=1)
        if dropout > 0 and self.training:
            x = F.dropout(x, dropout, training=True)
        return self.net(x)


class MetricsProjectionHead(nn.Module):
    """Small MLP to project embedding toward raw metrics for auxiliary loss."""

    def __init__(self, embedding_dim, metric_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, metric_dim),
        )

    def forward(self, x):
        return self.projection(x)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
def info_nce_loss(z_anchor, pos_indices, all_embeddings, temperature=0.15,
                  anchor_self_indices=None):
    """Standard InfoNCE / NT-Xent with one positive per anchor.

    z_anchor:             [B, d] anchor embeddings
    pos_indices:           [B] GLOBAL index of each anchor's positive in the pool
    all_embeddings:        [N, d] candidate/negative pool
    anchor_self_indices:   [B] global index of each anchor (masked out of the
                           denominator so an item is not its own negative)
    """
    z_anchor = F.normalize(z_anchor, dim=1)
    pool = F.normalize(all_embeddings, dim=1)
    sim = z_anchor @ pool.T / temperature  # [B, N]
    if anchor_self_indices is not None:
        sim[torch.arange(sim.shape[0]), anchor_self_indices] = float("-inf")
    log_prob = F.log_softmax(sim, dim=1)
    pos_log_prob = log_prob[torch.arange(sim.shape[0]), pos_indices]
    return -pos_log_prob.mean()


def triplet_loss_with_hard_negatives(z_anchor, z_positive, hard_neg_indices, z_all, temperature=0.07):
    margin = 0.3
    pos_dist = 1.0 - F.cosine_similarity(z_anchor, z_positive, dim=1)
    neg_sim = F.cosine_similarity(z_anchor.unsqueeze(1), z_all[hard_neg_indices], dim=2)
    hardest_neg_dist = (1.0 - neg_sim).min(dim=1).values
    return F.relu(pos_dist - hardest_neg_dist + margin).mean()


def cosine_preservation_loss(embedding, target_raw_metrics, temperature=1.0):
    emb_norm = F.normalize(embedding, dim=1)
    tgt_norm = F.normalize(target_raw_metrics, dim=1)
    sim = torch.sum(emb_norm * tgt_norm, dim=1) / temperature
    return 1.0 - sim.mean()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_neighbors(embeddings, labels, k=2):
    """Measure neighbor purity and lift over same-label fraction.

    Computes the full [N, N] similarity matrix on the provided device.
    """
    sim = torch.matmul(embeddings, embeddings.T)
    N = sim.shape[0]
    sim[torch.eye(N, dtype=torch.bool, device=sim.device)] = float("-inf")
    top_k_idx = sim.topk(k + 1, dim=1).indices  # [N, k+1]
    label_to_int = {name: i for i, name in enumerate(sorted(set(labels)))}
    labels_t = torch.tensor([label_to_int[l] for l in labels], device=embeddings.device)
    neighbor_labels = labels_t[top_k_idx]
    self_mask = top_k_idx == torch.arange(N, device=embeddings.device).unsqueeze(1)
    neighbor_labels[self_mask] = labels_t[N - 1] + 1  # mark self as different
    not_self = ~self_mask
    num_same = not_self.sum().item()
    num_match = (neighbor_labels[not_self] == labels_t.unsqueeze(1).expand_as(neighbor_labels)[not_self]).sum().item()
    neighbor_purity = num_match / max(num_same, 1)
    counts = Counter(labels)
    random_baseline = sum((count / len(labels)) ** 2 for count in counts.values())
    purity_lift = neighbor_purity - random_baseline
    return round(neighbor_purity, 5), round(random_baseline, 5), round(purity_lift, 5)


# ------------------------------------------------------------------
# LR schedule + branch selection
# ------------------------------------------------------------------
def cosine_lr_schedule(epoch, total_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def active_branches(feature_set):
    if feature_set == "hybrid":
        return ("metrics", "shape")
    if feature_set == "full":
        return ALL_BRANCHES
    if feature_set == "metrics":
        return ("metrics",)
    if feature_set == "shape":
        return ("shape",)
    if feature_set == "parity-sequence":
        return ("parity",)
    if feature_set == "residue-sequence":
        return ("residue",)
    raise ValueError(f"Unknown feature set: {feature_set}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_names = active_branches(args.feature_set)
    print(f"v8 feature_set={args.feature_set} loss={args.loss} primary_label={args.primary_label}")
    print(f"Branches: {branch_names} | metrics_loss={args.metrics_loss}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    m_starts, m_data = read_matrix(Path(args.metrics_safe), "m", args.limit)
    print(f"  metrics: {m_data.shape}")

    inputs = {"metrics": m_data}
    input_dims = {"metrics": int(m_data.shape[1])}

    if "shape" in branch_names:
        s_starts, s_data = read_matrix(Path(args.log_sketch), "s", args.limit)
        inputs["shape"] = s_data
        input_dims["shape"] = int(s_data.shape[1])
        assert len(s_starts) == len(m_starts), "shape row count mismatch"

    if "parity" in branch_names:
        p_data = read_token_sequence(Path(args.parity_runs), m_starts, args.sequence_len)
        inputs["parity"] = p_data
        input_dims["parity"] = int(p_data.shape[1])
        assert len(p_data) == len(m_starts), "parity row count mismatch"

    if "residue" in branch_names:
        r_data = read_token_sequence(Path(args.transitions), m_starts, args.sequence_len)
        inputs["residue"] = r_data
        input_dims["residue"] = int(r_data.shape[1])
        assert len(r_data) == len(m_starts), "residue row count mismatch"

    starts = m_starts
    labels, label_to_indices, label_counts = read_families(
        Path(args.families), starts, args.primary_label
    )
    N = len(starts)
    print(f"  Loaded {N} items, {len(label_to_indices)} {args.primary_label} classes")
    print(f"  Distribution: min={min(label_counts.values())}, max={max(label_counts.values())}, avg={sum(label_counts.values())/len(label_counts):.1f}")

    # Move inputs to GPU
    inputs = {k: t.to(device) for k, t in inputs.items()}
    m_data_gpu = m_data.to(device)

    # ------------------------------------------------------------------
    # Hard negatives (triplet loss only)
    # ------------------------------------------------------------------
    hard_neg_indices = None
    if args.loss == "triplet":
        print("\nMining hard negatives from raw metrics space...")
        hard_neg_indices = mine_hard_negatives_raw_metrics(m_data_gpu, label_to_indices, k=args.hard_negatives)
        print(f"  Mined {hard_neg_indices.numel()} hard negative edges (k={args.hard_negatives})")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    total_input = sum(input_dims[b] for b in branch_names)
    model = V1Encoder(branch_names, total_input, args.hidden_dims, args.embedding_dim).to(device)
    print(f"\nModel: V1 encoder, input_dim={total_input}, hidden={args.hidden_dims}, embed={args.embedding_dim}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    metrics_head = None
    if args.metrics_loss:
        metrics_head = MetricsProjectionHead(args.embedding_dim, m_data.shape[1]).to(device)
        print(f"  + metrics projection head; total params: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in metrics_head.parameters()):,}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + (list(metrics_head.parameters()) if metrics_head is not None else []),
        lr=3e-4, weight_decay=1e-4,
    )

    # Label bookkeeping for fast pair sampling
    name_to_int = {name: i for i, name in enumerate(sorted(label_to_indices.keys()))}
    label_to_indices_gpu = {name_to_int[l]: torch.tensor(indices, dtype=torch.long, device=device)
                            for l, indices in label_to_indices.items()}
    label_ids = torch.tensor([name_to_int[l] for l in labels], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}")
    losses = []
    best_lift = -999.0
    history = []
    base_lr = 3e-4

    for epoch in range(args.epochs):
        # Refresh the InfoNCE negative pool each epoch so it tracks the model
        if args.loss == "info_nce":
            model.eval()
            with torch.no_grad():
                all_emb = []
                for b_s in range(0, N, args.batch_size):
                    b_e = min(b_s + args.batch_size, N)
                    if b_e - b_s < 8:
                        continue
                    all_emb.append(model({name: t[b_s:b_e] for name, t in inputs.items()}, 0.0))
                all_embeddings = torch.cat(all_emb, dim=0)
        else:
            # triplet: reuse the one-shot full-embedding reference
            if epoch == 0:
                model.eval()
                with torch.no_grad():
                    all_emb = []
                    for b_s in range(0, N, args.batch_size):
                        b_e = min(b_s + args.batch_size, N)
                        if b_e - b_s < 8:
                            continue
                        all_emb.append(model({name: t[b_s:b_e] for name, t in inputs.items()}, 0.0))
                    all_embeddings = torch.cat(all_emb, dim=0)

        model.train()
        if metrics_head is not None:
            metrics_head.train()
        epoch_loss_accum = 0.0
        epoch_batches = 0

        perm = torch.randperm(N, device=device)
        total_batches = ((N - 8) // args.batch_size) + 1

        for batch_i, bs in enumerate(range(0, N, args.batch_size)):
            be = min(bs + args.batch_size, N)
            if be - bs < 8:
                continue
            if (batch_i + 1) % 50 == 0:
                print(f"    epoch {epoch+1}: batch {batch_i+1}/{total_batches} ({(batch_i+1)/total_batches*100:.0f}%)")

            b_idx = perm[bs:be]

            # Sample one positive per anchor from the same label group
            anchor_labels_int = label_ids[b_idx]
            pos_b_indices = torch.empty(b_idx.shape[0], dtype=torch.long, device=device)
            for lbl in torch.unique(anchor_labels_int):
                mask = (anchor_labels_int == lbl)
                group = label_to_indices_gpu[lbl.item()]
                n_in_group = int(mask.sum().item())
                pos_b_indices[mask] = group[torch.randint(0, len(group), (n_in_group,), device=device)]

            z_anchor = model({name: t[b_idx] for name, t in inputs.items()}, 0.0)

            if args.loss == "info_nce":
                loss = info_nce_loss(z_anchor, pos_b_indices, all_embeddings,
                                     args.temperature, anchor_self_indices=b_idx)
            else:
                z_positive = model({name: t[pos_b_indices] for name, t in inputs.items()}, 0.0)
                hard_neg = hard_neg_indices[b_idx]
                loss = triplet_loss_with_hard_negatives(z_anchor, z_positive, hard_neg,
                                                        all_embeddings, args.temperature)

            if metrics_head is not None:
                z_projected = F.normalize(metrics_head(z_anchor), dim=1)
                target = F.normalize(m_data_gpu[b_idx], dim=1)
                loss_metrics = cosine_preservation_loss(z_projected, target)
                loss = (1.0 - args.metrics_loss_weight) * loss + args.metrics_loss_weight * loss_metrics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_accum += float(loss.detach().cpu())
            epoch_batches += 1

        epoch_loss = epoch_loss_accum / max(epoch_batches, 1)
        losses.append(epoch_loss)

        # Evaluate every N epochs (and epoch 0)
        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                all_z = []
                for b_start in range(0, N, args.batch_size):
                    b_end = min(b_start + args.batch_size, N)
                    if b_end - b_start < 8:
                        continue
                    b_eval = torch.arange(b_start, b_end, device=device)
                    all_z.append(model({name: t[b_eval] for name, t in inputs.items()}, 0.0))
                z_full = torch.cat(all_z, dim=0)

            np_eval, rb_eval, lift_eval = evaluate_neighbors(z_full, labels, k=2)

            if lift_eval > best_lift:
                best_lift = lift_eval
                ckpt = {"model_state": model.state_dict()}
                if metrics_head is not None:
                    ckpt["metrics_head_state"] = metrics_head.state_dict()
                torch.save(ckpt, output_dir / "encoder_best.pt")

            print(f"  epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} lift={lift_eval:.4f} best_lift={best_lift:.4f}")
        else:
            print(f"  epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f}")

        new_lr = cosine_lr_schedule(epoch + 1, args.epochs, base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "lr": new_lr})

    # ------------------------------------------------------------------
    # Final evaluation with best checkpoint
    # ------------------------------------------------------------------
    print("\nFinal evaluation...")
    best_ckpt = output_dir / "encoder_best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        if "metrics_head_state" in checkpoint and metrics_head is not None:
            metrics_head.load_state_dict(checkpoint["metrics_head_state"])
    model.eval()
    with torch.no_grad():
        all_z = []
        for b_start in range(0, N, args.batch_size):
            b_end = min(b_start + args.batch_size, N)
            if b_end - b_start < 8:
                continue
            b_eval = torch.arange(b_start, b_end, device=device)
            all_z.append(model({name: t[b_eval] for name, t in inputs.items()}, 0.0))
        z_final = torch.cat(all_z, dim=0)

    neighbor_purity, random_baseline, purity_lift = evaluate_neighbors(z_final, labels, k=2)

    # Save embeddings
    with (output_dir / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label"] + [f"e{i}" for i in range(z_final.shape[1])])
        for n, label, vector in zip(starts, labels, z_final.tolist()):
            writer.writerow([n, label] + [f"{v:.9g}" for v in vector])

    torch.save({
        "model_state": model.state_dict(),
        "metrics_head_state": metrics_head.state_dict() if metrics_head is not None else None,
        "args": vars(args),
    }, output_dir / "encoder.pt")

    final_metrics = {
        "dataset_type": "collatz_contrastive_embeddings_v8",
        "tool": "research/contrastive_train_v8.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": N,
        "feature_set": args.feature_set,
        "branches": list(branch_names),
        "loss_function": args.loss,
        "primary_label": args.primary_label,
        "label_stats": {
            "unique_classes": len(label_to_indices),
            "min_per_class": min(label_counts.values()) if label_counts else 0,
            "max_per_class": max(label_counts.values()) if label_counts else 0,
            "avg_per_class": sum(label_counts.values()) / len(label_counts) if label_counts else 0,
            "top5_classes": label_counts.most_common(5),
        },
        "embedding_dims": args.embedding_dim,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "loss_start": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_history": losses,
        "neighbor_purity": neighbor_purity,
        "random_baseline_purity": random_baseline,
        "purity_lift": purity_lift,
        "best_lift": best_lift,
        "positive_pair_mode": "same_label_random",
        "metrics_loss_weight": args.metrics_loss_weight if args.metrics_loss else 0.0,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nv8 feature_set={args.feature_set} loss={args.loss}")
    print(f"  primary_label={args.primary_label} classes={len(label_to_indices)}")
    print(f"  rows={N} lift={purity_lift:.4f} best_lift={best_lift:.4f}")
    print(f"  loss history: start={losses[0] if losses else 0:.4f} final={losses[-1] if losses else 0:.4f}")
    print(f"  outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
