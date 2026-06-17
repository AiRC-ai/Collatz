#!/usr/bin/env python3
"""Train Collatz embeddings v7: single encoder, raw-metrics hard negatives, range_band positives.

Design from v4/v5/v6 learnings:
- v4 failed: mined hard negatives from model embeddings (bad early signal -> bad negatives).
- v5 failed: mined from raw metrics (correct!) but used coalescence_family_id (61K classes, ~1.6 each).
- v6 had: coarse labels (right), raw metrics mining (right), but overly complex pair sampling
  and wasteful precomputed embeddings.

v7 fixes:
- Single V1 encoder (concat all branches -> MLP -> embedding).
- Positive pairs sampled from coarse label groups (range_band: 16 classes, ~6.25K each).
- Hard negatives mined once from raw normalized metrics space (no embedding dependency).
- Auxiliary metrics preservation loss keeps embedding aligned with raw features.
- No self-noise pairs, no per-epoch negative mining.
- Evaluate every 10 epochs.
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
        description="v7: single-encoder, raw-metrics hard negatives, range_band positives"
    )
    parser.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    parser.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v7")
    parser.add_argument("--primary-label", default="range_band",
        choices=("range_band", "bit_length"))
    parser.add_argument("--feature-set", choices=("hybrid", "metrics", "shape",
                           "parity-sequence", "residue-sequence"), default="hybrid")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--metrics-loss-weight", type=float, default=0.2)
    parser.add_argument("--hard-negatives", type=int, default=50,
        help="Number of hard negatives per anchor")
    parser.add_argument("--max-pairs-per-batch", type=int, default=256,
        help="Max positive pairs per training batch")
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=192)
    parser.add_argument("--use-v1-encoder", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--no-metrics-loss", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def standardize(x):
    """Column-wise standardization."""
    m, s = x.mean(dim=0), x.std(dim=0)
    s = s.clamp(min=1e-6)
    return (x - m) / s


def read_matrix(path: Path, prefix: str, limit: int = 0):
    """Read numeric columns from CSV with given prefix."""
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
    """Load family labels, return (labels_list, label_to_indices_dict, counts)."""
    wanted = set(starts)
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
# Hard negative mining (raw metrics space, done once at start)
# ------------------------------------------------------------------

def mine_hard_negatives_raw_metrics(metrics_tensor, label_to_indices, k=50, chunk_size=10000):
    """Mine hard negatives from raw normalized metrics space.

    For each sample, find k nearest neighbors in metrics space that have
    a DIFFERENT label. Masks same-label and same-item entries from the
    similarity matrix.

    Returns [N, k] tensor of indices into the full dataset.
    Vectorized version: replaces the O(N * labels) Python loop with
    a single torch.eq broadcast for the same-label mask.
    """
    n = metrics_tensor.shape[0]
    device = metrics_tensor.device

    # Map each item to its label integer
    label_names = sorted(label_to_indices.keys())
    name_to_int = {name: i for i, name in enumerate(label_names)}
    label_arr = torch.zeros(n, dtype=torch.long, device=device)
    for label, indices in label_to_indices.items():
        label_arr[torch.tensor(indices, dtype=torch.long, device=device)] = name_to_int[label]

    # Normalize metrics for cosine similarity
    metrics_normed = F.normalize(metrics_tensor, dim=1)

    # Pre-allocate result
    neg_indices = torch.full((n, k), -1, dtype=torch.long, device=device)

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_sz = chunk_end - chunk_start

        # Similarity for this chunk against all items
        chunk_sim = metrics_normed[chunk_start:chunk_end] @ metrics_normed.T

        # Mask same-item diagonal
        diag = torch.arange(chunk_start, chunk_end, device=device)
        chunk_sim[torch.arange(chunk_sz), diag] = float("-inf")

        # Vectorized same-label mask: [chunk_sz, n] via label comparison
        chunk_labels = label_arr[chunk_start:chunk_end]   # [chunk_sz]
        all_labels = label_arr                             # [n]
        same_label_mask = chunk_labels.unsqueeze(1) == all_labels.unsqueeze(0)   # [chunk_sz, n]
        chunk_sim[same_label_mask] = float("-inf")

        # Top-k hardest negatives (highest similarity but wrong label)
        k_actual = min(k, n - 1)
        _, topk_indices = chunk_sim.topk(k_actual, dim=1)
        neg_indices[chunk_start:chunk_end] = topk_indices

    return neg_indices



# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class V1Encoder(nn.Module):
    """Single-branch encoder: concat all branches, project to embedding."""

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

def triplet_loss_with_hard_negatives(z_anchor, z_positive, hard_neg_indices, z_all, temperature=0.07):
    """Triplet loss: push positives closer, push hard negatives farther.

    For each anchor, compute:
      pos_dist = 1 - cosine(anchor, positive)
      neg_dist = min(1 - cosine(anchor, neg_i)) over all hard negatives
    loss = max(0, pos_dist - neg_dist + margin)
    """
    margin = 0.3
    pos_dist = 1.0 - F.cosine_similarity(z_anchor, z_positive, dim=1)
     # hard_neg_indices: [batch_size, k] -- indices into z_all
    neg_sim = F.cosine_similarity(z_anchor.unsqueeze(1), z_all[hard_neg_indices], dim=2)
    neg_dist = 1.0 - neg_sim
    # Use the hardest negative per anchor
    hardest_neg_dist = neg_dist.min(dim=1).values
    loss = F.relu(pos_dist - hardest_neg_dist + margin)
    return loss.mean()


def cosine_preservation_loss(embedding, target_raw_metrics, temperature=1.0):
    """Auxiliary loss: project embedding and match to raw metrics direction."""
    emb_norm = F.normalize(embedding, dim=1)
    tgt_norm = F.normalize(target_raw_metrics, dim=1)
    sim = torch.sum(emb_norm * tgt_norm, dim=1) / temperature
    return 1.0 - sim.mean()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_neighbors(embeddings, labels, k=2):
    """Measure lift: fraction of nearest neighbors with same label.

    Vectorized: computes full similarity matrix once, O(N^2) but fast via one CUDA op.
    """
    # Full similarity matrix: [N, N]
    sim = torch.matmul(embeddings, embeddings.T)
    N = sim.shape[0]
    # Zero out self-similarity on diagonal
    diag_mask = torch.eye(N, dtype=torch.bool, device=sim.device)
    sim[diag_mask] = float("-inf")
    # Top-k+1 for each item: [N, k+1]
    top_k_idx = sim.topk(k + 1, dim=1).indices  # [N, k+1]
    # Convert labels to int IDs for fast tensor ops
    label_to_int = {name: i for i, name in enumerate(sorted(set(labels)))}
    labels_int = [label_to_int[l] for l in labels]
    labels_t = torch.tensor(labels_int, device=embeddings.device)
    # Get labels of top-k neighbors
    neighbor_labels = labels_t[top_k_idx]  # [N, k+1]
    # Mask out self
    self_mask = top_k_idx == torch.arange(N, device=embeddings.device).unsqueeze(1)
    neighbor_labels[self_mask] = labels_t[N-1] + 1  # mark as different
    # Count matches (excluding self)
    not_self = ~self_mask
    num_same = not_self.sum().item()
    num_match = (neighbor_labels[not_self] == labels_t.unsqueeze(1).expand_as(neighbor_labels)[not_self]).sum().item()
    neighbor_purity = num_match / max(num_same, 1)
    counts = Counter(labels)
    random_baseline = sum((count / len(labels)) ** 2 for count in counts.values())
    purity_lift = neighbor_purity - random_baseline
    return round(neighbor_purity, 5), round(random_baseline, 5), round(purity_lift, 5)


# ------------------------------------------------------------------
# LR schedule
# ------------------------------------------------------------------

def cosine_lr_schedule(epoch, total_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


# ------------------------------------------------------------------
# Branch selection
# ------------------------------------------------------------------

def active_branches(feature_set):
    if feature_set == "hybrid":
        return ALL_BRANCHES
    elif feature_set == "metrics":
        return ("metrics",)
    elif feature_set == "shape":
        return ("shape",)
    elif feature_set == "parity-sequence":
        return ("parity",)
    elif feature_set == "residue-sequence":
        return ("residue",)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_names = active_branches(args.feature_set)
    print(f"v7 feature_set={args.feature_set} primary_label={args.primary_label}")
    print(f"Branches: {branch_names}")

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
    print(f"  Loaded {len(starts)} items, {len(label_to_indices)} {args.primary_label} classes")
    print(f"  Distribution: min={min(label_counts.values())}, max={max(label_counts.values())}, avg={sum(label_counts.values())/len(label_counts):.1f}")

    N = len(starts)

    # ------------------------------------------------------------------
    # Mine hard negatives from raw metrics (once, at start)
    # ------------------------------------------------------------------
    print("\nMining hard negatives from raw metrics space...")
    metrics_gpu = m_data.to(device)
    hard_neg_indices = mine_hard_negatives_raw_metrics(metrics_gpu, label_to_indices, k=args.hard_negatives)
    print(f"  Mined {hard_neg_indices.numel()} hard negative edges (k={args.hard_negatives})")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    total_input = sum(input_dims.values())
    model = V1Encoder(branch_names, total_input, args.hidden_dims, args.embedding_dim).to(device)
    print(f"\nModel: V1 single-branch encoder, input_dim={total_input}, hidden={args.hidden_dims}, embed={args.embedding_dim}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    if not args.no_metrics_loss:
        metrics_head = MetricsProjectionHead(args.embedding_dim, m_data.shape[1]).to(device)
        total_params += sum(p.numel() for p in metrics_head.parameters())
        print(f"  Total params (incl. metrics head): {total_params:,}")
    else:
        metrics_head = None

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optim_params = list(model.parameters())
    if metrics_head is not None:
        optim_params.extend(metrics_head.parameters())
    optimizer = torch.optim.AdamW(optim_params, lr=3e-4, weight_decay=1e-4)

    inputs = {k: t.to(device) for k, t in inputs.items()}
    m_data_gpu = m_data.to(device)

    # Precompute label-to-indices on GPU for pair sampling
    name_to_int = {name: i for i, name in enumerate(sorted(label_to_indices.keys()))}
    label_to_indices_gpu = {name_to_int[l]: torch.tensor(indices, dtype=torch.long, device=device)
                            for l, indices in label_to_indices.items()}
     # Integer label IDs for fast batch indexing
    label_ids = torch.tensor([name_to_int[l] for l in labels], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Precompute all embeddings for negative mining reference
    print("  Precomputing full embedding reference...")
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for b_s in range(0, N, args.batch_size):
            b_e = min(b_s + args.batch_size, N)
            if b_e - b_s < 8:
                continue
            z = model({name: tensor[b_s:b_e] for name, tensor in inputs.items()}, 0.0)
            all_embeddings.append(z)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"  Reference embeddings: {all_embeddings.shape[0]} x {all_embeddings.shape[1]}")

    # Training loop
    # ------------------------------------------------------------------
    print(f"\nTraining: {args.epochs} epochs, batch_size={args.batch_size}")
    losses = []
    best_lift = -999.0
    history = []

    for epoch in range(args.epochs):
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
            be = min(bs + args.batch_size, N)
            if be - bs < 8:
                continue

            b_idx = perm[bs:be].to(device)

             # --- Sample positive pairs from label groups ---
             # For each anchor in the batch, pick one positive from the same label group
               # --- Sample positive pairs from label groups ---
               # Precompute random offsets per label for fast batch sampling
            anchor_labels_int = label_ids[b_idx]     # [batch_size] int labels
            unique_labels = torch.unique(anchor_labels_int)
            pos_b_indices = torch.empty(b_idx.shape[0], dtype=torch.long, device=device)
            for lbl in unique_labels:
                mask = (anchor_labels_int == lbl)
                group = label_to_indices_gpu[lbl.item()]
                n_in_group = mask.sum().item()
                sampled = group[torch.randint(0, len(group), (n_in_group,), device=device)]
                pos_b_indices[mask] = sampled
            # pos_b_indices: [batch_size] indices of positive pairs

            # --- Get embeddings ---
            z_anchor = model({name: tensor[b_idx] for name, tensor in inputs.items()}, 0.0)
            z_positive = model({name: tensor[pos_b_indices] for name, tensor in inputs.items()}, 0.0)

            # --- Triplet loss with hard negatives ---
            hard_neg = hard_neg_indices[b_idx]  # [batch_size, k]
            loss_triplet = triplet_loss_with_hard_negatives(z_anchor, z_positive, hard_neg, all_embeddings, args.temperature)

            # --- Metrics preservation loss ---
            if metrics_head is not None:
                z_projected = F.normalize(metrics_head(z_anchor), dim=1)
                target = F.normalize(m_data_gpu[b_idx], dim=1)
                loss_metrics = cosine_preservation_loss(z_projected, target)
                loss = (1.0 - args.metrics_loss_weight) * loss_triplet + args.metrics_loss_weight * loss_metrics
            else:
                loss = loss_triplet

            # --- Optimize ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_accum += float(loss.detach().cpu())
            epoch_batches += 1
            if (batch_i + 1) % 20 == 0:
                import sys; sys.stderr.write(f"epoch {epoch+1}: batch {batch_i+1}/{total_batches}\n"); sys.stderr.flush()

        epoch_loss = epoch_loss_accum / max(epoch_batches, 1)
        losses.append(epoch_loss)

        # --- Evaluate every N epochs ---
        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                all_z = []
                for b_start in range(0, N, args.batch_size):
                    b_end = min(b_start + args.batch_size, N)
                    if b_end - b_start < 8:
                        continue
                    b_eval = torch.arange(b_start, b_end, device=device)
                    z = model({name: tensor[b_eval] for name, tensor in inputs.items()}, 0.0)
                    all_z.append(z)
                z_full = torch.cat(all_z, dim=0)

            np_eval, rb_eval, lift_eval = evaluate_neighbors(z_full.cpu(), labels, k=2)

            if lift_eval > best_lift:
                best_lift = lift_eval
                checkpoint = {
                    "model_state": model.state_dict(),
                }
                if metrics_head is not None:
                    checkpoint["metrics_head_state"] = metrics_head.state_dict()
                torch.save(checkpoint, output_dir / "encoder_best.pt")

            new_lr = cosine_lr_schedule(epoch + 1, args.epochs, 3e-4)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

            print(f"  epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} lift={lift_eval:.4f} best_lift={best_lift:.4f} lr={new_lr:.6f}")

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "lift": lift_eval if (epoch + 1) % args.evaluate_every == 0 or epoch == 0 else None,
            "best_lift": best_lift,
            "lr": cosine_lr_schedule(epoch + 1, args.epochs, 3e-4)
        })

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print("\nFinal evaluation...")
    checkpoint = torch.load(output_dir / "encoder_best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if "metrics_head_state" in checkpoint:
        metrics_head.load_state_dict(checkpoint["metrics_head_state"])
    model.eval()
    if metrics_head is not None:
        metrics_head.eval()
    with torch.no_grad():
        all_z = []
        for b_start in range(0, N, args.batch_size):
            b_end = min(b_start + args.batch_size, N)
            if b_end - b_start < 8:
                continue
            b_eval = torch.arange(b_start, b_end, device=device)
            z = model({name: tensor[b_eval] for name, tensor in inputs.items()}, 0.0)
            all_z.append(z)
        z_final = torch.cat(all_z, dim=0)

    neighbor_purity, random_baseline, purity_lift = evaluate_neighbors(z_final.cpu(), labels, k=2)

    # Save embeddings
    with (output_dir / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label"] + [f"e{i}" for i in range(z_final.shape[1])])
        for n, label, vector in zip(starts, labels, z_final.tolist()):
            writer.writerow([n, label] + [f"{v:.9g}" for v in vector])

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "metrics_head_state": metrics_head.state_dict() if metrics_head is not None else None,
        "args": vars(args),
    }, output_dir / "encoder.pt")

    # Save metrics
    final_metrics = {
        "dataset_type": "collatz_contrastive_embeddings_v7",
        "tool": "research/contrastive_train_v7.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": len(starts),
        "feature_set": args.feature_set,
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
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "loss_history": losses,
        "neighbor_purity": neighbor_purity,
        "random_baseline_purity": random_baseline,
        "purity_lift": purity_lift,
        "best_lift": best_lift,
        "hard_negative_source": "raw_metrics",
        "hard_negative_count": int(hard_neg_indices.numel()),
        "positive_pair_mode": "same_label_random",
        "metrics_loss_weight": args.metrics_loss_weight,
        "temperature": args.temperature,
        "n_folds": None, "n_seeds": None, "ci_95": None,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Print summary
    print(f"\nv7 feature_set={args.feature_set}")
    print(f"  primary_label={args.primary_label} classes={len(label_to_indices)}")
    print(f"  rows={len(starts)} lift={purity_lift:.4f} best_lift={best_lift:.4f}")
    print(f"  hard_negative_source=raw_metrics hard_negatives={args.hard_negatives}")
    print(f"  positive_pair_mode=same_label_random")
    print(f"  metrics_loss_weight={args.metrics_loss_weight}")
    print(f"  loss history: start={losses[0]:.4f} final={losses[-1]:.4f}")
    print(f"  outputs: {output_dir}")


if __name__ == "__main__":
    main()
