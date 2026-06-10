#!/usr/bin/env python3
"""Train Collatz embeddings v4: hard negative mining + metrics-aware loss.

Key changes from v3:
  - Hard negative mining: after each epoch, find nearest neighbors that are
    DIFFERENT family (misclassifications) and pull them away.
  - Metrics-aware loss: add a cosine similarity term that pulls the
    embedding close to the raw metrics vector (projected through a small MLP),
    preserving trajectory structure the encoder might otherwise compress away.
  - 200 epochs, higher learning rate, cosine LR schedule.
  - Keeps v1 single-encoder when --use-v1-encoder is set.
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
        description="Train Collatz embeddings v4 with hard negatives."
    )
    parser.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    parser.add_argument("--positive-pairs", default="/home/ryancox/3xN1/data/generated/ml_pairs/positive_pairs.csv")
    parser.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v4")
    parser.add_argument("--pair-mode", choices=("self_noise", "family_pairs", "combined"), default="family_pairs")
    parser.add_argument("--primary-label", default="tail_hash",
        choices=("tail_hash", "coalescence_family_id", "parity_motif_hash",
                 "residue_motif_hash", "range_band"))
    parser.add_argument("--feature-set", choices=("hybrid", "metrics", "shape",
                     "parity-sequence", "residue-sequence"), default="hybrid")
    parser.add_argument("--use-v1-encoder", action="store_true", default=False,
        help="Use single-branch encoder (v1 architecture) instead of multi-branch")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dims", type=int, default=192)
    parser.add_argument("--embedding-dims", type=int, default=64)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--representation-dropout", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=20260520)
    # v4 specific
    parser.add_argument("--hard-negatives", type=int, default=20,
        help="Number of hard negatives per positive pair (0 = off)")
    parser.add_argument("--metrics-loss-weight", type=float, default=0.3,
        help="Weight for metrics-aware supervision (0 = off)")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def standardize(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.nan_to_num((x - mean) / std)


def read_matrix(path, prefix, limit):
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
    wanted = set(starts)
    by_n = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n not in wanted:
                continue
            values = []
            for item in (row.get("tokens", "") or "").split(";"):
                if item:
                    values.append(float(int(item) & 0xffff) / 65535.0)
            by_n[n] = torch.tensor(values[:seq_len], dtype=torch.float32)
    sequences = []
    for n in starts:
        seq = by_n.get(n, torch.zeros(seq_len, dtype=torch.float32))
        if len(seq) < seq_len:
            seq = F.pad(seq, (0, seq_len - len(seq)), value=0.0)
        sequences.append(seq)
    return torch.stack(sequences, dim=0)


def read_families(path, starts, label_field):
    label_map = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n in starts:
                label_map[n] = row.get(label_field, row.get("family_id", "unknown"))
    labels = [label_map.get(n, "unknown") for n in starts]
    return labels, label_map


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class V1Encoder(nn.Module):
    """Single-branch encoder: concatenate all branches, project to embedding."""
    def __init__(self, branch_names, in_dims, hidden_dim, embedding_dim):
        super().__init__()
        self.branch_names = branch_names
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
          )

    def forward(self, inputs, rep_dropout):
        x = torch.cat([inputs[name] for name in self.branch_names], dim=1)
        if rep_dropout > 0 and self.training:
            x = F.dropout(x, p=rep_dropout, training=self.training)
        return F.normalize(self.net(x), dim=1)

class FamilyEncoder(nn.Module):
    """Multi-branch encoder from v3."""

    def __init__(self, input_dims, hidden_dims, output_dim, gated=True, branch_names=None):
        super().__init__()
        self.branch_names = branch_names or list(input_dims.keys())
        self.branches = nn.ModuleDict()
        self.aggregators = nn.ModuleDict()
        for name, dim in input_dims.items():
            self.branches[name] = nn.Sequential(
                nn.Linear(dim, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, output_dim),
            )
            self.aggregators[name] = nn.Sequential(
                nn.Linear(output_dim, hidden_dims),
                nn.GELU(),
                nn.Linear(hidden_dims, hidden_dims),
            )
        self.project = nn.Sequential(
            nn.Linear(hidden_dims * len(self.branch_names), hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims, output_dim),
        )
        if gated:
            self.gate_logit = nn.Parameter(torch.zeros(len(self.branch_names)))
        else:
            self.gate_logit = None

    def forward(self, inputs, dropout=0.0):
        outs = []
        for name in self.branch_names:
            if name not in inputs:
                continue
            branch_out = self.branches[name](inputs[name])
            agg = self.aggregators[name](branch_out)
            if self.gate_logit is not None:
                w = torch.softmax(self.gate_logit, dim=0)[
                    self.branch_names.index(name)
                ].unsqueeze(-1)
                branch_out = branch_out * w
            outs.append(branch_out)
        combined = torch.cat(outs, dim=-1)
        projected = F.layer_norm(combined, (combined.shape[-1],))
        projected = F.gelu(projected)
        projected = F.dropout(projected, p=0.1, training=self.training)
        return F.normalize(projected, dim=-1)


# ------------------------------------------------------------------
# Metrics-aware projection head
# ------------------------------------------------------------------

class MetricsProjectionHead(nn.Module):
    """Small head that maps embeddings back toward raw metrics for supervision."""

    def __init__(self, embedding_dim, metric_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, min(embedding_dim * 2, 512)),
            nn.LayerNorm(min(embedding_dim * 2, 512)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(min(embedding_dim * 2, 512), metric_dim),
        )

    def forward(self, x):
        return self.projection(x)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def info_nce_with_hard_negatives(z_a, z_b, hard_neg_indices, temperature=0.07):
    """InfoNCE where negatives are sampled from hard negatives (misclassifications).

    z_a, z_b: [B, D] positive pair embeddings
    hard_neg_indices: [B, K] indices of hard negatives (wrong family)
    """
    batch_size = z_a.shape[0]
    device = z_a.device

    # Compute positives
    pos_sim = torch.sum(z_a * z_b, dim=1) / temperature  # [B]

    # Hard negative similarities
    if hard_neg_indices is not None and hard_neg_indices.shape[1] > 0:
        neg_z = z_b[hard_neg_indices]  # [B, K, D]
        neg_sim = torch.sum(z_a.unsqueeze(1) * neg_z, dim=2) / temperature  # [B, K]

        # Combined: positives + hard negatives
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+K]
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
    else:
        # Fallback to standard InfoNCE
        z = torch.cat([z_a, z_b], dim=0)
        sim = torch.matmul(z, z.transpose(0, 1)) / temperature
        sim_ij = sim[:batch_size, batch_size:]
        sim_ji = sim[batch_size:, :batch_size]
        pos = torch.diag(sim[:batch_size, batch_size:])
        labels = torch.arange(batch_size, device=sim.device)
        loss_i = F.cross_entropy(torch.cat([sim_ij, pos.unsqueeze(1)], dim=1), labels)
        loss_j = F.cross_entropy(torch.cat([sim_ji, pos.unsqueeze(1)], dim=1), labels)
        loss = (loss_i + loss_j) / 2

    return loss


def cosine_loss(embedding, target, temperature=1.0):
    """Cosine similarity loss: pull embedding toward normalized target."""
    target_norm = F.normalize(target, dim=-1)
    emb_norm = F.normalize(embedding, dim=-1)
    return 1.0 - (emb_norm * target_norm).mean() / temperature


# ------------------------------------------------------------------
# Hard negative mining
# ------------------------------------------------------------------

def mine_hard_negatives(embeddings, labels, k=20, chunk_size=10000):
    """Find k nearest neighbors that have DIFFERENT labels for each item.

    Memory-efficient: computes the similarity matrix in chunks so a 100k
    sample set never holds a full 100k x 100k matrix in GPU memory.
    """
    n = embeddings.shape[0]
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    # Normalize embeddings for cosine similarity
    norms = embeddings.norm(dim=1, keepdim=True)
    emb_normed = embeddings / norms.clamp_min(1e-8)

    # Pre-allocate result: fill with -1, overwrite with top-k indices
    neg_indices = torch.full((n, k), -1, dtype=torch.long, device=embeddings.device)

    # Build a label mask: same_label[i, j] = True iff i and j share a label
    # Shape (n, n) boolean = ~100M bytes for 100k items, fits in memory.
    # Convert string labels to numeric indices
    unique_labels = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    labels_int = torch.tensor([label_to_int[l] for l in labels], device=embeddings.device)
    # For same-label mask, compare original string labels (works correctly)
    labels_t = torch.tensor([label_to_int[l] for l in labels], device=embeddings.device)
    same_label_mask = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1))

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_sz = chunk_end - chunk_start

        # Compute similarity for this chunk against all items
        chunk_sim = emb_normed[chunk_start:chunk_end] @ emb_normed.T    # (chunk, n)

        # Set same-item diagonal to -inf
        diag = torch.arange(chunk_start, chunk_end, device=embeddings.device)
        chunk_sim[torch.arange(chunk_sz), diag] = float('-inf')

        # Mask same-label entries to -inf
        chunk_sim = chunk_sim.masked_fill(same_label_mask[chunk_start:chunk_end], float('-inf'))

        # Get top-k hardest negatives (highest similarity but wrong label)
        k_actual = min(k, n - 1)
        topk_values, topk_indices = chunk_sim.topk(k_actual, dim=1)

         # topk returns valid indices; -inf masked positions ensure only different-label results.
        neg_indices[chunk_start:chunk_end] = topk_indices


        # Free chunk memory
        # del chunk_sim handled by GC
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return neg_indices

def evaluate_neighbors(embeddings, labels, k=2):
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)
    purities = []
    for i, label in enumerate(labels):
        same = [j for j in label_to_indices[label] if j != i]
        if not same:
            continue
        sim = (embeddings @ embeddings[i]).cpu()
        top_k = sim.topk(k + 1, dim=0).indices
        top_k = top_k[top_k != i].tolist()
        purities.append(sum(1 for j in top_k if labels[j] == label) / len(top_k))
    if not purities:
        return 0.0, 0.0, 0.0
    neighbor_purity = sum(purities) / len(purities)
    counts = Counter(labels)
    random_baseline = sum((count / len(labels)) ** 2 for count in counts.values())
    purity_lift = neighbor_purity - random_baseline
    return round(neighbor_purity, 5), round(random_baseline, 5), round(purity_lift, 5)


def write_metrics(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------------
# Training
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


def cosine_lr_schedule(epoch, total_epochs, initial_lr):
    """Cosine annealing LR schedule."""
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_names = active_branches(args.feature_set)
    print(f"Branches: {branch_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading metrics...")
    m_starts, m_data = read_matrix(Path(args.metrics_safe), "m", args.limit)
    print(f"  metrics: {m_data.shape}")

    inputs = {"metrics": m_data}
    input_dims = {"metrics": int(m_data.shape[1])}

    if "shape" in branch_names:
        print("Loading shape...")
        s_starts, s_data = read_matrix(Path(args.log_sketch), "s", args.limit)
        inputs["shape"] = s_data
        input_dims["shape"] = int(s_data.shape[1])
        assert len(s_starts) == len(m_starts), "shape row count mismatch"

    if "parity" in branch_names:
        print("Loading parity...")
        p_data = read_token_sequence(Path(args.parity_runs), m_starts, args.sequence_len)
        inputs["parity"] = p_data
        input_dims["parity"] = int(p_data.shape[1])
        assert len(p_data) == len(m_starts), "parity row count mismatch"

    if "residue" in branch_names:
        print("Loading residue...")
        r_data = read_token_sequence(Path(args.transitions), m_starts, args.sequence_len)
        inputs["residue"] = r_data
        input_dims["residue"] = int(r_data.shape[1])
        assert len(r_data) == len(m_starts), "residue row count mismatch"

    starts = m_starts
    labels, _ = read_families(Path(args.families), starts, args.primary_label)
    print(f"Loaded {len(starts)} items, {len(set(labels))} unique labels")

    # Model
    if args.use_v1_encoder:
        total_input = sum(input_dims.values())
        model = V1Encoder(branch_names, total_input, args.hidden_dims, args.embedding_dims)
        print(f"Using V1 single-branch encoder, input dims: {total_input}")
    else:
        model = FamilyEncoder(input_dims, args.hidden_dims, args.embedding_dims,
                               gated=False, branch_names=branch_names)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Metrics projection head for supervision
    metrics_head = MetricsProjectionHead(args.embedding_dims, m_data.shape[1]).to(device)
    total_params += sum(p.numel() for p in metrics_head.parameters())
    print(f"Total params (incl. metrics head): {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(metrics_head.parameters()),
        lr=3e-4, weight_decay=1e-4
    )

    inputs = {k: t.to(device) for k, t in inputs.items()}
    m_data_gpu = m_data.to(device)

    # Load positive pairs
    pos_a, pos_b, pair_type_counts = [], [], {}
    if Path(args.positive_pairs).exists():
        with open(args.positive_pairs) as f:
            reader = csv.DictReader(f)
            n_to_idx = {n: i for i, n in enumerate(starts)}
            for row in reader:
                na, nb = int(row["n_a"].strip()), int(row["n_b"].strip())
                pt = row.get("pair_type", "unknown").strip()

                if args.pair_mode == "family_pairs":
                    keep = pt in ("same_coalescence_family", "same_tail_hash")
                elif args.pair_mode == "self_noise":
                    keep = pt == "same_n_different_view"
                elif args.pair_mode == "combined":
                    keep = pt == "same_n_different_view" or pt in ("same_coalescence_family", "same_tail_hash")
                else:
                    keep = True

                if not keep:
                    continue

                ia, ib = n_to_idx.get(na), n_to_idx.get(nb)
                if ia is not None and ib is not None:
                    pos_a.append(ia)
                    pos_b.append(ib)
                    pair_type_counts[pt] = pair_type_counts.get(pt, 0) + 1

        pos_a = torch.tensor(pos_a, dtype=torch.long)
        pos_b = torch.tensor(pos_b, dtype=torch.long)
        print(f"Loaded {len(pos_a)} positive pairs ({pair_type_counts})")

    pos_dataset = torch.utils.data.TensorDataset(pos_a, pos_b)
    pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=256, shuffle=True, drop_last=True)

    # Mining buffer (persist between epochs)
    mining_buffer = {
        "neg_indices": None,
        "last_labels": list(labels),
        "last_embeddings": None,
    }

    # Training loop
    losses = []
    best_lift = -999
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        metrics_head.train()
        epoch_losses = []

        perm = torch.randperm(len(starts), device=device)
        pair_perm = torch.randperm(len(pos_a), device=device) if len(pos_a) > 0 else None

        # Mine hard negatives every 5 epochs
        if epoch % 5 == 0 or epoch == 0:
            model.eval()
            metrics_head.eval()
            with torch.no_grad():
                all_z = []
                for b_start in range(0, len(starts), args.batch_size):
                    b_end = min(b_start + args.batch_size, len(starts))
                    if b_end - b_start < 8:
                        continue
                    b_idx = perm[b_start:b_end].to(device)
                    view = {name: tensor[b_idx] for name, tensor in inputs.items()}
                    z = model(view, 0.0)
                    all_z.append(z)
                all_z = torch.cat(all_z, dim=0)
            mining_buffer["last_embeddings"] = all_z.cpu()
            mining_buffer["neg_indices"] = mine_hard_negatives(
                all_z, mining_buffer["last_labels"], k=args.hard_negatives
            ).to(device)
            hard_neg_count = int((mining_buffer["neg_indices"] != -1).sum().item()) if mining_buffer["neg_indices"] is not None else 0
            print(f"  mined {hard_neg_count} hard negative edges (k={args.hard_negatives})", flush=True)

        for bs in range(0, len(starts), args.batch_size):
            be = min(bs + args.batch_size, len(starts))
            if be - bs < 8:
                continue

            b_idx = perm[bs:be].to(device)

            # Self-noise pairs
            view_a = {name: tensor[b_idx] for name, tensor in inputs.items()}
            view_b = {name: value + torch.randn_like(value) * 0.025
                      for name, value in view_a.items()}

            z_a = model(view_a, args.representation_dropout)
            z_b = model(view_b, args.representation_dropout)

            # Hard negative InfoNCE loss
            hard_neg = mining_buffer["neg_indices"][b_idx] if mining_buffer["neg_indices"] is not None else None
            loss_contrastive = info_nce_with_hard_negatives(
                z_a, z_b, hard_neg, temperature=args.temperature
            )

            # Family pair loss
            loss_family = None
            if args.pair_mode in ("family_pairs", "combined") and len(pos_a) > 0:
                try:
                    fa, fb = next(iter(pos_loader))
                    fa, fb = fa.to(device), fb.to(device)
                    fam_a = {name: tensor[fa] for name, tensor in inputs.items()}
                    fam_b = {name: tensor[fb] for name, tensor in inputs.items()}
                    z_fa = model(fam_a, args.representation_dropout)
                    z_fb = model(fam_b, args.representation_dropout)
                    loss_family = info_nce_with_hard_negatives(
                        z_fa, z_fb, None, temperature=args.temperature
                    )
                except StopIteration:
                    pass

            # Metrics-aware loss
            loss_metrics = None
            if args.metrics_loss_weight > 0:
                emb_for_metrics = z_a
                projected_metrics = metrics_head(emb_for_metrics)
                loss_metrics = cosine_loss(projected_metrics, view_a["metrics"], temperature=1.0)

            # Combine losses
            loss = loss_contrastive
            if loss_family is not None:
                loss = loss + loss_family
            if loss_metrics is not None:
                loss = loss + args.metrics_loss_weight * loss_metrics

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))

        # Track best
        model.eval()
        with torch.no_grad():
            all_z_eval = []
            for b_start in range(0, len(starts), args.batch_size):
                b_end = min(b_start + args.batch_size, len(starts))
                if b_end - b_start < 8:
                    continue
                b_idx_e = perm[b_start:b_end].to(device)
                view_e = {name: tensor[b_idx_e] for name, tensor in inputs.items()}
                z_e = model(view_e, 0.0)
                all_z_eval.append(z_e)
            all_z_eval = torch.cat(all_z_eval, dim=0)

        np_eval, rb_eval, lift_eval = evaluate_neighbors(
            all_z_eval.cpu(), labels, k=2
        )

        if lift_eval > best_lift:
            best_lift = lift_eval
            torch.save({
                "model_state": model.state_dict(),
                "metrics_head_state": metrics_head.state_dict(),
                "args": vars(args),
            }, output_dir / "encoder_best.pt")

        # Cosine LR schedule
        new_lr = cosine_lr_schedule(epoch, args.epochs, 3e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{args.epochs} loss={losses[-1]:.4f} lift={lift_eval:.4f} best_lift={best_lift:.4f} lr={new_lr:.6f}", flush=True)

        write_metrics(output_dir / "metrics.json", {
            "dataset_type": "collatz_contrastive_embeddings_v4",
            "tool": "research/contrastive_train_v4.py",
            "status": "running",
            "device": str(device),
            "embedding_count": len(starts),
            "feature_set": args.feature_set,
            "primary_label": args.primary_label,
            "pair_mode": args.pair_mode,
            "epoch": epoch,
            "epochs": args.epochs,
            "loss_start": losses[0],
            "loss_final": losses[-1],
            "lift": lift_eval,
            "best_lift": best_lift,
            "temperature": args.temperature,
            "hard_negatives": args.hard_negatives,
            "metrics_loss_weight": args.metrics_loss_weight,
        })

    # Final eval
    checkpoint = torch.load(output_dir / "encoder_best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if "metrics_head_state" in checkpoint:
        metrics_head.load_state_dict(checkpoint["metrics_head_state"])
    model.eval()
    metrics_head.eval()
    with torch.no_grad():
        all_z = []
        for b_start in range(0, len(starts), args.batch_size):
            b_end = min(b_start + args.batch_size, len(starts))
            if b_end - b_start < 8:
                continue
            b_idx = torch.arange(b_start, b_end, device=device)
            view = {name: tensor[b_idx] for name, tensor in inputs.items()}
            z = model(view, 0.0)
            all_z.append(z)
        z = torch.cat(all_z, dim=0)

    neighbor_purity, random_baseline, purity_lift = evaluate_neighbors(z, labels, k=2)

    with (output_dir / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label"] + [f"e{i}" for i in range(z.shape[1])])
        for n, label, vector in zip(starts, labels, z.tolist()):
            writer.writerow([n, label] + [f"{v:.9g}" for v in vector])

    torch.save({
        "model_state": model.state_dict(),
        "metrics_head_state": metrics_head.state_dict(),
        "args": vars(args),
    }, output_dir / "encoder.pt")

    final_metrics = {
        "dataset_type": "collatz_contrastive_embeddings_v4",
        "tool": "research/contrastive_train_v4.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": len(starts),
        "feature_set": args.feature_set,
        "primary_label": args.primary_label,
        "pair_mode": args.pair_mode,
        "input_dims": {n: input_dims[n] for n in sorted(input_dims)},
        "embedding_dims": args.embedding_dims,
        "label_count": len(set(labels)),
        "epochs": args.epochs,
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "loss_history": losses,
        "neighbor_purity": neighbor_purity,
        "random_baseline_purity": random_baseline,
        "purity_lift": purity_lift,
        "best_lift": best_lift,
        "hard_negative_count": int(mining_buffer["neg_indices"].numel()) if mining_buffer["neg_indices"] is not None else 0,
        "hard_negative_match_rate": 0.0,
        "metrics_loss_weight": args.metrics_loss_weight,
        "temperature": args.temperature,
        "n_folds": None, "n_seeds": None, "ci_95": None,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    write_metrics(output_dir / "metrics.json", final_metrics)

    print(f"\ncontrastive_v4 feature_set={args.feature_set}", flush=True)
    print(f"  pair_mode={args.pair_mode} rows={len(starts)} lift={purity_lift:.4f} best_lift={best_lift:.4f}", flush=True)
    print(f"  hard_negatives={args.hard_negatives} metrics_loss_weight={args.metrics_loss_weight}", flush=True)
    print(f"  loss history: start={losses[0]:.4f} final={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
