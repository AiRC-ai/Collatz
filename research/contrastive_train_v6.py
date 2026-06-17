#!/usr/bin/env python3
"""Train Collatz embeddings v6: coarse-grained labels (range_band) with raw-metrics hard negatives.

Key insight from v4/v5 failure:
- v5 mined from raw metrics (correct direction) BUT used coalescence_family_id (61K classes, ~1.6 examples each)
- Contrastive learning needs multiple examples per class to learn discriminative features
- v5 never learned because every class had 1-2 members - no signal to distinguish

v6 changes:
- Primary label: range_band (16 classes, ~6.25K each) - learnable!
- Also support bit_length (17 classes, ~5.9K each) as alternative
- Raw metrics hard negatives (keep v5's correct insight)
- Cosine LR with warmup (from v5's built-in schedule)
- Lower metrics_loss_weight (0.15 vs 0.3) - let contrastive dominate
- Evaluate every 5 epochs (catch issues sooner)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="v6: coarse-grained labels (range_band) + raw-metrics hard negatives"
    )
    parser.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    parser.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v6")
    parser.add_argument("--primary-label", default="range_band",
        choices=("range_band", "bit_length", "peak_ratio_bucket"))
    parser.add_argument("--feature-set", choices=("hybrid", "metrics", "shape",
                          "parity-sequence", "residue-sequence"), default="hybrid")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--metrics-loss-weight", type=float, default=0.15)
    parser.add_argument("--hard-negatives", type=int, default=50)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, default=192)
    parser.add_argument("--use-v1-encoder", action="store_true", default=True)
    parser.add_argument("--representation-dropout", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--no-metrics-loss", action="store_true")
    parser.add_argument("--half-precision", action="store_true")
    return parser.parse_args()


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


def standardize(t):
    m, s = t.mean(dim=0), t.std(dim=0)
    s = s.clamp(min=1e-6)
    return (t - m) / s


def read_token_sequence(path, starts, seq_len):
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
    """Load family labels. With range_band, we get ~16 classes with ~6K examples each."""
    wanted = set(starts)
    n_to_idx = {n: i for i, n in enumerate(starts)}
    labels = [0] * len(starts)
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
            counts[label] += 1
    print(f"  Label '{primary_label}': {len(counts)} unique, top 5: {counts.most_common(5)}")
    return labels, counts


def mine_hard_negatives_raw_metrics(metrics_tensor, labels, k=50, chunk_size=10000):
    """Mine hard negatives from raw normalized metrics space.

    Labels are coarse (range_band: 16 classes), so mining excludes same-label
    items. With ~6K examples per class, there are plenty of same-label items
    to mask out, leaving a clean set of cross-class negatives.
    """
    n = metrics_tensor.shape[0]
    device = metrics_tensor.device

    # Build label-to-indices mapping
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    # Build label mask (same_label = positive, so mask it out for negatives)
    labels_t = torch.tensor([hash(str(l)) % (2**31) for l in labels], device=device)
    same_label_mask = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1))

    # Normalize metrics for cosine similarity
    metrics_normed = F.normalize(metrics_tensor, dim=1)

    # Pre-allocate result
    neg_indices = torch.full((n, k), -1, dtype=torch.long, device=device)

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_sz = chunk_end - chunk_start

        # Similarity for this chunk against all items (in metrics space)
        chunk_sim = metrics_normed[chunk_start:chunk_end] @ metrics_normed.T

        # Mask same-item diagonal
        diag = torch.arange(chunk_start, chunk_end, device=device)
        chunk_sim[torch.arange(chunk_sz), diag] = float("-inf")

        # Mask same-label entries (these are positives, not negatives)
        chunk_sim = chunk_sim.masked_fill(same_label_mask[chunk_start:chunk_end], float("-inf"))

        # Top-k hardest negatives (highest similarity but wrong label)
        k_actual = min(k, n - 1)
        _, topk_indices = chunk_sim.topk(k_actual, dim=1)
        neg_indices[chunk_start:chunk_end] = topk_indices

    return neg_indices


class V1Encoder(nn.Module):
    """Single forward pass: concat all branches through one MLP."""
    def __init__(self, branch_names, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.branch_names = branch_names
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, branches, dropout=0.0):
        x = torch.cat([branches[name] for name in self.branch_names], dim=1)
        if dropout > 0:
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


def info_nce_with_hard_negatives(z_a, z_b, hard_neg_indices, raw_negatives=None, temperature=0.05):
    """InfoNCE with hard negatives from raw metrics space."""
    batch_size = z_a.shape[0]
    device = z_a.device

    # Positives
    pos_sim = torch.sum(z_a * z_b, dim=1) / temperature

    if hard_neg_indices is not None and hard_neg_indices.shape[1] > 0:
        if raw_negatives is not None:
            neg_z = raw_negatives[hard_neg_indices]
        else:
            neg_z = z_b[hard_neg_indices]
        neg_sim = torch.sum(z_a.unsqueeze(1) * neg_z, dim=2) / temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
    else:
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
    """Preservation loss: project embedding and match to raw metrics."""
    emb_norm = F.normalize(embedding, dim=1)
    tgt_norm = F.normalize(target, dim=1)
    sim = torch.sum(emb_norm * tgt_norm, dim=1) / temperature
    loss = 1.0 - sim.mean()
    return loss


def evaluate_neighbors(embeddings, labels, k=2):
    """Measure lift: fraction of nearest neighbors with same label."""
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


def cosine_lr_schedule(epoch, total_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_names = active_branches(args.feature_set)
    print(f"Branches: {branch_names}")
    print(f"Primary label: {args.primary_label}")

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
    labels, label_counts = read_families(Path(args.families), starts, args.primary_label)
    print(f"Loaded {len(starts)} items, {len(set(labels))} unique {args.primary_label} classes")
    print(f"  Distribution: min={min(label_counts.values())}, max={max(label_counts.values())}, avg={sum(label_counts.values())/len(label_counts):.1f}")

    # Mine hard negatives from raw metrics space BEFORE training
    print("Mining hard negatives from raw metrics space...")
    metrics_gpu = m_data.to(device)
    raw_neg_indices = mine_hard_negatives_raw_metrics(metrics_gpu, labels, k=args.hard_negatives)
    print(f"  Mined {raw_neg_indices.numel()} hard negative edges (k={args.hard_negatives})")

    # Model
    if args.use_v1_encoder:
        total_input = sum(input_dims.values())
        model = V1Encoder(branch_names, total_input, args.hidden_dims, args.embedding_dims)
        print(f"Using V1 single-branch encoder, input dims: {total_input}")
    else:
        from research.contrastive_train_v5 import FamilyEncoder  # fallback
        model = FamilyEncoder(input_dims, args.hidden_dims, args.embedding_dims,
                               gated=False, branch_names=branch_names)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Metrics projection head
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

    # Generate positive pairs from coarse labels
    print("Generating positive pairs from coarse-grained labels...")
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    pos_a, pos_b, pair_type_counts = [], [], {}
    for label, indices in label_to_indices.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pos_a.append(indices[i])
                pos_b.append(indices[j])
                pair_type_counts[f"same_{args.primary_label}"] = pair_type_counts.get(f"same_{args.primary_label}", 0) + 1

    # Downsample to reasonable size (~100K pairs max)
    if len(pos_a) > 100000:
        perm = torch.randperm(len(pos_a))[:100000].tolist()
        pos_a = [pos_a[i] for i in perm]
        pos_b = [pos_b[i] for i in perm]

    pos_a = torch.tensor(pos_a, dtype=torch.long, device=device)
    pos_b = torch.tensor(pos_b, dtype=torch.long, device=device)
    print(f"Generated {len(pos_a)} positive pairs ({pair_type_counts})")

    pos_dataset = torch.utils.data.TensorDataset(pos_a, pos_b)
    pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=256, shuffle=True, drop_last=True)

    # Precompute all embeddings for mining reference
    print("Precomputing all embeddings for mining reference...")
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for b_start in range(0, len(starts), args.batch_size):
            b_end = min(b_start + args.batch_size, len(starts))
            if b_end - b_start < 8:
                continue
            b_idx = torch.arange(b_start, b_end, device=device)
            view = {name: tensor[b_idx] for name, tensor in inputs.items()}
            z = model(view, 0.0)
            all_embeddings.append(z)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"  Computed {all_embeddings.shape[0]} embeddings for mining reference")

    # Training loop
    losses = []
    best_lift = -999
    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        metrics_head.train()
        epoch_loss_accum = 0.0
        epoch_batches = 0

        perm = torch.randperm(len(starts), device=device)

        for bs in range(0, len(starts), args.batch_size):
            be = min(bs + args.batch_size, len(starts))
            if be - bs < 8:
                continue

            b_idx = perm[bs:be].to(device)

            # Use raw-metrics hard negatives
            hard_neg = raw_neg_indices[b_idx] if raw_neg_indices is not None else None

            # Compute embeddings
            view_a = {name: tensor[b_idx] for name, tensor in inputs.items()}
            z_a = model(view_a, args.representation_dropout)

            # Pair loss
            loss_pair = None
            if len(pos_a) > 0:
                try:
                    # Random offset for positive pairs
                    offset = torch.randint(0, len(pos_a) - 1, (b_idx.shape[0],), device=device)
                    idx_a = pos_a[offset].to(device)
                    idx_b = pos_b[offset].to(device)

                    z_a2 = model({name: tensor[idx_a] for name, tensor in inputs.items()}, 0.0)
                    z_b2 = model({name: tensor[idx_b] for name, tensor in inputs.items()}, 0.0)

                    loss_pair = info_nce_with_hard_negatives(z_a2, z_b2, hard_neg, all_embeddings, args.temperature)
                except Exception as e:
                    print(f"  Warning: positive pair loss failed: {e}")
                    loss_pair = None

            # Metrics preservation loss
            z_projected = F.normalize(metrics_head(z_a), dim=1)
            target = F.normalize(m_data_gpu[b_idx], dim=1)
            loss_metrics = cosine_loss(z_projected, target)

            # Combined loss
            if loss_pair is not None:
                loss = (1.0 - args.metrics_loss_weight) * loss_pair + args.metrics_loss_weight * loss_metrics
            else:
                loss = loss_metrics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_accum += float(loss.detach().cpu())
            epoch_batches += 1

        epoch_loss = epoch_loss_accum / max(epoch_batches, 1)
        losses.append(epoch_loss)

        # Evaluate every N epochs
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

        np_eval, rb_eval, lift_eval = evaluate_neighbors(z.cpu(), labels, k=2)

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
            param_group["lr"] = new_lr

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "lift": lift_eval,
            "best_lift": max(best_lift, lift_eval),
            "lr": new_lr
        })

        if (epoch + 1) % args.evaluate_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{args.epochs} loss={epoch_loss:.4f} lift={lift_eval:.4f} "
                  f"best_lift={best_lift:.4f} lr={new_lr:.6f}", flush=True)

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
        "dataset_type": "collatz_contrastive_embeddings_v6",
        "tool": "research/contrastive_train_v6.py",
        "status": "training",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": len(starts),
        "feature_set": args.feature_set,
        "primary_label": args.primary_label,
        "label_stats": {
            "unique_classes": len(set(labels)),
            "min_per_class": min(label_counts.values()) if label_counts else 0,
            "max_per_class": max(label_counts.values()) if label_counts else 0,
            "avg_per_class": sum(label_counts.values()) / len(label_counts) if label_counts else 0,
            "top5_classes": label_counts.most_common(5) if label_counts else [],
        },
        "embedding_dims": args.embedding_dims,
        "epochs": args.epochs,
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "loss_history": losses,
        "neighbor_purity": neighbor_purity,
        "random_baseline_purity": random_baseline,
        "purity_lift": purity_lift,
        "best_lift": best_lift,
        "hard_negative_source": "raw_metrics",
        "hard_negative_count": int(raw_neg_indices.numel()) if raw_neg_indices is not None else 0,
        "metrics_loss_weight": args.metrics_loss_weight,
        "temperature": args.temperature,
        "total_pairs": len(pos_a),
        "n_folds": None, "n_seeds": None, "ci_95": None,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }

    def write_metrics(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    write_metrics(output_dir / "metrics.json", final_metrics)

    print(f"\nv6 feature_set={args.feature_set}")
    print(f"  primary_label={args.primary_label} classes={len(set(labels))}")
    print(f"  pair_mode={args.primary_label}_pairs rows={len(starts)} lift={purity_lift:.4f} best_lift={best_lift:.4f}")
    print(f"  hard_negative_source=raw_metrics hard_negatives={args.hard_negatives}")
    print(f"  metrics_loss_weight={args.metrics_loss_weight}")
    print(f"  loss history: start={losses[0]:.4f} final={losses[-1]:.4f}")
    print(f"  total_pairs={len(pos_a)}")


if __name__ == "__main__":
    main()
