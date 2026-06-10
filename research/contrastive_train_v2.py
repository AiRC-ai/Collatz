#!/usr/bin/env python3
"""Train Collatz path-family embeddings with learnable branch gating.

Replaces the flat concatenation approach with gating so the model can
learn to upweight informative branches (metrics, residue) and downweight
weak ones (parity, shape).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


ALL_BRANCHES = ("metrics", "shape", "parity", "residue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Collatz path-family embeddings with branch gating."
    )
    parser.add_argument("--metrics-safe", default="/work/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/work/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/work/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/work/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/work/data/generated/ml_labels/families.csv")
    parser.add_argument("--positive-pairs", default="/work/data/generated/ml_pairs/positive_pairs.csv")
    parser.add_argument("--hard-negatives", default="/work/data/generated/ml_pairs/hard_negatives.csv")
    parser.add_argument("--pair-metrics", default="/work/data/generated/ml_pairs/metrics.json")
    parser.add_argument("--output-dir", default="/work/data/generated/contrastive_v2")
    parser.add_argument("--pair-mode", choices=("family_pairs", "self_noise"), default=os.getenv("CONTRASTIVE_PAIR_MODE", "family_pairs"))
    parser.add_argument("--primary-label",
        default=os.getenv("CONTRASTIVE_PRIMARY_LABEL", "tail_hash"),
        choices=("tail_hash", "coalescence_family_id", "parity_motif_hash",
                 "residue_motif_hash", "range_band"))
    parser.add_argument("--feature-set",
        choices=("hybrid", "metrics", "shape", "parity-sequence", "residue-sequence"),
        default=os.getenv("CONTRASTIVE_FEATURE_SET", "hybrid"))
    parser.add_argument("--branch-gate", action="store_true",
        default=os.getenv("CONTRASTIVE_BRANCH_GATE", "1") == "1")
    parser.add_argument("--gated-hybrid-branches", type=str,
        default=os.getenv("CONTRASTIVE_GATED_BRANCHES", "metrics,shape,parity,residue"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("CONTRASTIVE_LIMIT", "0")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("CONTRASTIVE_EPOCHS", "40")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CONTRASTIVE_BATCH_SIZE", "512")))
    parser.add_argument("--hidden-dims", type=int, default=int(os.getenv("CONTRASTIVE_HIDDEN_DIMS", "192")))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("CONTRASTIVE_EMBEDDING_DIMS", "64")))
    parser.add_argument("--sequence-len", type=int, default=int(os.getenv("CONTRASTIVE_SEQUENCE_LEN", "128")))
    parser.add_argument("--representation-dropout", type=float,
        default=float(os.getenv("CONTRASTIVE_REP_DROPOUT", "0.20")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("CONTRASTIVE_SEED", "20260520")))
    return parser.parse_args()


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

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


def standardize(x):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.nan_to_num((x - mean) / std)


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
                if len(values) >= seq_len:
                    break
            values.extend([0.0] * (seq_len - len(values)))
            by_n[n] = values
    return standardize(torch.tensor(
        [by_n.get(n, [0.0] * seq_len) for n in starts], dtype=torch.float32))


def read_families(path, starts, primary_label):
    wanted = set(starts)
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n not in wanted:
                continue
            rows[n] = row
    labels = []
    for n in starts:
        labels.append(rows.get(n, {}).get(primary_label, str(n)))
    return labels, rows


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class BranchEncoder(nn.Module):
    """Single-branch encoder: input -> hidden -> embedding."""
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, rep_dropout=0.0):
        z = self.net(x)
        if rep_dropout > 0.0:
            z = F.dropout(z, p=rep_dropout, training=self.training)
        return F.normalize(z, dim=-1)


class FamilyEncoder(nn.Module):
    """Multi-branch encoder with optional learnable gating."""
    def __init__(self, branch_configs, hidden_dim, embedding_dim, gated=True):
        super().__init__()
        self.branches = nn.ModuleDict()
        for name, input_dim in branch_configs.items():
            self.branches[name] = BranchEncoder(input_dim, hidden_dim, embedding_dim)
        self.branch_names = list(branch_configs.keys())
        self.gated = gated
        if gated and len(self.branches) > 1:
            self.gate_logit = nn.Parameter(torch.zeros(len(self.branch_names)))
        else:
            self.gate_logit = None
        self.output_dim = embedding_dim

    def forward(self, inputs, rep_dropout=0.0):
        branch_outs = []
        for name in self.branch_names:
            if name not in inputs:
                continue
            z = self.branches[name](inputs[name], rep_dropout)
            branch_outs.append(z)

        if self.gated and len(branch_outs) > 1:
            logits = self.gate_logit.to(branch_outs[0].device)
            weights = F.softmax(logits, dim=0)
            gated_branches = []
            for i, z in enumerate(branch_outs):
                w = weights[i].view(1, -1)
                gated_branches.append(z * w)
            combined = torch.cat(gated_branches, dim=-1)
        else:
            combined = torch.cat(branch_outs, dim=-1)

        projected = F.layer_norm(combined, (combined.shape[-1],))
        projected = F.gelu(projected)
        projected = F.dropout(projected, p=0.1, training=self.training)
        return F.normalize(projected, dim=-1)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------

def info_nce(z_a, z_b, temperature=0.07):
    batch_size = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    sim = torch.matmul(z, z.transpose(0, 1)) / temperature
    sim_ij = sim[:batch_size, batch_size:]
    sim_ji = sim[batch_size:, :batch_size]
    pos = torch.diag(sim[:batch_size, batch_size:])
    labels = torch.arange(batch_size, device=sim.device)
    loss_i = F.cross_entropy(torch.cat([sim_ij, pos.unsqueeze(1)], dim=1), labels)
    loss_j = F.cross_entropy(torch.cat([sim_ji, pos.unsqueeze(1)], dim=1), labels)
    return (loss_i + loss_j) / 2


def hard_negative_loss(anchor, positive, negative, temperature=0.07):
    pos_sim = torch.sum(anchor * positive, dim=1).mean() / temperature
    neg_sim = torch.sum(anchor * negative, dim=1).mean() / temperature
    return -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))).mean()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_neighbors(embeddings, labels):
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)
    k_min = 2
    purities = []
    for i, label in enumerate(labels):
        same = [j for j in label_to_indices[label] if j != i]
        if not same:
            continue
        sim = (embeddings @ embeddings[i]).cpu()
        top_k = sim.topk(k_min + 1, dim=0).indices
        top_k = top_k[top_k != i].tolist()
        purities.append(sum(1 for j in top_k if labels[j] == label) / len(top_k))
    if not purities:
        return 0.0, 0.0, 0.0
    neighbor_purity = sum(purities) / len(purities)
    label_counts = [len(v) for v in label_to_indices.values() if len(v) > 1]
    if label_counts:
        random_baseline = sum(c / (c - 1) for c in label_counts) / len(label_counts) * (k_min - 1) / k_min
    else:
        random_baseline = 0.0
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_names = active_branches(args.feature_set)
    print(f"Branches: {branch_names} | Gated: {args.branch_gate}")

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
    print(f"Loaded {len(starts)} families, {len(set(labels))} unique labels")

    # Model
    gated = args.branch_gate and len(branch_names) > 1 and args.feature_set == "hybrid"
    model = FamilyEncoder(input_dims, args.hidden_dims, args.embedding_dims, gated=gated)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    if gated:
        gw = torch.softmax(model.gate_logit, dim=0).detach().cpu().tolist()
        for name, w in zip(model.branch_names, gw):
            print(f"  Initial branch weight: {name} = {w:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    inputs = {k: t.to(device) for k, t in inputs.items()}

    # Load pairs
    neg_lookup = {}
    idx_to_n = {i: n for i, n in enumerate(starts)}
    n_to_idx = {n: i for i, n in enumerate(starts)}

    if Path(args.hard_negatives).exists():
        with open(args.hard_negatives) as f:
            reader = csv.DictReader(f)
            for row in reader:
                a, neg = int(row["n"]), int(row["negative_n"])
                ai, ni = n_to_idx.get(a), n_to_idx.get(neg)
                if ai is not None and ni is not None:
                    neg_lookup.setdefault(ai, []).append(ni)

    pair_data = {}
    pair_metrics = {}
    pair_type_counts = {}
    if Path(args.positive_pairs).exists():
        with open(args.positive_pairs) as f:
            reader = csv.DictReader(f)
            indices_a, indices_b = [], []
            pair_type_counts = {}
            for row in reader:
                na, nb = int(row["n_a"]), int(row["n_b"])
                ia, ib = n_to_idx.get(na), n_to_idx.get(nb)
                if ia is not None and ib is not None:
                    indices_a.append(ia)
                    indices_b.append(ib)
                    pt = row.get("pair_type", "unknown")
                    pair_type_counts[pt] = pair_type_counts.get(pt, 0) + 1
            if indices_a:
                pair_data = {"a": torch.tensor(indices_a, dtype=torch.long),
                             "b": torch.tensor(indices_b, dtype=torch.long)}
                print(f"Loaded {len(indices_a)} positive pairs ({pair_type_counts})")

    if Path(args.pair_metrics).exists():
        with open(args.pair_metrics) as f:
            pair_metrics = json.load(f)

    # Training loop
    losses = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(starts), device=device)
        epoch_losses = []

        for bs in range(0, len(starts), args.batch_size):
            be = min(bs + args.batch_size, len(starts))
            if be - bs < 8:
                continue

            b_idx = perm[bs:be].to(device)
            view_a = {name: tensor[b_idx] for name, tensor in inputs.items()}
            view_b = {name: value + torch.randn_like(value) * 0.025
                      for name, value in view_a.items()}

            z_a = model(view_a, args.representation_dropout)
            z_b = model(view_b, args.representation_dropout)
            loss = info_nce(z_a, z_b)

            hard_a, hard_b = [], []
            for anchor in b_idx.detach().cpu().tolist():
                candidates = neg_lookup.get(anchor, [])
                if candidates:
                    hard_a.append(anchor)
                    hard_b.append(random.choice(candidates))
            if hard_a:
                anchor_idx = torch.tensor(hard_a, dtype=torch.long, device=device)
                neg_idx = torch.tensor(hard_b, dtype=torch.long, device=device)
                z_anchor = model({name: tensor[anchor_idx] for name, tensor in inputs.items()},
                                 args.representation_dropout)
                # noise-augmented positive (same as main loop)
                z_pos_src = {name: tensor[anchor_idx] for name, tensor in inputs.items()}
                z_pos = model({name: v + torch.randn_like(v) * 0.025 for name, v in z_pos_src.items()},
                              args.representation_dropout)
                z_neg = model({name: tensor[neg_idx] for name, tensor in inputs.items()},
                              args.representation_dropout)
                loss = loss + hard_negative_loss(z_anchor, z_pos, z_neg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))

        if losses[-1] < best_loss:
            best_loss = losses[-1]
            torch.save({"model_state": model.state_dict(), "args": vars(args)},
                       output_dir / "encoder_best.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{args.epochs} loss={losses[-1]:.4f}", flush=True)
            if gated:
                gw = torch.softmax(model.gate_logit, dim=0).detach().cpu().tolist()
                for name, w in zip(model.branch_names, gw):
                    print(f"     {name}: {w:.4f}")

        write_metrics(output_dir / "metrics.json", {
            "dataset_type": "collatz_contrastive_embeddings_v2",
            "tool": "research/contrastive_train_v2.py",
            "status": "running",
            "device": str(device),
            "embedding_count": len(starts),
            "feature_set": args.feature_set,
            "primary_label": args.primary_label,
            "pair_mode": args.pair_mode,
            "branch_gated": gated,
            "epoch": epoch,
            "epochs": args.epochs,
            "loss_start": losses[0],
            "loss_final": losses[-1],
        })

    # Final eval
    checkpoint = torch.load(output_dir / "encoder_best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        z = model(inputs, 0.0).detach().cpu()

    neighbor_purity, random_baseline, purity_lift = evaluate_neighbors(z, labels)

    with (output_dir / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label"] + [f"e{i}" for i in range(z.shape[1])])
        for n, label, vector in zip(starts, labels, z.tolist()):
            writer.writerow([n, label] + [f"{v:.9g}" for v in vector])

    torch.save({"model_state": model.state_dict(), "args": vars(args)},
               output_dir / "encoder.pt")

    final_metrics = {
        "dataset_type": "collatz_contrastive_embeddings_v2",
        "tool": "research/contrastive_train_v2.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": len(starts),
        "feature_set": args.feature_set,
        "primary_label": args.primary_label,
        "pair_mode": args.pair_mode,
        "branch_gated": gated,
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
        "pair_type_distribution": pair_type_counts,
        "hard_negative_count": pair_metrics.get("hard_negative_count", 0),
        "hard_negative_match_rate": pair_metrics.get("hard_negative_match_rate", 0.0),
        "matched_controls": pair_metrics.get("matched_controls", {
            "bit_length": False, "range_band": False, "residue_class": False,
            "stopping_time_bucket": False, "peak_ratio_bucket": False,
            "first_drop_bucket": False,
        }),
        "n_folds": None, "n_seeds": None, "ci_95": None,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    write_metrics(output_dir / "metrics.json", final_metrics)

    print(f"\ncontrastive_v2 feature_set={args.feature_set} branch_gated={gated}", flush=True)
    print(f"  pair_mode={args.pair_mode} rows={len(starts)} purity_lift={purity_lift:.4f}", flush=True)
    if gated:
        gw = torch.softmax(model.gate_logit, dim=0).detach().cpu().tolist()
        for name, w in zip(model.branch_names, gw):
            print(f"  learned branch weight: {name} = {w:.4f}")


if __name__ == "__main__":
    main()
