#!/usr/bin/env python3
"""Train Collatz path-family embeddings v3: self-noise + family pairs, no gate.

Key changes from v2:
    - No branch gating (v2 gating was counterproductive)
    - Self-noise pairs for all items (noise invariance)
    - Coalescence family pairs for family structure
    - Standard v1 hyperparams (64-dim emb, 0.07 temp)
    - No hard negative loss
    - No sparsity penalty
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ALL_BRANCHES = ("metrics", "shape", "parity", "residue")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Collatz embeddings with clean signal (v3)."
     )
    parser.add_argument("--metrics-safe", default="/home/ryancox/3xN1/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--log-sketch", default="/home/ryancox/3xN1/data/generated/ml_stratified/log_sketch.csv")
    parser.add_argument("--parity-runs", default="/home/ryancox/3xN1/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/home/ryancox/3xN1/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--families", default="/home/ryancox/3xN1/data/generated/ml_labels/families.csv")
    parser.add_argument("--positive-pairs", default="/home/ryancox/3xN1/data/generated/ml_pairs/positive_pairs.csv")
    parser.add_argument("--hard-negatives", default="/home/ryancox/3xN1/data/generated/ml_pairs/hard_negatives.csv")
    parser.add_argument("--pair-metrics", default="/home/ryancox/3xN1/data/generated/ml_pairs/metrics.json")
    parser.add_argument("--output-dir", default="/home/ryancox/3xN1/data/generated/contrastive_v3")
    parser.add_argument("--pair-mode", choices=("self_noise", "family_pairs", "combined"), default="combined")
    parser.add_argument("--primary-label", default="tail_hash",
        choices=("tail_hash", "coalescence_family_id", "parity_motif_hash",
                  "residue_motif_hash", "range_band"))
    parser.add_argument("--feature-set", choices=("hybrid", "metrics", "shape",
                    "parity-sequence", "residue-sequence"), default="hybrid")
    parser.add_argument("--branch-gate", action="store_true", default=False)
    parser.add_argument("--no-branch-gate", action="store_false", dest="branch_gate")
    parser.add_argument("--gated-hybrid-branches", type=str, default="metrics,shape,parity,residue")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use-v1-encoder", action="store_true", default=False,
        help="Use single-branch encoder (v1 architecture) instead of multi-branch")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dims", type=int, default=192)
    parser.add_argument("--embedding-dims", type=int, default=64)
    parser.add_argument("--sequence-len", type=int, default=128)
    parser.add_argument("--representation-dropout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser.parse_args()


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
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
         )

    def forward(self, x, rep_dropout):
        if rep_dropout > 0 and self.training:
            x = F.dropout(x, p=rep_dropout, training=True)
        return self.net(x)




class V1Encoder(nn.Module):
    """Single-branch encoder (v1 architecture) — one encoder for all features."""
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
         # inputs is a dict like {name: tensor} — concatenate all feature tensors
        x = torch.cat([inputs[name] for name in self.branch_names], dim=1)
        if rep_dropout > 0 and self.training:
            x = F.dropout(x, p=rep_dropout, training=True)
        return F.normalize(self.net(x), dim=1)

class FamilyEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dim, embedding_dim, gated=False,
                 branch_names=None):
        super().__init__()
        self.branch_names = branch_names or list(input_dims.keys())
        self.embedding_dim = embedding_dim
        self.gated = gated and len(self.branch_names) > 1
        self.branches = nn.ModuleDict()
        for name in self.branch_names:
            self.branches[name] = BranchEncoder(
                input_dims[name], hidden_dim, embedding_dim
             )
        if self.gated:
            self.gate_logit = nn.Parameter(torch.zeros(len(self.branch_names)))

    def forward(self, inputs, rep_dropout):
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


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    branch_names = active_branches(args.feature_set)
    gated = args.branch_gate and len(branch_names) > 1 and args.feature_set == "hybrid"
    print(f"Branches: {branch_names} | Gated: {gated} | Pair mode: {args.pair_mode}")

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
                               gated=gated, branch_names=branch_names)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    if gated:
        gw = torch.softmax(model.gate_logit, dim=0).detach().cpu().tolist()
        for name, w in zip(model.branch_names, gw):
            print(f"  Initial branch weight: {name} = {w:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    inputs = {k: t.to(device) for k, t in inputs.items()}

     # Load positive pairs
    pos_a, pos_b = [], []
    if Path(args.positive_pairs).exists():
        with open(args.positive_pairs) as f:
            reader = csv.DictReader(f)
            pair_type_counts = {}
            n_to_idx = {n: i for i, n in enumerate(starts)}
            for row in reader:
                na, nb = int(row["n_a"].strip()), int(row["n_b"].strip())
                pt = row.get("pair_type", "unknown").strip()

                # combined mode: self-noise + coalescence family pairs
                if args.pair_mode == "combined":
                    keep = pt == "same_n_different_view" or pt in ("same_coalescence_family", "same_tail_hash")
                elif args.pair_mode == "self_noise":
                    keep = pt == "same_n_different_view"
                elif args.pair_mode == "family_pairs":
                    keep = pt in ("same_coalescence_family", "same_tail_hash")
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

     # Training loop
    losses = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(starts), device=device)
        epoch_losses = []

        # Sample family pairs for this epoch
        num_pairs = len(starts)
        pair_perm = torch.randperm(len(pos_a), device=device)

        for bs in range(0, len(starts), args.batch_size):
            be = min(bs + args.batch_size, len(starts))
            if be - bs < 8:
                continue

             # Self-noise pairs
            b_idx = perm[bs:be].to(device)
            view_a = {name: tensor[b_idx] for name, tensor in inputs.items()}
            view_b = {name: value + torch.randn_like(value) * 0.025
                      for name, value in view_a.items()}

            z_a = model(view_a, args.representation_dropout)
            z_b = model(view_b, args.representation_dropout)
            loss_self = info_nce(z_a, z_b, temperature=args.temperature)

             # Family pairs from pos_loader
            loss_family = None
            if args.pair_mode in ("family_pairs", "combined") and len(pos_a) > 0:
                try:
                    fa, fb = next(iter(pos_loader))
                    fa, fb = fa.to(device), fb.to(device)
                    fam_a = {name: tensor[fa] for name, tensor in inputs.items()}
                    fam_b = {name: tensor[fb] for name, tensor in inputs.items()}
                    z_fa = model(fam_a, args.representation_dropout)
                    z_fb = model(fam_b, args.representation_dropout)
                    loss_family = info_nce(z_fa, z_fb, temperature=args.temperature)
                except StopIteration:
                    pair_perm = torch.randperm(len(pos_a), device=device)

             # Combine losses
            if loss_family is not None:
                loss = loss_self + loss_family
            else:
                loss = loss_self

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
                    print(f"       {name}: {w:.4f}")

        write_metrics(output_dir / "metrics.json", {
             "dataset_type": "collatz_contrastive_embeddings_v3",
             "tool": "research/contrastive_train_v3.py",
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
             "temperature": args.temperature,
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
         "dataset_type": "collatz_contrastive_embeddings_v3",
         "tool": "research/contrastive_train_v3.py",
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
         "pair_type_distribution": pair_type_counts if 'pair_type_counts' in dir() else {},
         "hard_negative_count": 0,
         "hard_negative_match_rate": 0.0,
         "matched_controls": {},
         "temperature": args.temperature,
         "n_folds": None, "n_seeds": None, "ci_95": None,
         "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
     }
    write_metrics(output_dir / "metrics.json", final_metrics)

    print(f"\ncontrastive_v3 feature_set={args.feature_set} branch_gated={gated}", flush=True)
    print(f"  pair_mode={args.pair_mode} rows={len(starts)} purity_lift={purity_lift:.4f}", flush=True)
    if gated:
        gw = torch.softmax(model.gate_logit, dim=0).detach().cpu().tolist()
        for name, w in zip(model.branch_names, gw):
            print(f"  learned branch weight: {name} = {w:.4f}")


if __name__ == "__main__":
    main()
