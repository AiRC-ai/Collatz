#!/usr/bin/env python3
"""Train Collatz path-family embeddings from safe metrics, shape sketches, and family pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BRANCHES = ("metrics", "shape", "parity", "residue")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v2 contrastive Collatz path-family embeddings.")
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
    parser.add_argument(
        "--primary-label",
        default=os.getenv("CONTRASTIVE_PRIMARY_LABEL", "tail_hash"),
        choices=("tail_hash", "coalescence_family_id", "parity_motif_hash", "residue_motif_hash", "range_band"),
        help="Family label used for neighbor-purity reporting. Default tests shared-tail structure.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("hybrid", "metrics", "shape", "parity-sequence", "residue-sequence"),
        default=os.getenv("CONTRASTIVE_FEATURE_SET", "hybrid"),
    )
    parser.add_argument("--limit", type=int, default=int(os.getenv("CONTRASTIVE_LIMIT", "0")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("CONTRASTIVE_EPOCHS", "40")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CONTRASTIVE_BATCH_SIZE", "512")))
    parser.add_argument("--hidden-dims", type=int, default=int(os.getenv("CONTRASTIVE_HIDDEN_DIMS", "192")))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("CONTRASTIVE_EMBEDDING_DIMS", "64")))
    parser.add_argument("--sequence-len", type=int, default=int(os.getenv("CONTRASTIVE_SEQUENCE_LEN", "128")))
    parser.add_argument("--representation-dropout", type=float, default=float(os.getenv("CONTRASTIVE_REP_DROPOUT", "0.20")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("CONTRASTIVE_SEED", "20260520")))
    return parser.parse_args()


def read_matrix(path: Path, prefix: str, limit: int) -> tuple[list[int], torch.Tensor]:
    starts: list[int] = []
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in reader.fieldnames or [] if field.startswith(prefix)]
        for row in reader:
            starts.append(int(row["n"]))
            rows.append([float(row[field]) for field in fields])
            if limit > 0 and len(starts) >= limit:
                break
    if not rows:
        raise RuntimeError(f"no rows loaded from {path}")
    return starts, standardize(torch.tensor(rows, dtype=torch.float32))


def standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return torch.nan_to_num((x - mean) / std)


def read_token_sequence(path: Path, starts: list[int], seq_len: int) -> torch.Tensor:
    wanted = set(starts)
    by_n: dict[int, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n not in wanted:
                continue
            values: list[float] = []
            for item in (row.get("tokens", "") or "").split(";"):
                if item:
                    values.append(float(int(item) & 0xffff) / 65535.0)
                if len(values) >= seq_len:
                    break
            values.extend([0.0] * (seq_len - len(values)))
            by_n[n] = values
    return standardize(torch.tensor([by_n.get(n, [0.0] * seq_len) for n in starts], dtype=torch.float32))


def read_families(path: Path, starts: list[int], primary_label: str) -> tuple[list[str], dict[int, dict[str, str]]]:
    wanted = set(starts)
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n in wanted:
                rows[n] = row
    labels = [rows.get(n, {}).get(primary_label, "unknown") for n in starts]
    return labels, rows


def read_pairs(path: Path, index_by_n: dict[int, int]) -> list[tuple[int, int, str]]:
    if not path.exists():
        return []
    pairs: list[tuple[int, int, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            a = int(row["n_a"])
            b = int(row["n_b"])
            if a in index_by_n and b in index_by_n:
                pairs.append((index_by_n[a], index_by_n[b], row.get("pair_type", "family_pair")))
    return pairs


def read_hard_negatives(path: Path, index_by_n: dict[int, int]) -> dict[int, list[int]]:
    negatives: dict[int, list[int]] = {}
    if not path.exists():
        return negatives
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            neg = int(row["negative_n"])
            if n in index_by_n and neg in index_by_n:
                negatives.setdefault(index_by_n[n], []).append(index_by_n[neg])
    return negatives


class BranchEncoder(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.GELU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FamilyEncoder(nn.Module):
    def __init__(self, dims: dict[str, int], hidden_dims: int, embedding_dims: int, active: set[str]) -> None:
        super().__init__()
        self.active = active
        self.branches = nn.ModuleDict({name: BranchEncoder(dim, hidden_dims) for name, dim in dims.items() if name in active})
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims * len(self.branches), hidden_dims),
            nn.GELU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, embedding_dims),
        )

    def forward(self, inputs: dict[str, torch.Tensor], drop_prob: float = 0.0) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for name, branch in self.branches.items():
            encoded = branch(inputs[name])
            if self.training and drop_prob > 0.0 and len(self.branches) > 1:
                keep = torch.rand((encoded.shape[0], 1), device=encoded.device) > drop_prob
                encoded = encoded * keep
            outputs.append(encoded)
        return F.normalize(self.projection(torch.cat(outputs, dim=1)), dim=1)


def active_branches(feature_set: str) -> set[str]:
    if feature_set == "hybrid":
        return set(BRANCHES)
    if feature_set == "parity-sequence":
        return {"parity"}
    if feature_set == "residue-sequence":
        return {"residue"}
    return {feature_set}


def info_nce(z_anchor: torch.Tensor, z_pos: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    logits = z_anchor @ z_pos.T / temperature
    labels = torch.arange(z_anchor.shape[0], device=z_anchor.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def hard_negative_loss(z_anchor: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    pos = (z_anchor * z_pos).sum(dim=1, keepdim=True)
    neg = (z_anchor * z_neg).sum(dim=1, keepdim=True)
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(z_anchor.shape[0], dtype=torch.long, device=z_anchor.device)
    return F.cross_entropy(logits, labels)


def evaluate_neighbors(embeddings: torch.Tensor, labels: list[str]) -> tuple[float, float, float]:
    if embeddings.shape[0] < 2:
        return 0.0, 0.0, 0.0
    x = embeddings
    sim = x @ x.T
    sim.fill_diagonal_(-2.0)
    nearest = sim.argmax(dim=1).tolist()
    purity = sum(1 for i, j in enumerate(nearest) if labels[i] == labels[j]) / len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    baseline = sum((count / len(labels)) ** 2 for count in counts.values())
    return purity, baseline, purity - baseline


def write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    starts, metrics = read_matrix(Path(args.metrics_safe), "m", args.limit)
    starts_shape, shape = read_matrix(Path(args.log_sketch), "s", args.limit)
    if starts_shape != starts:
        raise RuntimeError("metrics_safe and log_sketch start rows are not aligned")
    parity = read_token_sequence(Path(args.parity_runs), starts, args.sequence_len)
    residue = read_token_sequence(Path(args.transitions), starts, args.sequence_len)
    labels, _families = read_families(Path(args.families), starts, args.primary_label)
    index_by_n = {n: i for i, n in enumerate(starts)}
    pairs = read_pairs(Path(args.positive_pairs), index_by_n) if args.pair_mode == "family_pairs" else []
    negatives = read_hard_negatives(Path(args.hard_negatives), index_by_n)
    if args.pair_mode == "family_pairs" and not pairs:
        raise RuntimeError("family_pairs mode requires positive_pairs.csv")

    inputs_cpu = {"metrics": metrics, "shape": shape, "parity": parity, "residue": residue}
    inputs = {name: value.to(device) for name, value in inputs_cpu.items()}
    dims = {name: value.shape[1] for name, value in inputs_cpu.items()}
    model = FamilyEncoder(dims, args.hidden_dims, args.embedding_dims, active_branches(args.feature_set)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    pair_items = pairs if pairs else [(i, i, "self_noise") for i in range(len(starts))]
    losses: list[float] = []
    pair_type_counts: dict[str, int] = {}
    for _a, _b, kind in pair_items:
        pair_type_counts[kind] = pair_type_counts.get(kind, 0) + 1

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pair_items)
        epoch_losses: list[float] = []
        for start in range(0, len(pair_items), args.batch_size):
            batch = pair_items[start : start + args.batch_size]
            if len(batch) < 2:
                continue
            a_idx = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            b_idx = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            view_a = {name: tensor[a_idx] for name, tensor in inputs.items()}
            view_b = {name: tensor[b_idx] for name, tensor in inputs.items()}
            if args.pair_mode == "self_noise":
                view_b = {name: value + torch.randn_like(value) * 0.025 for name, value in view_a.items()}
            z_a = model(view_a, args.representation_dropout)
            z_b = model(view_b, args.representation_dropout)
            loss = info_nce(z_a, z_b)

            hard_a: list[int] = []
            hard_b: list[int] = []
            for anchor in a_idx.detach().cpu().tolist():
                candidates = negatives.get(anchor, [])
                if candidates:
                    hard_a.append(anchor)
                    hard_b.append(random.choice(candidates))
            if hard_a:
                anchor_idx = torch.tensor(hard_a, dtype=torch.long, device=device)
                neg_idx = torch.tensor(hard_b, dtype=torch.long, device=device)
                pos_idx = anchor_idx
                z_anchor = model({name: tensor[anchor_idx] for name, tensor in inputs.items()}, args.representation_dropout)
                z_pos = model({name: tensor[pos_idx] for name, tensor in inputs.items()}, args.representation_dropout)
                z_neg = model({name: tensor[neg_idx] for name, tensor in inputs.items()}, args.representation_dropout)
                loss = loss + hard_negative_loss(z_anchor, z_pos, z_neg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        write_metrics(
            output_dir / "metrics.json",
            {
                "dataset_type": "collatz_contrastive_embeddings_v2",
                "tool": "research/contrastive_train_v2.py",
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
            },
        )

    model.eval()
    with torch.no_grad():
        z = model(inputs, 0.0).detach().cpu()
    neighbor_purity, random_baseline, purity_lift = evaluate_neighbors(z, labels)

    with (output_dir / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label", *[f"e{i}" for i in range(z.shape[1])]])
        for n, label, vector in zip(starts, labels, z.tolist()):
            writer.writerow([n, label, *[f"{value:.9g}" for value in vector]])
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, output_dir / "encoder.pt")

    pair_metrics = {}
    if Path(args.pair_metrics).exists():
        with Path(args.pair_metrics).open() as handle:
            pair_metrics = json.load(handle)
    metrics_out = {
        "dataset_type": "collatz_contrastive_embeddings_v2",
        "tool": "research/contrastive_train_v2.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": len(starts),
        "feature_set": args.feature_set,
        "primary_label": args.primary_label,
        "pair_mode": args.pair_mode,
        "input_dims": {name: dims[name] for name in sorted(active_branches(args.feature_set))},
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
            "bit_length": False,
            "range_band": False,
            "residue_class": False,
            "stopping_time_bucket": False,
            "peak_ratio_bucket": False,
            "first_drop_bucket": False,
        }),
        "n_folds": None,
        "n_seeds": None,
        "ci_95": None,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    write_metrics(output_dir / "metrics.json", metrics_out)
    print(
        f"contrastive_v2 feature_set={args.feature_set} pair_mode={args.pair_mode} "
        f"rows={len(starts)} lift={purity_lift:.4f}"
    )


if __name__ == "__main__":
    main()
