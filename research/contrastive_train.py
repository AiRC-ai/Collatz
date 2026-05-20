#!/usr/bin/env python3
"""Train a first-pass contrastive Collatz path embedding model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive embeddings over Collatz feature exports.")
    parser.add_argument("--metrics", default="/work/data/generated/ml/metrics.csv")
    parser.add_argument("--parity-runs", default="/work/data/generated/ml/parity_runs.csv")
    parser.add_argument("--transitions", default="/work/data/generated/ml/residue_transitions_mod32.csv")
    parser.add_argument("--sample", default="/work/data/generated/stratified/samples.csv")
    parser.add_argument("--output-dir", default="/work/data/generated/contrastive")
    parser.add_argument("--limit", type=int, default=int(os.getenv("CONTRASTIVE_LIMIT", "25000")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("CONTRASTIVE_EPOCHS", "40")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("CONTRASTIVE_BATCH_SIZE", "512")))
    parser.add_argument("--hidden-dims", type=int, default=int(os.getenv("CONTRASTIVE_HIDDEN_DIMS", "192")))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("CONTRASTIVE_EMBEDDING_DIMS", "64")))
    parser.add_argument("--token-bins", type=int, default=int(os.getenv("CONTRASTIVE_TOKEN_BINS", "64")))
    parser.add_argument(
        "--feature-set",
        choices=("hybrid", "metrics", "parity", "residue", "tokens"),
        default=os.getenv("CONTRASTIVE_FEATURE_SET", "hybrid"),
        help="Feature family to train on: hybrid, metrics, parity, residue, or parity+residue tokens.",
    )
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def read_sample(path: Path) -> tuple[set[int] | None, dict[int, str]]:
    if not path.exists():
        return None, {}
    selected: set[int] = set()
    labels: dict[int, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            reason = row.get("reasons", "sample") or "sample"
            selected.add(n)
            labels[n] = reason.split("|", 1)[0]
    return selected, labels


def read_token_hist(path: Path, selected: set[int] | None, bins: int) -> dict[int, list[float]]:
    histograms: dict[int, list[float]] = {}
    if not path.exists():
        return histograms
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if selected is not None and n not in selected:
                continue
            hist = [0.0] * bins
            tokens = row.get("tokens", "")
            if tokens:
                count = 0
                for item in tokens.split(";"):
                    if not item:
                        continue
                    token = int(item)
                    hist[token % bins] += 1.0
                    count += 1
                if count:
                    hist = [value / count for value in hist]
            histograms[n] = hist
    return histograms


def read_metrics(
    path: Path,
    selected: set[int] | None,
    labels_by_n: dict[int, str],
    parity: dict[int, list[float]],
    transitions: dict[int, list[float]],
    limit: int,
    token_bins: int,
    feature_set: str,
) -> tuple[list[int], list[str], torch.Tensor]:
    starts: list[int] = []
    labels: list[str] = []
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        metric_fields = [field for field in reader.fieldnames or [] if field.startswith("m")]
        for row in reader:
            n = int(row["n"])
            if selected is not None and n not in selected:
                continue
            feature: list[float] = []
            if feature_set in ("hybrid", "metrics"):
                feature.extend(float(row[field]) for field in metric_fields)
            if feature_set in ("hybrid", "parity", "tokens"):
                feature.extend(parity.get(n, [0.0] * token_bins))
            if feature_set in ("hybrid", "residue", "tokens"):
                feature.extend(transitions.get(n, [0.0] * token_bins))
            if not feature:
                raise RuntimeError(f"feature set produced no columns: {feature_set}")
            starts.append(n)
            labels.append(labels_by_n.get(n, "metric_export"))
            rows.append(feature)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("no rows available for contrastive training")
    x = torch.tensor(rows, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = torch.nan_to_num((x - mean) / std)
    return starts, labels, x


class Encoder(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int, embedding_dims: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.GELU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, embedding_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def evaluate_neighbors(embeddings: torch.Tensor, labels: list[str]) -> tuple[float, float, float]:
    if embeddings.shape[0] < 2:
        return 0.0, 0.0, 0.0
    chunk_size = max(1, int(os.getenv("CONTRASTIVE_EVAL_CHUNK", "2048")))
    eval_device_setting = os.getenv("CONTRASTIVE_EVAL_DEVICE", "auto")
    eval_device = torch.device(
        "cuda" if eval_device_setting != "cpu" and torch.cuda.is_available() else "cpu"
    )
    embeddings = embeddings.to(eval_device)
    nearest: list[int] = []
    all_embeddings = embeddings.T.contiguous()
    for start in range(0, embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, embeddings.shape[0])
        sim = embeddings[start:end] @ all_embeddings
        rows = torch.arange(end - start, device=eval_device)
        cols = torch.arange(start, end, device=eval_device)
        sim[rows, cols] = -2.0
        nearest.extend(sim.argmax(dim=1).cpu().tolist())
    same = sum(1 for i, j in enumerate(nearest) if labels[i] == labels[j])
    purity = same / len(labels)
    counts = Counter(labels)
    random_baseline = sum((count / len(labels)) ** 2 for count in counts.values())
    return purity, random_baseline, purity - random_baseline


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
    metrics_path = output_dir / "metrics.json"

    selected, labels_by_n = read_sample(Path(args.sample))
    parity = read_token_hist(Path(args.parity_runs), selected, args.token_bins)
    transitions = read_token_hist(Path(args.transitions), selected, args.token_bins)
    starts, labels, x_cpu = read_metrics(
        Path(args.metrics),
        selected,
        labels_by_n,
        parity,
        transitions,
        args.limit,
        args.token_bins,
        args.feature_set,
    )
    x = x_cpu.to(device)
    model = Encoder(x.shape[1], args.hidden_dims, args.embedding_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    losses: list[float] = []
    indices = torch.arange(x.shape[0], device=device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = indices[torch.randperm(x.shape[0], device=device)]
        epoch_losses: list[float] = []
        for start in range(0, x.shape[0], args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            batch = x[batch_idx]
            if batch.shape[0] < 2:
                continue
            noise_scale = 0.025
            view1 = batch + torch.randn_like(batch) * noise_scale
            view2 = batch + torch.randn_like(batch) * noise_scale
            z1 = model(view1)
            z2 = model(view2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        write_metrics(
            metrics_path,
            {
                "dataset_type": "collatz_contrastive_embeddings",
                "tool": "research/contrastive_train.py",
                "status": "running",
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "embedding_count": len(starts),
                "feature_set": args.feature_set,
                "input_dims": x.shape[1],
                "embedding_dims": args.embedding_dims,
                "epoch": epoch,
                "epochs": args.epochs,
                "loss_start": losses[0],
                "loss_final": losses[-1],
                "loss_history": losses,
            },
        )

    model.eval()
    with torch.no_grad():
        z = model(x).detach().cpu()
    neighbor_purity, random_baseline, purity_lift = evaluate_neighbors(z, labels)

    embeddings_path = output_dir / "embeddings.csv"
    with embeddings_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label", *[f"e{i}" for i in range(z.shape[1])]])
        for n, label, vector in zip(starts, labels, z.tolist()):
            writer.writerow([n, label, *[f"{value:.9g}" for value in vector]])

    checkpoint_path = output_dir / "encoder.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, checkpoint_path)

    metrics = {
        "dataset_type": "collatz_contrastive_embeddings",
        "tool": "research/contrastive_train.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_count": len(starts),
        "feature_set": args.feature_set,
        "input_dims": x.shape[1],
        "embedding_dims": args.embedding_dims,
        "label_count": len(set(labels)),
        "epochs": args.epochs,
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "loss_history": losses,
        "neighbor_purity": neighbor_purity,
        "random_baseline_purity": random_baseline,
        "purity_lift": purity_lift,
        "outputs": {"embeddings": "embeddings.csv", "checkpoint": "encoder.pt"},
    }
    write_metrics(metrics_path, metrics)
    print(
        f"contrastive feature_set={metrics['feature_set']} device={metrics['device']} rows={metrics['embedding_count']} "
        f"purity={neighbor_purity:.4f} baseline={random_baseline:.4f} lift={purity_lift:.4f}"
    )


if __name__ == "__main__":
    main()
