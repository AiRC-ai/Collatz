#!/usr/bin/env python3
"""Train image-only Collatz embeddings from 5-channel path tensors."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image-only Collatz path tensor embeddings.")
    parser.add_argument("--tensor-dir", default="/work/data/generated/image_tensors")
    parser.add_argument("--tensor", default="/work/data/generated/image_tensors/image_tensors.bin")
    parser.add_argument("--index", default="/work/data/generated/image_tensors/image_tensors_index.csv")
    parser.add_argument("--families", default="/work/data/generated/ml_labels/families.csv")
    parser.add_argument("--positive-pairs", default="/work/data/generated/ml_pairs/positive_pairs.csv")
    parser.add_argument("--hard-negatives", default="/work/data/generated/ml_pairs/hard_negatives.csv")
    parser.add_argument("--pair-metrics", default="/work/data/generated/ml_pairs/metrics.json")
    parser.add_argument("--output-dir", default="/work/data/generated/image_contrastive")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("IMAGE_CONTRASTIVE_EPOCHS", "30")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("IMAGE_CONTRASTIVE_BATCH_SIZE", "256")))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("IMAGE_CONTRASTIVE_EMBEDDING_DIMS", "64")))
    parser.add_argument("--seed", type=int, default=20260520)
    return parser.parse_args()


def read_index(path: Path) -> tuple[list[int], int, int, int]:
    starts: list[int] = []
    channels = height = width = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            starts.append(int(row["n"]))
            channels = int(row["channels"])
            height = int(row["height"])
            width = int(row["width"])
    if not starts:
        raise RuntimeError("image tensor index is empty")
    return starts, channels, height, width


def read_families(path: Path, starts: list[int]) -> list[str]:
    wanted = set(starts)
    labels: dict[int, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n in wanted:
                labels[n] = row.get("coalescence_family_id", "unknown")
    return [labels.get(n, "unknown") for n in starts]


def read_pairs(path: Path, index_by_n: dict[int, int]) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    pairs: list[tuple[int, int]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            a = int(row["n_a"])
            b = int(row["n_b"])
            if a in index_by_n and b in index_by_n:
                pairs.append((index_by_n[a], index_by_n[b]))
    return pairs


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dims: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, embedding_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return F.normalize(self.head(h), dim=1)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def evaluate(z: torch.Tensor, labels: list[str]) -> tuple[float, float, float]:
    sim = z @ z.T
    sim.fill_diagonal_(-2.0)
    nearest = sim.argmax(dim=1).tolist()
    purity = sum(1 for i, j in enumerate(nearest) if labels[i] == labels[j]) / len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    baseline = sum((count / len(labels)) ** 2 for count in counts.values())
    return purity, baseline, purity - baseline


def main() -> None:
    args = parse_args()
    tensor_dir = Path(args.tensor_dir)
    if args.tensor == "/work/data/generated/image_tensors/image_tensors.bin":
        args.tensor = str(tensor_dir / "image_tensors.bin")
    if args.index == "/work/data/generated/image_tensors/image_tensors_index.csv":
        args.index = str(tensor_dir / "image_tensors_index.csv")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    starts, channels, height, width = read_index(Path(args.index))
    tensor = torch.from_file(str(Path(args.tensor)), dtype=torch.uint8, size=len(starts) * channels * height * width)
    x_cpu = tensor.reshape(len(starts), channels, height, width).float() / 255.0
    labels = read_families(Path(args.families), starts)
    index_by_n = {n: i for i, n in enumerate(starts)}
    pairs = read_pairs(Path(args.positive_pairs), index_by_n) or [(i, i) for i in range(len(starts))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x_cpu.to(device)
    model = ImageEncoder(args.embedding_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses: list[float] = []
    for _epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pairs)
        epoch_losses: list[float] = []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            if len(batch) < 2:
                continue
            a = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            b = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            view_a = x[a]
            view_b = x[b]
            if all(left == right for left, right in batch):
                view_b = (view_b + torch.randn_like(view_b) * 0.025).clamp(0.0, 1.0)
            loss = info_nce(model(view_a), model(view_b))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
    if not losses:
        losses.append(0.0)

    model.eval()
    with torch.no_grad():
        z = model(x).detach().cpu()
    purity, baseline, lift = evaluate(z, labels)
    with (output_dir / "embeddings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "label", *[f"e{i}" for i in range(z.shape[1])]])
        for n, label, vector in zip(starts, labels, z.tolist()):
            writer.writerow([n, label, *[f"{value:.9g}" for value in vector]])
    pair_metrics = {}
    if Path(args.pair_metrics).exists():
        with Path(args.pair_metrics).open() as handle:
            pair_metrics = json.load(handle)
    metrics = {
        "dataset_type": "collatz_image_contrastive_embeddings",
        "tool": "research/image_contrastive_train.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "feature_set": "image",
        "embedding_count": len(starts),
        "embedding_dims": args.embedding_dims,
        "epochs": args.epochs,
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "neighbor_purity": purity,
        "random_baseline_purity": baseline,
        "purity_lift": lift,
        "matched_controls": pair_metrics.get("matched_controls", {}),
        "outputs": {"embeddings": "embeddings.csv"},
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"image_contrastive rows={len(starts)} lift={lift:.4f} output_dir={output_dir}")


if __name__ == "__main__":
    main()
