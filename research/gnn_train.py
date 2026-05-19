#!/usr/bin/env python3
"""Train a small unsupervised GNN over exported Collatz trajectory graphs."""

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


NODE_FEATURES = [
    "log2_value",
    "parity",
    "residue_mod3",
    "residue_mod4",
    "residue_mod8",
    "residue_mod16",
    "residue_mod32",
    "is_start",
    "is_terminal",
    "in_degree",
    "out_degree",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Collatz graph embeddings with a lightweight GraphSAGE encoder.")
    parser.add_argument("--nodes", default="/work/data/generated/graphs/nodes.csv")
    parser.add_argument("--edges", default="/work/data/generated/graphs/edges.csv")
    parser.add_argument("--output-dir", default="/work/data/generated/gnn")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("GNN_EPOCHS", "50")))
    parser.add_argument("--hidden-dims", type=int, default=int(os.getenv("GNN_HIDDEN_DIMS", "96")))
    parser.add_argument("--embedding-dims", type=int, default=int(os.getenv("GNN_EMBEDDING_DIMS", "32")))
    parser.add_argument("--negative-ratio", type=int, default=int(os.getenv("GNN_NEGATIVE_RATIO", "1")))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def read_nodes(path: Path) -> tuple[list[str], torch.Tensor]:
    values: list[str] = []
    features: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(row["value"])
            feature = [
                math.log1p(float(row["log2_value"])),
                float(row["parity"]),
                float(row["residue_mod3"]) / 2.0,
                float(row["residue_mod4"]) / 3.0,
                float(row["residue_mod8"]) / 7.0,
                float(row["residue_mod16"]) / 15.0,
                float(row["residue_mod32"]) / 31.0,
                float(row["is_start"]),
                float(row["is_terminal"]),
                math.log1p(float(row["in_degree"])),
                math.log1p(float(row["out_degree"])),
            ]
            features.append(feature)
    if not features:
        raise RuntimeError("nodes.csv contained no rows")
    x = torch.tensor(features, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    return values, x


def read_edges(path: Path) -> torch.Tensor:
    edges: list[tuple[int, int]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = int(row["source"])
            target = int(row["target"])
            edges.append((source, target))
            edges.append((target, source))
    if not edges:
        raise RuntimeError("edges.csv contained no rows")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class GraphSageLayer(nn.Module):
    def __init__(self, in_dims: int, out_dims: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(in_dims, out_dims)
        self.neigh_proj = nn.Linear(in_dims, out_dims)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=x.device, dtype=x.dtype))
        agg = agg / deg.clamp_min(1.0)
        return F.relu(self.self_proj(x) + self.neigh_proj(agg))


class GraphEncoder(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int, embedding_dims: int) -> None:
        super().__init__()
        self.layer1 = GraphSageLayer(in_dims, hidden_dims)
        self.layer2 = GraphSageLayer(hidden_dims, embedding_dims)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, edge_index)
        z = self.layer2(h, edge_index)
        return F.normalize(z, dim=1)


def sample_negative_edges(node_count: int, count: int, device: torch.device) -> torch.Tensor:
    src = torch.randint(0, node_count, (count,), device=device)
    dst = torch.randint(0, node_count, (count,), device=device)
    same = src == dst
    if same.any():
        dst[same] = (dst[same] + 1) % node_count
    return torch.stack([src, dst], dim=0)


def write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    tmp.replace(path)


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    values, x_cpu = read_nodes(Path(args.nodes))
    edge_index_cpu = read_edges(Path(args.edges))
    x = x_cpu.to(device)
    edge_index = edge_index_cpu.to(device)
    pos_edges = edge_index[:, ::2]

    model = GraphEncoder(x.shape[1], args.hidden_dims, args.embedding_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(x, edge_index)
        neg_edges = sample_negative_edges(x.shape[0], pos_edges.shape[1] * args.negative_ratio, device)
        pos_logits = (z[pos_edges[0]] * z[pos_edges[1]]).sum(dim=1)
        neg_logits = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        write_metrics(
            metrics_path,
            {
                "dataset_type": "collatz_gnn_embeddings",
                "tool": "research/gnn_train.py",
                "status": "running",
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "node_count": len(values),
                "directed_edge_count": int(pos_edges.shape[1]),
                "feature_schema": NODE_FEATURES,
                "epochs": args.epochs,
                "epoch": epoch,
                "hidden_dims": args.hidden_dims,
                "embedding_dims": args.embedding_dims,
                "loss_start": losses[0],
                "loss_final": losses[-1],
                "loss_history": losses,
            },
        )

    model.eval()
    with torch.no_grad():
        z = model(x, edge_index).detach().cpu()

    embeddings_path = output_dir / "embeddings.csv"
    with embeddings_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "value", *[f"e{i}" for i in range(z.shape[1])]])
        for node_id, value in enumerate(values):
            writer.writerow([node_id, value, *[f"{v:.9g}" for v in z[node_id].tolist()]])

    metrics = {
        "dataset_type": "collatz_gnn_embeddings",
        "tool": "research/gnn_train.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "node_count": len(values),
        "directed_edge_count": int(pos_edges.shape[1]),
        "feature_schema": NODE_FEATURES,
        "epochs": args.epochs,
        "hidden_dims": args.hidden_dims,
        "embedding_dims": args.embedding_dims,
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "loss_history": losses,
        "outputs": {"embeddings": "embeddings.csv"},
    }
    write_metrics(metrics_path, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print(
        f"gnn device={metrics['device']} nodes={metrics['node_count']} "
        f"edges={metrics['directed_edge_count']} loss_final={metrics['loss_final']:.6f}"
    )


if __name__ == "__main__":
    main()
