#!/usr/bin/env python3
"""Train a compact autoencoder and emit Collatz anomaly candidates."""

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
    parser = argparse.ArgumentParser(description="Train an autoencoder over Collatz metric features and rank reconstruction anomalies.")
    parser.add_argument("--metrics", default="/work/data/generated/ml/metrics.csv")
    parser.add_argument("--sample", default="/work/data/generated/stratified/samples.csv")
    parser.add_argument("--output-dir", default="/work/data/generated/anomalies")
    parser.add_argument("--limit", type=int, default=int(os.getenv("AUTOENCODER_LIMIT", "25000")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("AUTOENCODER_EPOCHS", "50")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("AUTOENCODER_BATCH_SIZE", "512")))
    parser.add_argument("--hidden-dims", type=int, default=int(os.getenv("AUTOENCODER_HIDDEN_DIMS", "128")))
    parser.add_argument("--latent-dims", type=int, default=int(os.getenv("AUTOENCODER_LATENT_DIMS", "24")))
    parser.add_argument("--top-anomalies", type=int, default=int(os.getenv("AUTOENCODER_TOP_ANOMALIES", "128")))
    parser.add_argument("--seed", type=int, default=20260520)
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
            selected.add(n)
            labels[n] = row.get("reasons", "sample") or "sample"
    return selected, labels


def read_metrics(path: Path, selected: set[int] | None, labels_by_n: dict[int, str], limit: int) -> tuple[list[int], list[str], torch.Tensor]:
    starts: list[int] = []
    labels: list[str] = []
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in reader.fieldnames or [] if field.startswith("m")]
        for row in reader:
            n = int(row["n"])
            if selected is not None and n not in selected:
                continue
            starts.append(n)
            labels.append(labels_by_n.get(n, "metric_export"))
            rows.append([float(row[field]) for field in fields])
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("no rows available for autoencoder training")
    x = torch.tensor(rows, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = torch.nan_to_num((x - mean) / std)
    return starts, labels, x


class Autoencoder(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int, latent_dims: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, latent_dims),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, in_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


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
    starts, labels, x_cpu = read_metrics(Path(args.metrics), selected, labels_by_n, args.limit)
    x = x_cpu.to(device)
    model = Autoencoder(x.shape[1], args.hidden_dims, args.latent_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    indices = torch.arange(x.shape[0], device=device)

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = indices[torch.randperm(x.shape[0], device=device)]
        epoch_losses: list[float] = []
        for start in range(0, x.shape[0], args.batch_size):
            batch = x[perm[start : start + args.batch_size]]
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / max(1, len(epoch_losses)))
        write_metrics(
            metrics_path,
            {
                "dataset_type": "collatz_autoencoder_anomalies",
                "tool": "research/autoencoder_anomaly.py",
                "status": "running",
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "row_count": len(starts),
                "epoch": epoch,
                "epochs": args.epochs,
                "loss_start": losses[0],
                "loss_final": losses[-1],
                "loss_history": losses,
            },
        )

    model.eval()
    with torch.no_grad():
        recon = model(x)
        errors = ((recon - x) ** 2).mean(dim=1).detach().cpu()

    ranked = sorted(
        ((float(error), n, label) for error, n, label in zip(errors.tolist(), starts, labels)),
        reverse=True,
    )
    anomalies = ranked[: args.top_anomalies]

    anomalies_path = output_dir / "anomalies.csv"
    with anomalies_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "n", "reconstruction_error", "reasons"])
        for rank, (error, n, label) in enumerate(anomalies, start=1):
            writer.writerow([rank, n, f"{error:.9g}", label])

    checkpoint_path = output_dir / "autoencoder.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, checkpoint_path)

    metrics = {
        "dataset_type": "collatz_autoencoder_anomalies",
        "tool": "research/autoencoder_anomaly.py",
        "status": "complete",
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "row_count": len(starts),
        "anomaly_count": len(anomalies),
        "input_dims": x.shape[1],
        "latent_dims": args.latent_dims,
        "epochs": args.epochs,
        "loss_start": losses[0],
        "loss_final": losses[-1],
        "loss_history": losses,
        "mean_reconstruction_error": float(errors.mean()),
        "max_reconstruction_error": float(errors.max()),
        "outputs": {"anomalies": "anomalies.csv", "checkpoint": "autoencoder.pt"},
    }
    write_metrics(metrics_path, metrics)
    print(
        f"autoencoder device={metrics['device']} rows={metrics['row_count']} "
        f"anomalies={metrics['anomaly_count']} max_error={metrics['max_reconstruction_error']:.6f}"
    )


if __name__ == "__main__":
    main()
