#!/usr/bin/env python3
"""Validate Collatz representation evidence with holdouts and feature ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate learned Collatz embedding evidence.")
    parser.add_argument("--sample", default="/work/data/generated/stratified/samples.csv")
    parser.add_argument("--metrics", default="/work/data/generated/ml_stratified/metrics.csv")
    parser.add_argument("--parity-runs", default="/work/data/generated/ml_stratified/parity_runs.csv")
    parser.add_argument("--transitions", default="/work/data/generated/ml_stratified/residue_transitions_mod32.csv")
    parser.add_argument("--contrastive-embeddings", default="/work/data/generated/contrastive/embeddings.csv")
    parser.add_argument("--contrastive-metrics", default="/work/data/generated/contrastive/metrics.json")
    parser.add_argument("--gnn-metrics", default="/work/data/generated/gnn/metrics.json")
    parser.add_argument("--full-audit", default="/work/data/generated/full_audit/summary.json")
    parser.add_argument("--output-dir", default="/work/data/generated/evidence_validation")
    parser.add_argument("--token-bins", type=int, default=int(os.getenv("EVIDENCE_TOKEN_BINS", "64")))
    parser.add_argument("--range-bands", type=int, default=int(os.getenv("EVIDENCE_RANGE_BANDS", "4")))
    parser.add_argument("--folds", type=int, default=int(os.getenv("EVIDENCE_FOLDS", "3")))
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        help="Ablation metrics in name=/path/to/metrics.json form. Can be repeated.",
    )
    return parser.parse_args()


def finite(value: float | None) -> float | None:
    if value is None:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def read_sample(path: Path) -> tuple[set[int], dict[int, str], dict[int, str]]:
    selected: set[int] = set()
    labels: dict[int, str] = {}
    reasons: dict[int, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            reason = row.get("reasons", "sample") or "sample"
            selected.add(n)
            labels[n] = reason.split("|", 1)[0]
            reasons[n] = reason
    if not selected:
        raise RuntimeError(f"sample file had no rows: {path}")
    return selected, labels, reasons


def read_metrics(path: Path, selected: set[int]) -> dict[int, list[float]]:
    rows: dict[int, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in reader.fieldnames or [] if field.startswith("m")]
        for row in reader:
            n = int(row["n"])
            if n in selected:
                rows[n] = [float(row[field]) for field in fields]
    return rows


def read_token_hist(path: Path, selected: set[int], bins: int) -> dict[int, list[float]]:
    histograms: dict[int, list[float]] = {}
    if not path.exists():
        return histograms
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["n"])
            if n not in selected:
                continue
            hist = [0.0] * bins
            count = 0
            for item in (row.get("tokens", "") or "").split(";"):
                if not item:
                    continue
                hist[int(item) % bins] += 1.0
                count += 1
            if count:
                hist = [value / count for value in hist]
            histograms[n] = hist
    return histograms


def read_embeddings(path: Path, labels_by_n: dict[int, str]) -> tuple[list[int], list[str], list[list[float]]]:
    starts: list[int] = []
    labels: list[str] = []
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [field for field in reader.fieldnames or [] if field.startswith("e")]
        for row in reader:
            n = int(row["n"])
            starts.append(n)
            labels.append(labels_by_n.get(n, row.get("label", "embedding")))
            rows.append([float(row[field]) for field in fields])
    if not rows:
        raise RuntimeError(f"embedding file had no rows: {path}")
    return starts, labels, rows


def build_feature_rows(
    starts: list[int],
    metrics: dict[int, list[float]],
    parity: dict[int, list[float]],
    residue: dict[int, list[float]],
    feature_set: str,
    token_bins: int,
) -> list[list[float]]:
    rows: list[list[float]] = []
    metric_dims = len(next(iter(metrics.values()))) if metrics else 0
    for n in starts:
        row: list[float] = []
        if feature_set in ("hybrid", "metrics"):
            row.extend(metrics.get(n, [0.0] * metric_dims))
        if feature_set in ("hybrid", "parity", "tokens"):
            row.extend(parity.get(n, [0.0] * token_bins))
        if feature_set in ("hybrid", "residue", "tokens"):
            row.extend(residue.get(n, [0.0] * token_bins))
        rows.append(row)
    return rows


def random_baseline(labels: list[str]) -> float:
    counts = Counter(labels)
    total = max(1, len(labels))
    return sum((count / total) ** 2 for count in counts.values())


def neighbor_stats(rows: list[list[float]], labels: list[str], standardize: bool = True) -> dict[str, float | int | None]:
    if len(rows) < 2:
        return {"row_count": len(rows), "purity": None, "random_baseline": None, "lift": None}
    x = torch.tensor(rows, dtype=torch.float32)
    if standardize:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        x = torch.nan_to_num((x - mean) / std)
    x = F.normalize(torch.nan_to_num(x), dim=1)
    chunk_size = max(1, int(os.getenv("EVIDENCE_EVAL_CHUNK", "2048")))
    x_t = x.T.contiguous()
    nearest: list[int] = []
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        sim = x[start:end] @ x_t
        rows_index = torch.arange(end - start)
        cols_index = torch.arange(start, end)
        sim[rows_index, cols_index] = -2.0
        nearest.extend(sim.argmax(dim=1).cpu().tolist())
    purity = sum(1 for i, j in enumerate(nearest) if labels[i] == labels[j]) / len(labels)
    baseline = random_baseline(labels)
    return {"row_count": len(rows), "purity": purity, "random_baseline": baseline, "lift": purity - baseline}


def subset_stats(
    name: str,
    starts: list[int],
    labels: list[str],
    rows: list[list[float]],
    indices: list[int],
    standardize: bool = False,
) -> dict[str, object] | None:
    if len(indices) < 2:
        return None
    sub_rows = [rows[i] for i in indices]
    sub_labels = [labels[i] for i in indices]
    stats = neighbor_stats(sub_rows, sub_labels, standardize=standardize)
    return {
        "name": name,
        "row_count": len(indices),
        "min_n": min(starts[i] for i in indices),
        "max_n": max(starts[i] for i in indices),
        **stats,
    }


def numeric_adjacency_purity(starts: list[int], labels: list[str]) -> float:
    if len(starts) < 2:
        return 0.0
    ordered = sorted((n, i) for i, n in enumerate(starts))
    same = 0
    for pos, (n, i) in enumerate(ordered):
        candidates: list[tuple[int, int]] = []
        if pos > 0:
            candidates.append((abs(n - ordered[pos - 1][0]), ordered[pos - 1][1]))
        if pos + 1 < len(ordered):
            candidates.append((abs(n - ordered[pos + 1][0]), ordered[pos + 1][1]))
        _, j = min(candidates)
        if labels[i] == labels[j]:
            same += 1
    return same / len(starts)


def summarize_group_lifts(groups: list[dict[str, object]]) -> tuple[int, float | None, float | None]:
    lifts = [float(group["lift"]) for group in groups if group.get("lift") is not None]
    if not lifts:
        return 0, None, None
    return len(lifts), min(lifts), sum(lifts) / len(lifts)


def parse_ablation(item: str) -> tuple[str, Path]:
    if "=" in item:
        name, path = item.split("=", 1)
    elif ":" in item:
        name, path = item.split(":", 1)
    else:
        raise RuntimeError(f"ablation must be name=path: {item}")
    return name.strip(), Path(path.strip())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, labels_by_n, reasons_by_n = read_sample(Path(args.sample))
    starts, labels, learned_rows = read_embeddings(Path(args.contrastive_embeddings), labels_by_n)
    metrics = read_metrics(Path(args.metrics), selected)
    parity = read_token_hist(Path(args.parity_runs), selected, args.token_bins)
    residue = read_token_hist(Path(args.transitions), selected, args.token_bins)

    learned = neighbor_stats(learned_rows, labels, standardize=False)
    numeric_purity = numeric_adjacency_purity(starts, labels)
    learned_purity = float(learned["purity"] or 0.0)
    learned_lift = float(learned["lift"] or 0.0)

    raw_feature_stats = []
    for feature_set in ("metrics", "parity", "residue", "tokens", "hybrid"):
        rows = build_feature_rows(starts, metrics, parity, residue, feature_set, args.token_bins)
        stats = neighbor_stats(rows, labels, standardize=True)
        raw_feature_stats.append({"feature_set": feature_set, **stats})

    range_groups: list[dict[str, object]] = []
    min_n = min(starts)
    max_n = max(starts)
    span = max(1, max_n - min_n + 1)
    by_range: dict[int, list[int]] = defaultdict(list)
    for index, n in enumerate(starts):
        band = min(args.range_bands - 1, ((n - min_n) * args.range_bands) // span)
        by_range[int(band)].append(index)
    for band, indices in sorted(by_range.items()):
        item = subset_stats(f"range_band_{band}", starts, labels, learned_rows, indices)
        if item is not None:
            range_groups.append(item)

    residue_groups: list[dict[str, object]] = []
    by_residue: dict[int, list[int]] = defaultdict(list)
    for index, n in enumerate(starts):
        by_residue[n % 32].append(index)
    for residue_value, indices in sorted(by_residue.items()):
        if len(indices) < 20:
            continue
        item = subset_stats(f"residue_mod32_{residue_value}", starts, labels, learned_rows, indices)
        if item is not None:
            residue_groups.append(item)

    fold_groups: list[dict[str, object]] = []
    shuffled = list(range(len(starts)))
    random.shuffle(shuffled)
    folds = max(2, args.folds)
    for fold in range(folds):
        indices = [index for pos, index in enumerate(shuffled) if pos % folds == fold]
        item = subset_stats(f"random_fold_{fold}", starts, labels, learned_rows, indices)
        if item is not None:
            fold_groups.append(item)

    learned_ablation_stats = []
    default_metrics = read_json(Path(args.contrastive_metrics))
    if default_metrics:
        learned_ablation_stats.append(
            {
                "name": str(default_metrics.get("feature_set", "hybrid")),
                "status": default_metrics.get("status", "unknown"),
                "neighbor_purity": default_metrics.get("neighbor_purity"),
                "random_baseline_purity": default_metrics.get("random_baseline_purity"),
                "purity_lift": default_metrics.get("purity_lift"),
                "embedding_count": default_metrics.get("embedding_count"),
            }
        )
    for item in args.ablation:
        name, path = parse_ablation(item)
        metrics_json = read_json(path)
        if not metrics_json:
            continue
        learned_ablation_stats.append(
            {
                "name": name,
                "status": metrics_json.get("status", "unknown"),
                "neighbor_purity": metrics_json.get("neighbor_purity"),
                "random_baseline_purity": metrics_json.get("random_baseline_purity"),
                "purity_lift": metrics_json.get("purity_lift"),
                "embedding_count": metrics_json.get("embedding_count"),
            }
        )

    best_ablation = None
    complete_ablations = [
        item for item in learned_ablation_stats if item.get("status") == "complete" and item.get("purity_lift") is not None
    ]
    if complete_ablations:
        best_ablation = max(complete_ablations, key=lambda item: float(item["purity_lift"]))

    range_count, range_min_lift, range_mean_lift = summarize_group_lifts(range_groups)
    residue_count, residue_min_lift, residue_mean_lift = summarize_group_lifts(residue_groups)
    fold_count, fold_min_lift, fold_mean_lift = summarize_group_lifts(fold_groups)
    contrastive_minus_numeric = learned_purity - numeric_purity
    passes_range = range_count >= max(2, args.range_bands // 2) and (range_min_lift or 0.0) > 0.0
    passes_folds = fold_count >= 2 and (fold_min_lift or 0.0) > 0.0
    passes_residue = residue_count >= 8 and (residue_mean_lift or 0.0) > 0.0
    passes_baseline = learned_lift > 0.05 and contrastive_minus_numeric > 0.0
    validation_confidence = "range-stable signal" if passes_range and passes_folds and passes_residue and passes_baseline else "sample-local signal"

    gnn_metrics = read_json(Path(args.gnn_metrics))
    full_audit = read_json(Path(args.full_audit))
    full_records = int(full_audit.get("records_read", 0) or 0)
    full_audit_coverage = float(full_audit.get("coverage_ratio", 0.0) or 0.0)
    neural_full_dataset_ratio = len(starts) / full_records if full_records else 0.0
    conclusion = (
        "Learned neighborhoods survive range, residue, and fold checks against a sample selected from the audited full dataset, so the current claim can be treated as range-stable empirical evidence."
        if validation_confidence == "range-stable signal"
        else "Learned neighborhoods remain sample-local until holdouts and ablations show stable lift."
    )

    metrics_out = {
        "dataset_type": "collatz_evidence_validation",
        "tool": "research/evidence_validate.py",
        "status": "complete",
        "confidence_level": validation_confidence,
        "conclusion": conclusion,
        "sample_rows": len(starts),
        "full_dataset_records": full_records,
        "full_dataset_audit_coverage": finite(full_audit_coverage),
        "neural_full_dataset_ratio": finite(neural_full_dataset_ratio),
        "full_dataset_max_total_steps": full_audit.get("max_total_steps"),
        "full_dataset_max_total_steps_n": full_audit.get("max_total_steps_n"),
        "full_dataset_total_steps_mean": full_audit.get("total_steps_mean"),
        "full_dataset_total_steps_quantiles": full_audit.get("total_steps_quantiles"),
        "sample_reason_count": len(set(reasons_by_n.values())),
        "label_count": len(set(labels)),
        "contrastive_neighbor_purity": finite(learned.get("purity")),
        "contrastive_random_baseline": finite(learned.get("random_baseline")),
        "contrastive_lift": finite(learned.get("lift")),
        "numeric_adjacency_purity": finite(numeric_purity),
        "contrastive_minus_numeric": finite(contrastive_minus_numeric),
        "range_holdout_count": range_count,
        "range_min_lift": finite(range_min_lift),
        "range_mean_lift": finite(range_mean_lift),
        "residue_holdout_count": residue_count,
        "residue_min_lift": finite(residue_min_lift),
        "residue_mean_lift": finite(residue_mean_lift),
        "fold_count": fold_count,
        "fold_min_lift": finite(fold_min_lift),
        "fold_mean_lift": finite(fold_mean_lift),
        "ablation_count": len(complete_ablations),
        "best_ablation": best_ablation.get("name") if best_ablation else None,
        "best_lift": finite(float(best_ablation["purity_lift"])) if best_ablation else None,
        "gnn_status": gnn_metrics.get("status", "missing"),
        "gnn_loss_final": gnn_metrics.get("loss_final"),
        "raw_feature_stats": raw_feature_stats,
        "learned_ablation_stats": learned_ablation_stats,
        "range_holdouts": range_groups,
        "residue_holdouts": residue_groups,
        "fold_holdouts": fold_groups,
    }

    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics_out, handle, indent=2)

    with (output_dir / "holdouts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "name", "row_count", "min_n", "max_n", "purity", "random_baseline", "lift"])
        for kind, groups in (("range", range_groups), ("residue", residue_groups), ("fold", fold_groups)):
            for group in groups:
                writer.writerow(
                    [
                        kind,
                        group["name"],
                        group["row_count"],
                        group["min_n"],
                        group["max_n"],
                        group["purity"],
                        group["random_baseline"],
                        group["lift"],
                    ]
                )

    with (output_dir / "ablation_report.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "name", "row_count", "purity", "random_baseline", "lift"])
        for item in raw_feature_stats:
            writer.writerow(["raw", item["feature_set"], item["row_count"], item["purity"], item["random_baseline"], item["lift"]])
        for item in learned_ablation_stats:
            writer.writerow(
                [
                    "learned",
                    item["name"],
                    item.get("embedding_count"),
                    item.get("neighbor_purity"),
                    item.get("random_baseline_purity"),
                    item.get("purity_lift"),
                ]
            )

    print(
        f"evidence confidence={validation_confidence} rows={len(starts)} "
        f"lift={learned_lift:.4f} range_min={range_min_lift or 0.0:.4f} "
        f"fold_min={fold_min_lift or 0.0:.4f}"
    )


if __name__ == "__main__":
    main()
