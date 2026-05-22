#!/usr/bin/env python3
"""Generate trajectory-family positive pairs and matched hard negatives."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CONTROL_FIELDS = [
    "bit_length",
    "range_band",
    "residue_class",
    "total_steps_bucket",
    "peak_ratio_bucket",
    "first_drop_bucket",
]

DIFFER_FIELDS = [
    "tail_hash",
    "coalescence_family_id",
    "parity_motif_hash",
    "residue_motif_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Collatz family positives and matched hard negatives.")
    parser.add_argument("--labels", default="/work/data/generated/ml_labels/families.csv")
    parser.add_argument("--metrics", default="/work/data/generated/ml_stratified/metrics_safe.csv")
    parser.add_argument("--sample", default="/work/data/generated/stratified/samples.csv")
    parser.add_argument("--output-dir", default="/work/data/generated/ml_pairs")
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--max-pairs-per-type", type=int, default=200000)
    parser.add_argument("--min-match-rate", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()
    if not 0.0 <= args.min_match_rate <= 1.0:
        raise SystemExit("--min-match-rate must be between 0 and 1")
    return args


def read_csv_by_n(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("n"):
                continue
            rows[int(row["n"])] = row
    return rows


def read_metric_starts(path: Path) -> set[int]:
    starts: set[int] = set()
    if not path.exists():
        return starts
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            starts.add(int(row["n"]))
    return starts


def control_match_rates(rows: dict[int, dict[str, str]]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {field: 0.0 for field in CONTROL_FIELDS}
    totals: dict[str, int] = {field: 0 for field in CONTROL_FIELDS}
    for row in rows.values():
        for field in CONTROL_FIELDS:
            if row.get(field, ""):
                totals[field] += 1
    return {field: totals[field] / total for field in CONTROL_FIELDS}


def attach_derived_controls(rows: dict[int, dict[str, str]]) -> None:
    for n, row in rows.items():
        row.setdefault("residue_class", str(n % 32))


def pair_key(a: int, b: int, pair_type: str) -> tuple[int, int, str]:
    lo, hi = sorted((a, b))
    return lo, hi, pair_type


def add_group_pairs(
    pairs: set[tuple[int, int, str]],
    distribution: Counter[str],
    rows: dict[int, dict[str, str]],
    field: str,
    pair_type: str,
    max_pairs: int,
) -> None:
    groups: dict[str, list[int]] = defaultdict(list)
    for n, row in rows.items():
        value = row.get(field, "")
        if value and value != "none":
            groups[value].append(n)
    added = 0
    for values in groups.values():
        if len(values) < 2:
            continue
        values = sorted(values)
        for i in range(len(values) - 1):
            key = pair_key(values[i], values[i + 1], pair_type)
            if key not in pairs:
                pairs.add(key)
                distribution[pair_type] += 1
                added += 1
                if added >= max_pairs:
                    return


def control_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in CONTROL_FIELDS)


def differs_by_family(anchor: dict[str, str], candidate: dict[str, str]) -> bool:
    return all(anchor.get(field, "") != candidate.get(field, "") for field in DIFFER_FIELDS)


def write_pairs(path: Path, pairs: set[tuple[int, int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_a", "n_b", "pair_type"])
        for n_a, n_b, pair_type in sorted(pairs, key=lambda item: (item[2], item[0], item[1])):
            writer.writerow([n_a, n_b, pair_type])


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_csv_by_n(Path(args.labels))
    metric_starts = read_metric_starts(Path(args.metrics))
    if metric_starts:
        labels = {n: row for n, row in labels.items() if n in metric_starts}
    if not labels:
        raise RuntimeError("no usable family labels found")
    attach_derived_controls(labels)

    pairs: set[tuple[int, int, str]] = set()
    distribution: Counter[str] = Counter()
    add_group_pairs(pairs, distribution, labels, "tail_hash", "same_tail_hash", args.max_pairs_per_type)
    add_group_pairs(pairs, distribution, labels, "coalescence_family_id", "same_coalescence_family", args.max_pairs_per_type)
    add_group_pairs(pairs, distribution, labels, "first_drop_bucket", "same_first_drop_family", args.max_pairs_per_type)
    add_group_pairs(pairs, distribution, labels, "parity_motif_hash", "same_parity_motif", args.max_pairs_per_type)
    add_group_pairs(pairs, distribution, labels, "residue_motif_hash", "same_residue_motif", args.max_pairs_per_type)
    add_group_pairs(pairs, distribution, labels, "source_family", "same_source_neighborhood", args.max_pairs_per_type)
    for n in labels:
        pairs.add((n, n, "same_n_different_view"))
        distribution["same_n_different_view"] += 1

    by_control: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for n, row in labels.items():
        by_control[control_key(row)].append(n)

    hard_negatives: list[tuple[int, int, str]] = []
    requested = 0
    matched = 0
    per_anchor_candidates: list[int] = []
    for n, row in sorted(labels.items()):
        candidates = [candidate for candidate in by_control[control_key(row)] if candidate != n and differs_by_family(row, labels[candidate])]
        random.shuffle(candidates)
        requested += args.hard_negatives
        per_anchor_candidates.append(len(candidates))
        for candidate in candidates[: args.hard_negatives]:
            hard_negatives.append((n, candidate, "|".join(control_key(row))))
            matched += 1

    with (output_dir / "hard_negatives.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "negative_n", "matched_control_key"])
        writer.writerows(hard_negatives)
    write_pairs(output_dir / "positive_pairs.csv", pairs)

    match_rate = matched / requested if requested else 0.0
    controls_pass = requested > 0 and match_rate >= args.min_match_rate and any(count > 0 for count in per_anchor_candidates)
    per_field_match_rate = control_match_rates(labels)
    controls_summary = {
        "bit_length": controls_pass,
        "range_band": controls_pass,
        "residue_class": controls_pass,
        "stopping_time_bucket": controls_pass,
        "peak_ratio_bucket": controls_pass,
        "first_drop_bucket": controls_pass,
    }
    controls_rate_summary = {field: per_field_match_rate[field] for field in CONTROL_FIELDS}
    metrics = {
        "dataset_type": "collatz_pair_sampler",
        "tool": "research/pair_sampler.py",
        "status": "complete",
        "schema_version": "pairs_v1",
        "label_count": len(labels),
        "positive_pair_count": len(pairs),
        "hard_negative_count": len(hard_negatives),
        "hard_negatives_per_anchor": args.hard_negatives,
        "hard_negative_match_rate": match_rate,
        "matched_control_min_match_rate": args.min_match_rate,
        "matched_control_pass": controls_pass,
        "matched_controls": controls_summary,
        "matched_control_rates": controls_rate_summary,
        "positive_pair_distribution": dict(sorted(distribution.items())),
        "control_fields": CONTROL_FIELDS,
        "different_family_fields": DIFFER_FIELDS,
        "outputs": {
            "positive_pairs": "positive_pairs.csv",
            "hard_negatives": "hard_negatives.csv",
        },
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        f"pairs positives={len(pairs)} hard_negatives={len(hard_negatives)} "
        f"match_rate={match_rate:.4f} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
