#!/usr/bin/env python3
"""Fixture checks for path-family export, labels, pairs, image tensors, and graph membership."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def die(message: str) -> None:
    raise SystemExit(f"path-family fixture check failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        die(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def control_key(row: dict[str, str]) -> tuple[str, ...]:
    n = int(row["n"])
    return (
        row["bit_length"],
        row["range_band"],
        str(n % 32),
        row["total_steps_bucket"],
        row["peak_ratio_bucket"],
        row["first_drop_bucket"],
    )


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: test_path_family_tools.py ML_DIR LABEL_DIR PAIR_DIR IMAGE_DIR GRAPH_DIR"
        )
    ml_dir = Path(sys.argv[1])
    label_dir = Path(sys.argv[2])
    pair_dir = Path(sys.argv[3])
    image_dir = Path(sys.argv[4])
    graph_dir = Path(sys.argv[5])

    metadata = json.loads((ml_dir / "metadata.json").read_text())
    require(metadata.get("metric_schema") == "safe_v1", "safe metric metadata schema missing")
    require(metadata.get("safe_metric_vector_dims") == 32, "safe metric dimension mismatch")
    require(metadata.get("outputs", {}).get("metrics_safe") == "metrics_safe.csv", "metrics_safe output not recorded")
    metric_rows = read_csv(ml_dir / "metrics_safe.csv")
    require(metric_rows, "metrics_safe.csv is empty")
    metric_fields = metric_rows[0].keys()
    require("n" in metric_fields, "metrics_safe.csv missing n")
    require(sum(1 for field in metric_fields if field.startswith("m")) == 32, "metrics_safe.csv must have 32 metric fields")

    family_rows = read_csv(label_dir / "families.csv")
    require(len(family_rows) == len(metric_rows), "family rows do not match safe metric rows")
    required_family_fields = {
        "n",
        "tail_entry_value",
        "tail_hash",
        "coalescence_family_id",
        "first_drop_bucket",
        "total_steps_bucket",
        "peak_ratio_bucket",
        "parity_motif_hash",
        "residue_motif_hash",
        "range_band",
        "bit_length",
        "source_family",
    }
    require(required_family_fields.issubset(family_rows[0].keys()), "families.csv missing required fields")
    require(all(row["tail_hash"] for row in family_rows), "family tail hashes must be populated")

    pair_metrics = json.loads((pair_dir / "metrics.json").read_text())
    require(pair_metrics.get("schema_version") == "pairs_v1", "pair metrics schema mismatch")
    require(pair_metrics.get("positive_pair_count", 0) > 0, "positive pairs not generated")
    require((pair_dir / "positive_pairs.csv").exists(), "positive_pairs.csv missing")
    require((pair_dir / "hard_negatives.csv").exists(), "hard_negatives.csv missing")
    controls = pair_metrics.get("matched_controls", {})
    require(set(controls) == {
        "bit_length",
        "range_band",
        "residue_class",
        "stopping_time_bucket",
        "peak_ratio_bucket",
        "first_drop_bucket",
    }, "matched control flags missing")
    require(pair_metrics.get("control_fields") == [
        "bit_length",
        "range_band",
        "residue_class",
        "total_steps_bucket",
        "peak_ratio_bucket",
        "first_drop_bucket",
    ], "hard-negative controls must include residue_class, not source_family")
    require(all(value == bool(pair_metrics.get("matched_control_pass")) for value in controls.values()),
            "matched-control flags must follow the sampler threshold result")
    family_by_n = {int(row["n"]): row for row in family_rows}
    for row in read_csv(pair_dir / "hard_negatives.csv"):
        anchor = family_by_n[int(row["n"])]
        negative = family_by_n[int(row["negative_n"])]
        require(control_key(anchor) == control_key(negative), "hard negative does not match every control field")
        require(anchor["tail_hash"] != negative["tail_hash"], "hard negative shares tail_hash")
        require(anchor["coalescence_family_id"] != negative["coalescence_family_id"], "hard negative shares coalescence family")
        require(anchor["parity_motif_hash"] != negative["parity_motif_hash"], "hard negative shares parity motif")
        require(anchor["residue_motif_hash"] != negative["residue_motif_hash"], "hard negative shares residue motif")

    image_meta = json.loads((image_dir / "metadata.json").read_text())
    require(image_meta.get("schema_version") == "image_tensor_v1", "image tensor schema mismatch")
    rows = int(image_meta.get("rows", 0))
    row_bytes = int(image_meta.get("row_bytes", 0))
    require(rows == len(metric_rows), "image tensor row count mismatch")
    require((image_dir / "image_tensors.bin").stat().st_size == rows * row_bytes, "image tensor byte size mismatch")
    require(len(read_csv(image_dir / "image_tensors_index.csv")) == rows, "image tensor index row count mismatch")

    manifest = json.loads((graph_dir / "trajectory_graph.json").read_text())
    require(manifest.get("files", {}).get("membership") == "start_membership.csv", "graph manifest missing membership file")
    membership_rows = read_csv(graph_dir / "start_membership.csv")
    require(membership_rows, "graph membership export empty")
    require({"n", "position", "node_id", "final_tail"}.issubset(membership_rows[0].keys()), "graph membership columns missing")

    print("path-family fixture checks passed")


if __name__ == "__main__":
    main()
