#!/usr/bin/env python3
"""Validate the public-safe canonical Collatz evidence summary."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


CONFIDENCE_LABELS = {
    "sample-local signal": 0,
    "range-stable signal": 1,
    "source-neighborhood-supported": 2,
    "candidate pattern": 3,
    "proof": 4,
}

PROMOTION_BLOCKERS = {
    "range_stable_gate_incomplete",
    "oeis_source_family_incomplete",
    "non_oeis_source_families_incomplete",
    "source_alignment_unknown_rows_present",
    "source_alignment_true_mismatch_present",
    "source_alignment_unmatched_rows_present",
    "metrics_only_exceeds_hybrid",
    "richer_representation_not_stronger",
    "matched_controls_incomplete",
    "missing_lift_statistics",
}

UNMATCHED_BUCKETS = {
    "above_active_scan_range",
    "missing_from_topology_sample",
    "missing_feature_row",
    "parser_error",
    "step_convention_mismatch",
    "peak_convention_mismatch",
    "true_mismatch",
    "missing_topology_projection_node",
    "duplicated_source_row",
    "future_source_target",
    "unknown",
}

UNSAFE_PATTERNS = [
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\b172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}\b"),
    re.compile(r"\bssh\s+"),
    re.compile(r"\bbash\s+-lc\b"),
    re.compile(r"\bnvidia-smi\b"),
]


def die(message: str) -> None:
    raise SystemExit(f"evidence summary validation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        die(message)


def walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(walk_strings(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(walk_strings(item))
        return out
    return []


def require_keys(obj: dict[str, Any], keys: list[str], prefix: str) -> None:
    for key in keys:
        require(key in obj, f"{prefix}.{key} is required")


def validate(path: Path) -> None:
    data = json.loads(path.read_text())
    require(data.get("schema_version") == "evidence_public_summary_v1", "schema_version must be evidence_public_summary_v1")
    require_keys(
        data,
        [
            "run_id",
            "git_commit",
            "generated_at_utc",
            "confidence",
            "audit",
            "coverage",
            "neural",
            "source_alignment",
            "next_experiment",
            "public_safety",
        ],
        "root",
    )

    confidence = data["confidence"]
    require(confidence.get("label") in CONFIDENCE_LABELS, "unknown confidence label")
    require(confidence.get("rank") == CONFIDENCE_LABELS[confidence["label"]], "confidence rank does not match label")
    require(confidence.get("claim_is_proof") is False, "claim_is_proof must be false")
    require(isinstance(confidence.get("interpretation"), str) and confidence["interpretation"], "confidence interpretation required")
    blockers = confidence.get("promotion_blockers")
    require(isinstance(blockers, list), "confidence.promotion_blockers must be a list")
    require(len(blockers) == len(set(blockers)), "confidence.promotion_blockers must be unique")
    require(set(blockers).issubset(PROMOTION_BLOCKERS), "confidence.promotion_blockers contains unknown blocker")

    audit = data["audit"]
    require(isinstance(audit.get("rows"), int), "audit.rows must be integer")
    require(isinstance(audit.get("range_start"), int), "audit.range_start must be integer")
    require(isinstance(audit.get("range_end"), int), "audit.range_end must be integer")
    require(audit["rows"] >= 0, "audit.rows must be non-negative")
    require(isinstance(audit.get("full_audit_completed"), bool), "audit.full_audit_completed must be boolean")

    coverage = data["coverage"]
    for name in ("topology", "stratified_evidence_sample"):
        item = coverage.get(name, {})
        require(isinstance(item.get("rows"), int), f"coverage.{name}.rows must be integer")
        require(isinstance(item.get("percent_of_audit"), (int, float)), f"coverage.{name}.percent_of_audit must be numeric")

    neural = data["neural"]
    require(isinstance(neural.get("leaderboard"), list) and neural["leaderboard"], "neural.leaderboard required")
    lifts: dict[str, Any] = {}
    all_controls_complete = True
    all_stats_complete = True
    for item in neural["leaderboard"]:
        require_keys(item, ["name", "lift_percent", "n_folds", "n_seeds", "n_samples", "matched_controls"], "leaderboard")
        lift = item["lift_percent"]
        require_keys(lift, ["mean", "std", "ci_95"], "leaderboard.lift_percent")
        lifts[item["name"]] = lift.get("mean")
        if lift.get("std") is None or lift.get("ci_95") is None or item.get("n_folds") is None or item.get("n_seeds") is None:
            all_stats_complete = False
        controls = item["matched_controls"]
        require(
            all(key in controls for key in (
                "bit_length",
                "range_band",
                "residue_class",
                "stopping_time_bucket",
                "peak_ratio_bucket",
                "first_drop_bucket",
            )),
            "matched_controls incomplete",
        )
        if not all(controls.values()):
            all_controls_complete = False

    source = data["source_alignment"]
    require(source.get("matched", 0) + source.get("unmatched", 0) == source.get("targets_total", -1), "source counts do not add up")
    breakdown = source.get("unmatched_breakdown", {})
    require(set(breakdown) == UNMATCHED_BUCKETS, "unmatched_breakdown buckets do not match contract")
    require(sum(int(value) for value in breakdown.values()) == source.get("unmatched"), "unmatched_breakdown total mismatch")
    require(
        ("source_alignment_unknown_rows_present" in blockers) == (int(breakdown["unknown"]) > 0),
        "unknown source blocker does not match unmatched_breakdown.unknown",
    )
    require(
        ("source_alignment_true_mismatch_present" in blockers) == (int(breakdown["true_mismatch"]) > 0),
        "true mismatch blocker does not match unmatched_breakdown.true_mismatch",
    )
    require(
        ("source_alignment_unmatched_rows_present" in blockers) == (source.get("unmatched", 0) > 0),
        "unmatched source blocker does not match source unmatched count",
    )
    metrics_lift = lifts.get("metrics-only")
    hybrid_lift = lifts.get("hybrid")
    if metrics_lift is not None and hybrid_lift is not None:
        require(
            ("metrics_only_exceeds_hybrid" in blockers) == (metrics_lift > hybrid_lift),
            "metrics-only dominance blocker does not match leaderboard",
        )
    require(
        ("matched_controls_incomplete" in blockers) == (not all_controls_complete),
        "matched-controls blocker does not match leaderboard controls",
    )
    require(
        ("missing_lift_statistics" in blockers) == (not all_stats_complete),
        "missing lift statistics blocker does not match leaderboard stats",
    )
    if confidence["label"] == "candidate pattern":
        require("metrics_only_exceeds_hybrid" not in blockers, "candidate pattern cannot be metric dominant")
        require("matched_controls_incomplete" not in blockers, "candidate pattern requires matched controls")
        require("missing_lift_statistics" not in blockers, "candidate pattern requires lift statistics")

    safety = data["public_safety"]
    require(safety.get("sanitized") is True, "public_safety.sanitized must be true")
    for key in (
        "contains_hostnames",
        "contains_internal_ips",
        "contains_usernames",
        "contains_absolute_local_paths",
        "contains_raw_internal_command_traces",
    ):
        require(safety.get(key) is False, f"public_safety.{key} must be false")

    for text in walk_strings(data):
        for pattern in UNSAFE_PATTERNS:
            require(pattern.search(text) is None, f"unsafe public value matched {pattern.pattern!r}: {text[:80]}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_evidence_summary.py SUMMARY.json")
    validate(Path(sys.argv[1]))
    print(f"validated {sys.argv[1]}")


if __name__ == "__main__":
    main()
