#!/usr/bin/env python3
"""Fixture-level regression tests for the public evidence contract."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import validate_evidence_summary


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/evidence_small/expected_latest_public_summary.json"


def expect_failure(data: dict, label: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(data, handle)
        temp_path = Path(handle.name)
    try:
        try:
            validate_evidence_summary.validate(temp_path)
        except SystemExit:
            return
        raise SystemExit(f"expected validation failure did not occur: {label}")
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    base = json.loads(FIXTURE.read_text())
    validate_evidence_summary.validate(FIXTURE)

    invalid = copy.deepcopy(base)
    invalid["confidence"]["label"] = "source-aligned candidate"
    expect_failure(invalid, "invalid confidence label")

    invalid = copy.deepcopy(base)
    invalid["next_experiment"]["summary"] = "review /Users/example/private/path"
    expect_failure(invalid, "unsafe public string")

    invalid = copy.deepcopy(base)
    del invalid["audit"]["rows"]
    expect_failure(invalid, "missing audit rows")

    invalid = copy.deepcopy(base)
    del invalid["coverage"]["topology"]["rows"]
    expect_failure(invalid, "coverage percent without row denominator")

    invalid = copy.deepcopy(base)
    invalid["confidence"]["promotion_blockers"].remove("source_alignment_unknown_rows_present")
    expect_failure(invalid, "unknown unmatched rows without blocker")

    invalid = copy.deepcopy(base)
    invalid["source_alignment"]["unmatched_breakdown"]["unknown"] = 0
    invalid["source_alignment"]["unmatched_breakdown"]["true_mismatch"] = 1
    invalid["confidence"]["promotion_blockers"].remove("source_alignment_unknown_rows_present")
    expect_failure(invalid, "true mismatch without blocker")

    invalid = copy.deepcopy(base)
    invalid["confidence"]["promotion_blockers"].remove("metrics_only_exceeds_hybrid")
    expect_failure(invalid, "metrics-only dominance without blocker")

    invalid = copy.deepcopy(base)
    invalid["confidence"]["promotion_blockers"].remove("matched_controls_incomplete")
    expect_failure(invalid, "incomplete matched controls without blocker")

    invalid = copy.deepcopy(base)
    invalid["confidence"]["promotion_blockers"].remove("missing_lift_statistics")
    expect_failure(invalid, "missing lift statistics without blocker")

    invalid = copy.deepcopy(base)
    invalid["confidence"]["label"] = "candidate pattern"
    invalid["confidence"]["rank"] = 3
    invalid["confidence"]["claim_is_candidate_pattern"] = True
    expect_failure(invalid, "candidate pattern with blockers")

    print("evidence contract fixture tests passed")


if __name__ == "__main__":
    main()
