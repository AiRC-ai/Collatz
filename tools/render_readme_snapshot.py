#!/usr/bin/env python3
"""Render or update the README evidence snapshot from canonical JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BEGIN = "<!-- BEGIN GENERATED EVIDENCE SNAPSHOT -->"
END = "<!-- END GENERATED EVIDENCE SNAPSHOT -->"


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.{digits}f}"


def render(data: dict) -> str:
    confidence = data["confidence"]
    audit = data["audit"]
    coverage = data["coverage"]
    neural = data["neural"]
    holdouts = neural["holdouts"]
    source = data["source_alignment"]
    interpretation = neural["interpretation"]
    leaderboard = neural["leaderboard"]
    best = max(
        (entry for entry in leaderboard if entry["lift_percent"]["mean"] is not None),
        key=lambda entry: entry["lift_percent"]["mean"],
        default=None,
    )
    best_text = "pending"
    if best:
        best_text = f"{best['name']} at {fmt(best['lift_percent']['mean'])}%"
    return "\n".join(
        [
            BEGIN,
            f"- Confidence: `{confidence['label']}`",
            f"- Meaning: {confidence['interpretation']}",
            f"- Audit: `{audit['rows']:,}` rows over `{audit['range_start']:,}..{audit['range_end']:,}`; full audit completed: `{str(audit['full_audit_completed']).lower()}`.",
            f"- Coverage: topology `{coverage['topology']['rows']:,}` rows (`{fmt(coverage['topology']['percent_of_audit'])}%` of audit); stratified evidence sample `{coverage['stratified_evidence_sample']['rows']:,}` rows (`{fmt(coverage['stratified_evidence_sample']['percent_of_audit'])}%`).",
            f"- Neural result: `{neural['latest_run']['sample_rows']:,}` sample rows; GPU used: `{str(neural['latest_run']['gpu_used']).lower()}`; parallel jobs completed: `{neural['latest_run']['parallel_jobs_completed']}`.",
            f"- Learned lift: weakest range `{fmt(holdouts['weakest_range_lift_percent'])}%`, fold minimum `{fmt(holdouts['fold_min_lift_percent'])}%`, numeric-adjacency lift `{fmt(holdouts['numeric_adjacency_lift_percent'])}%`.",
            f"- Best current ablation: `{best_text}`.",
            f"- Interpretation: `{interpretation['signal_type']}`; {interpretation['reason']}.",
            f"- Source alignment: `{source['matched']:,} / {source['targets_total']:,}` matched; unknown unmatched rows `{source['unmatched_breakdown']['unknown']}`.",
            f"- Next experiment: {data['next_experiment']['summary']}",
            "- This is empirical evidence, not a Collatz proof.",
            END,
        ]
    )


def update_readme(readme: Path, snapshot: str) -> None:
    text = readme.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"README is missing generated snapshot markers: {readme}")
    before, rest = text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    readme.write_text(before + snapshot + after)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README evidence snapshot from canonical JSON.")
    parser.add_argument("--input", default="data/generated/evidence/latest_public_summary.json")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--expect")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    snapshot = render(data) + "\n"
    if args.expect:
        expected = Path(args.expect).read_text()
        if snapshot != expected:
            raise SystemExit("rendered README snapshot differs from expected fixture")
    elif args.update_readme:
        update_readme(Path(args.readme), snapshot.rstrip("\n"))
    else:
        print(snapshot, end="")


if __name__ == "__main__":
    main()
