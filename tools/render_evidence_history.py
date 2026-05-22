#!/usr/bin/env python3
"""Render a public-safe historical evidence SVG from sanitized runner history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "cycle_count" in row and "evidence_score" in row:
            rows.append(row)
    return rows


def esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_millions(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "pending"
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.0f}M"
    return f"{num:,.0f}"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_canonical_row(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    audit = evidence.get("audit", {})
    source = evidence.get("source_alignment", {})
    if not evidence:
        return rows

    canonical_row = {
        "cycle_count": (int(rows[-1].get("cycle_count", 0)) + 1) if rows else 1,
        "timestamp": evidence.get("generated_at_utc", "pending"),
        "evidence_score": None,
        "source_match_rate": source.get("matched_fraction", 0),
        "matched_source_targets": source.get("matched", 0),
        "source_target_count": source.get("targets_total", 0),
        "full_audit_records": audit.get("rows", 0),
    }

    # If the last row already reflects canonical state, keep history unchanged.
    if rows:
        last = rows[-1]
        last_match_rate = _to_float(last.get("source_match_rate")) or 0.0
        canonical_match_rate = _to_float(canonical_row["source_match_rate"]) or 0.0
        if (
            int(last.get("matched_source_targets", -1)) == int(canonical_row["matched_source_targets"])
            and int(last.get("source_target_count", -1)) == int(canonical_row["source_target_count"])
            and abs(last_match_rate - canonical_match_rate) < 1e-9
            and int(last.get("full_audit_records", -1)) == int(audit.get("rows", 0))
        ):
            return rows

    rows.append(canonical_row)
    return rows


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render(history: list[dict[str, Any]], evidence: dict[str, Any]) -> str:
    audit = evidence.get("audit", {})
    source = evidence.get("source_alignment", {})
    neural = evidence.get("neural", {})
    confidence = evidence.get("confidence", {})

    width, height = 1200, 560
    left, right = 82, 1136
    top, bottom = 168, 430

    rows = sorted(history, key=lambda row: int(row.get("cycle_count", 0)))
    rows = _merge_canonical_row(rows, evidence)

    cycles = [int(row.get("cycle_count", index + 1)) for index, row in enumerate(rows)]

    parsed_scores = [_to_float(row.get("evidence_score", None)) for row in rows]
    has_plot_scores = [score for score in parsed_scores if score is not None]
    scores = []
    last_score = None
    for score in parsed_scores:
        if score is not None:
            last_score = score
            scores.append(score)
            continue
        scores.append(last_score if last_score is not None else 0.0)

    match_rates = [float(row.get("source_match_rate", 0.0)) * 100.0 for row in rows]
    audit_rows = [int(row.get("full_audit_records", 0) or 0) for row in rows]
    has_audit_history = any(value > 0 for value in audit_rows)

    min_cycle, max_cycle = min(cycles), max(cycles)
    score_min = min(70.0, min(has_plot_scores) if has_plot_scores else 70.0)
    score_max = max(76.0, max(has_plot_scores) if has_plot_scores else 76.0)
    match_min = min(97.5, min(match_rates))
    match_max = max(98.2, max(match_rates))
    max_audit = max(audit_rows) if has_audit_history else int(audit.get("rows", 0) or 0)

    score_points = [
        (
            scale(cycle, min_cycle, max_cycle, left, right),
            scale(score, score_min, score_max, bottom, top),
        )
        for cycle, score in zip(cycles, scores)
    ]
    match_points = [
        (
            scale(cycle, min_cycle, max_cycle, left, right),
            scale(rate, match_min, match_max, bottom, top),
        )
        for cycle, rate in zip(cycles, match_rates)
    ]

    bars = []
    if has_audit_history and max_audit > 0:
        bar_width = max(2.0, (right - left) / max(1, len(rows)) * 0.45)
        for cycle, value in zip(cycles, audit_rows):
            if value <= 0:
                continue
            x = scale(cycle, min_cycle, max_cycle, left, right)
            h = scale(float(value), 0.0, float(max_audit), 0.0, bottom - top)
            bars.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{bottom - h:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="#334155" opacity="0.45"/>'
            )
    bar_svg = ("\n    " + "\n    ".join(bars)) if bars else ""

    latest = rows[-1]
    latest_score = _to_float(latest.get("evidence_score"))
    latest_score_text = "pending" if latest_score is None else f"{latest_score:.2f}"
    latest_match = int(latest.get("matched_source_targets", source.get("matched", 0)) or 0)
    latest_total = int(latest.get("source_target_count", source.get("targets_total", 0)) or 0)
    latest_time = latest.get("timestamp", evidence.get("generated_at_utc", "pending"))
    try:
        latest_date = datetime.fromisoformat(str(latest_time).replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        latest_date = str(latest_time)

    audit_note = (
        "Audit row bars use recorded history."
        if has_audit_history
        else "Audit row history will populate on future cycles; headline shows the current canonical audit."
    )

    best = max(
        (entry for entry in neural.get("leaderboard", []) if entry.get("lift_percent", {}).get("mean") is not None),
        key=lambda entry: entry.get("lift_percent", {}).get("mean", 0),
        default={},
    )
    best_text = "pending"
    if best:
        best_text = f"{best.get('name')} {best.get('lift_percent', {}).get('mean'):.3f}%"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">Collatz evidence history</title>
  <desc id="desc">Public-safe trend graph generated from sanitized evidence history and canonical evidence JSON.</desc>
  <rect width="{width}" height="{height}" fill="#070b16"/>
  <text x="64" y="62" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="31" font-weight="800">Collatz Evidence History</text>
  <text x="64" y="92" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15">Every recorded runner iteration; public-safe evidence only.</text>

  <g>
    <rect x="64" y="116" width="232" height="76" rx="8" fill="#121a33" stroke="#2a3a68"/>
    <text x="82" y="145" fill="#94a3b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" letter-spacing="2">CURRENT AUDIT</text>
    <text x="82" y="174" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="800">{esc(fmt_millions(audit.get("rows")))}</text>
  </g>
  <g>
    <rect x="312" y="116" width="232" height="76" rx="8" fill="#121a33" stroke="#2a3a68"/>
    <text x="330" y="145" fill="#94a3b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" letter-spacing="2">ITERATIONS</text>
    <text x="330" y="174" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="800">{len(rows)}</text>
  </g>
  <g>
    <rect x="560" y="116" width="232" height="76" rx="8" fill="#121a33" stroke="#2a3a68"/>
    <text x="578" y="145" fill="#94a3b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" letter-spacing="2">EVIDENCE SCORE</text>
    <text x="578" y="174" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="800">{esc(latest_score_text)}</text>
  </g>
  <g>
    <rect x="808" y="116" width="328" height="76" rx="8" fill="#121a33" stroke="#2a3a68"/>
    <text x="826" y="145" fill="#94a3b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" letter-spacing="2">CONFIDENCE</text>
    <text x="826" y="174" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="23" font-weight="800">{esc(confidence.get("label", "pending"))}</text>
  </g>

  <g>
    <rect x="64" y="210" width="1072" height="250" rx="8" fill="#0b1020" stroke="#2a3a68"/>
    <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#253453"/>
    <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#253453"/>
    <text x="84" y="238" fill="#cbd5e1" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="700">Cycle {min_cycle} to {max_cycle}</text>
    <text x="84" y="258" fill="#7384ad" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">Latest: {esc(latest_date)}; source {latest_match:,}/{latest_total:,}; best ablation {esc(best_text)}.</text>
{bar_svg}
    <polyline points="{polyline(score_points)}" fill="none" stroke="#67e8f9" stroke-width="3"/>
    <polyline points="{polyline(match_points)}" fill="none" stroke="#f472b6" stroke-width="2.5" opacity="0.9"/>
    <circle cx="{score_points[-1][0]:.1f}" cy="{score_points[-1][1]:.1f}" r="5" fill="#67e8f9"/>
    <circle cx="{match_points[-1][0]:.1f}" cy="{match_points[-1][1]:.1f}" r="5" fill="#f472b6"/>
    <text x="{left}" y="{bottom + 26}" fill="#94a3b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">cycle {min_cycle}</text>
    <text x="{right - 72}" y="{bottom + 26}" fill="#94a3b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">cycle {max_cycle}</text>
  </g>

  <g>
    <rect x="64" y="482" width="18" height="5" fill="#67e8f9"/>
    <text x="90" y="489" fill="#cbd5e1" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13">Evidence score</text>
    <rect x="220" y="482" width="18" height="5" fill="#f472b6"/>
    <text x="246" y="489" fill="#cbd5e1" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13">Source match rate</text>
    <rect x="398" y="478" width="18" height="13" fill="#334155" opacity="0.45"/>
    <text x="424" y="489" fill="#cbd5e1" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13">Audit rows</text>
  </g>
  <text x="64" y="528" fill="#7384ad" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">{esc(audit_note)} Generated from sanitized runner history plus canonical evidence; operational telemetry does not raise confidence.</text>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render public-safe evidence history SVG.")
    parser.add_argument("--history", default="data/generated/runner/history.jsonl")
    parser.add_argument("--evidence", default="data/generated/evidence/latest_public_summary.json")
    parser.add_argument("--output", default="docs/media/evidence-history.svg")
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence).read_text())
    history = load_history(Path(args.history))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(history, evidence))


if __name__ == "__main__":
    main()
