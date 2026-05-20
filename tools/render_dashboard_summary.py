#!/usr/bin/env python3
"""Render the README dashboard SVG from canonical public evidence JSON."""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def compact_int(value: int | float | None) -> str:
    if value is None:
        return "pending"
    value = int(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.{digits}f}%"


def best_leaderboard(data: dict) -> dict | None:
    entries = data["neural"]["leaderboard"]
    present = [entry for entry in entries if entry["lift_percent"]["mean"] is not None]
    if not present:
        return None
    return max(present, key=lambda entry: entry["lift_percent"]["mean"])


def entry_lift(data: dict, name: str) -> float | None:
    for entry in data["neural"]["leaderboard"]:
        if entry["name"] == name:
            return entry["lift_percent"]["mean"]
    return None


def bar(label: str, value: float | None, max_value: float, y: int, color: str) -> str:
    width = 372
    fill_width = 0 if value is None or max_value <= 0 else int(width * min(max(value, 0.0) / max_value, 1.0))
    return f"""
    <text x="88" y="{y}" fill="#dbe7ff" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">{esc(label)}</text>
    <rect x="88" y="{y + 14}" width="{width}" height="18" rx="9" fill="#1f2a44"/>
    <rect x="88" y="{y + 14}" width="{fill_width}" height="18" rx="9" fill="{color}"/>
    <text x="474" y="{y + 29}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="800">{pct(value, 3)}</text>"""


def text_block(
    text: str,
    x: int,
    y: int,
    width: int,
    color: str = "#aab6d3",
    size: int = 14,
    weight: str | None = None,
    line_height: int = 20,
) -> str:
    chars = max(24, width // max(size // 2, 6))
    lines = textwrap.wrap(text, width=chars)
    if not lines:
        lines = [""]
    weight_attr = f' font-weight="{weight}"' if weight else ""
    return "\n".join(
        f'    <text x="{x}" y="{y + index * line_height}" fill="{color}" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="{size}"{weight_attr}>{esc(line)}</text>'
        for index, line in enumerate(lines)
    )


def render(data: dict) -> str:
    confidence = data["confidence"]
    audit = data["audit"]
    coverage = data["coverage"]
    neural = data["neural"]
    holdouts = neural["holdouts"]
    source = data["source_alignment"]
    best = best_leaderboard(data)
    best_name = best["name"] if best else "pending"
    best_lift = best["lift_percent"]["mean"] if best else None
    hybrid_lift = entry_lift(data, "hybrid")
    metrics_lift = entry_lift(data, "metrics-only")
    source_rate = float(source["matched_fraction"]) * 100.0
    topology_percent = coverage["topology"]["percent_of_audit"]
    sample_percent = coverage["stratified_evidence_sample"]["percent_of_audit"]
    max_bar = max(
        1.0,
        best_lift or 0.0,
        hybrid_lift or 0.0,
        holdouts["weakest_range_lift_percent"] or 0.0,
        holdouts["numeric_adjacency_lift_percent"] or 0.0,
    )
    max_bar = max_bar * 1.15
    blockers = confidence.get("promotion_blockers", [])
    blocker_text = ", ".join(blockers[:2]) if blockers else "none"
    if len(blockers) > 2:
        blocker_text += f", +{len(blockers) - 2} more"
    finding = (
        f"{best_name} is currently strongest at {pct(best_lift, 3)}, "
        f"while hybrid is {pct(hybrid_lift, 3)}. "
        "This keeps the interpretation metric-dominant."
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">
  <title id="title">Collatz evidence dashboard snapshot</title>
  <desc id="desc">Static public README snapshot rendered from the canonical public evidence JSON.</desc>
  <defs>
    <linearGradient id="panel" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#121a34"/>
      <stop offset="1" stop-color="#0b1020"/>
    </linearGradient>
    <linearGradient id="green" x1="0" x2="1">
      <stop offset="0" stop-color="#5eead4"/>
      <stop offset="1" stop-color="#86efac"/>
    </linearGradient>
    <linearGradient id="gold" x1="0" x2="1">
      <stop offset="0" stop-color="#facc15"/>
      <stop offset="1" stop-color="#fb7185"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#000000" flood-opacity=".28"/>
    </filter>
  </defs>

  <rect width="1200" height="680" fill="#080d1a"/>
  <text x="64" y="66" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="34" font-weight="800">3xN1 Collatz Evidence Dashboard</text>
  <text x="64" y="96" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="16">Generated from data/generated/evidence/latest_public_summary.json. No live runtime or infrastructure details are shown.</text>

  <g filter="url(#softShadow)">
    <rect x="64" y="132" width="260" height="118" rx="10" fill="url(#panel)" stroke="#2b385e"/>
    <text x="84" y="162" fill="#9fb0d4" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">CONFIDENCE</text>
    <text x="84" y="202" fill="#e2e8f0" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="23" font-weight="800">{esc(confidence["label"])}</text>
    <text x="84" y="226" fill="#86efac" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15" font-weight="700">empirical, not proof</text>

    <rect x="348" y="132" width="260" height="118" rx="10" fill="url(#panel)" stroke="#2b385e"/>
    <text x="368" y="162" fill="#9fb0d4" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">AUDIT</text>
    <text x="368" y="202" fill="#e2e8f0" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="28" font-weight="800">{compact_int(audit["rows"])}</text>
    <text x="458" y="202" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="17">rows</text>
    <text x="368" y="226" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">range {audit["range_start"]:,}..{audit["range_end"]:,}</text>

    <rect x="632" y="132" width="260" height="118" rx="10" fill="url(#panel)" stroke="#2b385e"/>
    <text x="652" y="162" fill="#9fb0d4" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">SOURCE CHECK</text>
    <text x="652" y="202" fill="#e2e8f0" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="28" font-weight="800">{source["matched"]:,} / {source["targets_total"]:,}</text>
    <text x="652" y="226" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">{pct(source_rate, 3)} public targets matched</text>

    <rect x="916" y="132" width="220" height="118" rx="10" fill="url(#panel)" stroke="#2b385e"/>
    <text x="936" y="162" fill="#9fb0d4" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">LIMIT</text>
    <text x="936" y="199" fill="#facc15" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="18" font-weight="800">not a proof</text>
    <text x="936" y="224" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">gate blockers remain</text>
  </g>

  <g filter="url(#softShadow)">
    <rect x="64" y="278" width="520" height="310" rx="10" fill="url(#panel)" stroke="#2b385e"/>
    <text x="88" y="314" fill="#e2e8f0" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="800">Neural Evidence</text>
    <text x="88" y="338" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">Lift values compare learned neighborhoods against controlled baselines.</text>
{bar(f"Best ablation: {best_name}", best_lift, max_bar, 384, "url(#green)")}
{bar("Hybrid representation", hybrid_lift, max_bar, 444, "#60a5fa")}
{bar("Weakest range holdout", holdouts["weakest_range_lift_percent"], max_bar, 504, "#a78bfa")}
    <text x="88" y="562" fill="#86efac" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15" font-weight="800">Numeric-adjacency lift: {pct(holdouts["numeric_adjacency_lift_percent"], 3)}</text>
  </g>

  <g filter="url(#softShadow)">
    <rect x="616" y="278" width="520" height="310" rx="10" fill="url(#panel)" stroke="#2b385e"/>
    <text x="640" y="314" fill="#e2e8f0" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="800">Current Finding</text>
{text_block(finding, 640, 342, 430)}

    <rect x="640" y="374" width="448" height="64" rx="8" fill="#070b16" stroke="#263450"/>
    <text x="660" y="400" fill="#9fb0d4" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">COVERAGE</text>
    <text x="660" y="424" fill="#dbe7ff" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">Topology {coverage["topology"]["rows"]:,} rows ({pct(topology_percent, 3)}); stratified sample {coverage["stratified_evidence_sample"]["rows"]:,} rows ({pct(sample_percent, 3)}).</text>

    <rect x="640" y="456" width="448" height="64" rx="8" fill="#070b16" stroke="#263450"/>
    <text x="660" y="482" fill="#9fb0d4" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">PROMOTION BLOCKERS</text>
{text_block(blocker_text, 660, 504, 395, "#facc15", 13, line_height=18)}

    <rect x="640" y="540" width="448" height="1" fill="#263450"/>
    <text x="640" y="566" fill="#e2e8f0" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15" font-weight="800">Next falsification test</text>
{text_block(data["next_experiment"]["summary"], 640, 590, 430, "#aab6d3", 13, line_height=17)}
  </g>

  <text x="64" y="642" fill="#7384ad" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12">Generated at {esc(data["generated_at_utc"])} from public-safe canonical evidence. Operations telemetry never raises confidence.</text>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README dashboard SVG from canonical evidence JSON.")
    parser.add_argument("--input", default="data/generated/evidence/latest_public_summary.json")
    parser.add_argument("--output", default="docs/media/dashboard-summary.svg")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(data))


if __name__ == "__main__":
    main()
