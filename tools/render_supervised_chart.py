#!/usr/bin/env python3
"""Render the headline model-comparison chart (SVG): original (metrics-only) vs
ours (hybrid), with the non-AI raw-metrics baseline as the floor.

Reads data/generated/evidence/model_comparison.json and emits
docs/media/v9-v10-supervised-chart.svg. One consistent metric throughout:
k=2 nearest-neighbor same-label purity minus the random baseline, over 100,000
Collatz trajectories. Both trained models use the v10 multi-task supervised
method; only the input features differ (original=metrics-only, ours=hybrid=all
branches).
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

COL = {"raw": "#64748b", "original": "#60a5fa", "ours": "#22c55e"}  # slate, blue, green
LEGEND = {"raw": "non-AI baseline (raw metrics)", "original": "original (metrics-only)", "ours": "ours (hybrid, all branches)"}
SHORT = {"range_band": "range_band", "bit_length": "bit_length", "peak_ratio_bucket": "peak_ratio"}
V7_RANGE_BAND_LIFT = 0.00233  # prior AI: v7 self-supervised contrastive (retired)


def esc(s): return html.escape(str(s), quote=True)


def load_data(path):
    d = json.loads(Path(path).read_text())
    return [{"label": r["label"], "n_classes": r["n_classes"], "baseline": r["random_baseline"],
             "raw": r["non_ai_raw_metrics_lift"], "original": r["original_metrics_only_lift"],
             "ours": r["ours_hybrid_lift"]} for r in d["results"]]


def render(rows):
    W, H, max_val = 1200, 680, 1.0
    left, top, plot_w, plot_h = 64, 196, 1072, 360
    group_w = plot_w / len(rows)
    bar_w, gap = 76, 14
    def y(v): return top + plot_h - plot_h * (max(v, 0) / max_val)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">']
    svg.append('<title id="title">Original (metrics-only) vs ours (hybrid)</title>')
    svg.append('<desc id="desc">k=2 neighbor-purity lift by label: non-AI raw baseline vs original metrics-only vs ours hybrid, all under the v10 supervised method.</desc>')
    svg.append('<defs><linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
               '<stop offset="0" stop-color="#101a33"/><stop offset="1" stop-color="#0b1326"/></linearGradient></defs>')
    svg.append(f'<rect width="{W}" height="{H}" fill="#080d1a"/>')
    svg.append(f'<text x="64" y="62" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="800">Original (metrics-only) vs ours (hybrid)</text>')
    svg.append(f'<text x="64" y="90" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15">k=2 neighbor-purity lift over random baseline (100k trajectories, one metric). Both models beat the non-AI baseline; ours (hybrid) is tied with original (metrics-only).</text>')
    lx = 64
    for key in ("raw", "original", "ours"):
        svg.append(f'<rect x="{lx}" y="112" width="16" height="16" rx="4" fill="{COL[key]}"/>')
        svg.append(f'<text x="{lx+24}" y="125" fill="#dbe7ff" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">{LEGEND[key]}</text>')
        lx += 290
    svg.append(f'<rect x="{left-20}" y="{top-30}" width="{plot_w+40}" height="{plot_h+70}" rx="10" fill="url(#panel)" stroke="#2b385e"/>')
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = top + plot_h - plot_h * frac
        svg.append(f'<line x1="{left}" y1="{gy}" x2="{left+plot_w}" y2="{gy}" stroke="#1f2a44" stroke-width="1"/>')
        svg.append(f'<text x="{left-10}" y="{gy+5}" fill="#64748b" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" text-anchor="end">{int(frac*100)}</text>')
    svg.append(f'<text x="{left-44}" y="{top+plot_h//2}" fill="#8693b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 {left-44} {top+plot_h//2})">lift (%)</text>')
    for gi, r in enumerate(rows):
        gx = left + group_w * gi + group_w / 2
        vals = [("raw", r["raw"]), ("original", r["original"]), ("ours", r["ours"])]
        total_w = 3 * bar_w + 2 * gap
        start_x = gx - total_w / 2
        for bi, (key, v) in enumerate(vals):
            bx = start_x + bi * (bar_w + gap)
            by, bh = y(v), top + plot_h - y(v)
            svg.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" rx="6" fill="{COL[key]}"/>')
            svg.append(f'<text x="{bx+bar_w/2}" y="{by-8}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="800" text-anchor="middle">{v*100:+.1f}</text>')
        by = y(r["baseline"])
        svg.append(f'<line x1="{gx-total_w/2-6}" y1="{by}" x2="{gx+total_w/2+6}" y2="{by}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5 4"/>')
        svg.append(f'<text x="{gx+total_w/2+10}" y="{by+4}" fill="#f59e0b" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11">random {r["baseline"]:.3f}</text>')
        svg.append(f'<text x="{gx}" y="{top+plot_h+34}" fill="#dbe7ff" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15" font-weight="700" text-anchor="middle">{SHORT[r["label"]]} ({r["n_classes"]})</text>')
    # progress strip
    py = 600
    svg.append(f'<rect x="64" y="{py-44}" width="1072" height="92" rx="10" fill="url(#panel)" stroke="#2b385e"/>')
    svg.append(f'<text x="84" y="{py-22}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="800">Progress on range_band lift: the win was the supervised METHOD over self-supervised contrastive, not the hybrid FEATURES over metrics-only.</text>')
    steps = [("prior AI: v7\nself-supervised", V7_RANGE_BAND_LIFT, "#ef4444"),
             ("non-AI baseline\nraw metrics", rows[0]["raw"], "#64748b"),
             ("original:\nmetrics-only", rows[0]["original"], "#60a5fa"),
             ("ours:\nhybrid", rows[0]["ours"], "#22c55e")]
    sw = 250
    for i, (name, v, color) in enumerate(steps):
        sx = 84 + i * sw; bw = 210
        svg.append(f'<rect x="{sx}" y="{py-6}" width="{bw}" height="22" rx="6" fill="#1f2a44"/>')
        svg.append(f'<rect x="{sx}" y="{py-6}" width="{int(bw*min(v/max_val,1.0))}" height="22" rx="6" fill="{color}"/>')
        svg.append(f'<text x="{sx+bw/2}" y="{py+8}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="700" text-anchor="middle">{v*100:+.2f}%</text>')
        for li, line in enumerate(name.split("\n")):
            svg.append(f'<text x="{sx+bw/2}" y="{py+26+li*13}" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" text-anchor="middle">{esc(line)}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description="Render original-vs-ours comparison chart SVG.")
    ap.add_argument("--input", default="data/generated/evidence/model_comparison.json")
    ap.add_argument("--output", default="docs/media/v9-v10-supervised-chart.svg")
    args = ap.parse_args()
    Path(args.output).write_text(render(load_data(args.input)))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
