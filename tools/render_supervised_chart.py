#!/usr/bin/env python3
"""Render the v9/v10 supervised-embedding comparison chart (SVG).

Shows, for each coarse label, the neighbor-purity lift (k=2) of:
  - raw-metrics k-NN (zero training, the honest baseline)
  - v9  single-task supervised embedding (trained on range_band)
  - v10 multi-task supervised embedding (trained on range_band+bit_length+peak_ratio)
plus the random baseline per label, and a progress strip that places the
self-supervised v7 collapse (0.23%) next to the supervised wins.

Lift numbers are read from the committed run metrics.json files where available;
v9's bit_length / peak_ratio lifts come from a held-out k-NN comparison run
(documented in docs/NEURAL_ENGINE_V9.md) and are recorded here as constants.
"""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

# v9 multi-label lifts not stored in v9 metrics.json (which only records the
# range_band training target). Values from the v9-vs-raw held-out comparison.
V9_EXTRA = {"bit_length": 0.7340, "peak_ratio_bucket": 0.2373}
V7_RANGE_BAND_LIFT = 0.00233  # v7 self-supervised contrastive, range_band

LABELS = ["range_band", "bit_length", "peak_ratio_bucket"]
SHORT = {"range_band": "range_band", "bit_length": "bit_length", "peak_ratio_bucket": "peak_ratio"}
COL = {"raw": "#64748b", "v9": "#a78bfa", "v10": "#22c55e"}  # slate, purple, green


def esc(s): return html.escape(str(s), quote=True)


def load_data(v9_path, v10_path):
    v10 = {c["label"]: c for c in json.loads(Path(v10_path).read_text())["comparison"]}
    v9m = json.loads(Path(v9_path).read_text())
    v9 = {"range_band": v9m["purity_lift"]}
    v9.update(V9_EXTRA)
    rows = []
    for lab in LABELS:
        c = v10[lab]
        rows.append({
            "label": lab,
            "baseline": c["random_baseline_purity"],
            "raw": c["raw_metrics_knn_lift"],
            "v9": v9[lab],
            "v10": c["v10_embedding_lift"],
            "acc": c["v10_test_accuracy"],
        })
    return rows


def render(rows):
    W, H = 1200, 680
    max_val = 1.0
    # main bar panel geometry
    left, top, plot_w, plot_h = 64, 196, 1072, 360
    n_groups = len(rows)
    group_w = plot_w / n_groups
    bar_w = 70
    gap = 14
    # y scale
    def y(v): return top + plot_h - plot_h * (max(v, 0) / max_val)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">']
    svg.append(f'<title id="title">Supervised embedding vs raw-metrics k-NN</title>')
    svg.append(f'<desc id="desc">Neighbor-purity lift (k=2) by label for raw-metrics k-NN, v9 single-task, and v10 multi-task supervised embeddings.</desc>')
    svg.append('<defs><linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
               '<stop offset="0" stop-color="#101a33"/><stop offset="1" stop-color="#0b1326"/></linearGradient></defs>')
    svg.append(f'<rect width="{W}" height="{H}" fill="#080d1a"/>')
    # title
    svg.append(f'<text x="64" y="62" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="26" font-weight="800">Supervised embedding beats raw metrics on every coarse label</text>')
    svg.append(f'<text x="64" y="90" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15">Neighbor-purity lift over random baseline (k=2, 100k trajectories). v10 multi-task trains one 64-d embedding to predict all three labels at once.</text>')
    # legend
    lx = 64
    for key, name in [("raw", "raw-metrics k-NN"), ("v9", "v9 single-task"), ("v10", "v10 multi-task")]:
        svg.append(f'<rect x="{lx}" y="112" width="16" height="16" rx="4" fill="{COL[key]}"/>')
        svg.append(f'<text x="{lx+24}" y="125" fill="#dbe7ff" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14">{name}</text>')
        lx += 200
    # main panel
    svg.append(f'<rect x="{left-20}" y="{top-30}" width="{plot_w+40}" height="{plot_h+70}" rx="10" fill="url(#panel)" stroke="#2b385e"/>')
    # gridlines + y labels
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gy = top + plot_h - plot_h * frac
        svg.append(f'<line x1="{left}" y1="{gy}" x2="{left+plot_w}" y2="{gy}" stroke="#1f2a44" stroke-width="1"/>')
        svg.append(f'<text x="{left-10}" y="{gy+5}" fill="#64748b" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" text-anchor="end">{int(frac*100)}</text>')
    svg.append(f'<text x="{left-44}" y="{top+plot_h//2}" fill="#8693b8" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="13" text-anchor="middle" transform="rotate(-90 {left-44} {top+plot_h//2})">lift (%)</text>')

    for gi, r in enumerate(rows):
        gx = left + group_w * gi + group_w / 2
        vals = [("raw", r["raw"]), ("v9", r["v9"]), ("v10", r["v10"])]
        total_bars_w = 3 * bar_w + 2 * gap
        start_x = gx - total_bars_w / 2
        for bi, (key, v) in enumerate(vals):
            bx = start_x + bi * (bar_w + gap)
            by = y(v)
            bh = top + plot_h - by
            svg.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" rx="6" fill="{COL[key]}"/>')
            svg.append(f'<text x="{bx+bar_w/2}" y="{by-8}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="800" text-anchor="middle">{v*100:+.1f}</text>')
        # random baseline dashed line across the group
        by = y(r["baseline"])
        svg.append(f'<line x1="{gx-total_bars_w/2-6}" y1="{by}" x2="{gx+total_bars_w/2+6}" y2="{by}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5 4"/>')
        svg.append(f'<text x="{gx+total_bars_w/2+10}" y="{by+4}" fill="#f59e0b" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11">random {r["baseline"]:.3f}</text>')
        # x label + v10 acc
        svg.append(f'<text x="{gx}" y="{top+plot_h+34}" fill="#dbe7ff" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="15" font-weight="700" text-anchor="middle">{SHORT[r["label"]]}</text>')
        svg.append(f'<text x="{gx}" y="{top+plot_h+54}" fill="#86efac" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" text-anchor="middle">v10 acc {r["acc"]*100:.1f}%</text>')

    # progress strip
    py = 600
    svg.append(f'<rect x="64" y="{py-44}" width="1072" height="92" rx="10" fill="url(#panel)" stroke="#2b385e"/>')
    svg.append(f'<text x="84" y="{py-22}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="800">Progress (range_band lift): self-supervised contrastive collapsed; supervised embedding recovers and generalizes.</text>')
    steps = [
        ("v7 self-supervised\ncontrastive", V7_RANGE_BAND_LIFT, "#ef4444"),
        ("raw-metrics k-NN\n(zero training)", rows[0]["raw"], "#64748b"),
        ("v9 single-task\nsupervised", rows[0]["v9"], "#a78bfa"),
        ("v10 multi-task\nsupervised (sweep)", rows[0]["v10"], "#22c55e"),
    ]
    sw = 250
    for i, (name, v, color) in enumerate(steps):
        sx = 84 + i * sw
        bw = 210
        svg.append(f'<rect x="{sx}" y="{py-6}" width="{bw}" height="22" rx="6" fill="#1f2a44"/>')
        fw = int(bw * min(v / max_val, 1.0))
        svg.append(f'<rect x="{sx}" y="{py-6}" width="{fw}" height="22" rx="6" fill="{color}"/>')
        svg.append(f'<text x="{sx+bw/2}" y="{py+8}" fill="#f8fafc" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="700" text-anchor="middle">{v*100:+.2f}%</text>')
        for li, line in enumerate(name.split("\n")):
            svg.append(f'<text x="{sx+bw/2}" y="{py+26+li*13}" fill="#aab6d3" font-family="Inter, ui-sans-serif, system-ui, sans-serif" font-size="11" text-anchor="middle">{esc(line)}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description="Render v9/v10 supervised comparison chart SVG.")
    ap.add_argument("--v9", default="data/generated/contrastive_v9/metrics.json")
    ap.add_argument("--v10", default="data/generated/contrastive_v10/metrics.json")
    ap.add_argument("--output", default="docs/media/v9-v10-supervised-chart.svg")
    args = ap.parse_args()
    rows = load_data(args.v9, args.v10)
    Path(args.output).write_text(render(rows))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
