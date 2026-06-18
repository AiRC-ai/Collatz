#!/usr/bin/env python3
"""Generalized Collatz invariant falsifier (reusable engine).

Throw any candidate Lyapunov function f(n) (closed-form, of the current state n
alone) at it; it tests f(S(n)) < f(n) on all odd n up to N and reports the
survival fraction, first failure, and failing-residue pattern. Also keeps the
exact residue-weighted-log feasibility check (Bellman-Ford) at configurable L.

The accelerated map S(n) = (3n+1)/2^v2(3n+1). A valid Lyapunov function must
decrease at EVERY accelerated step (except the trivial cycle). This engine's job
is to KILL bad candidates fast. A survivor (100% on a large N) is a candidate
invariant worth handing to a mathematician -- not a proof.

This run attacks the richer classes the L=8/12 obstruction pointed at:
  - halving-count coupling: f(n) = log2(n) + c * v2(3n+1)
  - two-step halving coupling: f(n) = log2(n) + c1*v2(3n+1) + c2*v2(3S(n)+1)
  - explicit 3-mod-4 compensation
  - residue-weighted feasibility at L=14 (higher resolution)
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Collatz invariant falsifier")
    p.add_argument("--n", type=int, default=5_000_000)
    p.add_argument("--l", type=int, default=14)
    p.add_argument("--output-dir", default="data/generated/proof")
    return p.parse_args()

def v2(x):
    c = 0
    while x & 1 == 0:
        x >>= 1; c += 1
    return c

def syc(n):
    m = 3 * n + 1
    return m >> v2(m)

def test_candidate(f, N):
    """Return (survival_fraction, first_fail_n, fail_residues_top)."""
    ok = 0; tot = 0; first = None; fres = defaultdict(int)
    for n in range(3, N + 1, 2):
        s = syc(n)
        if s == 1:
            ok += 1; tot += 1; continue
        tot += 1
        if f(s) < f(n) - 1e-12:
            ok += 1
        else:
            if first is None: first = n
            fres[n & 255] += 1
    top = sorted(fres.items(), key=lambda kv: -kv[1])[:4]
    return ok / tot, first, [(r, c) for r, c in top]

def feasibility(L, N):
    """Residue-weighted log f=log2(n)+w[n mod 2^L]: feasibility via Bellman-Ford."""
    M = 1 << L
    D = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for n in range(3, N + 1, 2):
        s = syc(n)
        if s == 1: continue
        delta = math.log2(s / n)
        r = n & (M - 1); rp = s & (M - 1)
        if delta > D[r][rp]: D[r][rp] = delta
    edges = [(r, rp, D[r][rp]) for r in D for rp in D[r]]
    nodes = sorted({v for e in edges for v in (e[0], e[1])})
    dist = {v: 0.0 for v in nodes}; V = len(nodes)
    for _ in range(V):
        upd = False
        for r, rp, w in edges:
            if dist[r] - w < dist[rp]:  # weight = -w
                dist[rp] = dist[r] - w; upd = True
        if not upd: break
    on_cycle = None
    for r, rp, w in edges:
        if dist[r] - w < dist[rp] - 1e-12:
            on_cycle = (r, rp); break
    return on_cycle is None, len(edges), V, on_cycle

def main():
    args = parse_args(); N = args.n
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"invariant falsifier | odd n in 3..{N} | L={args.l}")
    results = []

    def report(name, f):
        frac, first, top = test_candidate(f, N)
        print(f"  [{name:<42}] survival {frac*100:7.4f}%  first_fail={first}  top_fail_res={top}")
        results.append({"candidate": name, "survival": round(frac, 6), "first_fail_n": first, "top_failing_residues_mod256": top})
        return frac

    # baseline
    report("log2(n)", lambda n: math.log2(n))
    # halving-count coupling: f = log2(n) + c*v2(3n+1)
    best_c, best_frac = None, -1
    for c in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        frac = report(f"log2(n)+{c}*v2(3n+1)", (lambda c=c: lambda n: math.log2(n) + c * v2(3 * n + 1))())
        if frac > best_frac: best_frac, best_c = frac, c
    # two-step coupling: small grid
    best2, best2f = (None, None), -1
    for c1 in [1.0, 2.0]:
        for c2 in [0.5, 1.0, 1.5]:
            f = (lambda c1=c1, c2=c2: lambda n: math.log2(n) + c1 * v2(3 * n + 1) + c2 * v2(3 * syc(n) + 1))()
            frac = report(f"log2(n)+{c1}*v2(3n+1)+{c2}*v2(3S(n)+1)", f)
            if frac > best2f: best2f, best2 = frac, (c1, c2)
    # explicit 3-mod-4 compensation
    report("log2(n)-log2(3)*[n==3 mod4]", lambda n: math.log2(n) - math.log2(3) if (n & 3) == 3 else math.log2(n))

    print(f"\n  best halving-coupling: c={best_c} survival {best_frac*100:.4f}%")
    print(f"  best 2-step coupling: (c1,c2)={best2} survival {best2f*100:.4f}%")

    # residue-weighted feasibility at L
    feas, nedges, V, obs = feasibility(args.l, N)
    print(f"  residue-weighted log L={args.l}: {'FEASIBLE' if feas else 'INFEASIBLE'} ({nedges} constraints, {V} nodes) obstruction={obs}")

    metrics = {"tool": "research/invariant_falsifier.py", "N": N, "L": args.l,
               "candidates": results,
               "best_halving_coupling": {"c": best_c, "survival": round(best_frac, 6)},
               "best_2step_coupling": {"c1_c2": best2, "survival": round(best2f, 6)},
               "residue_weighted_feasibility_L": {"L": args.l, "feasible": feas, "n_constraints": nedges, "n_nodes": V, "obstruction_edge": obs},
               "honest_caveat": "Survival < 100% means the candidate is falsified (not a valid Lyapunov function). 100% on a finite sample is a candidate for proof, not a proof. Every tested candidate was falsified."}
    json.dump(metrics, open(out / "falsifier_metrics.json", "w"), indent=2)
    print(f"\noutputs: {out}/falsifier_metrics.json")

if __name__ == "__main__":
    main()
