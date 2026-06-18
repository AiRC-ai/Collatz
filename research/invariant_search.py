#!/usr/bin/env python3
"""Candidate-invariant discovery for Collatz (the honest ML->proof bridge).

This does NOT prove Collatz. It searches a bounded family of closed-form,
state-level candidate Lyapunov functions for the accelerated (Syracuse) map
S(n) = (3n+1)/2^v2(3n+1) (odd n -> odd n), and empirically stress-tests them on
all odd n up to N. Any candidate that holds on 100% of the sample is emitted as
a precise CONJECTURE (with the sample range) for a mathematician to prove /
formalize; failures are reported honestly.

Candidates:
  A. 1-step: S(n) < n. (Known to fail for n == 3 mod 4.)
  B. k-step: S^k(n) < n for fixed k = 1..6. (A fixed-k that holds for all n
     would be a Lyapunov function -> essentially a proof.)
  C. Residue-weighted log Lyapunov: f(n) = log2(n) + w[n mod 2^L]. The decrease
     condition f(S(n)) < f(n) is a system of linear difference constraints in w.
     We reduce to w[r]-w[r'] > D[r][r'] (D = max log2(S(n)/n) over sampled
     transitions r->r') and check feasibility via Bellman-Ford negative-cycle
     detection on the -D graph. Feasible => a residue-weighted function exists
     on the sample (candidate invariant). Infeasible => an obstructing residue
     cycle is reported. Pure Python, no LP solver needed.

Empirical 100% on a finite sample is necessary-but-not-sufficient evidence; it
is a candidate for rigorous proof, not a proof.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Collatz candidate-invariant search")
    p.add_argument("--n", type=int, default=10_000_000, help="test odd n in 1..N")
    p.add_argument("--l", type=int, default=8, help="residue bits for candidate C")
    p.add_argument("--kmax", type=int, default=6)
    p.add_argument("--output-dir", default="data/generated/proof")
    return p.parse_args()

def v2(x):
    c = 0
    while x & 1 == 0:
        x >>= 1; c += 1
    return c

def syc(n):
    """Syracuse/accelerated map: odd n -> odd (3n+1)/2^v2(3n+1)."""
    m = 3 * n + 1
    return m >> v2(m)

def main():
    args = parse_args()
    N = args.n; L = args.l; M = 1 << L
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"invariant search | odd n in 3..{N} (step 2) | L={L} ({M} residue classes) | kmax={args.kmax}")

    # ---- A. 1-step decrease ----
    a_fail = 0; a_total = 0; fail_res = defaultdict(int)
    for n in range(3, N + 1, 2):
        s = syc(n); a_total += 1
        if not (s < n):
            a_fail += 1; fail_res[n & (M - 1)] += 1
    a_rate = 1 - a_fail / a_total
    print(f"[A] 1-step S(n)<n: holds on {a_rate*100:.4f}% of {a_total} odd n (fails {a_fail})")
    top_fail = sorted(fail_res.items(), key=lambda kv: -kv[1])[:4]
    print(f"    top failing residues (mod {M}): {[(r,c) for r,c in top_fail]}")

    # ---- B. k-step decrease ----
    b = {}
    for k in range(1, args.kmax + 1):
        ok = 0; tot = 0
        for n in range(3, N + 1, 2):
            x = n; good = True
            for _ in range(k):
                x = syc(x)
                if x == 1: good = True; break
            if x == 1 or x < n: ok += 1
            tot += 1
        b[k] = ok / tot
        print(f"[B] k={k}: S^k(n)<n (or reach 1) on {b[k]*100:.4f}%")

    # ---- C. residue-weighted log Lyapunov feasibility ----
    # D[r][r'] = max log2(S(n)/n) over sampled odd transitions with n mod M = r, S(n) mod M = r'
    D = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    tight = 0
    for n in range(3, N + 1, 2):
        s = syc(n)
        if s == 1: continue
        delta = math.log2(s / n)  # want w[r]-w[r'] > delta
        r = n & (M - 1); rp = s & (M - 1)
        if delta > D[r][rp]:
            D[r][rp] = delta; tight += 1
    # feasibility: exists w with w[r]-w[r'] > D[r][r'] for all observed (r,r')
    # iff no cycle in graph (nodes residues, edge r->r' weight D[r][r']) has sum >= 0.
    # Check via Bellman-Ford for negative cycle on edge weight = -D.
    edges = [(r, rp, D[r][rp]) for r in D for rp in D[r]]
    nodes = set()
    for r, rp, _ in edges: nodes.add(r); nodes.add(rp)
    nodes = sorted(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    INF = float("inf")
    dist = {v: 0.0 for v in nodes}
    neg_edge_w = [(r, rp, -w) for r, rp, w in edges]  # weight = -D
    # Bellman-Ford from super-source (all dist 0); detect negative cycle
    V = len(nodes)
    for it in range(V):
        updated = False; upd = None
        for r, rp, w in neg_edge_w:
            if dist[r] + w < dist[rp]:
                dist[rp] = dist[r] + w; updated = True; upd = (r, rp, w)
        if not updated:
            break
    # one more relaxation pass to detect a node still on a negative cycle
    on_cycle = None
    for r, rp, w in neg_edge_w:
        if dist[r] + w < dist[rp] - 1e-12:
            on_cycle = (r, rp, w); break
    feasible = on_cycle is None
    print(f"[C] residue-weighted log Lyapunov (L={L}): {'FEASIBLE on sample' if feasible else 'INFEASIBLE'} ({len(edges)} binding pair-constraints, {V} residue nodes)")
    C = {"feasible_on_sample": feasible, "L": L, "n_residue_nodes": V, "n_pair_constraints": len(edges)}
    if feasible:
        # w = -dist (potentials) satisfy w[r]-w[r'] >= D[r][r']; verify on sample
        w = {v: -dist[v] for v in nodes}
        viol = 0; checked = 0
        for n in range(3, N + 1, 2):
            s = syc(n)
            if s == 1: continue
            r = n & (M - 1); rp = s & (M - 1)
            if r not in w or rp not in w: continue
            checked += 1
            if not (w[r] - w[rp] > math.log2(s / n)): viol += 1
        C["weights_sample"] = {str(k): round(v, 4) for k, v in list(w.items())[:16]}
        C["verified_transitions"] = checked; C["violations"] = viol
        print(f"    weights verify on {checked} transitions: {viol} violations")
    else:
        # report the obstructing edge / a short cycle through it
        C["obstruction_edge"] = {"from": on_cycle[0], "to": on_cycle[1], "log2_ratio_required": round(-on_cycle[2], 4)}
        print(f"    obstructing edge residue {on_cycle[0]} -> {on_cycle[1]} (negative-cycle in -D)")

    metrics = {"tool": "research/invariant_search.py", "N": N, "L": L, "kmax": args.kmax,
               "A_1step_decrease_rate": a_rate, "A_failures": a_fail, "A_total_odd_n": a_total,
               "B_kstep_decrease_rates": b,
               "C_residue_weighted_lyapunov": C,
               "honest_caveat": "Empirical 100% on a finite sample is necessary-but-not-sufficient. A feasible candidate C is a CONJECTURE to prove, not a proof. 1-step and k-step rates < 100% confirm no simple fixed-step Lyapunov function exists (consistent with Collatz being open)."}
    json.dump(metrics, open(out / "invariant_metrics.json", "w"), indent=2)
    print(f"\noutputs: {out}/invariant_metrics.json")
    # write conjectures doc
    doc = ["# Collatz candidate invariants (empirical, NOT a proof)", "",
           f"Searched closed-form Lyapunov candidates for the accelerated map S(n)=(3n+1)/2^v2(3n+1) on all odd n in 3..{N}.", "",
           "## A. 1-step S(n) < n", f"- holds on {a_rate*100:.4f}% of {a_total} odd n. Fails {a_fail} times.",
           "- Known: fails for n = 3 mod 4 (S(n) = (3n+1)/2 > n). NOT an invariant.", "",
           "## B. k-step S^k(n) < n"]
    for k, r in b.items():
        doc.append(f"- k={k}: {r*100:.4f}%")
    doc += ["- No fixed k reaches 100% on the sample; a fixed-k global decrease would be a Lyapunov function (essentially a proof). None found.", "",
            "## C. residue-weighted log Lyapunov f(n)=log2(n)+w[n mod 2^L]", f"- L={L}. Feasible on sample: {feasible}.",
            ("- FEASIBLE: a residue-weighted function decreases on 100% of sampled transitions. CONJECTURE: there exist weights w on residue classes mod 2^L such that f(S(n))<f(n) for all odd n. This is a candidate invariant for rigorous proof / formalization. (Finite-sample feasibility does not prove the infinite case.)" if feasible else "- INFEASIBLE on the sample: no such weights exist even on the sampled transitions; an obstructing residue cycle is reported. This shows a residue-weighted-log Lyapunov of this form cannot work."),
            "", "## Honest status", "- This is conjecture generation + empirical falsification, the legitimate ML->proof handoff. It is NOT a proof of Collatz.",
            "- The full conjecture remains open; realistic outcomes from here are partial theorems (dense subsets, cycle bounds) or a formalized proof attempt of candidate C if feasible.", ""]
    (out / "CONJECTURES.md").write_text("\n".join(doc))
    print(f"outputs: {out}/CONJECTURES.md")

if __name__ == "__main__":
    main()
