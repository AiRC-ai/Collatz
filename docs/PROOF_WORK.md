# Toward a proof: candidate-invariant search (and an honest negative)

## What this is, and what it is not

This is **not** a proof of the Collatz conjecture. It is the legitimate
ML-to-proof handoff step: a bounded, closed-form search for a candidate
Lyapunov (decrease) function for the accelerated Collatz map, followed by
rigorous empirical + linear-feasibility falsification on all odd n up to 2e7.
The output is either a precisely-stated conjecture for a mathematician to prove
(and formalize in Lean/Coq), or -- as happened here -- an explicit obstruction
showing why the natural simple candidates cannot work.

A proof of Collatz requires showing every trajectory reaches 1 (no divergent
trajectories, no non-trivial cycles). Empirical verification on any finite range
is not a proof.

## Setup

Accelerated (Syracuse) map on odd n: `S(n) = (3n+1) / 2^v2(3n+1)` (odd -> odd).
A Lyapunov function would be a state-level function f with `f(S(n)) < f(n)` for
all odd n (except the trivial cycle 1->1). `research/invariant_search.py` tests
three closed-form families on all odd n in 3..2e7:

- **A. 1-step:** `S(n) < n`.
- **B. k-step:** `S^k(n) < n` for fixed k = 1..8 (a fixed k that holds for all n
  would itself be a Lyapunov function -> essentially a proof).
- **C. residue-weighted log:** `f(n) = log2(n) + w[n mod 2^L]`. The decrease
  condition `f(S(n)) < f(n)` is a system of linear difference constraints in the
  weights w. Reducing to `w[r] - w[r'] > D[r][r']` (D = max log2(S(n)/n) over
  sampled transitions r -> r'), feasibility is checked exactly via Bellman-Ford
  negative-cycle detection on the -D graph -- no LP solver needed. Feasible =>
  a residue-weighted function decreases on 100% of sampled transitions
  (candidate invariant). Infeasible => an obstructing residue cycle is reported.

## Results (odd n in 3..20,000,000)

**A. 1-step `S(n) < n`: holds on 50.0000%** (fails exactly 5,000,000 of
9,999,999 odd n). Failing residues mod 256: 3, 7, 11, 15 -- i.e. all
n = 3 mod 4, where `S(n) = (3n+1)/2 > n`. Not an invariant.

**B. k-step `S^k(n) < n` (or reaches 1):**

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| % | 50.0 | 50.0 | 68.75 | 65.63 | 77.34 | 74.61 | 72.56 | 80.62 |

The rate climbs (and oscillates) with k but never reaches 100%. No fixed-k
global decrease exists on the sample -- so no simple fixed-step Lyapunov
function.

**C. residue-weighted log Lyapunov: INFEASIBLE**, at both L=8 (256 classes,
1,024 binding pair-constraints) and L=12 (4,096 classes, 22,615 constraints).
Obstructing edge in both: residue class **3 -> 5**, requiring a weight drop of
log2 ratio 0.737 that participates in a non-negative residue cycle. No choice of
bounded residue weights makes `log2(n) + w[n mod 2^L]` decrease on every
sampled transition -- not even on the finite sample.

## The obstruction, in words

For n = 3 mod 4, `3n+1 = 2 (mod 4)`, so `v2(3n+1) = 1` and `S(n) = (3n+1)/2`,
which is ~1.5 n -- a forced increase of ~log2(1.5). A residue-weighted log can
only compensate by reweighting residue classes, but the residue-transition
graph contains a cycle (through 3 -> 5 -> ...) whose required compensations sum
to >= 0, so no bounded weighting can make every step decrease. This is the
structural reason the simplest Lyapunov ansatz fails, and it is exactly where a
real proof has to do something harder than a bounded-residue weighting.

## What a proof would still need

The natural simple candidates are falsified above. A genuine proof would likely
require one of:
- an invariant depending on **more than a bounded residue window** (e.g. on the
  full parity prefix / 2-adic structure, or a non-local function of n);
- a **density-1-but-not-all** decrease (Terras/Korec style) **plus** a separate
  argument excluding the density-0 exceptional set -- which is the actual open
  part;
- **cycle exclusion + divergence exclusion** via deep number theory (non-trivial
  cycles are already known to be enormous; excluding divergence is the hard
  half); or
- a structurally-motivated **partial theorem** (termination for a proven-dense
  subset defined by an exact number-theoretic condition) -- a realistic,
  publishable outcome that is not the full conjecture.

Once a candidate invariant survives, the next step is a hand proof plus
machine-checked formalization (Lean/Coq). This script's job is to find or
falsify candidates; it does not replace the proof.

## Reproduce

```bash
python3 research/invariant_search.py --n 20000000 --kmax 8 --l 8 --output-dir data/generated/proof
python3 research/invariant_search.py --n 5000000  --kmax 4 --l 12 --output-dir /tmp/proof_l12
```

Outputs: `data/generated/proof/invariant_metrics.json`, `CONJECTURES.md`, `run.log`.

This is empirical evidence and a falsification of simple candidate invariants,
not a Collatz proof.

## Falsification of richer candidate classes (reusable engine)

`research/invariant_falsifier.py` is the reusable engine: give it any candidate
Lyapunov function f(n) and it reports the survival fraction (f(S(n))<f(n)) on all
odd n up to N, the first failure, and the failing-residue pattern, plus the exact
residue-weighted feasibility check at configurable L. Run at N=2e7 odd n / L=12
against the richer classes the obstruction pointed at (halving-count and
multi-step coupling -- i.e. using more than a bounded residue window):

| candidate | survival | first failure |
|---|---:|---:|
| log2(n) | 50.0% | n=3 |
| log2(n)+c*v2(3n+1), best c=0.5 | 43.3% | n=3 |
| log2(n)+c1*v2(3n+1)+c2*v2(3S(n)+1), best (1.0,1.5) | 45.6% | n=3 |
| log2(n)-log2(3)*[n=3 mod 4] (explicit exception) | 50.0% | n=3 |
| residue-weighted log, L=12 | INFEASIBLE | cycle 3->5 |

**Every richer candidate is falsified -- most already at n=3** (S(3)=5 > 3, and
the halving-count terms are anti-correlated with the needed direction, so they
make it *worse*, not better). The residue-weighted ansatz stays infeasible at
L=12 with the same 3->5 obstruction.

**Interpretation.** The obstruction is robust: compensating the `n = 3 mod 4`
increase with halving-count info or a bounded residue weighting does not work --
the v2 sequence is aligned the wrong way, and the residue cycle 3->5 forces a
non-negative compensation sum. This formally closes the "find a better
bounded-residue / halving-count weighting" line. A surviving invariant, if one
exists, must be non-local in a stronger sense (e.g. depend on the full parity
prefix as an unbounded object, or on magnitude in a non-monotone closed form) --
and finding such a closed form is exactly the open problem.

**The durable asset is the engine.** It is not a one-shot result: any future
candidate invariant can be tested with `python3 research/invariant_falsifier.py`
(add the callable to the candidates list). Its job is to kill bad conjectures
fast so effort concentrates on survivors. To date it has killed every closed-form
candidate thrown at it, which is the honest expected outcome for Collatz.
