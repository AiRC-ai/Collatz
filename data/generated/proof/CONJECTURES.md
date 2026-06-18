# Collatz candidate invariants (empirical, NOT a proof)

Searched closed-form Lyapunov candidates for the accelerated map S(n)=(3n+1)/2^v2(3n+1) on all odd n in 3..20000000.

## A. 1-step S(n) < n
- holds on 50.0000% of 9999999 odd n. Fails 5000000 times.
- Known: fails for n = 3 mod 4 (S(n) = (3n+1)/2 > n). NOT an invariant.

## B. k-step S^k(n) < n
- k=1: 50.0000%
- k=2: 50.0000%
- k=3: 68.7500%
- k=4: 65.6250%
- k=5: 77.3437%
- k=6: 74.6093%
- k=7: 72.5586%
- k=8: 80.6151%
- No fixed k reaches 100% on the sample; a fixed-k global decrease would be a Lyapunov function (essentially a proof). None found.

## C. residue-weighted log Lyapunov f(n)=log2(n)+w[n mod 2^L]
- L=8. Feasible on sample: False.
- INFEASIBLE on the sample: no such weights exist even on the sampled transitions; an obstructing residue cycle is reported. This shows a residue-weighted-log Lyapunov of this form cannot work.

## Honest status
- This is conjecture generation + empirical falsification, the legitimate ML->proof handoff. It is NOT a proof of Collatz.
- The full conjecture remains open; realistic outcomes from here are partial theorems (dense subsets, cycle bounds) or a formalized proof attempt of candidate C if feasible.
