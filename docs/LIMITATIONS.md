# Limitations

This project searches for empirical Collatz path structure. It does not prove
the Collatz conjecture.

- The 1.2-billion-row audit and GPU training inputs are too large to ship in the
  repository. A clean clone validates the software and evidence contract with
  deterministic fixtures; reproducing the largest experiments requires the
  separately generated, hash-pinned inputs and suitable hardware.
- Source alignment checks the project's parsing and trajectory conventions
  against public records. It is a correctness control, not independent proof of
  the conjecture.
- Held-out evaluation reduces memorization risk, but the learned labels are
  still derived from computed trajectory properties and do not establish a
  new mathematical invariant.
- The richer hybrid representation does not beat metrics-only features under
  the legacy family-pair control, and the v13 cluster-proxy representation
  underperforms raw metrics on the real fine-family target. These remain active
  promotion blockers rather than being hidden as unsuccessful experiments.
- CUDA scanning and GPU training are optional paths and are not exercised by
  the CPU-only continuous-integration workflow.
