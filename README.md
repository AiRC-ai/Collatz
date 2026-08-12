# Collatz Research System

[![CPU build and tests](https://github.com/AiRC-ai/Collatz/actions/workflows/cpu-ci.yml/badge.svg)](https://github.com/AiRC-ai/Collatz/actions/workflows/cpu-ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](https://en.cppreference.com/w/cpp/compiler_support/20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible C++20 research platform for generating Collatz trajectory
features, validating public record data, testing learned representations, and
falsifying candidate invariants. The default build is CPU-only; CUDA scanning
and GPU research workflows are optional.

> This is computational research, not a proof of the Collatz conjecture.

![Collatz path plot for n=27](collatz_27.svg)

## What it includes

- A reusable C++20 core plus CPU and optional CUDA scanners.
- Deterministic source-validation, evidence-publication, and visualization
  tools with a CTest fixture suite.
- Supervised and few-shot representation experiments with held-out evaluation,
  pinned input hashes, matched baselines, and recorded negative results.
- An invariant-falsification engine that rejects unsuccessful proof candidates
  and reports the counterexamples.

## Current findings

| Question | Result | Evaluation |
|---|---:|---|
| Fine-family retrieval | raw metrics `+25.1%`; direct prototypical model `+82.3%` | Family-disjoint test families, 3 seeds |
| Coarse range retrieval | raw `+12.0%`; supervised embedding `+84.8%` | Held-out rows |
| Coarse bit-length retrieval | raw `+25.1%`; supervised embedding `+70.9%` | Held-out rows |
| Coarse peak-ratio retrieval | raw `+50.4%`; supervised embedding `+83.3%` | Held-out rows |
| Cluster-proxy control | raw `+24.7%`; learned proxy `+21.0%` | Honest negative; proxy does not beat raw |

The direct fine-family model generalizes to unseen families. The re-clustered
proxy does not improve the real target, so it is retained as a diagnostic—not
promoted as a win. See the [held-out v12 report](docs/NEURAL_ENGINE_V12.md),
[v13 negative-control report](docs/NEURAL_ENGINE_V13.md), and
[proof-candidate work](docs/PROOF_WORK.md).

The canonical evidence status is `source-neighborhood-supported`, not a
candidate pattern. Promotion remains blocked because a legacy hybrid feature
set trails metrics-only features and some historical ablations lack complete
matched-control statistics.

## Quick start

Requirements: CMake 3.20 or newer, a C++20 compiler, and Python 3.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Generate a trajectory plot or inspect a sample from the command line:

```sh
./build/collatz_plot 27
./build/collatzctl sample 27
```

## Main components

| Area | Entry points |
|---|---|
| Core and scanning | `include/collatz/`, `collatz_scan_cpu`, optional `collatz_scan_cuda` |
| Evidence and validation | `collatz_validate_sources`, `collatz_evidence_publish`, `schemas/` |
| Representation research | `collatz_embed_export`, `research/`, `docs/NEURAL_PIPELINE.md` |
| Candidate falsification | `research/invariant_falsifier.py`, `docs/PROOF_WORK.md` |
| Operations and visualization | `collatzctl`, `collatz_web`, `ops/`, `compose.yaml` |

## Reproducibility and documentation

The clean-clone CPU path builds and tests without the large private runtime
artifacts. Large-scale experiments record their data provenance and SHA-256
input hashes in the public evidence files.

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Build notes](docs/BUILD.md)
- [Evidence contract](docs/EVIDENCE.md)
- [Source grounding](docs/SOURCES.md)
- [Known limitations](docs/LIMITATIONS.md)
- [Full research record and operating guide](docs/RESEARCH_RECORD.md)

## Project policy

This project is available under the [MIT License](LICENSE). Please use
[CITATION.cff](CITATION.cff) when citing the software, and follow the
[security policy](SECURITY.md) when reporting a vulnerability.
