# Source-Grounded Collatz Research Inputs

This project treats external sources in tiers. Authoritative record and
verification sources are used for correctness checks. Hypothesis and ML papers
are used for feature ideas, not proof.

## Validation Authorities

- Roosendaal, On The 3x + 1 Problem: https://www.ericr.nl/wondrous/
  - Use for delay records, path records, and residue/strength/level vocabulary.
- Oliveira e Silva, 3x+1 verification results: https://sweet.ua.pt/tos/3x%2B1.html
  - Use for verified stopping-time and maximum-excursion record files.
- Barina project: https://pcbarina.fit.vutbr.cz/
  - Use for current verification context and path-record references.
- OEIS A006577: https://oeis.org/A006577
  - Use for total stopping-time samples.
- OEIS A006884: https://oeis.org/A006884
  - Use for path-record and maximum-excursion samples.

## Implementation References

- Barina CUDA repository: https://github.com/xbarin02/collatz/
  - Study for GPU verification strategy. Reuse only license-compatible ideas.
- McDaMastR CollatzConjectureSimulator: https://github.com/McDaMastR/CollatzConjectureSimulator
  - Study as a separate GPU simulator reference. Do not copy GPL-style code
    unless licensing is explicitly handled.

## Feature And ML Hypothesis Sources

- Math StackExchange stopping-time curves:
  https://math.stackexchange.com/questions/4678861/collatz-stopping-time-curves
  - Use for residual and curve-family hypotheses, not correctness.
- MDPI clustering paper:
  https://www.mdpi.com/2227-7390/9/4/314
  - Use for sequence distance, clustering, and shape-analysis ideas.
- Kaggle Collatz sequences and metrics dataset:
  https://www.kaggle.com/datasets/clmentscipion/collatz-sequences-and-metrics-dataset
  - Use as convenience comparison data, not an authority.
- Barina paper:
  https://link.springer.com/article/10.1007/s11227-025-06961-0
  - Use for published GPU/CPU verification methodology and performance framing.

## Current V1 Source Validation

`collatz_validate_sources` reads `data/source_validation/reference_samples.csv`.
The first file is small by design: it proves that source validation is wired
into the build before large imports are added.

`collatz_source_targets` can build an expanded public target table from:

- `https://oeis.org/A006577/b006577.txt`
- `https://oeis.org/A006884/b006884.txt`
- `https://www.ericr.nl/wondrous/pathrecs.html`
- `https://www.ericr.nl/wondrous/delrecs.html`
- `https://pcbarina.fit.vutbr.cz/path-records.htm`
- Oliveira e Silva record files linked from `https://sweet.ua.pt/tos/3x%2B1.html`

The generated table keeps the compatibility header
`source,n,total_steps,peak_low`, followed by provenance columns:
`source_kind,source_rank,source_url,retrieved_utc,parser`.

The current canonical public state remains a `range-stable signal`, not a
source-neighborhood-supported candidate. The source gate reports matched and
unmatched rows plus a reason-bucket taxonomy. Unknown unmatched rows and any
true mismatch block confidence promotion. The next stronger gate requires
complete OEIS coverage plus at least two complete non-OEIS source families:
Roosendaal, Oliveira e Silva, and Barina.

Large imports should be named with source and retrieval date, for example:

- `data/imported/oeis_a006577_YYYY-MM-DD.csv`
- `data/imported/oliveira_t0_YYYY-MM-DD.txt`
- `data/imported/roosendaal_delay_records_YYYY-MM-DD.csv`
