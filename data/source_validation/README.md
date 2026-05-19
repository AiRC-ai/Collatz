# Source Validation Data

This directory is for small source-grounded samples and imported reference tables.

The first committed sample file is intentionally tiny. It gives the C++ validator
a stable smoke test while leaving the large reference imports outside git.

`public_source_targets.csv` is the first expanded public target table. It is
derived from OEIS A006577 and OEIS A006884 b-files and scoped to starts
`<= 100000` so it can be checked against the current 100K embedded topology
window. Its metadata sidecar records source URLs, row counts, and skipped rows.

Large imported datasets should live under `data/imported/` or `data/generated/`
and remain uncommitted unless explicitly requested.
