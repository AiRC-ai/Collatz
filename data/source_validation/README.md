# Source Validation Data

This directory is for small source-grounded samples and imported reference tables.

The first committed sample file is intentionally tiny. It gives the C++ validator
a stable smoke test while leaving the large reference imports outside git.

Large imported datasets should live under `data/imported/` or `data/generated/`
and remain uncommitted unless explicitly requested.
