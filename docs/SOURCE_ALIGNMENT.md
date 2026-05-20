# Source Alignment

Source alignment checks public validation starts against the current topology
sample and optional binary feature rows. The row-level output is
`data/generated/source_alignment/unmatched_rows.csv`.

Every unmatched row receives exactly one reason bucket:

- `above_active_scan_range`
- `missing_from_topology_sample`
- `parser_error`
- `step_convention_mismatch`
- `peak_convention_mismatch`
- `true_mismatch`
- `missing_topology_node`
- `duplicated_source_row`
- `future_source_target`
- `unknown`

Any `true_mismatch` blocks promotion. Any `unknown` row blocks promotion until
it is explained or resolved.
