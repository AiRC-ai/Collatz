# Source Alignment

Source alignment checks public validation starts against the current topology
sample and optional binary feature rows. The row-level output is
`data/generated/source_alignment/unmatched_rows.csv`.

The source-target builder dedupes public records by `source_family +
source_kind + n` before alignment and writes duplicate provenance beside the
target CSV. The stratified/topology flow should include the deduped public
source starts so in-range source records have projection rows before alignment
runs.

Every unmatched row receives exactly one reason bucket:

- `above_active_scan_range`
- `missing_from_topology_sample`
- `missing_feature_row`
- `parser_error`
- `step_convention_mismatch`
- `peak_convention_mismatch`
- `true_mismatch`
- `missing_topology_projection_node`
- `duplicated_source_row`
- `future_source_target`
- `unknown`

Any `true_mismatch` blocks promotion. Any `unknown` row blocks promotion until
it is explained or resolved.
