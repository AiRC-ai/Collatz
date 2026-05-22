<!-- BEGIN GENERATED EVIDENCE SNAPSHOT -->
- Confidence: `range-stable signal`
- Meaning: The learned neighborhood signal survives current range and holdout checks, but it is not proof and is not yet source-neighborhood-supported.
- Audit: `200,000,000` rows over `1..200,000,000`; full audit completed: `true`.
- Coverage: topology `100,000` rows (`0.050%` of audit); stratified evidence sample `108,000` rows (`0.054%`).
- Neural result: `100,000` sample rows; GPU used: `true`; parallel jobs completed: `6`.
- Learned lift: weakest range `7.459%`, fold minimum `8.604%`, numeric-adjacency lift `5.341%`.
- Best current ablation: `metrics-only at 12.165%`.
- Interpretation: `metric-dominant signal`; metrics-only lift exceeds hybrid lift under the current evidence run.
- Healthy negative control: `metrics-only` beats `hybrid` (`12.165%` vs `9.968%`), so richer neural structure has not yet earned promotion beyond simpler trajectory metrics.
- Promotion blockers: `source_alignment_unknown_rows_present, source_alignment_unmatched_rows_present, metrics_only_exceeds_hybrid, richer_representation_not_stronger, matched_controls_incomplete, missing_lift_statistics`.
- Source alignment: `4 / 5` matched; unknown unmatched rows `1`.
- Next experiment: Classify unmatched source targets, expand non-OEIS source imports, then rerun source-neighborhood, path-image, GNN, and matched-control ablations.
- This is empirical evidence, not a Collatz proof.
<!-- END GENERATED EVIDENCE SNAPSHOT -->
