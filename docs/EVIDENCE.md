# Evidence Contract

`data/generated/evidence/latest_public_summary.json` is the only source of
truth for public research claims. README text, dashboard evidence cards,
hypothesis wording, confidence labels, and source-alignment interpretation must
be generated from that file.

Confidence levels are conservative:

- `sample-local signal`: a model beats a random baseline on one sample.
- `range-stable signal`: full audit, range holdouts, fold checks, random
  baseline, and numeric-adjacency baseline are reported and positive.
- `source-neighborhood-supported`: public source targets agree with learned
  neighborhoods, unmatched rows are classified, and at least two non-OEIS source
  families are complete.
- `candidate pattern`: richer representations beat metrics-only under matched
  controls across holdouts, seeds, and ablations.
- `proof`: unavailable unless a formal independently checkable proof artifact
  exists.

Operational telemetry never raises scientific confidence.

`confidence.promotion_blockers` is machine-readable gate state. Public wording
should explain these blockers instead of inferring confidence from separate
README text, dashboard telemetry, or private runner status.
