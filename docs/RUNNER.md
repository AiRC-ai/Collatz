# Runner

The private-safe runner refreshes source targets, source alignment, full audit,
optional neural stages, canonical evidence, dashboard status, and a sanitized
ledger.

It does not auto-commit or auto-push public Git changes. Public README updates
remain explicit verified checkpoints.

When the neural stage is enabled, the current priority order is safe metrics,
family labels, matched hard-negative pairs, contrastive v2, retrieval
validation, then image/GNN evidence paths. This keeps the runner focused on
path-family learning rather than higher utilization alone.
