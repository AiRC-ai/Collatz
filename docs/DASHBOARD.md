# Dashboard

The dashboard API is split into:

- `evidence`: canonical public research evidence from
  `latest_public_summary.json`.
- `operations`: runner, scanner, CPU, GPU, neural-job, topology-preview, and
  GNN-preview telemetry.

The dashboard may show live GPU utilization, memory, power, CPU throughput, and
runner state, but those fields are operational only. They do not affect the
confidence label.
