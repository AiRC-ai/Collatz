# True Iteration Ledger

The project uses the user-global `true-iteration-until-done` workflow for long
research cycles.

Each stable cycle should record:

- mission slice
- source versions checked
- scan range
- binary or dataset version
- CPU throughput
- CUDA throughput when available
- validation result
- model metric result when training is active
- hypothesis confidence level
- strongest evidence
- weakest limitation
- next experiment

The private evidence runner appends sanitized entries to
`logs/iteration-ledger.jsonl`. The web dashboard does not expose raw ledger
lines. It reads public-safe automation state from
`data/generated/runner/status.json`.

The AI conclusion dashboard reads generated research claims from
`data/generated/hypotheses/summary.json` and detailed entries from
`data/generated/hypotheses/hypotheses.jsonl`.

The background evidence cycle lives in `ops/collatz-evidence-cycle.sh`.
It is idempotent, lock-protected, and intended for a user-level timer. It
refreshes public source imports, source-neighborhood alignment, hypotheses, and
runner status without committing or pushing public Git changes.

Example line:

```json
{"timestamp":"2026-05-19T12:00:00Z","slice":"cpu-scan-smoke","range":"1..1000000","result":"passed","next":"cuda parity check"}
```
