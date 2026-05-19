# Benchmarks

Benchmarks are small reproducibility notes, not final claims. Generated CSV and
JSONL benchmark artifacts are ignored by git.

## CPU Scanner Smoke

Date: 2026-05-19

Range: `1..1,000,000`

Command shape:

```sh
./build/collatz_scan_cpu \
  --start 1 \
  --end 1000000 \
  --output data/generated/bench_seq_<threads>t.csv \
  --progress logs/bench_seq_<threads>t.jsonl \
  --chunk-size 100000 \
  --threads <threads>
```

Results:

| Threads | Real Time | Final Throughput |
|---------|-----------|------------------|
| 1 | 5.16 s | 193,871 starts/s |
| 16 | 0.65 s | 1,527,500 starts/s |

The 1-thread and 16-thread CSV files had the same SHA-256:

```text
80f0a955091068c4a6d93940299ee6a6fdb125ab17f525ebd6aaeb531511c1de
```

The 16-thread output was also checked for ordered rows through `n=1,000,000`.
