#!/usr/bin/env python3
"""Check public source-target dedupe contract."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def source_family(source: str) -> str:
    if source.startswith("OEIS_"):
        return "oeis"
    if source.startswith("Roosendaal_"):
        return "roosendaal"
    if source.startswith("Oliveira_e_Silva_"):
        return "oliveira_e_silva"
    if source.startswith("Barina_"):
        return "barina"
    return source


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_source_targets_dedupe.py TARGETS.csv METADATA.json")
    targets = Path(sys.argv[1])
    metadata = Path(sys.argv[2])
    seen: set[tuple[str, str, str]] = set()
    with targets.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (source_family(row["source"]), row.get("source_kind", ""), row["n"])
            if key in seen:
                raise SystemExit(f"duplicate source target key survived dedupe: {key}")
            seen.add(key)

    meta = json.loads(metadata.read_text())
    if meta.get("dedupe_key") != "source_family + source_kind + n":
        raise SystemExit("source-target metadata does not document the dedupe key")
    duplicate_report = Path(str(meta.get("duplicates_output", "")))
    if not duplicate_report.exists():
        raise SystemExit("source-target duplicate report is missing")
    print("source target dedupe check passed")


if __name__ == "__main__":
    main()
