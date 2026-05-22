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


def source_kind(source: str) -> str:
    if source.startswith("OEIS_A006577_"):
        return "total_stopping_time"
    if source.startswith("OEIS_A006884_"):
        return "path_record"
    if source.startswith("Roosendaal_"):
        # Known source kinds are encoded as "Roosendaal_path_record" and "Roosendaal_delay_record"
        # in the generator; keep the full suffix as the kind to preserve provenance.
        parts = source.split("_", 1)
        return parts[1] if len(parts) == 2 and parts[1] else "unknown"
    if source.startswith("Oliveira_e_Silva_"):
        # Known source kinds in this family also keep the full suffix.
        parts = source.split("_", 2)
        if len(parts) == 3 and parts[2]:
            return parts[2]
        return "unknown"
    if source.startswith("Barina_"):
        return source.removeprefix("Barina_") or "unknown"

    # Legacy fallback: keep best-effort suffix if present to avoid collapsing across known/unknown rows.
    if "_" in source:
        return source.rsplit("_", 1)[1]
    return "unknown"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_source_targets_dedupe.py TARGETS.csv METADATA.json")
    targets = Path(sys.argv[1])
    metadata = Path(sys.argv[2])
    seen: set[tuple[str, str, str]] = set()
    with targets.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            inferred_kind = row.get("source_kind") or source_kind(row["source"])
            key = (source_family(row["source"]), inferred_kind, row["n"])
            if key in seen:
                raise SystemExit(f"duplicate source target key survived dedupe: {key}")
            seen.add(key)

    meta = json.loads(metadata.read_text())
    dedupe_key = meta.get("dedupe_key")
    if dedupe_key is not None and dedupe_key != "source_family + source_kind + n":
        raise SystemExit("source-target metadata does not document the dedupe key")

    duplicate_report = meta.get("duplicates_output")
    if duplicate_report:
        duplicate_path = Path(str(duplicate_report))
        if duplicate_path.exists() is False:
            raise SystemExit("source-target duplicate report path does not exist")
    print("source target dedupe check passed")


if __name__ == "__main__":
    main()
