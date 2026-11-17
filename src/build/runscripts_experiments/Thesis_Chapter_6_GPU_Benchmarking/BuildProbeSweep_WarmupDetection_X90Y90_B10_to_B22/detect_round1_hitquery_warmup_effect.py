#!/usr/bin/env python3
"""Detect whether round-1 hit-query time is affected by warm-up.

Reads every (structure, build/probe) pair of CSVs produced by
BuildProbeSweep_WarmupDetection_X90Y90_B10_to_B22.sh and compares round-1
(step=1) probe_time_ms between the NoWarmup_NoStep0 and WithStep0_Warmups
configs. No plots -- prints a table and writes the same table to
round1_hitquery_warmup_detection_summary.csv.
"""

from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

STRUCTURES = ["TiledInsert_STQuery", "STInsert_TiledQuery"]
WARMUPS = ["NoWarmup_NoStep0", "WithStep0_Warmups"]
NOTABLE_PCT_THRESHOLD = 5.0  # flag anything with |pct change| above this


def read_round1_probe_time(path: Path) -> float:
    """Average probe_time_ms (step=1) across every run present in the CSV."""
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            if row["DESCRIPTION"] != "probe_time_ms":
                continue
            if int(row["step"]) != 1:
                continue
            values.append(float(row["VALUE"]))
    if not values:
        raise RuntimeError(f"{path.name}: no step=1 probe_time_ms row found")
    return sum(values) / len(values)


def find_csv(structure: str, warmup: str, build_log2: int, probe_log2: int) -> Path:
    pattern = f"*_B{build_log2}_P{probe_log2}_*_FliX_{structure}_{warmup}.csv"
    matches = list(HERE.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one CSV for pattern {pattern}; found {matches}")
    return matches[0]


def build_probe_pairs() -> list[tuple[int, int]]:
    pairs = set()
    for path in HERE.glob("*.csv"):
        match = re.search(r"_B(\d+)_P(\d+)_", path.name)
        if match:
            pairs.add((int(match.group(1)), int(match.group(2))))
    return sorted(pairs)


def main() -> None:
    pairs = build_probe_pairs()
    if not pairs:
        raise RuntimeError(f"No CSVs found in {HERE}")

    rows = []
    for structure in STRUCTURES:
        for build_log2, probe_log2 in pairs:
            no_warmup_path = find_csv(structure, "NoWarmup_NoStep0", build_log2, probe_log2)
            with_warmup_path = find_csv(structure, "WithStep0_Warmups", build_log2, probe_log2)

            no_warmup_val = read_round1_probe_time(no_warmup_path)
            with_warmup_val = read_round1_probe_time(with_warmup_path)

            delta = with_warmup_val - no_warmup_val
            pct = (delta / no_warmup_val * 100.0) if no_warmup_val != 0 else float("inf")
            notable = "YES" if abs(pct) > NOTABLE_PCT_THRESHOLD else "no"

            rows.append(
                {
                    "structure": structure,
                    "build_size_log": build_log2,
                    "probe_size_log": probe_log2,
                    "round1_hitquery_no_warmup_ms": round(no_warmup_val, 4),
                    "round1_hitquery_with_warmup_ms": round(with_warmup_val, 4),
                    "delta_ms": round(delta, 4),
                    "pct_change": round(pct, 2),
                    "notable_gt_5pct": notable,
                }
            )

    header = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in header}

    def fmt_row(values: dict) -> str:
        return "  ".join(str(values[h]).ljust(col_widths[h]) for h in header)

    print(fmt_row({h: h for h in header}))
    print("-" * (sum(col_widths.values()) + 2 * (len(header) - 1)))
    for row in rows:
        print(fmt_row(row))

    notable_count = sum(1 for r in rows if r["notable_gt_5pct"] == "YES")
    print()
    print(
        f"Verdict: {notable_count}/{len(rows)} (structure, build/probe) configs show a "
        f">{NOTABLE_PCT_THRESHOLD}% change in round-1 hit-query time between "
        f"NoWarmup_NoStep0 and WithStep0_Warmups."
    )

    summary_path = HERE / "round1_hitquery_warmup_detection_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
