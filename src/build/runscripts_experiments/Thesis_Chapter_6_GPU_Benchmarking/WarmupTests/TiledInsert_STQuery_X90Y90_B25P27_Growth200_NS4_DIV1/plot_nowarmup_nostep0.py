#!/usr/bin/env python3
"""Plot the NoWarmup_NoStep0 experiment: tiled insert/delete, single-threaded
lookups, X90/Y90, B25/P27, no query-hit warm-up and no insert warm-up.

Shows update time (insert for R1-4, delete for R5-8), hit-query time, and
miss-query time round-by-round. Step 0 is excluded -- it does no meaningful
warm-up work in this configuration anyway.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABEL = "NoWarmup_NoStep0"
ROUNDS = list(range(1, 9))

METRIC_LABELS = {
    "insert_or_delete_time_ms": "Update Time (Insert/Delete)",
    "probe_time_ms": "Hit-Query Time",
    "probe_miss_time_ms": "Miss-Query Time",
}
METRIC_COLORS = {
    "insert_or_delete_time_ms": "#1f77b4",
    "probe_time_ms": "#2ca02c",
    "probe_miss_time_ms": "#d62728",
}


def find_csv() -> Path:
    matches = glob.glob(str(HERE / f"*_{LABEL}.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one CSV matching *_{LABEL}.csv; found {matches}")
    return Path(matches[0])


def read_round_data(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            step = int(row["step"])
            if step == 0:
                continue
            description = row["DESCRIPTION"]
            if description in METRIC_LABELS:
                rows.setdefault(step, {})[description] = float(row["VALUE"])

    if sorted(rows) != ROUNDS:
        raise RuntimeError(f"{path.name}: expected steps 1--8; found {sorted(rows)}")
    return rows


def main() -> None:
    csv_path = find_csv()
    data = read_round_data(csv_path)

    x = np.arange(len(ROUNDS), dtype=float)
    descriptions = list(METRIC_LABELS)
    total_width = 0.78
    bar_width = total_width / len(descriptions)

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)

    for index, description in enumerate(descriptions):
        offset = (index - (len(descriptions) - 1) / 2) * bar_width
        values = [data[r][description] for r in ROUNDS]
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=METRIC_COLORS[description],
            alpha=0.95,
            edgecolor="black",
            linewidth=0.7,
            label=METRIC_LABELS[description],
        )

    ax.axvline(3.5, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
    ax.set_xticks(x, [f"R{r}" for r in ROUNDS])
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.set_xlabel("Round (1-4 insert, 5-8 delete)", fontsize=24, labelpad=14)
    ax.set_ylabel("Time (ms)", fontsize=24, labelpad=14)
    ax.set_title(
        "No Query-Hit Warm-up, No Insert Warm-up\n"
        "(Tiled Insert/Delete, Single-Threaded Lookups, X90/Y90, B25/P27)",
        fontsize=26,
        pad=18,
    )
    ax.set_xlim(-0.6, len(ROUNDS) - 0.4)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True, fontsize=18)

    fig.tight_layout()
    stem = HERE / "TiledInsert_STQuery_X90Y90_nowarmup_nostep0"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created {stem.with_suffix('.png')}")
    print(f"Created {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
