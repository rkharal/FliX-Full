#!/usr/bin/env python3
"""Compare warm-up modes for single-threaded FliX inserts and queries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
NO_WARMUP_CSV = next(HERE.glob("*_FliX_STInsert_STQuery_NoWarmups.csv"))
WARMUP_CSV = next(HERE.glob("*_FliX_STInsert_STQuery_Warmups.csv"))

COLORS = {"No warm-up": "#1f77b4", "Warm-up": "#17becf"}
METRICS = {
    "update": "insert_or_delete_time_ms",
    "hit": "probe_time_ms",
    "miss": "probe_miss_time_ms",
}


def read_measurements(path: Path) -> dict[int, dict[str, float]]:
    measurements: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            if row["DESCRIPTION"] not in METRICS.values():
                continue
            step = int(row["step"])
            measurements.setdefault(step, {})[row["DESCRIPTION"]] = float(row["VALUE"])
    return measurements


def main() -> None:
    no_warmup = read_measurements(NO_WARMUP_CSV)
    warmup = read_measurements(WARMUP_CSV)
    steps = list(range(9))
    if sorted(no_warmup) != steps or sorted(warmup) != steps:
        raise RuntimeError("Expected complete measurements for rounds 0 through 8.")

    def measured(data: dict[int, dict[str, float]], step: int, metric: str) -> float:
        return data[step][METRICS[metric]]

    x = np.arange(9, dtype=float)
    width = 0.105
    centers = {"update": -0.25, "hit": 0.0, "miss": 0.25}
    offsets = {"No warm-up": -width / 2, "Warm-up": width / 2}
    hatches = {"insert": "///", "delete": "\\\\", "hit": "", "miss": "xx"}

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)

    for label, data in {"No warm-up": no_warmup, "Warm-up": warmup}.items():
        for step in range(1, 9):
            operation = "insert" if step <= 4 else "delete"
            ax.bar(
                x[step] + centers["update"] + offsets[label],
                measured(data, step, "update"),
                width=width,
                color=COLORS[label],
                alpha=0.95,
                edgecolor="black",
                linewidth=0.8,
                hatch=hatches[operation],
            )

        for metric in ("hit", "miss"):
            for step in steps:
                ax.bar(
                    x[step] + centers[metric] + offsets[label],
                    measured(data, step, metric),
                    width=width,
                    color=COLORS[label],
                    alpha=0.95,
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatches[metric],
                )

    labels = ["Round 0\nQuery only"]
    labels.extend(f"Round {step}\nInsert" for step in range(1, 5))
    labels.extend(f"Round {step}\nDelete" for step in range(5, 9))
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=36)
    ax.set_xlabel(
        "Experiment round (bar pairs: update, hit query, miss query)",
        fontsize=41,
        labelpad=20,
    )
    ax.set_ylabel("Time (ms)", fontsize=76, labelpad=22)
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(bottom=0)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["No warm-up"], edgecolor="black", label="No warm-up"),
            Patch(facecolor=COLORS["Warm-up"], edgecolor="black", label="Warm-up"),
        ],
        loc="upper left",
        borderaxespad=0.35,
        frameon=True,
        fontsize=51,
        ncol=2,
    )

    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.99))
    stem = HERE / "st_inserts_st_queries_warmup_vs_no_warmup_all_operations_linear"
    fig.savefig(
        stem.with_suffix(".png"), dpi=300, bbox_inches="tight",
        facecolor="white", transparent=False,
    )
    fig.savefig(
        stem.with_suffix(".pdf"), bbox_inches="tight",
        facecolor="white", transparent=False,
    )
    plt.close(fig)
    print(f"Created {stem.with_suffix('.png')}")
    print(f"Created {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
