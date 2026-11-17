#!/usr/bin/env python3
"""Plot the new build-key insertion warm-up comparison for steps 0--8."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
NO_WARMUP_CSV = next(HERE.glob("*_NoInsertWarmup_withstep0.csv"))
WARMUP_CSV = next(HERE.glob("*_Insert5000Warmup_withstep0.csv"))

COLORS = {
    "No insert warm-up": "#1f77b4",
    "5,000-key insert warm-up": "#17becf",
}
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
        raise RuntimeError("Expected complete measurements for steps 0 through 8.")

    def measured(data: dict[int, dict[str, float]], step: int, metric: str) -> float:
        return data[step][METRICS[metric]]

    x = np.arange(len(steps), dtype=float)
    width = 0.105
    centers = {"update": -0.25, "hit": 0.0, "miss": 0.25}
    offsets = {
        "No insert warm-up": -width / 2,
        "5,000-key insert warm-up": width / 2,
    }
    hatches = {"insert": "///", "delete": "\\", "hit": "", "miss": "xx"}

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)

    datasets = {
        "No insert warm-up": no_warmup,
        "5,000-key insert warm-up": warmup,
    }
    for label, data in datasets.items():
        for position, step in enumerate(steps):
            if step > 0:
                operation = "insert" if step <= 4 else "delete"
                ax.bar(
                    x[position] + centers["update"] + offsets[label],
                    measured(data, step, "update"),
                    width=width,
                    color=COLORS[label],
                    alpha=0.95,
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatches[operation],
                )
            for metric in ("hit", "miss"):
                ax.bar(
                    x[position] + centers[metric] + offsets[label],
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
    ax.set_xlim(-0.55, len(steps) - 0.45)
    ax.set_ylim(bottom=0)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS[label], edgecolor="black", label=label)
            for label in datasets
        ],
        loc="upper left",
        borderaxespad=0.35,
        frameon=True,
        fontsize=42,
        ncol=2,
    )

    fig.tight_layout(rect=(0.03, 0.03, 0.99, 0.99))
    stem = HERE / "X90Y90_NS5_Growth400_STInsert_TiledQuery_BuildKeyWarmup_comparison_22_23_withstep0"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    print(f"Created {stem.with_suffix('.png')}")
    print(f"Created {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
