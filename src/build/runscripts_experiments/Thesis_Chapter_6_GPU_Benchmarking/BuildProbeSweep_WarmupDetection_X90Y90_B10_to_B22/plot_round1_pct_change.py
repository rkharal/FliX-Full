#!/usr/bin/env python3
"""Plot the round-1 hit-query warm-up effect (% change) vs. build/probe size.

Single figure, one line per structure (single-threaded queries vs. tiled
queries), reading round1_hitquery_warmup_detection_summary.csv directly
(never hardcodes the numbers).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUMMARY_CSV = HERE / "round1_hitquery_warmup_detection_summary.csv"

STRUCTURE_LABELS = {
    "TiledInsert_STQuery": "Single-Threaded Queries",
    "STInsert_TiledQuery": "Tiled Queries",
}
STRUCTURE_COLORS = {
    "TiledInsert_STQuery": "#1f77b4",
    "STInsert_TiledQuery": "#d62728",
}
STRUCTURE_MARKERS = {
    "TiledInsert_STQuery": "o",
    "STInsert_TiledQuery": "s",
}
NOTABLE_PCT_THRESHOLD = 5.0
LABEL_GAP_POINTS = 14


def load_summary() -> dict[str, list[tuple[int, int, float]]]:
    data: dict[str, list[tuple[int, int, float]]] = {}
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            structure = row["structure"]
            build_log2 = int(row["build_size_log"])
            probe_log2 = int(row["probe_size_log"])
            pct_change = float(row["pct_change"])
            data.setdefault(structure, []).append((build_log2, probe_log2, pct_change))
    for structure in data:
        data[structure].sort(key=lambda t: t[0])
    return data


def main() -> None:
    data = load_summary()
    structures = list(STRUCTURE_LABELS)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axhline(
        NOTABLE_PCT_THRESHOLD,
        color="black",
        linewidth=2.0,
        linestyle=(0, (3, 2)),
        alpha=0.85,
        label=f"{NOTABLE_PCT_THRESHOLD:.0f}% notability threshold",
    )

    # pct_change is negative (warm-up reduces time); plot as a positive
    # speedup percentage instead.
    series_y = {}
    x_labels = None
    for structure in structures:
        points = data[structure]
        x_labels = [f"$2^{{{b}}}$/$2^{{{p}}}$" for b, p, _ in points]
        y = [-pct for _, _, pct in points]
        series_y[structure] = y
        ax.plot(
            range(len(points)),
            y,
            marker=STRUCTURE_MARKERS[structure],
            markersize=11,
            linewidth=2.5,
            color=STRUCTURE_COLORS[structure],
            label=STRUCTURE_LABELS[structure],
        )

    # Place each point's label on the side away from the other series at that
    # same x, so labels never collide even where the two lines cross or
    # converge (fixed per-series offsets would collide depending on which
    # line happens to be on top at a given x).
    for xi in range(len(x_labels)):
        values = {structure: series_y[structure][xi] for structure in structures}
        higher, lower = sorted(structures, key=lambda s: values[s], reverse=True)
        offsets = {higher: LABEL_GAP_POINTS, lower: -LABEL_GAP_POINTS}
        for structure in structures:
            ax.annotate(
                f"{values[structure]:.1f}%",
                (xi, values[structure]),
                textcoords="offset points",
                xytext=(0, offsets[structure]),
                ha="center",
                fontsize=11,
                color=STRUCTURE_COLORS[structure],
            )

    ax.set_xticks(range(len(x_labels)), x_labels)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel("Build / Probe Size Pairs", fontsize=18, labelpad=14)
    ax.set_ylabel("Percentage Difference", fontsize=26, labelpad=14)
    ax.set_ylim(-8, 80)
    ax.set_title("Warm-up Cycle Effect on Query Performance", fontsize=20, pad=18)
    ax.legend(loc="upper right", frameon=True, fontsize=15)

    fig.tight_layout()
    stem = HERE / "query_warmup_pct_change_by_build_probe_size"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created {stem.with_suffix('.png')}")
    print(f"Created {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
