#!/usr/bin/env python3
"""Create separate update and query plots for each update schedule.

Every figure includes FliX, LSMu, HT-Slab, and GPU B-tree. Experiment step 0
is excluded; the plotted rounds are the four insertion/deletion pairs defined
by the successive or alternating schedule.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent

STRUCTURES_ALL = ["FliX", "LSMu", "HashSlab", "GPUBTree"]
STRUCTURE_LABELS = {
    "FliX": "FliX",
    "LSMu": "LSMu",
    "HashSlab": "HT-Slab",
    "GPUBTree": "GPU B-tree",
}
STRUCTURE_COLORS = {
    "FliX": "#1f77b4",
    "LSMu": "#2ca02c",
    "HashSlab": "#8c564b",
    "GPUBTree": "#ff7f0e",
}
ROUNDS = list(range(1, 9))  # 4 inserts + 4 deletes (or alternating), step 0 excluded.

METRICS = {
    "query": ("probe_time_ms", "Query Time"),
    "update": ("insert_or_delete_time_ms", "Update Time (Insert/Delete)"),
}


def read_round_data(path: Path) -> dict[int, dict[str, float | bool]]:
    """Returns {step: {"insert_or_delete_time_ms": v, "probe_time_ms": v, "do_insert": bool}}."""
    rows: dict[int, dict[str, float | bool]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for raw_row in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw_row.items()}
            step = int(row["step"])
            if step == 0:
                continue
            entry = rows.setdefault(step, {"do_insert": row["do_insert"].strip() == "1"})
            description = row["DESCRIPTION"]
            if description in ("insert_or_delete_time_ms", "probe_time_ms"):
                entry[description] = float(row["VALUE"])

    if sorted(rows) != ROUNDS:
        raise RuntimeError(f"{path.name}: expected steps 1--8; found {sorted(rows)}")
    return rows


def load_schedule(
    schedule: str,
) -> dict[str, dict[int, dict[str, float | bool]]]:
    data: dict[str, dict[int, dict[str, float | bool]]] = {}
    for structure in STRUCTURES_ALL:
        matches = list(HERE.glob(f"*_{schedule}+{{{structure}}}.csv"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one CSV for schedule={schedule} structure={structure}; "
                f"found {len(matches)}"
            )
        data[structure] = read_round_data(matches[0])
    return data


def round_type_spans(
    data: dict[str, dict[int, dict[str, float | bool]]],
) -> list[tuple[int, bool]]:
    """do_insert per round, taken from the first structure (identical across structures)."""
    reference = data[STRUCTURES_ALL[0]]
    return [(round_number, bool(reference[round_number]["do_insert"])) for round_number in ROUNDS]


def shade_round_types(ax, spans: list[tuple[int, bool]]) -> None:
    insert_color = "#cfe8ff"
    delete_color = "#ffd9d9"
    for round_number, is_insert in spans:
        x = round_number - 1  # ROUNDS is 1-indexed, bar positions are 0-indexed
        ax.axvspan(
            x - 0.5,
            x + 0.5,
            color=insert_color if is_insert else delete_color,
            alpha=0.5,
            zorder=0,
            linewidth=0,
        )


def numeric_log_ticks(lower: float, upper: float) -> list[float]:
    """Return readable 1/2/5 ticks and label every displayed log-scale tick."""
    first_exponent = math.floor(math.log10(lower)) - 1
    last_exponent = math.ceil(math.log10(upper)) + 1
    ticks: list[float] = []
    for exponent in range(first_exponent, last_exponent + 1):
        decade = 10.0 ** exponent
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * decade
            if lower <= value <= upper:
                ticks.append(value)
    return ticks


def format_numeric_tick(value: float, _position: int) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def create_metric_figure(schedule: str, metric_key: str) -> None:
    description, title = METRICS[metric_key]
    data = load_schedule(schedule)
    spans = round_type_spans(data)

    x = np.arange(len(ROUNDS), dtype=float)
    total_width = 0.78
    bar_width = total_width / len(STRUCTURES_ALL)

    fig, ax = plt.subplots(figsize=(40, 24))
    shade_round_types(ax, spans)
    ax.set_axisbelow(True)

    all_values: list[float] = []
    clipped_bars: list[tuple[plt.Rectangle, float]] = []
    for index, structure in enumerate(STRUCTURES_ALL):
        offset = (index - (len(STRUCTURES_ALL) - 1) / 2) * bar_width
        values = [float(data[structure][round_number][description]) for round_number in ROUNDS]
        all_values.extend(values)
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=STRUCTURE_COLORS[structure],
            alpha=0.95,
            edgecolor="black",
            linewidth=0.8,
            label=STRUCTURE_LABELS[structure],
            zorder=3,
        )
        clipped_bars.extend(zip(bars, values))

    lower = min(all_values) / 1.4
    upper = 45.0
    ax.set_yscale("log")
    ax.set_ylim(lower, upper)
    y_ticks = numeric_log_ticks(lower, upper)
    if upper not in y_ticks:
        y_ticks.append(upper)
    ax.set_yticks(sorted(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(format_numeric_tick))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", which="major", linestyle="--", linewidth=1.0, alpha=0.28)
    ax.grid(axis="x", visible=False)

    operation_labels = [
        "Insert/\nQuery" if is_insert else "Delete/\nQuery"
        for _, is_insert in spans
    ]
    ax.set_xticks(
        x,
        [f"R{round_number}\n{operation}" for round_number, operation in zip(ROUNDS, operation_labels)],
    )
    ax.tick_params(axis="x", labelsize=51, width=1.4, length=7)
    ax.tick_params(axis="y", labelsize=53, width=1.4, length=7)
    ax.set_title(title, fontsize=77, pad=22)
    ax.set_xlabel("Rounds", fontsize=65, labelpad=18)
    ax.set_ylabel("Time (ms)", fontsize=103, labelpad=18)
    ax.set_xlim(-0.6, len(ROUNDS) - 0.4)
    for bar, value in clipped_bars:
        if value <= upper:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            upper / 1.035,
            f"{value:.0f}",
            ha="center",
            va="top",
            rotation=90,
            fontsize=40,
            fontweight="bold",
            color="#202020",
            clip_on=True,
            zorder=5,
        )
    ax.legend(
        loc="lower center",
        ncol=len(STRUCTURES_ALL),
        frameon=True,
        fontsize=68,
        bbox_to_anchor=(0.5, 1.08),
        bbox_transform=ax.transAxes,
    )

    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.99))

    stem = HERE / f"{schedule.lower()}_{metric_key}_time_legend_above_title_large_fonts"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created {stem.with_suffix('.png')}")
    print(f"Created {stem.with_suffix('.pdf')}")


def main() -> None:
    for schedule in ("Successive", "Alternating"):
        create_metric_figure(schedule, "query")
        create_metric_figure(schedule, "update")


if __name__ == "__main__":
    main()
