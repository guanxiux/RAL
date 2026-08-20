#!/usr/bin/env python3
"""Render stream-level batch sensitivity from batch_sensitivity.csv."""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from bar_patterns import COLORS, EDGE, HATCHES, legend_handles


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "batch_sensitivity.csv"
OUT_PDF = HERE / "batch_sensitivity.pdf"

for directory in (
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive",
):
    for path in glob.glob(
        os.path.join(directory, "**", "texgyretermes-*.otf"),
        recursive=True,
    ):
        fm.fontManager.addfont(path)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "TeX Gyre Termes",
        "mathtext.it": "TeX Gyre Termes:italic",
        "mathtext.bf": "TeX Gyre Termes:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "hatch.linewidth": 0.36,
    }
)

SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
BATCH_SIZES = (1, 2, 4, 8)

# One acmart column at physical PDF points: 3.334 in * 72 pt/in = 240.048 pt.
FIG_W, FIG_H = 3.334, 2.00
FS_TICK, FS_AXIS, FS_LEG = 6.4, 7.1, 6.4


def load_rows() -> dict[tuple[str, int], dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    expected = {
        (system, batch_size)
        for system in SYSTEMS
        for batch_size in BATCH_SIZES
    }
    by_key = {}
    for row in rows:
        key = (row["system"], int(row["batch_size"]))
        if key in by_key:
            raise ValueError(f"duplicate row {key} in {CSV_PATH}")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError(f"unexpected rows in {CSV_PATH}")
    return by_key


def main() -> None:
    rows = load_rows()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0.155, 0.19, 0.825, 0.60])

    group_centers = {
        batch_size: float(index)
        for index, batch_size in enumerate(BATCH_SIZES)
    }
    within_step = 0.185
    bar_width = 0.145
    midpoint = (len(SYSTEMS) - 1) / 2.0
    for batch_size in BATCH_SIZES:
        center = group_centers[batch_size]
        for system_index, system in enumerate(SYSTEMS):
            row = rows[(system, batch_size)]
            throughput = float(row["robust_throughput_frames_per_second"])
            if throughput <= 0.0:
                raise ValueError(f"non-positive throughput for {system}/B{batch_size}")
            x = center + (system_index - midpoint) * within_step
            ax.bar(
                x,
                throughput,
                bar_width,
                color=COLORS[system],
                edgecolor=EDGE,
                hatch=HATCHES[system],
                linewidth=0.65,
                zorder=3,
            )

    half_group = midpoint * within_step + bar_width / 2.0
    ax.set_xlim(
        group_centers[BATCH_SIZES[0]] - half_group - 0.08,
        group_centers[BATCH_SIZES[-1]] + half_group + 0.08,
    )
    ax.set_ylim(0.0, 1125.0)
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_xticks([group_centers[value] for value in BATCH_SIZES])
    ax.set_xticklabels([str(value) for value in BATCH_SIZES], fontsize=FS_TICK)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.2, length=2.0, width=0.6)
    ax.set_xlabel("Batch size", fontsize=FS_AXIS, labelpad=2.2)
    ax.set_ylabel("Aggregate throughput (frames/s)", fontsize=FS_AXIS, labelpad=2.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#bcbcbc", alpha=0.28, linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles_by_label = dict(
        zip(
            [SYSTEM_LABELS[system] for system in SYSTEMS],
            legend_handles(SYSTEMS),
            strict=True,
        )
    )
    # Matplotlib fills multi-row legends column-first.  This order renders
    # Dense / Tile skipping on the first row and Gather-scatter / WISEConv on
    # the second row.
    labels = ["Dense", "Gather-scatter", "Tile skipping", "WISEConv"]
    legend = fig.legend(
        handles=[handles_by_label[label] for label in labels],
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=ax.transAxes,
        ncol=2,
        frameon=False,
        fontsize=FS_LEG,
        handlelength=1.0,
        handletextpad=0.32,
        columnspacing=1.0,
        labelspacing=0.28,
        borderaxespad=0.0,
        borderpad=0.0,
    )
    for label in legend.get_texts():
        if label.get_text() == "WISEConv":
            label.set_fontweight("bold")

    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
