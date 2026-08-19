#!/usr/bin/env python3
"""Render the frame-equal effective-throughput stacked-bar figure from CSV."""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from bar_patterns import COLORS, EDGE, HATCHES, legend_handles


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "effective_throughput.csv"
OUT_PDF = HERE / "effective_throughput.pdf"

for directory in (
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive",
):
    for path in glob.glob(
        os.path.join(directory, "**", "texgyretermes-*.otf"), recursive=True
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


SYSTEMS = ["dense", "tile_skip", "gather_scatter", "wiseconv"]
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
WORKLOADS = ["fireflownet", "yolov8n", "yolov8m", "dynconv_pose"]
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}

# The source canvas is slightly wider than a column because tight cropping
# removes the unused exterior.  The resulting PDF is one acmart column wide.
FIG_W, FIG_H = 3.552, 2.02
FS_TICK, FS_AXIS, FS_WORKLOAD, FS_LEG = 6.3, 7.1, 6.7, 6.5

# A neutral style shared by every backend's non-useful issued-work segment.
WASTED_COLOR = "#e3e3e3"
WASTED_HATCH = "...."


def read_rows() -> dict[tuple[str, str], dict]:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    expected = len(WORKLOADS) * len(SYSTEMS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    by_key = {}
    for row in rows:
        key = (row["workload"], row["system"])
        if key in by_key:
            raise ValueError(f"duplicate row {key}")
        by_key[key] = row
    required = {(workload, system) for workload in WORKLOADS for system in SYSTEMS}
    if set(by_key) != required:
        raise ValueError("effective-throughput CSV has unexpected keys")
    return by_key


def relative_tick(value: float, _: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    rows = read_rows()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0.145, 0.195, 0.84, 0.61])

    group_step = 1.08
    within_step = 0.185
    bar_width = 0.148
    if within_step <= bar_width:
        raise ValueError("bars must retain a visible inter-bar gap")
    group_centers = {
        workload: index * group_step for index, workload in enumerate(WORKLOADS)
    }
    group_midpoint = (len(SYSTEMS) - 1) / 2.0

    for workload in WORKLOADS:
        center = group_centers[workload]
        for system_index, system in enumerate(SYSTEMS):
            row = rows[(workload, system)]
            effective = float(row["effective_throughput_vs_dense"])
            wasted = float(row["wasted_throughput_vs_dense"])
            raw = float(row["raw_throughput_vs_dense"])
            if effective < 0.0 or wasted < -1e-12:
                raise ValueError(f"negative stack segment for {workload}/{system}")
            if abs(raw - effective - wasted) > 1e-9:
                raise ValueError(f"stack does not close for {workload}/{system}")
            x = center + (system_index - group_midpoint) * within_step
            ax.bar(
                x,
                effective,
                bar_width,
                color=COLORS[system],
                edgecolor=EDGE,
                hatch=HATCHES[system],
                linewidth=0.65,
                zorder=3,
            )
            if wasted > 1e-12:
                ax.bar(
                    x,
                    wasted,
                    bar_width,
                    bottom=effective,
                    color=WASTED_COLOR,
                    edgecolor=EDGE,
                    hatch=WASTED_HATCH,
                    linewidth=0.65,
                    zorder=3,
                )

    first_center = group_centers[WORKLOADS[0]]
    last_center = group_centers[WORKLOADS[-1]]
    half_group = group_midpoint * within_step + bar_width / 2.0
    ax.set_xlim(first_center - half_group - 0.08, last_center + half_group + 0.08)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(relative_tick))
    ax.set_xticks([group_centers[workload] for workload in WORKLOADS])
    ax.set_xticklabels(
        [WORKLOAD_LABELS[workload] for workload in WORKLOADS],
        fontsize=FS_WORKLOAD,
    )
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.2, length=2.0, width=0.6)
    ax.set_ylabel("Relative throughput", fontsize=FS_AXIS, labelpad=2.0)
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
    handles_by_label["Wasted"] = Patch(
        facecolor=WASTED_COLOR,
        edgecolor=EDGE,
        hatch=WASTED_HATCH,
        linewidth=0.55,
    )
    # Matplotlib fills a multi-row legend column-first.  This order renders as
    # Dense / Tile skipping / Gather-scatter on the first row and
    # WISEConv / Wasted on the second.
    labels = ["Dense", "WISEConv", "Tile skipping", "Wasted", "Gather-scatter"]
    handles = [handles_by_label[label] for label in labels]
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=ax.transAxes,
        ncol=3,
        frameon=False,
        fontsize=FS_LEG,
        handlelength=0.9,
        handletextpad=0.28,
        columnspacing=0.72,
        borderaxespad=0.0,
        borderpad=0.0,
        labelspacing=0.3,
    )
    for label in legend.get_texts():
        if label.get_text() == "WISEConv":
            label.set_fontweight("bold")

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
