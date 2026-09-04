#!/usr/bin/env python3
"""Render the two-platform effective-throughput figure from the paper CSV."""

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

plt.rcParams.update({
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
})

DEVICES = ("rtx3080", "agx_orin")
DEVICE_LABELS = {"rtx3080": "RTX 3080", "agx_orin": "AGX Orin"}
SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
WORKLOADS = ("fireflownet", "yolov8n", "yolov8m", "dynconv_pose")
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}

# The 192 pt canvas is 80% of the measured 240 pt acmart column width and is
# included at that physical size. Both panels are deliberately shallow.
FIG_W, FIG_H = 192.0 / 72.0, 2.15
FS_TICK, FS_AXIS, FS_WORKLOAD, FS_PANEL, FS_LEG = 6.1, 6.9, 6.4, 6.7, 5.8
WASTED_COLOR = "#e3e3e3"
WASTED_HATCH = "...."


def read_rows() -> dict[tuple[str, str, str], dict]:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    expected = len(DEVICES) * len(WORKLOADS) * len(SYSTEMS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    by_key = {}
    for row in rows:
        key = (row["device"], row["workload"], row["system"])
        if key in by_key:
            raise ValueError(f"duplicate row {key}")
        by_key[key] = row
    required = {
        (device, workload, system)
        for device in DEVICES
        for workload in WORKLOADS
        for system in SYSTEMS
    }
    if set(by_key) != required:
        raise ValueError("effective-throughput CSV has unexpected keys")
    return by_key


def relative_tick(value: float, _: int) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def plot_panel(ax, rows, device, show_xlabels):
    group_step = 0.92
    within_step = 0.180
    bar_width = 0.145
    centers = {
        workload: index * group_step
        for index, workload in enumerate(WORKLOADS)
    }
    midpoint = (len(SYSTEMS) - 1) / 2.0
    for workload in WORKLOADS:
        for system_index, system in enumerate(SYSTEMS):
            row = rows[(device, workload, system)]
            effective = float(row["effective_throughput_vs_dense"])
            wasted = float(row["wasted_throughput_vs_dense"])
            raw = float(row["raw_throughput_vs_dense"])
            if abs(raw - effective - wasted) > 1e-9:
                raise ValueError(
                    f"stack does not close for {device}/{workload}/{system}"
                )
            x = centers[workload] + (system_index - midpoint) * within_step
            ax.bar(
                x, effective, bar_width, color=COLORS[system], edgecolor=EDGE,
                hatch=HATCHES[system], linewidth=0.62, zorder=3,
            )
            if wasted > 1e-12:
                ax.bar(
                    x, wasted, bar_width, bottom=effective,
                    color=WASTED_COLOR, edgecolor=EDGE, hatch=WASTED_HATCH,
                    linewidth=0.62, zorder=3,
                )

    half_group = midpoint * within_step + bar_width / 2.0
    ax.set_xlim(
        centers[WORKLOADS[0]] - half_group - 0.08,
        centers[WORKLOADS[-1]] + half_group + 0.08,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.yaxis.set_major_formatter(FuncFormatter(relative_tick))
    ax.set_xticks([centers[workload] for workload in WORKLOADS])
    ax.set_xticklabels(
        [WORKLOAD_LABELS[workload] for workload in WORKLOADS]
        if show_xlabels else []
    )
    ax.tick_params(axis="x", length=0, pad=2.0, labelsize=FS_WORKLOAD)
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.0, length=1.8, width=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#bcbcbc", alpha=0.28, linewidth=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    rows = read_rows()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = (
        fig.add_axes([0.14, 0.585, 0.845, 0.245]),
        fig.add_axes([0.14, 0.205, 0.845, 0.245]),
    )
    for ax, device in zip(axes, DEVICES, strict=True):
        plot_panel(ax, rows, device, show_xlabels=True)
    fig.text(
        0.562, 0.505, f"(a) {DEVICE_LABELS[DEVICES[0]]}",
        ha="center", va="center", fontsize=FS_PANEL,
    )
    fig.text(
        0.562, 0.115, f"(b) {DEVICE_LABELS[DEVICES[1]]}",
        ha="center", va="center", fontsize=FS_PANEL,
    )
    fig.text(
        0.025, 0.515, "Relative throughput", rotation=90,
        ha="center", va="center", fontsize=FS_AXIS,
    )

    handles_by_label = dict(zip(
        [SYSTEM_LABELS[system] for system in SYSTEMS],
        legend_handles(SYSTEMS), strict=True,
    ))
    handles_by_label["Wasted"] = Patch(
        facecolor=WASTED_COLOR, edgecolor=EDGE, hatch=WASTED_HATCH,
        linewidth=0.52,
    )
    labels = [
        "Dense", "WISEConv", "Tile skipping", "Wasted", "Gather-scatter"
    ]
    legend = fig.legend(
        handles=[handles_by_label[label] for label in labels], labels=labels,
        loc="upper center", bbox_to_anchor=(0.535, 0.985), ncol=3,
        frameon=False, fontsize=FS_LEG, handlelength=0.75,
        handletextpad=0.22, columnspacing=0.40, labelspacing=0.22,
        borderaxespad=0.0, borderpad=0.0,
    )
    for label in legend.get_texts():
        if label.get_text() == "WISEConv":
            label.set_fontweight("bold")

    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
