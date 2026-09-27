#!/usr/bin/env python3
"""FP32 (SIMT) vs FP16 (tensor core) full-model latency, Dense vs WISEConv.

This figure is drawn with the *same* plotting machinery as the end-to-end
latency figure (`plot_e2e.py`): identical fonts, rcParams, bar styling,
axis styling, panel geometry and legend.  Values are read from the unified
`logs/latency` tree so the figure tracks the deployment runs.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plot_e2e as e2e  # installs fonts + rcParams, exposes shared helpers
from bar_patterns import add_bar, legend_handles

# Reuse the end-to-end figure's typography and styling verbatim.
FS_TICK = e2e.FS_TICK
FS_AXIS = e2e.FS_AXIS
FS_PANEL = e2e.FS_PANEL
FS_LEG = e2e.FS_LEG
FS_WORKLOAD = e2e.FS_WORKLOAD
FS_DTYPE = 6.2

BACKENDS = ["dense", "wiseconv"]
LABELS = {"dense": "Dense", "wiseconv": "WISEConv"}

DEVICES = {"RTX 3080": "3080", "AGX Orin": "agx-orin"}
WORKLOADS = ["fireflownet", "yolov8n", "yolov8m", "dynconv_pose"]
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}
DTYPES = ["FP32", "FP16"]
RUN_TEMPLATES = {
    "fireflownet": {"FP32": "full-r3-{}", "FP16": "fp16-full-r3-{}"},
    "yolov8n": {"FP32": "yolov8n-full-r3-{}", "FP16": "yolov8n-fp16-full-r3-{}"},
    "yolov8m": {"FP32": "yolov8m-full-r3-{}", "FP16": "yolov8m-fp16-full-r3-{}"},
    "dynconv_pose": {"FP32": "full-s0125-r3-{}", "FP16": "fp16-full-s0125-r3-{}"},
}

FIG_W, FIG_H = 3.334, 2.30
LOGS = HERE.parents[1] / "logs"


def _latency(device_dir: str, workload: str, dtype: str, backend: str) -> float:
    run = RUN_TEMPLATES[workload][dtype].format(backend)
    path = (
        LOGS / "latency" / workload / device_dir / dtype.lower() / run
        / "summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [float(seq["all_frames_gpu_ms"]["mean"])
              for seq in payload["sequences"].values()]
    return statistics.fmean(values)


def load_data():
    data = {}
    for label, device_dir in DEVICES.items():
        data[label] = {}
        for workload in WORKLOADS:
            data[label][workload] = {
                dtype: (
                    _latency(device_dir, workload, dtype, "dense"),
                    _latency(device_dir, workload, dtype, "wiseconv"),
                )
                for dtype in DTYPES
            }
    return data


DATA = load_data()

# Four bars per workload panel: [Dense, WISEConv] x [FP32, FP16].
BAR_W = 0.48
BAR_X = [0.00, 0.48, 1.40, 1.88]
GROUP_CENTERS = [0.24, 1.64]
PANEL_XLIM = (-0.52, 2.40)


def style_axis(ax) -> None:
    """Identical axis treatment to plot_e2e.draw_latency_figure."""
    ax.set_xlim(*PANEL_XLIM)
    ax.set_xticks(GROUP_CENTERS)
    ax.set_xticklabels(DTYPES, fontsize=FS_DTYPE)
    ax.tick_params(axis="x", length=0, pad=4.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.0, length=2.0, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#bcbcbc", alpha=0.28, linewidth=0.35)


def draw_panel(fig, rect, values):
    ax = fig.add_axes(rect)
    for backend_index, backend in enumerate(BACKENDS):
        for dtype_index, dtype in enumerate(DTYPES):
            slot = dtype_index * 2 + backend_index
            dense_v, wise_v = values[dtype]
            value = dense_v if backend == "dense" else wise_v
            add_bar(ax, BAR_X[slot], value, BAR_W, backend, linewidth=0.65)
    style_axis(ax)
    top = max(max(values[dtype]) for dtype in DTYPES)
    ticks, decimals = e2e.natural_ticks(top)
    ax.set_ylim(0, ticks[-1])
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.{decimals}f}")
    )


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Geometry mirrors plot_e2e.draw_latency_figure, reduced to a single column
    # of platforms and two rows (RTX 3080, AGX Orin).
    margin_left, margin_right = 0.03, 0.03
    margin_top, margin_bottom = 0.28, 0.34
    panel_gap_y = 0.22
    platform_label_width = 0.15
    panel_unit_width = FIG_W - margin_left - margin_right
    panel_box_width = panel_unit_width - platform_label_width
    y_label_gutter, panel_right_pad = 0.30, 0.03
    axes_width = panel_box_width - y_label_gutter - panel_right_pad
    workload_gap = 0.13
    workload_width = (axes_width - 3 * workload_gap) / 4
    panel_height = (
        FIG_H - margin_top - margin_bottom - (len(DEVICES) - 1) * panel_gap_y
    ) / len(DEVICES)
    panel_letters = "ab"

    for panel_index, (label, _device_dir) in enumerate(DEVICES.items()):
        row_top = FIG_H - margin_top - panel_index * (panel_height + panel_gap_y)
        y_origin = row_top - panel_height
        panel_unit_left = margin_left
        panel_box_left = panel_unit_left + platform_label_width
        axes_left = panel_box_left + y_label_gutter
        box_left_fraction = panel_box_left / FIG_W
        y_fraction = y_origin / FIG_H
        panel_box_width_fraction = panel_box_width / FIG_W
        panel_height_fraction = panel_height / FIG_H

        fig.text(
            (panel_unit_left + 0.42 * platform_label_width) / FIG_W,
            (y_origin + 0.5 * panel_height) / FIG_H,
            f"({panel_letters[panel_index]}) {label}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=FS_PANEL,
            fontweight="bold",
        )
        fig.text(
            (panel_box_left + 0.055) / FIG_W,
            (y_origin + 0.5 * panel_height) / FIG_H,
            "Latency (ms)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=FS_AXIS,
        )

        for workload_index, workload in enumerate(WORKLOADS):
            left = (
                axes_left + workload_index * (workload_width + workload_gap)
            ) / FIG_W
            rect = [
                left,
                y_fraction,
                workload_width / FIG_W,
                panel_height_fraction,
            ]
            draw_panel(fig, rect, DATA[label][workload])
            if panel_index == len(DEVICES) - 1:
                # Only the bottom row carries workload names, as in the e2e
                # figure where columns are shared across platform rows.
                fig.text(
                    left + 0.5 * workload_width / FIG_W,
                    (y_origin - 0.15) / FIG_H,
                    WORKLOAD_LABELS[workload],
                    ha="center",
                    va="top",
                    fontsize=FS_WORKLOAD,
                )

        pad_y = 0.05
        border = Rectangle(
            (box_left_fraction, y_fraction - pad_y / FIG_H),
            panel_box_width_fraction,
            panel_height_fraction + 2 * pad_y / FIG_H,
            transform=fig.transFigure,
            linewidth=0.5,
            edgecolor="#2b2b2b",
            facecolor="none",
            clip_on=False,
            zorder=0,
        )
        fig.patches.append(border)

    handles = legend_handles(BACKENDS)
    legend = fig.legend(
        handles=handles,
        labels=[LABELS[backend] for backend in BACKENDS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(BACKENDS),
        frameon=False,
        fontsize=FS_LEG,
        handlelength=1.35,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    for text in legend.get_texts():
        if text.get_text() == "WISEConv":
            text.set_fontweight("bold")

    out = HERE / "dtype_sensitivity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
