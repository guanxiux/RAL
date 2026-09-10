#!/usr/bin/env python3
"""FP32 vs FP16 tensor-core sensitivity, e2e latency style."""

from __future__ import annotations
import math, os, glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from bar_patterns import add_bar, legend_handles

HERE = Path(__file__).resolve().parent

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
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.65, "hatch.linewidth": 0.36,
})

FIG_W, FIG_H = 3.334, 2.10
FS_TICK = 6.1; FS_AXIS = 7.2; FS_PANEL = 7.0
FS_LEG = 7.2; FS_WORKLOAD = 6.9; FS_DTYPE = 6.2

BACKENDS = ["dense", "wiseconv"]
LABELS = {"dense": "Dense", "wiseconv": "WISEConv"}

# (Dense, WISEConv) in ms
DATA = {
    "RTX 3080": {
        "FireFlowNet": {"FP32": (1.043, 0.399), "FP16": (0.347, 0.243)},
        "YOLOv8n":     {"FP32": (2.789, 1.582), "FP16": (1.212, 1.067)},
        "YOLOv8m":     {"FP32": (10.178, 5.967), "FP16": (3.532, 3.070)},
        "DynConv":     {"FP32": (5.986, 3.197), "FP16": (3.366, 2.459)},
    },
    "AGX Orin": {
        "FireFlowNet": {"FP32": (12.466, 5.390), "FP16": (4.310, 2.775)},
        "YOLOv8n":     {"FP32": (24.670, 16.738), "FP16": (8.194, 6.378)},
        "YOLOv8m":     {"FP32": (142.973, 90.565), "FP16": (29.616, 16.863)},
        "DynConv":     {"FP32": (61.720, 22.494), "FP16": (22.045, 15.553)},
    },
}
PLATFORMS = ["RTX 3080", "AGX Orin"]
WORKLOADS = ["FireFlowNet", "YOLOv8n", "YOLOv8m", "DynConv"]
DTYPES = ["FP32", "FP16"]


def natural_ticks(max_value):
    desired = max_value * 1.02
    exp = math.floor(math.log10(desired))
    best = None
    for pw in range(exp - 2, exp + 2):
        s = 10.0 ** pw
        for m in (1, 2, 2.5, 5):
            step = m * s
            n = math.ceil(desired / step - 1e-12)
            if 2 <= n <= 6:
                top = n * step
                score = ((top - desired) / desired, abs(n - 4))
                if best is None or score < best[0]:
                    best = (score, step, n)
    if best is None:
        return [0, desired], 1
    _, step, n = best
    ticks = [i * step for i in range(n + 1)]
    if math.isclose(step, round(step), abs_tol=1e-10): dec = 0
    elif math.isclose(step*10, round(step*10), abs_tol=1e-10): dec = 1
    else: dec = 2
    return ticks, dec


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # Layout: 2 rows (platforms), each row = platform_label + box with 4 workloads
    margin_left = 0.01
    margin_right = 0.01
    margin_top = 0.20
    margin_bottom = 0.02
    row_gap = 0.10
    platform_label_w = 0.28
    y_label_gutter = 0.28
    panel_right_pad = 0.03
    workload_gap = 0.08

    n_rows = len(PLATFORMS)
    n_workloads = len(WORKLOADS)
    row_height = (FIG_H - margin_top - margin_bottom - (n_rows - 1) * row_gap) / n_rows
    box_width = FIG_W - margin_left - platform_label_w - margin_right
    axes_total_w = box_width - y_label_gutter - panel_right_pad
    single_w = (axes_total_w - (n_workloads - 1) * workload_gap) / n_workloads

    bar_w = 0.15
    bar_gap = 0.02
    group_gap = 0.30

    for p_idx, platform in enumerate(PLATFORMS):
        row_top = FIG_H - margin_top - p_idx * (row_height + row_gap)
        row_bottom = row_top - row_height
        box_left = margin_left + platform_label_w

        # Box bounds in figure fraction
        bx = box_left / FIG_W
        by = row_bottom / FIG_H
        bw = box_width / FIG_W
        bh = row_height / FIG_H

        # Outer box
        border = Rectangle(
            (bx, by), bw, bh,
            transform=fig.transFigure,
            linewidth=0.5, edgecolor="#2b2b2b",
            facecolor="none", clip_on=False, zorder=0,
        )
        fig.patches.append(border)

        # Platform label
        fig.text(
            (margin_left + platform_label_w * 0.45) / FIG_W,
            by + bh / 2,
            f"({chr(97 + p_idx)}) {platform}",
            ha="center", va="center",
            rotation=90, fontsize=FS_PANEL,
        )

        # Y label
        fig.text(
            (box_left + 0.06) / FIG_W,
            by + bh / 2,
            "Latency (ms)",
            ha="center", va="center",
            rotation=90, fontsize=FS_AXIS,
        )

        axes_left_base = box_left + y_label_gutter
        axes_bottom = row_bottom + 0.32
        axes_height = row_height - 0.36

        for w_idx, workload in enumerate(WORKLOADS):
            ax_left = axes_left_base + w_idx * (single_w + workload_gap)
            rect = [ax_left / FIG_W, axes_bottom / FIG_H,
                    single_w / FIG_W, axes_height / FIG_H]
            ax = fig.add_axes(rect)

            within = bar_w + bar_gap
            gc0 = bar_w / 2 + within / 2
            gc1 = gc0 + within + group_gap
            group_centers = [gc0, gc1]
            all_vals = []
            for d_idx, dtype in enumerate(DTYPES):
                dense_v, wise_v = DATA[platform][workload][dtype]
                all_vals.extend([dense_v, wise_v])
                gc = group_centers[d_idx]
                for b_idx, backend in enumerate(BACKENDS):
                    val = [dense_v, wise_v][b_idx]
                    x = gc + (b_idx - 0.5) * within
                    add_bar(ax, x, val, bar_w, backend, linewidth=0.65)

            ticks, dec = natural_ticks(max(all_vals))
            ax.set_ylim(0, ticks[-1])
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _, d=dec: f"{v:.{d}f}")
            )
            ax.set_xlim(0, gc1 + within / 2 + bar_w / 2 + 0.02)
            ax.set_xticks(group_centers)
            ax.set_xticklabels(DTYPES, fontsize=FS_DTYPE)
            ax.tick_params(axis="x", length=0, pad=4)
            ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.0,
                           length=2.0, width=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="#bcbcbc", alpha=0.28, linewidth=0.35)

            # Workload label below (under FP32/FP16 labels)
            fig.text(
                (ax_left + single_w / 2) / FIG_W,
                (axes_bottom - 0.12) / FIG_H,
                workload,
                ha="center", va="top",
                fontsize=FS_WORKLOAD,
            )

    # Legend
    handles = legend_handles(BACKENDS)
    labels = [LABELS[b] for b in BACKENDS]
    fig.legend(
        handles=handles, labels=labels,
        fontsize=FS_LEG, ncol=len(BACKENDS), frameon=False,
        loc="upper center", bbox_to_anchor=(0.55, 1.01),
        handlelength=1.2, columnspacing=1.1, handletextpad=0.4,
    )

    out = HERE / "dtype_sensitivity.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
