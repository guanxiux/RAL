#!/usr/bin/env python3
"""Render the end-to-end latency, energy, and accuracy results from CSV."""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from bar_patterns import (
    add_bar,
    legend_handles,
)


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "e2e_stats.csv"
ACCURACY_CSV_PATH = HERE / "e2e_accuracy.csv"
LATENCY_PDF = HERE / "e2e_latency.pdf"
ACCURACY_PDF = HERE / "e2e_accuracy.pdf"
ENERGY_TEX = HERE / "e2e_energy_table.tex"

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
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.65,
        "hatch.linewidth": 0.36,
    }
)

BACKENDS = ["dense", "tile_skip", "gather_scatter", "wiseconv"]
LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}

PLATFORMS = [
    "tbd-a",
    "tbd-b",
    "3080",
    "4070-laptop",
    "agx-orin",
    "xavier-nx",
]
WORKLOADS = ["fireflownet", "yolov8n", "yolov8m", "dynconv_pose"]
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv\nPose",
}

# Exact ACM sigplan width under the repository's acmart.cls.  The slightly
# taller canvas reserves enough room for bottom-row workload labels at 1:1 size.
FIG_W, FIG_H = 7.00, 4.25
FS_TICK, FS_AXIS, FS_PANEL, FS_LEG, FS_WORKLOAD = 6.1, 7.2, 7.0, 7.2, 6.9
BREAK_RATIO = 1.75


def read_rows():
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    expected = len(PLATFORMS) * len(WORKLOADS) * len(BACKENDS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    by_key = {}
    for row in rows:
        key = (row["platform"], row["workload"], row["backend"])
        if key in by_key:
            raise ValueError(f"duplicate row {key}")
        by_key[key] = row
    return rows, by_key


def read_accuracy_rows():
    rows = list(csv.DictReader(ACCURACY_CSV_PATH.open(encoding="utf-8")))
    expected = len(WORKLOADS) * len(BACKENDS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} accuracy rows, found {len(rows)}")
    by_key = {}
    for row in rows:
        key = (row["workload"], row["backend"])
        if key in by_key:
            raise ValueError(f"duplicate accuracy row {key}")
        by_key[key] = row
    return rows, by_key


def bar(ax, index: int, backend: str, value: float, width: float):
    return add_bar(ax, index, value, width, backend, linewidth=0.65)


def contiguous_bar_x(index: int, width: float) -> float:
    """Center a gap-free backend group within the categorical x range."""
    group_center = (len(BACKENDS) - 1) / 2
    return group_center + (index - group_center) * width


def style_axis(ax) -> None:
    ax.set_xlim(-0.52, len(BACKENDS) - 0.48)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.0, length=2.0, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#bcbcbc", alpha=0.28, linewidth=0.35)


def natural_ticks(max_value: float):
    """Choose a 1/2/2.5/5-scaled axis with four to seven labeled ticks."""
    desired_top = max_value * 1.08
    exponent = math.floor(math.log10(desired_top))
    choices = []
    for power in range(exponent - 2, exponent + 2):
        scale = 10.0**power
        for multiplier in (1.0, 2.0, 2.5, 5.0):
            step = multiplier * scale
            intervals = math.ceil(desired_top / step - 1e-12)
            if 3 <= intervals <= 6:
                top = intervals * step
                choices.append(
                    (abs(intervals - 5), (top - desired_top) / desired_top, step,
                     intervals)
                )
    if not choices:
        raise ValueError(f"could not choose natural ticks for {max_value}")
    _, _, step, intervals = min(choices)
    ticks = [index * step for index in range(intervals + 1)]
    if math.isclose(step, round(step), rel_tol=0.0, abs_tol=1e-10):
        decimals = 0
    elif math.isclose(step * 10, round(step * 10), rel_tol=0.0,
                      abs_tol=1e-10):
        decimals = 1
    else:
        decimals = 2
    return ticks, decimals


def draw_regular_axis(fig, rect, values):
    ax = fig.add_axes(rect)
    patches = {}
    bar_width = 0.68
    for index, backend in enumerate(BACKENDS):
        bar_x = contiguous_bar_x(index, bar_width)
        patches[backend] = bar(
            ax, bar_x, backend, values[backend], bar_width
        )
    style_axis(ax)
    ticks, decimals = natural_ticks(max(values.values()))
    ax.set_ylim(0, ticks[-1])
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.{decimals}f}")
    )


def draw_broken_axis(fig, rect, values):
    left, bottom, width, height = rect
    bottom_fraction = 0.72
    gap_fraction = 0.07
    bottom_height = height * bottom_fraction
    gap_height = height * gap_fraction
    top_height = height - bottom_height - gap_height
    ax_bottom = fig.add_axes([left, bottom, width, bottom_height])
    ax_top = fig.add_axes(
        [left, bottom + bottom_height + gap_height, width, top_height]
    )

    patches = {}
    bar_width = 0.68
    for axis in (ax_bottom, ax_top):
        patches[axis] = {}
        for index, backend in enumerate(BACKENDS):
            bar_x = contiguous_bar_x(index, bar_width)
            patches[axis][backend] = bar(
                axis, bar_x, backend, values[backend], bar_width
            )
        style_axis(axis)

    outlier = values["gather_scatter"]
    body_max = max(value for key, value in values.items() if key != "gather_scatter")
    ax_bottom.set_ylim(0, body_max * 1.20)
    ax_top.set_ylim(outlier * 0.94, outlier * 1.07)
    ax_top.set_yticks([outlier])
    outlier_label = f"{outlier:.1f}" if outlier < 10 else f"{outlier:.0f}"
    ax_top.set_yticklabels([outlier_label])
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False)
    ax_top.yaxis.grid(False)

    # Match fig:gap: two dashed diagonal marks on the left spine.
    mark = 0.0065
    kwargs = {
        "transform": fig.transFigure,
        "color": "black",
        "linewidth": 0.7,
        "clip_on": False,
        "dashes": (2, 1.4),
    }
    for y_center in (
        bottom + bottom_height,
        bottom + bottom_height + gap_height,
    ):
        fig.lines.append(
            plt.Line2D(
                [left - mark, left + mark],
                [y_center - mark, y_center + mark],
                **kwargs,
            )
        )


def needs_break(values) -> bool:
    """Break a gather-scatter outlier before it compresses the other paths."""
    gather = values["gather_scatter"]
    body_max = max(value for key, value in values.items() if key != "gather_scatter")
    return gather >= BREAK_RATIO * body_max


def draw_latency_figure(by_key) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # The geometry mirrors ActiMM: a narrow platform-label gutter sits outside
    # each boxed panel; a second gutter inside the box holds one shared y label;
    # and the four independent workload axes receive enough horizontal space
    # for their own tick labels.
    margin_left, margin_right = 0.04, 0.04
    margin_top, margin_bottom = 0.36, 0.56
    panel_gap_x, panel_gap_y = 0.14, 0.18
    platform_label_width = 0.25
    panel_unit_width = (
        FIG_W - margin_left - margin_right - panel_gap_x
    ) / 2
    panel_box_width = panel_unit_width - platform_label_width
    y_label_gutter, panel_right_pad = 0.30, 0.05
    axes_width = panel_box_width - y_label_gutter - panel_right_pad
    workload_gap = 0.18
    workload_width = (axes_width - 3 * workload_gap) / 4
    panel_height = (
        FIG_H - margin_top - margin_bottom - 2 * panel_gap_y
    ) / 3
    panel_letters = "abcdef"

    platform_meta = {}
    for platform in PLATFORMS:
        row = by_key[(platform, WORKLOADS[0], BACKENDS[0])]
        platform_meta[platform] = (
            row["platform_label"],
            row["platform_status"],
        )

    for panel_index, platform in enumerate(PLATFORMS):
        row_index, column_index = divmod(panel_index, 2)
        panel_unit_left = (
            margin_left + column_index * (panel_unit_width + panel_gap_x)
        )
        panel_box_left = panel_unit_left + platform_label_width
        axes_left = panel_box_left + y_label_gutter
        y_origin = (
            FIG_H
            - margin_top
            - (row_index + 1) * panel_height
            - row_index * panel_gap_y
        )
        box_left_fraction = panel_box_left / FIG_W
        y_fraction = y_origin / FIG_H
        panel_box_width_fraction = panel_box_width / FIG_W
        panel_height_fraction = panel_height / FIG_H
        label, status = platform_meta[platform]

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

        if status == "placeholder":
            fig.text(
                box_left_fraction + panel_box_width_fraction / 2,
                y_fraction + panel_height_fraction / 2,
                "[Results pending]",
                ha="center",
                va="center",
                fontsize=FS_AXIS,
                color="#777777",
                style="italic",
            )
        else:
            fig.text(
                (panel_box_left + 0.095) / FIG_W,
                (y_origin + 0.5 * panel_height) / FIG_H,
                "Latency (ms)",
                ha="center",
                va="center",
                rotation=90,
                fontsize=FS_AXIS,
            )
            for workload_index, workload in enumerate(WORKLOADS):
                left = (
                    axes_left
                    + workload_index * (workload_width + workload_gap)
                ) / FIG_W
                rect = [
                    left,
                    y_fraction,
                    workload_width / FIG_W,
                    panel_height_fraction,
                ]
                values = {
                    backend: float(
                        by_key[(platform, workload, backend)]["latency_ms"]
                    )
                    for backend in BACKENDS
                }
                if needs_break(values):
                    draw_broken_axis(fig, rect, values)
                else:
                    draw_regular_axis(fig, rect, values)

                if row_index == 2:
                    fig.text(
                        left + 0.5 * workload_width / FIG_W,
                        (y_origin - 0.13) / FIG_H,
                        WORKLOAD_LABELS[workload],
                        ha="center",
                        va="top",
                        fontsize=FS_WORKLOAD,
                        linespacing=0.88,
                    )

        pad_y = 0.05
        border = Rectangle(
            (
                box_left_fraction,
                y_fraction - pad_y / FIG_H,
            ),
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
        ncol=4,
        frameon=False,
        fontsize=FS_LEG,
        handlelength=1.35,
        handletextpad=0.4,
        columnspacing=1.25,
    )
    for text in legend.get_texts():
        if text.get_text() == "WISEConv":
            text.set_fontweight("bold")

    fig.savefig(LATENCY_PDF)
    plt.close(fig)
    print(f"wrote {LATENCY_PDF}")


def draw_accuracy_figure(by_key) -> None:
    """Draw four metric-specific workload panels with a shared path legend."""
    # acmart's sigplan layout uses a 3.334-inch column.  Emit at that width so
    # LaTeX does not rescale the figure and its typography.
    fig_width, fig_height = 3.334, 1.65
    fig = plt.figure(figsize=(fig_width, fig_height))
    margin_left, margin_right = 0.31, 0.01
    margin_bottom, margin_top = 0.32, 0.34
    panel_gap = 0.265
    panel_width = (
        fig_width - margin_left - margin_right - 3 * panel_gap
    ) / 4
    panel_height = fig_height - margin_bottom - margin_top

    metric_labels = {
        "AEE": "AEE",
        "AP50:95 (%)": "AP (%)",
        "PCKh (%)": "PCKh (%)",
    }

    for workload_index, workload in enumerate(WORKLOADS):
        left = margin_left + workload_index * (panel_width + panel_gap)
        ax = fig.add_axes(
            [
                left / fig_width,
                margin_bottom / fig_height,
                panel_width / fig_width,
                panel_height / fig_height,
            ]
        )
        rows = {backend: by_key[(workload, backend)] for backend in BACKENDS}
        values = {backend: float(rows[backend]["value"]) for backend in BACKENDS}
        metrics = {row["metric"] for row in rows.values()}
        directions = {row["direction"] for row in rows.values()}
        if len(metrics) != 1 or len(directions) != 1:
            raise ValueError(f"inconsistent accuracy metadata for {workload}")
        metric = metrics.pop()

        bar_width = 0.68
        for backend_index, backend in enumerate(BACKENDS):
            # Match the latency bars' width while keeping the four paths in one
            # contiguous comparison group centered within the workload panel.
            bar_x = contiguous_bar_x(backend_index, bar_width)
            bar(ax, bar_x, backend, values[backend], bar_width)

        ticks, decimals = natural_ticks(max(values.values()))
        ax.set_xlim(-0.52, len(BACKENDS) - 0.48)
        ax.set_ylim(0, ticks[-1])
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _, digits=decimals: f"{value:.{digits}f}"
            )
        )
        ax.set_xticks([])
        ax.set_xlabel(
            WORKLOAD_LABELS[workload],
            fontsize=FS_WORKLOAD,
            labelpad=2.5,
            linespacing=0.88,
        )
        ax.set_ylabel(metric_labels[metric], fontsize=FS_AXIS, labelpad=0.8)
        ax.yaxis.set_label_coords(-0.28, 0.5)
        ax.tick_params(
            axis="y", labelsize=FS_TICK, pad=0.8, length=1.8, width=0.6
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#bcbcbc", alpha=0.28, linewidth=0.35)

    handles = legend_handles(BACKENDS)
    legend = fig.legend(
        handles=handles,
        labels=[LABELS[backend] for backend in BACKENDS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=FS_LEG,
        handlelength=0.9,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    for text in legend.get_texts():
        if text.get_text() == "WISEConv":
            text.set_fontweight("bold")

    fig.savefig(ACCURACY_PDF)
    plt.close(fig)
    print(f"wrote {ACCURACY_PDF}")


def format_energy(value: float, best: bool) -> str:
    formatted = f"{value:.3f}"
    return f"\\textbf{{{formatted}}}" if best else formatted


def write_energy_table(by_key) -> None:
    platforms = ["agx-orin", "xavier-nx"]
    lines = [
        "% Generated by figures/plot_e2e.py from figures/e2e_stats.csv.",
        "\\begin{table}[b]",
        "  \\centering",
        "  \\caption{Per-frame board energy (J) on the Jetson platforms. "
        "Lower is better.}",
        "  \\label{tab:e2e-energy}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{2.6pt}",
        "  \\renewcommand{\\arraystretch}{1.08}",
        "  \\begin{tabular}{@{}clrrrr@{}}",
        "    \\toprule",
        "    \\rotatebox[origin=c]{90}{GPU} & System & FireFlowNet & "
        "YOLOv8n & YOLOv8m & \\shortstack{DynConv\\\\Pose} \\\\",
        "    \\midrule",
    ]

    platform_labels = {
        "agx-orin": "AGX Orin",
        "xavier-nx": "Xavier NX",
    }
    system_labels = {
        "dense": "Dense",
        "tile_skip": "Tile skip",
        "gather_scatter": "Gather",
        "wiseconv": "\\textbf{WISEConv}",
    }

    for platform_index, platform in enumerate(platforms):
        if platform_index:
            lines.append("    \\midrule")
        best_values = {
            workload: min(
                float(
                    by_key[(platform, workload, backend)]["energy_j_per_frame"]
                )
                for backend in BACKENDS
            )
            for workload in WORKLOADS
        }
        for backend_index, backend in enumerate(BACKENDS):
            cells = []
            for workload in WORKLOADS:
                value = float(
                    by_key[(platform, workload, backend)]["energy_j_per_frame"]
                )
                cells.append(format_energy(value, value == best_values[workload]))
            platform_cell = ""
            if backend_index == 0:
                platform_cell = (
                    "\\multirow{4}{*}{\\rotatebox[origin=c]{90}{\\scriptsize "
                    + platform_labels[platform]
                    + "}}"
                )
            lines.append(
                f"    {platform_cell} & {system_labels[backend]} & "
                + " & ".join(cells)
                + " \\\\"
            )

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    ENERGY_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {ENERGY_TEX}")


def print_ranges(rows) -> None:
    latency_groups = defaultdict(dict)
    energy_groups = defaultdict(dict)
    for row in rows:
        if row["latency_ms"]:
            latency_groups[(row["platform"], row["workload"])][row["backend"]] = float(
                row["latency_ms"]
            )
        if row["energy_j_per_frame"]:
            energy_groups[(row["platform"], row["workload"])][row["backend"]] = float(
                row["energy_j_per_frame"]
            )

    latency_cuts = []
    wiseconv_fastest = 0
    sparse_not_faster_than_dense = 0
    for (platform, workload), values in latency_groups.items():
        rival = min(value for key, value in values.items() if key != "wiseconv")
        latency_cuts.append(
            (platform, workload, 1.0 - values["wiseconv"] / rival)
        )
        if values["wiseconv"] < rival:
            wiseconv_fastest += 1
        sparse_rival = min(values["tile_skip"], values["gather_scatter"])
        if sparse_rival >= values["dense"]:
            sparse_not_faster_than_dense += 1

    energy_rival_cuts, energy_dense_cuts = [], []
    wiseconv_lowest_energy = 0
    for values in energy_groups.values():
        rival = min(value for key, value in values.items() if key != "wiseconv")
        energy_rival_cuts.append(1.0 - values["wiseconv"] / rival)
        energy_dense_cuts.append(1.0 - values["wiseconv"] / values["dense"])
        if values["wiseconv"] < rival:
            wiseconv_lowest_energy += 1

    if wiseconv_fastest != len(latency_groups):
        raise ValueError(
            f"WISEConv is fastest in only {wiseconv_fastest}/{len(latency_groups)} "
            "measured latency pairs"
        )
    if wiseconv_lowest_energy != len(energy_groups):
        raise ValueError(
            f"WISEConv has lowest energy in only {wiseconv_lowest_energy}/"
            f"{len(energy_groups)} measured pairs"
        )

    print(
        "latency reduction vs fastest competitor: "
        f"{100 * min(cut for _, _, cut in latency_cuts):.1f}--"
        f"{100 * max(cut for _, _, cut in latency_cuts):.1f}%"
    )
    for label, platforms in (
        ("discrete GPUs", {"3080", "4070-laptop"}),
        ("Jetson platforms", {"agx-orin", "xavier-nx"}),
    ):
        cuts = [cut for platform, _, cut in latency_cuts if platform in platforms]
        print(
            f"  {label}: {100 * min(cuts):.1f}--{100 * max(cuts):.1f}%"
        )
    print(f"WISEConv fastest latency: {wiseconv_fastest}/{len(latency_groups)} pairs")
    print(
        "fastest existing sparse path does not beat dense: "
        f"{sparse_not_faster_than_dense}/{len(latency_groups)} pairs"
    )
    print(
        "energy reduction vs fastest competitor: "
        f"{100 * min(energy_rival_cuts):.1f}--{100 * max(energy_rival_cuts):.1f}%"
    )
    print(
        "energy reduction vs dense: "
        f"{100 * min(energy_dense_cuts):.1f}--{100 * max(energy_dense_cuts):.1f}%"
    )
    print(
        f"WISEConv lowest energy: {wiseconv_lowest_energy}/{len(energy_groups)} pairs"
    )


def main() -> None:
    rows, by_key = read_rows()
    _, accuracy_by_key = read_accuracy_rows()
    draw_latency_figure(by_key)
    draw_accuracy_figure(accuracy_by_key)
    write_energy_table(by_key)
    print_ranges(rows)


if __name__ == "__main__":
    main()
