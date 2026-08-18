#!/usr/bin/env python3
"""Render the Background sparsity-to-latency example from gap_stats.csv.

The single-column figure contains two equal-area panels:

* Panel (a) compares full-model latency and AEE for the three execution paths.
  The dashed line is dense latency scaled by the active ratio aggregated
  across layers.
* Panel (b) compares raw and effective throughput, each normalized to dense raw
  throughput.

The plotting script reads only the generated CSV.  Run gap_stats.py first to
refresh every number from the formal logs.
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from bar_patterns import add_bar, legend_handles


HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "gap_stats.csv")

for directory in (
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive",
):
    for path in glob.glob(
        os.path.join(directory, "**", "texgyretermes-*.otf"), recursive=True
    ):
        fm.fontManager.addfont(path)

# acmart's sigplan column is 240.945 TeX pt, or 3.334 physical inches.  Emit
# at that width so LaTeX does not rescale the typography.
FIG_W, FIG_H = 3.334, 1.82
FS_TICK, FS_LAB, FS_LEG, FS_CAP, FS_REF = 6.5, 7.0, 7.0, 7.5, 6.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "hatch.linewidth": 0.38,
    }
)

ORDER = ["Dense", "Tile skipping", "Gather-scatter"]
STYLE_KEY = {
    "Dense": "dense",
    "Tile skipping": "tile_skip",
    "Gather-scatter": "gather_scatter",
}
LABEL = {
    "Dense": "Dense",
    "Tile skipping": "Tile skipping",
    "Gather-scatter": "Gather-scatter",
}

rows = {row["system"]: row for row in csv.DictReader(open(CSV, encoding="utf-8"))}
if set(rows) != set(ORDER):
    raise ValueError(f"unexpected systems in {CSV}: {tuple(rows)}")
latency = {system: float(rows[system]["latency_ms"]) for system in ORDER}
aee = {system: float(rows[system]["aee"]) for system in ORDER}
raw_relative = {
    system: float(rows[system]["raw_throughput_vs_dense"]) for system in ORDER
}
effective_relative = {
    system: float(rows[system]["effective_throughput_vs_dense"])
    for system in ORDER
}
reference_ms = float(rows["Dense"]["proportional_reference_ms"])

# Equal-area panel geometry in figure fractions.  The central gutter carries
# both panel (a)'s right y label and panel (b)'s left y label.
LX0, RX0, PW = 0.120, 0.680, 0.300
PB, PT = 0.190, 0.895
PH = PT - PB

# Panel (a) contains two groups of three touching bars.  The gap between groups
# holds the proportional-reference annotation.
BW_A = 0.34
LAT_C, AEE_C = 0.0, 2.15
LAT_X = [LAT_C + (index - 1) * BW_A for index in range(len(ORDER))]
AEE_X = [AEE_C + (index - 1) * BW_A for index in range(len(ORDER))]
GROUP_HALF = 1.5 * BW_A
GAP_L = LAT_C + GROUP_HALF
GAP_R = AEE_C - GROUP_HALF
AX_LO = LAT_C - GROUP_HALF - 0.19
AX_HI = AEE_C + GROUP_HALF + 0.19


def draw_latency_and_aee(fig: plt.Figure) -> None:
    """Draw full-model latency and AEE on separate y axes."""
    ax_latency = fig.add_axes([LX0, PB, PW, PH], zorder=2)
    ax_aee = ax_latency.twinx()
    ax_aee.set_zorder(1)
    ax_latency.patch.set_visible(False)
    ax_aee.patch.set_visible(False)

    for index, system in enumerate(ORDER):
        add_bar(
            ax_latency,
            LAT_X[index],
            latency[system],
            BW_A,
            STYLE_KEY[system],
            linewidth=0.8,
        )
        add_bar(
            ax_aee,
            AEE_X[index],
            aee[system],
            BW_A,
            STYLE_KEY[system],
            linewidth=0.8,
        )

    for axis in (ax_latency, ax_aee):
        axis.set_xlim(AX_LO, AX_HI)
        axis.set_ylim(0.0, 2.5)
        axis.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        axis.tick_params(axis="y", labelsize=FS_TICK, pad=1.5)
        axis.spines["top"].set_visible(False)

    ax_latency.set_xticks([LAT_C, AEE_C])
    ax_latency.set_xticklabels(["Latency", "AEE"], fontsize=FS_TICK)
    ax_latency.tick_params(axis="x", length=0, pad=2.5)
    ax_latency.set_ylabel("Per-frame latency (ms)", fontsize=FS_LAB)
    ax_latency.yaxis.set_label_coords(-0.225, 0.5)
    ax_latency.spines["right"].set_visible(False)

    # ``twinx`` shares its x locator with ``ax_latency``. Hide only this
    # axis's labels so the primary axis retains the Latency/AEE group labels.
    ax_aee.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_aee.set_ylabel("AEE", fontsize=FS_LAB, rotation=-90, va="bottom")
    ax_aee.yaxis.set_label_coords(1.22, 0.5)
    ax_aee.spines["left"].set_visible(False)
    ax_aee.spines["bottom"].set_visible(False)

    line_right = GAP_L + 0.55 * (GAP_R - GAP_L)
    ax_latency.plot(
        [AX_LO, line_right],
        [reference_ms, reference_ms],
        linestyle=(0, (4, 2)),
        color="#b23b3b",
        linewidth=1.1,
        zorder=4,
    )
    ax_latency.annotate(
        f"{reference_ms:.2f} ms\nideal",
        (GAP_L + 0.02, reference_ms),
        xytext=(1, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=FS_REF,
        color="#b23b3b",
        linespacing=1.05,
    )


def relative_tick(value: float, _: int) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def draw_relative_throughput(fig: plt.Figure) -> None:
    """Draw raw and effective throughput relative to dense raw throughput."""
    ax = fig.add_axes([RX0, PB, PW, PH])
    group_centers = (0.0, 1.0)
    bar_width = 0.24
    offsets = {
        system: (index - 1) * bar_width for index, system in enumerate(ORDER)
    }
    for center, values in zip(
        group_centers, (raw_relative, effective_relative), strict=True
    ):
        for system in ORDER:
            add_bar(
                ax,
                center + offsets[system],
                values[system],
                bar_width,
                STYLE_KEY[system],
                linewidth=0.8,
            )

    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(FuncFormatter(relative_tick))
    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Raw", "Effective"], fontsize=FS_TICK)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.5)
    ax.set_ylabel("Relative throughput", fontsize=FS_LAB)
    ax.yaxis.set_label_coords(-0.25, 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def shared_legend(fig: plt.Figure) -> None:
    handles = legend_handles([STYLE_KEY[system] for system in ORDER])
    fig.legend(
        handles=handles,
        labels=[LABEL[system] for system in ORDER],
        fontsize=FS_LEG,
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.01),
        handlelength=1.2,
        columnspacing=1.1,
        handletextpad=0.4,
    )


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw_latency_and_aee(fig)
    draw_relative_throughput(fig)
    shared_legend(fig)
    fig.text(
        LX0 + PW / 2,
        0.015,
        "(a) Latency and accuracy",
        ha="center",
        va="bottom",
        fontsize=FS_CAP,
    )
    fig.text(
        RX0 + PW / 2,
        0.015,
        "(b) Relative throughput",
        ha="center",
        va="bottom",
        fontsize=FS_CAP,
    )
    output = os.path.join(HERE, "gap.pdf")
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
