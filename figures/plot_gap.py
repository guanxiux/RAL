#!/usr/bin/env python3
"""Render the Background sparsity-to-latency example (fig:gap) from gap_stats.csv.

Two equal-area panels, the three execution paths named on the x axis (no path legend):

* (a) end-to-end latency per batch step; the dashed line ("Ideal") is dense latency scaled by the active ratio of the
  MACs (the proportional reference).
* (b) throughput normalized to dense raw throughput, stacked: the useful share (the path's color) under the wasted
  share (gray dots); the bar height is the raw throughput, the number the useful throughput.

The plotting script reads only the CSV; run gap_stats.py first.
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, MultipleLocator

from bar_patterns import INK, MUTED, WASTED_COLOR, add_bar, add_wasted, wasted_handle

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "gap_stats.csv")

for directory in ("/usr/share/texmf/fonts/opentype/public/tex-gyre", "/usr/share/fonts/opentype/public/tex-gyre",
                  "/usr/local/texlive"):
    for path in glob.glob(os.path.join(directory, "**", "texgyretermes-*.otf"), recursive=True):
        fm.fontManager.addfont(path)

# acmart's sigplan column is 240.945 TeX pt (3.334 in): emit at that width so LaTeX does not rescale the type.
FIG_W, FIG_H = 3.334, 1.32
FS_TICK, FS_LAB, FS_CAP, FS_NOTE = 6.5, 7.0, 7.5, 6.0
GRID = "#e6e6e3"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK, "hatch.linewidth": 0.5,
})

ORDER = ["dense", "tile_skip", "gather_scatter"]
TICK = {"dense": "Dense", "tile_skip": "Tile\nskipping", "gather_scatter": "Gather-\nscatter"}

rows = {r["backend"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))}
val = lambda key, b: float(rows[b][key])
reference_ms = val("proportional_reference_ms", "dense")

# panel geometry (figure fractions): equal areas, a gutter wide enough for panel (b)'s y label
LX0, RX0, PW = 0.105, 0.625, 0.355
PB, PT = 0.30, 0.95
PH = PT - PB
XS = [0.0, 1.0, 2.0]
BW = 0.56


def style_axes(ax, ylabel, xmax=2.5):
    ax.set_xlim(-0.5, xmax)
    ax.set_xticks(XS)
    ax.set_xticklabels([TICK[b] for b in ORDER], fontsize=FS_TICK, linespacing=0.95)
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.5, length=2.0, width=0.5)
    ax.set_ylabel(ylabel, fontsize=FS_LAB, labelpad=2.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def draw_latency(fig):
    ax = fig.add_axes([LX0, PB, PW, PH])
    for x, b in zip(XS, ORDER):
        t = val("latency_ms", b)
        add_bar(ax, x, t, BW, b)
        ax.text(x, t + 1.0, f"{t:.1f}", ha="center", va="bottom", fontsize=FS_NOTE, color=INK)
    # the proportional reference: a dashed line, labeled in the free margin right of the last bar
    ax.axhline(reference_ms, color=MUTED, linestyle=(0, (3.5, 2)), linewidth=0.8, zorder=4)
    ax.text(3.08, reference_ms + 0.9, f"Ideal\n{reference_ms:.1f}", ha="right", va="bottom", fontsize=FS_NOTE,
            color=MUTED, linespacing=0.95)
    ax.set_ylim(0, 50)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    style_axes(ax, "Latency (ms)", xmax=3.1)


def draw_throughput(fig):
    ax = fig.add_axes([RX0, PB, PW, PH])
    for x, b in zip(XS, ORDER):
        eff, waste = val("effective_throughput_vs_dense", b), val("wasted_throughput_vs_dense", b)
        add_bar(ax, x, eff, BW, b)
        if waste > 1e-6:
            add_wasted(ax, x, waste, BW, bottom=eff)
        # the useful value at the top of the useful share (inside the wasted share where there is one)
        ax.text(x, eff + 0.02, f"{eff:.2f}", ha="center", va="bottom", fontsize=FS_NOTE, color=INK, zorder=6,
                path_effects=[withStroke(linewidth=1.6, foreground=WASTED_COLOR)] if waste > 1e-6 else None)
    # the wasted share's key, in the free area right of the dense bar (the colored share is the useful work)
    ax.legend([wasted_handle()], ["wasted"], loc="center", bbox_to_anchor=(1.45, 0.86), bbox_transform=ax.transData,
              frameon=False, fontsize=FS_NOTE, handlelength=1.5, handleheight=0.9, handletextpad=0.35, borderpad=0.1)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    style_axes(ax, "Rel. throughput")


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw_latency(fig)
    draw_throughput(fig)
    for x0, cap in ((LX0, "(a) Latency per batch"), (RX0, "(b) Relative throughput")):
        fig.text(x0 + PW / 2, 0.012, cap, ha="center", va="bottom", fontsize=FS_CAP)
    out = os.path.join(HERE, "gap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
