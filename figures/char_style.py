"""Shared style of the Characterization figures (plot_char_*.py): fonts, sizes, and the color roles.

Color roles (validated with the dataviz skill's validate_palette.js on a white surface, all pairs):
  precision   FP32 #1f7fa3, FP16 #c4553a            device is the line style / marker (RTX 3080 solid circle,
                                                      AGX Orin dashed square), never a second color
  configuration (Insight 1)   per-length best = the WISEConv color, best single = the gather-scatter violet,
                              dense-tuned = a neutral gray reference
  stall reasons (Insight 4)   the reference categorical order of the dataviz skill (adjacent pairs validated), with
                              "other" in neutral gray
"""
from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from bar_patterns import COLORS, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))

for directory in ("/usr/share/texmf/fonts/opentype/public/tex-gyre", "/usr/share/fonts/opentype/public/tex-gyre",
                  "/usr/local/texlive"):
    for path in glob.glob(os.path.join(directory, "**", "texgyretermes-*.otf"), recursive=True):
        fm.fontManager.addfont(path)

FIG_W = 3.334  # acmart sigplan \columnwidth (240.945 pt)
FS_TICK, FS_LAB, FS_CAP, FS_NOTE, FS_LEG = 7.0, 7.5, 8.0, 6.8, 8.0  # floor: ticks 7 pt, notes 6.8 pt; legends 8 pt
GRID = "#e6e6e3"
REF = "#b9b9b5"  # reference lines (proportional diagonal)

PREC = {"FP32": "#1f7fa3", "FP16": "#c4553a"}
DEV_LS = {"RTX 3080": "-", "AGX Orin": (0, (3.2, 1.6))}
DEV_MK = {"RTX 3080": "o", "AGX Orin": "s"}
CFG = {"best": COLORS["wiseconv"], "single": COLORS["tile_skip"], "tuned": "#a9a9a4"}
STALL = {"imc_miss": "#2a78d6", "barrier": "#eb6834", "long_scoreboard": "#1baf7a", "short_scoreboard": "#eda100",
         "wait": "#e87ba4", "membar": "#008300", "other": "#cfcfcb"}
STALL_LABEL = {"imc_miss": "const. cache miss", "barrier": "barrier", "long_scoreboard": "long scoreboard",
               "short_scoreboard": "short scoreboard", "wait": "fixed-latency wait", "membar": "memory barrier",
               "other": "other"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK, "hatch.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LAB,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2.0, "legend.fontsize": FS_LEG,
    "lines.linewidth": 1.1, "lines.markersize": 2.6,
})


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


__all__ = ["HERE", "CFG", "DEV_LS", "DEV_MK", "FIG_W", "FS_CAP", "FS_LAB", "FS_LEG", "FS_NOTE", "FS_TICK", "GRID", "INK",
           "MUTED", "PREC", "REF", "STALL", "STALL_LABEL", "plt", "read", "save", "tidy"]
