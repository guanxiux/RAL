#!/usr/bin/env python3
"""Render the fig:gap motivation figure from gap_stats.csv.

Produces a SINGLE self-contained figure (gap.pdf) sized to the paper's
\\columnwidth so LaTeX includes it at scale 1.0 (no downscaling, so the fonts
render at the sizes set here). It composes two equal-area panels with baked-in
"(a)/(b)" sub-captions and one shared legend centred above them:

  (a) Two groups sharing one panel: per-frame latency (ms, left broken axis,
      dashed sparsity-scaled floor) and PCKh accuracy (%, right axis). Only the
      left latency axis is broken (Gather-scatter overruns the ms scale); the
      right accuracy axis is a plain 0-100 scale, so the dashed floor line spans
      the latency group alone.
  (b) Raw throughput T and effective throughput eta*T (GFLOPS) for the same
      three systems.

Colours/textures are consistent per baseline. Reads only the CSV (no dependency
on the gitignored logs); rerun gap_stats.py to refresh the CSV.
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from bar_patterns import (
    add_bar,
    legend_handles,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "gap_stats.csv")

# Match the paper body font. IEEEtran typesets in Nimbus Roman No9 L (a Times
# clone); TeX Gyre Termes is its OTF descendant, so registering it here makes
# the figure text metrically match the body instead of falling back to the
# heavier DejaVu Serif.
_TERMES = []
for _d in ("/usr/share/texmf/fonts/opentype/public/tex-gyre",
           "/usr/share/fonts/opentype/public/tex-gyre",
           "/usr/local/texlive"):
    _TERMES += glob.glob(os.path.join(_d, "**", "texgyretermes-*.otf"),
                         recursive=True)
for _p in sorted(set(_TERMES)):
    fm.fontManager.addfont(_p)

# columnwidth = 252pt = 3.49in; design at that width so include is 1:1.
FIG_W, FIG_H = 3.49, 1.82

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.7,
    "hatch.linewidth": 0.38,
})
FS_TICK, FS_LAB, FS_LEG, FS_CAP, FS_FLOOR = 6.5, 7.0, 7.0, 7.5, 6.0

ORDER = ["Dense", "Tile skipping", "Gather-scatter"]
STYLE_KEY = {
    "Dense": "dense",
    "Tile skipping": "tile_skip",
    "Gather-scatter": "gather_scatter",
}
# Legend shows the concrete implementations; the paradigm-to-impl mapping is
# established in the gap section's text, so it needs no restating in the caption.
LABEL = {"Dense": "Dense", "Tile skipping": "DeltaCNN", "Gather-scatter": "DynConv"}

rows = {r["system"]: r for r in csv.DictReader(open(CSV))}
lat = {s: float(rows[s]["latency_ms_mean"]) for s in ORDER}
# throughput stored in TFLOPS; report in GFLOPS (x1000)
traw = {s: float(rows[s]["raw_tflops"]) * 1e3 for s in ORDER}
teff = {s: float(rows[s]["eff_tflops"]) * 1e3 for s in ORDER}
pckh = {s: float(rows[s]["pckh"]) for s in ORDER}
floor = float(rows["Dense"]["ideal_floor_ms"])

# --- panel geometry (figure fractions); equal width + equal height => equal area
LX0, RX0, PW = 0.120, 0.645, 0.330        # left/right panel x-origin, shared width
PB, PT = 0.190, 0.895                      # panel band bottom / top
PH = PT - PB
RB, RG = 0.80, 0.05                        # broken-axis: bottom frac, gap frac
BOTH = PH * RB
GAPH = PH * RG


# x-layout of panel (a): two groups of three touching bars (like panel b),
# a gap between them holds the floor label.
BW_A = 0.34
LAT_C, ACC_C = 0.0, 1.85                          # group centres
LAT_X = [LAT_C + (i - 1) * BW_A for i in range(len(ORDER))]
ACC_X = [ACC_C + (i - 1) * BW_A for i in range(len(ORDER))]
GHALF = 1.5 * BW_A                                # half-width of a 3-bar group
GAP_L, GAP_R = LAT_C + GHALF, ACC_C - GHALF       # inter-group gap edges
AX_LO, AX_HI = LAT_C - GHALF - 0.19, ACC_C + GHALF + 0.19
LAT_MID, ACC_MID = LAT_C, ACC_C


def draw_latency(fig):
    """Left panel: latency (left broken axis, dashed floor) + accuracy (right axis)."""
    gs = lat["Gather-scatter"]
    # accuracy axis spans the full panel height (plain 0-100), drawn under the
    # latency axes so its bars sit at their own x with no broken-axis coupling.
    ax_acc = fig.add_axes([LX0, PB, PW, PH], zorder=1)
    ax_bot = fig.add_axes([LX0, PB, PW, BOTH], zorder=2)
    ax_top = fig.add_axes([LX0, PB + BOTH + GAPH, PW, PH - BOTH - GAPH], zorder=2)

    # latency bars (both sub-axes for the broken scale)
    latency_patches = {}
    for a in (ax_bot, ax_top):
        a.patch.set_alpha(0)                 # let the accuracy axis show through
        latency_patches[a] = {}
        for i, s in enumerate(ORDER):
            latency_patches[a][s] = add_bar(
                a, LAT_X[i], lat[s], BW_A, STYLE_KEY[s], linewidth=0.8
            )
        a.set_xlim(AX_LO, AX_HI)
        a.set_xticks([])
    ax_bot.set_ylim(0, 70)
    ax_bot.set_yticks([0, 20, 40, 60])
    ax_top.set_ylim(gs * 0.985, gs * 1.03)
    ax_top.set_yticks([round(gs)])

    # dashed sparsity floor: spans the latency group and a bit into the gap;
    # label sits just past the bars so it reads as attached to the group
    line_r = GAP_L + 0.55 * (GAP_R - GAP_L)
    ax_bot.plot([AX_LO, line_r], [floor, floor],
                ls=(0, (4, 2)), color="#b23b3b", lw=1.1, zorder=4)
    ax_bot.annotate(f"{floor:.1f} ms\nideal", (GAP_L + 0.02, floor),
                    xytext=(1, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FS_FLOOR,
                    color="#b23b3b", linespacing=1.05)

    # accuracy bars on the right axis
    accuracy_patches = {}
    for i, s in enumerate(ORDER):
        accuracy_patches[s] = add_bar(
            ax_acc, ACC_X[i], pckh[s], BW_A, STYLE_KEY[s], linewidth=0.8
        )
    ax_acc.set_xlim(AX_LO, AX_HI)
    ax_acc.set_ylim(0, 130)               # headroom above 100 lowers the bars
    ax_acc.set_yticks([0, 50, 100])
    ax_acc.yaxis.set_label_position("right")
    ax_acc.yaxis.tick_right()
    ax_acc.set_ylabel("PCKh (%)", fontsize=FS_LAB, rotation=-90, va="bottom")
    ax_acc.yaxis.set_label_coords(1.16, 0.5)
    # group labels along the bottom
    ax_acc.set_xticks([LAT_MID, ACC_MID])
    ax_acc.set_xticklabels(["Latency", "Accuracy"], fontsize=FS_TICK)

    for sp in ("top", "left"):
        ax_acc.spines[sp].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_bot.tick_params(labelsize=FS_TICK, pad=1.5)
    ax_top.tick_params(bottom=False, labelbottom=False, labelsize=FS_TICK, pad=1.5)
    ax_acc.tick_params(axis="y", labelsize=FS_TICK, pad=1.5)
    ax_acc.tick_params(axis="x", length=0, pad=2.5)
    ax_bot.set_ylabel("Per-frame latency (ms)", fontsize=FS_LAB)
    ax_bot.yaxis.set_label_coords(-0.175, 0.62)



    # dashed break marks at the left spine boundary between the two sub-axes
    d = 0.013
    kw = dict(transform=fig.transFigure, color="k", lw=0.8, clip_on=False,
              dashes=(2, 1.4))
    for yc in (PB + BOTH, PB + BOTH + GAPH):
        fig.lines.append(plt.Line2D([LX0 - d, LX0 + d], [yc - d, yc + d], **kw))
    return ax_bot


def draw_throughput(fig):
    """Right panel: raw vs effective throughput, grouped by metric."""
    ax = fig.add_axes([RX0, PB, PW, PH])
    gx = np.array([0.0, 1.0])
    bw = 0.24
    offs = {s: (i - 1) * bw for i, s in enumerate(ORDER)}
    patches = []
    for gi, data in enumerate((traw, teff)):
        for s in ORDER:
            patch = add_bar(
                ax,
                gx[gi] + offs[s],
                data[s],
                bw,
                STYLE_KEY[s],
                linewidth=0.8,
            )
            patches.append((patch, STYLE_KEY[s]))
    ax.set_ylim(0, max(traw.values()) * 1.30)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlim(-0.55, gx[-1] + 0.55)
    ax.set_xticks(gx)
    ax.set_xticklabels(["Raw", "Effective"], fontsize=FS_TICK)
    ax.set_ylabel("Throughput (GFLOPS)", fontsize=FS_LAB)
    ax.yaxis.set_label_coords(-0.165, 0.5)
    ax.tick_params(axis="y", labelsize=FS_TICK, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2.5)   # match panel (a): no x tick marks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def shared_legend(fig):
    handles = legend_handles([STYLE_KEY[s] for s in ORDER])
    fig.legend(handles=handles, labels=[LABEL[s] for s in ORDER],
               fontsize=FS_LEG, ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.55, 1.01),
               handlelength=1.2, columnspacing=1.1, handletextpad=0.4)


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    draw_latency(fig)
    draw_throughput(fig)
    shared_legend(fig)
    cx_a = LX0 + PW / 2
    cx_b = RX0 + PW / 2
    fig.text(cx_a, 0.015, "(a) Latency and accuracy", ha="center", va="bottom",
             fontsize=FS_CAP)
    fig.text(cx_b, 0.015, "(b) Throughput", ha="center", va="bottom",
             fontsize=FS_CAP)
    out = os.path.join(HERE, "gap.pdf")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
