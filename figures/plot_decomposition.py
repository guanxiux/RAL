#!/usr/bin/env python3
"""fig:decomposition: per-frame YOLOv8m latency by stage on RTX 3080 (reads decomposition.csv).

Horizontal stacked bars, one per path, grouped by batch and precision; the total is printed at the end of each bar.
Stage colors are their own roles (not the path colors): convolution dark slate, construction bright yellow (far apart in
lightness: the two are always adjacent), gather/scatter
sea green with the gather-scatter hatch, other operators warm gray, GPU idle the wasted gray with dots.
"""
from __future__ import annotations

from matplotlib.patches import Patch

from bar_patterns import WASTED_COLOR, WASTED_HATCH, WASTED_INK
from char_style import FIG_W, INK, plt, read, save, tidy

FS_TICK, FS_NOTE, FS_LAB, FS_LEG = 7.3, 6.8, 7.5, 8.0  # larger than the shared sizes: the bars leave room

STAGES = [("construction", "construction", "#f2c14e", ""), ("conv", "convolution", "#2f4a68", ""),
          ("gather", "gather/scatter", "#5e9e8a", "\\\\\\\\"), ("other", "other ops", "#cfcac2", ""),
          ("idle", "GPU idle", WASTED_COLOR, WASTED_HATCH)]
INKS = {"gather": "#ffffff", "idle": WASTED_INK}
NAMES = {"dense": "Dense", "tile_skip": "Tile skipping", "gather_scatter": "Gather-scatter", "wiseconv": "WISEConv"}
H = 0.74  # bar height

rows = read("decomposition.csv")
PANELS = [("fp32", 14.0, [0, 2, 4, 6, 8, 10, 12, 14]), ("fp16", 4.4, [0, 1, 2, 3, 4])]
nrows = {d: sum(1 for r in rows if r["dtype"] == d) + 0.5 for d, _, _ in PANELS}
fig = plt.figure(figsize=(FIG_W, 2.95))
L, R, T, B, VG = 0.255, 0.97, 0.83, 0.105, 0.12
unit = (T - B - VG) / sum(nrows.values())
top = T
for pi, (dtype, xmax, xt) in enumerate(PANELS):
    h = nrows[dtype] * unit
    ax = fig.add_axes([L, top - h, R - L, h])
    top -= h + VG
    tidy(ax, "x")
    y, ticks, labels, group_mid = 0.0, [], [], []
    for gi, batch in enumerate((1, 4)):
        rs = [r for r in rows if int(r["batch"]) == batch and r["dtype"] == dtype]
        y0 = y
        for r in rs:
            left = 0.0
            for key, _, color, hatch in STAGES:
                w = float(r[f"{key}_ms"])
                if w <= 0:
                    continue
                ax.barh(y, w, H, left=left, color=color, lw=0, zorder=3)
                if hatch:
                    ax.barh(y, w, H, left=left, color="none", edgecolor=INKS[key], hatch=hatch, lw=0, zorder=3.1)
                left += w
            ours = r["path"] == "wiseconv"
            ax.text(left + xmax * 0.011, y, f"{float(r['total_ms']):.2f}", va="center", ha="left", fontsize=FS_NOTE,
                    color=INK, fontweight="bold" if ours else "normal")
            ticks.append(y)
            labels.append(NAMES[r["path"]])
            y += 1
        group_mid.append(((y0 + y - 1) / 2, f"b{batch}"))
        if gi == 0:
            ax.axhline(y - 0.5 + 0.25, color="#cfcfca", lw=0.6, zorder=1)
        y += 0.5
    ax.set_ylim(y - 0.5 - 0.3, -0.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    for t in ax.get_yticklabels():
        if t.get_text() == "WISEConv":
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    for mid, text in group_mid:
        ax.text(-0.30, mid, text, transform=ax.get_yaxis_transform(), ha="center", va="center", rotation=90,
                fontsize=FS_TICK, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_xticks(xt)
    ax.spines["left"].set_visible(False)
    ax.text(0.0, 1.02, dtype.upper(), transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_LAB,
            fontweight="bold")
    if pi == len(PANELS) - 1:
        ax.set_xlabel("Latency per frame (ms)", fontsize=FS_LAB, labelpad=1.5)
handles = [Patch(facecolor=c, edgecolor=INKS.get(k, "none"), hatch=h, lw=0) for k, _, c, h in STAGES]
fig.legend(handles, [n for _, n, _, _ in STAGES], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), fontsize=FS_LEG, handlelength=1.0, columnspacing=1.0, handletextpad=0.3, labelspacing=0.25)
save(fig, "decomposition.pdf")
