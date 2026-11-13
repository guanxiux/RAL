#!/usr/bin/env python3
"""fig:char-cost (Insight 5): the share of layers on which a selection mechanism costs more than following the worklist
length gains.  Per layer of YOLOv8m, the gain is the per-frame time saved by the per-length best over the latency-optimal
single configuration (real trace); the cost is the mechanism's per-layer cost on the same GPU (i5_cost): the
device-updatable node without a switch and with a switch every frame (per layer + per switch), host readback over the
static graph, and the IF conditional node.  One panel per GPU, one bar per precision.  Reads char_gain.csv and
char_cost.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_LEG, FS_NOTE, INK, PREC, plt, read, save, tidy

gain = read("char_gain.csv")
cost = read("char_cost.csv")
ROWS = ("Updatable node", "  + switch every frame", "Host readback", "Conditional node")
H = 0.42  # bar height (a row is 1)

fig = plt.figure(figsize=(FIG_W, 1.45))
B, T = 0.215, 0.86
axes = [fig.add_axes([0.335, B, 0.30, T - B]), fig.add_axes([0.685, B, 0.30, T - B])]
for ax, dev in zip(axes, ("RTX 3080", "AGX Orin")):
    c = {r["mechanism"]: (float(r["us"]), float(r["per_switch_us"])) for r in cost if r["device"] == dev}
    u0, sw = c["Updatable node"]
    us = (u0, u0 + sw, c["Host readback"][0], c["Conditional node"][0])
    for k, prec in enumerate(("FP32", "FP16")):
        g = np.array([float(r["gain_us"]) for r in gain if r["device"] == dev and r["dtype"] == prec])
        for i, u in enumerate(us):
            share = 100.0 * np.mean(g < u)
            y = i + (k - 0.5) * H
            ax.barh(y, share, H, color=PREC[prec], lw=0, zorder=3)
            ax.text(share + 2.0, y, (f"{share:.1f}" if share < 99.95 else "100"), ha="left", va="center", fontsize=FS_NOTE - 0.3, color=INK)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS if dev == "RTX 3080" else [])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 122)
    ax.set_xticks([0, 50, 100])
    ax.set_title(f"{'(a)' if dev == 'RTX 3080' else '(b)'} {dev}", fontsize=FS_CAP, pad=2.0)
    ax.spines["left"].set_visible(False)
    tidy(ax, grid_axis="x")
fig.text((0.335 + 0.985) / 2, 0.0, "Layers where the cost exceeds the gain (%)", ha="center", va="bottom",
         fontsize=FS_CAP)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=PREC[p]) for p in ("FP32", "FP16")], ["FP32", "FP16"],
           loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=FS_LEG, handlelength=0.9,
           handleheight=0.8, columnspacing=0.8, handletextpad=0.35)
save(fig, "char_cost.pdf")
