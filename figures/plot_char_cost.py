#!/usr/bin/env python3
"""fig:char-cost (Insight 5): the gain of following the worklist length against the cost of each mechanism that
follows it, per layer.  CDF over the layers of YOLOv8m of the per-frame gain of the per-length best over the best
single configuration (real trace), per precision; vertical lines: the per-layer cost of each selection mechanism on
the same GPU (i5_cost; host readback over the static graph, device-updatable without switches).  Reads char_gain.csv
and char_cost.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_LEG, FS_NOTE, INK, MUTED, PREC, plt, read, save, tidy

gain = read("char_gain.csv")
cost = read("char_cost.csv")
MECH = (("Updatable node", "updatable", "-"), ("Host readback", "readback", (0, (3, 1.5))),
        ("Conditional node", "conditional", (0, (1, 1.2))))

fig = plt.figure(figsize=(FIG_W, 1.62))
B, T = 0.23, 0.76
axes = [fig.add_axes([0.115, B, 0.40, T - B]), fig.add_axes([0.57, B, 0.40, T - B])]
for ax, dev in zip(axes, ("RTX 3080", "AGX Orin")):
    for prec in ("FP32", "FP16"):
        g = np.sort([max(float(r["gain_us"]), 1e-2) for r in gain if r["device"] == dev and r["dtype"] == prec])
        ax.step(np.concatenate([[g[0]], g]), np.concatenate([[0], np.arange(1, len(g) + 1) / len(g)]), where="post",
                color=PREC[prec], lw=1.2, zorder=4)
    for name, short, ls in MECH:
        us = next(float(r["us"]) for r in cost if r["device"] == dev and r["mechanism"] == name)
        ax.axvline(us, color=INK if short == "updatable" else MUTED, ls=ls, lw=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xticklabels(["0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(f"{'(a)' if dev == 'RTX 3080' else '(b)'} {dev}: µs per layer", fontsize=FS_CAP)
    tidy(ax, grid_axis="x")
axes[0].set_ylabel("CDF over layers")
axes[1].set_yticklabels([])
hg = [plt.Line2D([], [], color=PREC[p], lw=1.4) for p in ("FP32", "FP16")]
hm = [plt.Line2D([], [], color=INK if s == "updatable" else MUTED, ls=ls, lw=0.9) for _, s, ls in MECH]
blank = plt.Line2D([], [], color="none")
# matplotlib fills legend columns first: rows = (gains, mechanism costs)
fig.legend([hg[0], hm[0], hg[1], hm[1], blank, hm[2]],
           ["gain, FP32", "cost: updatable node", "gain, FP16", "host readback", "", "conditional node"],
           loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=3, frameon=False, fontsize=FS_LEG, handlelength=1.8,
           columnspacing=1.0, handletextpad=0.4)
save(fig, "char_cost.pdf")
