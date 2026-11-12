#!/usr/bin/env python3
"""fig:char-cost (Insight 5): the gain of following the worklist length against the cost of each mechanism that
follows it, per layer.  CDF over the layers of YOLOv8m of the per-frame gain of the per-length best over the best
single configuration (real trace), per precision (lines labeled directly); against the per-layer cost of each selection
mechanism on the same GPU (i5_cost): the device-updatable node as a band from no switch to a switch every frame
(per layer + per switch), host readback over the static graph and the IF conditional node as lines, labeled at the
top.  Reads char_gain.csv and char_cost.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_NOTE, INK, MUTED, PREC, plt, read, save, tidy

gain = read("char_gain.csv")
cost = read("char_cost.csv")
BAND = "#d9d9d5"

fig = plt.figure(figsize=(FIG_W, 1.62))
B, T = 0.23, 0.86
axes = [fig.add_axes([0.115, B, 0.40, T - B]), fig.add_axes([0.57, B, 0.40, T - B])]
for ax, dev in zip(axes, ("RTX 3080", "AGX Orin")):
    c = {r["mechanism"]: (float(r["us"]), float(r["per_switch_us"])) for r in cost if r["device"] == dev}
    u0, sw = c["Updatable node"]
    ax.axvspan(u0, u0 + sw, color=BAND, lw=0, zorder=1)
    ax.text(np.sqrt(u0 * (u0 + sw)), 1.04, "updatable\nnode", ha="center", va="bottom", fontsize=FS_NOTE - 0.5,
            color=INK, linespacing=0.95, zorder=5, clip_on=False)
    rb, cn = c["Host readback"][0], c["Conditional node"][0]
    for us in (rb, cn):
        ax.axvline(us, color=MUTED, lw=0.8, zorder=2)
    ax.text(np.sqrt(rb * cn), 1.04, "readback,\nconditional", ha="center", va="bottom", fontsize=FS_NOTE - 0.5,
            color=MUTED, linespacing=0.95, clip_on=False)
    for prec in ("FP32", "FP16"):
        g = np.sort([max(float(r["gain_us"]), 1e-2) for r in gain if r["device"] == dev and r["dtype"] == prec])
        y = np.arange(1, len(g) + 1) / len(g)
        ax.step(np.concatenate([[g[0]], g]), np.concatenate([[0], y]), where="post", color=PREC[prec], lw=1.2,
                zorder=4)
        q = 0.75 if prec == "FP16" else 0.35
        x = np.quantile(g, q)
        ax.text(x / 1.2 if prec == "FP16" else x * 1.2, q, f"gain, {prec}", ha="right" if prec == "FP16" else "left",
                va="center", fontsize=FS_NOTE - 0.5, color=PREC[prec], zorder=5)
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
save(fig, "char_cost.pdf")
