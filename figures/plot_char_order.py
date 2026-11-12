#!/usr/bin/env python3
"""fig:char-order (Insight 3): the same worklist in four orders, same length and tile.  (a) time relative to the
construction order over every layer x captured frame (box: 25th-75th percentile, whiskers: 5th-95th, tick: median),
per GPU and precision.  (b) DRAM read bytes and (c) L1 hit rate of the representative layers (Nsight Compute, cold
L2; one dot per layer, GPU and precision) relative to the construction order.  Reads char_order.csv and
char_order_counters.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_LEG, FS_NOTE, INK, MUTED, plt, read, save, tidy

ORDERS = ("sorted", "shuffled", "uniform")
OC = {"sorted": "#2a78d6", "shuffled": "#eb6834", "uniform": "#1baf7a"}  # dataviz reference slots 1-3 (all pairs)
OLABEL = {"sorted": "sorted globally", "shuffled": "shuffled", "uniform": "uniform positions (control)"}
GROUPS = (("RTX 3080", "FP32"), ("RTX 3080", "FP16"), ("AGX Orin", "FP32"), ("AGX Orin", "FP16"))

t = read("char_order.csv")
c = read("char_order_counters.csv")
fig = plt.figure(figsize=(FIG_W, 1.62))
B, T = 0.26, 0.84
ax = fig.add_axes([0.105, B, 0.50, T - B])
axd = fig.add_axes([0.675, B, 0.12, T - B])
axl = fig.add_axes([0.875, B, 0.12, T - B])

bw = 0.24
for gi, (dev, prec) in enumerate(GROUPS):
    for oi, o in enumerate(ORDERS):
        r = next(x for x in t if x["device"] == dev and x["dtype"] == prec and x["order"] == o)
        q = {k: float(r[k]) for k in ("p5", "p25", "p50", "p75", "p95")}
        x = gi + (oi - 1) * (bw + 0.03)
        ax.plot([x, x], [q["p5"], q["p25"]], color=OC[o], lw=0.7, zorder=3)
        ax.plot([x, x], [q["p75"], q["p95"]], color=OC[o], lw=0.7, zorder=3)
        ax.add_patch(plt.Rectangle((x - bw / 2, q["p25"]), bw, max(q["p75"] - q["p25"], 0.002), facecolor=OC[o],
                                   edgecolor="none", alpha=0.9, zorder=4))
        ax.plot([x - bw / 2, x + bw / 2], [q["p50"], q["p50"]], color="white", lw=0.9, zorder=5)
ax.axhline(1.0, color=MUTED, lw=0.6, zorder=2)
ax.set_xlim(-0.5, len(GROUPS) - 0.5)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([f"{'3080' if d == 'RTX 3080' else 'Orin'}\n{p}" for d, p in GROUPS], linespacing=0.95)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0.96, 1.34)
ax.set_yticks([1.0, 1.1, 1.2, 1.3])
ax.set_ylabel("Time (×)")
ax.set_xlabel("(a) Time, every layer × frame", fontsize=FS_CAP)
tidy(ax)


def strip(axis, key, transform, ylabel, ylim, yticks, label):
    rng = np.random.default_rng(3)
    for oi, o in enumerate(ORDERS):
        base = {(x["device"], x["dtype"], x["layer"]): float(x[key]) for x in c if x["order"] == "real"}
        v = [transform(float(x[key]), base[(x["device"], x["dtype"], x["layer"])]) for x in c if x["order"] == o]
        axis.scatter(oi + rng.uniform(-0.22, 0.22, len(v)), v, s=4.0, color=OC[o], linewidths=0, alpha=0.85, zorder=3)
    axis.set_xlim(-0.6, 2.6)
    axis.set_xticks([])
    axis.set_ylim(*ylim)
    axis.set_yticks(yticks)
    axis.set_xlabel(label, fontsize=FS_CAP, linespacing=1.0)
    tidy(axis)


strip(axd, "dram", lambda v, b: v, None, (0.8, 4.4), [1, 2, 3, 4], "(b) DRAM\nbytes (×)")
axd.axhline(1.0, color=MUTED, lw=0.6, zorder=2)
strip(axl, "l1_hit", lambda v, b: v - b, None, (-36, 10), [-30, -20, -10, 0], "(c) L1 hit\nrate (Δ pp)")
axl.axhline(0.0, color=MUTED, lw=0.6, zorder=2)
fig.text(0.01, 0.985, "vs. construction order:", ha="left", va="top", fontsize=FS_LEG, color=MUTED)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=OC[o]) for o in ORDERS], [OLABEL[o] for o in ORDERS], loc="upper left",
           bbox_to_anchor=(0.24, 1.035), ncol=3, frameon=False, fontsize=FS_LEG, handlelength=0.9, handleheight=0.8, columnspacing=0.9,
           handletextpad=0.4)
save(fig, "char_order.pdf")
