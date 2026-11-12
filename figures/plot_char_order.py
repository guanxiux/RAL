#!/usr/bin/env python3
"""fig:char-order (Insight 3): the same worklist in four orders, same length and tile.  (a) time relative to the
tile-local (construction) order over every layer x captured frame (box: 25th-75th percentile with a minimum height so
that a narrow box stays visible, whisker: 5th-95th, dark line: median), per GPU and precision.  (b) DRAM read bytes and (c) L1 hit rate of the representative layers (Nsight Compute, cold
L2; one dot per layer, GPU and precision) relative to the construction order.  Reads char_order.csv and
char_order_counters.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_LAB, FS_LEG, INK, MUTED, plt, read, save, tidy

ORDERS = ("sorted", "shuffled", "uniform")
OC = {"sorted": "#2a78d6", "shuffled": "#eb6834", "uniform": "#1baf7a"}  # dataviz reference slots 1-3 (all pairs)
OLABEL = {"sorted": "sorted globally", "shuffled": "shuffled", "uniform": "uniform positions (control)"}
GROUPS = (("RTX 3080", "FP32"), ("RTX 3080", "FP16"), ("AGX Orin", "FP32"), ("AGX Orin", "FP16"))

t = read("char_order.csv")
c = read("char_order_counters.csv")
fig = plt.figure(figsize=(FIG_W, 1.9))
B, T = 0.215, 0.865
ax = fig.add_axes([0.155, B, 0.43, T - B])
axd = fig.add_axes([0.665, B, 0.115, T - B])
axl = fig.add_axes([0.865, B, 0.115, T - B])

bw = 0.24
for gi, (dev, prec) in enumerate(GROUPS):
    for oi, o in enumerate(ORDERS):
        r = next(x for x in t if x["device"] == dev and x["dtype"] == prec and x["order"] == o)
        q = {k: float(r[k]) for k in ("p5", "p25", "p50", "p75", "p95")}
        x = gi + (oi - 1) * (bw + 0.03)
        ax.plot([x, x], [q["p5"], q["p95"]], color=OC[o], lw=0.8, zorder=3)
        lo, hi = q["p25"], max(q["p75"], q["p25"] + 0.006)  # a minimum height keeps a narrow box visible
        ax.add_patch(plt.Rectangle((x - bw / 2, lo), bw, hi - lo, facecolor=OC[o], edgecolor="none", zorder=4))
        ax.plot([x - bw / 2, x + bw / 2], [q["p50"], q["p50"]], color=INK, lw=1.0, solid_capstyle="butt", zorder=5)
ax.axhline(1.0, color=MUTED, lw=0.6, zorder=2)
ax.set_xlim(-0.5, len(GROUPS) - 0.5)
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([f"{'3080' if d == 'RTX 3080' else 'Orin'}\n{p}" for d, p in GROUPS], linespacing=0.95)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0.96, 1.34)
ax.set_yticks([1.0, 1.1, 1.2, 1.3])
ax.set_yticklabels(["1.0×", "1.1×", "1.2×", "1.3×"])
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


strip(axd, "dram", lambda v, b: v, None, (0.8, 4.4), [1, 2, 3, 4], "(b) DRAM\nbytes")
axd.set_yticklabels(["1×", "2×", "3×", "4×"])
axd.axhline(1.0, color=MUTED, lw=0.6, zorder=2)
strip(axl, "l1_hit", lambda v, b: v - b, None, (-36, 10), [-30, -20, -10, 0], "(c) L1 hit\nrate (pp)")
axl.set_yticklabels(["−30", "−20", "−10", "0"])
axl.axhline(0.0, color=MUTED, lw=0.6, zorder=2)
fig.text(0.04, (B + T) / 2, "Relative to\ntile-local order", rotation=90, ha="center", va="center", fontsize=FS_LAB,
         linespacing=1.0)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=OC[o]) for o in ORDERS], [OLABEL[o] for o in ORDERS], loc="upper center",
           bbox_to_anchor=(0.55, 1.01), ncol=3, frameon=False, fontsize=FS_LEG, handlelength=0.9, handleheight=0.8,
           columnspacing=0.9, handletextpad=0.4)
save(fig, "char_order.pdf")
