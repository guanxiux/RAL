#!/usr/bin/env python3
"""fig:char-bound (Insight 4): the two per-layer fixed costs.  (a) warp stall reasons of the compute kernel with no
work (n_W = 0: launch and exit) and with a single tile (n_W = 1: one tile's dependent chain), the median share over
the representative layers (Nsight Compute, warm caches); the launch times and the intercept b are in the text.
(b) construction time over the per-length-best convolution time of the whole model, against the active ratio of
synthetic clustered masks.  Reads char_intercept.csv and char_construction.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import CFG, FS_LAB, DEV_LS, DEV_MK, FIG_W, FS_CAP, FS_LEG, FS_NOTE, INK, MUTED, PREC, plt, read, save, tidy

# stall reasons grouped by cause; only memory latency (the cost a worklist could add) is drawn in a strong color
CAUSE = (("param", ("imc_miss",), "constant-cache miss", "#a9c8e8"),
         ("sync", ("barrier", "membar"), "barrier, memory fence", "#f2b98f"),
         ("pipe", ("short_scoreboard", "wait"), "on-chip latency", "#f0d98a"),
         ("mem", ("long_scoreboard",), "DRAM/L2 latency", "#127a55"),
         ("other", ("other",), "other", "#e2e2de"))
KEYS = tuple(k for c in CAUSE for k in c[1])
GROUPS = (("RTX 3080", "FP32"), ("RTX 3080", "FP16"), ("AGX Orin", "FP32"), ("AGX Orin", "FP16"))
POINTS = (("nw0", "no work"), ("nw1", "1 tile"))

ic = read("char_intercept.csv")
cs = read("char_construction.csv")
fig = plt.figure(figsize=(FIG_W, 2.25))
B, T = 0.13, 0.80
ax = fig.add_axes([0.225, B, 0.38, T - B])
axb = fig.add_axes([0.75, B, 0.235, T - B])

ys, labels, y = [], [], 0.0

for dev, prec in GROUPS:
    for pt, plabel in POINTS:
        rows = [r for r in ic if r["device"] == dev and r["dtype"] == prec and r["point"] == pt]
        share = {k: float(np.median([float(r[k]) for r in rows])) for k in KEYS}
        tot = sum(share.values())
        left = 0.0
        for _, keys, _, color in CAUSE:
            w = sum(share[k] for k in keys) / tot
            ax.barh(y, w, 0.78, left=left, color=color, edgecolor="white", linewidth=0.4, zorder=3)
            left += w
        us = float(np.median([float(r["us"]) for r in rows]))
        ys.append(y)
        labels.append(f"{plabel}, {us:.1f} µs")
        y -= 1.0
    b = float(np.median([float(r["intercept_us"]) for r in ic
                         if r["device"] == dev and r["dtype"] == prec and r["point"] == "nw1"]))
    ax.text(-0.52, y + 2.8, f"{'RTX 3080' if dev == 'RTX 3080' else 'AGX Orin'} {prec}:  intercept $b$ = {b:.1f} µs",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=FS_LAB, color=INK)
    y -= 1.5
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=FS_LAB)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["0", "50", "100"])
ax.set_ylim(y + 1.95, 1.2)
ax.set_xlabel("(a) Share of stall cycles (%)", fontsize=FS_CAP)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c[3]) for c in CAUSE], [c[2] for c in CAUSE],
           loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=FS_LEG, handlelength=0.9,
           handleheight=0.8, columnspacing=0.8, handletextpad=0.35)

for dev, prec in GROUPS:
    d = sorted((r for r in cs if r["device"] == dev and r["dtype"] == prec), key=lambda r: float(r["alpha"]))
    a = [float(r["alpha"]) for r in d]
    v = [100 * float(r["share"]) for r in d]
    axb.plot(a, v, color=PREC[prec], ls=DEV_LS[dev], marker=DEV_MK[dev], ms=2.2, lw=1.0, zorder=3,
             label=f"{'3080' if dev == 'RTX 3080' else 'Orin'} {prec}")
axb.set_xscale("log")
axb.set_xlim(0.008, 1.2)
axb.set_xticks([0.01, 0.1, 1])
axb.set_xticklabels(["0.01", "0.1", "1"])
axb.minorticks_off()
axb.set_ylim(0, 66)
axb.set_yticks([0, 10, 20, 30, 40])
axb.set_ylabel("Construction / convolution (%)")
axb.set_xlabel(r"(b) Active ratio $\alpha$", fontsize=FS_CAP)
tidy(axb)
axb.legend(loc="upper right", bbox_to_anchor=(1.04, 1.04), frameon=False, fontsize=FS_LEG, handlelength=1.3,
           handletextpad=0.35, labelspacing=0.25, borderaxespad=0.1)
save(fig, "char_bound.pdf")
