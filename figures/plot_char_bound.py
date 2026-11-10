#!/usr/bin/env python3
"""fig:char-bound (Insight 4): the two per-layer fixed costs.  (a) warp stall reasons of the compute kernel with no
work (n_W = 0: launch and exit) and with a single tile (n_W = 1: one tile's dependent chain), the median share over
the representative layers (Nsight Compute, warm caches), with the median time per launch (graph timing).
(b) construction time over the per-length-best convolution time of the whole model, against the active ratio of
synthetic clustered masks.  Reads char_intercept.csv and char_construction.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import (DEV_LS, DEV_MK, FIG_W, FS_CAP, FS_LEG, FS_NOTE, INK, MUTED, PREC, STALL, STALL_LABEL, plt,
                        read, save, tidy)

KEYS = ("imc_miss", "barrier", "long_scoreboard", "short_scoreboard", "wait", "membar", "other")
GROUPS = (("RTX 3080", "FP32"), ("RTX 3080", "FP16"), ("AGX Orin", "FP32"), ("AGX Orin", "FP16"))
POINTS = (("nw0", "no work"), ("nw1", "1 tile"))

ic = read("char_intercept.csv")
cs = read("char_construction.csv")
fig = plt.figure(figsize=(FIG_W, 2.10))
B, T = 0.15, 0.74
ax = fig.add_axes([0.245, B, 0.32, T - B])
axb = fig.add_axes([0.715, B, 0.27, T - B])

ys, labels, y = [], [], 0.0
for dev, prec in GROUPS:
    for pt, plabel in POINTS:
        rows = [r for r in ic if r["device"] == dev and r["dtype"] == prec and r["point"] == pt]
        share = {k: float(np.median([float(r[k]) for r in rows])) for k in KEYS}
        tot = sum(share.values())
        left = 0.0
        for k in KEYS:
            w = share[k] / tot
            ax.barh(y, w, 0.78, left=left, color=STALL[k], edgecolor="white", linewidth=0.4, zorder=3)
            left += w
        us = float(np.median([float(r["us"]) for r in rows]))
        ax.text(1.02, y, f"{us:.1f}", ha="left", va="center", fontsize=FS_NOTE, color=INK)
        ys.append(y)
        labels.append(plabel)
        y -= 1.0
    ax.text(-0.40, y + 1.5, f"{'3080' if dev == 'RTX 3080' else 'Orin'}\n{prec}", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=FS_NOTE, color=INK, linespacing=0.95)
    y -= 0.35
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=FS_NOTE)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["0", "50", "100"])
ax.set_ylim(y + 0.6, 0.6)
ax.text(1.02, 0.95, "µs", ha="left", va="center", fontsize=FS_NOTE, color=MUTED)
ax.set_xlabel("(a) Share of stall cycles (%)", fontsize=FS_CAP)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=STALL[k]) for k in KEYS], [STALL_LABEL[k] for k in KEYS],
           loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, fontsize=FS_LEG, handlelength=0.9,
           handleheight=0.8, columnspacing=0.8, handletextpad=0.35)

for dev, prec in GROUPS:
    d = sorted((r for r in cs if r["device"] == dev and r["dtype"] == prec), key=lambda r: float(r["alpha"]))
    a = [float(r["alpha"]) for r in d]
    axb.plot(a, [100 * float(r["share"]) for r in d], color=PREC[prec], ls=DEV_LS[dev], marker=DEV_MK[dev], ms=2.2,
             lw=1.0, zorder=3)
axb.set_xscale("log")
axb.set_xlim(0.008, 1.2)
axb.set_xticks([0.01, 0.1, 1])
axb.set_xticklabels(["0.01", "0.1", "1"])
axb.minorticks_off()
axb.set_ylim(0, 42)
axb.set_yticks([0, 10, 20, 30, 40])
axb.set_ylabel("Construction / conv. (%)")
axb.set_xlabel(r"(b) Active ratio $\alpha$", fontsize=FS_CAP)
tidy(axb)
hs = [plt.Line2D([], [], color=PREC[p], lw=1.4) for p in ("FP32", "FP16")] + \
     [plt.Line2D([], [], color=MUTED, ls=DEV_LS[d], marker=DEV_MK[d], ms=2.2, lw=1.0) for d in ("RTX 3080", "AGX Orin")]
axb.legend(hs, ["FP32", "FP16", "3080", "Orin"], loc="lower center", bbox_to_anchor=(0.5, 0.98), frameon=False,
           fontsize=FS_NOTE, ncol=2, handlelength=1.6, columnspacing=0.6, handletextpad=0.25, borderaxespad=0.15)
save(fig, "char_bound.pdf")
