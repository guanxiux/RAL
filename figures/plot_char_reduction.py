#!/usr/bin/env python3
"""fig:char-reduction (Insight 2): a worklist tile against a dense tile of the same configuration on the
representative layers, RTX 3080 (Nsight Compute, cold L2).  (a) global-load sectors per request, dense tile vs
worklist tiles of real frames (range: the 25th-75th percentile frames), FP32 and FP16.  (b) per-tile counts of the
worklist tiles relative to the dense tile, FP32: executed instructions, L2 read sectors, DRAM read bytes of the
features (weights excluded: read once per launch).  Reads char_reduction.csv (char_stats.py)."""
from __future__ import annotations

from char_style import FIG_W, FS_CAP, FS_LEG, FS_NOTE, GRID, INK, MUTED, PREC, plt, read, save, tidy

DEV = "RTX 3080"
rows = [r for r in read("char_reduction.csv") if r["device"] == DEV]
layers = list(dict.fromkeys(r["layer"] for r in rows))
short = {r["layer"]: r["short"].replace("x", "×") for r in rows}
Y = {l: len(layers) - 1 - i for i, l in enumerate(layers)}

fig = plt.figure(figsize=(FIG_W, 1.78))
B, T = 0.155, 0.80
axa = fig.add_axes([0.285, B, 0.30, T - B])
axb = fig.add_axes([0.665, B, 0.32, T - B])

off = {"FP32": 0.14, "FP16": -0.14}
for prec in ("FP32", "FP16"):
    for l in layers:
        g = {r["variant"]: float(r["sectors_per_request"]) for r in rows if r["layer"] == l and r["dtype"] == prec}
        y = Y[l] + off[prec]
        lo, hi = min(g["real_p25"], g["real_p50"], g["real_p75"]), max(g["real_p25"], g["real_p50"], g["real_p75"])
        axa.plot([lo, hi], [y, y], color=PREC[prec], lw=1.0, solid_capstyle="butt", zorder=3)
        axa.plot(g["real_p50"], y, "o", color=PREC[prec], ms=3.0, zorder=4)
        axa.plot(g["dense"], y, "o", mfc="white", mec=PREC[prec], mew=0.8, ms=3.4, zorder=5)
axa.set_yticks(list(Y.values()))
axa.set_yticklabels([short[l] for l in layers], fontsize=FS_NOTE)
axa.set_ylim(-0.6, len(layers) - 0.4)
axa.set_xlim(11, 16.5)
axa.set_xticks([11, 12, 13, 14, 15, 16])
axa.set_xticklabels(["11", "", "13", "", "15", ""])
axa.set_xlabel("(a) Sectors per request", fontsize=FS_CAP)
axa.tick_params(axis="y", length=0)
tidy(axa, grid_axis="x")
h1 = plt.Line2D([], [], marker="o", color="none", mfc="white", mec=INK, mew=0.8, ms=3.4)
h2 = plt.Line2D([], [], marker="o", color=INK, ms=3.0, lw=1.0)
hp = [plt.Line2D([], [], color=PREC[p], lw=2.5) for p in ("FP32", "FP16")]
fig.legend([h1, h2] + hp, ["dense tile", "worklist tiles (p25-p75 frames)", "FP32", "FP16"], loc="upper center",
           bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=FS_LEG, handlelength=1.3, columnspacing=0.9,
           handletextpad=0.35, borderaxespad=0.1)

METRIC = (("inst", "D", "instructions", INK, INK), ("l2", "^", "L2 sectors", MUTED, MUTED),
          ("dram_feature", "o", "DRAM (features)", "white", INK))
moff = {"inst": 0.2, "l2": 0.0, "dram_feature": -0.2}
handles = []
for key, mk, label, fc, ec in METRIC:
    for l in layers:
        v = [float(r[key]) for r in rows if r["layer"] == l and r["dtype"] == "FP32" and r["variant"] != "dense"]
        y = Y[l] + moff[key]
        axb.plot([min(v), max(v)], [y, y], color=ec, lw=0.8, zorder=3)
        axb.plot(sorted(v)[1], y, mk, mfc=fc, mec=ec, mew=0.7, ms=3.0, zorder=4)
    handles.append(plt.Line2D([], [], marker=mk, color="none", mfc=fc, mec=ec, mew=0.7, ms=3.0))
axb.axvline(1.0, color=MUTED, lw=0.6, zorder=2)
axb.set_xscale("log")
axb.set_xlim(0.45, 2.6)
axb.set_xticks([0.5, 1, 2])
axb.set_xticklabels(["0.5", "1", "2"])
axb.minorticks_off()
axb.set_yticks(list(Y.values()))
axb.set_yticklabels([])
axb.set_ylim(-0.6, len(layers) - 0.4)
axb.tick_params(axis="y", length=0)
axb.set_xlabel("(b) Per tile vs. dense (FP32)", fontsize=FS_CAP)
tidy(axb, grid_axis="x")
fig.legend(handles, [m[2] for m in METRIC], loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False,
           fontsize=FS_LEG, handlelength=0.9, columnspacing=0.6, handletextpad=0.25, borderaxespad=0.1)
save(fig, "char_reduction.pdf")
