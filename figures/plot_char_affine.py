#!/usr/bin/env python3
"""fig:char-affine (Insight 1): layer-averaged convolution time over dense time against the active ratio, one panel
per GPU and precision.  Per-length best (markers; its affine fit as text), the best single configuration, the
dense-tuned configuration, and the proportional diagonal.  Reads char_affine.csv (char_stats.py)."""
from __future__ import annotations

from bar_patterns import IDEAL, IDEAL_LS
from char_style import CFG, FIG_W, FS_CAP, FS_LEG, FS_NOTE, INK, plt, read, save, tidy

rows = read("char_affine.csv")
DEVS, PRECS = ("RTX 3080", "AGX Orin"), ("FP32", "FP16")

fig = plt.figure(figsize=(FIG_W, 2.55))
L, R, B, T, GX, GY = 0.115, 0.985, 0.11, 0.815, 0.07, 0.09
pw, ph = (R - L - GX) / 2, (T - B - GY) / 2
handles = {}
for i, dev in enumerate(DEVS):
    for j, prec in enumerate(PRECS):
        ax = fig.add_axes([L + j * (pw + GX), T - (i + 1) * ph - i * GY, pw, ph])
        d = sorted((r for r in rows if r["device"] == dev and r["dtype"] == prec), key=lambda r: float(r["alpha"]))
        a = [float(r["alpha"]) for r in d]
        get = lambda k: [float(r[k]) for r in d]
        fa, fb = float(d[0]["fit_a"]), float(d[0]["fit_b"])
        ax.plot([0, 1], [0, 1], color=IDEAL, ls=IDEAL_LS, lw=1.0, zorder=2)
        handles["tuned"], = ax.plot(a, get("tuned"), color=CFG["tuned"], lw=1.0, ls=(0, (4, 1.5, 1, 1.5)), zorder=3)
        handles["single"], = ax.plot(a, get("single"), color=CFG["single"], lw=1.0, ls=(0, (3, 1.5)), zorder=3)
        handles["best"], = ax.plot(a, get("best"), color=CFG["best"], lw=1.2, marker="o", ms=2.4, zorder=4)
        ax.text(0.97, 0.05, f"best $\\approx$ {fa:.2f}$\\alpha$ + {fb:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_NOTE, color=INK)
        ax.text(0.03, 0.97, f"{dev}, {prec}", transform=ax.transAxes, ha="left", va="top", fontsize=FS_NOTE,
                color=INK, fontweight="bold")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.25)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.5, 1])
        if i == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        if j == 1:
            ax.set_yticklabels([])
        tidy(ax, grid_axis="y")
fig.text(L + (R - L) / 2, 0.012, r"Active ratio $\alpha$", ha="center", va="bottom", fontsize=FS_CAP)
fig.text(0.012, B + (T - B) / 2, r"$t\,/\,t_{\mathrm{dense}}$", ha="left", va="center", rotation=90, fontsize=FS_CAP)
handles["ideal"] = plt.Line2D([], [], color=IDEAL, ls=IDEAL_LS, lw=1.0)
fig.legend([handles[k] for k in ("best", "single", "tuned", "ideal")],
           ["best configuration per length", "fixed: latency-optimal",
            "fixed: dense-tuned", "ideal ($t \\propto \\alpha$)"], loc="upper center",
           bbox_to_anchor=(0.54, 1.0), ncol=2, frameon=False, fontsize=FS_LEG, handlelength=1.8, columnspacing=0.8,
           handletextpad=0.4)
save(fig, "char_affine.pdf")
