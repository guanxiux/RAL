#!/usr/bin/env python3
"""fig:char-reduction (Insight 2): a worklist tile against a dense tile of the same configuration on the
representative layers (Nsight Compute, cold L2), every value a worklist / dense ratio on one shared axis.
(a) global-load sectors per request (the median frame's worklist launch over the dense launch).  (b)-(d) the marginal
cost of one more tile: the slope over the tile count of six worklist prefixes over that of six dense prefixes, for
executed instructions, SM active cycles, and DRAM read bytes (2026-11-12 re-measurement: the per-launch fixed traffic,
such as the cold weights, cancels in the slope).  Each row: range over RTX 3080 / AGX Orin x FP32 / FP16 (bar) and
their median (dot); below each, the coordinate-indexed variant (kernel map per row and tap, i2_indexed.py) over the
dense tile of the same run.  Reads char_reduction.csv, char_reduction_marginal.csv, and char_reduction_indexed.csv
(char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_NOTE, INK, MUTED, plt, read, save, tidy

ACC = "#1f7fa3"
red = read("char_reduction.csv")
marg = read("char_reduction_marginal.csv")
ixr = read("char_reduction_indexed.csv")
IDX = "#7660a6"  # the purple of the baseline frames: an existing execution paradigm
DY = 0.17
layers = list(dict.fromkeys(r["layer"] for r in marg))
Y = {l: len(layers) - 1 - i for i, l in enumerate(layers)}


def label(l):
    """'640² 3→48 3x3s2' (FP32) -> '640², 3→48, 3×3/2'."""
    s = next(r["short"] for r in marg if r["layer"] == l and r["dtype"] == "FP32")
    hw, ck, k = s.split(" ")
    k = k.replace("x", "×").replace("s2", "/2")
    return f"{hw}, {ck}, {k}"


def spr(l):
    v = []
    for dev in ("RTX 3080", "AGX Orin"):
        for p in ("FP32", "FP16"):
            g = {r["variant"]: float(r["sectors_per_request"]) for r in red
                 if r["layer"] == l and r["device"] == dev and r["dtype"] == p}
            v.append(g["real_p50"] / g["dense"])
    return v


col = lambda rows, k: (lambda l: [float(r[k]) for r in rows if r["layer"] == l])
PANELS = (("(a) Sectors\nper request", spr, col(ixr, "sectors_per_request")),
          ("(b) Executed\ninstructions", col(marg, "inst"), col(ixr, "inst")),
          ("(c) SM active\ncycles", col(marg, "cycles"), col(ixr, "cycles")),
          ("(d) DRAM\nread bytes", col(marg, "dram_bytes"), col(ixr, "dram_bytes")))

fig = plt.figure(figsize=(FIG_W, 1.72))
L, R, B, T = 0.315, 0.985, 0.175, 0.80
gap = 0.018
w = (R - L - gap * (len(PANELS) - 1)) / len(PANELS)
for i, (title, fn, fx) in enumerate(PANELS):
    ax = fig.add_axes([L + i * (w + gap), B, w, T - B])
    ax.axvline(1.0, color=MUTED, lw=0.6, zorder=2)
    for l in layers:
        for v, y, c, mk in ((fn(l), Y[l] + DY, ACC, "o"), (fx(l), Y[l] - DY, IDX, "^")):
            ax.plot([min(v), max(v)], [y] * 2, color=c, lw=1.6, alpha=0.45, solid_capstyle="round", zorder=3)
            ax.plot(float(np.median(v)), y, mk, color=c, ms=2.8, zorder=4)
    ax.set_xlim(0.55, 1.75)
    ax.set_xticks([0.6, 1.0, 1.4])
    ax.set_xticklabels(["0.6", "1", "1.4"])
    ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_yticks(list(Y.values()))
    ax.set_yticklabels([label(l) for l in layers] if i == 0 else [], fontsize=FS_NOTE)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=FS_NOTE + 0.5, linespacing=1.0, pad=2.5)
    tidy(ax, grid_axis="x")
fig.text(L - 0.012, T + 0.015, "input, $C_{in}$→$C_{out}$, kernel", ha="right", va="bottom", fontsize=FS_NOTE,
         color=INK)
fig.text((L + R) / 2, 0.01, "Tile / dense tile", ha="center", va="bottom", fontsize=FS_CAP)
fig.legend([plt.Line2D([], [], color=ACC, marker="o", ms=2.8, lw=1.6, alpha=0.8),
            plt.Line2D([], [], color=IDX, marker="^", ms=2.8, lw=1.6, alpha=0.8)], ["worklist", "indexed"],
           loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, frameon=False, fontsize=FS_NOTE, handlelength=1.4,
           handletextpad=0.4, borderaxespad=0.1)
save(fig, "char_reduction.pdf")
