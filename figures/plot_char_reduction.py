#!/usr/bin/env python3
"""fig:char-reduction (Insight 2): a worklist tile against a dense tile of the same configuration on the
representative layers (Nsight Compute, cold L2), every value a worklist / dense ratio on one shared axis.
(a) global-load sectors per request (the median frame's worklist launch over the dense launch).  (b)-(d) the marginal
cost of one more tile: the slope over the tile count of six worklist prefixes over that of six dense prefixes, for
executed instructions, SM active cycles, and DRAM read bytes (2026-11-12 re-measurement: the per-launch fixed traffic,
such as the cold weights, cancels in the slope).  Each row: range over RTX 3080 / AGX Orin x FP32 / FP16 (bar) and
their median (dot).  Reads char_reduction.csv and char_reduction_marginal.csv (char_stats.py)."""
from __future__ import annotations

import numpy as np

from char_style import FIG_W, FS_CAP, FS_NOTE, INK, MUTED, plt, read, save, tidy

ACC = "#1f7fa3"
red = read("char_reduction.csv")
marg = read("char_reduction_marginal.csv")
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


PANELS = (("(a) Sectors\nper request", spr),
          ("(b) Executed\ninstructions", lambda l: [float(r["inst"]) for r in marg if r["layer"] == l]),
          ("(c) SM active\ncycles", lambda l: [float(r["cycles"]) for r in marg if r["layer"] == l]),
          ("(d) DRAM\nread bytes", lambda l: [float(r["dram_bytes"]) for r in marg if r["layer"] == l]))

fig = plt.figure(figsize=(FIG_W, 1.72))
L, R, B, T = 0.315, 0.985, 0.175, 0.80
gap = 0.018
w = (R - L - gap * (len(PANELS) - 1)) / len(PANELS)
for i, (title, fn) in enumerate(PANELS):
    ax = fig.add_axes([L + i * (w + gap), B, w, T - B])
    ax.axvline(1.0, color=MUTED, lw=0.6, zorder=2)
    for l in layers:
        v = fn(l)
        ax.plot([min(v), max(v)], [Y[l]] * 2, color=ACC, lw=1.6, alpha=0.45, solid_capstyle="round", zorder=3)
        ax.plot(float(np.median(v)), Y[l], "o", color=ACC, ms=2.8, zorder=4)
    ax.set_xlim(0.8, 1.45)
    ax.set_xticks([0.8, 1.0, 1.2, 1.4])
    ax.set_xticklabels(["", "1", "", "1.4"])
    ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_yticks(list(Y.values()))
    ax.set_yticklabels([label(l) for l in layers] if i == 0 else [], fontsize=FS_NOTE)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=FS_NOTE + 0.5, linespacing=1.0, pad=2.5)
    tidy(ax, grid_axis="x")
fig.text(L - 0.012, T + 0.015, "input, $C_{in}$→$C_{out}$, kernel", ha="right", va="bottom", fontsize=FS_NOTE,
         color=INK)
fig.text((L + R) / 2, 0.01, "Worklist tile / dense tile", ha="center", va="bottom", fontsize=FS_CAP)
save(fig, "char_reduction.pdf")
