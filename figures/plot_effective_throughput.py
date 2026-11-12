#!/usr/bin/env python3
"""fig:effective-throughput: effective (eta*T) and wasted ((1-eta)*T) throughput of every path, relative to Dense raw
throughput of the same precision (reads effective_throughput.csv).

One panel per platform; each workload has an FP32 cluster (four paths) and an FP16 cluster (Dense, WISEConv), styled
as fig:e2e-latency.  The effective share is the path's color, the wasted share the dotted gray of fig:gap; WISEConv
bars carry their effective throughput.
"""
from __future__ import annotations

from collections import defaultdict

from bar_patterns import add_bar, add_wasted, legend_handles, wasted_handle
from char_style import FIG_W, FS_CAP, FS_LEG, FS_NOTE, FS_TICK, INK, MUTED, plt, read, save, tidy

PLATFORMS = ["RTX 3080", "AGX Orin"]
WORKLOADS = [("fireflownet", "FireFlowNet"), ("yolov8m_b1", "YOLOv8m b1"), ("yolov8m_b4", "YOLOv8m b4"),
             ("dynconv", "DynConv")]
CLUSTERS = [("fp32", ("dense", "tile_skip", "gather_scatter", "wiseconv")), ("fp16", ("dense", "wiseconv"))]
NAMES = {"dense": "Dense", "tile_skip": "Tile skipping", "gather_scatter": "Gather-scatter", "wiseconv": "WISEConv"}
CGAP, GGAP, YMAX = 0.4, 1.6, 1.25
BW = 0.84  # bar width (slot = 1): a small gap between bars
FP16_BAND = "#f3f1ec"
OURS_INK = "#a8521a"
FS_IN = 5.0

data = defaultdict(dict)
for r in read("effective_throughput.csv"):
    data[(r["platform"], r["workload"], r["dtype"])][r["path"]] = (float(r["effective"]), float(r["wasted"]))

fig = plt.figure(figsize=(FIG_W, 2.55))
L, R, T, B, VG = 0.075, 0.995, 0.865, 0.105, 0.115
ph = (T - B - VG) / 2
for i, plat in enumerate(PLATFORMS):
    ax = fig.add_axes([L, T - (i + 1) * ph - i * VG, R - L, ph])
    tidy(ax)
    x, centers, marks = 0.0, [], []
    for wl, _ in WORKLOADS:
        start = x
        for ci, (dtype, paths) in enumerate(CLUSTERS):
            if ci:
                x += CGAP
            c0 = x
            if dtype == "fp16":
                ax.axvspan(c0 - 0.5, c0 + len(paths) - 0.5, color=FP16_BAND, lw=0, zorder=0.5)
            cell = data.get((plat, wl, dtype), {})
            for k, p in enumerate(paths):
                if p not in cell:
                    continue
                eff, wasted = cell[p]
                add_bar(ax, c0 + k, eff, BW, p)
                if wasted > 0:
                    add_wasted(ax, c0 + k, min(wasted, YMAX - eff), BW, bottom=eff)
                if p == "wiseconv":
                    ax.text(c0 + k, eff + wasted + 0.03, f"{eff:.2f}", rotation=90, rotation_mode="anchor", ha="left",
                            va="center_baseline", fontsize=FS_IN, color=OURS_INK, fontweight="bold", zorder=6)
            marks.append((c0 + (len(paths) - 1) / 2, dtype.upper()))
            x += len(paths)
        centers.append((start + x - 1) / 2)
        x += GGAP
    ax.set_xlim(-0.8, x - GGAP - 0.2)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(axis="x", length=0)
    ax.set_xticks([c for c, _ in marks])
    ax.set_xticklabels([t for _, t in marks], fontsize=FS_NOTE - 0.5, color=INK)
    ax.text(0.0, 1.04, plat, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_CAP - 0.5, fontweight="bold")
    if i == len(PLATFORMS) - 1:
        for c, (_, name) in zip(centers, WORKLOADS):
            ax.text(c, -0.21, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_TICK)
fig.text(0.012, (T + B) / 2, "Throughput relative to Dense", rotation=90, ha="center", va="center", fontsize=FS_TICK)
handles = legend_handles(["dense", "tile_skip", "gather_scatter", "wiseconv"]) + [wasted_handle()]
fig.legend(handles, [NAMES[k] for k in NAMES] + ["wasted"], loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.54, 1.0), fontsize=FS_LEG - 0.5, handlelength=1.2, columnspacing=0.8, handletextpad=0.35)
save(fig, "effective_throughput.pdf")
