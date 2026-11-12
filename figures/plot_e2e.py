#!/usr/bin/env python3
"""fig:e2e-latency: full-model latency of every path, normalized to Dense of the same precision (reads e2e_stats.csv).

One panel per platform (3 x 2).  Each workload is a group with an FP32 cluster (the four paths) and an FP16 cluster
(Dense and WISEConv; the baselines have no FP16 path).  Dense = 1 in each cluster, so the Ideal line of proportional
scaling is the active ratio alpha.  Dense bars carry their absolute latency, WISEConv bars their speedup over the
fastest path of the cluster, and bars above the axis are clipped with their value printed inside.
"""
from __future__ import annotations

from collections import defaultdict

from matplotlib.lines import Line2D

from bar_patterns import COLORS, IDEAL, IDEAL_LS, INK, MUTED, add_bar, legend_handles
from char_style import FS_CAP, FS_LEG, FS_NOTE, FS_TICK, plt, read, save, tidy

FIG_W, FIG_H = 7.0, 3.42  # acmart \textwidth (505.9 pt)
YMAX = 1.6
PLATFORMS = ["A100", "RTX 4090", "RTX 3080", "RTX 4070 Laptop", "AGX Orin", "Xavier NX"]
WORKLOADS = [("fireflownet", "FireFlowNet"), ("yolov8m_b1", "YOLOv8m b1"), ("yolov8m_b4", "YOLOv8m b4"),
             ("dynconv", "DynConv")]
CLUSTERS = [("fp32", ("dense", "tile_skip", "gather_scatter", "wiseconv")), ("fp16", ("dense", "wiseconv"))]
NAMES = {"dense": "Dense", "tile_skip": "Tile skipping", "gather_scatter": "Gather-scatter", "wiseconv": "WISEConv"}
BW = 1.0           # bar width (slot = 1): the bars of a cluster touch
CGAP = 0.45        # between the FP32 and FP16 clusters
GGAP = 2.3         # between workloads
FP16_BAND = "#f3f1ec"
OURS_INK = "#a8521a"  # WISEConv labels: a darker step of the WISEConv orange
TRUNC = "#c62828"     # the truncation mark
FS_IN = 6.0        # numbers inside / above bars

data = defaultdict(dict)
alpha = {}
for r in read("e2e_stats.csv"):
    key = (r["platform"], r["workload"], r["dtype"])
    data[key][r["path"]] = (float(r["latency_ms"]), float(r["norm"]))
    alpha[key] = float(r["alpha"])


# the mean active ratio of each workload (FP32 replay; FP16 runs the same masks)
alpha_of = {name: alpha[("RTX 3080", wl, "fp32")] for wl, name in WORKLOADS}


def fmt_ms(ms):
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}"


def panel(ax, platform, left, bottom_row):
    tidy(ax)
    x = 0.0
    centers, cluster_marks = [], []
    for wl, _ in WORKLOADS:
        start = x
        for ci, (dtype, paths) in enumerate(CLUSTERS):
            if ci:
                x += CGAP
            c0 = x
            cell = data.get((platform, wl, dtype))
            if dtype == "fp16":
                ax.axvspan(c0 - 0.5, c0 + len(paths) - 0.5, color=FP16_BAND, lw=0, zorder=0.5)
            if not cell:
                ax.text(c0 + (len(paths) - 1) / 2, 0.06, "no FP16", rotation=90, ha="center", va="bottom",
                        fontsize=FS_IN, color=INK)
            else:
                fastest = min(v[1] for p, v in cell.items() if p != "wiseconv")
                a = alpha[(platform, wl, dtype)]
                ax.plot([c0 - 0.5, c0 + len(paths) - 0.5], [a, a], color=IDEAL, lw=1.2, ls=IDEAL_LS, zorder=5)
                for k, p in enumerate(paths):
                    if p not in cell:
                        continue
                    ms, h = cell[p]
                    xc = c0 + k
                    add_bar(ax, xc, min(h, YMAX), BW, p)
                    if p == "dense":
                        ax.text(xc, 0.95, f"{fmt_ms(ms)} ms", rotation=90, ha="center", va="top", fontsize=FS_IN,
                                color="white", zorder=6)
                    elif h > YMAX:  # truncated at the axis top
                        ax.plot([xc], [YMAX], marker="x", ms=4.2, mew=1.3, color=TRUNC, clip_on=False, zorder=8)
                    if p == "wiseconv":
                        ax.text(xc, h + 0.04, f"{fastest / h:.2f}" + r"$\times$", rotation=90, rotation_mode="anchor",
                                ha="left", va="center_baseline", fontsize=FS_IN, color=OURS_INK, fontweight="bold",
                                zorder=6)
            cluster_marks.append((c0 + (len(paths) - 1) / 2, dtype.upper()))
            x += len(paths)
        centers.append((start + x - 1) / 2)
        x += GGAP
    ax.set_xlim(-0.8, x - GGAP - 0.2)
    ax.set_ylim(0, YMAX)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["0", "0.5", "1", "1.5"] if left else [])
    ax.tick_params(axis="x", length=0)
    ax.set_xticks([c for c, _ in cluster_marks])
    ax.set_xticklabels([t for _, t in cluster_marks], fontsize=FS_NOTE, color=INK)
    if bottom_row:
        for c, (_, name) in zip(centers, WORKLOADS):
            ax.text(c, -0.20, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS_TICK + 0.5)
            ax.text(c, -0.40, r"$\hat{\alpha}$ = " + f"{100 * alpha_of[name]:.1f}%", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=FS_NOTE, color=INK)
    ax.text(0.0, 1.07, platform, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_CAP, fontweight="bold")


fig = plt.figure(figsize=(FIG_W, FIG_H))
L, R, T, B = 0.058, 0.995, 0.875, 0.115
HG, VG = 0.035, 0.085
pw = (R - L - HG) / 2
ph = (T - B - 2 * VG) / 3
for i, platform in enumerate(PLATFORMS):
    row, col = divmod(i, 2)
    ax = fig.add_axes([L + col * (pw + HG), T - (row + 1) * ph - row * VG, pw, ph])
    panel(ax, platform, col == 0, row == 2)
    if col == 0 and row == 1:
        ax.set_ylabel("Latency normalized to Dense", fontsize=FS_TICK + 0.5)

handles = legend_handles(["dense", "tile_skip", "gather_scatter", "wiseconv"])
handles.append(Line2D([], [], color=IDEAL, lw=1.2, ls=IDEAL_LS))
handles.append(Line2D([], [], color=TRUNC, marker="x", ms=4.2, mew=1.3, lw=0))
labels = [NAMES[k] for k in ("dense", "tile_skip", "gather_scatter", "wiseconv")] + [r"Ideal ($\hat{\alpha}$)",
                                                                                    "beyond axis"]
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.53, 1.005),
           fontsize=FS_LEG + 0.5, handlelength=1.6, columnspacing=1.6, handletextpad=0.5)
save(fig, "e2e_latency.pdf")
