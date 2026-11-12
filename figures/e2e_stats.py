#!/usr/bin/env python3
"""End-to-end latency of every path, platform, workload, and precision -> e2e_stats.csv (fig:e2e-latency).

Reads ``logs/wise2_eval/`` (collected by ``experiments/wise2/report/collect.py``) through the loaders of
``experiments/wise2/report/eval_tables.py``, so the numbers are the ones in ``logs/wise2_eval/TABLES.md``.

  * YOLOv8m (batch 1 and batch 4): the whole-video runs; the two WildTrack cameras are one dataset, so their means are
    merged frame-weighted.
  * FireFlowNet: the mean of the per-sequence means (every held-out MVSEC sequence counts equally).  DynConv: the median of rounds (one MPII split).
  * alpha (the Ideal reference, latency normalized to Dense = alpha): required / dense MACs of the RTX 3080 work replay
    over the whole evaluation set (work_full_fp32.json, work_counts.py --split full), one per workload: the masks do
    not depend on the precision
    of each workload and precision; all platforms run the same masks.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/wise2/report"))
import eval_tables as et  # noqa: E402

OUT = HERE / "e2e_stats.csv"
PLATFORMS = ["a100_sxm4_40gb_sm80", "rtx_4090_sm89", "rtx_3080_sm86", "rtx_4070_laptop_gpu_sm89", "orin_sm87",
             "xavier_sm72"]
# (key, task directory, run workload, gather-scatter backend in the run files)
WORKLOADS = [
    ("fireflownet", "mvsec_fireflownet_select", "fireflownet", "gather_scatter"),
    ("yolov8m_b1", "wildtrack640_yolov8m_select", None, None),
    ("yolov8m_b4", "wildtrack640_yolov8m_b4_select", None, None),
    ("dynconv", "mpii_dynconv_select", "dynconv_pose", "dynconv"),
]
PATHS = ("dense", "tile_skip", "gather_scatter", "wiseconv")


def yolo(dev, task, dtype):
    res = {}
    rows = et.yolo_rows(dev, task).get(dtype, {})
    d = et.yolo_dir(dev, task)
    for path in PATHS:
        if path not in rows:
            continue
        # frame-weighted over the cameras: the per-camera frame counts sit next to the means in the result files
        frames = None
        for f in (f"{dtype}_static.json", f"{dtype}_updatable.json", f"{dtype}_baselines.json"):
            j = et.load(d / f)
            if j and path in j.get("backends", {}):
                L = j["backends"][path]["latency_ms"]
                frames = [L[c]["frames"] for c in ("C3", "C6")]
                means = [L[c]["mean"] for c in ("C3", "C6")]
                assert all(abs(a - b) < 1e-9 for a, b in zip(means, rows[path]["lat"])), (dev, task, dtype, path)
        lat = rows[path]["lat"]
        res[path] = sum(l * n for l, n in zip(lat, frames)) / sum(frames) if frames else sum(lat) / len(lat)
    return res


def runs(dev, wl, dtype, gs):
    res = {}
    for path, backend in (("dense", "dense"), ("tile_skip", "tile_skip"), ("gather_scatter", gs), ("wiseconv", "wiseconv")):
        mode = et.ours_run(dev, wl, dtype) if path == "wiseconv" else "static"
        if mode is None:
            continue
        s, _ = et.run_summary(dev, "latency", wl, dtype, backend, mode)
        if not s:
            continue
        seqs = s["sequences"].values()
        if wl == "fireflownet":  # each MVSEC sequence counts equally (the protocol of the registered abstract)
            res[path] = sum(v["all_frames_gpu_ms"]["mean"] for v in seqs) / len(seqs)
        else:
            vals = [v["all_frames_gpu_ms"]["median"] for v in seqs]
            res[path] = sum(vals) / len(vals)
    return res


def alpha(task):
    w = et.load(et.EV / "rtx_3080_sm86" / task / "micro_upd" / "work_full_fp32.json")
    return w["paths"]["wiseconv"]["required_macs"] / w["paths"]["dense"]["dense_macs"] if w else None


def main():
    rows = []
    for dev in PLATFORMS:
        for key, task, wl, gs in WORKLOADS:
            for dtype in ("fp32", "fp16"):
                lat = yolo(dev, task, dtype) if wl is None else runs(dev, wl, dtype, gs)
                if "dense" not in lat:
                    continue
                a = alpha(task)
                for path, ms in lat.items():
                    rows.append({"platform": et.DEV_NAMES[dev], "workload": key, "dtype": dtype, "path": path,
                                 "latency_ms": f"{ms:.5f}", "norm": f"{ms / lat['dense']:.5f}",
                                 "alpha": f"{a:.5f}" if a else ""})
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
