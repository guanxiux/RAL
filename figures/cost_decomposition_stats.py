#!/usr/bin/env python3
"""Build the paper-facing latency decomposition (wise2, YOLOv8m batch 1, RTX 3080) from the collected logs.

Sources (logs/wise2_eval/<device>/wildtrack640_yolov8m_select/, experiments/wise2/report/collect.py):
  micro_upd/bd_<dtype>.json      per-stage kernel time and GPU busy time of each path under nsys (16 clips x 250 frames)
  micro_upd/memory_<dtype>.json  the same frames without the profiler (the clip latency)
  e2e_full/<dtype>_*.json        whole-video E2E latency of each path (mean of C3 / C6)
Each path's stages are its kernels' shares of the GPU busy time; idle = latency - busy (host synchronization, gaps);
every time is scaled by the path's E2E / clip latency (shares unchanged).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# ---- CONFIG
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = REPO / "logs/wise2_eval"
DEVICE = "rtx_3080_sm86"
TASK = "wildtrack640_yolov8m_select"
OUT_CSV = HERE / "cost_decomposition.csv"
PATHS = {"fp32": ("dense", "tile_skip", "gather_scatter", "wiseconv"), "fp16": ("dense", "wiseconv")}
LABELS = {"dense": "Dense", "tile_skip": "Tile skipping", "gather_scatter": "Gather-scatter", "wiseconv": "WISEConv"}
E2E_FILES = {"dense": ("static", "dense"), "tile_skip": ("baselines", "tile_skip"),
             "gather_scatter": ("baselines", "gather_scatter"), "wiseconv": ("updatable", "wiseconv")}


def e2e(d, dtype, path):
    f, key = E2E_FILES[path]
    L = json.loads((d / f"e2e_full/{dtype}_{f}.json").read_text())["backends"][key]["latency_ms"]
    return float(np.mean([L["C3"]["mean"], L["C6"]["mean"]]))


def main() -> None:
    d = EV / DEVICE / TASK
    rows = []
    for dtype, paths in PATHS.items():
        bd = json.loads((d / f"micro_upd/bd_{dtype}.json").read_text())["backends"]
        mem = json.loads((d / f"micro_upd/memory_{dtype}.json").read_text())["backends"]
        for path in paths:
            b, clip, tot = bd[path], mem[path]["latency_ms"], e2e(d, dtype, path)
            s = b["stage_us_per_frame"]
            ksum = sum(s.values())
            busy = min(clip, b["busy_us_per_frame"] / 1e3)
            f = tot / clip
            share = {k: busy * s.get(k, 0.0) / ksum for k in ("construction", "conv", "gather", "other")}
            row = dict(dtype=dtype, system=path, system_label=LABELS[path], total_ms=tot, clip_ms=clip, alignment=f,
                       construction_ms=f * share["construction"], convolution_ms=f * share["conv"],
                       gather_ms=f * share["gather"], other_ms=f * share["other"], idle_ms=f * (clip - busy),
                       source=str((d / "micro_upd").relative_to(REPO)))
            parts = row["construction_ms"] + row["convolution_ms"] + row["gather_ms"] + row["other_ms"] + row["idle_ms"]
            assert abs(parts - tot) < 1e-9, (path, parts, tot)  # the parts add up to the E2E latency
            assert 0.9 < f < 1.1, (path, f)  # the clips stand for the whole video
            rows.append(row)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows; alignment {', '.join('%s %.3f' % (r['system'], r['alignment']) for r in rows)})")


if __name__ == "__main__":
    main()
