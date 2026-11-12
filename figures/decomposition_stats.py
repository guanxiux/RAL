#!/usr/bin/env python3
"""fig:decomposition data: per-frame latency of YOLOv8m split by stage (RTX 3080; batch 1 and 4; FP32 and FP16).

Sources (logs/wise2_eval/rtx_3080_sm86/<task>/micro_upd/, collected by experiments/wise2/report/collect.py):
  bd_<dtype>.json      per-stage kernel time and GPU busy time of each path under nsys (random clips)
  memory_<dtype>.json  the same clips without the profiler (clip latency)
and the whole-video E2E latency of each path from e2e_stats.csv (the cameras merged frame-weighted).
Each path's stages are its kernels' shares of the GPU busy time; idle = clip latency - busy (host synchronization and
gaps); every time is scaled by E2E / clip latency (shares unchanged) and divided by the batch, so all rows are per frame.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = REPO / "logs/wise2_eval/rtx_3080_sm86"
OUT = HERE / "decomposition.csv"
TASKS = {1: ("wildtrack640_yolov8m_select", "yolov8m_b1"), 4: ("wildtrack640_yolov8m_b4_select", "yolov8m_b4")}
PATHS = {"fp32": ("dense", "tile_skip", "gather_scatter", "wiseconv"), "fp16": ("dense", "wiseconv")}


def main():
    e2e = {(r["workload"], r["dtype"], r["path"]): float(r["latency_ms"])
           for r in csv.DictReader((HERE / "e2e_stats.csv").open()) if r["platform"] == "RTX 3080"}
    rows = []
    for batch, (task, key) in TASKS.items():
        d = EV / task / "micro_upd"
        for dtype, paths in PATHS.items():
            bd = json.loads((d / f"bd_{dtype}.json").read_text())["backends"]
            mem = json.loads((d / f"memory_{dtype}.json").read_text())["backends"]
            for path in paths:
                b, clip, tot = bd[path], mem[path]["latency_ms"], e2e[(key, dtype, path)]
                s = b["stage_us_per_frame"]
                ksum = sum(s.values())
                busy = min(clip, b["busy_us_per_frame"] / 1e3)
                f = tot / clip
                assert 0.85 < f < 1.2, (batch, dtype, path, f)  # the clips stand for the whole video
                part = {k: f * busy * s.get(k, 0.0) / ksum / batch for k in ("construction", "conv", "gather", "other")}
                part["gather"] += part.pop("construction") if path == "gather_scatter" else 0.0  # its mask
                part.setdefault("construction", 0.0)  # compaction into an index list is part of its gather pass
                part["idle"] = f * (clip - busy) / batch
                assert abs(sum(part.values()) - tot / batch) < 1e-9
                rows.append({"batch": batch, "dtype": dtype, "path": path, "total_ms": f"{tot / batch:.5f}",
                             **{f"{k}_ms": f"{v:.5f}" for k, v in part.items()}, "alignment": f"{f:.4f}",
                             "source": str(d.relative_to(REPO))})
    with OUT.open("w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
