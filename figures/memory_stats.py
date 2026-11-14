#!/usr/bin/env python3
"""Build the paper-facing peak-memory data (wise2, RTX 3080, FP32) from the collected logs.

Sources (logs/wise2_eval/rtx_3080_sm86/, experiments/wise2/report/collect.py):
  runs/latency/<fireflownet|dynconv_pose>/fp32/memory2-<path>/summary.json   the E2E latency protocol, one round
  <wildtrack640_yolov8m|_b4>_select/micro_upd/memory2_fp32.json            e2e_breakdown.py (16 clips x 250 frames at
                                                                             batch 1, 4 x 4 streams x 125 frames at batch 4)
The value is the peak live allocation (torch.cuda.max_memory_allocated) from before the CUDA-graph capture to the end
of the timed frames (a graph's activations are allocated at capture; replays do not allocate), without the preloaded
input frames; for runs over several sequences the largest sequence peak.  FP32 only: in FP16 the cuDNN workspace of the
remaining dense convolutions depends on cudnn.benchmark's per-run choice (up to +120 MiB on DynConv).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# ---- CONFIG
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = REPO / "logs/wise2_eval/rtx_3080_sm86"
DTYPE = "fp32"
OUT_CSV = HERE / "memory.csv"
PATHS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
# (key, label, source kind, source)
WORKLOADS = (
    ("fireflownet", "FireFlowNet", "run", "fireflownet"),
    ("yolov8m", "YOLOv8m b1", "breakdown", "wildtrack640_yolov8m_select"),
    ("yolov8m_b4", "YOLOv8m b4", "breakdown", "wildtrack640_yolov8m_b4_select"),
    ("dynconv", "DynConv", "run", "dynconv_pose"),
)


def peaks(kind: str, src: str) -> dict:
    if kind == "breakdown":
        b = json.loads((EV / src / f"micro_upd/memory2_{DTYPE}.json").read_text())["backends"]
        # the re-measured dense baseline (2026-11-13: cudnn.benchmark, --dense-layout auto)
        b["dense"] = json.loads((EV / src / f"micro_upd/memory_dense_{DTYPE}.json").read_text())["backends"]["dense"]
        return {k: v["peak_mib"] for k, v in b.items()}
    out = {}
    for path in PATHS:
        name = "dynconv" if src == "dynconv_pose" and path == "gather_scatter" else path
        s = json.loads((EV / f"runs/latency/{src}/{DTYPE}/memory2-{name}/summary.json").read_text())
        found = []

        def walk(v):  # every sequence's peak_mib
            if isinstance(v, dict):
                for k, u in v.items():
                    if k == "peak_mib":
                        found.append(float(u))
                    else:
                        walk(u)
            elif isinstance(v, list):
                for u in v:
                    walk(u)
        walk(s)
        out[path] = max(found)
    return out


def main() -> None:
    rows = []
    for key, label, kind, src in WORKLOADS:
        p = peaks(kind, src)
        rows.append({"workload": key, "workload_label": label, **{k: f"{p[k]:.12g}" for k in PATHS}})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")
    for r in rows:
        print(f"{r['workload_label']:<12} " + " ".join(f"{k}={float(r[k]):.1f}" for k in PATHS))


if __name__ == "__main__":
    main()
