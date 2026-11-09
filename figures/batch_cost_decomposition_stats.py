#!/usr/bin/env python3
"""Build the stream-batching latency decomposition (wise2, YOLOv8m, RTX 3080, FP32, batch 1 and 4) from the logs.

Batch 1 is the cost-decomposition table's FP32 rows (cost_decomposition.csv, cost_decomposition_stats.py).  Batch 4
uses the same rule on logs/wise2_eval/rtx_3080_sm86/wildtrack640_yolov8m_b4_select/:
  micro_upd/bd_fp32.json       per-stage kernel time and GPU busy time under nsys (4 clip groups x 4 streams x 125 frames)
  micro_upd/memory2_fp32.json  the same frames without the profiler (the clip latency)
  e2e_full/fp32_*.json         whole-video E2E latency per batch of 4 frames (mean of C3 / C6)
Stages are the kernels' shares of the GPU busy time, idle = clip latency - busy, every time scaled by the path's E2E /
clip latency; batch-4 times are per frame (per batch / 4).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# ---- CONFIG
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D4 = REPO / "logs/wise2_eval/rtx_3080_sm86/wildtrack640_yolov8m_b4_select"
B1_CSV = HERE / "cost_decomposition.csv"
OUT_CSV = HERE / "batch_cost_decomposition.csv"
PATHS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
LABELS = {"dense": "Dense", "tile_skip": "Tile skipping", "gather_scatter": "Gather-scatter", "wiseconv": "WISEConv"}
E2E_FILES = {"dense": ("static", "dense"), "tile_skip": ("baselines", "tile_skip"),
             "gather_scatter": ("baselines", "gather_scatter"), "wiseconv": ("updatable", "wiseconv")}
FIELDS = ("total_ms", "construction_ms", "convolution_ms", "gather_ms", "other_ms", "idle_ms")


def e2e(path):
    f, key = E2E_FILES[path]
    L = json.loads((D4 / f"e2e_full/fp32_{f}.json").read_text())["backends"][key]["latency_ms"]
    return float(np.mean([L["C3"]["mean"], L["C6"]["mean"]]))


def main() -> None:
    rows = []
    with B1_CSV.open(encoding="utf-8", newline="") as src:
        for r in csv.DictReader(src):
            if r["dtype"] == "fp32":
                rows.append({"batch_size": 1, "system": r["system"], "system_label": r["system_label"],
                             "alignment": r["alignment"], "source": r["source"], **{k: r[k] for k in FIELDS}})
    bd = json.loads((D4 / "micro_upd/bd_fp32.json").read_text())["backends"]
    mem = json.loads((D4 / "micro_upd/memory2_fp32.json").read_text())["backends"]
    for path in PATHS:
        b, clip, tot = bd[path], mem[path]["latency_ms"], e2e(path)
        s = b["stage_us_per_frame"]
        ksum = sum(s.values())
        busy = min(clip, b["busy_us_per_frame"] / 1e3)
        f = tot / clip
        share = {k: busy * s.get(k, 0.0) / ksum for k in ("construction", "conv", "gather", "other")}
        row = {"batch_size": 4, "system": path, "system_label": LABELS[path], "alignment": f,
               "source": str((D4 / "micro_upd").relative_to(REPO)), "total_ms": tot / 4,
               "construction_ms": f * share["construction"] / 4, "convolution_ms": f * share["conv"] / 4,
               "gather_ms": f * share["gather"] / 4, "other_ms": f * share["other"] / 4, "idle_ms": f * (clip - busy) / 4}
        assert abs(sum(row[k] for k in FIELDS[1:]) - row["total_ms"]) < 1e-9
        if not 0.9 < f < 1.1:
            print(f"warning: batch 4 {path}: E2E / clip latency {f:.3f}")
        rows.append(row)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}: " + ", ".join(f"b{r['batch_size']} {r['system']} {float(r['total_ms']):.2f} ms "
                                           f"(x{float(r['alignment']):.3f})" for r in rows))


if __name__ == "__main__":
    main()
