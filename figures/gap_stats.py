#!/usr/bin/env python3
"""Derive the Background sparsity-to-latency example (fig:gap) from the evaluation logs.

Workload: YOLOv8m at batch 4 on WildTrack 640, RTX 3080, FP32, the temporal difference masks of the task policy
(DeltaCNN).  Every number comes from the files the evaluation tables use (experiments/wise2/report/eval_tables.py,
``Effective throughput / worklist efficiency``):

  * ``logs/wise2_eval/rtx_3080_sm86/wildtrack640_yolov8m_b4_select/micro_upd/work_fp32.json``: issued and required
    MACs of each path (tile skipping: the MAC-level useful-compute ratio ``eta_mac``);
  * the end-to-end latency per batch step of each path (eval_tables._yolo_latencies).

Throughput is normalized to dense raw throughput (cuDNN):

  T_path / T_dense     = (issued_path / dense MACs) * (t_dense / t_path)
  T_eff,path / T_dense = eta_path * T_path / T_dense,        wasted = (1 - eta_path) * T_path / T_dense

The proportional reference is dense latency scaled by the required / dense MAC ratio (alpha_MAC).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/wise2/report"))
import eval_tables as T  # noqa: E402

DEVICE, TASK, DTYPE = "rtx_3080_sm86", "wildtrack640_yolov8m_b4_select", "fp32"
OUT_CSV = HERE / "gap_stats.csv"
SYSTEMS = [("dense", "Dense"), ("tile_skip", "Tile skipping"), ("gather_scatter", "Gather-scatter"),
           ("wiseconv", "WISEConv")]  # WISEConv: for the evaluation figure, not drawn in fig:gap


def main() -> None:
    work_path = T.EV / DEVICE / TASK / "micro_upd" / f"work_{DTYPE}.json"
    work = json.loads(work_path.read_text())
    lats = T._yolo_latencies(DEVICE, TASK, DTYPE)
    dense_macs = work["paths"]["dense"]["dense_macs"]
    required = work["paths"]["wiseconv"]["required_macs"]
    alpha = required / dense_macs
    t_dense = lats["dense"]
    rows = []
    for key, label in SYSTEMS:
        w = work["paths"][key]
        eta = w["eta_mac"] if key == "tile_skip" else w["required_macs"] / w["issued_macs"]
        t = lats[key]
        t_rel = (w["issued_macs"] / dense_macs) * (t_dense / t)
        rows.append(dict(system=label, backend=key, latency_ms=t, useful_compute_ratio=eta,
                         raw_throughput_vs_dense=t_rel, effective_throughput_vs_dense=eta * t_rel,
                         wasted_throughput_vs_dense=(1 - eta) * t_rel, latency_reduction_vs_dense=1 - t / t_dense,
                         active_ratio_mac=alpha, proportional_reference_ms=alpha * t_dense,
                         source=f"{work_path.relative_to(REPO)}; e2e latency per batch step (eval_tables)"))
    # cross-checks against the numbers quoted in the text (intro / background)
    r = {x["backend"]: x for x in rows}
    assert abs(r["dense"]["useful_compute_ratio"] - (1 - 0.822)) < 5e-4
    assert abs(r["gather_scatter"]["raw_throughput_vs_dense"] - 0.317) < 5e-4
    assert abs(r["tile_skip"]["useful_compute_ratio"] - 0.674) < 5e-4
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for x in rows:
            wr.writerow({k: f"{v:.9g}" if isinstance(v, float) else v for k, v in x.items()})
    for x in rows:
        print(f"{x['system']:15s} {x['latency_ms']:7.2f} ms  eta {x['useful_compute_ratio']:.3f}  T {x['raw_throughput_vs_dense']:.3f}"
              f"  eff {x['effective_throughput_vs_dense']:.3f}  wasted {x['wasted_throughput_vs_dense']:.3f}")
    print(f"proportional reference {alpha * t_dense:.2f} ms (alpha_MAC {alpha:.3f}) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
