#!/usr/bin/env python3
"""Build the WISEConv efficiency / worklist table data (wise2, RTX 3080, FP32) from the collected logs.

Source: logs/wise2_eval/rtx_3080_sm86/<task>_select/micro_upd/work_full_fp32.json (the whole evaluation set; experiments/wise2/microbench/work_counts.py
on the deployment's updatable plan): the position-level eta decomposition (eta_construct = sum ||M_out||_0 / sum n_W
from work_etac2_fp32.json, a replay that traces each reusing layer's M_out (q3080_etaconstruct2.sh, 2026-11-17); eta_compute = required / issued positions with every layer's tile padding at the
candidate its n_W selects) and the worklist lengths of every layer invocation pooled over layers and frames.
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
OUTPUT = HERE / "worklist_efficiency.csv"
WORKLOADS = (("fireflownet", "FireFlowNet", "mvsec_fireflownet_select"),
             ("yolov8m", "YOLOv8m b1", "wildtrack640_yolov8m_select"),
             ("yolov8m_b4", "YOLOv8m b4", "wildtrack640_yolov8m_b4_select"),
             ("dynconv_pose", "DynConv", "mpii_dynconv_select"))


def main() -> None:
    rows = []
    for key, label, task in WORKLOADS:
        path = EV / task / f"micro_upd/work_full_{DTYPE}.json"
        d = json.loads(path.read_text())
        w, n = d["paths"]["wiseconv"], d["worklist_length"]
        etac_path = EV / task / f"micro_upd/work_etac2_{DTYPE}.json"  # the longer replay (q3080_etaconstruct2.sh)
        etac = json.loads(etac_path.read_text())["paths"]["wiseconv"]["eta_construct_measured"]
        rows.append({"workload": key, "workload_label": label, "frames": d["frames"],
                     "invocations": n["invocations"], "p10_worklist_length": n["p10"],
                     "p50_worklist_length": n["p50"], "p90_worklist_length": n["p90"],
                     "eta_construct_frame_mean": etac, "eta_compute_frame_mean": w["eta_compute"],
                     "useful_compute_ratio_frame_mean": etac * w["eta_compute"],
                     "source": f"{path.relative_to(REPO)}; {etac_path.relative_to(REPO)}",
                     "aggregation": "percentiles_across_layer_frame_invocations"})
    with OUTPUT.open("w", encoding="utf-8", newline="") as out:
        wr = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
