#!/usr/bin/env python3
"""fig:effective-throughput data: useful-compute ratio eta and relative throughput T of every path -> effective_throughput.csv.

Work counts come from the RTX 3080 work replay over the whole evaluation set (logs/wise2_eval/rtx_3080_sm86/<task>/
micro_upd/work_full_fp32.json, work_counts.py --split full: required, issued, and dense MACs per path), shared by every
platform and precision (the masks do not depend on either); FP16 WISEConv issues the FP16 plan's tiles on the same
worklists (path wiseconv_fp16, --issued-for fp16); latencies from e2e_stats.csv (the E2E figure's numbers).

  eta    = required / issued MACs (tile skipping: its MAC-level eta_mac)
  T      = (issued / dense MACs) * (t_dense / t_path)   raw throughput relative to Dense of the same precision
  eta*T  = effective throughput (= t_dense / t_path * alpha), (1 - eta)*T = wasted throughput
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EV = REPO / "logs/wise2_eval"
OUT = HERE / "effective_throughput.csv"
DEVICES = {"rtx_3080_sm86": "RTX 3080", "orin_sm87": "AGX Orin"}
TASKS = [("fireflownet", "mvsec_fireflownet_select"), ("yolov8m_b1", "wildtrack640_yolov8m_select"),
         ("yolov8m_b4", "wildtrack640_yolov8m_b4_select"), ("dynconv", "mpii_dynconv_select")]


def work(dev, task, dtype):
    f = EV / "rtx_3080_sm86" / task / "micro_upd" / "work_full_fp32.json"
    if not f.exists():
        return None, None
    paths = json.loads(f.read_text())["paths"]
    if dtype == "fp16":
        paths = {"dense": paths["dense"], "wiseconv": paths["wiseconv_fp16"]}
    return paths, f


def main():
    lat = {(r["platform"], r["workload"], r["dtype"], r["path"]): float(r["latency_ms"])
           for r in csv.DictReader((HERE / "e2e_stats.csv").open())}
    rows = []
    for dev, plat in DEVICES.items():
        for key, task in TASKS:
            for dtype in ("fp32", "fp16"):
                w, src = work(dev, task, dtype)
                if not w or (plat, key, dtype, "dense") not in lat:
                    continue
                dense_macs = w["dense"]["dense_macs"]
                required = w["wiseconv"]["required_macs"]
                t_dense = lat[(plat, key, dtype, "dense")]
                for path in ("dense", "tile_skip", "gather_scatter", "wiseconv"):
                    if path not in w or (plat, key, dtype, path) not in lat:
                        continue
                    issued = w[path]["issued_macs"]
                    eta = w[path]["eta_mac"] if path == "tile_skip" and "eta_mac" in w[path] else required / issued
                    T = issued / dense_macs * t_dense / lat[(plat, key, dtype, path)]
                    rows.append({"platform": plat, "workload": key, "dtype": dtype, "path": path, "eta": f"{eta:.5f}",
                                 "T": f"{T:.5f}", "effective": f"{eta * T:.5f}", "wasted": f"{(1 - eta) * T:.5f}",
                                 "source": str(src.relative_to(REPO))})
    with OUT.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {OUT.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
