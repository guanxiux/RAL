#!/usr/bin/env python3
"""Build the paper-facing ablation data (wise2, YOLOv8m batch 1, FP16; RTX 3080 and, since 2026-11-14, AGX Orin) from the
collected logs, one block of rows per device.

Sources (logs/wise2_eval/<device>/wildtrack640_yolov8m_select/, written by experiments/wise2/report/collect.py):
  ablation_upd/lat_<dtype>.json     CUDA-event latency of every variant (e2e_breakdown.py, 8 clips x 250 frames)
  ablation_upd/stage_<dtype>.json   per-stage kernel time of the same variants under nsys
  ablation_upd/stage_B1_<dtype>.json  the same for B1 (eager, profiled separately on 2026-11-12, deploy/q3080_b1stage.sh)
  ablation_upd/oracle_<dtype>.json  the zero-cost oracle (oracle_bench.py)
  e2e_full/<dtype>_updatable.json   Ours' whole-video E2E latency (mean of C3 / C6)
Every time is aligned to the E2E: multiplied by Ours' E2E / clip latency (ratios within the table unchanged).
The oracle's latency is C2's minus the oracle's saving over C2 in its own protocol (the saving is conv time only);
its construction is C2's and its convolution C2's minus that saving.
A1 and B2 run on the mainline's plan (2026-11-14; before, on the single-tile C2 plan) and are compared with Ours of the same
session (ours_fix, deploy/q_abl_upd_fix.sh).  A1's clip latency includes the per-frame permutation: its total is
ours_fix's with ours_fix's convolution time replaced by A1's.
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
DEVICES = ("rtx_3080_sm86", "orin_sm87")
TASK = "wildtrack640_yolov8m_select"
DTYPE = "fp16"
OUT_CSV = HERE / "ablation.csv"
# (key, group, label, latency tag, stage tag, reference tag of the ratio)
VARIANTS = [
    ("no_reuse", "construction", "No reuse", "A3_noreuse", "A3_noreuse", "ours"),
    ("shuffled", "compute", "Shuffled order", None, "A1_shuffle_upd", "ours_fix"),
    ("readback", "compute", "Readback \\& launch", "B1_readback", "B1_readback", "ours"),
    ("full_grid", "compute", "Full-layer grid", "B2_fixedgrid_upd", "B2_fixedgrid_upd", "ours_fix"),
    ("dense_tile", "selection", "Dense-tuned config.", "C1m_anchor", "C1m_anchor", "ours"),
    ("speedup_tile", "selection", "Speedup-optimal config.", "C2m_speedup", "C2m_speedup", "ours"),
    ("latency_tile", "selection", "Latency-optimal config.", "C3m_latency", "C3m_latency", "ours"),
    ("kmax", "selection", "Cost-unaware $K_{\\max}$", "D1_kmax", "D1_kmax", "ours"),
    ("oracle", "selection", "Zero-cost oracle", "oracle", None, "ours"),
    ("ours", "", "Ours", "ours", "ours", "ours"),
]


def device_rows(DEVICE):
    d = EV / DEVICE / TASK
    lat = json.loads((d / f"ablation_upd/lat_{DTYPE}.json").read_text())["backends"]
    stage = json.loads((d / f"ablation_upd/stage_{DTYPE}.json").read_text())["backends"]
    stage.update(json.loads((d / f"ablation_upd/stage_B1_{DTYPE}.json").read_text())["backends"])
    oracle = json.loads((d / f"ablation_upd/oracle_{DTYPE}.json").read_text())
    e2e = json.loads((d / f"e2e_full/{DTYPE}_updatable.json").read_text())["backends"]["wiseconv"]["latency_ms"]
    e2e_ours = float(np.mean([e2e["C3"]["mean"], e2e["C6"]["mean"]]))
    clip = {k: v["latency_ms"] for k, v in lat.items()}
    clip["oracle"] = clip["C2_speedup"] - (oracle["ref_ms"] - oracle["oracle_ms"])
    f = e2e_ours / clip["ours"]
    rows = []
    for key, group, label, lt, st, ref in VARIANTS:
        s = stage.get(st, {}).get("stage_us_per_frame", {}) if st else {}
        cons = f * s["construction"] / 1e3 if s else None
        conv = f * s["conv"] / 1e3 if s else None
        if key == "shuffled":  # its clip latency includes the per-frame shuffle: swap only the reference's convolution time
            c2 = stage[ref]["stage_us_per_frame"]
            t = clip[ref] + (s["conv"] - c2["conv"]) / 1e3
            total, ratio = f * t, t / clip[ref]
        elif key == "oracle":  # same graph as C2 apart from the per-layer tile: only convolution time changes
            c2 = stage["C2_speedup"]["stage_us_per_frame"]
            cons = f * c2["construction"] / 1e3
            conv = f * (c2["conv"] / 1e3 - (oracle["ref_ms"] - oracle["oracle_ms"]))
            total, ratio = f * clip[lt], clip[lt] / clip[ref]
        else:
            total, ratio = f * clip[lt], clip[lt] / clip[ref]
        rows.append(dict(variant=key, group=group, variant_label=label, construction_ms=cons, convolution_ms=conv,
                         total_ms=total, ratio=ratio, reference=ref, dtype=DTYPE, device=DEVICE,
                         alignment=f, source=str((d / "ablation_upd").relative_to(REPO))))
    assert abs(rows[-1]["total_ms"] - e2e_ours) < 1e-9  # Ours equals its whole-video E2E latency
    print(f"{DEVICE}: {len(rows)} rows, alignment {f:.3f}")
    return rows


def main() -> None:
    rows = [r for dev in DEVICES if (EV / dev / TASK / f"ablation_upd/lat_{DTYPE}.json").exists() for r in device_rows(dev)]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
