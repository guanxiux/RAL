#!/usr/bin/env python3
"""Ablation on every workload, device and precision (2026-11-14, deploy/q_ablation2.sh): one row per (device, task,
dtype, variant) with the total latency, construction and convolution time, and the ratio to the variant's reference,
computed as ablation_stats.py does for the single YOLOv8m / RTX 3080 / FP16 table.

Sources: logs/wise2_abl2/<device>/<task>/{lat,stage}_<dtype>.json (ablation_bench.py; stages profiled in parts
<tag>@<k>, merged weighted by frames) and oracle_<dtype>.json (oracle_bench.py).  Times are the sample's (workloads.py
"sample" split), not aligned to the E2E; the ratios are the comparison.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs/wise2_abl2"
OUT_CSV = HERE / "ablation2.csv"
DEVICES = {"rtx_3080_sm86": "RTX 3080", "orin_sm87": "AGX Orin"}
TASKS = {"mvsec_fireflownet": "FireFlowNet", "wildtrack640_yolov8m": "YOLOv8m", "mpii_dynconv": "DynConv"}
# (key, latency tag, stage tag, reference tag)
VARIANTS = [
    ("no_reuse", "A3_noreuse", "A3_noreuse", "ours"),
    ("shuffled", None, "A1_shuffle_C2", "C2_speedup"),
    ("readback", "B1_readback", "B1_readback", "ours"),
    ("full_grid", "B2_fixedgrid", "B2_fixedgrid", "C2_speedup"),
    ("dense_tile", "C1m_anchor", "C1m_anchor", "ours"),
    ("speedup_tile", "C2m_speedup", "C2m_speedup", "ours"),
    ("latency_tile", "C3m_latency", "C3m_latency", "ours"),
    ("kmax", "D1_kmax", "D1_kmax", "ours"),
    ("oracle", "oracle", None, "ours"),
    ("ours", "ours", "ours", "ours"),
]


def merged_stages(path):
    """{tag: {stage: us per frame}} of a stage file, the parts <tag>@<k> merged weighted by their frames."""
    acc = {}
    for key, r in json.loads(path.read_text())["backends"].items():
        if "stage_us_per_frame" not in r:
            continue
        tag = key.split("@")[0]
        a = acc.setdefault(tag, {"frames": 0, "us": {}})
        a["frames"] += r["frames"]
        for s, v in r["stage_us_per_frame"].items():
            a["us"][s] = a["us"].get(s, 0.0) + v * r["frames"]
    return {t: {s: v / a["frames"] for s, v in a["us"].items()} for t, a in acc.items()}


def main():
    rows = []
    for dev, dev_label in DEVICES.items():
        for task, task_label in TASKS.items():
            for p in ("fp32", "fp16"):
                d = LOGS / dev / task
                lat_f, st_f, or_f = d / f"lat_{p}.json", d / f"stage_{p}.json", d / f"oracle_{p}.json"
                if not lat_f.exists():
                    continue
                lat = {k: v["latency_ms"] for k, v in json.loads(lat_f.read_text())["backends"].items()}
                stage = merged_stages(st_f) if st_f.exists() else {}
                oracle = json.loads(or_f.read_text()) if or_f.exists() else None
                if oracle and "C2_speedup" in lat:
                    lat["oracle"] = lat["C2_speedup"] - (oracle["ref_ms"] - oracle["oracle_ms"])
                for key, lt, st, ref in VARIANTS:
                    s = stage.get(st, {}) if st else {}
                    cons = s.get("construction", 0.0) / 1e3 if s else None
                    conv = s.get("conv", 0.0) / 1e3 if s else None
                    if key == "shuffled":  # its latency includes the per-frame shuffle: swap only C2's convolution
                        if not s or "C2_speedup" not in stage or ref not in lat:
                            continue
                        total = lat[ref] + conv - stage["C2_speedup"]["conv"] / 1e3
                    elif key == "oracle":
                        if "oracle" not in lat or "C2_speedup" not in stage:
                            continue
                        c2 = stage["C2_speedup"]
                        cons = c2.get("construction", 0.0) / 1e3
                        conv = c2["conv"] / 1e3 - (oracle["ref_ms"] - oracle["oracle_ms"])
                        total = lat["oracle"]
                    else:
                        if lt not in lat:
                            continue
                        total = lat[lt]
                    if ref not in lat:
                        continue
                    rows.append(dict(device=dev_label, task=task_label, dtype=p.upper(), variant=key,
                                     construction_ms=cons, convolution_ms=conv, total_ms=total,
                                     ratio=total / lat[ref], reference=ref,
                                     source=str(d.relative_to(REPO))))
    with OUT_CSV.open("w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")
    # console summary: ratio per variant, one column per (device, task, dtype)
    cols = sorted({(r["device"], r["task"], r["dtype"]) for r in rows})
    print("variant        " + " ".join(f"{d[:4]}/{t[:4]}/{p}" for d, t, p in cols))
    for key, *_ in VARIANTS:
        vals = {(r["device"], r["task"], r["dtype"]): r["ratio"] for r in rows if r["variant"] == key}
        print(f"{key:14s} " + " ".join(f"{vals[c]:13.3f}" if c in vals else " " * 13 for c in cols))


if __name__ == "__main__":
    main()
