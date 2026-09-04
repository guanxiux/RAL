#!/usr/bin/env python3
"""Emit the two-platform effective-throughput figure data from audited logs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs" / "effective_throughput"
OUT_CSV = HERE / "effective_throughput.csv"

DEVICES = {
    "rtx3080": {
        "label": "RTX 3080",
        "source": LOGS / "3080-streamk-v10-summary" / "summary.json",
    },
    "agx_orin": {
        "label": "AGX Orin",
        "source": LOGS / "agx-orin-v5" / "summary.json",
    },
}
WORKLOADS = ("fireflownet", "yolov8n", "yolov8m", "dynconv_pose")
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}
SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_rows() -> list[dict]:
    rows = []
    frame_populations: dict[str, int] = {}
    for device, device_config in DEVICES.items():
        source = device_config["source"]
        summary = load_json(source)
        if summary.get("schema_version") != 2:
            raise ValueError(f"unsupported summary schema: {source}")
        workloads = summary.get("workloads")
        if set(workloads or {}) != set(WORKLOADS):
            raise ValueError(f"unexpected workload set: {source}")

        for workload in WORKLOADS:
            result = workloads[workload]
            frames = int(result["frames"])
            previous = frame_populations.setdefault(workload, frames)
            if frames != previous:
                raise ValueError(f"frame population differs for {workload}")
            systems = result.get("systems")
            if set(systems or {}) != set(SYSTEMS):
                raise ValueError(f"unexpected system set for {device}/{workload}")
            for system in SYSTEMS:
                entry = systems[system]
                raw = float(entry["raw_throughput_vs_dense"])
                effective = float(entry["effective_throughput_vs_dense"])
                wasted = float(entry["wasted_throughput_vs_dense"])
                if min(raw, effective, wasted) < -1e-12:
                    raise ValueError(
                        f"negative throughput for {device}/{workload}/{system}"
                    )
                if not math.isclose(
                    raw, effective + wasted, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("throughput stack does not close")
                if system == "dense" and not math.isclose(
                    raw, 1.0, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("dense normalization does not close")
                rows.append({
                    "device": device,
                    "device_label": device_config["label"],
                    "workload": workload,
                    "workload_label": WORKLOAD_LABELS[workload],
                    "system": system,
                    "system_label": SYSTEM_LABELS[system],
                    "frames": frames,
                    "latency_ms": f"{float(entry['latency_ms']):.12g}",
                    "useful_compute_ratio": (
                        f"{float(entry['useful_compute_ratio']):.12g}"
                    ),
                    "raw_throughput_vs_dense": f"{raw:.12g}",
                    "effective_throughput_vs_dense": f"{effective:.12g}",
                    "wasted_throughput_vs_dense": f"{wasted:.12g}",
                    "source": relative(source),
                })
    return rows


def main() -> None:
    rows = build_rows()
    expected = len(DEVICES) * len(WORKLOADS) * len(SYSTEMS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)
    print(f"wrote {OUT_CSV}")
    for row in rows:
        print(
            f"{row['device']:<10} {row['workload']:<13} {row['system']:<15} "
            f"effective={float(row['effective_throughput_vs_dense']):.4f} "
            f"wasted={float(row['wasted_throughput_vs_dense']):.4f}"
        )


if __name__ == "__main__":
    main()
