#!/usr/bin/env python3
"""Build the cost-decomposition paper CSV from the unified cost logs.

Each ``logs/latency_cost_decomposition/<workload>/<device>/<dtype>/<run>/``
summary records one deployed system (``backend``) with per-category GPU time.
Devices/systems without a summary are omitted; the table generator renders the
missing cells as dashes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs" / "latency_cost_decomposition"
OUT_CSV = HERE / "cost_decomposition.csv"
MODEL = "yolov8n"
DTYPE = "fp32"

DEVICES = {
    "3080": ("rtx3080", "RTX 3080"),
    "agx-orin": ("agx_orin", "AGX Orin"),
}
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather",
    "dynconv": "Gather",
    "wiseconv": "WISEConv",
}
CATEGORIES = ("construction", "convolution", "elementwise", "other")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def build_rows() -> list[dict]:
    rows = []
    for device_dir, (device, label) in DEVICES.items():
        base = LOGS / MODEL / device_dir / DTYPE
        if not base.is_dir():
            continue
        for run in sorted(p for p in base.iterdir() if p.is_dir()):
            summary = run / "summary.json"
            if not summary.is_file():
                continue
            data = json.loads(summary.read_text(encoding="utf-8"))
            system = data.get("backend") or run.name.split("-")[-1]
            total = float(data["formal_latency_ms"])
            categories = data["categories_ms"]
            frames = data.get("frames")
            row = {
                "device": device,
                "device_label": label,
                "system": system,
                "system_label": SYSTEM_LABELS.get(system, system),
                "formal_frames": "" if frames is None else int(frames),
                "stage_sample_frames": "" if frames is None else int(frames),
                "total_ms": f"{total:.12g}",
                "source": relative(summary),
            }
            for category in CATEGORIES:
                milliseconds = float(categories.get(category, 0.0))
                row[f"{category}_ms"] = f"{milliseconds:.12g}"
                row[f"{category}_percent"] = f"{100.0 * milliseconds / total:.12g}"
            rows.append(row)
    if not rows:
        raise ValueError(f"no cost-decomposition summaries under {LOGS}")
    return rows


def main() -> None:
    rows = build_rows()
    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
