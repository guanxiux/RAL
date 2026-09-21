#!/usr/bin/env python3
"""Build the stream-batching cost-decomposition paper CSV.

Reads ``logs/stream_batch_cost_decomposition/<workload>/<device>/<dtype>/
batch<B>-<system>/summary.json`` for every measured (batch size, system) cell
and emits the per-frame stage decomposition.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "logs" / "stream_batch_cost_decomposition" / "yolov8n" / "3080" / "fp32"
OUT_CSV = HERE / "batch_cost_decomposition.csv"

SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
BATCH_SIZES = (1, 4, 8)
CATEGORIES = ("construction", "convolution", "elementwise", "other")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_rows() -> list[dict]:
    rows = []
    for batch_size in BATCH_SIZES:
        for system in SYSTEMS:
            summary_path = ROOT / f"batch{batch_size}-{system}" / "summary.json"
            data = load(summary_path)
            categories = data["categories_ms_per_frame"]
            total = float(data["formal_latency_ms"])
            row = {
                "batch_size": batch_size,
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "timed_batches": int(data.get("profiled_batches", 0)),
                "timed_frames": int(data.get("timed_frames", 0)),
                "total_latency_ms_per_frame": f"{total:.12g}",
                "actual_batch_latency_ms": f"{float(data.get('kernel_total_ms_per_frame', 0.0)):.12g}",
                "construction_applicable": "true" if system == "wiseconv" else "false",
                "source": relative(summary_path),
            }
            for category in CATEGORIES:
                milliseconds = float(categories.get(category, 0.0))
                row[f"{category}_ms_per_frame"] = f"{milliseconds:.12g}"
                row[f"{category}_percent"] = f"{100.0 * milliseconds / total:.12g}"
            rows.append(row)

    by_key = {(row["batch_size"], row["system"]): row for row in rows}
    for batch_size in BATCH_SIZES:
        competitors = {
            system: float(by_key[(batch_size, system)]["total_latency_ms_per_frame"])
            for system in SYSTEMS if system != "wiseconv"
        }
        fastest = min(competitors, key=competitors.get)
        wise = by_key[(batch_size, "wiseconv")]
        wise["fastest_competing_system"] = fastest
        wise["speedup_vs_fastest_competing"] = (
            competitors[fastest] / float(wise["total_latency_ms_per_frame"])
        )
    for row in rows:
        row.setdefault("fastest_competing_system", "")
        row.setdefault("speedup_vs_fastest_competing", "")
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
