#!/usr/bin/env python3
"""Build the paper-facing peak-memory table data from the unified logs.

Reads ``logs/peak_memory/<workload>/<device>/<dtype>/<run>/summary.json`` for
every workload/system cell and emits ``memory.csv``.  The reported value is the
peak live allocation (``peak_allocated_mb``), i.e. the high-water mark of tensors
actually alive on the device, rather than the allocator-reserved pool (which is
quantized to 2~MiB segments).
DynConv's gather-scatter backend is stored under the ``dynconv`` cell key;
every other workload uses ``gather_scatter``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs"
OUT_CSV = HERE / "memory.csv"

WORKLOADS = (
    ("fireflownet", "FireFlowNet"),
    ("yolov8n", "YOLOv8n"),
    ("yolov8m", "YOLOv8m"),
    ("dynconv", "DynConv"),
)
# Summary cell key for each column.
COLUMNS = {
    "dense": "dense",
    "tile_skip": "tile_skip",
    "gather_scatter": "gather_scatter",
    "wiseconv": "wiseconv",
}
GATHER_ALIASES = ("gather_scatter", "dynconv")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def cell_value(cells: dict, key: str) -> float | None:
    if key == "gather_scatter":
        for alias in GATHER_ALIASES:
            if alias in cells:
                return float(cells[alias]["peak_allocated_mb"])
        return None
    if key not in cells:
        return None
    return float(cells[key]["peak_allocated_mb"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="3080")
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--run", default="full")
    args = parser.parse_args()

    rows = []
    for workload, label in WORKLOADS:
        path = LOGS / "peak_memory" / workload / args.device / args.dtype \
            / args.run / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        cells = load_json(path)["cells"]
        row = {"workload": workload, "workload_label": label}
        for column, key in COLUMNS.items():
            value = cell_value(cells, key)
            row[column] = "" if value is None else f"{value:.12g}"
        rows.append(row)

    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)
    print(f"wrote {OUT_CSV}")
    for row in rows:
        print(f"{row['workload_label']:<12} dense={row['dense']:<7} "
              f"tile={row['tile_skip']:<7} gather={row['gather_scatter']:<7} "
              f"wise={row['wiseconv']}")


if __name__ == "__main__":
    main()
