#!/usr/bin/env python3
"""Build the two-platform YOLOv8n cost-decomposition paper CSV."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_CSV = HERE / "cost_decomposition.csv"
DEVICES = {
    "rtx3080": {
        "label": "RTX 3080",
        "source": REPO / "logs" / "microbenchmarks" / "cost_decomposition"
        / "3080-random100-streamk-v1-semantic"
        / "cost_decomposition.summary.json",
    },
    "agx_orin": {
        "label": "AGX Orin",
        "source": REPO / "logs" / "microbenchmarks" / "cost_decomposition"
        / "agx-yolov8n-common-v5" / "summary.json",
    },
}
SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather",
    "wiseconv": "WISEConv",
}
CATEGORIES = ("construction", "convolution", "elementwise", "other")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalized_rows(summary: dict, source: Path) -> list[dict]:
    """Return one workload-level row per system.

    The new semantic profiler stores three equally sized activity-bin rows
    for each workload/system.  The paper table intentionally has no activity
    bins, so combine those rows with their selected-frame counts before
    validating and emitting the table.  Older compact summaries already have
    one row per system and are accepted unchanged for the AGX artifact.
    """

    rows = summary.get("rows") or []
    if not rows:
        raise ValueError(f"summary has no rows: {source}")
    if any("workload" in row for row in rows):
        rows = [row for row in rows if row.get("workload") == "yolov8n"]
        by_system: dict[str, list[dict]] = {}
        for row in rows:
            by_system.setdefault(str(row["system"]), []).append(row)
        if any(len(group) == 0 for group in by_system.values()):
            raise ValueError(f"empty workload row group: {source}")
        combined = []
        for system, group in by_system.items():
            frames = sum(int(row["frames"]) for row in group)
            if frames <= 0:
                raise ValueError(f"invalid selected-frame count: {source}")
            total = sum(
                int(row["frames"]) * float(row["total_latency_ms"])
                for row in group
            ) / frames
            stage_ms = {
                category: sum(
                    int(row["frames"]) * float(row["stage_ms"][category])
                    for row in group
                ) / frames
                for category in CATEGORIES
            }
            combined.append({
                "system": system,
                "label": system,
                "formal_frames": 2572,
                "stage_sample_frames": frames,
                "total_ms": total,
                "stage_ms": stage_ms,
                "stage_percent": {
                    category: 100.0 * stage_ms[category] / total
                    for category in CATEGORIES
                },
            })
        return combined
    return rows


def build_rows() -> list[dict]:
    output = []
    for device, config in DEVICES.items():
        source = config["source"]
        summary = load_json(source)
        rows = normalized_rows(summary, source)
        by_system = {row["system"]: row for row in rows or ()}
        if set(by_system) != set(SYSTEMS) or len(rows or ()) != len(SYSTEMS):
            raise ValueError(f"unexpected systems: {source}")
        for system in SYSTEMS:
            row = by_system[system]
            if int(row["formal_frames"]) != 2572:
                raise ValueError(f"unexpected frame count: {source}")
            total = float(row["total_ms"])
            stages = {name: float(row["stage_ms"][name]) for name in CATEGORIES}
            if not math.isclose(
                sum(stages.values()), total, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"stage times do not close: {device}/{system}")
            output_row = {
                "device": device,
                "device_label": config["label"],
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "formal_frames": 2572,
                "stage_sample_frames": int(row["stage_sample_frames"]),
                "total_ms": f"{total:.12g}",
                "source": relative(source),
            }
            for category in CATEGORIES:
                milliseconds = stages[category]
                percent = float(row["stage_percent"][category])
                if not math.isclose(
                    percent, 100.0 * milliseconds / total,
                    rel_tol=0.0, abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"stage percentage does not close: "
                        f"{device}/{system}/{category}"
                    )
                output_row[f"{category}_ms"] = f"{milliseconds:.12g}"
                output_row[f"{category}_percent"] = f"{percent:.12g}"
            output.append(output_row)
    return output


def main() -> None:
    rows = build_rows()
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
            f"{row['device']:<10} {row['system']:<15} "
            f"total={float(row['total_ms']):.3f} "
            f"cons={float(row['construction_ms']):.3f} "
            f"conv={float(row['convolution_ms']):.3f}"
        )


if __name__ == "__main__":
    main()
