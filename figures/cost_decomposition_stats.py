#!/usr/bin/env python3
"""Validate the formal YOLOv8n cost decomposition and emit paper data.

The formal log contains one row for each activity bin and execution system.
This statistics script verifies the profiling protocol and the stage-total
identities before copying the paper-facing fields to a compact CSV.  The LaTeX
table is generated from that CSV by ``make_cost_decomposition_table.py``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = (
    REPO
    / "logs"
    / "microbenchmarks"
    / "cost_decomposition"
    / "3080-yolov8n-two-bin-v3-semantic"
)
SOURCE_CSV = RUN / "cost_decomposition.csv"
SOURCE_SUMMARY = RUN / "cost_decomposition.summary.json"
SOURCE_MANIFEST = RUN / "manifest.json"
OUT_CSV = HERE / "cost_decomposition.csv"

BINS = ("low", "high")
SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
CATEGORIES = ("construction", "convolution", "elementwise", "other")
EXPECTED_FRAMES_PER_BIN = 1286
EXPECTED_TRACE_FRAMES = 2572


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def as_float(row: dict, field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} in source row {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in source row {row}")
    return value


def validate_protocol(summary: dict, manifest: dict) -> None:
    if summary.get("kind") != "yolov8n_two_bin_full_trace_cost_decomposition":
        raise ValueError("unexpected cost-decomposition summary kind")
    if manifest.get("kind") != "yolov8n_two_bin_cost_manifest":
        raise ValueError("unexpected cost-decomposition manifest kind")
    if summary.get("device", {}).get("name") != "NVIDIA GeForce RTX 3080":
        raise ValueError("the paper table must use the RTX 3080 run")
    workload = manifest.get("workload", {})
    if workload.get("name") != "yolov8n":
        raise ValueError("the paper table must use YOLOv8n")
    if int(workload.get("complete_trace_frames", -1)) != EXPECTED_TRACE_FRAMES:
        raise ValueError("the cost decomposition does not cover the full trace")

    aggregation = summary.get("aggregation", {})
    expected = {
        "activity_metric": "required_macs / dense_macs",
        "global_bins": "median split over every frame in the complete held-out trace",
        "profiled_frames": "every held-out frame",
        "frame_weighting": "arithmetic mean over every frame in each bin",
        "formal_frame_latency": "median of three formal CUDA-event rounds",
        "stage_scaling": (
            "formal_frame_ms * nsys_stage_timeline_ns / "
            "nsys_full_frame_timeline_ns"
        ),
    }
    for key, value in expected.items():
        if aggregation.get(key) != value:
            raise ValueError(f"unexpected aggregation protocol for {key}")

    category_keys = tuple(item.get("key") for item in summary.get("categories", ()))
    if category_keys != CATEGORIES:
        raise ValueError(f"unexpected stage categories: {category_keys}")


def validate_source_rows(rows: list[dict], summary: dict, manifest: dict) -> None:
    expected_keys = {(activity_bin, system) for activity_bin in BINS for system in SYSTEMS}
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row.get("workload") != "yolov8n":
            raise ValueError("unexpected workload in cost-decomposition CSV")
        key = (row.get("activity_bin"), row.get("system"))
        if key in by_key:
            raise ValueError(f"duplicate cost-decomposition row: {key}")
        by_key[key] = row
    if set(by_key) != expected_keys:
        raise ValueError("cost-decomposition CSV has missing or unexpected rows")

    summary_rows = {
        (row["activity_bin"], row["system"]): row
        for row in summary.get("rows", ())
    }
    if set(summary_rows) != expected_keys:
        raise ValueError("summary rows do not match the expected bins and systems")

    for bin_index, activity_bin in enumerate(BINS):
        population = manifest["workload"]["bin_population"][activity_bin]
        if int(population["frames"]) != EXPECTED_FRAMES_PER_BIN:
            raise ValueError(f"unexpected population for {activity_bin}")
        bin_means = set()
        for system in SYSTEMS:
            row = by_key[(activity_bin, system)]
            if int(row["activity_bin_index"]) != bin_index:
                raise ValueError(f"wrong index for {activity_bin}/{system}")
            if int(row["frames"]) != EXPECTED_FRAMES_PER_BIN:
                raise ValueError(f"wrong frame count for {activity_bin}/{system}")

            required_mean = as_float(row, "required_work_ratio_mean")
            bin_means.add(round(required_mean, 15))
            if not math.isclose(
                required_mean,
                float(population["required_work_ratio_mean"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(f"bin mean mismatch for {activity_bin}/{system}")

            total_ms = as_float(row, "total_latency_ms")
            if total_ms <= 0.0:
                raise ValueError(f"non-positive latency for {activity_bin}/{system}")
            stage_total = sum(as_float(row, f"{category}_ms") for category in CATEGORIES)
            if not math.isclose(stage_total, total_ms, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"stage total does not close for {activity_bin}/{system}")

            percent_total = 0.0
            for category in CATEGORIES:
                stage_ms = as_float(row, f"{category}_ms")
                percent = as_float(row, f"{category}_percent")
                expected_percent = 100.0 * stage_ms / total_ms
                if not math.isclose(
                    percent, expected_percent, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"stage percentage mismatch for "
                        f"{activity_bin}/{system}/{category}"
                    )
                percent_total += percent
            if not math.isclose(percent_total, 100.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"stage percentages do not close for {activity_bin}/{system}")

            construction_ms = as_float(row, "construction_ms")
            if system == "wiseconv":
                if construction_ms <= 0.0:
                    raise ValueError("WISEConv construction must be measured")
            elif construction_ms != 0.0:
                raise ValueError(
                    f"construction must be inapplicable for {activity_bin}/{system}"
                )

            summary_row = summary_rows[(activity_bin, system)]
            summary_total = float(summary_row["total_latency_ms"])
            if not math.isclose(total_ms, summary_total, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"CSV/summary latency mismatch for {activity_bin}/{system}")
            for category in CATEGORIES:
                if not math.isclose(
                    as_float(row, f"{category}_ms"),
                    float(summary_row["stage_ms"][category]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"CSV/summary stage mismatch for "
                        f"{activity_bin}/{system}/{category}"
                    )
        if len(bin_means) != 1:
            raise ValueError(f"systems do not share the {activity_bin} bin population")


def paper_rows(rows: list[dict]) -> list[dict]:
    by_key = {(row["activity_bin"], row["system"]): row for row in rows}
    output = []
    for bin_index, activity_bin in enumerate(BINS):
        for system in SYSTEMS:
            row = by_key[(activity_bin, system)]
            output_row = {
                "activity_bin": activity_bin,
                "activity_bin_index": bin_index,
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "frames": int(row["frames"]),
                "required_work_ratio_mean": f"{as_float(row, 'required_work_ratio_mean'):.12g}",
                "total_latency_ms": f"{as_float(row, 'total_latency_ms'):.12g}",
                "construction_applicable": "true" if system == "wiseconv" else "false",
            }
            for category in CATEGORIES:
                output_row[f"{category}_ms"] = f"{as_float(row, f'{category}_ms'):.12g}"
                output_row[f"{category}_percent"] = (
                    f"{as_float(row, f'{category}_percent'):.12g}"
                )
            output_row["source"] = str(SOURCE_CSV.relative_to(REPO))
            output.append(output_row)
    return output


def write_csv(rows: list[dict]) -> None:
    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)


def main() -> None:
    summary = read_json(SOURCE_SUMMARY)
    manifest = read_json(SOURCE_MANIFEST)
    validate_protocol(summary, manifest)
    with SOURCE_CSV.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    validate_source_rows(rows, summary, manifest)
    output = paper_rows(rows)
    write_csv(output)

    print(f"wrote {OUT_CSV}")
    print("bin   system             total   cons.    conv.   elem.   other")
    for row in output:
        construction = (
            f"{float(row['construction_ms']):.3f}"
            if row["construction_applicable"] == "true"
            else "  N/A"
        )
        print(
            f"{row['activity_bin']:<6}{row['system_label']:<19}"
            f"{float(row['total_latency_ms']):>6.3f}"
            f"{construction:>8}"
            f"{float(row['convolution_ms']):>8.3f}"
            f"{float(row['elementwise_ms']):>8.3f}"
            f"{float(row['other_ms']):>8.3f}"
        )


if __name__ == "__main__":
    main()
