#!/usr/bin/env python3
"""Build the WISEConv efficiency/worklist table data from the unified logs.

For each workload (RTX 3080, fp32) it combines:

  * ``logs/effective_throughput/3080/summary.json`` for the position-level
    eta decomposition (E/W construct, W/C compute, E/C total) and frames;
  * ``logs/work/<workload>/3080/fp32/<run>-wiseconv/summary.json`` for the
    per-invocation output-worklist length histograms, pooled across sequences
    to the requested percentiles.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs"
DEVICE = "3080"
DTYPE = "fp32"
EFFECTIVE_THROUGHPUT = LOGS / "effective_throughput" / DEVICE / "summary.json"
OUTPUT = HERE / "worklist_efficiency.csv"

WORKLOADS = ("fireflownet", "yolov8n", "yolov8m", "dynconv_pose")
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}
RUN_PREFIX = {
    "fireflownet": "full-r3",
    "yolov8n": "yolov8n-full-r3",
    "yolov8m": "yolov8m-full-r3",
    "dynconv_pose": "full-s0125-r3",
}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def histogram_percentile(histogram: Counter, quantile: float) -> float:
    count = sum(histogram.values())
    if count <= 0:
        raise ValueError("cannot take a percentile of an empty histogram")
    position = (count - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, count - 1)

    def value_at(index: int) -> int:
        cumulative = 0
        for value in sorted(histogram):
            cumulative += histogram[value]
            if index < cumulative:
                return value
        raise RuntimeError("histogram order statistic is out of range")

    lower_value = value_at(lower)
    if lower == upper:
        return float(lower_value)
    fraction = position - lower
    return lower_value * (1.0 - fraction) + value_at(upper) * fraction


def aggregate_worklists(workload: str) -> dict:
    path = (LOGS / "work" / workload / DEVICE / DTYPE
            / f"{RUN_PREFIX[workload]}-wiseconv" / "summary.json")
    sequences = load_json(path).get("sequences") or {}
    if not sequences:
        raise ValueError(f"no work sequences in {path}")
    histogram: Counter = Counter()
    frames = 0
    for sequence in sequences.values():
        block = sequence.get("worklist_length")
        if not block or not block.get("histogram"):
            raise ValueError(
                f"worklist_length histogram missing in {path}; rerun the "
                "WISEConv work pass")
        for length, count in block["histogram"]:
            histogram[int(length)] += int(count)
        frames += int(sequence.get("frames") or sequence.get("frame_count") or 0)
    return {
        "frames": frames,
        "invocations": sum(histogram.values()),
        "p10_worklist_length": histogram_percentile(histogram, 0.10),
        "p50_worklist_length": histogram_percentile(histogram, 0.50),
        "p90_worklist_length": histogram_percentile(histogram, 0.90),
        "worklist_source": relative(path),
    }


def build_rows() -> list[dict]:
    effective = load_json(EFFECTIVE_THROUGHPUT)["workloads"]
    rows = []
    for workload in WORKLOADS:
        worklist = aggregate_worklists(workload)
        entry = effective[workload]
        decomposition = entry["wiseconv_decomposition"]
        rows.append({
            "workload": workload,
            "workload_label": WORKLOAD_LABELS[workload],
            **worklist,
            "eta_construct_frame_mean": float(
                decomposition["eta_coverage_E_over_W"]),
            "eta_compute_frame_mean": float(
                decomposition["eta_packing_W_over_C"]),
            "useful_compute_ratio_frame_mean": float(
                decomposition["useful_compute_ratio_E_over_C"]),
            "effective_throughput_source": relative(EFFECTIVE_THROUGHPUT),
            "aggregation": "percentiles_across_layer_frame_invocations",
        })
    return rows


def main() -> None:
    rows = build_rows()
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
