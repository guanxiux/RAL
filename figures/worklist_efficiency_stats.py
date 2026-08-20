#!/usr/bin/env python3
"""Build the WISEConv efficiency/worklist table data from audited artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EFFECTIVE_THROUGHPUT = HERE / "effective_throughput.csv"
WORKLIST_ROOT = (
    REPO / "logs" / "microbenchmarks" / "worklist_statistics" / "3080-full-v2"
)
OUTPUT = HERE / "worklist_efficiency.csv"

WORKLOADS = (
    "fireflownet",
    "yolov8n",
    "yolov8m",
    "dynconv_pose",
)
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}
WORKLIST_SUMMARIES = {
    "fireflownet": (
        WORKLIST_ROOT / "optical_flow" / "wiseconv" / "fireflownet"
        / "summary.json"
    ),
    "yolov8n": WORKLIST_ROOT / "yolov8_mot16" / "yolov8n" / "summary.json",
    "yolov8m": WORKLIST_ROOT / "yolov8_mot16" / "yolov8m" / "summary.json",
    "dynconv_pose": WORKLIST_ROOT / "dynconv_pose.summary.json",
}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def histogram_percentile(histogram: Counter, quantile: float) -> float:
    count = sum(histogram.values())
    if count <= 0:
        raise ValueError("cannot take a percentile of an empty histogram")
    position = (count - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)

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


def workload_records(workload: str, payload: dict) -> list[dict]:
    if workload == "fireflownet":
        sequences = payload.get("sequences")
        if not isinstance(sequences, dict):
            raise ValueError("FireFlowNet summary has no sequence mapping")
        return [row["work"] for row in sequences.values()]
    if workload in ("yolov8n", "yolov8m"):
        backends = payload.get("backends")
        if not isinstance(backends, dict) or set(backends) != {"wiseconv"}:
            raise ValueError(f"{workload} summary has unexpected backends")
        sequences = backends["wiseconv"]
        return [row["work"] for row in sequences.values()]
    return [payload["work"]]


def aggregate_worklists(workload: str) -> dict:
    source = WORKLIST_SUMMARIES[workload]
    records = workload_records(workload, load_json(source))
    if not records:
        raise ValueError(f"{workload} contains no worklist records")

    histogram = Counter()
    layer_names = None
    frames = 0
    invocations = 0
    entries = 0
    for work in records:
        overall = work["worklist_statistics"]
        layers = work["worklist_statistics_by_layer"]
        if overall.get("schema_version") != 2:
            raise ValueError(f"unsupported worklist schema for {workload}")
        if overall.get("aggregation") != (
            "call_equal_across_layer_frame_invocations"
        ):
            raise ValueError(f"unexpected worklist aggregation for {workload}")
        current_names = set(layers)
        if layer_names is None:
            layer_names = current_names
        elif current_names != layer_names:
            raise ValueError(f"layer names change across {workload} sequences")

        measured_frames = int(overall["measured_frames"])
        layer_invocations = sum(int(row["invocations"]) for row in layers.values())
        layer_entries = sum(
            int(row["sum_worklist_entries"]) for row in layers.values()
        )
        if layer_invocations != int(overall["invocations"]):
            raise ValueError(f"layer invocation total does not close for {workload}")
        if layer_entries != int(overall["sum_worklist_entries"]):
            raise ValueError(f"layer worklist total does not close for {workload}")
        if any(
            int(row["invocations"]) != measured_frames
            for row in layers.values()
        ):
            raise ValueError(f"a {workload} layer is not invoked once per frame")

        sequence_histogram = Counter()
        for pair in overall.get("worklist_length_histogram", ()):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"invalid worklist histogram for {workload}")
            length, count = map(int, pair)
            if length < 0 or count <= 0 or length in sequence_histogram:
                raise ValueError(f"invalid worklist histogram bin for {workload}")
            sequence_histogram[length] = count
        if sum(sequence_histogram.values()) != layer_invocations:
            raise ValueError(f"histogram invocation total does not close for {workload}")
        if sum(
            length * count for length, count in sequence_histogram.items()
        ) != layer_entries:
            raise ValueError(f"histogram worklist total does not close for {workload}")
        if not math.isclose(
            histogram_percentile(sequence_histogram, 0.10),
            float(overall["p10_worklist_length"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            histogram_percentile(sequence_histogram, 0.90),
            float(overall["p90_worklist_length"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"histogram percentiles do not close for {workload}")

        frames += measured_frames
        invocations += layer_invocations
        entries += layer_entries
        histogram.update(sequence_histogram)

    if not layer_names or invocations != frames * len(layer_names):
        raise ValueError(f"invalid layer-frame population for {workload}")
    return {
        "frames": frames,
        "layers": len(layer_names),
        "invocations": invocations,
        "p10_worklist_length": histogram_percentile(histogram, 0.10),
        "p50_worklist_length": histogram_percentile(histogram, 0.50),
        "p90_worklist_length": histogram_percentile(histogram, 0.90),
        "worklist_source": relative(source),
    }


def load_wiseconv_efficiency() -> dict[str, dict]:
    with EFFECTIVE_THROUGHPUT.open(encoding="utf-8", newline="") as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["system"] == "wiseconv"
        ]
    by_workload = {row["workload"]: row for row in rows}
    if set(by_workload) != set(WORKLOADS) or len(rows) != len(WORKLOADS):
        raise ValueError("effective-throughput CSV has unexpected WISEConv rows")
    return by_workload


def build_rows() -> list[dict]:
    efficiency = load_wiseconv_efficiency()
    rows = []
    for workload in WORKLOADS:
        worklist = aggregate_worklists(workload)
        eta = efficiency[workload]
        if int(eta["frames"]) != worklist["frames"]:
            raise ValueError(f"frame count mismatch for {workload}")
        eta_construct = float(eta["eta_coverage_frame_mean"])
        eta_compute = float(eta["eta_packing_frame_mean"])
        eta_total = float(eta["useful_compute_ratio_frame_mean"])
        if not all(0.0 <= value <= 1.0 for value in (
            eta_construct, eta_compute, eta_total
        )):
            raise ValueError(f"invalid efficiency for {workload}")
        rows.append({
            "workload": workload,
            "workload_label": WORKLOAD_LABELS[workload],
            **worklist,
            # E/W and W/C are named after their owning stages in the paper.
            "eta_construct_frame_mean": eta_construct,
            "eta_compute_frame_mean": eta_compute,
            "useful_compute_ratio_frame_mean": eta_total,
            "effective_throughput_source": relative(EFFECTIVE_THROUGHPUT),
            "aggregation": "percentiles_across_layer_frame_invocations",
        })
    return rows


def main() -> None:
    rows = build_rows()
    fieldnames = list(rows[0])
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT}")
    for row in rows:
        print(
            f"{row['workload_label']:<12} "
            f"eta=({row['eta_construct_frame_mean']:.4f}, "
            f"{row['eta_compute_frame_mean']:.4f}, "
            f"{row['useful_compute_ratio_frame_mean']:.4f}) "
            f"n_W=[{row['p10_worklist_length']:.1f}, "
            f"{row['p50_worklist_length']:.1f}, "
            f"{row['p90_worklist_length']:.1f}]"
        )


if __name__ == "__main__":
    main()
