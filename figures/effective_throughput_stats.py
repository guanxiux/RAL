#!/usr/bin/env python3
"""Aggregate frame-paired RTX 3080 effective-throughput measurements.

The operator-level work quantities come from the existing work-only replays,
while latency comes from the formal full-model logs.  For every held-out frame
``f`` and backend ``b`` this script computes

    raw_b,f = (I_b,f / D_f) * (t_dense,f / t_b,f)
    eff_b,f = (E_f   / D_f) * (t_dense,f / t_b,f)

where ``D`` and ``E`` are dense and policy-required convolution MACs and
``I_b`` is the backend's issued convolution MACs.  The plotted workload value
is the arithmetic mean of these frame-level ratios.  Three latency rounds are
first reduced to a per-frame median, matching the per-frame records already
stored by the formal FireFlowNet and YOLO runs; pose applies the same reduction
to its three round records.

This is a statistics script: it reads formal logs and writes a CSV consumed by
``plot_effective_throughput.py``.  It never modifies the formal logs.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BENCHMARKS = REPO / "benchmarks"
LOGS = REPO / "logs"
OUT_CSV = HERE / "effective_throughput.csv"

# Reuse the audited work/provenance loaders rather than reimplementing their
# application-specific schemas in the figure pipeline.
sys.path.insert(0, str(BENCHMARKS))
import effective_throughput as work_stats  # noqa: E402


SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
WORKLOAD_LABELS = {
    "fireflownet": "FireFlowNet",
    "yolov8n": "YOLOv8n",
    "yolov8m": "YOLOv8m",
    "dynconv_pose": "DynConv",
}
ISSUED_FIELD = {
    "dense": "D",
    "tile_skip": "A",
    "gather_scatter": "E",
    "wiseconv": "C",
}


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def frame_key(sequence: str, frame_index: int) -> tuple[str, int]:
    return str(sequence), int(frame_index)


def insert_unique(mapping: dict, key, value, source: Path) -> None:
    if key in mapping:
        raise ValueError(f"duplicate frame {key} in {source}")
    mapping[key] = value


def checked_frame_median(row: dict, source: Path) -> float:
    samples = row.get("gpu_latency_ms")
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError(f"expected three latency rounds in {source}")
    samples = [float(value) for value in samples]
    if min(samples) <= 0.0:
        raise ValueError(f"non-positive latency in {source}")
    measured = statistics.median(samples)
    stored = float(row["gpu_latency_median_ms"])
    if not math.isclose(measured, stored, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"stored frame median does not close in {source}")
    return stored


def load_aligned_work(workload: str, config: dict) -> tuple[list[dict], list[str]]:
    """Return frame-aligned D/E/A/W/C convolution-MAC records."""
    work_stats.validate_work_provenance(workload, config)
    wise = work_stats.load_wise_work(config)
    sequences = list(wise["sequences"])
    frame_counts = {
        sequence: sum(
            1 for row in wise["frame_records"] if row["sequence"] == sequence
        )
        for sequence in sequences
    }
    tile = work_stats.load_tile_work(config["tile_work"], sequences, frame_counts)

    wise_by_key = {}
    for row in wise["frame_records"]:
        key = frame_key(row["sequence"], row["frame_index"])
        insert_unique(wise_by_key, key, row, config["wise_work"])
    tile_by_key = {}
    for row in tile["frame_records"]:
        key = frame_key(row["sequence"], row["frame_index"])
        insert_unique(tile_by_key, key, row, config["tile_work"])
    if set(wise_by_key) != set(tile_by_key):
        missing_tile = sorted(set(wise_by_key) - set(tile_by_key))[:3]
        missing_wise = sorted(set(tile_by_key) - set(wise_by_key))[:3]
        raise ValueError(
            f"work-frame mismatch for {workload}: "
            f"missing tile={missing_tile}, missing WISEConv={missing_wise}"
        )

    aligned = []
    for key, wise_row in wise_by_key.items():
        tile_row = tile_by_key[key]
        dense = float(wise_row["dense_macs"])
        required = float(wise_row["required_macs"])
        covered = float(wise_row["covered_macs"])
        issued = float(wise_row["issued_macs"])
        tile_issued = float(tile_row["issued_macs"])
        tolerance = max(1.0, dense * 1e-9)
        if dense <= 0.0:
            raise ValueError(f"non-positive dense work for {workload}/{key}")
        if abs(float(tile_row["dense_macs"]) - dense) > tolerance:
            raise ValueError(f"dense-work mismatch for {workload}/{key}")
        if required > covered + tolerance or covered > issued + tolerance:
            raise ValueError(f"invalid E/W/C ordering for {workload}/{key}")
        aligned.append(
            {
                "key": key,
                "D": dense,
                "E": required,
                "A": tile_issued,
                "W": covered,
                "C": issued,
            }
        )

    expected = int(wise["frames"])
    if len(aligned) != expected or len(aligned) != int(tile["frames"]):
        raise ValueError(f"work frame count mismatch for {workload}")
    return aligned, [wise["source"], tile["source"]]


def load_flow_latency(config: dict, sequences: list[str]) -> tuple[dict, list[str]]:
    values = {system: {} for system in SYSTEMS}
    sources = []
    for system in SYSTEMS:
        run_dir = LOGS / "optical_flow" / system / config["full_run"]
        for sequence in sequences:
            path = run_dir / f"latency_work_curve_{sequence}.jsonl"
            sources.append(relative(path))
            for row in read_jsonl(path):
                key = frame_key(sequence, row["frame_index"])
                insert_unique(
                    values[system], key, checked_frame_median(row, path), path
                )
    return values, sources


def load_yolo_latency(config: dict, sequences: list[str]) -> tuple[dict, list[str]]:
    path = config["full_dir"] / "frames.jsonl"
    values = {system: {} for system in SYSTEMS}
    sequence_set = set(sequences)
    for row in read_jsonl(path):
        system = row.get("backend")
        sequence = row.get("sequence")
        if system not in values or sequence not in sequence_set:
            continue
        key = frame_key(sequence, row["frame_index"])
        insert_unique(values[system], key, checked_frame_median(row, path), path)
    return values, [relative(path)]


def load_pose_latency(config: dict, sequences: list[str]) -> tuple[dict, list[str]]:
    if len(sequences) != 1:
        raise ValueError(f"expected one pose stream, found {sequences}")
    run_dir = config["full_dir"]
    filenames = {
        "dense": "dense.jsonl",
        "tile_skip": "tile_skip.jsonl",
        "gather_scatter": "dynconv.jsonl",
        "wiseconv": "wiseconv.jsonl",
    }
    values = {system: {} for system in SYSTEMS}
    sources = []
    for system, filename in filenames.items():
        path = run_dir / filename
        sources.append(relative(path))
        samples = defaultdict(dict)
        for row in read_jsonl(path):
            if row.get("kind") != "latency":
                continue
            frame = int(row["frame"])
            round_index = int(row["round"])
            if round_index in samples[frame]:
                raise ValueError(f"duplicate pose round/frame in {path}")
            samples[frame][round_index] = float(row["gpu_ms"])
        for frame, round_values in samples.items():
            if set(round_values) != {0, 1, 2}:
                raise ValueError(f"missing pose latency round for frame {frame}")
            ordered = [round_values[index] for index in range(3)]
            if min(ordered) <= 0.0:
                raise ValueError(f"non-positive pose latency in {path}")
            key = frame_key(sequences[0], frame)
            insert_unique(values[system], key, statistics.median(ordered), path)
    return values, sources


def load_latency_frames(config: dict, sequences: list[str]):
    if config["kind"] == "optical_flow":
        return load_flow_latency(config, sequences)
    if config["kind"] == "yolo":
        return load_yolo_latency(config, sequences)
    return load_pose_latency(config, sequences)


def safe_ratio(numerator: float, denominator: float) -> float:
    return work_stats.safe_ratio(float(numerator), float(denominator))


def mean(values) -> float:
    return statistics.fmean(float(value) for value in values)


def aggregate_workload(workload: str, config: dict) -> list[dict]:
    work_frames, work_sources = load_aligned_work(workload, config)
    sequences = list(dict.fromkeys(row["key"][0] for row in work_frames))
    latency, latency_sources = load_latency_frames(config, sequences)
    expected_keys = {row["key"] for row in work_frames}
    for system in SYSTEMS:
        actual_keys = set(latency[system])
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:3]
            extra = sorted(actual_keys - expected_keys)[:3]
            raise ValueError(
                f"latency-frame mismatch for {workload}/{system}: "
                f"missing={missing}, extra={extra}"
            )

    rows = []
    for system in SYSTEMS:
        issued_field = ISSUED_FIELD[system]
        raw_relative = []
        effective_relative = []
        wasted_relative = []
        raw_gflops = []
        effective_gflops = []
        eta = []
        latencies_ms = []
        active_ratios = []
        coverage = []
        packing = []
        undercoverage = 0
        max_identity_error = 0.0

        for frame in work_frames:
            key = frame["key"]
            dense_macs = frame["D"]
            required_macs = frame["E"]
            issued_macs = frame[issued_field]
            dense_ms = latency["dense"][key]
            backend_ms = latency[system][key]
            raw = safe_ratio(issued_macs, dense_macs) * dense_ms / backend_ms
            effective = (
                safe_ratio(required_macs, dense_macs) * dense_ms / backend_ms
            )
            wasted = raw - effective
            frame_eta = safe_ratio(required_macs, issued_macs)
            max_identity_error = max(
                max_identity_error, abs(effective - frame_eta * raw)
            )
            if issued_macs < required_macs:
                undercoverage += 1

            raw_relative.append(raw)
            effective_relative.append(effective)
            wasted_relative.append(wasted)
            # 2 * MAC / ms / 1e6 = GFLOP/s.
            raw_gflops.append(2.0 * issued_macs / backend_ms / 1e6)
            effective_gflops.append(2.0 * required_macs / backend_ms / 1e6)
            eta.append(frame_eta)
            latencies_ms.append(backend_ms)
            active_ratios.append(safe_ratio(required_macs, dense_macs))
            if system == "wiseconv":
                coverage.append(safe_ratio(required_macs, frame["W"]))
                packing.append(safe_ratio(frame["W"], frame["C"]))

        raw_mean = mean(raw_relative)
        effective_mean = mean(effective_relative)
        wasted_mean = mean(wasted_relative)
        if not math.isclose(
            raw_mean, effective_mean + wasted_mean, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"stack does not close for {workload}/{system}")
        if system == "dense" and not math.isclose(
            raw_mean, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"dense normalization does not close for {workload}")
        if max_identity_error > 1e-10:
            raise ValueError(
                f"per-frame effective-throughput identity failed for "
                f"{workload}/{system}: {max_identity_error}"
            )

        totals = {
            field: sum(frame[field] for frame in work_frames)
            for field in ("D", "E", "A", "W", "C")
        }
        rows.append(
            {
                "workload": workload,
                "workload_label": WORKLOAD_LABELS[workload],
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "frames": len(work_frames),
                "latency_ms_frame_mean": f"{mean(latencies_ms):.12g}",
                "active_work_ratio_frame_mean": f"{mean(active_ratios):.12g}",
                "useful_compute_ratio_frame_mean": f"{mean(eta):.12g}",
                "eta_ratio_of_totals": f"{safe_ratio(totals['E'], totals[issued_field]):.12g}",
                "eta_coverage_frame_mean": (
                    f"{mean(coverage):.12g}" if coverage else ""
                ),
                "eta_packing_frame_mean": (
                    f"{mean(packing):.12g}" if packing else ""
                ),
                "raw_throughput_gflops_frame_mean": f"{mean(raw_gflops):.12g}",
                "effective_throughput_gflops_frame_mean": (
                    f"{mean(effective_gflops):.12g}"
                ),
                "raw_throughput_vs_dense": f"{raw_mean:.12g}",
                "effective_throughput_vs_dense": f"{effective_mean:.12g}",
                "wasted_throughput_vs_dense": f"{wasted_mean:.12g}",
                "undercoverage_frames": undercoverage,
                "latency_reduction": "median_across_three_rounds_per_frame",
                "frame_aggregation": "arithmetic_mean_over_frames",
                "work_sources": ";".join(work_sources),
                "latency_sources": ";".join(latency_sources),
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no effective-throughput rows")
    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)


def main() -> None:
    rows = []
    for workload, config in work_stats.CONFIG.items():
        rows.extend(aggregate_workload(workload, config))
    expected = len(WORKLOAD_LABELS) * len(SYSTEMS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    write_csv(rows)

    print(f"wrote {OUT_CSV}")
    print("workload      system          eta      effective    wasted      raw")
    for row in rows:
        print(
            f"{row['workload']:<14}{row['system']:<16}"
            f"{float(row['useful_compute_ratio_frame_mean']):>7.4f}"
            f"{float(row['effective_throughput_vs_dense']):>13.4f}"
            f"{float(row['wasted_throughput_vs_dense']):>10.4f}"
            f"{float(row['raw_throughput_vs_dense']):>10.4f}"
        )


if __name__ == "__main__":
    main()
