#!/usr/bin/env python3
"""Derive YOLOv8n stream-batching sensitivity data from the formal logs.

The formal runner partitions MOT16-03 into eight fixed contiguous streams and
changes only how those streams are grouped into tensor batches.  This script
audits the run manifest, raw latency arrays, frame population, CUDA Graph
protocol, and stored aggregation before writing the CSV consumed by
``plot_batch_sensitivity.py``.  No measured value is entered by hand.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_NAME = "3080-yolov8n-mot16-03-b1248-r3"
RUN_DIR = REPO / "logs" / "yolov8_batch_sensitivity" / RUN_NAME
WISE_RUN_DIR = (
    REPO / "logs" / "yolov8_batch_sensitivity"
    / "3080-streamk-v10-yolov8n-batch-wiseconv-r3"
)
OUT_CSV = HERE / "batch_sensitivity.csv"

SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
BATCH_SIZES = (1, 2, 4, 8)
EXPECTED_GRAPHS = {
    "dense": 2,
    "tile_skip": 2,
    "gather_scatter": 0,
    "wiseconv": 20,
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def close(left: float, right: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"values do not close: {left} != {right}")


def load_protocol() -> tuple[dict, dict, int]:
    run_path = RUN_DIR / "run.json"
    summary_path = RUN_DIR / "summary.json"
    run = load_json(run_path)
    summary = load_json(summary_path)
    if run.get("status") != "complete":
        raise ValueError(f"incomplete run at {run_path}")
    if run.get("workload") != "yolov8n_mot16_stream_batch_sensitivity":
        raise ValueError(f"unexpected workload at {run_path}")
    if run.get("model") != "yolov8n":
        raise ValueError(f"unexpected model at {run_path}")
    device_name = str((run.get("device") or {}).get("name") or "")
    if device_name != "NVIDIA GeForce RTX 3080":
        raise ValueError(f"unexpected device at {run_path}: {device_name}")
    if tuple(run.get("sequences") or ()) != ("MOT16-03",):
        raise ValueError(f"unexpected sequence at {run_path}")
    if int(run.get("base_streams_per_sequence", 0)) != 8:
        raise ValueError(f"unexpected base-stream count at {run_path}")
    if tuple(run.get("batch_sizes") or ()) != BATCH_SIZES:
        raise ValueError(f"unexpected batch sizes at {run_path}")
    if tuple(run.get("backends") or ()) != SYSTEMS:
        raise ValueError(f"unexpected backends at {run_path}")
    if int(run.get("repetitions", 0)) != 3:
        raise ValueError(f"unexpected repetitions at {run_path}")
    if run.get("tf32") is not False or run.get("autotune") is not True:
        raise ValueError(f"unexpected execution settings at {run_path}")
    if run.get("fresh_model_per_batch_size") is not True:
        raise ValueError(f"batch shapes were not isolated at {run_path}")
    if run.get("git_status") != []:
        raise ValueError(f"formal run did not use a clean tree at {run_path}")
    graph_protocol = run.get("cuda_graph_by_backend") or {}
    for system in SYSTEMS:
        if bool(graph_protocol.get(system)) != (EXPECTED_GRAPHS[system] > 0):
            raise ValueError(f"unexpected graph protocol for {system}")

    timed_frames = int(run.get("timed_frames_per_round", 0))
    manifest = (run.get("stream_manifest") or {}).get("MOT16-03") or []
    if len(manifest) != 8:
        raise ValueError("stream manifest does not contain eight streams")
    manifest_keys = []
    for stream in manifest:
        indices = tuple(int(value) for value in stream.get("frame_indices") or ())
        if len(indices) < 2 or indices != tuple(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"non-contiguous stream {stream.get('stream_id')}")
        if int(stream.get("timed_transitions", -1)) != len(indices) - 1:
            raise ValueError(f"invalid transition count for {stream.get('stream_id')}")
        manifest_keys.extend(
            (str(stream["stream_id"]), frame_index)
            for frame_index in indices[1:]
        )
    if len(manifest_keys) != len(set(manifest_keys)) or len(manifest_keys) != timed_frames:
        raise ValueError("manifest timed-frame population is invalid")

    configs = summary.get("configs") or {}
    if set(configs) != set(SYSTEMS):
        raise ValueError(f"summary has unexpected systems at {summary_path}")
    if any(set(configs[system]) != {str(value) for value in BATCH_SIZES} for system in SYSTEMS):
        raise ValueError(f"summary has incomplete batch sizes at {summary_path}")
    return run, summary, timed_frames


def load_config(
    summary: dict,
    system: str,
    batch_size: int,
    timed_frames: int,
    run_dir: Path = RUN_DIR,
) -> tuple[dict, set[tuple[str, int]]]:
    stored = summary["configs"][system][str(batch_size)]
    expected_batches = timed_frames // batch_size
    if timed_frames % batch_size:
        raise ValueError("timed frame population is not divisible by batch size")
    if int(stored["rounds"]) != 3:
        raise ValueError(f"unexpected rounds for {system}/B{batch_size}")
    if int(stored["timed_batches_per_round"]) != expected_batches:
        raise ValueError(f"unexpected batch count for {system}/B{batch_size}")
    if int(stored["timed_frames_per_round"]) != timed_frames:
        raise ValueError(f"unexpected frame count for {system}/B{batch_size}")
    if int(stored["cuda_graph_count"]) != EXPECTED_GRAPHS[system]:
        raise ValueError(f"unexpected graph count for {system}/B{batch_size}")

    latency_path = run_dir / f"latency_{system}_batch{batch_size}.npz"
    with np.load(latency_path) as arrays:
        latency = arrays["gpu_latency_ms"].astype(np.float64)
        stream_ids = arrays["stream_ids"]
        frame_indices = arrays["frame_indices"]
        sequences = arrays["sequence"]
    if latency.shape != (3, expected_batches):
        raise ValueError(f"unexpected latency shape in {latency_path}")
    if stream_ids.shape != (expected_batches, batch_size):
        raise ValueError(f"unexpected stream-id shape in {latency_path}")
    if frame_indices.shape != (expected_batches, batch_size):
        raise ValueError(f"unexpected frame-index shape in {latency_path}")
    if sequences.shape != (expected_batches,) or set(sequences) != {"MOT16-03"}:
        raise ValueError(f"unexpected sequences in {latency_path}")

    frame_keys = {
        (str(stream_id), int(frame_index))
        for batch_streams, batch_frames in zip(
            stream_ids, frame_indices, strict=True
        )
        for stream_id, frame_index in zip(
            batch_streams, batch_frames, strict=True
        )
    }
    if len(frame_keys) != timed_frames:
        raise ValueError(f"duplicate or missing timed frames in {latency_path}")

    round_ms_per_frame = latency.sum(axis=1) / timed_frames
    stored_rounds = tuple(float(value) for value in stored["round_mean_gpu_ms_per_frame"])
    if len(stored_rounds) != 3:
        raise ValueError(f"unexpected stored rounds for {system}/B{batch_size}")
    for measured, expected in zip(round_ms_per_frame, stored_rounds, strict=True):
        close(float(measured), expected)
    robust_ms_per_frame = float(statistics.median(round_ms_per_frame))
    robust_throughput = float(
        statistics.median(1000.0 / round_ms_per_frame)
    )
    close(robust_ms_per_frame, float(stored["robust_mean_gpu_ms_per_frame"]))
    close(
        robust_throughput,
        float(stored["robust_throughput_frames_per_second"]),
    )

    tracker = [
        group
        for repetition in stored.get("activity_tracker_rounds") or []
        for group in repetition
        if group is not None
    ]
    max_fallbacks = max(
        (int(value["nearest_bucket_fallbacks"]) for value in tracker),
        default=0,
    )
    if max_fallbacks != 0:
        raise ValueError(f"activity-bucket fallback for {system}/B{batch_size}")

    return (
        {
            "system": system,
            "system_label": SYSTEM_LABELS[system],
            "batch_size": batch_size,
            "rounds": 3,
            "timed_batches_per_round": expected_batches,
            "timed_frames_per_round": timed_frames,
            "robust_batch_gpu_ms": robust_ms_per_frame * batch_size,
            "robust_amortized_gpu_ms_per_frame": robust_ms_per_frame,
            "robust_throughput_frames_per_second": robust_throughput,
            "cuda_graph": str(EXPECTED_GRAPHS[system] > 0).lower(),
            "cuda_graph_count": EXPECTED_GRAPHS[system],
            "max_activity_bucket_fallbacks": max_fallbacks,
            "source": str(latency_path.relative_to(REPO)),
        },
        frame_keys,
    )


def main() -> None:
    run, summary, timed_frames = load_protocol()
    wise_run = load_json(WISE_RUN_DIR / "run.json")
    wise_summary = load_json(WISE_RUN_DIR / "summary.json")
    if (
        wise_run.get("status") != "complete"
        or wise_run.get("workload") != "yolov8n_mot16_stream_batch_sensitivity"
        or wise_run.get("model") != "yolov8n"
        or tuple(wise_run.get("sequences") or ()) != ("MOT16-03",)
        or int(wise_run.get("base_streams_per_sequence", 0)) != 8
        or tuple(wise_run.get("batch_sizes") or ()) != BATCH_SIZES
        or tuple(wise_run.get("backends") or ()) != ("wiseconv",)
        or int(wise_run.get("repetitions", 0)) != 3
        or wise_run.get("tf32") is not False
        or wise_run.get("autotune") is not True
        or wise_run.get("fresh_model_per_batch_size") is not True
        or not bool((wise_run.get("cuda_graph_by_backend") or {}).get("wiseconv"))
    ):
        raise ValueError(f"unexpected isolated WISEConv batch protocol at {WISE_RUN_DIR}")
    if int(wise_run.get("timed_frames_per_round", 0)) != timed_frames:
        raise ValueError("WISEConv batch run has a different timed-frame count")
    if set(wise_summary.get("configs") or {}) != {"wiseconv"}:
        raise ValueError("isolated WISEConv batch summary has unexpected systems")
    rows = []
    canonical_frames = None
    for system in SYSTEMS:
        for batch_size in BATCH_SIZES:
            source_summary = wise_summary if system == "wiseconv" else summary
            source_run_dir = WISE_RUN_DIR if system == "wiseconv" else RUN_DIR
            source_run = wise_run if system == "wiseconv" else run
            row, frame_keys = load_config(
                source_summary, system, batch_size, timed_frames, source_run_dir
            )
            row.update(
                sequence="MOT16-03",
                device=str(source_run["device"]["name"]),
                tf32=str(source_run["tf32"]).lower(),
                run_git_commit=str(source_run["git_commit"]),
            )
            if canonical_frames is None:
                canonical_frames = frame_keys
            elif frame_keys != canonical_frames:
                raise ValueError(
                    f"timed-frame mismatch for {system}/B{batch_size}"
                )
            rows.append(row)

    by_key = {
        (str(row["system"]), int(row["batch_size"])): row for row in rows
    }
    for system in SYSTEMS:
        base = float(
            by_key[(system, 1)]["robust_amortized_gpu_ms_per_frame"]
        )
        for batch_size in BATCH_SIZES:
            row = by_key[(system, batch_size)]
            latency = float(row["robust_amortized_gpu_ms_per_frame"])
            row["per_frame_speedup_vs_batch1"] = base / latency

    for batch_size in BATCH_SIZES:
        competitors = {
            system: float(
                by_key[(system, batch_size)][
                    "robust_amortized_gpu_ms_per_frame"
                ]
            )
            for system in SYSTEMS
            if system != "wiseconv"
        }
        fastest = min(competitors, key=competitors.get)
        wise = by_key[("wiseconv", batch_size)]
        wise_latency = float(wise["robust_amortized_gpu_ms_per_frame"])
        wise["fastest_competing_system"] = fastest
        wise["speedup_vs_fastest_competing"] = (
            competitors[fastest] / wise_latency
        )
        wise["latency_reduction_vs_fastest_competing"] = (
            1.0 - wise_latency / competitors[fastest]
        )

    fieldnames = list(rows[0])
    for optional in (
        "fastest_competing_system",
        "speedup_vs_fastest_competing",
        "latency_reduction_vs_fastest_competing",
    ):
        if optional not in fieldnames:
            fieldnames.append(optional)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"# baseline source: {RUN_DIR.relative_to(REPO)}")
    print(f"# WISEConv source: {WISE_RUN_DIR.relative_to(REPO)}")
    print(f"# baseline commit: {run['git_commit']}")
    print(f"# identical timed-frame population: {len(canonical_frames or ())}")
    print("system             B   batch ms   ms/frame        fps   vs B1")
    for row in rows:
        print(
            f"{str(row['system_label']):18s} "
            f"{int(row['batch_size']):1d} "
            f"{float(row['robust_batch_gpu_ms']):10.4f} "
            f"{float(row['robust_amortized_gpu_ms_per_frame']):10.4f} "
            f"{float(row['robust_throughput_frames_per_second']):10.1f} "
            f"{float(row['per_frame_speedup_vs_batch1']):8.3f}x"
        )
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
