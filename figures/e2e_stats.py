#!/usr/bin/env python3
"""Derive end-to-end latency, energy, and accuracy from the formal logs.

The output contains one row per platform, workload, and execution path.  The
two unmeasured discrete-GPU slots are emitted with empty measurements so the
plot keeps its final 3x2 layout while those platforms are being selected.

Latency follows the aggregation used by benchmarks/.agg_intro.py: optical-flow
and detection latencies are averaged across evaluation sequences, and pose
latency is read from the full-run aggregate.  Energy uses the formal Jetson
protocol: each steady-state round's mean VDD_IN power is multiplied by that
round's cumulative CUDA-event GPU time, then summed over rounds and divided by
the total timed frames.  AGX Orin summaries already contain this quantity.  The
legacy Xavier NX markers are upgraded in memory from their original latency
JSONL files; the source logs are not modified.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs"
OUT_CSV = HERE / "e2e_stats.csv"
OUT_ACCURACY_CSV = HERE / "e2e_accuracy.csv"

PLATFORMS = [
    ("tbd-a", "TBD Discrete GPU A", "placeholder"),
    ("tbd-b", "TBD Discrete GPU B", "placeholder"),
    ("3080", "RTX 3080", "measured"),
    ("4070-laptop", "RTX 4070 Laptop", "measured"),
    ("agx-orin", "Jetson AGX Orin", "measured"),
    ("xavier-nx", "Jetson Xavier NX", "measured"),
]

WORKLOADS = [
    ("fireflownet", "FireFlowNet"),
    ("yolov8n", "YOLOv8n"),
    ("yolov8m", "YOLOv8m"),
    ("dynconv_pose", "DynConv Pose"),
]

BACKENDS = [
    ("dense", "Dense"),
    ("tile_skip", "Tile skipping"),
    ("gather_scatter", "Gather-scatter"),
    ("wiseconv", "WISEConv"),
]

OPTICAL_RUN = {
    "3080": "3080-v23-full-r3-1800mhz",
    "4070-laptop": "4070-laptop-full-r3",
    "agx-orin": "agx-orin-30w-locked-energy-v4-full-r3",
    "xavier-nx": "xavier-nx-20w-locked-full-r3",
}

YOLO_RUN = {
    "3080": {
        "yolov8n": "3080-v23-yolov8n-full-r3-1800mhz",
        "yolov8m": "3080-v23-yolov8m-full-r3-1800mhz",
    },
    "4070-laptop": {
        "yolov8n": "4070-laptop-yolov8n-full-r3",
        "yolov8m": "4070-laptop-yolov8m-full-r3",
    },
    "agx-orin": {
        "yolov8n": "agx-orin-30w-locked-energy-v4-yolov8n-full-r3",
        "yolov8m": "agx-orin-30w-locked-energy-v4-yolov8m-full-r3",
    },
    "xavier-nx": {
        "yolov8n": "xavier-nx-20w-locked-yolov8n-full-r3",
        "yolov8m": "xavier-nx-20w-locked-yolov8m-full-r3",
    },
}

POSE_RUN = {
    "3080": "3080-v23-full-s0125-r3-1800mhz",
    "4070-laptop": "4070-laptop-full-s0125-r3",
    "agx-orin": "agx-orin-30w-locked-energy-v4-full-s0125-r3",
    "xavier-nx": "xavier-nx-20w-locked-full-s0125-r3",
}

AGX_ENERGY = {
    "fireflownet": "fireflownet-full-r3-energy-v4.per_backend_energy.json",
    "yolov8n": "yolov8n-full-r3-energy-v4.per_backend_energy.json",
    "yolov8m": "yolov8m-full-r3-energy-v4.per_backend_energy.json",
    "dynconv_pose": "dynconv-pose-full-s0125-r3-energy-v4.per_backend_energy.json",
}

XAVIER_ENERGY = {
    "fireflownet": "fireflownet-full-r3-energy-v3",
    "yolov8n": "yolov8n-full-r3-energy-v3",
    "yolov8m": "yolov8m-full-r3-energy-v3",
    "dynconv_pose": "dynconv-pose-full-s0125-r3-energy-v3",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def actual_backend(workload: str, backend: str) -> str:
    if workload == "dynconv_pose" and backend == "gather_scatter":
        return "dynconv"
    return backend


def optical_latency(platform: str, backend: str) -> tuple[float, str]:
    run = OPTICAL_RUN[platform]
    path = LOGS / "optical_flow" / backend / run / "summary.json"
    summary = load_json(path)
    values = [
        sequence["latency"]["round_mean_gpu_ms"]["mean"]
        for sequence in summary["sequences"].values()
    ]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid optical-flow latency in {path}")
    return statistics.fmean(values), str(path.relative_to(REPO))


def yolo_latency(
    platform: str, workload: str, backend: str
) -> tuple[float, str]:
    path = LOGS / "yolov8_mot16" / YOLO_RUN[platform][workload] / "summary.json"
    summary = load_json(path)
    values = [
        sequence["latency"]["all_frames_gpu_ms"]["mean"]
        for name, sequence in summary["backends"][backend].items()
        if name != "aggregate_accuracy"
    ]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid YOLO latency in {path}")
    return statistics.fmean(values), str(path.relative_to(REPO))


def pose_latency(platform: str, backend: str) -> tuple[float, str]:
    path = LOGS / "dynconv_pose" / POSE_RUN[platform] / "summary.json"
    summary = load_json(path)
    value = summary["backends"][actual_backend("dynconv_pose", backend)]["latency"][
        "mean"
    ]
    if value <= 0:
        raise ValueError(f"invalid pose latency in {path}")
    return value, str(path.relative_to(REPO))


def latency(platform: str, workload: str, backend: str) -> tuple[float, str]:
    actual = actual_backend(workload, backend)
    if workload == "fireflownet":
        return optical_latency(platform, actual)
    if workload in ("yolov8n", "yolov8m"):
        return yolo_latency(platform, workload, actual)
    if workload == "dynconv_pose":
        return pose_latency(platform, backend)
    raise KeyError(workload)


def optical_accuracy(backend: str) -> tuple[float, str]:
    """Aggregate AEE over every valid ground-truth timestamp."""
    run = OPTICAL_RUN["3080"]
    path = LOGS / "optical_flow" / backend / run / "summary.json"
    summary = load_json(path)
    weighted_sum = 0.0
    sample_count = 0
    for sequence in summary["sequences"].values():
        accuracy = sequence.get("accuracy") or {}
        aee = accuracy.get("aee")
        if isinstance(aee, dict):
            aee = aee.get("mean")
        count = accuracy.get("valid_aee_frames")
        if not isinstance(aee, (int, float)) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"invalid optical-flow accuracy in {path}")
        weighted_sum += float(aee) * count
        sample_count += count
    if sample_count <= 0:
        raise ValueError(f"empty optical-flow accuracy in {path}")
    return weighted_sum / sample_count, str(path.relative_to(REPO))


def yolo_accuracy(workload: str, backend: str) -> tuple[float, str]:
    """Read COCO-style AP50:95 against the fixed teacher detections."""
    path = LOGS / "yolov8_mot16" / YOLO_RUN["3080"][workload] / "summary.json"
    summary = load_json(path)
    aggregate = summary["backends"][backend].get("aggregate_accuracy") or {}
    value = aggregate.get("pseudo_map50_95")
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"invalid detection accuracy for {backend} in {path}")
    return 100.0 * float(value), str(path.relative_to(REPO))


def pose_accuracy(backend: str) -> tuple[float, str]:
    """Read model-level PCKh from the formal pose run."""
    path = LOGS / "dynconv_pose" / POSE_RUN["3080"] / "summary.json"
    summary = load_json(path)
    actual = actual_backend("dynconv_pose", backend)
    accuracy = summary["backends"][actual].get("accuracy") or {}
    value = accuracy.get("pckh")
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 100.0:
        raise ValueError(f"invalid pose accuracy for {actual} in {path}")
    return float(value), str(path.relative_to(REPO))


def accuracy_rows() -> list[dict[str, str]]:
    """Return one platform-independent task-quality value per backend."""
    rows = []
    for workload, workload_label in WORKLOADS:
        for backend, backend_label in BACKENDS:
            if workload == "fireflownet":
                value, source = optical_accuracy(backend)
                metric, direction = "AEE", "lower"
            elif workload in ("yolov8n", "yolov8m"):
                value, source = yolo_accuracy(workload, backend)
                metric, direction = "AP50:95 (%)", "higher"
            elif workload == "dynconv_pose":
                value, source = pose_accuracy(backend)
                metric, direction = "PCKh (%)", "higher"
            else:
                raise KeyError(workload)
            rows.append(
                {
                    "workload": workload,
                    "workload_label": workload_label,
                    "backend": backend,
                    "backend_label": backend_label,
                    "metric": metric,
                    "direction": direction,
                    "value": f"{value:.9f}",
                    "source": source,
                }
            )
    if len(rows) != len(WORKLOADS) * len(BACKENDS):
        raise AssertionError(len(rows))
    return rows


def load_power_tools():
    path = REPO / "benchmarks" / "summarize_power_intervals.py"
    spec = importlib.util.spec_from_file_location("wiseconv_power_tools", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def xavier_latency_sources(workload: str) -> list[tuple[str | None, Path]]:
    if workload == "fireflownet":
        run = OPTICAL_RUN["xavier-nx"]
        return [
            (
                actual_backend(workload, backend),
                LOGS
                / "optical_flow"
                / actual_backend(workload, backend)
                / run
                / "sequences.jsonl",
            )
            for backend, _ in BACKENDS
        ]
    if workload in ("yolov8n", "yolov8m"):
        return [
            (
                None,
                LOGS
                / "yolov8_mot16"
                / YOLO_RUN["xavier-nx"][workload]
                / "sequences.jsonl",
            )
        ]
    if workload == "dynconv_pose":
        run = LOGS / "dynconv_pose" / POSE_RUN["xavier-nx"]
        return [
            (None, run / f"{actual_backend(workload, backend)}.jsonl")
            for backend, _ in BACKENDS
        ]
    raise KeyError(workload)


def energy_summaries() -> tuple[dict[tuple[str, str, str], float], dict]:
    values: dict[tuple[str, str, str], float] = {}
    sources: dict[tuple[str, str], str] = {}

    agx_root = LOGS / "nx_state" / "agx-orin-30w-locked-energy-v4"
    for workload, filename in AGX_ENERGY.items():
        path = agx_root / filename
        summary = load_json(path)
        for backend, _ in BACKENDS:
            actual = actual_backend(workload, backend)
            entry = summary["backends"][actual]
            value = entry.get("gpu_time_normalized_joules_per_timed_frame")
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"missing normalized energy for {actual} in {path}")
            values[("agx-orin", workload, backend)] = float(value)
        sources[("agx-orin", workload)] = str(path.relative_to(REPO))

    tools = load_power_tools()
    xavier_root = LOGS / "nx_state" / "xavier-nx-20w-locked"
    for workload, stem in XAVIER_ENERGY.items():
        legacy_path = xavier_root / f"{stem}.per_backend_energy.json"
        power_path = xavier_root / f"{stem}.power.jsonl"
        legacy = load_json(legacy_path)
        intervals = tools.attach_gpu_times(
            legacy["intervals"], xavier_latency_sources(workload)
        )
        summary = tools.summarize(
            tools.read_jsonl(power_path), intervals, "VDD_IN"
        )
        expected = {actual_backend(workload, backend) for backend, _ in BACKENDS}
        if set(summary["backends"]) != expected:
            raise ValueError(f"unexpected Xavier backends in {legacy_path}")
        expected_intervals = 3 if workload == "dynconv_pose" else 9
        for backend, _ in BACKENDS:
            actual = actual_backend(workload, backend)
            entry = summary["backends"][actual]
            if entry["interval_count"] != expected_intervals:
                raise ValueError(f"unexpected interval count for {actual} in {legacy_path}")
            value = entry["gpu_time_normalized_joules_per_timed_frame"]
            values[("xavier-nx", workload, backend)] = float(value)
        sources[("xavier-nx", workload)] = (
            f"{legacy_path.relative_to(REPO)} + CUDA-event latency JSONL"
        )

    return values, sources


def main() -> None:
    energies, energy_sources = energy_summaries()
    rows = []
    for platform, platform_label, status in PLATFORMS:
        for workload, workload_label in WORKLOADS:
            for backend, backend_label in BACKENDS:
                latency_ms = None
                latency_source = ""
                if status == "measured":
                    latency_ms, latency_source = latency(platform, workload, backend)
                energy_j = energies.get((platform, workload, backend))
                rows.append(
                    {
                        "platform": platform,
                        "platform_label": platform_label,
                        "platform_status": status,
                        "workload": workload,
                        "workload_label": workload_label,
                        "backend": backend,
                        "backend_label": backend_label,
                        "latency_ms": "" if latency_ms is None else f"{latency_ms:.9f}",
                        "energy_j_per_frame": "" if energy_j is None else f"{energy_j:.9f}",
                        "latency_source": latency_source,
                        "energy_source": energy_sources.get((platform, workload), ""),
                    }
                )

    expected_rows = len(PLATFORMS) * len(WORKLOADS) * len(BACKENDS)
    if len(rows) != expected_rows:
        raise AssertionError((len(rows), expected_rows))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    task_quality = accuracy_rows()
    with OUT_ACCURACY_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(task_quality[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(task_quality)

    measured = sum(bool(row["latency_ms"]) for row in rows)
    powered = sum(bool(row["energy_j_per_frame"]) for row in rows)
    print(f"wrote {OUT_CSV}")
    print(f"latency cells: {measured}; energy cells: {powered}")
    print(f"wrote {OUT_ACCURACY_CSV}; accuracy cells: {len(task_quality)}")


if __name__ == "__main__":
    main()
