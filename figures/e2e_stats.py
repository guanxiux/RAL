#!/usr/bin/env python3
"""Derive end-to-end latency, energy, accuracy, and reference data from logs.

The unified logs split the three passes into sibling trees:

  * latency  ``logs/latency/<workload>/<device>/<dtype>/<run-prefix>-<backend>/``
  * accuracy ``logs/accuracy/<workload>/<device>/<dtype>/<run-prefix>-<backend>/``
  * work     ``logs/work/<workload>/<device>/<dtype>/<run-prefix>-<backend>/``

Only platforms that actually have logs are emitted; missing cells stay blank so
the figure degrades gracefully instead of failing.

Latency is the mean of the per-sequence ``all_frames_gpu_ms.mean``.  Active
ratios come from the work pass (or the effective-throughput work-only replay
when a workload's work pass has no per-sequence counters).  Energy is the AGX
Orin run's own ``energy`` block (total joules over attributed frames).  The
proportional reference scales each sequence's dense latency by that sequence's
active ratio before matching the latency aggregation.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs"
OUT_CSV = HERE / "e2e_stats.csv"
OUT_ACCURACY_CSV = HERE / "e2e_accuracy.csv"

# Platforms with logs in the unified tree.  Device-directory names equal ids.
PLATFORMS = [
    ("a100", "NVIDIA A100", "measured"),
    ("4090", "RTX 4090", "measured"),
    ("3080", "RTX 3080", "measured"),
    ("4070-laptop", "RTX 4070 Laptop", "measured"),
    ("agx-orin", "AGX Orin", "measured"),
    ("xavier-nx", "Xavier NX", "measured"),
]

WORKLOADS = [
    ("fireflownet", "FireFlowNet"),
    ("yolov8n", "YOLOv8n"),
    ("yolov8m", "YOLOv8m"),
    ("dynconv_pose", "DynConv"),
]

BACKENDS = [
    ("dense", "Dense"),
    ("tile_skip", "Tile skipping"),
    ("gather_scatter", "Gather-scatter"),
    ("wiseconv", "WISEConv"),
]

DTYPE = "fp32"
RUN_PREFIX = {
    "fireflownet": "full-r3",
    "yolov8n": "yolov8n-full-r3",
    "yolov8m": "yolov8m-full-r3",
    "dynconv_pose": "full-s0125-r3",
}
ACTIVE_RATIO_PLATFORM = "3080"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def backend_dir(workload: str, backend: str) -> str:
    if workload == "dynconv_pose" and backend == "gather_scatter":
        return "dynconv"
    return backend


def run_path(tree: str, platform: str, workload: str, backend: str) -> Path:
    run = f"{RUN_PREFIX[workload]}-{backend_dir(workload, backend)}"
    return LOGS / tree / workload / platform / DTYPE / run / "summary.json"


def latency(platform: str, workload: str, backend: str):
    path = run_path("latency", platform, workload, backend)
    if not path.is_file():
        return None, ""
    sequences = load_json(path)["sequences"]
    values = [float(seq["all_frames_gpu_ms"]["mean"]) for seq in sequences.values()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid latency in {path}")
    return statistics.fmean(values), str(path.relative_to(REPO))


def energy(platform: str, workload: str, backend: str):
    path = run_path("latency", platform, workload, backend)
    if not path.is_file():
        return None
    block = load_json(path).get("energy")
    if not block:
        return None
    frames = block.get("attributed_frames") or block.get("frame_count")
    total = block.get("total_energy_j")
    if not frames or not total:
        return None
    return float(total) / float(frames)


def active_ratio_trace(workload: str):
    """Return per-sequence active ratios and the source path."""
    path = (
        LOGS / "work" / workload / ACTIVE_RATIO_PLATFORM / DTYPE
        / f"{RUN_PREFIX[workload]}-wiseconv" / "summary.json"
    )
    ratios: dict[str, float] = {}
    if path.is_file():
        sequences = load_json(path).get("sequences", {})
        for name, seq in sequences.items():
            if workload == "fireflownet" and "exact_flops_reduction" in seq:
                ratios[name] = 1.0 - float(seq["exact_flops_reduction"])
            elif workload == "dynconv_pose" and "conditional_mac_ratio" in seq:
                ratios[name] = float(seq["conditional_mac_ratio"])
            elif workload in ("yolov8n", "yolov8m") and "exact_flops_ratio" in seq:
                ratios[name] = float(seq["exact_flops_ratio"])
    if ratios:
        return ratios, str(path.relative_to(REPO))
    # Fallback: the effective-throughput replay carries the same masks.
    et = LOGS / "effective_throughput" / ACTIVE_RATIO_PLATFORM / "summary.json"
    if et.is_file():
        work = (load_json(et).get("workloads", {}).get(workload, {}) or {}).get(
            "work", {}
        )
        required = work.get("E_required_positions")
        dense = work.get("D_dense_positions")
        if required and dense:
            return {"_model": float(required) / float(dense)}, str(
                et.relative_to(REPO))
    raise ValueError(f"no active-ratio source for {workload}")


def _sequence_latencies(platform: str, workload: str, backend: str) -> dict:
    path = run_path("latency", platform, workload, backend)
    return load_json(path)["sequences"]


def proportional_reference(platform: str, workload: str):
    ratios, ratio_source = active_ratio_trace(workload)
    sequences = _sequence_latencies(platform, workload, "dense")
    names = set(sequences)
    if set(ratios) == {"_model"}:
        shared = next(iter(ratios.values()))
        values = [
            float(seq["all_frames_gpu_ms"]["mean"]) * shared
            for seq in sequences.values()
        ]
    else:
        missing = names - set(ratios)
        if missing:
            raise ValueError(f"ratio coverage mismatch for {workload}: {missing}")
        values = [
            float(seq["all_frames_gpu_ms"]["mean"]) * ratios[name]
            for name, seq in sequences.items()
        ]
    reference = statistics.fmean(values)
    if reference <= 0.0:
        raise ValueError(f"invalid proportional reference for {platform}/{workload}")
    return reference, f"{run_path('latency', platform, workload, 'dense')} + {ratio_source}"


def _optical_accuracy(backend: str):
    path = run_path("accuracy", "3080", "fireflownet", backend)
    if not path.is_file():
        return None, ""
    weighted = 0.0
    count = 0
    for seq in load_json(path)["sequences"].values():
        aee = seq.get("aee")
        if isinstance(aee, dict):
            aee = aee.get("mean")
        frames = seq.get("valid_aee_frames")
        if isinstance(aee, (int, float)) and isinstance(frames, int) and frames > 0:
            weighted += float(aee) * frames
            count += frames
    if count <= 0:
        return None, ""
    return weighted / count, str(path.relative_to(REPO))


def _yolo_accuracy(workload: str, backend: str):
    path = run_path("accuracy", "3080", workload, backend)
    if not path.is_file():
        return None, ""
    sequences = load_json(path)["sequences"]
    values = [
        float(seq["pseudo_map50_95"])
        for seq in sequences.values()
        if "pseudo_map50_95" in seq
    ]
    if not values:
        return None, ""
    return 100.0 * statistics.fmean(values), str(path.relative_to(REPO))


def _pose_accuracy(backend: str):
    path = run_path("accuracy", "3080", "dynconv_pose", backend)
    if not path.is_file():
        return None, ""
    sequences = load_json(path)["sequences"]
    values = [float(seq["pckh"]) for seq in sequences.values() if "pckh" in seq]
    if not values:
        return None, ""
    return statistics.fmean(values), str(path.relative_to(REPO))


def accuracy_rows() -> list[dict]:
    rows = []
    for workload, workload_label in WORKLOADS:
        for backend, backend_label in BACKENDS:
            if workload == "fireflownet":
                value, source = _optical_accuracy(backend)
                metric, direction = "AEE", "lower"
            elif workload in ("yolov8n", "yolov8m"):
                value, source = _yolo_accuracy(workload, backend)
                metric, direction = "AP50:95 (%)", "higher"
            elif workload == "dynconv_pose":
                value, source = _pose_accuracy(backend)
                metric, direction = "PCKh (%)", "higher"
            else:
                raise KeyError(workload)
            rows.append({
                "workload": workload,
                "workload_label": workload_label,
                "backend": backend,
                "backend_label": backend_label,
                "metric": metric,
                "direction": direction,
                "value": "" if value is None else f"{value:.9f}",
                "source": source,
            })
    return rows


def main() -> None:
    model_active_ratios = {}
    active_ratio_sources = {}
    for workload, _ in WORKLOADS:
        ratios, source = active_ratio_trace(workload)
        model_active_ratios[workload] = statistics.fmean(ratios.values())
        active_ratio_sources[workload] = source

    rows = []
    for platform, platform_label, status in PLATFORMS:
        for workload, workload_label in WORKLOADS:
            reference_ms = reference_source = ""
            if run_path("latency", platform, workload, "dense").is_file():
                reference_ms, reference_source = proportional_reference(
                    platform, workload
                )
            for backend, backend_label in BACKENDS:
                latency_ms, latency_source = latency(platform, workload, backend)
                energy_j = energy(platform, workload, backend)
                rows.append({
                    "platform": platform,
                    "platform_label": platform_label,
                    "platform_status": status,
                    "workload": workload,
                    "workload_label": workload_label,
                    "model_active_ratio": f"{model_active_ratios[workload]:.9f}",
                    "backend": backend,
                    "backend_label": backend_label,
                    "latency_ms": "" if latency_ms is None else f"{latency_ms:.9f}",
                    "ideal_latency_ms": (
                        "" if reference_ms == "" else f"{reference_ms:.9f}"
                    ),
                    "energy_j_per_frame": "" if energy_j is None else f"{energy_j:.9f}",
                    "latency_source": latency_source,
                    "ideal_source": reference_source,
                    "active_ratio_source": active_ratio_sources[workload],
                    "energy_source": latency_source if energy_j is not None else "",
                })

    expected_rows = len(PLATFORMS) * len(WORKLOADS) * len(BACKENDS)
    if len(rows) != expected_rows:
        raise AssertionError((len(rows), expected_rows))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    task_quality = accuracy_rows()
    with OUT_ACCURACY_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(task_quality[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_quality)

    measured = sum(bool(row["latency_ms"]) for row in rows)
    referenced = sum(bool(row["ideal_latency_ms"]) for row in rows)
    powered = sum(bool(row["energy_j_per_frame"]) for row in rows)
    print(f"wrote {OUT_CSV}")
    print(f"latency cells: {measured}; proportional-reference cells: "
          f"{referenced}; energy cells: {powered}")
    print(f"wrote {OUT_ACCURACY_CSV}; accuracy cells: "
          f"{sum(bool(r['value']) for r in task_quality)}/{len(task_quality)}")


if __name__ == "__main__":
    main()
