#!/usr/bin/env python3
"""Audit batched YOLOv8n decomposition logs and emit paper-facing data.

The formal CUDA-event run remains the source of truth for total latency.  The
NSYS captures supply only a per-tensor-batch split among construction,
convolution, elementwise, and other work.  This script independently checks the
formal arrays, capture provenance, graph mapping, per-batch closure, and the
12-row aggregate before writing the CSV consumed by
``make_batch_cost_decomposition_table.py``.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = (
    REPO
    / "logs"
    / "microbenchmarks"
    / "batch_cost_decomposition"
    / "3080-yolov8n-mot16-03-b148-v2-semantic"
)
SOURCE_CSV = RUN / "batch_cost_decomposition.csv"
SOURCE_BATCHES = RUN / "batch_cost_decomposition.batches.jsonl"
SOURCE_SUMMARY = RUN / "batch_cost_decomposition.summary.json"
SOURCE_AUDIT = RUN / "batch_cost_decomposition.kernel_audit.json"
SOURCE_MANIFEST = RUN / "manifest.json"
OUT_CSV = HERE / "batch_cost_decomposition.csv"

SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
BATCH_SIZES = (1, 4, 8)
PAPER_BATCH_SIZES = (1, 4, 8)
CATEGORIES = ("construction", "convolution", "elementwise", "other")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather-scatter",
    "wiseconv": "WISEConv",
}
EXPECTED_GRAPHS = {
    "dense": 2,
    "tile_skip": 2,
    "gather_scatter": 0,
    "wiseconv": 20,
}
EXPECTED_DEVICE = "NVIDIA GeForce RTX 3080"
EXPECTED_TIMED_FRAMES = 1488
EXPECTED_AGGREGATION = {
    "timed_frame_population": EXPECTED_TIMED_FRAMES,
    "frame_weighting": "equal weight over the fixed stream population",
    "formal_batch_latency": (
        "per-batch samples from the round whose additive ms/frame is the "
        "median of three formal CUDA-event rounds"
    ),
    "stage_scaling": (
        "formal_batch_ms * nsys_stage_timeline_ns / "
        "nsys_full_batch_timeline_ns"
    ),
    "reported_unit": "sum of scaled batch stage time / timed frames",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_recorded(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def close(
    left: float,
    right: float,
    *,
    tolerance: float = 1e-9,
    context: str,
) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{context}: {left} != {right}")


def finite_float(row: dict, field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {field} for {context}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} for {context}")
    return value


def normalized_batch_key(row: dict) -> dict:
    return {
        "sequence": str(row["sequence"]),
        "group_index": int(row["group_index"]),
        "step_index": int(row["step_index"]),
        "stream_ids": [str(value) for value in row["stream_ids"]],
        "frame_indices": [int(value) for value in row["frame_indices"]],
    }


def load_formal_config(config: dict, batch_size: int) -> tuple[dict, set[tuple[str, int]]]:
    path = resolve_recorded(config["formal_latency_npz"])
    if sha256_file(path) != config["formal_latency_sha256"]:
        raise ValueError(f"formal latency artifact changed: {path}")
    with np.load(path) as arrays:
        latency = arrays["gpu_latency_ms"].astype(np.float64)
        sequences = arrays["sequence"].astype(str)
        group_indices = arrays["group_index"].astype(np.int64)
        step_indices = arrays["step_index"].astype(np.int64)
        stream_ids = arrays["stream_ids"].astype(str)
        frame_indices = arrays["frame_indices"].astype(np.int64)

    expected_batches = EXPECTED_TIMED_FRAMES // batch_size
    if latency.shape != (3, expected_batches):
        raise ValueError(f"unexpected latency shape at {path}: {latency.shape}")
    if sequences.shape != (expected_batches,):
        raise ValueError(f"unexpected sequence shape at {path}")
    if group_indices.shape != (expected_batches,) or step_indices.shape != (
        expected_batches,
    ):
        raise ValueError(f"unexpected group/step shape at {path}")
    if stream_ids.shape != (expected_batches, batch_size):
        raise ValueError(f"unexpected stream-id shape at {path}")
    if frame_indices.shape != (expected_batches, batch_size):
        raise ValueError(f"unexpected frame-index shape at {path}")
    if set(sequences) != {"MOT16-03"}:
        raise ValueError(f"unexpected sequence at {path}")

    batch_keys = [
        {
            "sequence": str(sequences[index]),
            "group_index": int(group_indices[index]),
            "step_index": int(step_indices[index]),
            "stream_ids": [str(value) for value in stream_ids[index]],
            "frame_indices": [int(value) for value in frame_indices[index]],
        }
        for index in range(expected_batches)
    ]
    if sha256_json(batch_keys) != config["batch_keys_sha256"]:
        raise ValueError(f"formal batch-key digest changed at {path}")

    frame_population = {
        (str(stream_id), int(frame_index))
        for batch_streams, batch_frames in zip(
            stream_ids, frame_indices, strict=True
        )
        for stream_id, frame_index in zip(
            batch_streams, batch_frames, strict=True
        )
    }
    if len(frame_population) != EXPECTED_TIMED_FRAMES:
        raise ValueError(f"formal frame population does not close at {path}")

    round_totals = latency.sum(axis=1) / EXPECTED_TIMED_FRAMES
    selected_round = sorted(
        range(3), key=lambda index: (round_totals[index], index)
    )[1]
    if selected_round != int(config["selected_median_total_round"]):
        raise ValueError(f"selected formal round changed at {path}")
    formal_total = float(round_totals[selected_round])
    close(
        formal_total,
        float(config["formal_gpu_ms_per_frame"]),
        context=f"formal total at {path}",
    )
    return {
        "path": path,
        "latency": latency,
        "batch_keys": batch_keys,
        "selected_round": selected_round,
        "total_ms_per_frame": formal_total,
    }, frame_population


def validate_protocol(manifest: dict, summary: dict, audit: dict) -> str:
    if manifest.get("kind") != "yolov8n_stream_batch_cost_decomposition_manifest":
        raise ValueError("unexpected batch-decomposition manifest kind")
    if summary.get("kind") != "yolov8n_stream_batch_cost_decomposition":
        raise ValueError("unexpected batch-decomposition summary kind")
    if audit.get("kind") != "yolov8n_stream_batch_semantic_attribution_audit":
        raise ValueError("unexpected batch-decomposition audit kind")
    if tuple(manifest.get("systems") or ()) != SYSTEMS:
        raise ValueError("manifest system order changed")
    if tuple(manifest.get("batch_sizes") or ()) != BATCH_SIZES:
        raise ValueError("manifest batch-size order changed")
    category_keys = tuple(
        item.get("key") for item in manifest.get("categories") or ()
    )
    if category_keys != CATEGORIES:
        raise ValueError(f"unexpected stage categories: {category_keys}")
    if tuple(item.get("key") for item in summary.get("categories") or ()) != CATEGORIES:
        raise ValueError("summary stage categories changed")
    if manifest.get("aggregation") != EXPECTED_AGGREGATION:
        raise ValueError("manifest aggregation protocol changed")
    if summary.get("aggregation") != EXPECTED_AGGREGATION:
        raise ValueError("summary aggregation protocol changed")
    if (manifest.get("device") or {}).get("name") != EXPECTED_DEVICE:
        raise ValueError("manifest uses the wrong device")
    if (summary.get("device") or {}).get("name") != EXPECTED_DEVICE:
        raise ValueError("summary uses the wrong device")
    if int((manifest.get("device") or {}).get("expected_sm_clock_mhz", 0)) != 1800:
        raise ValueError("unexpected RTX 3080 SM clock")
    if manifest.get("git_status") != []:
        raise ValueError("decomposition manifest was not made from a clean tree")

    manifest_digest = sha256_file(SOURCE_MANIFEST)
    if summary.get("manifest_sha256") != manifest_digest:
        raise ValueError("summary refers to a different manifest")
    if summary.get("capture_manifest_sha256") != [manifest_digest]:
        raise ValueError("captures do not share exactly one manifest")
    if audit.get("manifest_sha256") != manifest_digest:
        raise ValueError("kernel audit refers to a different manifest")

    for path_field, digest_field in (
        ("formal_run_json", "formal_run_json_sha256"),
        ("formal_summary_json", "formal_summary_json_sha256"),
        ("policy_path", "policy_sha256"),
        ("mapping_path", "mapping_sha256"),
    ):
        path = resolve_recorded(manifest[path_field])
        if sha256_file(path) != manifest[digest_field]:
            raise ValueError(f"recorded input changed: {path}")

    formal_run = read_json(resolve_recorded(manifest["formal_run_json"]))
    formal_graphs = formal_run.get("cuda_graph_by_backend") or {}
    if (
        formal_run.get("status") != "complete"
        or formal_run.get("workload")
        != "yolov8n_mot16_stream_batch_sensitivity"
        or formal_run.get("model") != "yolov8n"
        or tuple(formal_run.get("sequences") or ()) != ("MOT16-03",)
        or tuple(formal_run.get("backends") or ()) != SYSTEMS
        or tuple(formal_run.get("batch_sizes") or ()) != (1, 2, 4, 8)
        or int(formal_run.get("base_streams_per_sequence", 0)) != 8
        or int(formal_run.get("timed_frames_per_round", 0))
        != EXPECTED_TIMED_FRAMES
        or int(formal_run.get("repetitions", 0)) != 3
        or formal_run.get("tf32") is not False
        or formal_run.get("autotune") is not True
        or formal_run.get("fresh_model_per_batch_size") is not True
        or formal_run.get("git_status") != []
        or formal_run.get("git_commit") != manifest["formal_git_commit"]
        or (formal_run.get("device") or {}).get("name") != EXPECTED_DEVICE
    ):
        raise ValueError("formal batch-sensitivity protocol changed")
    for system in SYSTEMS:
        if bool(formal_graphs.get(system)) != (EXPECTED_GRAPHS[system] > 0):
            raise ValueError(f"formal CUDA Graph protocol changed for {system}")
    return manifest_digest


def validate_capture(
    system: str,
    batch_size: int,
    config: dict,
    formal: dict,
    manifest: dict,
    manifest_digest: str,
    audit_row: dict,
) -> None:
    stem = RUN / "captures" / f"yolov8n_{system}_batch{batch_size}"
    metadata_path = stem.with_suffix(".capture.json")
    metadata = read_json(metadata_path)
    if (
        metadata.get("status") != "complete"
        or metadata.get("kind") != "yolov8n_stream_batch_nsys_capture"
        or metadata.get("system") != system
        or int(metadata.get("batch_size", 0)) != batch_size
        or bool(metadata.get("cuda_graph")) != bool(config["cuda_graph"])
        or metadata.get("manifest_sha256") != manifest_digest
        or metadata.get("git_commit") != manifest["git_commit"]
        or metadata.get("git_status") != []
    ):
        raise ValueError(f"capture metadata mismatch for {system}/B{batch_size}")
    gpu = metadata.get("gpu") or {}
    if (
        gpu.get("name") != EXPECTED_DEVICE
        or gpu.get("pstate") != "P0"
        or int(gpu.get("sm_clock_mhz", 0)) != 1800
    ):
        raise ValueError(f"capture clock/device mismatch for {system}/B{batch_size}")

    report_path = resolve_recorded(metadata["report"])
    profile_path = resolve_recorded(metadata["profile_json"])
    if sha256_file(report_path) != metadata["report_sha256"]:
        raise ValueError(f"NSYS report changed for {system}/B{batch_size}")
    if sha256_file(profile_path) != metadata["profile_json_sha256"]:
        raise ValueError(f"profile JSON changed for {system}/B{batch_size}")
    profile = read_json(profile_path)
    expected_batches = EXPECTED_TIMED_FRAMES // batch_size
    if (
        profile.get("status") != "complete"
        or profile.get("kind") != "yolov8n_stream_batch_nsys_profile"
        or profile.get("backend") != system
        or int(profile.get("batch_size", 0)) != batch_size
        or int(profile.get("timed_batches", 0)) != expected_batches
        or int(profile.get("timed_frames", 0)) != EXPECTED_TIMED_FRAMES
        or profile.get("sequence") != "MOT16-03"
        or profile.get("tf32") is not False
        or profile.get("autotune") is not True
        or profile.get("policy_artifact_status") != "complete"
        or bool(profile.get("cuda_graph")) != bool(config["cuda_graph"])
        or int(profile.get("cuda_graph_count", -1)) != EXPECTED_GRAPHS[system]
    ):
        raise ValueError(f"profile protocol mismatch for {system}/B{batch_size}")
    if resolve_recorded(profile["formal_latency"]).resolve() != formal["path"].resolve():
        raise ValueError(f"profile uses the wrong formal latency for {system}/B{batch_size}")
    profile_keys = []
    for batch_id, row in enumerate(profile.get("batches") or ()):
        if int(row.get("batch_id", -1)) != batch_id:
            raise ValueError(f"non-contiguous profile batches for {system}/B{batch_size}")
        profile_keys.append(normalized_batch_key(row))
    if profile_keys != formal["batch_keys"]:
        raise ValueError(f"profile/formal batch mismatch for {system}/B{batch_size}")

    expected_graphs = EXPECTED_GRAPHS[system]
    graph = audit_row.get("graph_mapping") or {}
    if (
        int(audit_row.get("captured_frame_ranges", 0)) != expected_batches
        or int(audit_row.get("selected_frame_ranges", 0)) != expected_batches
        or int(audit_row.get("semantic_nvtx_range_count", 0)) <= 0
        or int(graph.get("graph_capture_count", -1)) != expected_graphs
        or int(graph.get("mapped_graph_count", -1)) != expected_graphs
    ):
        raise ValueError(f"NSYS attribution mismatch for {system}/B{batch_size}")
    mapped_nodes = int(graph.get("mapped_graph_node_count", -1))
    semantic_nodes = sum(
        int(value) for value in (graph.get("semantic_graph_node_counts") or {}).values()
    )
    if mapped_nodes != semantic_nodes or (mapped_nodes > 0) != (expected_graphs > 0):
        raise ValueError(f"graph-node mapping does not close for {system}/B{batch_size}")
    if int(audit_row.get("associated_gpu_events", 0)) <= 0:
        raise ValueError(f"no attributed GPU events for {system}/B{batch_size}")

    construction_ms = 0.0
    for kernel in audit_row.get("kernels") or ():
        category = kernel.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"unknown kernel category for {system}/B{batch_size}")
        if category == "construction":
            construction_ms += float(kernel["raw_duration_ms"])
    if (construction_ms > 0.0) != (system == "wiseconv"):
        raise ValueError(f"construction attribution mismatch for {system}/B{batch_size}")


def validate_aggregate_row(
    source_row: dict,
    summary_row: dict,
    config: dict,
    system: str,
    batch_size: int,
) -> None:
    context = f"{system}/B{batch_size}"
    if (
        int(source_row["timed_batches"]) != EXPECTED_TIMED_FRAMES // batch_size
        or int(source_row["timed_frames"]) != EXPECTED_TIMED_FRAMES
        or int(source_row["selected_median_total_round"])
        != int(config["selected_median_total_round"])
    ):
        raise ValueError(f"aggregate population mismatch for {context}")
    total = finite_float(source_row, "total_latency_ms_per_frame", context)
    close(total, float(config["formal_gpu_ms_per_frame"]), context=f"formal {context}")
    close(
        total,
        float(summary_row["total_latency_ms_per_frame"]),
        tolerance=1e-12,
        context=f"CSV/summary total {context}",
    )
    stage_total = 0.0
    percent_total = 0.0
    for category in CATEGORIES:
        value = finite_float(source_row, f"{category}_ms_per_frame", context)
        percent = finite_float(source_row, f"{category}_percent", context)
        if value < 0.0:
            raise ValueError(f"negative {category} time for {context}")
        close(
            value,
            float(summary_row["stage_ms_per_frame"][category]),
            tolerance=1e-12,
            context=f"CSV/summary {category} {context}",
        )
        close(
            percent,
            100.0 * value / total,
            context=f"stage percentage {category} {context}",
        )
        stage_total += value
        percent_total += percent
    close(stage_total, total, context=f"stage closure {context}")
    close(percent_total, 100.0, context=f"percentage closure {context}")
    construction = finite_float(source_row, "construction_ms_per_frame", context)
    if (construction > 0.0) != (system == "wiseconv"):
        raise ValueError(f"construction applicability mismatch for {context}")


def validate_per_batch_rows(
    rows: list[dict],
    formal_by_key: dict[tuple[str, int], dict],
    source_by_key: dict[tuple[str, int], dict],
) -> None:
    grouped: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for row in rows:
        key = (str(row.get("system")), int(row.get("batch_size", 0)))
        grouped[key].append(row)
    expected = {(system, batch_size) for batch_size in BATCH_SIZES for system in SYSTEMS}
    if set(grouped) != expected:
        raise ValueError("per-batch JSONL has missing or unexpected configurations")

    for key in expected:
        system, batch_size = key
        batch_rows = sorted(grouped[key], key=lambda row: int(row["batch_id"]))
        expected_batches = EXPECTED_TIMED_FRAMES // batch_size
        if [int(row["batch_id"]) for row in batch_rows] != list(range(expected_batches)):
            raise ValueError(f"non-contiguous per-batch rows for {system}/B{batch_size}")
        formal = formal_by_key[key]
        stage_sums = {category: 0.0 for category in CATEGORIES}
        formal_sum = 0.0
        for batch_id, row in enumerate(batch_rows):
            context = f"{system}/B{batch_size}/batch{batch_id}"
            if normalized_batch_key(row) != formal["batch_keys"][batch_id]:
                raise ValueError(f"per-batch frame key mismatch for {context}")
            formal_batch_ms = finite_float(row, "formal_batch_ms", context)
            close(
                formal_batch_ms,
                float(formal["latency"][formal["selected_round"], batch_id]),
                context=f"formal batch latency {context}",
            )
            formal_per_frame = finite_float(row, "formal_ms_per_frame", context)
            close(
                formal_per_frame,
                formal_batch_ms / batch_size,
                context=f"per-frame conversion {context}",
            )
            stages = row.get("stage_ms_per_frame") or {}
            stage_total = 0.0
            for category in CATEGORIES:
                value = float(stages[category])
                if value < 0.0 or not math.isfinite(value):
                    raise ValueError(f"invalid {category} value for {context}")
                stage_sums[category] += value
                stage_total += value
            close(stage_total, formal_per_frame, context=f"batch stage closure {context}")
            busy = finite_float(row, "nsys_gpu_busy_fraction", context)
            if not 0.0 <= busy <= 1.0:
                raise ValueError(f"invalid GPU busy fraction for {context}")
            formal_sum += formal_batch_ms

        aggregate = source_by_key[key]
        close(
            formal_sum / EXPECTED_TIMED_FRAMES,
            float(aggregate["total_latency_ms_per_frame"]),
            context=f"per-batch total aggregation for {system}/B{batch_size}",
        )
        for category in CATEGORIES:
            # Every tensor batch in a configuration has the same B, and each
            # row is already divided by B.  The mean over tensor batches is
            # therefore also the equal-weight mean over the fixed frames.
            close(
                stage_sums[category] / expected_batches,
                float(aggregate[f"{category}_ms_per_frame"]),
                context=f"per-batch {category} aggregation for {system}/B{batch_size}",
            )


def paper_rows(source_by_key: dict[tuple[str, int], dict]) -> list[dict]:
    output = []
    for batch_size in PAPER_BATCH_SIZES:
        competitors = {
            system: float(source_by_key[(system, batch_size)]["total_latency_ms_per_frame"])
            for system in SYSTEMS
            if system != "wiseconv"
        }
        fastest = min(competitors, key=competitors.get)
        for system in SYSTEMS:
            source = source_by_key[(system, batch_size)]
            total = float(source["total_latency_ms_per_frame"])
            row = {
                "batch_size": batch_size,
                "system": system,
                "system_label": SYSTEM_LABELS[system],
                "timed_batches": int(source["timed_batches"]),
                "timed_frames": int(source["timed_frames"]),
                "total_latency_ms_per_frame": f"{total:.12g}",
                "actual_batch_latency_ms": f"{total * batch_size:.12g}",
                "construction_applicable": "true" if system == "wiseconv" else "false",
            }
            for category in CATEGORIES:
                row[f"{category}_ms_per_frame"] = (
                    f"{float(source[f'{category}_ms_per_frame']):.12g}"
                )
                row[f"{category}_percent"] = (
                    f"{float(source[f'{category}_percent']):.12g}"
                )
            row["fastest_competing_system"] = fastest if system == "wiseconv" else ""
            row["speedup_vs_fastest_competing"] = (
                f"{competitors[fastest] / total:.12g}" if system == "wiseconv" else ""
            )
            row["source"] = str(SOURCE_CSV.relative_to(REPO))
            output.append(row)
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
    manifest = read_json(SOURCE_MANIFEST)
    summary = read_json(SOURCE_SUMMARY)
    audit = read_json(SOURCE_AUDIT)
    manifest_digest = validate_protocol(manifest, summary, audit)

    with SOURCE_CSV.open(encoding="utf-8", newline="") as source:
        source_rows = list(csv.DictReader(source))
    source_by_key = {
        (str(row["system"]), int(row["batch_size"])): row for row in source_rows
    }
    expected = {(system, batch_size) for batch_size in BATCH_SIZES for system in SYSTEMS}
    if len(source_rows) != len(expected) or set(source_by_key) != expected:
        raise ValueError("aggregate CSV must contain exactly 12 expected rows")
    summary_by_key = {
        (str(row["system"]), int(row["batch_size"])): row
        for row in summary.get("rows") or ()
    }
    if set(summary_by_key) != expected:
        raise ValueError("summary must contain exactly 12 expected rows")
    audit_by_key = audit.get("captures") or {}
    if set(audit_by_key) != {
        f"{system}/B{batch_size}" for batch_size in BATCH_SIZES for system in SYSTEMS
    }:
        raise ValueError("kernel audit must contain exactly 12 expected captures")

    formal_by_key = {}
    canonical_population = None
    for batch_size in BATCH_SIZES:
        for system in SYSTEMS:
            key = (system, batch_size)
            config = manifest["configs"][system][str(batch_size)]
            if (
                int(config["timed_batches"]) != EXPECTED_TIMED_FRAMES // batch_size
                or int(config["timed_frames"]) != EXPECTED_TIMED_FRAMES
                or bool(config["cuda_graph"]) != (EXPECTED_GRAPHS[system] > 0)
                or int(config["cuda_graph_count"]) != EXPECTED_GRAPHS[system]
            ):
                raise ValueError(f"manifest config mismatch for {system}/B{batch_size}")
            formal, population = load_formal_config(config, batch_size)
            formal_by_key[key] = formal
            if canonical_population is None:
                canonical_population = population
            elif population != canonical_population:
                raise ValueError(f"timed-frame population changed for {system}/B{batch_size}")
            validate_capture(
                system,
                batch_size,
                config,
                formal,
                manifest,
                manifest_digest,
                audit_by_key[f"{system}/B{batch_size}"],
            )
            validate_aggregate_row(
                source_by_key[key],
                summary_by_key[key],
                config,
                system,
                batch_size,
            )

    batch_rows = read_jsonl(SOURCE_BATCHES)
    expected_batch_rows = sum(
        EXPECTED_TIMED_FRAMES // batch_size
        for batch_size in BATCH_SIZES
        for _ in SYSTEMS
    )
    if len(batch_rows) != expected_batch_rows:
        raise ValueError(
            f"per-batch JSONL has {len(batch_rows)} rows; expected {expected_batch_rows}"
        )
    validate_per_batch_rows(batch_rows, formal_by_key, source_by_key)

    output = paper_rows(source_by_key)
    write_csv(output)
    print(f"# source: {RUN.relative_to(REPO)}")
    print(f"# provenance commit: {manifest['git_commit']}")
    print(f"# identical timed-frame population: {len(canonical_population or ())}")
    print("B  system             total   cons.   conv.   elem.  other")
    for row in output:
        construction = (
            f"{float(row['construction_ms_per_frame']):.3f}"
            if row["construction_applicable"] == "true"
            else "  N/A"
        )
        print(
            f"{int(row['batch_size']):<3}"
            f"{row['system_label']:<19}"
            f"{float(row['total_latency_ms_per_frame']):>6.3f}"
            f"{construction:>8}"
            f"{float(row['convolution_ms_per_frame']):>8.3f}"
            f"{float(row['elementwise_ms_per_frame']):>8.3f}"
            f"{float(row['other_ms_per_frame']):>7.3f}"
        )
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
