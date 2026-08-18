#!/usr/bin/env python3
"""Derive the Background sparsity-to-latency example from measured logs.

The figure combines two sources without hand-entered results:

1. ``logs/tile_skip_amplification/summary.json`` supplies the mask-geometry
   ratios for FireFlowNet.  ``E/D`` is the required convolution work, while
   ``A/D`` is the work scheduled by DeltaCNN tile skipping.
2. The formal RTX 3080 FireFlowNet summaries supply full-model GPU latency and
   AEE for Dense, Tile skipping, and Gather-scatter on the same held-out
   sequences and mask policy.

Raw and effective throughput are normalized to dense raw throughput.  This
keeps the comparison tied to measured latency and work ratios:

  T_path / T_dense       = (issued_path / D) * (t_dense / t_path)
  T_eff,path / T_dense   = (E / D) * (t_dense / t_path)

The proportional-latency reference is ``t_dense * E/D``.  It is a reference
under unchanged per-work throughput, not a hard lower bound.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs"
AMP_SUMMARY = LOGS / "tile_skip_amplification" / "summary.json"
WORKLOAD = "fireflownet"
RUN = "3080-v23-full-r3-1800mhz"
DEVICE_LABEL = "RTX 3080"
OUT_CSV = HERE / "gap_stats.csv"

# Log backend, paper label, and the issued-work ratio used by that path.
SYSTEMS = [
    ("dense", "Dense", "D"),
    ("tile_skip", "Tile skipping", "A"),
    ("gather_scatter", "Gather-scatter", "E"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def load_work_ratios() -> dict[str, float | int]:
    """Load hardware-independent D, A, and E ratios for FireFlowNet."""
    workload = load_json(AMP_SUMMARY)["workloads"][WORKLOAD]
    scheduled = float(workload["A_over_D"])
    required = float(workload["E_over_D"])
    eta = float(workload["eta"])
    amplification = float(workload["alpha"])
    if not 0.0 < required <= scheduled <= 1.0:
        raise ValueError((required, scheduled))
    if not math.isclose(required / scheduled, eta, rel_tol=2e-3):
        raise ValueError("E/A does not reproduce the stored eta")
    if not math.isclose(scheduled / required, amplification, rel_tol=2e-3):
        raise ValueError("A/E does not reproduce the stored amplification")
    return {
        "frames": int(workload["frames"]),
        "D": 1.0,
        "A": scheduled,
        "E": required,
        "eta_tile": eta,
        "amplification": amplification,
    }


def read_aee(sequence: dict, path: Path) -> tuple[float, int]:
    accuracy = sequence.get("accuracy") or {}
    value = accuracy.get("aee")
    if isinstance(value, dict):
        value = value.get("mean")
    count = accuracy.get("valid_aee_frames")
    if not isinstance(value, (int, float)) or not isinstance(count, int):
        raise ValueError(f"invalid AEE in {path}")
    if value < 0.0 or count <= 0:
        raise ValueError(f"invalid AEE in {path}")
    return float(value), count


def load_backend(backend: str) -> dict[str, float | int | bool | str]:
    """Aggregate one formal backend run using the Evaluation methodology."""
    run_dir = LOGS / "optical_flow" / backend / RUN
    summary_path = run_dir / "summary.json"
    run_path = run_dir / "run.json"
    summary = load_json(summary_path)
    metadata = load_json(run_path)
    if metadata.get("status") != "complete" or metadata.get("backend") != backend:
        raise ValueError(f"incomplete or mismatched run at {run_path}")

    sequence_names = tuple(metadata.get("evaluated_sequences") or ())
    if set(sequence_names) != set(summary.get("sequences") or {}):
        raise ValueError(f"sequence mismatch at {summary_path}")

    sequence_latencies = []
    timed_frames = 0
    weighted_aee = 0.0
    aee_frames = 0
    for name in sequence_names:
        sequence = summary["sequences"][name]
        latency = sequence["latency"]
        sequence_latencies.append(float(latency["round_mean_gpu_ms"]["mean"]))
        frames_per_round = latency["frames_per_round"]
        if not frames_per_round or len(set(frames_per_round)) != 1:
            raise ValueError(f"inconsistent rounds for {name} in {summary_path}")
        timed_frames += int(frames_per_round[0])
        aee, count = read_aee(sequence, summary_path)
        weighted_aee += aee * count
        aee_frames += count

    if not sequence_latencies or min(sequence_latencies) <= 0.0:
        raise ValueError(f"invalid latency at {summary_path}")
    return {
        "latency_ms": statistics.fmean(sequence_latencies),
        "aee": weighted_aee / aee_frames,
        "timed_frames": timed_frames,
        "aee_frames": aee_frames,
        "tf32": bool(metadata.get("tf32")),
        "source": str(summary_path.relative_to(REPO)),
    }


def main() -> None:
    work = load_work_ratios()
    measured = {backend: load_backend(backend) for backend, _, _ in SYSTEMS}
    frame_counts = {int(value["timed_frames"]) for value in measured.values()}
    aee_counts = {int(value["aee_frames"]) for value in measured.values()}
    tf32_values = {bool(value["tf32"]) for value in measured.values()}
    if frame_counts != {int(work["frames"])}:
        raise ValueError((frame_counts, work["frames"]))
    if len(aee_counts) != 1 or len(tf32_values) != 1:
        raise ValueError((aee_counts, tf32_values))

    dense_latency = float(measured["dense"]["latency_ms"])
    active_ratio = float(work["E"])
    reference_ms = dense_latency * active_ratio
    issued_ratio = {
        "D": float(work["D"]),
        "A": float(work["A"]),
        "E": active_ratio,
    }

    rows = []
    for backend, name, quantity in SYSTEMS:
        latency = float(measured[backend]["latency_ms"])
        issued = issued_ratio[quantity]
        raw_relative = issued * dense_latency / latency
        effective_relative = active_ratio * dense_latency / latency
        rows.append(
            {
                "system": name,
                "backend": backend,
                "device": DEVICE_LABEL,
                "tf32": next(iter(tf32_values)),
                "frames": int(work["frames"]),
                "latency_ms": f"{latency:.9f}",
                "aee": f"{float(measured[backend]['aee']):.9f}",
                "issued_work_ratio": f"{issued:.9f}",
                "active_ratio": f"{active_ratio:.9f}",
                "eta": f"{effective_relative / raw_relative:.9f}",
                "raw_throughput_vs_dense": f"{raw_relative:.9f}",
                "effective_throughput_vs_dense": f"{effective_relative:.9f}",
                "speedup_vs_dense": f"{dense_latency / latency:.9f}",
                "proportional_reference_ms": f"{reference_ms:.9f}",
                "gap_to_reference_x": f"{latency / reference_ms:.9f}",
                "source": measured[backend]["source"],
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"# work anchor : {AMP_SUMMARY} ({WORKLOAD}, {work['frames']} frames)")
    print(f"# latency/AEE : optical_flow/{{backend}}/{RUN}/summary.json")
    print(
        f"# active E/D={active_ratio:.4f}; tile A/D={float(work['A']):.4f}; "
        f"tile eta={float(work['eta_tile']):.4f}"
    )
    print(f"# proportional reference = {reference_ms:.3f} ms")
    header = ("system", "lat(ms)", "AEE", "eta", "T/Td", "Teff/Td", "gap/ref")
    print("{:<16}{:>9}{:>9}{:>9}{:>9}{:>11}{:>10}".format(*header))
    for row in rows:
        print(
            "{:<16}{:>9.3f}{:>9.4f}{:>9.3f}{:>9.3f}{:>11.3f}{:>10.2f}".format(
                row["system"],
                float(row["latency_ms"]),
                float(row["aee"]),
                float(row["eta"]),
                float(row["raw_throughput_vs_dense"]),
                float(row["effective_throughput_vs_dense"]),
                float(row["gap_to_reference_x"]),
            )
        )
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
