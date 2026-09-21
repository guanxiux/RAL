#!/usr/bin/env python3
"""Derive the Background sparsity-to-latency example from the unified logs.

Sources (RTX 3080, fireflownet, fp32):

  * ``logs/effective_throughput/3080/summary.json`` supplies the position-count
    work ratios ``E/D`` (required) and ``A/D`` (tile-skip issued), matching the
    paper's spatial alpha/eta definitions.
  * ``logs/latency/fireflownet/3080/fp32/full-r3-<backend>/summary.json``
    supplies full-model GPU latency.
  * ``logs/accuracy/fireflownet/3080/fp32/full-r3-<backend>/summary.json``
    supplies AEE.

Raw and effective throughput are normalized to dense raw throughput:

  T_path / T_dense     = (issued_path / D) * (t_dense / t_path)
  T_eff,path / T_dense = (E / D) * (t_dense / t_path)
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "logs"
DEVICE = "3080"
DTYPE = "fp32"
WORKLOAD = "fireflownet"
RUN_PREFIX = "full-r3"
DEVICE_LABEL = "RTX 3080"
OUT_CSV = HERE / "gap_stats.csv"

SYSTEMS = [
    ("dense", "Dense", "D"),
    ("tile_skip", "Tile skipping", "A"),
    ("gather_scatter", "Gather-scatter", "E"),
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def latency(backend: str):
    path = (LOGS / "latency" / WORKLOAD / DEVICE / DTYPE
            / f"{RUN_PREFIX}-{backend}" / "summary.json")
    sequences = load_json(path)["sequences"]
    values = [float(s["all_frames_gpu_ms"]["mean"]) for s in sequences.values()]
    return statistics.fmean(values), str(path.relative_to(REPO))


def aee(backend: str):
    path = (LOGS / "accuracy" / WORKLOAD / DEVICE / DTYPE
            / f"{RUN_PREFIX}-{backend}" / "summary.json")
    if not path.is_file():
        return None, ""
    weighted = count = 0
    for seq in load_json(path)["sequences"].values():
        value = seq.get("aee")
        if isinstance(value, dict):
            value = value.get("mean")
        frames = seq.get("valid_aee_frames")
        if isinstance(value, (int, float)) and isinstance(frames, int) and frames > 0:
            weighted += float(value) * frames
            count += frames
    return (weighted / count if count else None), str(path.relative_to(REPO))


def main() -> None:
    et_path = LOGS / "effective_throughput" / DEVICE / "summary.json"
    workload = load_json(et_path)["workloads"][WORKLOAD]
    work = workload["work"]
    dense = float(work["D_dense_positions"])
    required = float(work["E_required_positions"]) / dense
    scheduled = float(work["A_tile_skip_issued_positions"]) / dense
    frames = int(workload["frames"])

    measured = {b: latency(b) for b, _, _ in SYSTEMS}
    dense_latency = measured["dense"][0]
    reference_ms = dense_latency * required
    issued_ratio = {"D": 1.0, "A": scheduled, "E": required}

    rows = []
    for backend, name, quantity in SYSTEMS:
        latency_ms, source = measured[backend]
        aee_value, _ = aee(backend)
        issued = issued_ratio[quantity]
        raw_relative = issued * dense_latency / latency_ms
        effective_relative = required * dense_latency / latency_ms
        rows.append({
            "system": name,
            "backend": backend,
            "device": DEVICE_LABEL,
            "frames": frames,
            "latency_ms": f"{latency_ms:.9f}",
            "aee": "" if aee_value is None else f"{aee_value:.9f}",
            "issued_work_ratio": f"{issued:.9f}",
            "active_ratio": f"{required:.9f}",
            "eta": f"{effective_relative / raw_relative:.9f}",
            "raw_throughput_vs_dense": f"{raw_relative:.9f}",
            "effective_throughput_vs_dense": f"{effective_relative:.9f}",
            "speedup_vs_dense": f"{dense_latency / latency_ms:.9f}",
            "proportional_reference_ms": f"{reference_ms:.9f}",
            "gap_to_reference_x": f"{latency_ms / reference_ms:.9f}",
            "source": source,
        })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"# work anchor : {et_path.relative_to(REPO)} "
          f"({WORKLOAD}, {frames} frames)")
    print(f"# active E/D={required:.4f}; tile A/D={scheduled:.4f}")
    print(f"# proportional reference = {reference_ms:.3f} ms")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
