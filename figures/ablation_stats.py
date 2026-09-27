#!/usr/bin/env python3
"""Build the paper-facing YOLOv8n ablation data from the unified logs.

The ablation is reported as a stage decomposition.  Construction and
convolution milliseconds come from the cost-decomposition harness run under each
variant's own environment::

    latency_cost_decomposition/<model>/<device>/<dtype>/ablation-<variant>/

and are the **raw profiled per-frame GPU kernel times** of each category
(``raw_categories_ms``), produced with ``--kernel-total`` in a single profiling
mode (eager, because ``Readback & launch`` host-synchronizes and cannot be
CUDA-graph captured) so the categories are comparable across variants instead
of being apportioned by ``share * E2E`` (which mixed eager profiling with a
graph E2E and let construction cost leak into the convolution column).
``total_ms`` is the variant's full-model latency from the staged
``ablation_and_sensitivity`` summary, and ``eta_percent`` is its useful-compute
ratio.  The variant environment is set by the caller when the decomposition
runs are produced (see ``benchmarks/ablation``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_CSV = HERE / "ablation.csv"
LOGS = REPO / "logs"
MODEL = "yolov8n"
DTYPE = "fp32"

DEVICES = ("rtx3080", "agx_orin")
DEVICE_DIRS = {"rtx3080": "3080", "agx_orin": "agx-orin"}
DEVICE_LABELS = {"rtx3080": "RTX 3080", "agx_orin": "AGX Orin"}
VARIANTS = (
    "atomic_append",
    "global_order",
    "no_reuse",
    "dp_only",
    "no_hybrid",
    "sync_exact_selector",
    "ours",
)
VARIANT_LABELS = {
    "atomic_append": "Atomic append",
    "global_order": "Global order",
    "no_reuse": "No reuse",
    "dp_only": "DP only",
    "no_hybrid": "No hybrid",
    "sync_exact_selector": "Readback & launch",
    "ours": "Ours",
}
# Directory suffix of the staged variant run and its decomposition run.
VARIANT_DIRS = {v: v for v in VARIANTS}
VARIANT_DIRS["sync_exact_selector"] = "sync_exact"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def build_rows() -> list[dict]:
    rows = []
    for device in DEVICES:
        for variant in VARIANTS:
            variant_dir = VARIANT_DIRS[variant]
            cost_path = (
                LOGS / "latency_cost_decomposition" / MODEL / DEVICE_DIRS[device]
                / DTYPE / f"ablation-{variant_dir}" / "summary.json"
            )
            if not cost_path.is_file():
                raise FileNotFoundError(cost_path)
            cost = load_json(cost_path)
            # Self-consistent kernel decomposition: construction/convolution are
            # raw profiled per-frame kernel times, not shares scaled to an E2E
            # measured in a different execution mode.  Fall back to the legacy
            # scaled categories for old summaries.
            categories = cost.get("raw_categories_ms")
            if categories is None:
                # Legacy scaled summary: recover raw kernel time from the
                # profiled total and the category shares.
                profiled = float(cost["profiled_kernel_total_ms"])
                frames = float(cost["frames"])
                categories = {
                    name: share * profiled / frames
                    for name, share in cost["categories_share"].items()
                }
            staged = (
                LOGS / "ablation_and_sensitivity" / MODEL / DEVICE_DIRS[device]
                / DTYPE / variant_dir / "summary.json"
            )
            staged_summary = load_json(staged) if staged.is_file() else {}
            eta = staged_summary.get("eta")
            total = staged_summary.get("total_ms")
            if total is None:
                total = cost["formal_latency_ms"]
            rows.append({
                "device": device,
                "device_label": DEVICE_LABELS[device],
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "eta_percent": "" if eta is None else f"{100.0 * eta:.12g}",
                "construction_ms": f"{float(categories['construction']):.12g}",
                "convolution_ms": f"{float(categories['convolution']):.12g}",
                "total_ms": f"{float(total):.12g}",
                "source": relative(cost_path),
            })
    return rows


def main() -> None:
    rows = build_rows()
    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
