#!/usr/bin/env python3
"""Build the paper-facing YOLOv8n ablation data.

The RTX~3080 rows use the final device-resident Stream-K runs.  Those runs
record full-trace latency and work efficiency, but do not include a separate
semantic NSYS stage attribution for every variant.  Consequently, Cons./Conv.
are emitted only for Ours, for which the current cost-decomposition capture is
available; the other stage cells remain explicit dashes in the generated table.
AGX Orin rows are kept as placeholders until the multi-platform refresh.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_CSV = HERE / "ablation.csv"

DEVICES = ("rtx3080", "agx_orin")
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
    "no_hybrid": "No Hybrid",
    "sync_exact_selector": "Sync exact selector",
    "ours": "Ours",
}
RUNS = {
    "atomic_append": "3080-streamk-v7-atomic-append-final-policy-r3",
    "global_order": "3080-streamk-v7-global-order-final-policy-r3",
    "no_reuse": "3080-streamk-v7-no-reuse-final-policy-r3",
    "dp_only": "3080-streamk-v7-dp-only-capability-r3",
    "no_hybrid": "3080-streamk-v7-no-hybrid-capability-r3",
    "sync_exact_selector": "3080-streamk-v7-sync-exact-selector-r3",
    "ours": "3080-streamk-v7-ours-final-policy-r3",
}
RUN_ROOT = REPO / "logs" / "yolov8n_ablation"
WORK_SUMMARY = (
    REPO / "logs" / "effective_throughput" / "3080"
    / "3080-streamk-v10-yolov8n-work-final-policy" / "summary.json"
)
LEGACY_ETA_SOURCE = REPO / "logs" / "yolov8n_ablation" / "ablation_table_v4_gridcap.json"
COST_CSV = HERE / "cost_decomposition.csv"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_totals() -> dict[str, float]:
    totals = {}
    for variant, run_name in RUNS.items():
        run_dir = RUN_ROOT / run_name
        run = load_json(run_dir / "run.json")
        summary = load_json(run_dir / "summary.json")
        if run.get("status") != "complete":
            raise ValueError(f"incomplete ablation run: {run_dir}")
        if run.get("variant", {}).get("key") != variant:
            raise ValueError(f"variant mismatch at {run_dir}")
        aggregate = (
            summary.get("latency", {})
            .get("aggregate", {})
            .get("per_frame_round_median_gpu_ms", {})
        )
        if int(aggregate.get("count", 0)) != 2572:
            raise ValueError(f"unexpected frame count at {run_dir}")
        totals[variant] = float(aggregate["mean"])
    return totals


def load_eta() -> dict[str, float]:
    """Load exact work η where available without mixing latency protocols."""

    old_rows = {
        row["variant"]: row
        for row in load_json(LEGACY_ETA_SOURCE).get("rows", ())
    }
    values = {
        variant: float(old_rows[variant]["eta_percent"])
        for variant in ("atomic_append", "global_order", "no_reuse")
    }
    work = load_json(WORK_SUMMARY)
    totals = {
        key: sum(int(value[key]) for value in work["sequences"].values())
        for key in ("dense_macs", "required_macs", "issued_macs")
    }
    values["ours"] = 100.0 * totals["required_macs"] / totals["issued_macs"]
    # DP/aligned/Hybrid mode selection does not change requested or packed
    # work, so they share Ours' useful-compute ratio.
    for variant in ("dp_only", "no_hybrid", "sync_exact_selector"):
        values[variant] = values["ours"]
    return values


def load_ours_stages() -> tuple[float, float]:
    with COST_CSV.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if row.get("device") == "rtx3080" and row.get("system") == "wiseconv":
            return float(row["construction_ms"]), float(row["convolution_ms"])
    raise ValueError("updated RTX 3080 WISEConv cost row is missing")


def build_rows() -> list[dict]:
    totals = load_totals()
    eta = load_eta()
    ours_construction, ours_convolution = load_ours_stages()
    rows = []
    for device in DEVICES:
        for variant in VARIANTS:
            is_rtx = device == "rtx3080"
            is_ours = variant == "ours"
            rows.append({
                "device": device,
                "device_label": DEVICE_LABELS[device],
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "eta_percent": f"{eta[variant]:.12g}" if is_rtx else "",
                "eta_coverage_percent": (
                    "100.0" if is_rtx and variant == "no_reuse"
                    else ("98.54" if is_rtx else "")
                ),
                "construction_ms": (
                    f"{ours_construction:.12g}" if is_rtx and is_ours else ""
                ),
                "convolution_ms": (
                    f"{ours_convolution:.12g}" if is_rtx and is_ours else ""
                ),
                "total_ms": f"{totals[variant]:.12g}" if is_rtx else "",
                "cuda_graph": (
                    ("false" if variant == "sync_exact_selector" else "true")
                    if is_rtx else ""
                ),
                "source": (
                    f"logs/yolov8n_ablation/{RUNS[variant]}/summary.json"
                    if is_rtx else ""
                ),
                "work_source": str(WORK_SUMMARY.relative_to(REPO)) if is_rtx else "",
            })
    return rows


def main() -> None:
    rows = build_rows()
    temporary = OUT_CSV.with_suffix(OUT_CSV.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUT_CSV)
    print(f"wrote {OUT_CSV}")
    for row in rows:
        if row["device"] == "rtx3080":
            print(
                f"{row['variant_label']:<24} "
                f"eta={float(row['eta_percent']):.2f}% "
                f"total={float(row['total_ms']):.3f} ms"
            )


if __name__ == "__main__":
    main()
