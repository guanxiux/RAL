#!/usr/bin/env python3
"""Build the paper-facing YOLOv8n ablation data from the unified logs.

Sources (all under ``logs/``):

  * variant rows   ``ablation_and_sensitivity/<model>/<device>/<dtype>/<variant>/summary.json``
  * Ours row       the deployed E2E run
                   ``latency/<model>/<device>/<dtype>/<run>-wiseconv/summary.json``
                   with its ``eta`` taken from the matching ``work/`` run when
                   present, otherwise from a mode-only variant (see below)
  * Ours stages    ``latency_cost_decomposition/<model>/<device>/<dtype>/<run>/summary.json``

Stage attribution is reported only for Ours; the other rows intentionally
remain latency/useful-compute measurements without a variant-specific split.
"""

from __future__ import annotations

import csv
import json
import statistics
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
    "no_hybrid": "No Hybrid",
    "sync_exact_selector": "Sync exact selector",
    "ours": "Ours",
}
# Directory name of the staged variant run.
VARIANT_DIRS = {v: v for v in VARIANTS}
VARIANT_DIRS["sync_exact_selector"] = "sync_exact"
# The decomposition-mode variants never change the requested or packed work, so
# they share Ours' useful-compute ratio.  Used only when the reused Ours work
# pass is unavailable (e.g. the deployed run was measured latency-only).
ETA_FALLBACK = ("dp_only", "no_hybrid")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def ablation_dir(device: str) -> Path:
    return LOGS / "ablation_and_sensitivity" / MODEL / DEVICE_DIRS[device] / DTYPE


def find_e2e_run(device: str) -> str | None:
    base = LOGS / "latency" / MODEL / DEVICE_DIRS[device] / DTYPE
    if not base.is_dir():
        return None
    runs = sorted(p.name for p in base.iterdir()
                  if p.is_dir() and p.name.endswith("-wiseconv"))
    return runs[-1] if runs else None


def latency_total(summary: dict) -> float | None:
    medians = [
        value["all_frames_gpu_ms"]["median"]
        for value in summary.get("sequences", {}).values()
        if "all_frames_gpu_ms" in value
    ]
    return statistics.fmean(medians) if medians else None


def work_eta(summary: dict) -> float | None:
    position, mac = [], []
    for value in summary.get("sequences", {}).values():
        if value.get("useful_position_ratio") is not None:
            position.append(value["useful_position_ratio"])
        if value.get("useful_compute_ratio") is not None:
            mac.append(value["useful_compute_ratio"])
    if position:
        return statistics.fmean(position)
    if mac:
        return statistics.fmean(mac)
    return None


def read_variant_rows(device: str) -> dict[str, dict]:
    """Total and eta per variant, from the staged ablation summaries."""
    base = ablation_dir(device)
    rows: dict[str, dict] = {}
    for variant in VARIANTS:
        if variant == "ours":
            continue
        path = base / VARIANT_DIRS[variant] / "summary.json"
        if not path.is_file():
            continue
        summary = load_json(path)
        rows[variant] = {
            "total_ms": summary.get("total_ms"),
            "eta": summary.get("eta"),
        }
    return rows


def read_ours_row(device: str, variants: dict[str, dict]) -> dict:
    """Ours total from the deployed E2E run and eta from its work pass."""
    run = find_e2e_run(device)
    total = None
    eta = None
    source = ""
    if run is not None:
        lat = LOGS / "latency" / MODEL / DEVICE_DIRS[device] / DTYPE / run \
            / "summary.json"
        if lat.is_file():
            total = latency_total(load_json(lat))
            source = f"latency/{MODEL}/{DEVICE_DIRS[device]}/{DTYPE}/{run}/summary.json"
        work = LOGS / "work" / MODEL / DEVICE_DIRS[device] / DTYPE / run \
            / "summary.json"
        if work.is_file():
            eta = work_eta(load_json(work))
    if eta is None:
        for variant in ETA_FALLBACK:
            if variants.get(variant, {}).get("eta") is not None:
                eta = variants[variant]["eta"]
                break
    return {"total_ms": total, "eta": eta, "source": source}


def load_ours_stages(device: str, run: str | None) -> tuple[float, float] | None:
    if run is None:
        return None
    path = LOGS / "latency_cost_decomposition" / MODEL / DEVICE_DIRS[device] \
        / DTYPE / run / "summary.json"
    for candidate in (path, path.parent.parent / "full" / "summary.json"):
        if not candidate.is_file():
            continue
        categories = load_json(candidate).get("categories_ms", {})
        if "construction" in categories and "convolution" in categories:
            return float(categories["construction"]), \
                float(categories["convolution"])
    return None


def build_rows() -> list[dict]:
    rows = []
    for device in DEVICES:
        variants = read_variant_rows(device)
        ours = read_ours_row(device, variants)
        stages = load_ours_stages(device, find_e2e_run(device))
        for variant in VARIANTS:
            if variant == "ours":
                total, eta = ours["total_ms"], ours["eta"]
                construction = stages[0] if stages else None
                convolution = stages[1] if stages else None
                source = ours["source"]
            else:
                entry = variants.get(variant, {})
                total, eta = entry.get("total_ms"), entry.get("eta")
                construction = convolution = None
                source = f"ablation_and_sensitivity/{MODEL}/{DEVICE_DIRS[device]}" \
                    f"/{DTYPE}/{VARIANT_DIRS[variant]}/summary.json"
            rows.append({
                "device": device,
                "device_label": DEVICE_LABELS[device],
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "eta_percent": "" if eta is None else f"{100.0 * eta:.12g}",
                "construction_ms": "" if construction is None else f"{construction:.12g}",
                "convolution_ms": "" if convolution is None else f"{convolution:.12g}",
                "total_ms": "" if total is None else f"{total:.12g}",
                "source": source,
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


if __name__ == "__main__":
    main()
