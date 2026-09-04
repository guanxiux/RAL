#!/usr/bin/env python3
"""Build the paired FP32/FP16 sensitivity paper CSV."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_CSV = HERE / "dtype_decomposition.csv"
SETTINGS = {
    "dynconv_rtx3080": {
        "workload_label": "DynConv",
        "device_label": "RTX 3080",
        "source": REPO / "logs" / "microbenchmarks"
        / "dynconv_dtype_decomposition" / "3080-fp16-cutlass-v1-semantic"
        / "dynconv_dtype_decomposition.summary.json",
        "scope": "overall",
    },
    "fireflownet_agx": {
        "workload_label": "FireFlowNet",
        "device_label": "AGX Orin",
        "source": REPO / "logs" / "microbenchmarks"
        / "fireflownet_dtype_decomposition" / "agx-orin-v5" / "summary.json",
        "scope": None,
    },
}
DTYPES = ("fp32", "fp16")
SYSTEMS = ("dense", "wiseconv")
CATEGORIES = ("construction", "convolution", "elementwise", "other")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_rows() -> list[dict]:
    output = []
    for setting, config in SETTINGS.items():
        source = config["source"]
        payload = load_json(source)
        source_rows = payload.get("rows") or ()
        if config["scope"] is not None:
            source_rows = [
                row for row in source_rows if row.get("scope") == config["scope"]
            ]
        source_rows = [
            row for row in source_rows
            if row.get("dtype") in DTYPES and row.get("system") in SYSTEMS
        ]
        by_key = {(row["dtype"], row["system"]): row for row in source_rows}
        expected = {(dtype, system) for dtype in DTYPES for system in SYSTEMS}
        if set(by_key) != expected or len(source_rows) != len(expected):
            raise ValueError(f"unexpected dtype rows: {source}")
        for dtype in DTYPES:
            for system in SYSTEMS:
                row = by_key[(dtype, system)]
                total = float(row.get("total_ms", row.get("total_latency_ms")))
                stages = {
                    category: float(row["stage_ms"][category])
                    for category in CATEGORIES
                }
                if not math.isclose(
                    sum(stages.values()), total, rel_tol=0.0, abs_tol=1e-9
                ):
                    raise ValueError(f"stages do not close: {setting}/{dtype}/{system}")
                output_row = {
                    "setting": setting,
                    "workload_label": config["workload_label"],
                    "device_label": config["device_label"],
                    "dtype": dtype,
                    "system": system,
                    "system_label": "WISEConv" if system == "wiseconv" else "Dense",
                    "total_ms": f"{total:.12g}",
                    "source": relative(source),
                }
                for category in CATEGORIES:
                    milliseconds = stages[category]
                    percent = float(row["stage_percent"][category])
                    if not math.isclose(
                        percent, 100.0 * milliseconds / total,
                        rel_tol=0.0, abs_tol=1e-8,
                    ):
                        raise ValueError(
                            f"percentage does not close: "
                            f"{setting}/{dtype}/{system}/{category}"
                        )
                    output_row[f"{category}_ms"] = f"{milliseconds:.12g}"
                    output_row[f"{category}_percent"] = f"{percent:.12g}"
                output.append(output_row)
    return output


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
    for row in rows:
        print(
            f"{row['setting']:<20} {row['dtype']}/{row['system']:<8} "
            f"total={float(row['total_ms']):.3f} ms"
        )


if __name__ == "__main__":
    main()
