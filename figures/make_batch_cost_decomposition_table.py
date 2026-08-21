#!/usr/bin/env python3
"""Generate the stream-batching decomposition LaTeX table from CSV."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "batch_cost_decomposition.csv"
OUTPUT = HERE / "batch_cost_decomposition_table.tex"

SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
BATCH_SIZES = (4, 8)
CATEGORIES = ("construction", "convolution", "elementwise", "other")
TABLE_SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather",
    "wiseconv": "WISEConv",
}


def stage_cell(row: dict, category: str) -> str:
    if category == "construction" and row["construction_applicable"] != "true":
        return r"\textemdash{}"
    milliseconds = float(row[f"{category}_ms_per_frame"])
    percent = float(row[f"{category}_percent"])
    return rf"{milliseconds:.2f}({percent:.1f}\%)"


def make_table(rows: list[dict]) -> str:
    expected = {
        (batch_size, system)
        for batch_size in BATCH_SIZES
        for system in SYSTEMS
    }
    by_key = {
        (int(row["batch_size"]), str(row["system"])): row for row in rows
    }
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("paper batch-decomposition CSV has unexpected rows")

    lines = [
        r"\begin{table}[!ht]",
        r"  \centering",
        r"  \caption{YOLOv8n latency decomposition under stream-level",
        r"  batching on RTX~3080. Stage definitions, units, and cell format follow",
        r"  Table~\ref{tab:cost-decomposition}.}",
        r"  \Description{A table compares Dense, Tile skipping, Gather-scatter,",
        r"  and WISEConv at batch sizes four and eight. Each row gives",
        r"  amortized total latency per frame and its division among worklist",
        r"  construction, convolution, elementwise, and other work.}",
        r"  \label{tab:batch-cost-decomposition}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{1.2pt}",
        r"  \renewcommand{\arraystretch}{1.15}",
        r"  \begin{tabular}{@{}c c c c c c c@{}}",
        r"    \hline",
        r"      & \textbf{System} & \textbf{Tot.} & \textbf{Cons.} &",
        r"    \textbf{Conv.} & \textbf{Elem.} & \textbf{Other} \\",
        r"    \hline",
    ]

    for batch_size in BATCH_SIZES:
        for system_index, system in enumerate(SYSTEMS):
            row = by_key[(batch_size, system)]
            prefix = (
                rf"    \multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}{{Batch size = {batch_size}}}}}"
                if system_index == 0
                else "   "
            )
            system_label = TABLE_SYSTEM_LABELS[system]
            if system == "wiseconv":
                system_label = rf"\textbf{{{system_label}}}"
            cells = [
                f"{float(row['total_latency_ms_per_frame']):.2f}",
                *(stage_cell(row, category) for category in CATEGORIES),
            ]
            lines.append(
                f"{prefix} & {system_label} & " + " & ".join(cells) + r" \\"
            )
        lines.append(r"    \hline")

    lines.extend(
        [
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(make_table(rows), encoding="utf-8")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
