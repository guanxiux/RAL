#!/usr/bin/env python3
"""Generate the single-column cost-decomposition LaTeX table from CSV."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cost_decomposition.csv"
OUTPUT = HERE / "cost_decomposition_table.tex"

BINS = ("low", "high")
SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
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
    milliseconds = float(row[f"{category}_ms"])
    percent = float(row[f"{category}_percent"])
    return rf"{milliseconds:.2f}({percent:.1f}\%)"


def make_table(rows: list[dict]) -> str:
    expected = {(activity_bin, system) for activity_bin in BINS for system in SYSTEMS}
    by_key = {(row["activity_bin"], row["system"]): row for row in rows}
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError("paper cost-decomposition CSV has unexpected rows")

    lines = [
        r"\begin{table}[!ht]",
        r"  \centering",
        r"  \caption{YOLOv8n latency decomposition on RTX~3080 into worklist",
        r"  construction (Cons.), convolution (Conv.), elementwise operators",
        r"  (Elem., e.g., activation and normalization),",
        r"  and other work, with stage entries in ms and percentages of total latency",
        r"  in parentheses.}",
        r"  \Description{A table divides every YOLOv8n frame into bins with low and high",
        r"  active ratios and compares latency attributed to",
        r"  construction, convolution, elementwise, and other work for Dense, Tile",
        r"  skipping, Gather-scatter, and WISEConv. Construction is applicable only",
        r"  to WISEConv.}",
        r"  \label{tab:cost-decomposition}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{1.2pt}",
        r"  \renewcommand{\arraystretch}{1.15}",
        r"  \begin{tabular}{@{}c c c c c c c@{}}",
        r"    \hline",
        r"      & \textbf{System} & \textbf{Tot.} & \textbf{Cons.} &",
        r"      \textbf{Conv.} & \textbf{Elem.} & \textbf{Other} \\",
        r"    \hline",
    ]

    for bin_index, activity_bin in enumerate(BINS):
        first = by_key[(activity_bin, SYSTEMS[0])]
        ratio = 100.0 * float(first["required_work_ratio_mean"])
        bin_label = f"{activity_bin.capitalize()} ({ratio:.1f}\\%)"
        for system_index, system in enumerate(SYSTEMS):
            row = by_key[(activity_bin, system)]
            prefix = (
                rf"    \multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}{{{bin_label}}}}}"
                if system_index == 0
                else "   "
            )
            system_label = TABLE_SYSTEM_LABELS[system]
            if system == "wiseconv":
                system_label = rf"\textbf{{{system_label}}}"
            cells = [
                f"{float(row['total_latency_ms']):.2f}",
                *(stage_cell(row, category) for category in CATEGORIES),
            ]
            lines.append(f"{prefix} & {system_label} & " + " & ".join(cells) + r" \\")
        lines.append(r"    \hline")

    lines.extend(
        [
            r"  \end{tabular}",
            r"  \par\vspace{6pt}",
            r"  \noindent\parbox{\columnwidth}{\footnotesize\textit{Notes.} Low and High",
            r"  median-split the evaluated MOT16 frames by active ratio; labels report bin",
            r"  means. Each stage time multiplies its per-frame NSYS fraction by the",
            r"  corresponding end-to-end CUDA-event latency.}",
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
