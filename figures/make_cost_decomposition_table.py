#!/usr/bin/env python3
"""Generate the wise2 latency-decomposition table (YOLOv8m batch 1, RTX 3080) from cost_decomposition.csv."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cost_decomposition.csv"
OUTPUT = HERE / "cost_decomposition_table.tex"
FIELDS = ("construction_ms", "convolution_ms", "gather_ms", "other_ms", "idle_ms")


def cell(row: dict, field: str) -> str:
    v = float(row[field])
    if v < 0.005:
        return r"\textemdash{}"
    return f"{v:.2f}\\,({100 * v / float(row['total_ms']):.0f})"


def make_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{Full-model YOLOv8m latency decomposition (batch 1, RTX~3080).}",
        r"  \Description{A table decomposes the full-model latency of Dense, Tile skipping, Gather-scatter and",
        r"  WISEConv in FP32, and of Dense and WISEConv in FP16, into construction, convolution, gather and",
        r"  scatter, other GPU work, and GPU idle time.}",
        r"  \label{tab:cost-decomposition}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{1.5pt}",
        r"  \renewcommand{\arraystretch}{1.13}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}c l c c c c c c@{}}",
        r"    \hline",
        r"      & \textbf{System} & \textbf{Tot.} & \textbf{Cons.} & \textbf{Conv.} & \textbf{G/S} &",
        r"      \textbf{Other} & \textbf{Idle} \\",
        r"    \hline",
    ]
    for di, dtype in enumerate(("fp32", "fp16")):
        group = [r for r in rows if r["dtype"] == dtype]
        if di:
            lines.append(r"    \hline")
        for i, row in enumerate(group):
            lead = rf"\multirow{{{len(group)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\strut {dtype.upper()}}}}}" if i == 0 else ""
            name = rf"\textbf{{{row['system_label']}}}" if row["system"] == "wiseconv" else row["system_label"]
            lines.append(f"    {lead} & {name} & {float(row['total_ms']):.2f} & " +
                         " & ".join(cell(row, f) for f in FIELDS) + r" \\")
    lines.extend([
        r"    \hline",
        r"  \end{tabular*}",
        r"  \par\vspace{5pt}",
        r"  \noindent\parbox{\columnwidth}{\footnotesize\textit{Notes.} Entries are milliseconds per frame;",
        r"  parenthesized numbers are percentages of each row's total, the whole-video end-to-end latency. Stages are the",
        r"  kernels' shares of the GPU busy time (nsys); Cons. is the worklist construction (gather-scatter's mask",
        r"  compaction), G/S gather-scatter's feature extraction and scatter, Other the remaining operators;",
        r"  Idle is the GPU idle time within the frame (host synchronization). The baselines have no FP16",
        r"  implementation.}",
        r"\end{table}",
        "",
    ])
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
