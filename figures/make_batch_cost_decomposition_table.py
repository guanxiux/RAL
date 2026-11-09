#!/usr/bin/env python3
"""Generate the stream-batching latency-decomposition table (YOLOv8m, RTX 3080, FP32) from batch_cost_decomposition.csv."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "batch_cost_decomposition.csv"
OUTPUT = HERE / "batch_cost_decomposition_table.tex"
FIELDS = ("construction_ms", "convolution_ms", "gather_ms", "other_ms", "idle_ms")
BATCH_SIZES = (1, 4)


def cell(row: dict, field: str) -> str:
    v = float(row[field])
    if v < 0.005:
        return r"\textemdash{}"
    return f"{v:.2f}\\,({100 * v / float(row['total_ms']):.0f})"


def make_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{YOLOv8m latency decomposition under stream-level batching (RTX~3080, FP32). Each batch",
        r"  combines one current frame of each of $B$ disjoint temporal streams; times are per frame. Stage",
        r"  definitions and cell format follow Table~\ref{tab:cost-decomposition}.}",
        r"  \Description{A table compares Dense, Tile skipping, Gather-scatter and WISEConv at batch sizes one and",
        r"  four, dividing the per-frame latency into construction, convolution, gather and scatter, other GPU work,",
        r"  and GPU idle time.}",
        r"  \label{tab:batch-cost-decomposition}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{1.5pt}",
        r"  \renewcommand{\arraystretch}{1.13}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}c l c c c c c c@{}}",
        r"    \hline",
        r"      & \textbf{System} & \textbf{Tot.} & \textbf{Cons.} & \textbf{Conv.} & \textbf{G/S} &",
        r"      \textbf{Other} & \textbf{Idle} \\",
        r"    \hline",
    ]
    for bi, batch in enumerate(BATCH_SIZES):
        group = [r for r in rows if int(r["batch_size"]) == batch]
        if len(group) != 4:
            raise ValueError(f"batch {batch}: expected 4 rows")
        if bi:
            lines.append(r"    \hline")
        for i, row in enumerate(group):
            lead = rf"\multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}{{\strut $B{{=}}{batch}$}}}}" if i == 0 else ""
            name = rf"\textbf{{{row['system_label']}}}" if row["system"] == "wiseconv" else row["system_label"]
            lines.append(f"    {lead} & {name} & {float(row['total_ms']):.2f} & " +
                         " & ".join(cell(row, f) for f in FIELDS) + r" \\")
    lines.extend([
        r"    \hline",
        r"  \end{tabular*}",
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
