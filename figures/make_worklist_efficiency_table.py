#!/usr/bin/env python3
"""Generate the single-column WISEConv efficiency/worklist LaTeX table."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "worklist_efficiency.csv"
OUTPUT = HERE / "worklist_efficiency_table.tex"
WORKLOADS = ("fireflownet", "yolov8n", "yolov8m", "dynconv_pose")


def percentage(value: str) -> str:
    return f"{100.0 * float(value):.1f}\\%"


def percentile_length(value: str) -> str:
    return f"{float(value):,.0f}".replace(",", "{,}")


def make_table(rows: list[dict]) -> str:
    by_workload = {row["workload"]: row for row in rows}
    if set(by_workload) != set(WORKLOADS) or len(rows) != len(WORKLOADS):
        raise ValueError("paper worklist-efficiency CSV has unexpected rows")

    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{WISEConv $\eta$ decomposition and worklist-length statistics",
        r"  on RTX~3080. The $n_{\mathcal{W}}$ percentiles are computed from worklist",
        r"  lengths pooled across all layers and frames.}",
        r"  \Description{A table reports eta construct, eta compute, eta, and the",
        r"  P10, P50, and P90 worklist lengths for FireFlowNet, YOLOv8n, YOLOv8m,",
        r"  and DynConv on RTX 3080. Worklist-length statistics are computed from samples",
        r"  pooled across all layers and frames.}",
        r"  \label{tab:worklist-efficiency}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{1.5pt}",
        r"  \renewcommand{\arraystretch}{1.12}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l c c c r r r@{}}",
        r"    \hline",
        r"    \multirow{2}{*}{\textbf{Workload}} &",
        r"    \multirow{2}{*}{$\boldsymbol{\eta}_{\mathrm{construct}}$} &",
        r"    \multirow{2}{*}{$\boldsymbol{\eta}_{\mathrm{compute}}$} &",
        r"    \multirow{2}{*}{$\boldsymbol{\eta}$} &",
        r"    \multicolumn{3}{c}{$\boldsymbol{n}_{\mathcal{W}}$} \\",
        r"    \cline{5-7}",
        r"    & & & & \textbf{P10} & \textbf{P50} & \textbf{P90} \\",
        r"    \hline",
    ]
    for workload in WORKLOADS:
        row = by_workload[workload]
        lines.append(
            f"    {row['workload_label']} & "
            f"{percentage(row['eta_construct_frame_mean'])} & "
            f"{percentage(row['eta_compute_frame_mean'])} & "
            f"{percentage(row['useful_compute_ratio_frame_mean'])} & "
            f"${percentile_length(row['p10_worklist_length'])}$ & "
            f"${percentile_length(row['p50_worklist_length'])}$ & "
            f"${percentile_length(row['p90_worklist_length'])}$ \\\\"
        )
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
