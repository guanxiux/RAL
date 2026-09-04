#!/usr/bin/env python3
"""Generate the paired FP32/FP16 sensitivity LaTeX table."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dtype_decomposition.csv"
OUTPUT = HERE / "dtype_decomposition_table.tex"
SETTINGS = ("dynconv_rtx3080", "fireflownet_agx")
DTYPES = ("fp32", "fp16")
SYSTEMS = ("dense", "wiseconv")
CATEGORIES = ("construction", "convolution", "elementwise", "other")


def stage_cell(row: dict, category: str) -> str:
    if category == "construction" and row["system"] == "dense":
        return r"\textemdash{}"
    milliseconds = float(row[f"{category}_ms"])
    percent = float(row[f"{category}_percent"])
    return rf"{milliseconds:.2f}({percent:.1f}\%)"


def make_table(rows: list[dict]) -> str:
    expected = {
        (setting, dtype, system)
        for setting in SETTINGS for dtype in DTYPES for system in SYSTEMS
    }
    by_key = {
        (row["setting"], row["dtype"], row["system"]): row for row in rows
    }
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError("dtype-decomposition CSV has unexpected rows")

    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{FP32 (SIMT) and FP16 (tensor core) full-model latency",
        r"  decomposition for DynConv on RTX~3080 and FireFlowNet on AGX Orin.",
        r"  Entries are milliseconds, with shares of total latency in parentheses.}",
        r"  \Description{A table compares Dense and WISEConv in FP32 and FP16 for",
        r"  DynConv on RTX 3080 and FireFlowNet on AGX Orin, decomposing latency",
        r"  into construction, convolution, elementwise, and other work.}",
        r"  \label{tab:dtype-decomposition}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.0pt}",
        r"  \renewcommand{\arraystretch}{1.10}",
        r"  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}c c c c c c c c@{}}",
        r"    \hline",
        r"    \textbf{Setting} & \textbf{Type} & \textbf{System} & \textbf{Tot.}",
        r"    & \textbf{Cons.} & \textbf{Conv.} & \textbf{Elem.} & \textbf{Other} \\",
        r"    \hline",
    ]
    for setting in SETTINGS:
        first = by_key[(setting, DTYPES[0], SYSTEMS[0])]
        setting_label = (
            rf"\shortstack{{{first['workload_label']}\\{first['device_label']}}}"
        )
        row_index = 0
        for dtype in DTYPES:
            for system in SYSTEMS:
                row = by_key[(setting, dtype, system)]
                prefix = (
                    rf"    \multirow{{4}}{{*}}{{{setting_label}}}"
                    if row_index == 0 else "   "
                )
                dtype_cell = (
                    rf"\multirow{{2}}{{*}}{{{dtype.upper()}}}"
                    if system == SYSTEMS[0] else ""
                )
                system_label = row["system_label"]
                if system == "wiseconv":
                    system_label = rf"\textbf{{{system_label}}}"
                cells = [
                    f"{float(row['total_ms']):.2f}",
                    *(stage_cell(row, category) for category in CATEGORIES),
                ]
                lines.append(
                    f"{prefix} & {dtype_cell} & {system_label} & "
                    + " & ".join(cells) + " \\\\"
                )
                row_index += 1
        lines.append(r"    \hline")
    lines.extend([
        r"  \end{tabular*}",
        r"  \par\vspace{5pt}",
        r"  \noindent\parbox{\textwidth}{\footnotesize\textit{Notes.} Stage",
        r"  fractions use the same semantic attribution as",
        r"  Table~\ref{tab:cost-decomposition} and are scaled to each setting's",
        r"  full-trace CUDA-event latency.}",
        r"\end{table*}",
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
