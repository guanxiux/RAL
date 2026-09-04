#!/usr/bin/env python3
"""Generate the paired-device YOLOv8n ablation LaTeX table."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ablation.csv"
OUTPUT = HERE / "ablation_table.tex"
DEVICES = ("rtx3080", "agx_orin")
VARIANTS = (
    "atomic_append",
    "global_order",
    "no_reuse",
    "dp_only",
    "no_hybrid",
    "sync_exact_selector",
    "ours",
)


def cell(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value in (None, ""):
        return r"\textemdash{}"
    return f"{float(value):.3f}"


def eta_cell(row: dict) -> str:
    value = row.get("eta_percent", "")
    if value in (None, ""):
        return r"\textemdash{}"
    return f"{float(value):.1f}\%"


def make_table(rows: list[dict]) -> str:
    expected = {
        (device, variant) for device in DEVICES for variant in VARIANTS
    }
    by_key = {(row["device"], row["variant"]): row for row in rows}
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError("ablation CSV has unexpected rows")

    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{YOLOv8n component ablation on RTX~3080; AGX Orin is",
        r"  reserved for the multi-platform refresh. Times are milliseconds.}",
        r"  \Description{A table compares seven WISEConv design variants on RTX",
        r"  3080 and reserves the corresponding AGX Orin columns. Each platform",
        r"  reports useful-compute ratio, construction time, convolution time,",
        r"  and full-model latency when available.}",
        r"  \label{tab:ablation}",
        r"  \fontsize{7.0}{7.8}\selectfont",
        r"  \setlength{\tabcolsep}{0.7pt}",
        r"  \renewcommand{\arraystretch}{1.06}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l c r r r c r r r@{}}",
        r"    \hline",
        r"    \multirow{2}{*}{\textbf{Variant}} &",
        r"    \multicolumn{4}{c}{\textbf{RTX 3080}} &",
        r"    \multicolumn{4}{c}{\textbf{AGX Orin}} \\",
        r"    \cline{2-5}\cline{6-9}",
        r"    & $\boldsymbol{\eta}$ & \textbf{Cons.} & \textbf{Conv.} & \textbf{Total}",
        r"    & $\boldsymbol{\eta}$ & \textbf{Cons.} & \textbf{Conv.} & \textbf{Total} \\",
        r"    \hline",
    ]
    for variant in VARIANTS:
        left = by_key[(DEVICES[0], variant)]
        label = left["variant_label"]
        if variant == "ours":
            label = rf"\textbf{{{label}}}"
        cells = []
        for device in DEVICES:
            row = by_key[(device, variant)]
            cells.extend([
                eta_cell(row),
                cell(row, "construction_ms"),
                cell(row, "convolution_ms"),
                cell(row, "total_ms"),
            ])
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    lines.extend([
        r"    \hline",
        r"  \end{tabular*}",
        r"  \par\vspace{5pt}",
        r"  \noindent\parbox{\columnwidth}{\scriptsize\textit{Notes.}",
        r"  $\eta$ is the full-model useful-compute ratio. Cons. and Conv. are",
        r"  construction- and convolution-stage latency; dashes indicate that",
        r"  variant-level stage attribution was not collected. AGX Orin entries",
        r"  are intentionally blank pending the multi-platform refresh.}",
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
