#!/usr/bin/env python3
"""Generate the paired-device YOLOv8n ablation stage-decomposition table."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ablation.csv"
OUTPUT = HERE / "ablation_table.tex"
DEVICES = ("rtx3080", "agx_orin")
GROUPS = (
    ("Const.", ("atomic_append", "global_order", "no_reuse")),
    ("Comp.", ("dp_only", "no_hybrid", "sync_exact_selector")),
)
VARIANTS = tuple(variant for _, keys in GROUPS for variant in keys) + ("ours",)


def cell(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value in (None, ""):
        return r"\textemdash{}"
    return f"{float(value):.2f}"


def make_table(rows: list[dict]) -> str:
    expected = {(device, variant) for device in DEVICES for variant in VARIANTS}
    by_key = {(row["device"], row["variant"]): row for row in rows}
    if set(by_key) != expected or len(rows) != len(expected):
        raise ValueError("ablation CSV has unexpected rows")

    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{YOLOv8n component ablation on RTX~3080 and AGX~Orin.",
        r"  \textit{Const.} and \textit{Comp.} group the variants that replace one",
        r"  construction-stage and compute-stage mechanism, respectively; all other",
        r"  settings---weights, inputs, upstream masks, and deployed tile",
        r"  configuration---are held fixed. Cons. and Conv. are the construction- and",
        r"  compute-stage GPU kernel time (ms) profiled in a single eager pass;",
        r"  Tot. is the full-model end-to-end latency (ms), which also includes",
        r"  non-kernel time and therefore need not equal Cons.$+$Conv.}",
        r"  \Description{A table compares seven WISEConv design variants on RTX",
        r"  3080 and AGX Orin, grouped into construction-stage (Const.) and",
        r"  compute-stage (Comp.) variants. Each platform reports construction,",
        r"  convolution, and total latency in milliseconds.}",
        r"  \label{tab:ablation}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{1.6pt}",
        r"  \renewcommand{\arraystretch}{1.06}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}c c c c c c c c@{}}",
        r"    \hline",
        r"    & \multirow{2}{*}{\textbf{Variant}} &",
        r"    \multicolumn{3}{c}{\textbf{RTX 3080}} &",
        r"    \multicolumn{3}{c}{\textbf{AGX Orin}} \\",
        r"    \cline{3-5}\cline{6-8}",
        r"    & & \textbf{Cons.} & \textbf{Conv.} & \textbf{Tot.}",
        r"    & \textbf{Cons.} & \textbf{Conv.} & \textbf{Tot.} \\",
        r"    \hline",
    ]

    def emit(variant: str, bold: bool = False, lead: str = "") -> None:
        left = by_key[(DEVICES[0], variant)]
        label = left["variant_label"].replace("&", r"\&")
        if bold:
            label = rf"\textbf{{{label}}}"
        cells = []
        for device in DEVICES:
            row = by_key[(device, variant)]
            cells.extend([
                cell(row, "construction_ms"),
                cell(row, "convolution_ms"),
                cell(row, "total_ms"),
            ])
        lines.append(f"    {lead} & {label} & " + " & ".join(cells) + r" \\")

    for group_index, (group, keys) in enumerate(GROUPS):
        if group_index:
            lines.append(r"    \hline")
        for index, variant in enumerate(keys):
            lead = (
                rf"\multirow{{3}}{{*}}{{\rotatebox[origin=c]{{90}}"
                rf"{{\strut {group}}}}}"
                if index == 0 else ""
            )
            emit(variant, lead=lead)
    lines.append(r"    \hline")
    emit("ours", bold=True)
    lines.extend([
        r"    \hline",
        r"  \end{tabular*}",
        r"  \par\vspace{5pt}",
        r"  \noindent\parbox{\columnwidth}{\footnotesize\textit{Variants.}",
        r"  \textit{Atomic append}: one global atomic per coordinate instead of",
        r"  per-segment reservation.",
        r"  \textit{Global order}: two-pass stable compaction for full",
        r"  dense-traversal order.",
        r"  \textit{No reuse}: rebuild worklist at every layer.",
        r"  \textit{DP only}: force data-parallel decomposition, disable",
        r"  Stream-$K$ adaptation.",
        r"  \textit{No hybrid}: select between data-parallel and full Stream-$K$",
        r"  only, disabling the hybrid DP-prefix\,+\,Stream-$K$-tail combination.",
        r"  \textit{Readback \& launch}: synchronize worklist length to host and",
        r"  relaunch per operator.}",
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
