#!/usr/bin/env python3
"""Generate the two-platform cost-decomposition LaTeX table."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "cost_decomposition.csv"
OUTPUT = HERE / "cost_decomposition_table.tex"
DEVICES = ("rtx3080", "agx_orin")
SYSTEMS = ("dense", "tile_skip", "gather_scatter", "wiseconv")
SYSTEM_LABELS = {
    "dense": "Dense",
    "tile_skip": "Tile skipping",
    "gather_scatter": "Gather",
    "wiseconv": "WISEConv",
}
CATEGORIES = ("construction", "convolution", "elementwise", "other")


def stage_cell(row: dict, category: str) -> str:
    if row is None:
        return r"\textemdash{}"
    if category == "construction" and row["system"] != "wiseconv":
        return r"\textemdash{}"
    milliseconds = float(row[f"{category}_ms"])
    percent = float(row[f"{category}_percent"])
    return rf"{milliseconds:.2f}({percent:.1f}\%)"


def make_table(rows: list[dict]) -> str:
    by_key = {(row["device"], row["system"]): row for row in rows}
    present_devices = [d for d in DEVICES
                       if any((d, s) in by_key for s in SYSTEMS)]
    if not present_devices:
        raise ValueError("cost-decomposition CSV has no rows")

    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{Full-model YOLOv8n latency decomposition on RTX~3080 and",
        r"  AGX~Orin.}",
        r"  \Description{A table compares Dense, Tile skipping, Gather-scatter,",
        r"  and WISEConv on RTX 3080 and AGX Orin, decomposing full-model latency",
        r"  into construction, convolution, elementwise, and other work.}",
        r"  \label{tab:cost-decomposition}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2.35pt}",
        r"  \renewcommand{\arraystretch}{1.13}",
        r"  \begin{tabular}{@{}c c c c c c c@{}}",
        r"    \hline",
        r"      & \textbf{System} & \textbf{Tot.} & \textbf{Cons.} &",
        r"      \textbf{Conv.} & \textbf{Elem.} & \textbf{Other} \\",
        r"    \hline",
    ]
    for device in present_devices:
        device_row = next(by_key[(device, s)] for s in SYSTEMS
                          if (device, s) in by_key)
        platform = device_row["device_label"].replace(" ", "~")
        for index, system in enumerate(SYSTEMS):
            row = by_key.get((device, system))
            prefix = (
                rf"    \multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}"
                rf"{{\strut {platform}}}}}"
                if index == 0 else "   "
            )
            label = SYSTEM_LABELS.get(system, system)
            if system == "wiseconv":
                label = rf"\textbf{{{label}}}"
            if row is None:
                cells = [r"\textemdash{}"] * (1 + len(CATEGORIES))
            else:
                cells = [
                    f"{float(row['total_ms']):.2f}",
                    *(stage_cell(row, category) for category in CATEGORIES),
                ]
            lines.append(f"{prefix} & {label} & " + " & ".join(cells) + " \\\\")
        lines.append(r"    \hline")
    lines.extend([
        r"  \end{tabular}",
        r"  \par\vspace{5pt}",
        r"  \noindent\parbox{\columnwidth}{\footnotesize\textit{Notes.} Entries",
        r"  are milliseconds; percentages are shares of each row's total. Cons. is the",
        r"  WISEConv worklist construction, which the baselines do not have; Other is",
        r"  extraction, scatter, mask, and setup work.}",
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
