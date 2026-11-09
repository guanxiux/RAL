#!/usr/bin/env python3
"""Generate the peak-memory LaTeX table from ``memory.csv`` (rows are systems)."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "memory.csv"
OUTPUT = HERE / "memory_table.tex"

SYSTEMS = (
    ("dense", "Dense"),
    ("tile_skip", "Tile skipping"),
    ("gather_scatter", "Gather-scatter"),
    ("wiseconv", "WISEConv"),
)


def cell(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value in (None, ""):
        return r"\textemdash{}"
    return f"{float(value):.1f}"


def make_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \caption{Peak live GPU allocation (MiB) on RTX~3080 (FP32; batch 1 unless noted).}",
        r"  \label{tab:memory}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \hline",
        r"    \textbf{System} & " + " & ".join(rf"\textbf{{{row['workload_label']}}}" for row in rows) + r" \\",
        r"    \hline",
    ]
    for field, label in SYSTEMS:
        cells = [cell(row, field) for row in rows]
        if field == "wiseconv":
            label = rf"\textbf{{{label}}}"
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"    \hline",
        r"  \end{tabular}",
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
