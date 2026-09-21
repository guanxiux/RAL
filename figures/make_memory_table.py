#!/usr/bin/env python3
"""Generate the peak-memory LaTeX table from ``memory.csv``."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "memory.csv"
OUTPUT = HERE / "memory_table.tex"


def cell(row: dict, field: str) -> str:
    value = row.get(field, "")
    if value in (None, ""):
        return r"\textemdash{}"
    return f"{float(value):.1f}"


def make_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \caption{Peak GPU memory (MiB) on RTX~3080.}",
        r"  \label{tab:memory}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \renewcommand{\arraystretch}{1.08}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \hline",
        r"    \textbf{Workload} & \textbf{Dense} & \textbf{Tile skip} & "
        r"\textbf{Gather-scatter} & \textbf{\sysname} \\",
        r"    \hline",
    ]
    for row in rows:
        lines.append(
            f"    {row['workload_label']} & {cell(row, 'dense')} & "
            f"{cell(row, 'tile_skip')} & {cell(row, 'gather_scatter')} & "
            f"{cell(row, 'wiseconv')} \\\\"
        )
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
