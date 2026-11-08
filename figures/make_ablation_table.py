#!/usr/bin/env python3
"""Generate the wise2 ablation table (YOLOv8m batch 1, RTX 3080, FP16) from ablation.csv (ablation_stats.py)."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ablation.csv"
OUTPUT = HERE / "ablation_table.tex"
GROUPS = (("Cons.", ("no_reuse",)), ("Comp.", ("shuffled", "readback", "full_grid")),
          ("Sel.", ("dense_tile", "speedup_tile", "latency_tile", "kmax", "oracle")))


def cell(row: dict, field: str) -> str:
    value = row.get(field, "")
    return r"\textemdash{}" if value in (None, "") else f"{float(value):.2f}"


def make_table(rows: list[dict]) -> str:
    by = {row["variant"]: row for row in rows}
    expected = {k for _, keys in GROUPS for k in keys} | {"ours"}
    if set(by) != expected:
        raise ValueError("ablation CSV has unexpected rows")
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{Co-design ablation on YOLOv8m (batch 1, RTX~3080, FP16). Each variant replaces one",
        r"  mechanism and keeps the weights, inputs, upstream masks and every other component. Cons. and",
        r"  Conv. are the construction- and compute-stage GPU kernel time (ms); Tot. is the full-model latency",
        r"  (ms, aligned to the whole-video end-to-end latency of Ours); $\times$ is relative to Ours unless noted.}",
        r"  \Description{A table compares WISEConv design variants on RTX 3080 in FP16, grouped into",
        r"  construction-stage, compute-stage and tile-selection variants, reporting construction, convolution",
        r"  and total latency in milliseconds and the ratio to the full design.}",
        r"  \label{tab:ablation}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{2.2pt}",
        r"  \renewcommand{\arraystretch}{1.06}",
        r"  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}c l c c c c@{}}",
        r"    \hline",
        r"    & \textbf{Variant} & \textbf{Cons.} & \textbf{Conv.} & \textbf{Tot.} & \textbf{$\times$} \\",
        r"    \hline",
    ]

    def emit(key: str, lead: str = "", bold: bool = False) -> None:
        row = by[key]
        label = rf"\textbf{{{row['variant_label']}}}" if bold else row["variant_label"]
        ratio = f"{float(row['ratio']):.2f}" + (r"$^{\dagger}$" if row["reference"] != "ours" else "")
        lines.append(f"    {lead} & {label} & {cell(row, 'construction_ms')} & {cell(row, 'convolution_ms')} & "
                     f"{cell(row, 'total_ms')} & {ratio} \\\\")

    for gi, (group, keys) in enumerate(GROUPS):
        if gi:
            lines.append(r"    \hline")
        for i, key in enumerate(keys):
            lead = (rf"\multirow{{{len(keys)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\strut {group}}}}}"
                    if i == 0 and len(keys) > 1 else (group if i == 0 else ""))
            emit(key, lead)
    lines.append(r"    \hline")
    emit("ours", bold=True)
    lines.extend([
        r"    \hline",
        r"  \end{tabular*}",
        r"  \par\vspace{5pt}",
        r"  \noindent\parbox{\columnwidth}{\footnotesize\textit{Variants.}",
        r"  \textit{No reuse}: rebuild the worklist at every layer.",
        r"  \textit{Shuffled order}: permute each worklist on the device before the compute stage (permutation",
        r"  not counted; convolution time).",
        r"  \textit{Readback \& launch}: read $n_W$ back to the host per layer and launch the exact grid.",
        r"  \textit{Full-layer grid}: launch the dense layer's grid; CTAs beyond the worklist exit.",
        r"  \textit{Dense-tuned / Best single / Latency-optimal tile}: Ours' candidate structure (the same",
        r"  layers, candidate counts and thresholds), every candidate the tile fastest at $\alpha{=}1$, with the",
        r"  best expected speedup, or with the lowest expected latency (uniform $n_W$).",
        r"  \textit{Cost-unaware $K_{\max}$}: up to four candidates at every layer, ignoring their cost.",
        r"  \textit{Zero-cost oracle}: those candidates, each frame on a graph captured for its own $n_W$, with",
        r"  no selection mechanism.",
        r"  $^{\dagger}$Relative to the best single tile without the selection mechanism.}",
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
