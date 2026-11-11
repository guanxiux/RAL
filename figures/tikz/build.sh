#!/bin/bash
# Build the Background schematics (bg_*.tex, shared style bgstyle.tex) into PDFs next to them.
cd "$(dirname "$0")"
for f in bg_*.tex; do pdflatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null || { echo "failed: $f"; exit 1; }; done
rm -f bg_*.aux bg_*.log
