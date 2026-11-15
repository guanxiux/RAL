#!/bin/bash
# Build the Background schematics (bg_*.tex), the Design overview (overview.tex) and the intro figure (teaser.tex), shared style bgstyle.tex,
# into PDFs next to them.
cd "$(dirname "$0")"
python3 teaser_assets.py && python3 teaser_traj.py > /dev/null || { echo "failed: teaser assets"; exit 1; }
for f in bg_*.tex overview.tex teaser.tex; do pdflatex -interaction=nonstopmode -halt-on-error "$f" > /dev/null || { echo "failed: $f"; exit 1; }; done
rm -f bg_*.aux bg_*.log overview.aux overview.log teaser.aux teaser.log
