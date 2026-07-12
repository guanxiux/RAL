# RAL Paper Draft

This directory contains the current WISEConv RAL paper draft.

The LaTeX files are the source of truth for the paper story:

- `main.tex`: title, abstract, keywords, and section order.
- `src/intro.tex`: polished introduction and highest-priority story text.
- `src/background.tex`: dense convolution, mask sources, execution paradigms,
  and the sparsity-to-latency gap.
- `src/design.tex`: WISEConv two-stage design.
- `src/impl.tex`: CUDA/PyTorch implementation sketch.

`outline.md` is a derived working summary. It should be updated from the LaTeX,
not used to override the LaTeX.
