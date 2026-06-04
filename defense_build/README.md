# ARSPI-Net defense deck — build

Reproducible build for the restructured PowerPoint defense and its spoken transcript.

## Outputs
- `../defense/ARSPINet_Defense.pptx` — ~40 main + ~15 backup slides, 16:9, Stony Brook theme.
- `../defense/spoken_transcript.md` — word-for-word spoken script (also in each slide's notes).

## How to regenerate
```bash
pip install python-pptx Pillow PyMuPDF matplotlib
# LaTeX (equations + TikZ diagrams + tables):
apt-get install -y --no-install-recommends \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-pictures texlive-fonts-recommended

python3 defense_build/latexgen.py     # -> figpng/eq_*, tbl_*, tikz_* (transparent PNGs)
python3 defense_build/build_deck.py    # -> defense/ARSPINet_Defense.pptx + spoken_transcript.md
python3 defense_build/preview.py       # -> defense_build/preview/*.png  (visual QA)
```

## Where each visual comes from
- **Equations & diagrams** — authored in LaTeX/TikZ in `latex/*.tex`, rendered to transparent
  PNGs by `latexgen.py`. Equations reuse the exact source from `../ARSPI-Net_Defense.tex`.
  TikZ originals: volume-conduction head, EEG-vs-fMRI resolution, reservoir/LSM schematic,
  Cover's-theorem lift, neuromorphic energy chart. Tables (booktabs): interpretability
  landscape, centered baselines, 4-level taxonomy, disorder→layer, three generations.
- **Result figures** — the dissertation's own PDFs under `../pictures/**`, rasterized on demand
  by `build_deck.py` (PyMuPDF, ~300 dpi), cached in `figpng/`.
- **External images** — `assets/`; sources and licenses in `ATTRIBUTIONS.md`.
- **Animation** — one embedded clip (`../pictures/animations/lsm_membrane_raster.mp4`) with a
  poster frame. Plays in PowerPoint/Keynote.

## Editing
- Slides + narration live in one place: the `SLIDES` list in `build_deck.py`. Each entry's
  `say` is the spoken text (→ speaker notes **and** the transcript). Re-run `build_deck.py`.
- To change an equation/diagram, edit its `latex/*.tex` (or the body in `latexgen.py`) and
  re-run `latexgen.py`, then `build_deck.py`.

## Notes
- LibreOffice headless does not run in some sandboxes; `preview.py` reproduces the exact slide
  geometry in PIL for visual QA without it.
- `figpng/`, `preview/`, and `latex/_build/` are regenerable and git-ignored.
