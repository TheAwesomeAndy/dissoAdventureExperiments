# Andrew Lane Dissertation — Final ProQuest Source

This package contains the final LaTeX source, bibliography, referenced figures,
and compiled PDF for:

*Affective Reservoir-Spike Processing and Inference Network (ARSPI-Net): A
Four-Level Interpretable Neuromorphic Framework for Clinical EEG Analysis*

## Build

From the package root, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The build requires a current TeX distribution with `pdfx`, `graphicx`,
`subcaption`, `mathtools`, `amsmath`, `float`, `booktabs`, `tabularx`,
`geometry`, `setspace`, `etoolbox`, and `pdfpages`.

`main.xmpdata` supplies archival metadata, and `main.tex` requests PDF/A-2b.
The supplied compiled PDF is letter size, unencrypted, and contains embedded
fonts.

## Front matter

The unsigned committee page is intentionally retained as page ii. Stony Brook
University's June 2026 dissertation guidelines require that page in the
dissertation PDF while the separately signed approval form is submitted as a
different document. Committee names follow the signed form: Wendy Tang, Alex
Doboli, Brady Nelson, and Sangjin Hong.

## Interpretation boundary

The package does not contain the raw Stress, Health, and the Psychophysiology of
Emotion (SHAPE) inputs. The corrected fold-local real-data pipeline therefore
could not be rerun during the post-defense audit. The dissertation identifies
affected archived numerical results as descriptive legacy estimates and states
the corrected specification without inventing replacement values.

## Included files

- `main.tex` and chapter/appendix source files
- `Disso.bib`
- `main.xmpdata`
- only the figure files referenced by the manuscript
- the compiled PDF
- `FINAL_AUDIT_CHANGELOG.md`

Temporary build files are intentionally excluded.
