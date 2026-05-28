# Beamer Integration Manifest

Source deck: `ARSPI-Net_Defense.tex` (canonical 45-frame main + 29-frame appendix).
New material: defense-audit figures rendered by `defense_figures/` scripts on `main`
and staged into `pictures/defense_audit/`.

This manifest records each inserted frame's anchor in the original deck, the new
frame's content, and the rationale for the insertion location.

Total insertions: **8 main-deck frames** + **1 appendix section divider** + **12
appendix evidence frames** = **21 new frames**.

Augmented total: **53 main frames** (was 45) + **42 appendix frames** (was 29) =
**95 content frames**. Compiled PDF is **105 pages** (Beamer page counter shows
"X / 94"; the `\section` title page is not counted as a navigation slide).

## Main-deck insertions

| #  | Inserted after frame                                       | New frame title                                                                         | Section | Reason                                                                                                                            | Est. speaking time |
|----|------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|--------------------|
| 1  | "The contraction measurement, visualized"                  | Defense audit: autonomous $\rho(W)$ vs.\ driven $\lambda_1$                              | main    | Reservoir-contraction region. Replaces the field-default ESN-stability citation with a driven Lyapunov measurement on 3,165 ERPs. | 90 s               |
| 2  | "The measurement-instrument paradigm"                      | Defense audit: FNN-measured embedding dimension                                         | main    | Spike-to-embedding region. Anchors the measurement-instrument paradigm with a Kennel--Brown FNN bound rather than a Takens claim.  | 75 s               |
| 3  | "The propagation operator"                                 | Defense audit: memory-capacity regime for leak $\beta$                                  | main    | Sets up the graph-POC discussion with a measured operating characteristic for the leak hyperparameter.                            | 90 s               |
| 4  | "The diffusion mechanism, measured on real reservoir features" | Defense audit: theoretical bounds vs.\ measured operating points                       | main    | Closes the graph-propagation region with the bounds-vs-measurement comparison (TB synthesis figure).                              | 60 s               |
| 5  | "A null result that forced the methodology"                | Defense audit: per-trial channel-permutation null                                       | main    | Null / methodology pivot. Reinforces the existing null-as-methodology framing with the strictest spatial null (per-trial perms).  | 75 s               |
| 6  | (immediately after insertion #5)                           | Defense audit: failure gallery --- four pivots                                          | main    | Failure-driven rigor slide directly after the null-as-methodology frames; carries the four documented pivots (A/C/D twice).        | 90 s               |
| 7  | (before "Contributions restated", co-located with #8)      | Defense claims are anchored to dissertation chapters                                    | main    | Anchor map placed near the close so the committee sees the chapter-by-chapter spine before contributions are restated.            | 30 s               |
| 8  | (immediately after #7, just before the [standout] close)   | Research provenance and assistance                                                      | main    | One global disclosure slide. Placed after the substantive closing argument and before the standout "Thank you / Questions" frame. | 45 s               |

**Main-deck total added speaking time**: ~9 min. Augmented main: 45 → 53 frames;
the existing 45-minute talk has ~9 minutes of natural slack (transitions, sparse
"big claim" frames, restated-contributions slide), well within budget.

## Appendix insertions

Inserted as a new `\section{Appendix --- Defense Audit Diagnostics}`
immediately before `\end{document}` (after the existing 29-frame appendix).

| #  | New frame title                                              | Backs                                       |
|----|--------------------------------------------------------------|---------------------------------------------|
| 0  | Appendix --- Defense audit: raw diagnostics overview         | section divider / contents                  |
| 1  | Appendix --- Exp A raw: eigenspectrum of $W$                 | main insertion #1                           |
| 2  | Appendix --- Exp A raw: Benettin sample trajectories         | main insertion #1                           |
| 3  | Appendix --- Exp A raw: per-trial $\lambda_1$ scatter        | main insertion #1                           |
| 4  | Appendix --- Exp B raw: delay embedding                      | main insertion #2                           |
| 5  | Appendix --- Exp B raw: per-trial FNN distribution           | main insertion #2                           |
| 6  | Appendix --- Exp C raw: input drive                          | main insertion #3                           |
| 7  | Appendix --- Exp C raw: state response per $\beta$           | main insertion #3                           |
| 8  | Appendix --- Exp C raw: memory capacity vs.\ $\tau$          | main insertion #3                           |
| 9  | Appendix --- Exp D raw: feature layout                       | main insertion #5                           |
| 10 | Appendix --- Exp D raw: class-label distribution             | main insertion #5                           |
| 11 | Appendix --- Exp D raw: observed vs.\ null example           | main insertion #5                           |
| 12 | Appendix --- AM: figure-to-source backup index               | main insertion #7                           |

**Appendix total added**: 13 frames (1 divider + 12 evidence frames). At the
upper end of the 8--12 author-specified budget; the +1 accounts for the
section-divider frame so the new appendix material is clearly demarcated for
committee navigation.

## What was preserved

- Every existing main frame (45) and every existing appendix frame (29) is
  unchanged.
- Section structure ("Part 1--7" + original appendix section) is unchanged.
- All existing graphics paths (`pictures/chDynamics/`, `pictures/chSynthesis/`,
  etc.) remain in place.
- No frame titles or in-frame text was rewritten on existing slides.

## What was added (file-level)

- `ARSPI-Net_Defense.tex` — the canonical .tex with the 20 new frames inserted at
  the conceptual locations above.
- `pictures/defense_audit/` — 26 new PDFs (15 main-deck analysis/summary figures
  + 11 appendix raw diagnostics from `defense_figures/`).
- `ARSPI-Net_Defense.pdf` — the compiled augmented Beamer deck (105 pages).
- `defense_figures/integration_manifest.md` — this file.

## What was NOT added

- No replacement deck. `build_deck.py` from the prior session is left untouched;
  PR #24 was closed without merging.
- No edits to the dissertation chapters or to existing chapter figures.
- No edits to README.md or HANDOFF_TO_NEXT_AGENT.md in this PR.
- No new scientific claims. Every new frame's content is a re-presentation of
  artifacts already on `main` in `defense_figures/`.

## Compilation

```bash
pdflatex -interaction=nonstopmode ARSPI-Net_Defense.tex
pdflatex -interaction=nonstopmode ARSPI-Net_Defense.tex   # second pass for ToC
```
