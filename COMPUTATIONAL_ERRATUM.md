# Computational erratum — confirmatory classification values

**Status:** authoritative for code-derived numbers. This document records a
reconciliation between the values reported in the submitted dissertation and the
values the committed pipeline in this repository actually produces.

## Why this document exists

The dissertation is a submitted degree document. Its text and tables report the
confirmatory BSC₆/SRP-64 balanced accuracy as **60.6%** (three conditions) and
**53.2%** (four categories), taken from the original computational record that
accompanied the submission (see `dissertation/main.tex`, `dissertation/chgraph.tex`,
`dissertation/chConclusion.tex`).

The committed canonical pipeline
(`experiments/confirmatory/run_confirmatory_validation.py` +
`confirmatory_pipeline.py`) is an **independent reconstruction** of that
protocol. The original analysis code is not part of this repository, and the
SRP/reservoir path was reimplemented from the described method. Run on the
authorized SHAPE data it produces **64.3%** and **54.6%** — it agrees with the
dissertation on the raw-ERP and conventional-baseline rows (within ≈1 pp) but
diverges on the reservoir/SRP rows and, more strongly, on the destruction
controls.

Both statements are true at once. They are reconciled here rather than by
silently overwriting either record.

## Authority model (Model B: frozen dissertation + erratum)

- The **dissertation PDF and LaTeX source** remain the historical degree
  document. Their reported confirmatory values (60.6% / 53.2%) are **not edited**
  to match the reconstruction; they stand as the submitted record.
- The **committed pipeline and its execution record**
  (`experiments/confirmatory/confirmatory_results.json`) are **authoritative for
  code-derived numbers**. Where a number in this repository describes what the
  committed code produces, it is the reconstruction value.
- `results_manifest.json` stores the reconstruction values under `value` and the
  submitted values under `dissertation_reported`, with a top-level
  `confirmatory_reconciliation` note.

The `dissertation/` tree is "authoritative" for the **language- and
consistency-corrected manuscript**, not for asserting that the reported
confirmatory *values* equal the committed runner's output. That equality does
not hold and is not claimed.

## Reconciliation table (BSC₆/SRP-64, balanced accuracy)

| Quantity | Dissertation (submitted) | Committed runner (reproduced) |
|---|---:|---:|
| Three conditions | 60.6% [57.8, 63.4] | **64.3% [61.1, 67.2]** |
| Four categories | 53.2% [50.6, 55.8] | **54.6% [51.9, 57.3]** |
| Advantage over conventional (3-cond) | +11.3 pp [7.9, 14.7] | +15.7 pp [11.9, 19.5] |
| Advantage over conventional (4-cat) | +16.5 pp [13.3, 19.8] | +18.5 pp [15.2, 21.7] |
| Participant-label permutation | p = 0.0099 (both) | p = 0.0099 (both) — **exact** |
| Temporal-bin-shuffle drop (3-cond) | −7.4 pp | −24.8 pp |
| Temporal-bin-shuffle drop (4-cat) | −12.9 pp | −27.8 pp |

Rows that agree (raw ERP-16, conventional baseline, permutation p) and rows that
diverge (reservoir/SRP accuracies, destruction magnitudes) are both shown, so the
scope of the divergence is explicit. The divergence is concentrated on exactly
the path that was reimplemented.

## What reproduces exactly, and what drifts

- **Accuracies and the permutation p reproduce exactly** across library versions
  (verified: three-condition SRP-64 = 64.329%, PCA-64 = 59.968%, gap +4.36 pp,
  byte-identical on reruns).
- **SVD-based descriptive variance ratios (ρ) in the exploratory study** drift at
  the ~1-unit level across BLAS/LAPACK builds; the committed JSONs and manifest
  hold the values produced by the pinned CI environment, and the qualitative
  conclusions are invariant to that drift.

## Verification

`experiments/confirmatory/check_against_manifest.py confirmatory_results.json --tol 0.05`
compares every confirmatory manifest estimand against the committed execution
record and must pass with no mismatches; CI runs exactly this. Passing tests
verify code behavior, not scientific validity.
