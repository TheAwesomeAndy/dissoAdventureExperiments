# Novel experiments: what the inspectable chain buys you

These experiments target the hardest reviewer question for an
interpretability-first neuromorphic model that is deliberately **not** an
accuracy leader: *if the raw ERP classifies affect more accurately, why use the
fixed, inspectable measurement chain?* Each experiment answers with a
quantitative property that a black-box model cannot expose, computed on the same
verified operators as the confirmatory stream and reported at honest scope.

## The three contributions

### N1 — A data-independent random projection beats data-fitted PCA
The confirmatory pipeline replaced a globally fitted PCA (which leaks held-out
participants) with a fixed, data-independent SRP-64. N1 shows this is not merely
a leak-free compromise — it is *better*:

- **SRP-64 > fold-local PCA-64.** Against a *fold-local* (leak-free) PCA fitted
  only on training participants, the fixed random projection is significantly
  more accurate — the paired participant-bootstrap difference *excludes zero* at
  every tested dimension (on the three-condition cohort, +4.4pp at 64 dims,
  interval [1.5, 7.2]). The mechanism: on subject-entangled EEG the
  variance-maximizing PCA directions align to subject nuisance (the ρ≈7–10
  entanglement quantified in N3), while an isotropic random projection preserves
  condition geometry. Leak-free *and* more discriminative.
- **Johnson–Lindenstrauss dimensionality law.** Balanced accuracy across
  compression dimensions {16, 32, 64, 128} traces the expected geometry-
  preservation curve.
- **Seed stability.** The reported SRP-64 is not a lucky draw: accuracy across
  many random projection seeds has a small spread (~±1pp).

*Why it matters:* on nuisance-dominated grouped clinical EEG, unsupervised
variance-maximizing compression is actively counterproductive because it fits
the nuisance; a data-independent random projection is leak-free by construction
*and* more discriminative — a clean, reusable methods result and a concrete
argument against the default PCA step.

### N2 — The reservoir is a calibrated instrument (at the decision level)
The dissertation frames the fixed reservoir as a measurement instrument. N2
quantifies that — and is careful about *where* the reproducibility lives:

- **Reproducibility is decision-level, not feature-level.** Individual SRP
  coordinates have near-zero cross-seed ICC(2,1) (≈0.00 on real data — expected
  for a random projection: no individual coordinate is privileged). What *is*
  reproducible is the readout: accuracy is stable across projection and reservoir
  seeds (~±1pp). We report both, and do not overclaim feature-wise reliability.
- **Transfer function.** Total spike output is a monotonic function of input
  amplitude (Spearman ρ = 1.0) with a quantified dynamic range — a calibratable
  dose–response.
- **Graceful degradation.** Balanced accuracy under channel dropout and additive
  noise versus the raw-ERP reference. The reservoir readout is notably
  *noise-insensitive* (near-flat under additive noise up to 1 SD), a robustness
  property of the fixed nonlinear transform.

*Why it matters:* reframes "fixed weights" from a limitation into an instrument
guarantee — decision-level reproducibility, monotonic calibration, and known
noise behavior — while stating honestly that individual random features are not
themselves reproducible.

### N3 — Representational necessity and subject/condition information
Extends the destruction controls into a systematic map of *which structure is
necessary for what*:

- For the intact SRP-64 features and under each destruction (temporal-bin order,
  temporal resolution, electrode identity), N3 reports both the **held-out
  condition-accuracy drop** and the **subject-to-condition variance ratio ρ**.
- ρ ties directly to the dissertation's reported ρ ≈ 7.2 subject-covariance
  entanglement and to the *IEEE SPL* submission on the same phenomenon.

*Why it matters:* an interpretability *necessity map* — evidence that the
inspectable factors are causally load-bearing for condition decoding, and a
quantitative account of how subject nuisance is distributed across them.

## Files

| File | Purpose |
|---|---|
| `novel_pipeline.py` | Estimators (fold-local PCA, ICC, transfer function, robustness, variance ratio, necessity map), reusing the committed confirmatory operators |
| `run_novel_experiments.py` | Runs N1–N3 on authorized SHAPE data → `novel_results.json` |
| `verify_novel.py` | Synthetic verification of every estimator (CI-safe, no SHAPE data) |

## Running

```text
# machinery (no data, CI-safe)
python experiments/novel/verify_novel.py

# real data (restricted; not shipped)
python experiments/novel/run_novel_experiments.py \
    --data3 ./batch_data --granularity three_condition --out novel_results.json
```

Add `--fast` for a quicker pass (fewer dimensions and projection seeds).
`--granularity four_category` runs the four-category cohort.

## Scope

`verify_novel.py` establishes estimator behavior on synthetic data: fold-local
PCA does not leak, ICC recovers reliability, the transfer function is monotonic,
and the variance decomposition recovers a planted subject-nuisance structure. It
does **not** establish the empirical SHAPE results, which require running the
committed pipeline on the authorized restricted dataset. These experiments are
positioned as **exploratory, hypothesis-strengthening** analyses; like the rest
of the repository, passing tests verify code behavior, not scientific validity.
