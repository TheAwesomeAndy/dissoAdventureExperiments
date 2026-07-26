# Novel experiments: what the inspectable chain buys you

These experiments target the hardest reviewer question for an
interpretability-first neuromorphic model that is deliberately **not** an
accuracy leader: *if the raw ERP classifies affect more accurately, why use the
fixed, inspectable measurement chain?* Each experiment answers with a
quantitative property that a black-box model cannot expose, computed on the same
verified operators as the confirmatory stream and reported at honest scope.

## The three contributions

### N1 — Leak-free compression is free
The confirmatory pipeline replaced a globally fitted PCA (which leaks held-out
participants) with a fixed, data-independent SRP-64. N1 shows this is not a
compromise:

- **SRP-64 ≈ fold-local PCA-64.** Against a *fold-local* (leak-free) PCA fitted
  only on training participants, the paired participant-bootstrap difference
  includes zero at every tested dimension. A projection that never looks at the
  data matches one that is optimally fitted per fold.
- **Johnson–Lindenstrauss dimensionality law.** Balanced accuracy across
  compression dimensions {16, 32, 64, 128} traces the expected geometry-
  preservation curve.
- **Seed stability.** The reported SRP-64 is not a lucky draw: accuracy across
  many random projection seeds has a small spread.

*Why it matters:* leak-free-by-construction compression, provably independent of
the evaluation set, at no accuracy cost — a clean, reusable methods result for
grouped clinical-EEG pipelines where transductive leakage is the dominant
failure mode.

### N2 — The reservoir is a calibrated instrument
The dissertation frames the fixed reservoir as a measurement instrument. N2
quantifies that:

- **Cross-seed reliability.** BSC6/SRP-64 features have a median intraclass
  correlation (ICC(2,1)) across independent reservoir initializations — a
  reproducibility statement a trained network cannot make about its weights.
- **Transfer function.** Total spike output is a monotonic function of input
  amplitude with a quantified dynamic range: a calibratable dose–response.
- **Graceful degradation.** Balanced accuracy under channel dropout and additive
  noise, compared against the raw-ERP reference, characterizes robustness of the
  fixed transform.

*Why it matters:* reframes "fixed weights" from a limitation into an instrument
guarantee — reproducibility, calibration, and known failure behavior.

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
