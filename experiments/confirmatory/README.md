# Confirmatory validation pipeline

This directory contains the canonical, committed pipeline that generates the
**corrected confirmatory** classification results reported in the dissertation
(`dissertation/chgraph.tex`, abstract in `dissertation/main.tex`). It closes
the reproducibility gap noted in earlier revisions, where the headline numbers
were documented but had no generating code.

## Protocol (exactly what the dissertation describes)

- Five repeats of five-fold **participant-grouped** cross-validation
  (`StratifiedGroupKFold`); every participant's full condition panel stays in
  one fold.
- **Fold-local** preprocessing: `StandardScaler` fitted on training
  participants only.
- A fixed **SRP-64** projection (`SparseRandomProjection`, prespecified seed,
  shared across electrodes). It is **data-independent** — its matrix depends
  only on the seed and dimensionality, never on the observations — which is
  the correction that replaces the archived globally fitted PCA-64.
- `LogisticRegression` readout for accuracy; a separate deterministic
  `RidgeClassifier` for within-participant label permutation.
- Participant-bootstrap 95% intervals.
- Mechanistic destruction controls (temporal-bin shuffle, temporal collapse,
  electrode shuffle), three reservoir seeds (7, 42, 99), and temporal-window
  ablations.
- Three-condition and four-category granularities.

The reservoir and BSC₆ encoder are imported from the already-validated
`chapter5Experiments/experiment_zero.py`, so this pipeline uses the exact same
operators the rest of the repository is verified against.

## Files

| File | Purpose |
|---|---|
| `confirmatory_pipeline.py` | Library: SRP-64, fold-local CV, bootstrap CI, permutation, controls, windows |
| `run_confirmatory_validation.py` | Real-data runner → writes `confirmatory_results.json` |
| `verify_confirmatory.py` | Synthetic-data verification of the machinery (**no SHAPE data needed**) |
| `check_against_manifest.py` | Compares produced results to `results_manifest.json`; schema-only mode in CI |

## Running

**Verify the machinery (no data, CI-safe):**
```
python verify_confirmatory.py
python check_against_manifest.py            # schema-only: manifest provenance
```

**Regenerate the headline numbers (requires authorized SHAPE data):**
```
python run_confirmatory_validation.py --data3 ./batch_data --data4 ./categories \
    --out confirmatory_results.json
python check_against_manifest.py confirmatory_results.json --tol 1.5
```

## Important scope note

`verify_confirmatory.py` proves **code behavior and the preprocessing
boundary** (SRP data-independence, fold-local fitting, participant grouping,
determinism, output schema). It does **not** establish the scientific findings.
The SHAPE dataset is restricted (https://lab-can.com/shape/) and is not shipped
in this repository; reproducing the exact 60.6% / 53.2% headline values
requires feeding the authorized data to the runner. This separation — verified
machinery here, numbers regenerated from authorized data — is the honest
reproducibility contract for a restricted-data study.
