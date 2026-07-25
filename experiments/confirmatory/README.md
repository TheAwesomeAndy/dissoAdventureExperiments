# Confirmatory validation pipeline

This directory contains the canonical pipeline for the corrected confirmatory
classification results reported in `dissertation/chgraph.tex` and the abstract
of `dissertation/main.tex`.

## Protocol

- Five repeats of five-fold participant-grouped cross-validation with
  `StratifiedGroupKFold`. A participant's complete condition panel remains in
  one fold.
- Raw-epoch reconstruction removes samples 0–204, anti-alias decimates samples
  205–1228 by four to 256 samples, and z-scores each epoch and channel.
- Subject 127 is excluded from both granularities. Subject 195 is additionally
  excluded from the four-category stream.
- `StandardScaler` and the linear readout are fitted on training participants
  only within each fold.
- BSC6 is compressed by a fixed, data-independent SRP-64 projection generated
  from a prespecified seed and shared across electrodes. The projection does not
  estimate a basis from the observations.
- `LogisticRegression` produces the point estimates. A deterministic
  `RidgeClassifier` is used for within-participant label permutations.
- Participant-cluster bootstrap intervals pool the out-of-fold predictions from
  all five repeats. Representation differences use paired participant-cluster
  bootstrap intervals.
- Mechanistic controls destroy temporal-bin order, temporal resolution, or
  electrode identity while preserving the remaining representation structure.
- Reservoir-seed and temporal-window analyses characterize operating-point
  sensitivity.

The reservoir and BSC6 encoder are imported from
`chapter5Experiments/experiment_zero.py`; the corrected runner replaces the
legacy epoch slicing and globally fitted PCA stream.

## Files

| File | Purpose |
|---|---|
| `confirmatory_pipeline.py` | SRP-64, grouped cross-validation, pooled OOF predictions, bootstrap estimators, permutation test, and controls |
| `run_confirmatory_validation.py` | Authorized-data reconstruction and `confirmatory_results.json` generation |
| `verify_confirmatory.py` | Synthetic verification of code behavior and preprocessing boundaries |
| `check_against_manifest.py` | Schema validation in CI and complete confirmatory-result comparison after an authorized run |

## Running

Data-independent verification:

```text
python experiments/confirmatory/verify_confirmatory.py
python experiments/confirmatory/check_against_manifest.py
```

Authorized-data reconstruction:

```text
python experiments/confirmatory/run_confirmatory_validation.py \
    --data3 ./batch_data \
    --data4 ./categories \
    --out confirmatory_results.json

python experiments/confirmatory/check_against_manifest.py \
    confirmatory_results.json --tol 1.5
```

## Scope

The synthetic verification establishes implementation behavior, including the
participant-group boundary, data-independent projection, repeated-prediction
pooling, bootstrap pairing, destruction operators, and raw-epoch preprocessing.
It does not establish the empirical SHAPE findings. Exact agreement with the
reported 60.6% and 53.2% estimates must be checked by running the committed
pipeline on the authorized restricted dataset and comparing the generated JSON
against `results_manifest.json`.
