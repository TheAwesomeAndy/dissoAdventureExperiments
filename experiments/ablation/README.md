# Layer Ablation — Keystone Experiment

## Overview

This is the dissertation's keystone experiment. It directly tests whether ARSPI-Net's three response layers are redundant, complementary, or if one subsumes the others.

---

## Dissertation Context

The central thesis of the dissertation is:

> "ARSPI-Net reveals three operationally distinct response layers in affective EEG — discriminative representation, dynamical response, and topology/coupling — each sensitive to different aspects of the signal."

That thesis is a claim. This experiment is the test. Without this ablation, the three-layer decomposition is narrative consistency, not empirical evidence. This experiment applies Methodology Rules 1 (horizontal before vertical) and 2 (claims require direct tests) from the dissertation's scientific voice directive.

---

## Script: `layer_ablation.py`

**Purpose:** Systematically ablate and combine feature blocks from Chapters 5, 6, and 7 to measure their individual and combined discriminative value.

### Feature Blocks

| Block | Source | Shape | Description |
|-------|--------|-------|-------------|
| E | Chapter 5 | (633, 2176) | BSC6-PCA64 embeddings (34 ch x 64 dims) |
| D | Chapter 6 | (633, 238) | 7 dynamical metrics (34 ch x 7 metrics) |
| T | Chapter 5/7 | (633, 68) | tPLV strength + clustering (34 ch x 2) |
| C | Chapter 7 | (633, 3) | Coupling scalar kappa + mean signed coupling |
| BandPower | Baseline | (633, 170) | 5-band spectral power (34 ch x 5 bands) |

### Methodology

- **Primary readout:** L2-regularized logistic regression (Rule 5: linear readouts for content comparison)
- **Sensitivity check:** RBF-SVM (appendix)
- **Validation:** Subject-stratified 10-fold CV
- **Scope:** Tests all single blocks, all pairwise combinations, and the full concatenation

### Ablation Matrix

**Emotion discrimination (A0-A9):** 3-class affective classification (Negative / Neutral / Pleasant)

| Condition | Features | Purpose |
|-----------|----------|---------|
| A0 | BandPower | Conventional baseline |
| A1 | E only | Embedding-layer contribution |
| A2 | D only | Dynamics-layer contribution |
| A3 | T only | Topology-layer contribution |
| A4 | C only | Coupling-layer contribution |
| A5 | E + D | Embedding + dynamics |
| A6 | E + T | Embedding + topology |
| A7 | D + T | Dynamics + topology |
| A8 | E + D + T | All three primary layers |
| A9 | E + D + T + C | Full ARSPI-Net feature set |

**Clinical detection (C1-C6):** Binary classification for each diagnosis

| Condition | Task | Purpose |
|-----------|------|---------|
| C1 | MDD detection | Most prevalent diagnosis |
| C2 | SUD detection | Strongest topological phenotype |
| C3 | PTSD detection | Threat-specific processing |
| C4 | GAD detection | Anxiety-related organization |
| C5 | ADHD detection | Dynamical hyperactivation signature |
| C6 | Medication effects | Pharmacological confound control |

---

## Expected Results and Interpretation

### What Each Outcome Pattern Means

| Pattern | Interpretation |
|---------|---------------|
| A8 > A1, A2, A3 individually | Layers carry complementary information (thesis supported) |
| A8 ~ max(A1, A2, A3) | One layer subsumes the others (thesis weakened) |
| A9 > A8 | Coupling adds discriminative value beyond individual layers |
| A1 >> A2, A3 | Embedding dominates; dynamics and topology are secondary |
| Different layers dominate for different C tasks | Layers are sensitive to different clinical dimensions (thesis supported) |

### Critical Predictions from Prior Chapters

Based on Chapter 7 Experiment E findings (T+D <= max(T,D) for all diagnoses):

- **Dynamics (D) should dominate for ADHD** (AUC 0.622 in Ch7 Exp E, highest single-family result)
- **Topology (T) should dominate for GAD** (AUC 0.581 in Ch7 Exp E)
- **Neither D nor T should detect SUD** (both at chance in Ch7 Exp E) — but embedding (E) may succeed since it captures different information
- **E is expected to be the strongest individual block** for emotion discrimination (it contains the full 2176-dim BSC6-PCA64 representation that achieved ~79% at 3-class)

### Honest Assessment Framework

The key question is NOT whether combining helps (Chapter 7 already showed it often doesn't at the linear-readout level). The key question is whether the layers are *operationally distinct* — whether they are sensitive to *different* aspects of the signal:

- If E dominates emotion but D dominates ADHD and T dominates GAD, that is operational distinctness
- If E dominates everything, that is subsumption — the other layers are interpretive tools, not independent measurement axes
- Both outcomes are scientifically informative; only the first supports the central thesis

---

## Verification Results

<!-- Last run: 2026-03-20, Result: 23/23 PASS -->

```bash
python experiments/ablation/verify_ablation.py
```

**Result: 23/23 PASS.** Infrastructure tests on synthetic data:

- **Syntax validation (1 test):** layer_ablation.py parses without errors
- **Dependency imports (9 tests):** numpy, sklearn (linear_model, svm, preprocessing, metrics, model_selection), scipy.stats, pandas, matplotlib — all available
- **Coupling computation (3 tests):** 7x2 Spearman coupling matrix computed correctly, values in [-1,1], kappa scalar in [0,1]
- **Classification pipeline (3 tests):** StratifiedGroupKFold CV runs without error, returns accuracy (0.667 on synthetic data), above chance (>0.33)
- **Feature dimensions (7 tests):** E=34x64=2176, D=34x7=238, T=34x2=68, C=3, BP=34x5=170; 10 ablation conditions (A0-A9), 6 clinical conditions (C1-C6)

## Dependencies

numpy, scikit-learn, pandas, matplotlib, pickle

## Data Requirements

Requires two pre-extracted pickle files that are **not included** in the repository (too large for git):

1. `shape_features_211.pkl` — Chapter 5 baseline features (generated by `chapter5Experiments/run_chapter5_experiments.py`)
2. `ch6_ch7_3class_features.pkl` — 3-class dynamical + topological features (generated by `experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py`)

**To generate these files**, run the feature extraction pipeline first:

```bash
# Step 1: Generate Chapter 5 features
python chapter5Experiments/run_chapter5_experiments.py --data_dir /path/to/batch_data/

# Step 2: Generate 3-class Ch6/Ch7 features
python experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py
```

Place both pickle files in the `experiments/ablation/` directory before running `layer_ablation.py`.
