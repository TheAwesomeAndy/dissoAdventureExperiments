# Chapters 6 & 7 — 3-Class Experimental Pipeline

## Overview
This pipeline extracts dynamical and topological features from the [SHAPE EEG dataset](https://lab-can.com/shape/) at 3-class granularity (Negative/Neutral/Pleasant) for Chapters 6 (Dynamical Characterization) and 7 (Structure-Function Coupling).

## Dissertation Context
The 3-class problem is the primary experimental vehicle for Chapters 6 and 7. Variance decomposition (Chapter 5) establishes that condition-related signal accounts for 8.7% of total embedding variance at 3-class versus 2.4% at 4-class — a 3.6x signal advantage. Classification accuracy with subject-centering reaches 79.4% at 3-class. Every experiment in this pipeline operates in the stronger signal regime.

The 4-class analyses (in `chapter6Experiments/` and `chapter7Experiments/` at the repository root) provide within-valence resolution for specific findings. The two granularities are complementary, not redundant.

## Architecture Context
ARSPI-Net is a three-stage hybrid neuromorphic architecture:
- **Stage 1 (Temporal Encoding):** LIF reservoir converts continuous EEG into spike trains, encoded via BSC6 binning and PCA-64 compression
- **Stage 2 (Spatial Relational Analysis):** 34 per-channel embeddings as node features on a sensor graph with tPLV-derived edge weights
- **Stage 3 (Readout):** Classification and clinical interpretability

## Scripts (run in order)

### 1. `ch6_ch7_01_feature_extraction.py`
**Purpose:** Drive the validated LIF reservoir on 211 subjects x 3 conditions x 34 channels = 21,522 reservoir runs. Extract 7 core dynamical metrics + 4 extra metrics per channel, plus theta-band tPLV matrices.
- **Input:** Raw 3-class EEG files from `batch_data/`
- **Output:** `ch6_ch7_3class_features.pkl` (~50-200 MB)
- **Metrics extracted:**
  - Amplitude-tracking: total_spikes, MFR, rate_entropy, rate_variance
  - Temporal-structure: permutation_entropy, tau_AC
  - Sparsity: temporal_sparsity
  - Extra: CLZ, lambda_proxy, tau_relax, T_RTB
  - Topological: tPLV strength, weighted clustering coefficient
- **Runtime:** ~10-20 minutes

### 2. `ch6_ch7_02_raw_observations.py`
**Purpose:** Characterize dynamical and topological features BEFORE clinical/statistical analysis. "Observe before you analyze."
- **Input:** `ch6_ch7_3class_features.pkl`
- **Output:** 8 observation PDFs + terminal statistics
- **Observations:** Metric distributions, condition-dependent profiles, per-channel heterogeneity, population firing rate timecourses, topological structure, variance decomposition, inter-metric correlations, clinical metadata coverage
- **Runtime:** ~3-5 minutes

### 3. `ch6_03_experiments.py`
**Purpose:** Seven experiments testing dynamical metrics as condition-sensitive and clinically informative descriptors.
- **Input:** `ch6_ch7_3class_features.pkl`, `clinical_profile.csv`
- **Output:** `ch6_results.pkl`, 7 PDF figures in `ch6_figures/`
- **Experiments:**
  - EXP-6.1: Condition effects on dynamical metrics
  - EXP-6.2: Metric family decomposition (amplitude-tracking vs temporal-structure)
  - EXP-6.3: Transdiagnostic clinical comparisons (MDD, PTSD, SUD, GAD, ADHD)
  - EXP-6.4: Condition x clinical interactions
  - EXP-6.5: Sparse coding efficiency (Phi)
  - EXP-6.6: HC vs MDD hypothesis test (3 directional predictions)
  - EXP-6.7: Dynamical metric discriminative value (3-class + binary clinical)

### 4. `ch7_04_experiments.py`
**Purpose:** Five experiments characterizing coupling between temporal (dynamical) and spatial (topological) descriptor families.
- **Input:** `ch6_ch7_3class_features.pkl`, `clinical_profile.csv`
- **Output:** `ch7_3class_results.pkl`, 5 PDF figures in `ch7_figures/`
- **Experiments:**
  - EXP-7.1: Coupling existence at 3-class (kappa vs permutation null)
  - EXP-7.2: Variance decomposition of kappa (subject/condition/residual)
  - EXP-7.3: Clinical coupling differences by diagnosis
  - EXP-7.4: Augmentation ablation (T-only vs D-only vs T+D)
  - EXP-7.5: Within-valence coupling structure (uses 4-class results)

## Verification Results

<!-- Last run: 2026-03-20, Result: 28/28 PASS -->

```bash
python experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py
```

**Result: 28/28 PASS.** Infrastructure tests on synthetic data:

- **Syntax validation (4 tests):** All 4 scripts parse without errors
- **LIF Reservoir (8 tests):** Module imports, init_reservoir returns weights, W_in (64,1) and W_rec (64,64) shapes, spike shape (64,256) and membrane shape (64,256), binary spikes, non-silent
- **Dynamical metrics (8 tests):** total_spikes, MFR, population rate shape, rate entropy, rate variance, tau_ac, permutation entropy in (0,1], temporal sparsity in [0,1]
- **Topological metrics (5 tests):** tPLV matrix shape (34,34), values in [0,1], symmetric, node strength shape (34,), clustering coefficient computable
- **Configuration (3 tests):** N_RES=256, BETA=0.05, THRESHOLD=0.5

Note: This script uses functional-style reservoir (init_reservoir/run_reservoir) rather than a class, with output shape (n_res, T) transposed from the chapter4/5 convention (T, n_res).

## Dependencies
numpy, scipy, scikit-learn, matplotlib, pandas, pickle

## Data Requirements
SHAPE Community 3-class EEG files in `batch_data/`. Each file: (1229, 34) float64.
Clinical metadata: `clinical_profile.csv` in `data/` directory.
