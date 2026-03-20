# Chapter 5 — 4-Class Classification Extension

## Overview
These scripts extend the Chapter 5 clinical EEG classification pipeline from 3-class (Negative/Neutral/Pleasant) to 4-class (Threat/Mutilation/Cute/Erotic). This tests whether within-valence subcategory pairs carry distinct spatiotemporal signatures in the ARSPI-Net embedding space.

## Dissertation Context
Chapter 5 originally classified 3 broad affective conditions. The 4-class extension is critical because it reveals whether the 3-class collapse lost discriminative structure. If Threat and Mutilation (both negative) produce different reservoir embeddings, the temporal coding captures finer affective distinctions than valence alone. This finding feeds directly into Chapter 6's within-valence dissociation analysis and Chapter 7's category-conditioned coupling structure.

## Scripts (run in order)

### 1. `ch5_4class_01_feature_extraction.py`
**Purpose:** Extract BSC6-PCA64 and band-power features from all 4 IAPS subcategories.
- **Input:** Raw EEG from `categoriesbatch{1-4}/` (844 observations: 211 subjects x 4 categories)
- **Output:** `shape_features_4class.pkl` containing downsampled EEG, BSC6 features, PCA-64 embeddings, band-power features, MFR vectors
- **Runtime:** ~20-40 minutes
- **Key parameters:** N_res=256, beta=0.05, threshold=0.5, PCA components=64, BSC bins=6

### 2. `ch5_4class_02_raw_observations.py`
**Purpose:** Characterize the 4-class data at every pipeline stage BEFORE classification modeling.
- **Input:** `shape_features_4class.pkl`, `clinical_profile.csv`
- **Output:** 7 PDF figures + 7 .npz data files in `ch5_4class_raw_data/`
- **Observations:** Grand-average ERPs, spike statistics, embedding structure, inter-channel connectivity, graph variability, clinical metadata, within-vs-between valence distances
- **Runtime:** ~3-5 minutes

### 3. `ch5_4class_03_classification_full.py`
**Purpose:** Complete 4-class experimental program with 11 experiments spanning classification and clinical interpretability.
- **Input:** `shape_features_4class.pkl`, `clinical_profile.csv`
- **Output:** `ch5_4class_results.pkl`, PDF figures in `ch5_4class_figures/`
- **Classification experiments (EXP 1-6):** Baseline comparison, GNN architecture comparison, confusion matrices, pairwise contrasts, variance decomposition, per-channel discrimination
- **Clinical experiments (EXP 7-11):** Channel-level biomarkers, condition x clinical interactions, edge-level biomarkers, comorbidity burden, within-valence x clinical interactions
- **Validation:** Subject-stratified 10-fold CV, PCA fitted per fold on training data only

## Verification Results

<!-- Last run: 2026-03-20, Result: 25/25 PASS -->

```bash
python experiments/ch5_4class/verify_ch5_4class.py
```

**Result: 25/25 PASS.** Infrastructure tests on synthetic data without requiring SHAPE EEG:

- **Syntax validation (3 tests):** All 3 scripts parse without errors
- **LIF Reservoir (8 tests):** Module imports, instantiation, W_in (64,1) and W_rec (64,64) shapes, forward pass shapes (256,64), binary spikes, finite membrane, non-negative membrane, non-silent spikes
- **BSC extraction (3 tests):** Produces 384-dim vector (64 neurons x 6 bins), non-negative, nonzero entries
- **Band power (3 tests):** Produces 5-dim vector per channel, non-negative, nonzero entries
- **Configuration consistency (7 tests):** N_RES=256, BETA=0.05, THRESHOLD=0.5, SEED=42, BSC_N_BINS=6, PCA_N_COMPONENTS=64, 4 categories defined

Note: One numpy compatibility fix was applied — `np.trapz` replaced with `np.trapezoid` for numpy >= 2.0.

## Dependencies
numpy, scipy, scikit-learn, matplotlib, pandas, pickle

## Data Requirements
SHAPE Community EEG subcategory files in `categoriesbatch{1-4}/` directories. Each file: (1229, 34) float64, baseline-corrected, trial-averaged microvolts.
