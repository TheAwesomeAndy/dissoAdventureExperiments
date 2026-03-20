# Layer Ablation — Keystone Experiment

## Overview
This is the dissertation's keystone experiment. It directly tests whether ARSPI-Net's three response layers are redundant, complementary, or if one subsumes the others.

## Dissertation Context
The central thesis of the dissertation is:

> "ARSPI-Net reveals three operationally distinct response layers in affective EEG — discriminative representation, dynamical response, and topology/coupling — each sensitive to different aspects of the signal."

That thesis is a claim. This experiment is the test. Without this ablation, the three-layer decomposition is narrative consistency, not empirical evidence. This experiment applies Methodology Rules 1 (horizontal before vertical) and 2 (claims require direct tests) from the dissertation's scientific voice directive.

## Script

### `layer_ablation.py`
**Purpose:** Systematically ablate and combine feature blocks from Chapters 5, 6, and 7 to measure their individual and combined discriminative value.

**Feature Blocks:**
| Block | Source | Shape | Description |
|-------|--------|-------|-------------|
| E | Chapter 5 | (633, 2176) | BSC6-PCA64 embeddings (34 ch x 64 dims) |
| D | Chapter 6 | (633, 238) | 7 dynamical metrics (34 ch x 7 metrics) |
| T | Chapter 5/7 | (633, 68) | tPLV strength + clustering (34 ch x 2) |
| C | Chapter 7 | (633, 3) | Coupling scalar kappa + mean signed coupling |
| BandPower | Baseline | (633, 170) | 5-band spectral power (34 ch x 5 bands) |

**Methodology:**
- Primary readout: L2-regularized logistic regression (Rule 5: linear readouts for content comparison)
- Sensitivity check: RBF-SVM (appendix)
- Subject-stratified 10-fold CV
- Tests all single blocks, all pairwise combinations, and the full concatenation

**Input:** `ch6_ch7_3class_features.pkl`, `clinical_profile.csv`
**Output:** Ablation matrix (A1-A9), comparison tables, PDF figures

## Dependencies
numpy, scikit-learn, pandas, matplotlib, pickle

## Data Requirements
Requires pre-extracted features from the 3-class pipeline (`experiments/ch6_ch7_3class/`).
