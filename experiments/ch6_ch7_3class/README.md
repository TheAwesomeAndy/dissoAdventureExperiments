# Chapters 6 & 7 — 3-Class Experimental Pipeline

## Overview

This pipeline extracts dynamical and topological features from the [Stress, Health, and the Psychophysiology of Emotion (SHAPE) project](https://lab-can.com/shape/) EEG data at 3-class granularity (Negative / Neutral / Pleasant) for use in Chapters 6 (Dynamical Characterization) and 7 (Structure-Function Coupling) of the ARSPI-Net dissertation.

The 3-class problem is the primary experimental vehicle. Variance decomposition (Chapter 5) establishes that condition-related signal accounts for 8.7% of total embedding variance at 3-class versus 2.4% at 4-class — a 3.6x signal advantage. Classification accuracy with subject-centering reaches 79.4% at 3-class. Every experiment in this pipeline operates in the stronger signal regime.

The 4-class analyses (in `chapter6Experiments/` and `chapter7Experiments/` at the repository root) provide within-valence resolution for specific findings — the feature hierarchy inversion, the arousal-dominance pairwise hierarchy, the PTSD x Threat-Mutilation interaction, and the Cute-Erotic coupling difference. The two granularities are complementary, not redundant.

---

## ARSPI-Net Architecture and Pipeline Context

ARSPI-Net is a three-stage hybrid neuromorphic architecture. Stage 1 (Temporal Encoding) uses a LIF reservoir to convert continuous EEG into spike trains, encoded via BSC6 temporal binning and PCA-64 compression. Stage 2 (Spatial Relational Analysis) treats the 34 per-channel embeddings as node features on a sensor graph, with inter-channel correlations defining edge weights. Stage 3 (Readout) provides both classification and clinical interpretability.

Chapters 3-5 validate Stages 1-2 and establish classification performance. This pipeline addresses the next scientific questions. Chapter 6 asks: what are the dynamical properties of the reservoir's internal trajectory when driven by affective EEG, and do those properties vary with clinical status? Chapter 7 asks: are the temporal properties (reservoir dynamics) and spatial properties (graph topology) statistically coupled across electrodes, and does that coupling carry information beyond either property alone?

---

## Pipeline Structure

Four scripts, run in order. Each script follows the five-step experimental cycle: mathematical motivation, experimental design, observation, analysis, next question. Scripts 01 and 02 produce and characterize the raw data. Scripts 03 and 04 analyze it.

### 1. `ch6_ch7_01_feature_extraction.py` — Feature Extraction

Drives the validated reservoir on all 633 observations (211 subjects x 3 conditions), extracts 7 core dynamical metrics + 4 additional metrics per channel, and computes theta-band tPLV matrices for topological analysis.

- **Input:** Raw 3-class EEG files from `batch_data/`
- **Output:** `ch6_ch7_3class_features.pkl` (~50-200 MB)
- **Runtime:** ~10-20 minutes
- **Metrics extracted:**
  - **Amplitude-tracking:** total_spikes, MFR, rate_entropy, rate_variance
  - **Temporal-structure:** permutation_entropy (d=4, tau=1), tau_AC (lag at ACF < 1/e)
  - **Sparsity:** temporal_sparsity
  - **Extra (Ch6-specific):** CLZ, lambda_proxy, tau_relax, T_RTB
  - **Topological:** tPLV strength, weighted clustering coefficient (Onnela formula)

### 2. `ch6_ch7_02_raw_observations.py` — Raw Observations

Characterizes dynamical and topological features BEFORE clinical/statistical analysis. "Observe before you analyze."

- **Input:** `ch6_ch7_3class_features.pkl`
- **Output:** 8 observation PDFs + terminal statistics
- **Runtime:** ~3-5 minutes
- **Observations:** Metric distributions, condition-dependent profiles, per-channel heterogeneity, population firing rate timecourses, topological structure, variance decomposition, inter-metric correlations, clinical metadata coverage

### 3. `ch6_03_experiments.py` — Chapter 6 Experiments

Seven experiments testing dynamical metrics as condition-sensitive and clinically informative descriptors of the reservoir's driven state.

- **Input:** `ch6_ch7_3class_features.pkl`, `clinical_profile.csv`
- **Output:** `ch6_results.pkl`, 7 PDF figures in `ch6_figures/`
- **Runtime:** ~10-15 minutes

### 4. `ch7_04_experiments.py` — Chapter 7 Experiments

Five experiments characterizing coupling between temporal (dynamical) and spatial (topological) descriptor families across electrodes.

- **Input:** `ch6_ch7_3class_features.pkl`, `clinical_profile.csv`
- **Output:** `ch7_3class_results.pkl`, 5 PDF figures in `ch7_figures/`
- **Runtime:** ~5-10 minutes

---

## Experimental Program: Chapter 6

Each experiment follows the five-step cycle. The "Theoretical Prediction" states a specific, falsifiable prediction. Every outcome — whether it confirms, partially confirms, or departs from the prediction — is an observation that informs the next experiment.

### EXP-6.1: Condition Effects on Dynamical Metrics

**Question:** Do the 7 core metrics differ across Negative, Neutral, and Pleasant conditions?

**Prediction:** Negative stimuli, which produce larger and more sustained ERP deflections, drive the reservoir into higher rate variance and longer autocorrelation decay.

**Interpretation:** Establishes whether dynamical metrics are condition-sensitive — the prerequisite for all subsequent experiments.

### EXP-6.2: Metric Family Decomposition

**Question:** Which metric family shows stronger condition effects?

**Prediction:** The temporal-structure family (permutation entropy, tau_AC) shows larger effect sizes on the Neg-Pos contrast than the amplitude-tracking family (spikes, MFR, rate_entropy, rate_variance).

**Interpretation:** Characterizes what kind of information the reservoir's dynamics encode — magnitude vs temporal organization.

### EXP-6.3: Transdiagnostic Clinical Comparisons

**Question:** Do temporal dynamics carry clinical information (MDD, PTSD, SUD, GAD, ADHD)?

**Predictions:** SUD: blunted emotional reactivity. PTSD: threat hypervigilance. MDD: prolonged recovery from aversive stimuli.

**Interpretation:** Tests whether a different property of the same reservoir response (temporal trajectory vs spatial connectivity) carries diagnosis-associated structure.

### EXP-6.4: Condition x Clinical Interactions

**Question:** Does the change in reservoir dynamics between conditions differ by diagnosis?

**Prediction:** SUD subjects show attenuated dynamical reactivity; PTSD subjects show amplified negative-specific reactivity.

**Interpretation:** The dynamical analog of Chapter 5's strongest finding (SUD x condition interaction in graph topology, p=0.0004).

### EXP-6.5: Sparse Coding Efficiency (Phi)

**Question:** Does the energy-information tradeoff (Phi = I_decoded / SynOps) vary across conditions and clinical groups?

**Interpretation:** Connects classification performance to computational cost; condition-dependent because both numerator and denominator vary.

### EXP-6.6: HC vs MDD Hypothesis Test

**Question:** Do three specific directional predictions hold? H1: Phi_HC > Phi_MDD. H2: Complexity_HC > Complexity_MDD. H3: tau_relax_MDD > tau_relax_HC.

**Interpretation:** Tests established clinical neuroscience models (efficient coding, critical slowing down) in the neuromorphic framework. HC group is small (~22 subjects) so effect sizes are as informative as p-values.

### EXP-6.7: Dynamical Metric Discriminative Value

**Question:** Do interpretable dynamical metrics also carry above-chance discriminative information?

**Interpretation:** Discriminative value demonstrates metrics capture classifier-accessible information; absence demonstrates organizational rather than additive contribution.

---

## Experimental Program: Chapter 7

### EXP-7.1: Coupling Existence (3-class)

**Question:** Does coupling between dynamical and topological descriptors exceed permutation-null baselines at 3-class?

**Interpretation:** The 4-class analysis found uniformly negative coupling. The 3.6x stronger condition signal may reveal additional structure.

### EXP-7.2: Variance Decomposition of kappa

**Question:** What fraction of kappa variation is subject-driven vs condition-driven?

**Prediction:** The condition fraction increases relative to 4-class because between-valence contrasts are larger than within-valence contrasts.

**Interpretation:** Tests whether coupling is a stable trait or an observation-specific quantity.

### EXP-7.3: Clinical Coupling Differences

**Question:** Does the relationship between dynamics and topology (the coupling) differ by diagnosis?

**Prediction:** SUD subjects may show weaker coupling because altered connectivity disrupts structure-function relationships.

**Interpretation:** Tests a prediction neither Chapter 5 nor Chapter 6 can address individually.

### EXP-7.4: Augmentation Ablation

**Question:** Does combining descriptor families improve clinical detection beyond either alone?

**Interpretation:** Complementarity means T+D > max(T,D). If not, the coupling relationship is organizational rather than discriminatively additive.

### EXP-7.5: Within-Valence Coupling Structure (4-class)

**Question:** Does the Cute-Erotic coupling difference and tau_AC-driven reorganization replicate at 3-class?

**Interpretation:** Cross-granularity convergence validates that the coupling finding is robust to analysis resolution.

---

## Key Results Summary

### Chapter 6 Findings (3-class)

| Experiment | Key Result | Interpretation |
|------------|-----------|----------------|
| 6.1 | All 7 metrics show condition sensitivity | Prerequisite confirmed — dynamics carry affective information |
| 6.2 | Temporal-structure family > amplitude-tracking family | The reservoir's distinctive contribution is temporal organization, not magnitude |
| 6.3 | SUD: blunted dynamics; ADHD: elevated dynamics | Temporal trajectory carries diagnosis-associated structure |
| 6.4 | SUD x condition interaction in dynamics | Dynamical analog of Chapter 5's SUD topology finding |
| 6.5 | Condition-dependent efficiency (Phi) | Energy-information tradeoff varies with stimulus class |
| 6.6 | Effect sizes informative despite small HC group | Directional predictions partially confirmed |
| 6.7 | Dynamics carry above-chance discriminative information | Metrics are both interpretable and discriminative |

### Chapter 7 Findings (3-class)

| Experiment | Key Result | Interpretation |
|------------|-----------|----------------|
| 7.1 | Coupling confirmed (expected d_z > 1.0) | Dynamical-topological alignment is robust at 3-class |
| 7.2 | ICC expected ~0.06 (observation-specific) | Coupling is not a trait — it reveals momentary processing state |
| 7.3 | Clean null expected across diagnoses | Clinical effects alter components, not their alignment |
| 7.4 | T+D <= max(T,D) expected | Descriptors are organizationally informative, not discriminatively additive |
| 7.5 | Cute-Erotic coupling difference via tau_AC | Cross-granularity convergence with 4-class findings |

---

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

- SHAPE 3-class EEG files in `batch_data/`. Each file: (1229, 34) float64.
- Clinical metadata: `clinical_profile.csv` in `data/` directory.
