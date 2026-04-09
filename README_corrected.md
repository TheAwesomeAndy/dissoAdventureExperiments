# ARSPI-Net: A Staged Neuromorphic Signal Processing Framework for EEG

**Dissertation:** Lane, A. (2026). *ARSPI-Net: Hybrid Neuromorphic Signal Processing Architecture for Affective EEG Analysis.* PhD Dissertation, Department of Electrical and Computer Engineering.

**Defense version:** `v1.0-defense` (tag after final push)

This repository contains the complete experimental pipeline for Chapters 3–8 of the ARSPI-Net dissertation. The framework converts raw multichannel EEG into a staged neuromorphic representation whose internal layers are independently analyzable—providing discriminative classification, dynamical trajectory characterization, graph-topological phenotyping, and structure-function coupling analysis within a single architecture.

**ARSPI-Net is not an accuracy-maximizing classifier.** EEGNet (72.0%) and raw EEG with a linear readout (70.5%) both outperform the reservoir embedding (59.4%) on the three-class SHAPE task. The reservoir's contribution is a structured, compressed representation that enables the three-layer measurement framework validated across Chapters 5–7. The accuracy-interpretability tradeoff is quantified at 12.6 pp relative to EEGNet.

---

## Table of Contents

1. [Key Results Summary](#key-results-summary)
2. [Repository Structure](#repository-structure)
3. [Reproduction Map](#reproduction-map)
4. [Architecture Overview](#architecture-overview)
5. [Chapter 3: Reservoir Characterization](#chapter-3-reservoir-characterization)
6. [Chapter 4: Spike-to-Embedding Pipeline](#chapter-4-spike-to-embedding-pipeline)
7. [Chapter 5: Graph Analysis and External Baselines](#chapter-5-graph-analysis-and-external-baselines)
8. [Chapter 6: Dynamical Characterization](#chapter-6-dynamical-characterization)
9. [Chapter 7: Structure-Function Coupling](#chapter-7-structure-function-coupling)
10. [Extended Experiments](#extended-experiments)
11. [Dependencies](#dependencies)
12. [Input Data Format](#input-data-format)
13. [Data Validation and QC](#data-validation-and-quality-control)
14. [Key Parameters Reference](#key-parameters-reference)
15. [Verification Summary](#verification-summary)
16. [Adapting to Your Own EEG Data](#adapting-to-your-own-eeg-data)

---

## Key Results Summary

All classification results use 10-fold stratified cross-validation with subject-level splitting on the SHAPE Community dataset (211 subjects, 34 channels, trial-averaged ERPs).

### Complete Baseline Ranking (3-Class: Negative / Neutral / Pleasant)

| Rank | Method | Bal. Acc | Type | Script |
|---|---|---|---|---|
| 1 | **EEGNet** (Lawhern 2018) | **72.0% ± 4.9%** | Trained CNN (PyTorch) | `chapter5Experiments/eegnet_gru_lstm_baselines.py` |
| 2 | Raw EEG (8704d) + LogReg | 70.5% ± 3.0% | Linear on raw | `chapter5Experiments/sklearn_baselines.py` |
| 3 | Raw EEG PCA-200 + LogReg | 64.9% ± 5.0% | Linear on compressed | `chapter5Experiments/sklearn_baselines.py` |
| 4 | GRU (bidir, 64 hidden) | 59.9% ± 6.4% | Trained RNN (PyTorch) | `chapter5Experiments/eegnet_gru_lstm_baselines.py` |
| 5 | **Reservoir + LogReg** | **59.4% ± 3.6%** | **ARSPI-Net (no training)** | `chapter5Experiments/sklearn_baselines.py` |
| 6 | LSTM (bidir, 64 hidden) | 58.0% ± 5.5% | Trained RNN (PyTorch) | `chapter5Experiments/eegnet_gru_lstm_baselines.py` |
| 7 | BandPower + LinearSVM | 47.7% ± 5.1% | Spectral + linear | `chapter5Experiments/sklearn_baselines.py` |

**Key findings:**
- The reservoir (59.4%) matches trained recurrent models (GRU 59.9%, LSTM 58.0%) with zero trainable parameters
- Subject-centering is a general SHAPE task property: Raw EEG 70.5% → 88.4%, Reservoir 59.4% → 78.8%
- The reservoir's contribution is multi-layer decomposition, not peak accuracy

### Three-Layer Architecture

| Layer | What It Measures | Key Result |
|---|---|---|
| Embedding (Ch5) | Discriminative condition signal | 59.4% (outperforms spectral by +11.7 pp) |
| Dynamics (Ch6) | Temporal trajectory sensitivity | 7/7 metrics condition-sensitive; temporal family 2.4× > amplitude |
| Topology/Coupling (Ch5, Ch7) | Inter-channel clinical phenotypes | SUD p=0.0004; GAD p=0.032; coupling κ=0.22 |

Different clinical disorders are most detectable in different layers: dynamics for SUD/PTSD, topology for GAD, coupling for ADHD.

### Graph Propagation Regime Boundary

No graph operator (smoothing, high-pass, residual, attention-based) exceeds the non-propagated per-channel baseline on this 34-node, 20%-density electrode graph. The concatenated per-channel representation maximizes classification; message-passing monotonically suppresses inter-channel discriminative contrast.

---

## Repository Structure

```
dissoAdventureExperiments/
├── README.md                              # This file
├── validate_shape_data.py                 # QC: broad-condition SHAPE data
├── validate_subcategory_data.py           # QC: fine-grained subcategory data
├── verify_validators.py                   # Verification for validators
│
├── chapter4Experiments/                   # Ch3–4: Reservoir characterization + embedding pipeline
│   ├── run_chapter4_experiments.py        #   Ablation, FDR, coding, PCA, robustness, sensitivity
│   ├── run_chapter4_observations.py       #   Raw observation figures
│   └── verify_chapter4.py                #   Verification (31 tests)
│
├── chapter5Experiments/                   # Ch5: Graph analysis, baselines, clinical interpretability
│   ├── run_chapter5_experiments.py        #   Internal GNN comparison (legacy, see note below)
│   ├── reproduce_chapter5.py             #   Standalone reproducibility pipeline
│   ├── sklearn_baselines.py              #   All sklearn baselines (Table 5.4)
│   ├── eegnet_gru_lstm_baselines.py      #   EEGNet + GRU + LSTM (PyTorch, canonical)
│   ├── experiment_zero.py                #   Baseline disambiguation (centered vs uncentered)
│   └── verify_chapter5.py                #   Verification (32 tests)
│
├── chapter6Experiments/                   # Ch6: Dynamical characterization (4-class)
│   ├── run_chapter6_exp1_esp.py          #   Echo State Property verification
│   ├── run_chapter6_exp2_reliability.py  #   Cross-seed reliability (ICC)
│   ├── run_chapter6_exp3_surrogate.py    #   Surrogate sensitivity testing
│   ├── run_chapter6_exp3_valueadd.py     #   Value-add vs raw EEG
│   ├── run_chapter6_exp4_dissociation.py #   Within-valence dissociation
│   ├── run_chapter6_exp5_interaction.py  #   Diagnosis × category interaction
│   ├── run_chapter6_exp6_temporal.py     #   Sliding-window temporal localisation
│   ├── reproduce_chapter6.py             #   Standalone reproducibility pipeline
│   └── verify_chapter6.py                #   Verification (31 tests)
│
├── chapter7Experiments/                   # Ch7: Dynamical-topological coupling (4-class)
│   ├── run_chapter7_experiment_A.py      #   Coupling existence
│   ├── run_chapter7_experiment_B.py      #   Variance decomposition
│   ├── run_chapter7_experiment_C.py      #   Category-conditioned coupling
│   ├── run_chapter7_experiment_D.py      #   Diagnosis-associated coupling
│   ├── run_chapter7_experiment_E.py      #   Augmentation ablation
│   └── verify_chapter7.py                #   Verification (38 tests)
│
├── experiments/                           # Cross-chapter extensions (March 2026)
│   ├── ch5_4class/                       #   4-class regime comparison
│   ├── ch6_ch7_3class/                   #   3-class dynamical + coupling pipeline
│   └── ablation/                         #   Layer ablation keystone experiment
│
├── data/                                  # Clinical metadata
│   └── clinical_profile.csv              #   211-subject clinical profiles
├── docs/
│   ├── methodology_rules.md              #   12 governing methodology rules
│   ├── REPRODUCTION_MAP.md               #   Script → table/figure mapping (17 tables, 117 figures)
│   └── GITHUB_UPDATE_GUIDE.md            #   Repository management guide
├── latex/                                 # Revised LaTeX chapter sources
└── pictures/chLSMEmbeddings/             # Ch4 publication figures
```

### Important: Directory Naming Convention

The `chapter4Experiments/` directory contains scripts for both **Chapter 3** (reservoir characterization) and **Chapter 4** (embedding pipeline) of the dissertation. Chapter numbering shifted during writing; the directory name reflects the original numbering. The reproduction map in `docs/REPRODUCTION_MAP.md` provides the exact mapping from dissertation chapter/table/figure to script.

### Important: Legacy vs. Corrected Chapter 5 Framing

The original `chapter5Experiments/run_chapter5_experiments.py` contains a 7-row baseline table that presents "LSM-BSC₆-PCA64 + GAT + Spatial" as the "Full ARSPI-Net (best)" configuration. **This framing is superseded by the dissertation's corrected analysis.** The corrected Chapter 5 demonstrates that:

- No graph propagation operator exceeds the non-propagated per-channel baseline
- The concatenated per-channel embedding (59.4%) is the correct primary result
- The Propagation Operating Characteristic documents when and why message-passing reduces performance

The legacy 7-row table in `run_chapter5_experiments.py` is retained for completeness and reproducibility of the original exploration, but the dissertation's conclusions are based on the corrected analysis in `reproduce_chapter5.py`, which includes the propagation regime characterization, variance decomposition, and clinical interpretability experiments that supersede the original GAT comparison.

---

## Reproduction Map

See `docs/REPRODUCTION_MAP.md` for the complete mapping of every table (17) and figure (117) in the dissertation to the script that produces it. Summary:

| Chapter | Tables | Figures | Primary Script(s) |
|---|---|---|---|
| Ch3 (Reservoir) | 1 | 22 | `chapter4Experiments/run_chapter4_*.py` |
| Ch4 (Embedding) | 2 | 7 | `chapter4Experiments/run_chapter4_experiments.py` |
| Ch5 (Graph + Baselines) | 7 | 35 | `chapter5Experiments/reproduce_chapter5.py`, `sklearn_baselines.py`, `eegnet_gru_lstm_baselines.py` |
| Ch6 (Dynamics) | 2 | 7 | `experiments/ch6_ch7_3class/ch6_ch7_04_ch6_experiments.py` |
| Ch7 (Coupling) | 5 | 19 | `chapter7Experiments/run_chapter7_experiment_*.py`, `experiments/ch6_ch7_3class/ch6_ch7_05_ch7_experiments.py` |
| App A | 0 | 27 | `experiments/ch5_4class/*.py`, `experiments/ablation/layer_ablation.py`, `experiments/ch6_ch7_3class/*.py` |

---

## Architecture Overview

```
Raw EEG (34 channels, 1024 Hz, trial-averaged)
  → Preprocessing (baseline removal, downsample to 256 Hz, z-score)
    → LIF Reservoir (per-channel, N=256, β=0.05, θ=0.5, fixed random weights)
      → Temporal Coding (BSC₆ binned spike counts)
        → PCA-64 per channel → 2,176-dim concatenated embedding
          → Classification (LogReg / SVM — linear readouts only)
          → Dynamical Metrics (7 trajectory descriptors per channel)
          → Graph Topology (strength, efficiency, clustering on tPLV graph)
          → Structure-Function Coupling (CCA between dynamics and topology)
```

The reservoir weights are **not trained**. The only learned component is the downstream readout (logistic regression or SVM). This makes the internal dynamics fully analyzable as a driven nonlinear system.

---

## Chapter 3: Reservoir Characterization

**Directory:** `chapter4Experiments/` (see naming note above)

**Scripts:** `run_chapter4_observations.py` (6 raw observation figures), `run_chapter4_experiments.py` (6 experiments)

**Data:** Synthetic — no EEG data required. Generates 200 trials per class with identical total energy but different temporal profiles.

**Key result:** BSC₆ achieves >90% on temporal discrimination where Mean Firing Rate fails at ~50%, proving temporal coding captures structure that rate coding misses.

```bash
python chapter4Experiments/run_chapter4_experiments.py
python chapter4Experiments/run_chapter4_observations.py
```

---

## Chapter 4: Spike-to-Embedding Pipeline

**Directory:** `chapter4Experiments/`

**Key results:** BSC₆ + PCA-64 is the selected embedding (99.5% on synthetic, <2% variance across 10 random seeds, FDR 8840× above raw input).

---

## Chapter 5: Graph Analysis and External Baselines

**Directory:** `chapter5Experiments/`

### Classification Experiments

| Script | What It Produces | Status |
|---|---|---|
| `reproduce_chapter5.py` | Raw observations, variance decomposition, propagation regime, clinical interpretability | **Primary — produces the dissertation's corrected results** |
| `sklearn_baselines.py` | All sklearn baseline rows of Table 5.4 | **Primary** |
| `eegnet_gru_lstm_baselines.py` | EEGNet, GRU, LSTM (PyTorch, canonical implementations) | **Primary** |
| `experiment_zero.py` | Disambiguation: confirms 70.5% was uncentered | **Verification** |
| `run_chapter5_experiments.py` | Original 7-row GNN comparison (GAT/GCN/GraphSAGE) | **Legacy — superseded by corrected analysis** |

### Note on Baseline Implementations

**Canonical PyTorch implementations:** `eegnet_gru_lstm_baselines.py` implements EEGNet following Lawhern et al. (2018, J Neural Eng) with depthwise and separable convolutions, bidirectional GRU, and bidirectional LSTM using standard PyTorch modules. These are trained end-to-end with AdamW, cosine annealing, and early stopping. Each model runs 10 folds × 5 random seeds with majority-vote ensembling. These are standard trained baselines.

**NumPy GNN implementations (legacy):** The original `run_chapter5_experiments.py` contains from-scratch NumPy implementations of GCN, GraphSAGE, and GAT. These were used for the initial internal comparison but are **not** canonical optimized implementations. The dissertation's corrected Chapter 5 does not rely on these for its primary claims — the corrected analysis shows that no graph propagation improves over the non-propagated baseline, making the specific GNN implementation irrelevant to the conclusion.

### Running Chapter 5

```bash
# Primary analysis (requires SHAPE EEG data):
python chapter5Experiments/reproduce_chapter5.py \
    --data_dir /path/to/shape_eeg_files/ \
    --labels /path/to/Psychopathology.xlsx \
    --output_dir ./figures/ch5/

# Sklearn baselines (requires shape_features_211.pkl):
python chapter5Experiments/sklearn_baselines.py

# Deep learning baselines (requires PyTorch + shape_features_211.pkl):
python chapter5Experiments/eegnet_gru_lstm_baselines.py

# Disambiguation (requires raw EEG batch_data/):
python chapter5Experiments/experiment_zero.py --data_dir /path/to/batch_data/
```

---

## Chapter 6: Dynamical Characterization

**Directories:** `chapter6Experiments/` (4-class subcategory), `experiments/ch6_ch7_3class/` (3-class — used for the revised dissertation figures)

**Key results:** 7/7 dynamical metrics condition-sensitive; temporal-structure family 2.4× larger effects than amplitude; 0/49 clinical comparisons significant at channel-averaged level; clinical signal requires per-channel spatial resolution.

```bash
# 3-class (produces dissertation figures):
python experiments/ch6_ch7_3class/ch6_ch7_04_ch6_experiments.py

# 4-class subcategory experiments:
python chapter6Experiments/run_chapter6_exp1_esp.py \
    --category-dirs categoriesbatch1 categoriesbatch2 categoriesbatch3 categoriesbatch4
```

---

## Chapter 7: Structure-Function Coupling

**Directories:** `chapter7Experiments/` (4-class), `experiments/ch6_ch7_3class/` (3-class)

**Key results:** Coupling exists (κ=0.22, p<0.001), is observation-specific (not trait-like), is organizational rather than strongly discriminative, and does not carry independent clinical signal.

```bash
# 4-class experiments:
python chapter7Experiments/run_chapter7_experiment_A.py --analyze
python chapter7Experiments/run_chapter7_experiment_B.py
python chapter7Experiments/run_chapter7_experiment_C.py
python chapter7Experiments/run_chapter7_experiment_D.py
python chapter7Experiments/run_chapter7_experiment_E.py

# 3-class replication:
python experiments/ch6_ch7_3class/ch6_ch7_05_ch7_experiments.py
```

---

## Extended Experiments

### 4-Class Regime Comparison (`experiments/ch5_4class/`)

Tests whether within-valence subcategory pairs (Threat/Mutilation, Cute/Erotic) carry distinct signatures. Key finding: the reservoir outperforms band-power by +12.5 pp at 3-class but is outperformed by −6.2 pp at 4-class, identifying a regime boundary for BSC₆ temporal coding.

### Layer Ablation (`experiments/ablation/`)

The dissertation's keystone experiment. Tests 10 ablation conditions (A0–A9) and 6 clinical conditions (C1–C6), demonstrating that the three response layers carry operationally distinct information with different clinical disorders most detectable in different layers.

### 3-Class Pipeline (`experiments/ch6_ch7_3class/`)

Consolidated dynamical (Ch6) and coupling (Ch7) analysis at 3-class granularity, which provides a 3.6× signal advantage over 4-class. Produces the revised dissertation figures for Chapters 6 and 7.

---

## Dependencies

```bash
# Core (all chapters):
pip install numpy scipy scikit-learn matplotlib pandas openpyxl

# Deep learning baselines only (Chapter 5):
pip install torch
```

No deep learning frameworks required for Chapters 3–4 or 6–7. All GNN implementations in the legacy `run_chapter5_experiments.py` are from-scratch NumPy.

---

## Input Data Format

This project was developed in collaboration with the [**SHAPE study**](https://lab-can.com/shape/) at Stony Brook University. The EEG data is not included in this repository for participant privacy.

### Broad-Condition EEG Files

**Pattern:** `SHAPE_Community_{SUBJECT_ID}_IAPS{Neg,Neu,Pos}_BC.txt`

- 1229 rows × 34 columns, space-separated, microvolts
- Rows 0–204: baseline (200 ms at 1024 Hz), rows 205–1228: post-stimulus (1000 ms)
- 34 EEG channels (Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO7, PO3, POz, PO4, PO8, O1, Oz, O2, REF)

### Subcategory Files (4-class)

**Pattern:** `SHAPE_Community_{SUBJECT_ID}_IAPS{Neg,Pos}_{Threat,Mutilation,Cute,Erotic}_BC.txt`

Same structure, organized in category batch directories.

---

## Data Validation and Quality Control

```bash
python validate_shape_data.py \
    --batch1 batch1.zip --batch2 batch2.zip --batch3 batch3.zip \
    --participant_info ParticipantInfo.csv \
    --psychopathology Psychopathology.xlsx

python validate_subcategory_data.py \
    --category-dirs categoriesbatch1 categoriesbatch2 categoriesbatch3 categoriesbatch4
```

10 automated checks: file inventory, dimensional consistency, subject completeness, numerical integrity, amplitude range, flat channels, outliers, baseline verification, cross-batch duplicates, clinical cross-reference.

**Known exclusion:** Subject 127 is excluded from all analyses due to a recording anomaly (Neutral condition entirely flat).

---

## Key Parameters Reference

| Parameter | Value | Used In |
|---|---|---|
| Reservoir size | 256 neurons | All chapters |
| Membrane leak (β) | 0.05 | All chapters |
| Firing threshold | 0.5 | All chapters |
| Spectral radius | 0.9 | All chapters |
| Temporal bins (BSC₆) | 6 | All chapters |
| PCA components | 64 per channel | All chapters |
| CV folds | 10, subject-stratified | All classification |
| Random seed | 42 | All scripts |

---

## Verification Summary

| Component | Script | Tests | Status |
|---|---|---|---|
| Chapter 3–4 | `chapter4Experiments/verify_chapter4.py` | 31 | 31/31 PASS |
| Chapter 5 | `chapter5Experiments/verify_chapter5.py` | 32 | 32/32 PASS |
| Chapter 6 | `chapter6Experiments/verify_chapter6.py` | 31 | 31/31 PASS |
| Chapter 7 | `chapter7Experiments/verify_chapter7.py` | 38 | 38/38 PASS |
| 4-class ext | `experiments/ch5_4class/verify_ch5_4class.py` | 25 | 25/25 PASS |
| 3-class ext | `experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py` | 28 | 28/28 PASS |
| Ablation | `experiments/ablation/verify_ablation.py` | 23 | 23/23 PASS |
| Validators | `verify_validators.py` | 20 | 20/20 PASS |
| **Total** | | **228** | **228/228 PASS** |

These are **software infrastructure verification tests** (correct shapes, parameter consistency, numerical stability). They run without the proprietary SHAPE dataset. Empirical replication of the reported scientific results requires access to the SHAPE EEG data from Stony Brook University.

---

## Adapting to Your Own EEG Data

The pipeline expects trial-averaged, baseline-corrected ERP matrices of shape `(time_samples, n_channels)` per subject per condition. To adapt:

1. Produce per-subject, per-condition epoch matrices (average across trials if needed)
2. Ensure 1024 Hz sampling (or adjust the downsample factor)
3. Update the channel count (search for `34` in scripts)
4. Update electrode positions for the spatial graph (in `reproduce_chapter5.py`)
5. Update file discovery patterns and subject/condition parsing

See the [full adaptation guide](https://github.com/TheAwesomeAndy/dissoAdventureExperiments#adapting-to-your-own-eeg-data) for detailed instructions.

---

## Citation

```
Lane, A. (2026). ARSPI-Net: Hybrid Neuromorphic Signal Processing
Architecture for Affective EEG Analysis. PhD Dissertation,
Department of Electrical and Computer Engineering.
```
