# ARSPI-Net: A Four-Level Interpretable Neuromorphic Framework for Clinical EEG Analysis

**Dissertation:** Lane, A. A. (2026). *Affective Reservoir-Spike Processing and Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic Framework for Clinical EEG Analysis.* PhD Dissertation, Department of Electrical and Computer Engineering, Stony Brook University.

**Advisor:** K. Wendy Tang (Senior Member, IEEE)

This repository contains the complete experimental pipeline for the ARSPI-Net dissertation. The framework converts raw multichannel EEG into a staged neuromorphic representation — fixed-weight LIF spiking reservoir → BSC₆ temporal coding → PCA-64 compression → graph-organized spatial analysis — whose internal layers are independently analyzable, providing a four-level interpretability taxonomy for clinical EEG.

**ARSPI-Net is not an accuracy-maximizing classifier.** EEGNet achieves 89.1% (centered) with no decomposition; ARSPI-Net achieves 78.8% (centered) with full decomposition across temporal window, dynamical regime, graph topology, and structure-function coupling. Intrinsic interpretability — not peak accuracy — is the primary contribution. Every output variable traces directly to input ERP transients through an explicit signal chain with no learned recurrent weights.

---

## Publications

1. **Lane, A., Nelson, B. D., & Tang, K. W.** (2023). "Towards ARSPI-Net: Development of an Efficient Hybrid Deep Learning Framework." *Proc. IEEE Long Island Systems, Applications and Technology Conference (LISAT)*. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10179592)

2. **Lane, A., Nelson, B. D., & Tang, K. W.** (2024). "Towards ARSPI-NET: Advancing EEG Feature Extraction with Neuromorphic Algorithms." *Proc. IEEE Long Island Systems, Applications and Technology Conference (LISAT)*.

3. **Lane, A. A., Tang, K. W., & Nelson, B. D.** (2026). "Subject-Covariance Entanglement in Affective EEG Embeddings." *IEEE Signal Processing Letters* — **submitted, under review**.

---

## Key Results Summary

All classification results use 10-fold StratifiedGroupKFold cross-validation (random_state=42) with subject-level splitting on the SHAPE Community dataset (N=211 transdiagnostic subjects, 34 channels, trial-averaged ERPs). Dataset: [https://lab-can.com/shape/](https://lab-can.com/shape/)

### Four-Level Interpretability Taxonomy

| Level | Property | Key Metric | Chapter |
|---|---|---|---|
| 1 | Temporal traceability | r = 0.82 LPP recovery | Ch4 |
| 2 | Geometric transparency | ρ = 7.2 variance ratio, +13 to +21 pp centering gain | Ch5 |
| 3 | Dynamical characterization | R² = 0.661 LPP prediction | Ch6 |
| 4 | Systems-level correspondence | κ = 0.22, p < 0.001 | Ch7 |

### Complete Baseline Table (Uncentered → Centered)

All deep learning baselines trained with canonical PyTorch implementations: Adam optimizer, early stopping, gradient clipping, ReduceLROnPlateau, 10 × 5 seeds.

| Model | Uncentered | Centered | Gain |
|---|---|---|---|
| EEGNet (Lawhern 2018) | 72.0% ± 4.9% | 89.1% ± 3.1% | **+17.1 pp** |
| Raw EEG + LogReg | 70.5% ± 3.0% | 88.4% | **+17.9 pp** |
| PCA-200 + LogReg | 64.9% ± 5.0% | 86.4% | **+21.5 pp** |
| GRU (2-layer bidir) | 59.9% ± 6.4% | 78.4% ± 3.5% | **+18.5 pp** |
| **ARSPI-Net (reservoir)** | **59.4% ± 3.6%** | **78.8%** | **+19.4 pp** |
| LSTM (2-layer bidir) | 58.0% ± 5.5% | 71.1% ± 6.8% | **+13.1 pp** |
| Band Power + SVM | 47.7% ± 5.1% | 61.0% | **+13.3 pp** |

**Centering is the dominant intervention; architecture choice is secondary.** Subject-mean centering removes the subject-specific DC offset, revealing +13 to +21 pp of hidden condition signal across all 7 representations. Variance decomposition: 62.6% subject, 8.7% condition, 28.7% residual (ρ = 7.2×).

### Key Experimental Results

- **3-class:** Cross-subject accuracy 63.4%; subject-centering reveals latent capacity of 79.4% (SUD p=0.0004)
- **4-class:** Cross-subject accuracy 40.4%; centered 52.0% (PTSD threat-specificity p=0.036)
- **Regime boundary:** Reservoir +12.5 pp at 3-class, −6.2 pp at 4-class
- **Dynamical descriptors (Ch6):** 7/7 condition-sensitive, 0/49 clinical significant; temporal family 2.4× > amplitude
- **Structure-function coupling (Ch7):** κ = 0.22, p < 0.001; no clinical coupling; Cute-Erotic p=0.025 (4-class only)
- **Latent axis:** Continuous excitability-persistence axis (PC1, 19.1%), condition-modulated (Friedman p=6×10⁻⁶), diagnostically independent

---

## Repository Structure

```
dissoAdventureExperiments/
├── README.md                              # This file
│
├── chapter4Experiments/                   # Ch4: Spike-to-embedding pipeline (synthetic + EEG)
│   ├── run_chapter4_experiments.py        #   6 experiments: ablation, FDR, coding, PCA, robustness, sensitivity
│   ├── run_chapter4_observations.py       #   6 raw observation figures
│   └── verify_chapter4.py                #   Verification (31 tests)
│
├── chapter5Experiments/                   # Ch5: Clinical EEG classification + baselines
│   ├── run_chapter5_experiments.py        #   7-row baseline table + GNN experiments
│   ├── sklearn_baselines.py              #   8 sklearn classifiers
│   ├── deprecated/
│   │   └── eegnet_gru_lstm_baselines.py  #   NumPy reference implementations (superseded, see note below)
│   ├── canonical_pytorch_baselines.py    #   Canonical PyTorch EEGNet/GRU/LSTM (produces centered table)
│   ├── experiment_zero.py                #   Baseline disambiguation (centered vs uncentered)
│   ├── reproduce_chapter5.py             #   Standalone reproducibility pipeline
│   └── verify_*.py                       #   Verification scripts (157 tests total)
│
├── chapter6Experiments/                   # Ch6: Dynamical characterization of LIF reservoir
│   ├── run_chapter6_exp1-6.py            #   ESP, reliability, surrogate, value-add, dissociation, temporal
│   ├── reproduce_chapter6.py             #   Standalone reproducibility pipeline
│   └── verify_*.py                       #   Verification scripts (73 tests)
│
├── chapter7Experiments/                   # Ch7: Dynamical-topological coupling
│   ├── run_chapter7_experiment_A-E.py    #   Coupling, variance, category, diagnosis, ablation
│   ├── extract_kappa_matrix.py           #   Utility: export coupling as CSV
│   └── verify_*.py                       #   Verification scripts (78 tests)
│
├── experiments/                           # Extended experiments (March 2026)
│   ├── ch5_4class/                       #   4-class classification extension
│   ├── ch6_ch7_3class/                   #   3-class dynamical + coupling pipeline (PRIMARY for Ch6/Ch7)
│   ├── ablation/                         #   Layer ablation keystone experiment
│   ├── chapter3/                         #   Ch3: Controlled LIF reservoir characterization (synthetic)
│   └── interpretability/                 #   Four-level interpretability validation scripts
│
├── validation/                            # Data quality control
│   ├── validate_shape_data.py            #   QC for 3-class SHAPE data
│   └── validate_subcategory_data.py      #   QC for 4-class subcategory data
│
├── data/clinical_profile.csv             # 211-subject clinical profiles
├── docs/
│   ├── methodology_rules.md              #   12 governing methodology rules
│   ├── VERIFICATION_METHODOLOGY.md       #   Verification trustworthiness rationale
│   └── REPRODUCTION_MAP.md              #   Script → dissertation table/figure mapping
├── latex/                                 # LaTeX chapter sources
├── pictures/chLSMEmbeddings/             # Ch4 publication figures
└── .gitignore
```

### Note on Baseline Implementations

`deprecated/eegnet_gru_lstm_baselines.py` contains simplified NumPy reference implementations where EEGNet trains only the FC layer and GRU/LSTM use fixed random recurrent weights. These were the original exploration baselines and are retained for provenance only. The dissertation's final baseline table (above) uses `canonical_pytorch_baselines.py`, which implements full end-to-end training with canonical architectures (Lawhern et al. 2018 for EEGNet), Adam optimization, early stopping, gradient clipping, and ReduceLROnPlateau scheduling.

---

## Reproduction Map

### Primary Results (3-Class: Negative / Neutral / Pleasant)

The dissertation's final results for Chapters 5–7 use the **3-class** design. The 4-class experiments (chapter6Experiments/, chapter7Experiments/) represent the original exploration; the 3-class scripts under `experiments/ch6_ch7_3class/` produce the numbers reported in the final dissertation.

| Dissertation Element | Script | Key Result |
|---|---|---|
| Deep learning baselines (EEGNet/GRU/LSTM) | `chapter5Experiments/canonical_pytorch_baselines.py` | 3 deep learning rows |
| Full 7-row baseline table | `chapter5Experiments/run_chapter5_experiments.py` | All 7 rows (incl. Raw EEG, PCA-200, Reservoir, BandPower) |
| Experiment Zero disambiguation | `chapter5Experiments/experiment_zero.py` | 70.5% confirmed uncentered |
| Ch6 3-class dynamical descriptors | `experiments/ch6_ch7_3class/ch6_03_experiments.py` | 7/7 condition-sensitive |
| Ch7 3-class coupling | `experiments/ch6_ch7_3class/ch7_04_experiments.py` | κ = 0.22, p < 0.001 |
| Layer ablation | `experiments/ablation/layer_ablation.py` | A1–A9, C1–C6 complete |

### Extended Results (4-Class)

| Dissertation Element | Script | Key Result |
|---|---|---|
| 4-class classification | `experiments/ch5_4class/ch5_4class_03_classification_full.py` | 52.0% centered, 40.4% raw |
| 4-class Ch6 experiments | `chapter6Experiments/run_chapter6_exp*.py` | Original exploration |
| 4-class Ch7 experiments | `chapter7Experiments/run_chapter7_experiment_*.py` | Original exploration |

### Interpretability Validation & Chapter 3

| Dissertation Element | Script | Key Result |
|---|---|---|
| Ch3 LIF reservoir characterization | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Separation, fading memory, kernel quality (synthetic) |
| Level 1 temporal traceability | `experiments/interpretability/run_level1_temporal_traceability.py` | r = 0.82 LPP recovery, R² = 0.661 prediction |
| EEGNet saliency comparison | `experiments/interpretability/run_eegnet_saliency_comparison.py` | EEGNet 402–691 ms vs ARSPI-Net 176–254 ms |
| Attention-prototype readout | `experiments/interpretability/run_arspinet_v2_attention_prototype.py` | 66.7%, permutation p = 0.634 (not significant) |

---

## Architecture Overview

The ARSPI-Net pipeline implements a Koopman-theoretic signal processing chain:

1. **LIF Spiking Reservoir** (Ch3) — Fixed-weight leaky integrate-and-fire reservoir converts continuous EEG into spike trains. The separation property (Maass 2002) guarantees distinct inputs produce distinct reservoir states. Chapter 3 is theoretical/analytical — it derives the reservoir properties and separation guarantees. The computational validation (reservoir size ablation, coding scheme comparison, cross-seed robustness, parameter sensitivity) is performed in `chapter4Experiments/`.

2. **BSC₆ Temporal Coding** (Ch4) — Binned Spike Count with 6 temporal bins discretizes the continuous spike response into a structured temporal representation. Each bin is independently analyzable (temporal traceability).

3. **PCA-64 Compression** (Ch5) — Truncated PCA compresses the BSC₆ output while preserving condition-discriminative geometry. Subject-mean centering removes the dominant subject variance (62.6%) to expose the condition signal (8.7%).

4. **Graph-Organized Spatial Analysis** (Ch5, Ch7) — Electrode-level features organized by channel topology enable clinical phenotyping through graph metrics and structure-function coupling (κ).

### Interpretability Signal Chain

Every output variable traces to input through: membrane voltage → spike times → BSC₆ bins → PCA components → named descriptors → graph topology → coupling κ. No learned recurrent weights. No black-box layers. This is intrinsic interpretability, not post-hoc explanation.

---

## Dependencies

```
numpy>=1.21
scipy>=1.7
scikit-learn>=1.0
matplotlib>=3.5
torch>=1.12          # For canonical_pytorch_baselines.py only
```

## Input Data Format

The SHAPE Community dataset is available at [https://lab-can.com/shape/](https://lab-can.com/shape/). Raw EEG data (not included in this repository) should be placed in `batch_data/` (3-class) or `categories/` (4-class) following the structure described in the validation scripts.

**3-class data** (`batch_data/`): Baseline-corrected text files, one per subject per condition.
- Format: `SHAPE_Community_{SUBJECT_ID}_IAPS{Neg,Neu,Pos}_BC.txt`
- Dimensions: 1229 rows (timepoints at 1024 Hz) x 34 columns (EEG channels)
- Units: microvolts, loadable with `np.loadtxt(filepath)`

**4-class data** (`categories/`): `.mat` files organized by affective subcategory (Threat / Mutilation / Cute / Erotic), loadable with `scipy.io.loadmat()`.

**Intermediate pickle files** (generated by the pipeline, not included in the repo):
- `shape_features_211.pkl` — Chapter 5 features (from `run_chapter5_experiments.py`)
- `ch6_ch7_3class_features.pkl` — 3-class features (from `ch6_ch7_01_feature_extraction.py`)

These pickles are required by the ablation script (`experiments/ablation/layer_ablation.py`).

---

## Verification Summary

| Suite | Script | Tests | Status |
|---|---|---|---|
| Ch4 Core | `verify_chapter4.py` | 31 | PASS |
| Ch5 Core | `verify_chapter5.py` | 32 | PASS |
| Ch5 Exp Zero | `verify_experiment_zero.py` | 37 | PASS |
| Ch5 Reproduce | `verify_reproduce_chapter5.py` | 33 | PASS |
| Ch5 Baselines | `verify_baselines.py` | 55 | PASS |
| Ch6 Core | `verify_chapter6.py` | 31 | PASS |
| Ch6 Reproduce | `verify_reproduce_chapter6.py` | 42 | PASS |
| Ch7 Experiments | `verify_chapter7.py` | 38 | PASS |
| Ch7 Utilities | `verify_extract_utilities.py` | 40 | PASS |
| Validation | `verify_validators.py` | 20 | PASS |
| Ch5 4-class | `verify_ch5_4class.py` | 25 | PASS |
| Ch6/7 3-class | `verify_ch6_ch7_3class.py` | 28 | PASS |
| Ablation | `verify_ablation.py` | 23 | PASS |
| **Total** | | **435** | **435/435 PASS** |

Methodology: [docs/VERIFICATION_METHODOLOGY.md](docs/VERIFICATION_METHODOLOGY.md)
