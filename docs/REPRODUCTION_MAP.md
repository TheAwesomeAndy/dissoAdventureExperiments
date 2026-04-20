# Reproduction Map: Script → Dissertation Table/Figure

This document maps every table, figure, and statistical result in the ARSPI-Net dissertation to the specific script that produces it. Any reader can verify any claim by running the corresponding script.

**Important:** The dissertation's final results for Chapters 6 and 7 use the **3-class** design (Negative / Neutral / Pleasant). The original 4-class experiments in `chapter6Experiments/` and `chapter7Experiments/` represent the initial exploration and are retained for completeness. The 3-class scripts under `experiments/ch6_ch7_3class/` produce the numbers reported in the final dissertation.

---

## Chapter 4: Spike-to-Embedding Pipeline (Synthetic + EEG)

| Dissertation Element | Script | Output |
|---|---|---|
| Table 4.1: Reservoir size ablation | `chapter4Experiments/run_chapter4_experiments.py` | Exp 1 results + `ablation_reservoir_size.pdf` |
| Table 4.2: Coding scheme comparison | `chapter4Experiments/run_chapter4_experiments.py` | Exp 3 results + `coding_scheme_accuracy_comparison.pdf` |
| Figure 4.1: FDR three-way comparison | `chapter4Experiments/run_chapter4_experiments.py` | `fdr_three_way_comparison.pdf` |
| Figure 4.2: PCA explained variance | `chapter4Experiments/run_chapter4_experiments.py` | `pca_explained_variance.pdf` |
| Figure 4.3: Cross-seed robustness | `chapter4Experiments/run_chapter4_experiments.py` | `cross_initialization_robustness.pdf` |
| Figure 4.4: Parameter sensitivity | `chapter4Experiments/run_chapter4_experiments.py` | `parameter_sensitivity_heatmap.pdf` |
| Figures 4.5–4.10: Raw observations | `chapter4Experiments/run_chapter4_observations.py` | `obs01` through `obs06` PDFs |

---

## Chapter 5: Clinical EEG Classification + Baselines

### PRIMARY: Complete Centered Baseline Table (Dissertation Table 5.X)

**The dissertation's headline 7-row centered baseline table is assembled from multiple scripts; no single script produces all 7 rows.**

| Dissertation Row | Uncentered → Centered | Script |
|---|---|---|
| EEGNet (Lawhern 2018) | 72.0% → 89.1% | `chapter5Experiments/canonical_pytorch_baselines.py` |
| GRU (2-layer bidir) | 59.9% → 78.4% | `chapter5Experiments/canonical_pytorch_baselines.py` |
| LSTM (2-layer bidir) | 58.0% → 71.1% | `chapter5Experiments/canonical_pytorch_baselines.py` |
| Raw EEG + LogReg | 70.5% → 88.4% | `chapter5Experiments/experiment_zero.py` |
| ARSPI-Net reservoir + LogReg | 59.4% → 78.8% | `chapter5Experiments/experiment_zero.py` |
| PCA-200 + LogReg | 64.9% → 86.4% | (not in repo — see note below) |
| Band Power + SVM | 47.7% → 61.0% | `chapter5Experiments/sklearn_baselines.py` |

**Note on PCA-200:** The dissertation reports 64.9% → 86.4% for PCA-200 + LogReg, but no committed script currently produces this row. The result was computed during Chapter 5 development and the number has been retained as an uncommitted intermediate. Running `sklearn_baselines.py` with the PCA-200 feature stack as input is the intended reproduction path.

| Supporting Analysis | Script | Output |
|---|---|---|
| Variance decomposition (ρ = 7.2×) | `chapter5Experiments/canonical_pytorch_baselines.py` (`--centered` flag run) | 62.6% subject, 8.7% condition, 28.7% residual |
| Experiment Zero disambiguation | `chapter5Experiments/experiment_zero.py` | Confirms 70.5% is uncentered; centered Raw EEG = 88.4% |

### Supporting: GNN Ablation Tables and Alternative Readouts

| Dissertation Element | Script | Output |
|---|---|---|
| Table 5.1: GNN ablation (7 architecture variants — NOT the headline baseline table above) | `chapter5Experiments/run_chapter5_experiments.py` | 7 rows: BandPower+LogReg, BandPower+MLP, LSM+PCA+MLP, BandPower+GAT(spat), ARSPI-Net (full), ARSPI-Net (func), LSM-MFR+GAT(spat) |
| Table 5.2: GNN architecture comparison | `chapter5Experiments/run_chapter5_experiments.py` | Experiment 2 |
| Table 5.3: Graph sparsity sweep | `chapter5Experiments/run_chapter5_experiments.py` | Experiment 3 |
| Table 5.4: Conventional sklearn baselines | `chapter5Experiments/sklearn_baselines.py` | 8 classifier results |
| Table 5.5: Deep baselines (NumPy reference, superseded) | `chapter5Experiments/deprecated/eegnet_gru_lstm_baselines.py` | NumPy implementations (see note) |
| Figure 5.1: Confusion matrix | `chapter5Experiments/run_chapter5_experiments.py` | `confusion_matrix.pdf` |
| Full reproducibility | `chapter5Experiments/reproduce_chapter5.py` | All Ch5 figures |

**Note:** `deprecated/eegnet_gru_lstm_baselines.py` contains simplified NumPy reference implementations retained for reproducibility. The canonical results use `canonical_pytorch_baselines.py` with full end-to-end PyTorch training.

---

## Chapter 6: Dynamical Characterization — PRIMARY: 3-Class

| Dissertation Element | Script | Output |
|---|---|---|
| **All Ch6 3-class experiments** | **`experiments/ch6_ch7_3class/ch6_03_experiments.py`** | **7/7 condition-sensitive, 0/49 clinical** |
| 3-class features | `experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py` | `ch6_ch7_3class_features.pkl` |
| 3-class observations | `experiments/ch6_ch7_3class/ch6_ch7_02_raw_observations.py` | 8 observation PDFs |

### Exploratory: 4-Class (Original)

| Dissertation Element | Script | Output |
|---|---|---|
| ESP verification | `chapter6Experiments/run_chapter6_exp1_esp.py` | lambda_1 values |
| Cross-seed reliability (ICC) | `chapter6Experiments/run_chapter6_exp2_reliability.py` | ICC per metric |
| Surrogate sensitivity | `chapter6Experiments/run_chapter6_exp3_surrogate.py` | Significance per metric |
| Value-add vs raw EEG | `chapter6Experiments/run_chapter6_exp3_valueadd.py` | Ratio comparisons |
| Subcategory dissociation | `chapter6Experiments/run_chapter6_exp4_dissociation.py` | Within-valence d_z values |
| Diagnosis × category interaction | `chapter6Experiments/run_chapter6_exp5_interaction.py` | SUD/ADHD effects |
| Temporal localization | `chapter6Experiments/run_chapter6_exp6_temporal.py` | Peak at 708 ms |

---

## Chapter 7: Structure-Function Coupling — PRIMARY: 3-Class

| Dissertation Element | Script | Output |
|---|---|---|
| **All Ch7 3-class experiments** | **`experiments/ch6_ch7_3class/ch7_04_experiments.py`** | **κ = 0.22, p < 0.001** |

### Exploratory: 4-Class (Original)

| Dissertation Element | Script | Output |
|---|---|---|
| Coupling existence | `chapter7Experiments/run_chapter7_experiment_A.py` | d_z = 1.063 |
| Variance decomposition | `chapter7Experiments/run_chapter7_experiment_B.py` | 29% subject, 1% category |
| Category-conditioned coupling | `chapter7Experiments/run_chapter7_experiment_C.py` | tau_AC × clustering |
| Diagnosis coupling differences | `chapter7Experiments/run_chapter7_experiment_D.py` | Clean null (all d < 0.12) |
| Augmentation ablation | `chapter7Experiments/run_chapter7_experiment_E.py` | ADHD D-only AUC 0.622 |

---

## Layer Ablation (Keystone Experiment)

| Dissertation Element | Script | Output |
|---|---|---|
| Ablation matrix (A1–A9, C1–C6) | `experiments/ablation/layer_ablation.py` | Full ablation results |

---

## Chapter 3: LIF Reservoir Characterization (Synthetic)

| Dissertation Element | Script | Output |
|---|---|---|
| Membrane dynamics verification | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 1: leak, fire, reset checks |
| Separation property (Maass 2002) | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 2: input–state distance correlation |
| Fading memory decay | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 3: τ decay constant |
| Spectral radius sweep | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 4: accuracy vs ρ |
| Kernel quality | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 5: effective rank |
| Cross-seed robustness | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 6: 10 seeds |

---

## Interpretability Validation (Four-Level Taxonomy)

| Dissertation Element | Script | Output |
|---|---|---|
| Level 1: LPP recovery (r ≈ 0.23 post-reservoir) | `experiments/interpretability/run_level1_temporal_traceability.py` | Per-channel correlation + PDF |
| Level 1: LPP prediction (R² = 0.661) | `experiments/interpretability/run_level1_temporal_traceability.py` | Ridge regression + PDF |
| EEGNet saliency peaks (402–691 ms) | `experiments/interpretability/run_eegnet_saliency_comparison.py` | Saliency profile + PDF |
| Attention-prototype readout (66.7%, p = 0.634) | `experiments/interpretability/run_arspinet_v2_attention_prototype.py` | Readout comparison + PDF |

---

## Extended: 4-Class Classification

| Dissertation Element | Script | Output |
|---|---|---|
| 4-class features | `experiments/ch5_4class/ch5_4class_01_feature_extraction.py` | `shape_features_4class.pkl` |
| 4-class observations | `experiments/ch5_4class/ch5_4class_02_raw_observations.py` | 7 observation PDFs |
| 4-class classification | `experiments/ch5_4class/ch5_4class_03_classification_full.py` | 52.0% centered, 40.4% raw |

---

## Data Validation

| Purpose | Script |
|---|---|
| 3-class EEG QC (10 checks) | `validation/validate_shape_data.py` |
| 4-class subcategory QC (12 checks) | `validation/validate_subcategory_data.py` |
| Validator meta-verification | `validation/verify_validators.py` |

---

## Verification Scripts

| Verified Scripts | Verification Script | Tests |
|---|---|---|
| Ch4 experiments + observations | `chapter4Experiments/verify_chapter4.py` | 31 |
| Ch5 core infrastructure | `chapter5Experiments/verify_chapter5.py` | 32 |
| Ch5 Experiment Zero | `chapter5Experiments/verify_experiment_zero.py` | 37 |
| Ch5 reproducibility pipeline | `chapter5Experiments/verify_reproduce_chapter5.py` | 33 |
| Ch5 sklearn + deep baselines | `chapter5Experiments/verify_baselines.py` | 55 |
| Ch6 all experiment scripts | `chapter6Experiments/verify_chapter6.py` | 31 |
| Ch6 reproducibility pipeline | `chapter6Experiments/verify_reproduce_chapter6.py` | 42 |
| Ch7 experiments A–E | `chapter7Experiments/verify_chapter7.py` | 38 |
| Ch7 extract utilities | `chapter7Experiments/verify_extract_utilities.py` | 40 |
| Ch5 4-class extension | `experiments/ch5_4class/verify_ch5_4class.py` | 25 |
| Ch6/7 3-class pipeline | `experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py` | 28 |
| Ablation keystone | `experiments/ablation/verify_ablation.py` | 23 |
| Data validators | `validation/verify_validators.py` | 20 |
| **Total** | | **435** |
