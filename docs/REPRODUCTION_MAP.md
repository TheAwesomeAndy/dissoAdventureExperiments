# Reproduction Map: Script → Dissertation Table/Figure

This document maps every table, figure, and statistical result in the ARSPI-Net dissertation to the specific script that produces it. Any reader can verify any claim by running the corresponding script.

---

## Chapter 4: Temporal Pattern Discrimination

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

## Chapter 5: Clinical EEG Classification

| Dissertation Element | Script | Output |
|---|---|---|
| Table 5.1: 7-row baseline table | `chapter5Experiments/run_chapter5_experiments.py` | Rows 1-7 results + `baseline_comparison.pdf` |
| Table 5.2: GNN architecture comparison | `chapter5Experiments/run_chapter5_experiments.py` | Experiment 2 + `architecture_comparison.pdf` |
| Table 5.3: Graph sparsity sweep | `chapter5Experiments/run_chapter5_experiments.py` | Experiment 3 + `sparsity_sweep.pdf` |
| Table 5.4: Conventional sklearn baselines | `chapter5Experiments/sklearn_baselines.py` | 8 classifier results |
| Table 5.5: Deep learning baselines | `chapter5Experiments/eegnet_gru_lstm_baselines.py` | EEGNet/GRU/LSTM results |
| Table 5.6: Experiment Zero disambiguation | `chapter5Experiments/experiment_zero.py` | 4-number centered/uncentered table |
| Figure 5.1: Confusion matrix | `chapter5Experiments/run_chapter5_experiments.py` | `confusion_matrix.pdf` |
| Figure 5.2: Deep baseline comparison | `chapter5Experiments/results/fig_deep_baselines.pdf` | Pre-generated |
| Full reproducibility | `chapter5Experiments/reproduce_chapter5.py` | All Ch5 figures + `ch5_all_results.pkl` |

---

## Chapter 6: Dynamical Characterization (4-class)

| Dissertation Element | Script | Output |
|---|---|---|
| Table 6.1: ESP verification | `chapter6Experiments/run_chapter6_exp1_esp.py` | lambda_1 values |
| Table 6.2: Cross-seed reliability (ICC) | `chapter6Experiments/run_chapter6_exp2_reliability.py` | ICC per metric |
| Table 6.3: Surrogate sensitivity | `chapter6Experiments/run_chapter6_exp3_surrogate.py` | Significance per metric |
| Table 6.4: Value-add vs raw EEG | `chapter6Experiments/run_chapter6_exp3_valueadd.py` | Ratio comparisons |
| Table 6.5: Subcategory dissociation | `chapter6Experiments/run_chapter6_exp4_dissociation.py` | Within-valence d_z values |
| Table 6.6: Diagnosis × category interaction | `chapter6Experiments/run_chapter6_exp5_interaction.py` | SUD/ADHD effects |
| Table 6.7: Temporal localization | `chapter6Experiments/run_chapter6_exp6_temporal.py` | Peak at 708 ms |
| Full reproducibility | `chapter6Experiments/reproduce_chapter6.py` | All Ch6 figures + results |

---

## Chapter 7: Dynamical-Topological Coupling (4-class)

| Dissertation Element | Script | Output |
|---|---|---|
| Table 7.1: Coupling existence | `chapter7Experiments/run_chapter7_experiment_A.py` | d_z = 1.063, p < 10^-100 |
| Table 7.2: Variance decomposition | `chapter7Experiments/run_chapter7_experiment_B.py` | 29% subject, 1% category, 70% residual |
| Table 7.3: Category-conditioned C matrices | `chapter7Experiments/run_chapter7_experiment_C.py` | tau_AC × clustering finding |
| Table 7.4: Diagnosis coupling differences | `chapter7Experiments/run_chapter7_experiment_D.py` | Clean null (all d < 0.12) |
| Table 7.5: Augmentation ablation | `chapter7Experiments/run_chapter7_experiment_E.py` | ADHD D-only AUC 0.622 |
| Utility: kappa CSV export | `chapter7Experiments/extract_kappa_matrix.py` | `kappa_matrix.csv` |
| Utility: C matrices CSV export | `chapter7Experiments/extract_C_matrices.py` | `C_matrices.csv` |

---

## Extended Experiments (March 2026)

| Dissertation Element | Script | Output |
|---|---|---|
| Ch5 §5.5: 4-class features | `experiments/ch5_4class/ch5_4class_01_feature_extraction.py` | `shape_features_4class.pkl` |
| Ch5 §5.5: 4-class observations | `experiments/ch5_4class/ch5_4class_02_raw_observations.py` | 7 observation PDFs |
| Ch5 §5.5: 4-class classification | `experiments/ch5_4class/ch5_4class_03_classification_full.py` | 11 experiments |
| Ch6/7 3-class: features | `experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py` | `ch6_ch7_3class_features.pkl` |
| Ch6/7 3-class: observations | `experiments/ch6_ch7_3class/ch6_ch7_02_raw_observations.py` | 8 observation PDFs |
| Ch6/7 3-class: Ch6 experiments | `experiments/ch6_ch7_3class/ch6_03_experiments.py` | 7 experiments |
| Ch6/7 3-class: Ch7 experiments | `experiments/ch6_ch7_3class/ch7_04_experiments.py` | 5 experiments |
| Keystone ablation matrix | `experiments/ablation/layer_ablation.py` | A0-A9 + C1-C6 |

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
| Ch5 run_chapter5_experiments.py | `chapter5Experiments/verify_chapter5.py` | 32 |
| Ch5 experiment_zero.py | `chapter5Experiments/verify_experiment_zero.py` | 37 |
| Ch5 reproduce_chapter5.py | `chapter5Experiments/verify_reproduce_chapter5.py` | 33 |
| Ch5 sklearn + deep baselines | `chapter5Experiments/verify_baselines.py` | see script |
| Ch6 all 7 experiment scripts | `chapter6Experiments/verify_chapter6.py` | 31 |
| Ch6 reproduce_chapter6.py | `chapter6Experiments/verify_reproduce_chapter6.py` | 42 |
| Ch7 experiments A-E | `chapter7Experiments/verify_chapter7.py` | 38 |
| Ch7 extract utilities | `chapter7Experiments/verify_extract_utilities.py` | 40 |
| Ch5 4-class extension | `experiments/ch5_4class/verify_ch5_4class.py` | 25 |
| Ch6/7 3-class pipeline | `experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py` | 28 |
| Ablation keystone | `experiments/ablation/verify_ablation.py` | 23 |
| Data validators | `validation/verify_validators.py` | 20 |
