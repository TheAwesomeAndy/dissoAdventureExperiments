# Reproduction Map: Script → Dissertation Table/Figure

This document maps the tables, figures, and statistical results in the ARSPI-Net dissertation to the scripts that produce them, organized into three tiers so a reader can tell what the corrected primary inference rests on versus what is retained for provenance.

- **Confirmatory** — the corrected estimates the dissertation reports as its primary inference: five repeats of five-fold participant-grouped cross-validation, fold-local preprocessing, and a data-independent SRP-64 projection.
- **Exploratory** — analyses that generate hypotheses (3-class dynamical/coupling pipeline, 4-class extensions).
- **Archived** — legacy pipelines (globally fitted PCA, transductive subject-centering, message-passing GNN sweep, original 4-class experiments) retained for provenance and as prespecified replication targets. These are **descriptive**, not prospective, and are **not** used for the primary inference.

Every number is also indexed, with full provenance fields, in the machine-readable manifest [`../results_manifest.json`](../results_manifest.json).

> **Important scope note.** Passing verification tests establish code behavior, not scientific validity (see [`VERIFICATION_METHODOLOGY.md`](VERIFICATION_METHODOLOGY.md)). The tier of a result — confirmatory vs. archived — is a statement about its study design and preprocessing boundary, not about whether its test suite passes. All 435 tests pass regardless of tier.

---

## TIER 1 — CONFIRMATORY (primary inference)

**Protocol.** Five repeats of five-fold participant-grouped cross-validation; every participant's full condition panel stays in one fold. Per-feature standardization and the linear logistic readout are fitted on training participants only (fold-local). BSC₆ is compressed 1,536→64 with a fixed sparse random projection (SRP-64) from a prespecified seed. Intervals are participant-bootstrap 95% intervals; a deterministic ridge readout drives 100 within-participant label permutations.

**Samples.** 211 participants / 633 observations (three conditions: Negative, Neutral, Pleasant); 210 participants / 840 observations (four categories: Threat, Mutilation, Cute, Erotic).

| Dissertation Element | Value | Script |
|---|---|---|
| Table (corrected validation): BSC₆/SRP-64 | 60.6% [57.8, 63.4] (3-cond); 53.2% [50.6, 55.8] (4-cat) | *not yet committed — see note* |
| Raw ERP-16 reference | 68.1% [65.1, 71.1] (3-cond); 60.5% [57.9, 63.0] (4-cat) | *not yet committed* |
| Conventional spectral/Hjorth baseline | 49.4% (3-cond); 36.7% (4-cat) | *not yet committed* |
| Fusion (BSC₆/SRP-64 + conventional) | 62.0% (3-cond); 53.5% (4-cat) | *not yet committed* |
| Advantage over conventional | +11.3 pp [7.9, 14.7] (3-cond); +16.5 pp [13.3, 19.8] (4-cat) | *not yet committed* |
| Participant-label permutation (ridge) | p = 0.0099 (both granularities) | *not yet committed* |
| Destruction controls (temporal-bin shuffle / collapse / electrode shuffle) | −7.4/−4.9/−13.9 pp (3-cond); −12.9/−8.6/−20.1 pp (4-cat) | *not yet committed* |
| Seed robustness (7, 42, 99) | 60.5–62.1%; full window 63.3% (3-cond) | *not yet committed* |

> **Confirmatory-pipeline gap.** These corrected numbers are reported in `dissertation/chgraph.tex` and `dissertation/main.tex`, but a committed end-to-end script that regenerates them from the SHAPE data is **not yet in this repository**. The manifest marks these rows `"script": null`. This gap is stated explicitly rather than papered over with the archived scripts, which produce the descriptive legacy estimates below, not the confirmatory ones. Committing that pipeline (a `SparseRandomProjection`-based `RepeatedStratifiedGroupKFold` runner with fold-local standardization) is the top reproduction to-do.

---

## TIER 2 — EXPLORATORY (hypothesis-generating)

### Chapter 6: Dynamical Characterization — 3-class

| Dissertation Element | Script | Output |
|---|---|---|
| All Ch6 3-class experiments | `experiments/ch6_ch7_3class/ch6_03_experiments.py` | 7/7 condition-sensitive, 0/49 clinical significant |
| 3-class features | `experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py` | `ch6_ch7_3class_features.pkl` |
| 3-class observations | `experiments/ch6_ch7_3class/ch6_ch7_02_raw_observations.py` | 8 observation PDFs |

### Chapter 7: Structure-Function Coupling — 3-class

| Dissertation Element | Script | Output |
|---|---|---|
| All Ch7 3-class experiments | `experiments/ch6_ch7_3class/ch7_04_experiments.py` | median κ ≈ 0.22, p < 0.001 |

### Extended 4-class classification

| Dissertation Element | Script | Output |
|---|---|---|
| 4-class features | `experiments/ch5_4class/ch5_4class_01_feature_extraction.py` | `shape_features_4class.pkl` |
| 4-class observations | `experiments/ch5_4class/ch5_4class_02_raw_observations.py` | 7 observation PDFs |
| 4-class classification | `experiments/ch5_4class/ch5_4class_03_classification_full.py` | 52.0% centered, 40.4% raw (archived preprocessing) |

### Chapter 3: LIF Reservoir Characterization (synthetic)

| Dissertation Element | Script | Output |
|---|---|---|
| Membrane dynamics, separation, fading memory, spectral radius, kernel quality, cross-seed | `experiments/chapter3/run_chapter3_lsm_characterization.py` | Exp 1–6 |

### Chapter 4: Spike-to-Embedding Pipeline (synthetic + EEG)

| Dissertation Element | Script | Output |
|---|---|---|
| Table 4.1: Reservoir size ablation | `chapter4Experiments/run_chapter4_experiments.py` | Exp 1 + `ablation_reservoir_size.pdf` |
| Table 4.2: Coding scheme comparison | `chapter4Experiments/run_chapter4_experiments.py` | Exp 3 + `coding_scheme_accuracy_comparison.pdf` |
| Figures 4.1–4.4 | `chapter4Experiments/run_chapter4_experiments.py` | FDR, PCA variance, robustness, sensitivity |
| Figures 4.5–4.10: Raw observations | `chapter4Experiments/run_chapter4_observations.py` | `obs01`–`obs06` PDFs |

### Interpretability validation (four-level audit framework)

| Dissertation Element | Script | Output |
|---|---|---|
| Level 1: LPP recovery (post-reservoir \|r\| ≈ 0.23) | `experiments/interpretability/run_level1_temporal_traceability.py` | Per-channel correlation + PDF |
| Level 3: Descriptor-ERP alignment (amp \|r\| = 0.82, spike \|r\| = 0.23) | `experiments/interpretability/run_level3_descriptor_erp_alignment.py` | Per-channel correlation table + PDF |
| EEGNet saliency peaks (402–691 ms) | `experiments/interpretability/run_eegnet_saliency_comparison.py` | Saliency profile + PDF |
| Attention-prototype readout (66.7%, p = 0.634, not significant) | `experiments/interpretability/run_arspinet_v2_attention_prototype.py` | Readout comparison + PDF |

---

## TIER 3 — ARCHIVED (descriptive / provenance only)

These use the archived input pickle, globally fitted PCA, and — in centered columns — the complete held-out participant panel (transductive). They are **not** prospective performance and are **not** used for the primary inference. Retained to motivate follow-up and define replication targets.

### Archived baseline / centered comparison

| Dissertation Row | Uncentered → Centered (transductive) | Script |
|---|---|---|
| EEGNet (Lawhern 2018) | 72.0% → 89.1% | `chapter5Experiments/canonical_pytorch_baselines.py` |
| GRU (2-layer bidir) | 59.9% → 78.4% | `chapter5Experiments/canonical_pytorch_baselines.py` |
| LSTM (2-layer bidir) | 58.0% → 71.1% | `chapter5Experiments/canonical_pytorch_baselines.py` |
| Raw EEG + LogReg | 70.5% → 88.4% | `chapter5Experiments/experiment_zero.py` |
| ARSPI-Net reservoir + LogReg | 59.4% → 78.8% | `chapter5Experiments/experiment_zero.py` |
| PCA-200 + LogReg | 64.9% → 86.4% | (not committed — run `sklearn_baselines.py` on the PCA-200 stack) |
| Band Power + SVM | 47.7% → 61.0% | `chapter5Experiments/sklearn_baselines.py` |

> **Why archived.** The centered columns compute the subject mean using held-out data (transductive), which inflates apparent accuracy relative to a prospective setting. The uncentered columns use a globally fitted PCA basis. Neither matches the corrected fold-local / SRP-64 specification, so these values are descriptive legacy estimates.

Supporting archived analyses: variance decomposition (62.6% subject, 8.7% condition, 28.7% residual, ρ = 7.2×) via `canonical_pytorch_baselines.py --centered`; Experiment Zero disambiguation (confirms 70.5% is uncentered) via `experiment_zero.py`.

### Archived message-passing GNN sweep

| Architecture | Bal. Acc. | Script |
|---|---|---|
| PCA-64 concat + SVM (non-propagated) | 63.4% ± 4.4% (MCC 0.457) | `chapter5Experiments/run_chapter5_experiments.py` |
| GIN (5-seed) | 49.8% ± 3.7% | `chapter5Experiments/run_chapter5_experiments.py` |
| GCN (5-seed) | 56.4% ± 4.3% | `chapter5Experiments/run_chapter5_experiments.py` |
| EdgeConv (5-seed) | 48.5% ± 5.1% | `chapter5Experiments/run_chapter5_experiments.py` |
| Over-smoothing mechanism (Dirichlet energy, cosine sim) | 33% inter-channel variance drop at K=2 | `chapter5Experiments/graph_diffusion_oversmoothing.py` |

Global preprocessing and dependent cross-validation folds mean this sweep cannot establish a universal graph operating boundary.

### Archived 4-class Ch6/Ch7 exploration

| Dissertation Element | Script | Output |
|---|---|---|
| Ch6 ESP / reliability / surrogate / value-add / dissociation / interaction / temporal | `chapter6Experiments/run_chapter6_exp*.py` | Original 4-class exploration |
| Ch7 coupling A–E | `chapter7Experiments/run_chapter7_experiment_*.py` | Original 4-class exploration |
| Layer ablation matrix (A1–A9, C1–C6) | `experiments/ablation/layer_ablation.py` | Full ablation (near-chance clinical readouts) |

The layer-specific clinical screens are exploratory and cohort-level. They yield nominal candidates but no validated biomarkers.

### Deprecated NumPy baselines

`chapter5Experiments/deprecated/eegnet_gru_lstm_baselines.py` contains simplified NumPy reference implementations (EEGNet trains only the FC layer; GRU/LSTM use fixed random recurrent weights). Retained for provenance only; superseded by `canonical_pytorch_baselines.py`.

---

## Data Validation

| Purpose | Script |
|---|---|
| 3-class EEG QC (10 checks) | `validation/validate_shape_data.py` |
| 4-class subcategory QC (12 checks) | `validation/validate_subcategory_data.py` |
| Repo↔dissertation consistency checker | `validation/validate_dissertation_claims.py` |
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
