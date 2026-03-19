# Chapter 4: Temporal Pattern Discrimination (Spike-to-Embedding Pipeline)

This folder contains the experimental scripts and figures for Chapter 4 of the ARSPI-Net dissertation. Chapter 4 validates the core claim of the neuromorphic architecture: that temporal spike coding (BSC6) captures discriminative information that rate coding (MFR) cannot, using a controlled synthetic task where the two stimulus classes have identical total energy but different temporal profiles.

## Experiment Overview

| Experiment | Script | Purpose |
|------------|--------|---------|
| 4.1-4.6 | `run_chapter4_experiments.py` | Complete 6-experiment pipeline: reservoir ablation, FDR analysis, coding comparison, PCA reduction, robustness, parameter sensitivity |
| Observations | `run_chapter4_observations.py` | 6 raw data visualization figures showing the data at every stage of the BSC6-to-PCA-64 pipeline |

---

## The Temporal Pattern Discrimination Task

### Why Synthetic Data?

Chapter 4 uses synthetic data by design. The controlled task isolates the specific question — *does temporal coding capture information that rate coding cannot?* — without confounds from real EEG preprocessing, subject variability, or label noise. If temporal coding cannot outperform rate coding on a task specifically constructed to require temporal information, it will not do so on real data either.

### Task Design

Two stimulus classes with **identical total energy** but **different temporal profiles**:

| Class | Name | Amplitude Profile | Purpose |
|-------|------|-------------------|---------|
| 0 | "Early Burst" | Strong -> Medium -> Weak | Temporal emphasis on early sub-window |
| 1 | "Late Burst" | Weak -> Medium -> Strong | Temporal emphasis on late sub-window |

Both classes have the same total energy (sum of amplitudes) — a rate-based code that collapses temporal structure will see both classes as identical.

### Task Parameters

| Parameter | Value |
|-----------|-------|
| Trials per class | 200 (400 total) |
| Input dimensionality | 33 features (5 active, 28 silent) |
| Simulation length | 150 timesteps |
| Noise (sigma) | 0.5 |
| Amplitude jitter | +/- 30% |
| Timing jitter | +/- 5 steps |
| Sub-pulse duration | 8 steps |
| Sub-pulse centers | Steps 20, 35, 50 |
| Base amplitude | 1.5 |

The jitter and noise make the task genuinely non-trivial — individual trials can look quite different from each other, and the class boundary is not trivially separable in the raw input space (FDR = 0.13).

---

## The LIF Reservoir

### Architecture

| Parameter | Value | Source |
|-----------|-------|--------|
| Reservoir neurons (N_res) | 256 | Experiment 1 ablation |
| Membrane leak (beta) | 0.05 | Chapter 3, Experiment 6 sensitivity |
| Firing threshold (M_th) | 0.5 | Chapter 3, Experiment 6 sensitivity |
| Spectral radius | 0.9 | Controlled via eigenvalue scaling |
| Input weights | Xavier-uniform, (N_res, N_input) | |
| Recurrent weights | Xavier-uniform, (N_res, N_res), scaled to spectral radius | |
| Random seed | 42 (baseline) | Experiment 5 tests 10 seeds |

### Membrane Dynamics

```
V_i(t+1) = (1 - beta) * V_i(t) * (1 - s_i(t)) + W_in @ x(t) + W_rec @ s(t)
s_i(t) = 1 if V_i(t) >= M_th, else 0
V_i(t) = max(V_i(t) - s_i(t) * M_th, 0)    # soft reset with floor
```

The `(1 - s_i(t))` term provides a hard reset on spike: the membrane potential is cleared after firing, preventing sustained high-voltage states.

### Feature Extraction Window

Timesteps 10-70 (out of 150). This window captures the main stimulus-driven response while excluding the initial transient (steps 0-9) and the post-stimulus decay (steps 70+).

---

## Coding Schemes

Four spike coding methods are compared, all operating on the same spike trains:

| Scheme | Method | Dimensions | Temporal? |
|--------|--------|-----------|-----------|
| **MFR** (Mean Firing Rate) | Average spike count per neuron across window | 256 | No |
| **LFS** (Last Frame Snapshot) | Time of first spike per neuron (normalized) | 256 | Partial |
| **BSC3** (Binned Spike Counts, 3 bins) | Divide window into 3 bins, count spikes per neuron per bin | 768 | Yes |
| **BSC6** (Binned Spike Counts, 6 bins) | Divide window into 6 bins, count spikes per neuron per bin | 1,536 | Yes |

**BSC6** is the primary ARSPI-Net coding scheme. It preserves temporal structure by maintaining separate spike counts for 6 contiguous time windows, producing a 256 x 6 = 1,536-dimensional feature vector per trial. PCA then reduces this to 64 dimensions.

**MFR** is the critical control: it collapses all temporal information into a single rate per neuron. On an equal-energy task, MFR should perform at chance (~50%) because both classes produce the same total spike counts.

---

## Experiment 1: Reservoir Size Ablation

### Research Question

How does reservoir size affect classification accuracy? Is N_res = 256 sufficient or does performance improve with larger reservoirs?

### Method

BSC6 + PCA-64 features extracted from reservoirs of size N_res in {64, 128, 256, 512}. Logistic regression with 5-fold stratified CV.

### Verified Results

| N_res | Accuracy |
|-------|----------|
| 64 | 98.0% +/- 2.3% |
| 128 | 99.0% +/- 0.9% |
| **256** | **98.8% +/- 0.8%** |
| 512 | 99.0% +/- 1.2% |

### Interpretation

Performance saturates at N_res = 128. The selected size N_res = 256 is on the plateau — larger reservoirs provide no meaningful benefit. This validates the Chapter 3 parameter choice.

### Output Figure

- `pictures/chLSMEmbeddings/ablation_reservoir_size.pdf`

---

## Experiment 2: FDR Three-Way Comparison

### Research Question

Does the LIF reservoir enhance class separability compared to raw input or a simple linear filter? The Fisher Discriminant Ratio (FDR = trace(S_W^{-1} S_B)) quantifies multivariate class separation.

### Method

Three representations of the same data are compared:
1. **Raw Input (Binned):** Direct BSC6-style binning of the raw input signal (no reservoir)
2. **Linear Filter (5-pt MA):** 5-point moving average smoothing, then BSC6-style binning
3. **LSM Reservoir (BSC6):** Full LIF reservoir processing, then BSC6 feature extraction

FDR is computed with Tikhonov regularization (epsilon = 1e-4 * trace(S_W) / d) to handle rank-deficient covariance matrices.

### Verified Results

| Representation | FDR |
|---------------|-----|
| Raw Input (Binned) | 0.131 |
| Linear Filter (5-pt MA) | 0.127 |
| **LSM Reservoir (BSC6)** | **1158.008** |

**The LSM reservoir achieves ~8,840x higher FDR than raw input.** The linear filter provides no improvement over raw input, confirming that the separability enhancement is due to the nonlinear reservoir dynamics, not simple smoothing.

### Interpretation

The reservoir's nonlinear integrate-and-fire dynamics transform a weakly separable input space (FDR = 0.13) into a highly separable embedding space (FDR = 1158). This is the core mechanism that makes neuromorphic processing valuable: the reservoir acts as a nonlinear kernel that amplifies temporal structure into separable spatial patterns.

### Output Figure

- `pictures/chLSMEmbeddings/fdr_three_way_comparison.pdf`

---

## Experiment 3: Neural Coding Scheme Comparison

### Research Question

Which spike coding scheme best preserves the temporal information needed for classification? This is the chapter's central experiment.

### Method

All four coding schemes are extracted from the same set of spike trains (N_res = 256, seed = 42). Two classifiers are tested: Logistic Regression (linear) and SVM-RBF (nonlinear).

### Verified Results

| Coding | LogReg | SVM-RBF |
|--------|--------|---------|
| MFR | 47.5% +/- 4.1% | 51.2% +/- 4.9% |
| LFS | 70.0% +/- 4.4% | 70.8% +/- 3.3% |
| BSC3 | 98.3% +/- 1.3% | 98.3% +/- 1.3% |
| **BSC6** | **99.5% +/- 0.6%** | **99.5% +/- 0.6%** |

### Interpretation

**MFR at ~50% (chance) is the chapter's key result.** Both classes produce identical mean firing rates (equal energy), so collapsing temporal structure destroys all discriminative information. This proves that temporal coding is necessary — not just helpful — for this class of temporal discrimination tasks.

**LFS at 70%** captures partial temporal information (time of first spike) but loses the multi-bin structure. **BSC3 at 98%** and **BSC6 at 99.5%** show that finer temporal binning progressively recovers the full temporal profile.

The LogReg and SVM-RBF results are nearly identical for BSC6, indicating the separability is linear — the reservoir does the nonlinear work, and a simple linear readout suffices.

### Output Figure

- `pictures/chLSMEmbeddings/coding_scheme_accuracy_comparison.pdf`

---

## Experiment 4: PCA Dimensionality Reduction

### Research Question

BSC6 produces 1,536-dimensional feature vectors. How much can PCA compress them without losing discriminative information?

### Method

PCA is fitted on the full BSC6 feature matrix (400 x 1536). Classification accuracy is evaluated for component counts in {5, 10, 20, 32, 64, 128, 256} plus the full 1536 dimensions.

### Verified Results

| PCA Components | Accuracy | Cumulative Variance |
|---------------|----------|-------------------|
| 5 | 99.5% +/- 0.6% | 15.4% |
| 10 | 99.5% +/- 0.6% | 19.9% |
| 20 | 98.8% +/- 0.8% | 27.5% |
| 32 | 98.8% +/- 0.8% | 35.1% |
| **64** | **98.5% +/- 1.2%** | **50.5%** |
| 128 | 96.7% +/- 2.2% | 70.1% |
| 256 | 86.8% +/- 3.2% | 90.7% |
| Full (1536) | 99.5% +/- 0.6% | 100% |

### Interpretation

Discriminative information is concentrated in the first few PCs — just 5 components achieve 99.5% accuracy despite capturing only 15.4% of total variance. This means the class-relevant structure occupies a low-dimensional subspace of the BSC6 feature space.

The selected PCA-64 achieves 98.5% accuracy — a negligible drop from full BSC6 while reducing dimensionality by 24x (1536 -> 64). Higher component counts (128, 256) actually degrade performance, likely because they introduce noisy dimensions that hurt the regularized classifier.

### Output Figures

- `pictures/chLSMEmbeddings/pca_explained_variance.pdf` — Cumulative variance curve with PCA-64 cutoff annotated
- `pictures/chLSMEmbeddings/pca_component_visualization.pdf` — Temporal loading profiles of the top 4 PCs

---

## Experiment 5: Cross-Initialization Robustness

### Research Question

Are the results stable across different random weight initializations, or do they depend on a lucky seed?

### Method

10 independent reservoir initializations (seeds: 13, 20, 27, 34, 41, 48, 55, 62, 69, 76). For each seed, BSC6+PCA-64, BSC3, and MFR features are extracted and classified.

### Verified Results

| Coding | Mean Accuracy | Std | Min | Max |
|--------|-------------|-----|-----|-----|
| **BSC6+PCA-64** | **98.6%** | **0.7%** | 97.5% | 100.0% |
| BSC3 | 99.1% | 0.4% | 98.3% | 99.5% |
| MFR | 50.8% | 2.7% | 46.8% | 54.5% |

### Interpretation

BSC6+PCA-64 is highly robust: all 10 seeds achieve >97.5% accuracy. The standard deviation (0.7%) is smaller than the classification CV variance, confirming that results are insensitive to weight initialization.

MFR remains at chance (~50%) across all seeds — confirming that the MFR failure is structural (equal-energy classes) rather than a property of a specific initialization.

### Output Figure

- `pictures/chLSMEmbeddings/cross_initialization_robustness.pdf` — Box plots with individual seed points

---

## Experiment 6: Parameter Sensitivity

### Research Question

How sensitive is classification accuracy to the LIF parameters beta (membrane leak) and M_th (firing threshold)?

### Method

Grid search over beta in {0.03, 0.04, 0.05, 0.06, 0.07, 0.08} and M_th in {0.3, 0.4, 0.5, 0.6, 0.7}. BSC6+PCA-64 features, logistic regression, 5-fold CV.

### Verified Results

Accuracy ranges from **97.7% to 99.8%** across all 30 parameter combinations. The selected parameters (beta=0.05, M_th=0.5) achieve 98.5% — solidly on the plateau.

| | M_th=0.3 | M_th=0.4 | M_th=0.5 | M_th=0.6 | M_th=0.7 |
|---|---------|---------|---------|---------|---------|
| beta=0.03 | 99.0 | 98.8 | 99.8 | 98.8 | 99.5 |
| beta=0.04 | 99.0 | 98.8 | 97.7 | 98.0 | 98.5 |
| **beta=0.05** | 98.5 | 98.0 | **98.5** | 97.8 | 98.2 |
| beta=0.06 | 99.0 | 99.2 | 98.8 | 99.0 | 98.5 |
| beta=0.07 | 98.8 | 98.8 | 97.8 | 99.0 | 98.5 |
| beta=0.08 | 98.8 | 98.5 | 98.3 | 98.5 | 99.0 |

### Interpretation

The parameter landscape is remarkably flat — all combinations exceed 97.7% accuracy. This means the reservoir's temporal coding capability is robust to parameter choice, not a fragile property of a specific operating point. The Chapter 3 parameters (beta=0.05, M_th=0.5) are validated as a safe operating point within a broad high-performance region.

### Output Figure

- `pictures/chLSMEmbeddings/parameter_sensitivity_heatmap.pdf` — Color-coded accuracy grid with selected parameters highlighted

---

## Raw Observation Figures

The observation script generates 6 figures that visualize the data at every stage of the BSC6-to-PCA-64 pipeline, before any classification is performed:

| Figure | File | What It Shows |
|--------|------|--------------|
| Obs 1 | `obs01_raw_input_signals.pdf` | Raw input stimuli for both classes (3 trials each). Shading shows temporal amplitude profile difference. |
| Obs 2 | `obs02_raw_spike_rasters.pdf` | LIF reservoir spike rasters (80 of 256 neurons). Green shading marks the feature extraction window (steps 10-70). |
| Obs 3 | `obs03_raw_bsc6_features.pdf` | BSC6 feature matrices (256 neurons x 6 bins) for individual trials + class means + class difference map. |
| Obs 4 | `obs04_raw_embedding_space.pdf` | PCA-2 projections of BSC6 (clear linear separability) vs MFR (complete class overlap). Plus PCA eigenspectrum. |
| Obs 5 | `obs05_population_dynamics.pdf` | Population firing rates (class-averaged), sparsity distributions, and per-neuron sorted rates — shows why rate coding fails. |
| Obs 6 | `obs06_membrane_dynamics.pdf` | Membrane potential heatmaps and individual neuron traces — shows integrate-and-fire dynamics during stimulus processing. |

---

## Verification

Both scripts were executed to verify all claims. The verification confirms:

1. **BSC6 achieves 99.5% accuracy** (Experiment 3) — above the >90% stated in the root README
2. **MFR achieves 47.5-51.2%** (Experiment 3) — at chance, confirming temporal coding is necessary
3. **FDR ratio is ~8,840x** (Experiment 2) — above the ~6-7x stated in the root README (the README's conservative estimate refers to earlier task parameters)
4. **Results are stable across 10 random seeds** (Experiment 5) — BSC6 range: 97.5-100.0%
5. **Results are stable across 30 parameter combinations** (Experiment 6) — minimum accuracy 97.7%
6. **All 13 figures generated successfully** — 7 experimental + 6 observational

### Running Verification

```bash
# Run all 6 experiments and generate 7 figures (~2-3 minutes):
python chapter4Experiments/run_chapter4_experiments.py

# Run observation pipeline and generate 6 figures (~1-2 minutes):
python chapter4Experiments/run_chapter4_observations.py [--output_dir pictures/chLSMEmbeddings]
```

**No external data required** — both scripts generate synthetic data internally.

**Dependencies:** `numpy`, `scipy`, `scikit-learn`, `matplotlib`

### Verification Checklist

| Claim | Experiment | Expected | Verified |
|-------|-----------|----------|----------|
| BSC6 > 90% accuracy | Exp 3 | > 90% | 99.5% |
| MFR ~ 50% (chance) | Exp 3 | ~ 50% | 47.5% (LogReg), 51.2% (SVM) |
| FDR: LSM >> Raw | Exp 2 | > 6x | 8,840x (1158 vs 0.13) |
| N_res = 256 sufficient | Exp 1 | Plateau by 256 | 98.8% (256) vs 99.0% (512) |
| PCA-64 preserves accuracy | Exp 4 | Minimal drop from full | 98.5% (PCA-64) vs 99.5% (full) |
| Robust across seeds | Exp 5 | Low variance | 98.6% +/- 0.7% over 10 seeds |
| Robust across parameters | Exp 6 | Broad plateau | 97.7-99.8% across 30 combinations |
| Rate coding fails on equal-energy | Exp 3+5 | MFR ~ chance | 50.8% +/- 2.7% across 10 seeds |

---

## Shared Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Reservoir neurons | 256 | Experiment 1 ablation |
| Leak rate (beta) | 0.05 | Chapter 3, Experiment 6 validation |
| Spike threshold (M_th) | 0.5 | Chapter 3, Experiment 6 validation |
| Spectral radius | 0.9 | Eigenvalue scaling |
| BSC temporal bins | 6 | Experiment 3 comparison |
| PCA components | 64 | Experiment 4 analysis |
| Feature window | Steps 10-70 | Post-transient, pre-decay |
| CV folds | 5 | Stratified k-fold |
| LogReg regularization | C=0.1, solver=liblinear | |
| SVM kernel | RBF, C=1.0 | |
| Random seed | 42 (baseline) | Experiment 5 tests 10 seeds |
| Figure DPI | 300 | Publication quality |
| Figure format | PDF | Vector graphics |

## Files

### Scripts

| File | Lines | Description |
|------|-------|-------------|
| `run_chapter4_experiments.py` | 609 | Complete 6-experiment pipeline: LIF reservoir, 4 coding schemes, FDR analysis, size ablation, PCA reduction, robustness, parameter sensitivity. Generates 7 publication figures. |
| `run_chapter4_observations.py` | 582 | Raw data observation pipeline: generates 6 figures visualizing every stage of the BSC6-to-PCA-64 embedding, from input stimuli through spike rasters to embedding geometry. |

### Output Figures (`pictures/chLSMEmbeddings/`)

| File | Experiment | Description |
|------|-----------|-------------|
| `ablation_reservoir_size.pdf` | Exp 1 | Bar chart: accuracy vs N_res in {64, 128, 256, 512} |
| `fdr_three_way_comparison.pdf` | Exp 2 | Bar chart: FDR for Raw / Linear Filter / LSM |
| `coding_scheme_accuracy_comparison.pdf` | Exp 3 | Grouped bar chart: MFR / LFS / BSC3 / BSC6 x LogReg / SVM |
| `pca_explained_variance.pdf` | Exp 4 | Cumulative variance curve with PCA-64 cutoff |
| `pca_component_visualization.pdf` | Exp 4 | Temporal loading profiles of top 4 PCs |
| `cross_initialization_robustness.pdf` | Exp 5 | Box plots with individual seed points for 3 coding schemes |
| `parameter_sensitivity_heatmap.pdf` | Exp 6 | Color-coded accuracy grid: beta x M_th |
| `obs01_raw_input_signals.pdf` | Obs | Raw input stimuli for both classes |
| `obs02_raw_spike_rasters.pdf` | Obs | LIF reservoir spike rasters (80 neurons) |
| `obs03_raw_bsc6_features.pdf` | Obs | BSC6 feature matrices with class difference |
| `obs04_raw_embedding_space.pdf` | Obs | PCA-2 projections: BSC6 (separable) vs MFR (overlap) |
| `obs05_population_dynamics.pdf` | Obs | Population firing rates, sparsity, per-neuron rates |
| `obs06_membrane_dynamics.pdf` | Obs | Membrane potential heatmaps and neuron traces |

---

## Relationship to Other Chapters

- **Chapter 3** established the LIF reservoir parameters through systematic characterization. Chapter 4 validates these choices on a controlled discrimination task.
- **Chapter 4** proves that temporal coding (BSC6) is necessary for temporal pattern discrimination. This is the foundational result for all subsequent chapters.
- **Chapter 5** applies the BSC6+PCA-64 pipeline validated here to real clinical EEG data (SHAPE Community dataset), adding graph-structured classification.
- **Chapter 6** uses the same reservoir to characterize dynamical properties (Lyapunov exponents, permutation entropy, relaxation time) per affective condition.
- **Chapter 7** couples Chapter 5's topological descriptors with Chapter 6's dynamical descriptors — both built on the reservoir validated here.

## Sample

- 400 synthetic trials (200 per class)
- No external data requirements
- Fully deterministic with seed=42
- Runtime: ~2-3 minutes for experiments, ~1-2 minutes for observations
