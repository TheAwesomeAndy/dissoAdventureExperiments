# ARSPI-Net: Hybrid Neuromorphic Affective Computing for EEG

**Publication:** Lane, A. (2026). *ARSPI-Net: Hybrid Neuromorphic Affective Computing Architecture for EEG Signal Processing.* PhD Dissertation.

This repository contains the complete experimental pipeline for Chapters 4–6 of the ARSPI-Net dissertation. The core innovation is a **Leaky Integrate-and-Fire (LIF) spiking neural network reservoir** that transforms continuous EEG signals into sparse binary spike codes, capturing temporal structure that rate-based approaches miss. These spike codes are then classified using Graph Neural Networks (GNNs) that exploit spatial electrode topology.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Architecture Overview](#architecture-overview)
3. [Dependencies](#dependencies)
4. [Input Data Format](#input-data-format)
5. [Adapting to Your Own EEG Data](#adapting-to-your-own-eeg-data)
6. [Preprocessing Pipeline](#preprocessing-pipeline)
7. [Feature Extraction](#feature-extraction)
8. [Chapter 4: Temporal Pattern Discrimination](#chapter-4-temporal-pattern-discrimination)
9. [Chapter 5: Clinical EEG Classification](#chapter-5-clinical-eeg-classification)
10. [Chapter 6: Dynamical Characterization](#chapter-6-dynamical-characterization)
11. [Data Validation and QC](#data-validation-and-quality-control)
12. [Output Files](#output-files)
13. [Key Parameters Reference](#key-parameters-reference)
14. [Troubleshooting and Notes](#troubleshooting-and-notes)

---

## Repository Structure

```
dissoAdventureExperiments/
├── README.md                          # This file
├── run_chapter4_experiments.py        # Ch4: Temporal coding experiments (synthetic data)
├── run_chapter4_observations.py       # Ch4: Raw observation figures (6 figures)
├── run_chapter5_experiments.py        # Ch5: Full EEG classification pipeline with GNN
├── reproduce_chapter5.py              # Ch5: Standalone reproducibility pipeline
├── reproduce_chapter6.py              # Ch6: Standalone reproducibility pipeline
├── validate_shape_data.py             # QC for broad-condition SHAPE data (3 conditions)
├── validate_subcategory_data.py       # QC for fine-grained subcategory data (4 categories)
└── chapter6Experiments/
    └── run_chapter6_exp1_esp.py       # Ch6 Exp 6.1: Echo State Property verification
```

---

## Architecture Overview

The ARSPI-Net pipeline has four stages:

```
Raw EEG (continuous µV)
  → Preprocessing (baseline removal, downsampling, z-score)
    → LIF Reservoir (per-channel spiking neural network → binary spike trains)
      → Temporal Feature Extraction (BSC₆ binned spike counts + PCA)
        → Classification (GNN with spatial electrode graph, or SVM/MLP)
```

**Why spiking?** Traditional EEG features (band power, Hjorth parameters) collapse temporal dynamics into summary statistics. The LIF reservoir preserves *when* neural events occur, not just *how much* activity there is. The key experiment (Chapter 4) shows that rate-based coding (Mean Firing Rate) fails at ~50% accuracy on temporal discrimination tasks, while Binned Spike Counts (BSC₆) achieve >90%.

---

## Dependencies

Install all requirements:

```bash
pip install numpy scipy scikit-learn matplotlib pandas openpyxl
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, linear algebra, all LIF/GNN implementations |
| `scipy` | Signal processing (`decimate`, `welch`), statistics (`ttest_rel`, `kruskal`, `mannwhitneyu`) |
| `scikit-learn` | Classifiers (SVM, LogReg, MLP), PCA, cross-validation, metrics |
| `matplotlib` | Figure generation (all output is PDF at 300 dpi) |
| `pandas` | DataFrame operations, metadata cross-referencing |
| `openpyxl` | Reading `.xlsx` clinical labels |

**No deep learning frameworks required.** All GNN implementations (GCN, GraphSAGE, GAT) are written from scratch in NumPy.

---

## Input Data Format

This project was built for the **SHAPE Community Emotional Interrupt Task** dataset. If you have different EEG data, you will need to adapt the loading code. Here is exactly what the code expects:

### Broad-Condition EEG Files

**Filename pattern:** `SHAPE_Community_{SUBJECT_ID}_{CONDITION}_BC.txt`

- `{SUBJECT_ID}`: 3-digit zero-padded subject number (e.g., `007`, `042`, `127`)
- `{CONDITION}`: One of `IAPSNeg`, `IAPSNeu`, `IAPSPos`
- `_BC`: Indicates baseline-corrected data

**Example filenames:**
```
SHAPE_Community_007_IAPSNeg_BC.txt
SHAPE_Community_007_IAPSNeu_BC.txt
SHAPE_Community_007_IAPSPos_BC.txt
```

**File contents:**
- **Format:** Space-separated plain text, no header row
- **Dimensions:** 1229 rows × 34 columns
- **Units:** Microvolts (µV)
- **Sampling rate:** 1024 Hz
- **Time structure:**
  - Rows 0–204 (205 samples): Baseline period (200 ms), already baseline-corrected to ~0
  - Rows 205–1228 (1024 samples): Post-stimulus period (1000 ms)
- **Columns:** 34 EEG channels in this order:

```
Col 0:  Fp1    Col 1:  Fp2    Col 2:  F7     Col 3:  F3     Col 4:  Fz
Col 5:  F4     Col 6:  F8     Col 7:  FC5    Col 8:  FC1    Col 9:  FC2
Col 10: FC6    Col 11: T7     Col 12: C3     Col 13: Cz     Col 14: C4
Col 15: T8     Col 16: CP5    Col 17: CP1    Col 18: CP2    Col 19: CP6
Col 20: P7     Col 21: P3     Col 22: Pz     Col 23: P4     Col 24: P8
Col 25: PO7    Col 26: PO3    Col 27: POz    Col 28: PO4    Col 29: PO8
Col 30: O1     Col 31: Oz     Col 32: O2     Col 33: REF
```

**File organization:** Data is delivered in ZIP batches (`batch1.zip`, `batch2.zip`, `batch3.zip`), each containing the `.txt` files directly.

### Subcategory EEG Files (Chapter 6, Experiment 6.1)

**Filename pattern:** `SHAPE_Community_{SUBJECT_ID}_IAPS{VALENCE}_{CATEGORY}_BC.txt`

- `{VALENCE}`: `Neg` or `Pos`
- `{CATEGORY}`: `Threat`, `Mutilation`, `Cute`, or `Erotic`

**Example:** `SHAPE_Community_042_IAPSNeg_Threat_BC.txt`

Same file structure (1229 × 34), but organized in category directories (`categoriesbatch1/`, `categoriesbatch2/`, etc.).

### Metadata Files

| File | Format | Purpose |
|------|--------|---------|
| `SHAPE_Community_Andrew_Psychopathology.xlsx` | Excel | Clinical diagnoses: columns `ID`, `MDD` (0/1), `Depression_Type`, `GAD`, `PTSD`, etc. |
| `ParticipantInfo.csv` | CSV | Demographic information |

---

## Adapting to Your Own EEG Data

If your data is in a different format (e.g., `.edf`, `.set`, `.fif`, MNE-Python objects, BIDS format), you need to produce the same intermediate representation. Here is what to do:

### Step 1: Produce Per-Subject, Per-Condition Epoch Matrices

Each experiment expects a **trial-averaged, baseline-corrected ERP matrix** of shape `(time_samples, n_channels)` per subject per condition. If you have single-trial data, average across trials first.

### Step 2: Match the Expected Dimensions

The code expects `(1229, 34)` — but this is specific to the SHAPE dataset. You have two options:

**Option A: Reshape your data to match (recommended for minimal code changes)**

1. Ensure your data is at **1024 Hz** (resample if needed)
2. Include a **200 ms baseline** (205 samples at 1024 Hz) followed by **1000 ms post-stimulus** (1024 samples)
3. Total: 1229 samples per epoch
4. If you have a different number of channels, update the channel count everywhere (search for `34` in the scripts)

**Option B: Modify the loading code to match your data**

The key loading function in each script follows this pattern:

```python
# In reproduce_chapter5.py / run_chapter5_experiments.py:
data = np.loadtxt(filepath)           # → (1229, 34)
data = data[205:]                     # Remove baseline → (1024, 34)
data = decimate(data, 4, axis=0)      # Downsample 1024→256 Hz → (256, 34)
```

To adapt, write your own loader that returns `(N_timepoints_after_downsample, N_channels)`. The critical output shape the reservoir sees is `(256, N_channels)` — a 256-timestep signal per channel.

### Step 3: Update File Discovery

The scripts discover files using `glob` patterns. Modify the glob pattern to match your naming:

```python
# Original (in reproduce_chapter5.py):
files = glob.glob(os.path.join(data_dir, 'SHAPE_Community_*_IAPS*_BC.txt'))

# Your adaptation, e.g. for BIDS:
files = glob.glob(os.path.join(data_dir, 'sub-*/eeg/sub-*_task-*_eeg.txt'))
```

### Step 4: Update Subject/Condition Parsing

Subject IDs and conditions are parsed from filenames with regex:

```python
# Original parsing:
base = os.path.basename(f)   # "SHAPE_Community_042_IAPSNeg_BC.txt"
parts = base.split('_')
subj_id = parts[2]           # "042"
condition = parts[3]          # "IAPSNeg"
```

Replace this with your own parsing logic. The code needs:
- A **subject identifier** (string or int) — used for subject-stratified cross-validation
- A **condition label** (string) — used as the classification target

### Step 5: Update Electrode Positions (for GNN spatial graph)

Chapter 5 constructs a spatial adjacency graph from 3D electrode coordinates. These are hardcoded in `run_chapter5_experiments.py` (approx. lines 410–455) and `reproduce_chapter5.py`. If you have different channels:

```python
# Original 34-channel positions (approximate 10-20 montage, unit sphere):
CHANNEL_POSITIONS = {
    'Fp1': (-0.31, 0.95, 0.03),
    'Fp2': (0.31, 0.95, 0.03),
    'F7':  (-0.81, 0.59, -0.04),
    # ... etc for all 34 channels
}
```

Replace with your electrode positions. The GNN builds a k-nearest-neighbors graph (default k=5) from these coordinates.

### Step 6: Update Channel Count References

Search all scripts for the number `34` and replace with your channel count. Key locations:
- Array shapes: `(1229, 34)`, `(1024, 34)`, `(256, 34)`
- Channel loops: `for ch in range(34)`
- Reservoir instantiation: one reservoir per channel
- Electrode position arrays
- Validation expected shape: `EXPECTED_SHAPE = (1229, 34)`

### Quick Checklist for Custom Data

| Item | What to change |
|------|---------------|
| File format | `np.loadtxt(...)` → your loader (MNE, EDF, etc.) |
| Filename pattern | `glob.glob(...)` pattern and regex parsing |
| Subject ID extraction | Filename parsing logic |
| Condition label extraction | Filename parsing logic |
| Sampling rate | Adjust `decimate` factor or resample to 256 Hz |
| Baseline samples | Change `205` to your baseline length |
| Number of channels | Replace `34` everywhere |
| Electrode positions | Update 3D coordinates for GNN graph |
| Clinical labels (optional) | Provide your own metadata mapping subject → diagnosis |

---

## Preprocessing Pipeline

All scripts apply identical preprocessing:

```
1. Load:       np.loadtxt(file)              → (1229, 34)
2. Baseline:   data = data[205:]             → (1024, 34)  # discard 200ms baseline
3. Downsample: decimate(data, 4, axis=0)     → (256, 34)   # 1024 Hz → 256 Hz
4. Normalize:  z-score per epoch per channel  → (256, 34)   # mean=0, std=1
```

The result is an array of shape `(N_subjects * N_conditions, 256, 34)` ready for feature extraction.

---

## Feature Extraction

### Conventional EEG Features

| Feature | Method | Dimensions per channel |
|---------|--------|----------------------|
| **Band power** (5 bands) | Welch PSD: delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–100 Hz) | 5 |
| **Hjorth parameters** (3) | Activity (variance), Mobility (√(var(dx)/var(x))), Complexity | 3 |
| **Combined** | Concatenation | 8 |

Output shape: `(N, 34, 8)` — flattened to `(N, 272)` for non-graph classifiers.

### LSM Spiking Features

**LIF Reservoir** (one independent reservoir per channel):
- 256 neurons, β=0.05 (leak), threshold=0.5, spectral radius=0.9
- Xavier-uniform weight initialization
- Each channel gets a unique random seed: `seed + ch * 17`

**LIF neuron dynamics per timestep:**
```
I_total[t]  = W_in * x[t] + W_rec @ spikes[t-1]
mem[t]      = (1 - β) * mem[t-1] * (1 - spikes[t-1]) + I_total[t]
spikes[t]   = 1 if mem[t] >= threshold else 0
mem[t]      = mem[t] - spikes[t] * threshold    # reset by subtraction
mem[t]      = max(mem[t], 0)                     # non-negative constraint
```

**Output coding schemes:**

| Scheme | Description | Dimensions per channel |
|--------|-------------|----------------------|
| **BSC₆** (Binned Spike Counts) | Divide spike train into 6 time bins, count spikes per neuron per bin | 6 × 256 = 1536, then PCA → 64 |
| **MFR** (Mean Firing Rate) | Average spike count per neuron | 256 |
| **BSC₃** | Same as BSC₆ but with 3 bins | 3 × 256 = 768 |
| **LFS** (Last Frame Snapshot) | Spike state at final timestep | 256 |

**Feature extraction window:** Timesteps 10–70 (out of 256) by default, capturing the main ERP components.

---

## Chapter 4: Temporal Pattern Discrimination

**Purpose:** Prove that temporal coding (BSC₆) captures information that rate coding (MFR) cannot.

**Data:** Synthetic — no EEG data required. The script generates 200 trials per class of two stimulus patterns with identical total energy but different temporal profiles:
- Class 0 ("Early burst"): Strong → Medium → Weak
- Class 1 ("Late burst"): Weak → Medium → Strong

Both have noise (σ=0.5), amplitude jitter (±30%), and timing jitter (±5 steps).

### Running Chapter 4

```bash
# Experimental results (7 experiments, all figures):
python chapter4Experiments/run_chapter4_experiments.py

# Raw observation figures (6 detailed visualization figures):
python chapter4Experiments/run_chapter4_observations.py [--output_dir pictures/chLSMEmbeddings]
```

**No command-line arguments required** — Chapter 4 uses synthetic data only.

### Chapter 4 Experiments

| # | Experiment | What it tests |
|---|-----------|---------------|
| 1 | Reservoir size ablation | Accuracy vs. N_res ∈ {64, 128, 256, 512} |
| 2 | FDR three-way comparison | Fisher Discriminant Ratio: Raw vs. Linear Filter vs. LSM |
| 3 | Coding scheme comparison | MFR, LFS, BSC₃, BSC₆ — accuracy comparison |
| 4 | PCA dimensionality reduction | Cumulative variance and component visualization |
| 5 | Cross-initialization robustness | 10 random seeds → box plots |
| 6 | Parameter sensitivity | β × M_th grid heatmap |

### Chapter 4 Key Results
- **BSC₆ + PCA-64:** >90% accuracy
- **MFR:** ~50% (complete failure — proves temporal coding is necessary)
- **FDR:** LSM features have ~6–7× better class separability than raw input

---

## Chapter 5: Clinical EEG Classification

**Purpose:** Classify affective EEG responses (Negative/Neutral/Pleasant) on the SHAPE Community dataset using the full ARSPI-Net pipeline.

### Running Chapter 5

```bash
# Full experimental pipeline (7-row baseline table + GNN experiments):
python chapter5Experiments/run_chapter5_experiments.py \
    --data_dir /path/to/shape_eeg_files/ \
    [--demo]  # optional: run with synthetic data for testing

# Standalone reproducibility script:
python chapter5Experiments/reproduce_chapter5.py \
    --data_dir /path/to/shape_eeg_files/ \
    --labels /path/to/SHAPE_Community_Andrew_Psychopathology.xlsx \
    --output_dir ./figures/ch5/
```

**`--demo` mode:** If you don't have the SHAPE dataset, `chapter5Experiments/run_chapter5_experiments.py --demo` generates synthetic data to verify the pipeline runs end-to-end.

### Chapter 5 Experiments: The 7-Row Baseline Table

| Row | Features | Classifier | Graph? | Purpose |
|-----|----------|-----------|--------|---------|
| 1 | BandPower | LogReg | No | Baseline |
| 2 | BandPower | MLP | No | Nonlinear baseline |
| 3 | LSM-BSC₆-PCA64 | MLP | No | LSM benefit without graph |
| 4 | BandPower | GAT | Spatial | Graph benefit on conventional features |
| 5 | **LSM-BSC₆-PCA64** | **GAT** | **Spatial** | **Full ARSPI-Net (best)** |
| 6 | LSM-BSC₆-PCA64 | GAT | Functional | Functional adjacency variant |
| 7 | LSM-MFR | GAT | Spatial | Rate-based variant (expected worse) |

### GNN Details

All GNN implementations are **from-scratch NumPy** (no PyTorch/TF):

- **GCN:** `H' = D^{-1/2} A_tilde D^{-1/2} @ H @ W` (symmetric normalized message passing)
- **GraphSAGE:** Concatenate self-features with mean-aggregated neighbor features
- **GAT:** Multi-head attention-weighted neighbor aggregation (softmax attention)

**Graph construction:**
- **Spatial adjacency:** k-nearest neighbors (k=5) from 3D electrode positions
- **Functional adjacency:** Feature correlation matrix, thresholded at 75th percentile

**Validation:** Subject-stratified 10-fold cross-validation (`StratifiedGroupKFold`) — all conditions from one subject stay in the same fold to prevent data leakage.

---

## Chapter 6: Dynamical Characterization

**Purpose:** Characterize the reservoir's dynamical properties — prove it operates in a stable driven regime and show how dynamics differ across affective conditions.

### Running Chapter 6

```bash
# Experiment 6.1: Echo State Property verification (requires subcategory data):
python chapter6Experiments/run_chapter6_exp1_esp.py \
    --category-dirs categoriesbatch1 categoriesbatch2 categoriesbatch3 categoriesbatch4 \
    --output-dir results/ \
    [--channels 0 8 16 24 33] \
    [--esp-subjects 30] \
    [--seed 42]

# Full Chapter 6 reproducibility (requires Chapter 5 features):
python reproduce_chapter6.py \
    --features shape_features.pkl \
    --labels SHAPE_Community_Andrew_Psychopathology.xlsx \
    --output_dir ./figures/ch6/
```

### Chapter 6 Experiments

| # | Experiment | What it measures |
|---|-----------|-----------------|
| 1 | ESP Convergence | Two trajectories from different initial conditions converge under same input |
| 2 | Driven Lyapunov Exponent (λ₁) | Benettin renormalization → λ₁ < 0 confirms stable contraction |
| 3 | Per-condition Φ (sparse coding efficiency) | Var[population_rate] / mean_activity |
| 4 | Permutation Entropy (H_π) | Order-based entropy of PC1 of membrane potentials |
| 5 | Relaxation time (τ_relax) | Steps to 1/e decay after stimulus peak |
| 6 | Autocorrelation decay (τ_ac) | Lag to 1/e autocorrelation |
| 7 | Sliding window classification | 50 ms windows → time-resolved accuracy curve |
| 8 | ERP-motivated windows | P1/N1 (~100ms), EPN (~250ms), LPP (~400–800ms) |
| 9 | HC vs MDD dynamical differences | Clinical group comparison |
| 10 | Surrogate testing | Phase-randomized controls |

### Chapter 6 Key Results
- **λ₁ = -0.054 ± 0.0001** — 100% negative across 4,220 measurements
- The reservoir is uniformly contracting: ESP is verified
- Affective conditions produce distinct dynamical signatures

---

## Data Validation and Quality Control

Run these **before** any experiments to verify data integrity.

### Broad-Condition Validation (3 conditions per subject)

```bash
python validate_shape_data.py \
    --batch1 /path/to/batch1.zip \
    --batch2 /path/to/batch2.zip \
    --batch3 /path/to/batch3.zip \
    --participant_info /path/to/ParticipantInfo.csv \
    --psychopathology /path/to/Psychopathology.xlsx \
    --output_dir qc_output/
```

**10 automated checks:**

| Check | What it verifies |
|-------|-----------------|
| 1. File inventory | Naming convention matches `SHAPE_Community_XXX_IAPSYYY_BC.txt` |
| 2. Dimensional consistency | Every file is exactly 1229 × 34 |
| 3. Subject completeness | Each subject has all 3 conditions |
| 4. Numerical integrity | No NaN, Inf, or non-numeric values |
| 5. Amplitude range | Values within ±500 µV (warn at ±200 µV) |
| 6. Flat/dead channels | Variance > 0.01 µV² per channel |
| 7. Extreme outliers | No values > 5 SD from channel mean |
| 8. Baseline period | First 205 rows near zero (mean < 5 µV) |
| 9. Cross-batch duplicates | No duplicate subjects across ZIP batches |
| 10. Clinical cross-reference | Subject IDs match psychopathology database |

### Subcategory Validation (4 categories per subject)

```bash
python validate_subcategory_data.py \
    --category-dirs categoriesbatch1 categoriesbatch2 categoriesbatch3 categoriesbatch4 \
    --broad-zips batch1.zip batch2.zip batch3.zip \
    --output validation_report.txt
```

---

## Output Files

### Figures (all PDF, 300 dpi)

**Chapter 4** (saved to `pictures/chLSMEmbeddings/`):
- `obs01_raw_input_signals.pdf` — Example trials from both classes
- `obs02_raw_spike_rasters.pdf` — Spike raster plots
- `obs03_raw_bsc6_features.pdf` — BSC₆ feature heatmaps
- `obs04_raw_embedding_space.pdf` — PCA projections (BSC₆ vs MFR)
- `obs05_population_dynamics.pdf` — Population rate and sparsity
- `obs06_membrane_dynamics.pdf` — Membrane potential heatmaps
- `ablation_reservoir_size.pdf`, `fdr_three_way_comparison.pdf`, `coding_scheme_accuracy_comparison.pdf`, `pca_explained_variance.pdf`, `cross_initialization_robustness.pdf`, `parameter_sensitivity_heatmap.pdf`

**Chapter 5** (saved to `--output_dir`, default `./figures/ch5/`):
- `baseline_comparison.pdf` — 7-row bar chart
- `architecture_comparison.pdf` — GCN vs GraphSAGE vs GAT
- `sparsity_sweep.pdf` — k ∈ {3, 5, 7, 10, 15} vs accuracy
- `depth_ablation.pdf` — GNN depth (1–4 layers) vs accuracy
- `confusion_matrix.pdf` — 3×3 heatmap
- `raw_data/*.pdf` — Spike rasters, phase portraits, etc.

**Chapter 6** (saved to `--output_dir`, default `./figures/ch6/`):
- `sliding_window_50ms.pdf`, `condition_phi_tau.pdf`, `esp_gate.pdf`
- `raw_data/lyapunov_convergence.pdf`, `raw_data/dimensionality.pdf`

### Data Files

| File | Contents |
|------|----------|
| `ch5_all_results.pkl` | All Chapter 5 numerical results (accuracies, F1 scores, confusion matrices) |
| `ch6_all_results.pkl` | All Chapter 6 dynamical metrics (Φ, τ, H_π, sliding window results, λ₁) |
| `ch6_exp1_esp.pkl` | Detailed ESP convergence and Lyapunov measurements |
| `shape_features.pkl` | Preprocessed features (input for Chapter 6) |

### QC Reports

| File | Contents |
|------|----------|
| `qc_report.txt` | Full text QC report |
| `qc_summary.csv` | Per-file QC metrics |
| `qc_flagged_subjects.csv` | Subjects needing manual review |
| `qc_amplitude_distributions.pdf` | Channel amplitude box plots |
| `qc_subject_heatmap.pdf` | Per-subject quality heatmap |

---

## Key Parameters Reference

All parameters are consistent across scripts. Change them if adapting to different data.

### Reservoir (LIF Spiking Network)

```python
N_RES            = 256    # Number of reservoir neurons
BETA             = 0.05   # Membrane leak rate (higher = faster decay)
THRESHOLD        = 0.5    # Firing threshold (M_th)
SPECTRAL_RADIUS  = 0.9    # Recurrent weight scaling (controls echo state property)
```

### Preprocessing

```python
DOWNSAMPLE_FACTOR = 4     # 1024 Hz → 256 Hz
BASELINE_SAMPLES  = 205   # First 205 rows to discard (200 ms at 1024 Hz)
```

### Feature Extraction

```python
N_BINS           = 6      # BSC temporal bins
PCA_COMPONENTS   = 64     # Per-channel PCA reduction
FEATURE_WINDOW   = (10, 70)  # Timestep range for feature extraction (out of 256)
```

### Classification

```python
N_FOLDS          = 10     # Subject-stratified cross-validation folds
RANDOM_STATE     = 42     # All random seeds for reproducibility
K_NEIGHBORS      = 5      # Spatial graph connectivity
```

### Data QC Thresholds

```python
AMPLITUDE_WARN_UV  = 200    # Warn if max amplitude exceeds this
AMPLITUDE_FAIL_UV  = 500    # Fail if max amplitude exceeds this
FLAT_CHANNEL_STD   = 0.01   # "Dead" channel detection threshold
BASELINE_MEAN_WARN = 5.0    # Baseline mean should be near 0 µV
OUTLIER_SD         = 5.0    # Flag values > 5 SD from mean
```

---

## Troubleshooting and Notes

### Known Exclusions
- **Subject 127** is explicitly excluded in all scripts (`EXCLUDED_SUBJECTS = {127}`)

### Runtime Estimates
- Chapter 4 (synthetic): ~2–5 minutes
- Chapter 5 (full SHAPE dataset): ~15–20 minutes (reservoir processing is the bottleneck)
- Chapter 6 (dynamical analysis): ~10–15 minutes

### Matplotlib Backend
All scripts set `matplotlib.use('Agg')` for headless/server environments. If you want interactive plots, remove or change this line.

### Reproducibility
- All random seeds are hardcoded (`seed=42`)
- Per-channel reservoir seeds: `seed + ch * 17` (ensures each channel gets a unique but deterministic reservoir)
- PCA is fit per-fold on training data only (no data leakage)

### Common Adaptation Pitfalls

1. **Channel count mismatch:** If you have 64 channels instead of 34, the reservoir loop, electrode position array, and all shape assertions must be updated.
2. **Sampling rate mismatch:** If your data is at 512 Hz, change the downsample factor from 4 to 2 (target is 256 Hz).
3. **No baseline period:** If your data is already baseline-removed, skip the `data[205:]` slicing.
4. **Single-trial data:** The SHAPE files are trial-averaged ERPs. If you have single trials, either average them first or modify the pipeline to handle trial-level data (which would give you more samples but noisier signals).
5. **Different conditions:** The code expects exactly 3 conditions for Chapter 5 classification. If you have 2 or 5 conditions, update the label encoding and confusion matrix dimensions.

---

## Citation

```
Lane, A. (2026). ARSPI-Net: Hybrid Neuromorphic Affective Computing
Architecture for EEG Signal Processing. PhD Dissertation.
```
