# Chapter 7, Experiment A: Dynamical-Topological Coupling Existence

This folder contains the script, results, and figures for Chapter 7 Experiment A of the ARSPI-Net dissertation. This experiment tests whether electrode-wise reservoir dynamical descriptors and graph-topological descriptors are coupled within subjects.

## Research Question

Does a meaningful coupling exist between the LIF reservoir's internal dynamics (spike patterns, firing variability, temporal structure) and the EEG's external connectivity topology (theta-band phase locking)?

## Method

For each of 211 subjects x 4 affective categories (844 observations):

1. **Dynamical profiling:** Run each of 34 EEG channels through a 256-neuron LIF reservoir and extract 7 dynamical metrics per channel, producing a D matrix (34 x 7).
2. **Connectivity:** Compute theta-band (4-8 Hz) phase locking value (tPLV) across all electrode pairs, producing a 34 x 34 PLV matrix.
3. **Topological extraction:** From the PLV matrix, compute strength and weighted clustering per electrode, producing a T matrix (34 x 2).
4. **Coupling:** Compute all 14 Spearman rank correlations between D and T, yielding a C matrix (7 x 2). Normalise to a scalar kappa = ||C||_Fro / sqrt(14).
5. **Permutation null:** Shuffle electrode labels in T 2000 times to generate a null distribution of kappa per observation.

## Files

| File | Description |
|------|-------------|
| `run_chapter7_experiment_A.py` | Main experiment pipeline (batch processing + analysis) |
| `extract_kappa_matrix.py` | Utility to export kappa values as CSV |
| `kappa_matrix.csv` | Per-subject kappa values (211 subjects x 4 categories) |
| `chapter7_results/ch7_full_results.pkl` | Complete results: D, T, PLV, C, kappa, null distributions (23.7 MB) |
| `chapter7_results/ch7_expA_analysis.pkl` | Aggregated statistics and figure data (128 KB) |
| `chapter7_results/figures/fig7_A1_mean_coupling_matrix.pdf` | Group-mean 7x2 coupling heatmap with Bonferroni significance |
| `chapter7_results/figures/fig7_A2_kappa_vs_null.pdf` | Observed vs null kappa distributions + p-value histogram |
| `chapter7_results/figures/fig7_A3_example_subjects.pdf` | Three example coupling matrices (weak/medium/strong) |
| `chapter7_results/figures/fig7_A4_kappa_by_category.pdf` | Violin plots of kappa by affective category |

## Usage

```bash
# Process subjects in batches (distributed execution)
python3 run_chapter7_experiment_A.py 0 8      # subjects 0-7
python3 run_chapter7_experiment_A.py 8 8      # subjects 8-15
# ... continue until all 211 subjects are processed

# Generate figures and tables after all batches complete
python3 run_chapter7_experiment_A.py --analyze

# Export kappa matrix as CSV
python3 extract_kappa_matrix.py > kappa_matrix.csv
```

## Parameters

| Parameter | Value |
|-----------|-------|
| Reservoir neurons | 256 |
| Leak rate (beta) | 0.05 |
| Spike threshold | 0.5 |
| Reservoir seed | 42 |
| EEG channels | 34 |
| Sampling rate | 1024 Hz |
| Theta band | 4-8 Hz |
| Permutations | 2000 |
| Categories | Threat, Mutilation, Cute, Erotic |

## 7 Dynamical Metrics

1. total_spikes
2. mean_firing_rate
3. rate_entropy
4. rate_variance
5. temporal_sparsity
6. permutation_entropy (d=4, mean membrane potential)
7. tau_ac (autocorrelation decay to 1/e)

## 2 Topological Metrics

1. strength (PLV row sum)
2. weighted clustering (Onnela et al.)

## Key Results

| Statistic | Value |
|-----------|-------|
| N observations | 844 |
| Median kappa (observed) | 0.2733 |
| Median kappa (null) | 0.1385 |
| % significant at p < 0.05 | 39.8% |
| % significant at p < 0.01 | 23.6% |
| Wilcoxon statistic | 338,144 |
| p-value | 4.98e-113 |
| Effect size (d_z) | 1.063 |

### Coupling by Category

| Category | N | Median kappa | Mean kappa | SD |
|----------|---|-------------|-----------|-----|
| Threat | 211 | 0.2685 | 0.2862 | 0.1388 |
| Mutilation | 211 | 0.2696 | 0.2853 | 0.1415 |
| Cute | 211 | 0.2906 | 0.3085 | 0.1450 |
| Erotic | 211 | 0.2654 | 0.2783 | 0.1414 |

### Strongest Coupling Pairs (Bonferroni-corrected)

| Dynamical Metric | Topological Metric | Mean rho | p-value |
|-----------------|-------------------|----------|---------|
| permutation_entropy | clustering | -0.0512 | 1.91e-06 |
| tau_ac | clustering | -0.0499 | 3.49e-06 |
| permutation_entropy | strength | -0.0483 | 1.11e-05 |
| tau_ac | strength | -0.0441 | 4.71e-05 |

All significant correlations are negative: higher dynamical complexity is associated with sparser, less clustered network organisation.

## Interpretation

Coupling between reservoir dynamics and EEG connectivity topology is statistically robust (d_z = 1.063, p < 10^-100). This validates the foundational assumption for subsequent Chapter 7 experiments (B-E) that dynamical and topological dimensions carry related but non-redundant information.

## Sample

- 211 subjects from the SHAPE Community dataset
- Subject 127 excluded
- Kappa range: [0.0296, 0.7341]
