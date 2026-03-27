# Chapter 6: Dynamical Characterisation of the LIF Reservoir

This folder contains the experimental scripts for Chapter 6 of the ARSPI-Net dissertation. Chapter 6 validates the Leaky Integrate-and-Fire (LIF) spiking reservoir as a measurement instrument before using it for scientific discovery. The experiments operate at 4-class granularity (Threat / Mutilation / Cute / Erotic), enabling within-valence subcategory contrasts.

---

## Scientific Rationale

Chapters 4-5 established that the LIF reservoir produces discriminative embeddings (BSC6 at 99.5% on synthetic data, ~79% on real EEG). But classification accuracy alone does not characterize *what* the reservoir computes. Chapter 6 asks: what are the dynamical properties of the reservoir's internal trajectory when driven by affective EEG? Are those properties reliable, meaningful, and clinically informative?

The answer requires a progressive validation pipeline: first verify the reservoir behaves correctly (ESP), then confirm measurements are reliable (ICC), then validate that metrics detect real structure (surrogates), and only then use validated metrics to investigate affective and clinical differences.

---

## Experiment Overview

| Experiment | Script | Purpose |
|------------|--------|---------|
| 6.1 | `run_chapter6_exp1_esp.py` | Echo State Property verification via driven Lyapunov exponent |
| 6.2 | `run_chapter6_exp2_reliability.py` | Cross-seed reliability of 11 dynamical metrics (ICC 3,1) |
| 6.3a | `run_chapter6_exp3_surrogate.py` | Surrogate sensitivity testing (3 null families) |
| 6.3b | `run_chapter6_exp3_valueadd.py` | Nonlinear transformation value-add vs raw EEG |
| 6.4 | `run_chapter6_exp4_dissociation.py` | Affective subcategory dissociation (within-valence contrasts) |
| 6.5 | `run_chapter6_exp5_interaction.py` | Diagnosis x subcategory interaction (5 clinical groups) |
| 6.6 | `run_chapter6_exp6_temporal.py` | Sliding-window temporal localisation (22 windows, 146 ms each) |

---

## Reservoir Parameters

| Parameter | Value |
|-----------|-------|
| Neurons | 256 |
| Leak rate (beta) | 0.05 |
| Spike threshold | 0.5 |
| Connectivity | 10% sparse |
| Input scaling | 0.3 (Gaussian std) |
| Spectral radius | ~0.8 |

## Dynamical Metrics (11 computed, 9 validated)

| # | Metric | Family | Status |
|---|--------|--------|--------|
| 1 | total_spikes | Amplitude-tracking | Validated |
| 2 | mean_firing_rate | Amplitude-tracking | Validated |
| 3 | phi_proxy (efficiency) | Amplitude-tracking | Validated |
| 4 | population_sparsity | Amplitude-tracking | **Dropped** (ICC < 0.75) |
| 5 | temporal_sparsity | Sparsity | Validated |
| 6 | lz_complexity | Temporal-structure | **Dropped** (ICC < 0.75) |
| 7 | permutation_entropy (d=4) | Temporal-structure | Validated |
| 8 | tau_relax (relaxation time) | Temporal-structure | Validated |
| 9 | tau_ac (autocorrelation decay) | Temporal-structure | Validated |
| 10 | rate_entropy | Amplitude-tracking | Validated |
| 11 | rate_variance | Amplitude-tracking | Validated |

Metrics 4 and 6 failed the ICC >= 0.75 reliability gate in Experiment 6.2 and were excluded from downstream analyses.

## Data Flow

```
Raw EEG (SHAPE dataset) --> Exp 6.1 (ESP gate)
                          --> Exp 6.2 (reliability gate) --> 9 validated metrics
                          --> Exp 6.3 (surrogate gate)
                                                         --> Exp 6.4 (dissociation)
                                                         --> Exp 6.5 (interaction) [requires Psychopathology.xlsx]
                                                         --> Exp 6.6 (temporal)
```

---

## Detailed Results and Interpretation

### Experiment 6.1: Echo State Property (ESP) Verification

**Question:** Does the reservoir operate in the echo state regime, where driven trajectories converge regardless of initial conditions?

**Method:** Compute the driven Lyapunov exponent (lambda_1) across all subjects and categories. lambda_1 < 0 confirms trajectory convergence.

**Result:** lambda_1 = -0.054, 100% negative across all subjects and categories.

**Interpretation:** The reservoir operates firmly in the echo state regime. Different initial conditions converge to the same driven trajectory, meaning the reservoir's response is a deterministic function of its input. This is the prerequisite for using dynamical metrics as *measurements* of the input signal — if the reservoir were chaotic (lambda_1 > 0), different runs would produce different metrics for the same input, and no downstream analysis would be meaningful.

---

### Experiment 6.2: Cross-Seed Reliability (ICC)

**Question:** Are dynamical metrics stable across different random weight initializations?

**Method:** Compute each metric for the same subjects across 10 independent reservoir initializations. Calculate ICC(3,1) — the intraclass correlation measuring absolute agreement.

**Result:** 9 of 11 metrics achieve ICC >= 0.75. Population sparsity and Lempel-Ziv complexity fail the reliability gate and are excluded.

**Interpretation:** The reservoir produces consistent dynamical measurements despite random weight initialization. The 9 validated metrics reflect properties of the *input signal*, not properties of a particular random weight matrix. The two failures are informative: population sparsity is too sensitive to the specific neuron-level firing pattern, and Lempel-Ziv complexity is sensitive to the binary sequence structure which varies across initializations.

---

### Experiment 6.3a: Surrogate Sensitivity Testing

**Question:** Do the metrics detect genuine temporal structure, or would they produce the same values from random signals?

**Method:** Three surrogate families that systematically destroy different signal properties:
- **Phase-randomized:** Preserves power spectrum, destroys temporal correlations
- **Time-shuffled:** Preserves amplitude distribution, destroys temporal order
- **Block-shuffled:** Preserves local structure, destroys global temporal organization

**Result:** 9 of 11 metrics detect genuine temporal structure (significant difference from all three surrogate types).

**Interpretation:** The metrics capture real properties of the EEG signal that are destroyed by randomization. This is a necessary validity check — metrics that cannot distinguish real signals from surrogates carry no meaningful information.

---

### Experiment 6.3b: Value-Add vs Raw EEG

**Question:** Does the LIF reservoir transformation add measurement value beyond what can be computed directly from raw EEG?

**Result:**
- Permutation entropy detectability: **6.8x amplification** (reservoir vs raw EEG)
- tau_AC: **0.59x** (raw EEG is superior)

**Interpretation:** The reservoir is not uniformly better. It amplifies temporal complexity signals (permutation entropy) by providing a nonlinear expansion of the input into a high-dimensional spike space. But for autocorrelation structure, the raw EEG's continuous signal carries more information than the binary spike train. This characterizes the reservoir's *specific* contribution — it is a nonlinear complexity amplifier, not a universal signal enhancer.

---

### Experiment 6.4: Affective Subcategory Dissociation

**Question:** Do within-valence subcategory pairs (Threat vs Mutilation, Cute vs Erotic) produce distinguishable reservoir dynamics?

**Result:**
- **Within-negative:** Permutation entropy d_z = -0.31 (Mutilation > Threat)
- **Within-positive:** Total spikes d_z = -0.40 (Erotic > Cute)

**Interpretation:** The reservoir's dynamics distinguish stimuli that share the same valence but differ in content. Mutilation produces more temporally complex reservoir trajectories than Threat, while Erotic produces more total spiking activity than Cute. Critically, different metric families carry the dissociation for different valences — the within-negative contrast is carried by temporal-structure metrics, while the within-positive contrast is carried by amplitude-tracking metrics. This suggests the two metric families reflect genuinely different aspects of affective processing.

---

### Experiment 6.5: Diagnosis x Subcategory Interaction

**Question:** Does clinical status alter the pattern of dynamical reactivity across affective subcategories?

**Result:**
- **SUD:** Category-dependent hypoactivation, strongest for Mutilation (d = -0.46)
- **ADHD:** Global hyperactivation across all categories (d = +0.40)

**Interpretation:** SUD subjects show selectively blunted reservoir dynamics in response to aversive content (Mutilation), consistent with clinical models of emotional hyporesponsivity in substance use disorders. ADHD subjects show globally elevated dynamics regardless of stimulus content, consistent with models of cortical hyperexcitability. The SUD pattern is category-specific (interaction); the ADHD pattern is category-independent (main effect). This establishes that the reservoir's temporal descriptors carry clinically meaningful information about affective processing.

---

### Experiment 6.6: Sliding-Window Temporal Localisation

**Question:** When during the EEG epoch is affective information most discriminable in the reservoir dynamics?

**Method:** 22 overlapping windows of 146 ms each, spanning the full post-stimulus epoch. Dynamical metrics computed per window.

**Result:** Peak discriminability at 708 ms (d_z = -0.83 for Cute-Erotic contrast).

**Interpretation:** The peak at 708 ms maps onto the late positive potential (LPP) — a well-established ERP component associated with sustained emotional processing. The reservoir's temporal dynamics capture the same processing stage identified by decades of ERP research, but through a completely different measurement modality (spiking dynamics vs voltage deflection). This convergence validates the reservoir as a biologically meaningful measurement instrument.

---

## Reproducibility

`reproduce_chapter6.py` is a standalone pipeline that reproduces all figures and tables from the chapter in a single run.

## Output

Each experiment saves a pickle file (`ch6_exp{N}_full.pkl`) and generates publication-quality PDF figures.

---

## Verification Results

<!-- Last run: 2026-03-20, Result: 31/31 PASS -->

### Automated Verification (verify_chapter6.py)

```bash
python chapter6Experiments/verify_chapter6.py
```

**Result: 31/31 PASS.** The verification script tests all core infrastructure on synthetic data:

- **Syntax validation (8 tests):** All 8 scripts (7 experiments + reproduce) parse without errors
- **LIF Reservoir (8 tests):** Module imports, reservoir instantiation, weight shapes (256,) and (256,256), output shapes (1229,256) for both membrane and spikes, binary spikes, finite membrane, non-silent, sparse (<30% active)
- **Dynamical metrics (6 tests):** total_spikes > 0, MFR in (0,1), population rate shape, rate entropy > 0, rate variance > 0, tau_ac computable, permutation entropy in (0,1]
- **ESP convergence (1 test):** Late trajectory distance < early distance (trajectories converge under same input from different initial conditions)
- **Surrogate generation (5 tests):** Phase-randomized preserves power spectrum and changes signal; time-shuffled preserves amplitude distribution and changes temporal order
- **Documentation (1 test):** CHAPTER6_VERIFICATION_REPORT.md exists

### Independent Code Review (CHAPTER6_VERIFICATION_REPORT.md)

A 746-line independent verification report provides 27 additional synthetic unit tests covering:

- **Algorithm correctness:** Lyapunov exponent computation, ICC formula, surrogate generation, permutation interaction test — all verified correct on synthetic data with known ground truth
- **Scientific methodology:** Progressive validation-to-discovery pipeline rated "exemplary"
- **Identified issues:**
  - HIGH: Reservoir architecture inconsistency between experiment scripts and `reproduce_chapter6.py` (threshold subtraction + floor vs different implementation)
  - MEDIUM: No multiple-comparison corrections across Experiments 6.4-6.6 (18-308 uncorrected tests)
  - MEDIUM: Effect size metric is paired d_z (inflates magnitudes ~2x vs independent d)
  - LOW: PE computation parameter differences between full-epoch (d=4) and windowed analyses

### Relationship to Extended 3-Class Experiments

The `experiments/ch6_ch7_3class/` directory runs 7 equivalent Chapter 6 experiments at 3-class granularity, where the condition signal is 3.6x stronger. That extension includes its own verification script (`verify_ch6_ch7_3class.py`, 28/28 PASS).

---

## Sample

- 211 subjects from the [SHAPE dataset](https://lab-can.com/shape/)
- 4 affective categories: Threat, Mutilation, Cute, Erotic
- Subject 127 excluded
