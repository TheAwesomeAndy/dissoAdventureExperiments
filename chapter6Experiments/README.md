# Chapter 6: Dynamical Characterisation of the LIF Reservoir

This folder contains the experimental scripts for Chapter 6 of the ARSPI-Net dissertation. Chapter 6 validates the Leaky Integrate-and-Fire (LIF) spiking reservoir as a measurement instrument before using it for scientific discovery.

## Experiment Overview

The experiments follow a progressive validation pipeline: verify the reservoir behaves correctly, confirm measurements are reliable, then use validated metrics to investigate affective subcategory differences.

| Experiment | Script | Purpose |
|------------|--------|---------|
| 6.1 | `run_chapter6_exp1_esp.py` | Echo State Property verification via driven Lyapunov exponent |
| 6.2 | `run_chapter6_exp2_reliability.py` | Cross-seed reliability of 11 dynamical metrics (ICC 3,1) |
| 6.3a | `run_chapter6_exp3_surrogate.py` | Surrogate sensitivity testing (3 null families) |
| 6.3b | `run_chapter6_exp3_valueadd.py` | Nonlinear transformation value-add vs raw EEG |
| 6.4 | `run_chapter6_exp4_dissociation.py` | Affective subcategory dissociation (within-valence contrasts) |
| 6.5 | `run_chapter6_exp5_interaction.py` | Diagnosis x subcategory interaction (5 clinical groups) |
| 6.6 | `run_chapter6_exp6_temporal.py` | Sliding-window temporal localisation (22 windows, 146 ms each) |

## Reproducibility

`reproduce_chapter6.py` is a standalone pipeline that reproduces all figures and tables from the chapter in a single run.

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

1. total_spikes
2. mean_firing_rate
3. phi_proxy (efficiency)
4. population_sparsity -- dropped after reliability gate
5. temporal_sparsity
6. lz_complexity -- dropped after reliability gate
7. permutation_entropy (d=4, mean membrane potential)
8. tau_relax (relaxation time)
9. tau_ac (autocorrelation decay)
10. rate_entropy
11. rate_variance

Metrics 4 and 6 failed the ICC >= 0.75 reliability gate in Experiment 6.2 and were excluded from downstream analyses.

## Data Flow

```
Raw EEG (SHAPE dataset) --> Exp 6.1 (ESP gate)
                          --> Exp 6.2 (reliability gate) --> 9 validated metrics
                          --> Exp 6.3 (surrogate gate)
                                                         --> Exp 6.4 (dissociation)
                                                         --> Exp 6.5 (interaction) [requires Exp 6.4 + Psychopathology.xlsx]
                                                         --> Exp 6.6 (temporal)
```

## Key Results

- **Exp 6.1:** Driven Lyapunov exponent lambda_1 = -0.054, 100% negative across all subjects/categories. ESP verified.
- **Exp 6.2:** 9/11 metrics pass ICC >= 0.75 reliability threshold across 10 random seeds.
- **Exp 6.3a:** 9/11 metrics detect genuine temporal structure vs phase-randomised, time-shuffled, and block-shuffled surrogates.
- **Exp 6.3b:** LIF reservoir amplifies permutation entropy detectability 6.8x over raw EEG; raw EEG superior for tau_AC (0.59x).
- **Exp 6.4:** Within-negative: permutation entropy dz = -0.31 (Mutilation > Threat). Within-positive: total spikes dz = -0.40 (Erotic > Cute).
- **Exp 6.5:** SUD shows category-dependent hypoactivation (Mutilation d = -0.46). ADHD shows global hyperactivation (d = +0.40).
- **Exp 6.6:** Peak discriminability at 708 ms (dz = -0.83 for Cute-Erotic), mapping onto the late positive potential (LPP) ERP component.

## Output

Each experiment saves a pickle file (`ch6_exp{N}_full.pkl`) and generates publication-quality PDF figures to `/mnt/user-data/outputs/pictures/`.

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
  - MEDIUM: Effect size metric is paired dz (inflates magnitudes ~2x vs independent d)
  - LOW: PE computation parameter differences between full-epoch (d=4) and windowed analyses

### Relationship to Extended 3-Class Experiments

The `experiments/ch6_ch7_3class/` directory runs 7 equivalent Chapter 6 experiments at 3-class granularity, where the condition signal is 3.6x stronger. That extension includes its own verification script (`verify_ch6_ch7_3class.py`, 28/28 PASS).

## Sample

- 211 subjects from the [SHAPE dataset](https://lab-can.com/shape/)
- 4 affective categories: Threat, Mutilation, Cute, Erotic
- Subject 127 excluded
