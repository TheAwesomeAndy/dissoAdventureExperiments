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
Raw EEG (SHAPE Community) --> Exp 6.1 (ESP gate)
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

## Verification

See `CHAPTER6_VERIFICATION_REPORT.md` for 27 synthetic unit tests and a full static analysis of all experiment scripts.

## Sample

- 211 subjects from the SHAPE Community dataset
- 4 affective categories: Threat, Mutilation, Cute, Erotic
- Subject 127 excluded
