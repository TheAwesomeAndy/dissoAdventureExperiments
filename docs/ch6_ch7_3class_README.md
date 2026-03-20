# ARSPI-Net Chapters 6 & 7: 3-Class Experimental Pipeline

## Overview

This pipeline extracts dynamical and topological features from the [SHAPE EEG dataset](https://lab-can.com/shape/) at 3-class granularity (Negative / Neutral / Pleasant) for use in Chapters 6 (Dynamical Characterization) and 7 (Structure-Function Coupling) of the ARSPI-Net dissertation.

The 3-class problem is the primary experimental vehicle. Variance decomposition (Chapter 5) establishes that condition-related signal accounts for 8.7% of total embedding variance at 3-class versus 2.4% at 4-class — a 3.6× signal advantage. Classification accuracy with subject-centering reaches 79.4% at 3-class. Every experiment in this pipeline operates in the stronger signal regime.

The 4-class analyses (Threat / Mutilation / Cute / Erotic) provide specific findings that require within-valence resolution — the feature hierarchy inversion, the arousal-dominance pairwise hierarchy, the PTSD × Threat–Mutilation interaction, and the Cute–Erotic coupling difference. Those findings appear in Chapter 5 (§5.5) and Chapter 7 (Experiment 7.5) as paired-granularity comparisons.

---

## ARSPI-Net Architecture and Pipeline Context

ARSPI-Net is a three-stage hybrid neuromorphic architecture. Stage 1 (Temporal Encoding) uses a LIF reservoir to convert continuous EEG into spike trains, encoded via BSC6 temporal binning and PCA-64 compression. Stage 2 (Spatial Relational Analysis) treats the 34 per-channel embeddings as node features on a sensor graph, with inter-channel correlations defining edge weights. Stage 3 (Readout) provides both classification and clinical interpretability.

Chapters 3–5 validate Stages 1–2 and establish classification performance. This pipeline addresses the next scientific questions. Chapter 6 asks: what are the dynamical properties of the reservoir's internal trajectory when driven by affective EEG, and do those properties vary with clinical status? Chapter 7 asks: are the temporal properties (reservoir dynamics) and spatial properties (graph topology) statistically coupled across electrodes, and does that coupling carry information beyond either property alone?

---

## Pipeline Structure

Four scripts, run in order. Each script follows the five-step experimental cycle: mathematical motivation → experimental design → observation → analysis → next question. Scripts 01 and 02 produce and characterize the raw data. Scripts 03 and 04 analyze it.

**Script 01: Feature Extraction** (`ch6_ch7_01_feature_extraction.py`). Drives the validated reservoir on all 633 observations (211 subjects × 3 conditions), extracts 7 core dynamical metrics + 4 additional metrics per channel, and computes theta-band tPLV matrices for topological analysis. Input: raw 3-class EEG files. Output: `ch6_ch7_3class_features.pkl`. Runtime: ~10–20 minutes.

**Script 02: Raw Observations** (`ch6_ch7_02_raw_observations.py`). Characterizes the dynamical and topological features before any clinical or statistical analysis. Eight observation sets establish the empirical properties of the extracted features: their distributions, condition sensitivity, channel heterogeneity, temporal structure, connectivity patterns, variance decomposition, and clinical metadata coverage. Input: `ch6_ch7_3class_features.pkl`. Output: 8 observation PDFs + terminal statistics. Runtime: ~3–5 minutes.

**Script 03: Chapter 6 Experiments** (`ch6_03_experiments.py`). Seven experiments testing the dynamical metrics as condition-sensitive and clinically informative descriptors of the reservoir's driven state. Each experiment reports raw observations (group means, distributions, effect magnitudes) before performing statistical tests. Input: `ch6_ch7_3class_features.pkl` + `clinical_profile.csv`. Output: `ch6_results.pkl` + experiment PDFs. Runtime: ~10–15 minutes.

**Script 04: Chapter 7 Experiments** (`ch7_04_experiments.py`). Five experiments characterizing the coupling between temporal (dynamical) and spatial (topological) descriptor families across electrodes. Input: `ch6_ch7_3class_features.pkl` + `clinical_profile.csv`. Output: `ch7_results.pkl` + experiment PDFs. Runtime: ~5–10 minutes.

---

## Script 01: Feature Extraction

### Mathematical Motivation

The LIF reservoir at the validated operating point (N=256, β=0.05, θ=0.5) maps a continuous EEG channel signal u(t) into a spike train matrix S ∈ {0,1}^{256×T} and a membrane potential matrix M ∈ R^{256×T}. The reservoir's driven trajectory {M(t)} is a deterministic function of u(t) for fixed weights (Definition 6.5).

Seven core dynamical metrics characterize the driven trajectory. They decompose into two families identified in Chapter 6's theoretical framework.

Amplitude-tracking metrics measure the gross activation of the reservoir response. They respond to input magnitude and energy: total spikes (ΣΣ S_{i,t}), mean firing rate (total_spikes / (N_res × T)), rate entropy (H of per-neuron firing rate distribution), and rate variance (Var of population firing rate r(t)).

Temporal-structure metrics measure the complexity and persistence of the reservoir's state-space trajectory. They respond to the temporal organization of the input, not its magnitude: permutation entropy (Ĥ_π, normalized PE of mean membrane potential, d=4, τ=1; Bandt & Pompe 2002, Def. 6.14) and autocorrelation decay (τ_AC, lag at which ACF(r(t)) < 1/e; Def. 6.17–6.18).

Temporal sparsity (fraction of timesteps with r(t) < 1/N_res) measures the energy efficiency of the spiking response.

Four additional metrics support specific Chapter 6 experiments: Lempel-Ziv complexity (C_LZ, Def. 6.13), Lyapunov proxy (λ_proxy, approximate driven Lyapunov exponent), relaxation time (τ_relax, Def. 6.16), and return-to-baseline time (T_RTB, Def. 6.19).

Topological metrics for Chapter 7 are derived from the theta-band (4–8 Hz) time-averaged phase-locking value (tPLV) matrix (Lachaux et al. 1999), computed from full-resolution 1024 Hz EEG. The two topological descriptors are weighted node strength (sum of tPLV edge weights per channel) and weighted clustering coefficient (Onnela formula; Onnela et al. 2005).

### Expected Output

File: `ch6_ch7_3class_features.pkl` (~50–200 MB) containing D (633,34,7), D_extra (633,34,4), T_topo (633,34,2), tPLV_mats (633,34,34), pop_rate_ts (633,34,256), y (633,), subjects (633,), and cond_names dict.

### Run Command

```
python ch6_ch7_01_feature_extraction.py
```

Adjust `DATA_DIR` at the top of the script if `batch_data/` is elsewhere. Dependencies: numpy, scipy, pickle.

---

## Experimental Program: Chapter 6

Each experiment follows the five-step cycle. The "Theoretical Prediction" column states a specific, falsifiable prediction derived from the motivating mathematics or from established clinical neuroscience models. Every outcome — whether it confirms, partially confirms, or departs from the prediction — is an observation about the system that informs the next experiment.

### EXP-6.1: Condition Effects on Dynamical Metrics

The reservoir processes different emotional inputs. The data processing inequality constrains what information the driven trajectory can carry about the stimulus class. This experiment measures whether the 7 core metrics differ across Negative, Neutral, and Pleasant conditions. The theoretical prediction is that Negative stimuli, which produce larger and more sustained ERP deflections (OBS-1 in Chapter 5), drive the reservoir into a regime of higher rate variance and longer autocorrelation decay. This observation establishes whether the dynamical metrics are condition-sensitive — the prerequisite for every subsequent experiment.

### EXP-6.2: Metric Family Decomposition

Chapter 6 decomposes the dynamical metrics into amplitude-tracking and temporal-structure families. The LIF reservoir's distinctive contribution over simpler amplitude-based encoders is temporal structure encoding. The theoretical prediction is that the temporal-structure family (Ĥ_π, τ_AC) shows larger effect sizes on the Neg–Pos contrast than the amplitude-tracking family (spikes, MFR, rate_entropy, rate_variance). This observation characterizes what kind of information the reservoir's dynamics encode about affective input.

### EXP-6.3: Transdiagnostic Clinical Comparisons

Chapter 5 established that graph topology carries disorder-specific signatures (SUD p=0.0004, PTSD 11 channel-metric pairs). The dynamical metrics measure a different property of the same reservoir response — temporal trajectory behavior rather than spatial connectivity structure. This experiment tests whether temporal dynamics also carry clinical information, using the same diagnostic variables (MDD, PTSD, SUD, GAD, ADHD, medication, sex). The theoretical predictions derive from established clinical neuroscience: SUD is associated with blunted emotional reactivity, PTSD with threat hypervigilance, and MDD with prolonged recovery from aversive stimuli.

### EXP-6.4: Condition × Clinical Interactions

Chapter 5's strongest clinical finding is a condition × diagnosis interaction: SUD subjects reorganize their network connectivity during emotional processing in the opposite direction to non-SUD subjects (p=0.0004). This experiment tests the dynamical analog — does the change in reservoir dynamics between conditions differ by diagnosis? The theoretical prediction is that SUD subjects show attenuated dynamical reactivity and PTSD subjects show amplified negative-specific reactivity.

### EXP-6.5: Sparse Coding Efficiency (Φ)

Φ = I_decoded / SynOps measures bits of stimulus information per synaptic operation — the energy-information tradeoff of the neuromorphic representation. This metric connects the reservoir's classification performance to its computational cost. Φ varies across conditions because both the numerator (decoded information) and the denominator (total spikes) are condition-dependent. Whether Φ varies with clinical status is an open empirical question — this experiment measures it.

### EXP-6.6: HC vs MDD Hypothesis Test

Chapter 6's theoretical framework generates three specific directional predictions: H1 (Φ_HC > Φ_MDD, from the efficient coding hypothesis), H2 (Λ_HC > Λ_MDD for complexity metrics CLZ and Ĥ_π), and H3 (τ_relax_MDD > τ_relax_HC, from the critical slowing down literature). This experiment tests all three predictions. The HC group is small (~22 subjects) and the MDD group is large (~142), so statistical power is asymmetric — the observation of effect sizes is as informative as the p-values.

### EXP-6.7: Dynamical Metric Discriminative Value

The dynamical metrics are designed as interpretable biomarkers, not classification features. This experiment measures whether they also carry discriminative information — above-chance accuracy on 3-class emotion and binary clinical detection tasks. Both outcomes are informative: discriminative value demonstrates that the metrics capture information accessible to downstream classifiers; absence of discriminative value demonstrates that their contribution is organizational rather than additive, which itself characterizes the information structure of the reservoir's temporal encoding.

---

## Experimental Program: Chapter 7

### EXP-7.1: Coupling Existence (3-class)

The coupling scalar κ measures the alignment between dynamical and topological descriptor profiles across 34 electrodes. The 4-class analysis found uniformly negative coupling. This experiment tests whether coupling exceeds permutation-null baselines at 3-class, where the condition signal is 3.6× stronger.

### EXP-7.2: Variance Decomposition of κ

The 4-class analysis found 29.2% subject variance, 0.6% condition variance, and 70.1% residual variance in κ. The condition fraction did not reach significance (p=0.133). At 3-class with stronger condition signal, the theoretical prediction is that the condition fraction increases because the between-valence contrast (Negative vs Pleasant) is larger than any within-valence contrast.

### EXP-7.3: Clinical Coupling Differences

Chapters 5 and 6 independently test whether spatial and temporal properties differ by diagnosis. This experiment tests whether the relationship between them — the coupling — differs by diagnosis. From the coupling interpretation (high-connectivity channels have lower input diversity → simpler reservoir trajectories), SUD subjects may show weaker coupling because their altered connectivity disrupts the normal structure-function relationship. This is a prediction that neither Chapter 5 nor Chapter 6 can test individually.

### EXP-7.4: Augmentation Ablation

If the two descriptor families carry complementary discriminative information, combining them improves clinical detection beyond either alone. If not, their relationship is organizational (characterized by coupling) rather than discriminatively additive. The 4-class Chapter 7 found that dynamical descriptors are "analytically informative rather than discriminatively additive." The 3-class signal regime may produce a different observation.

### EXP-7.5: Within-Valence Coupling Structure (4-class)

This uses existing 4-class results. The Cute–Erotic coupling difference (Δκ = +0.031, p = 0.025) and the category-conditioned coupling reorganization concentrated in temporal-structure metrics establish a cross-chapter convergence with Chapter 6's metric family decomposition. This experiment requires subcategory resolution and remains at 4-class.
