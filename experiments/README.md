# Extended Experiments

This directory contains experimental extensions that build on the core chapter experiments. These scripts were added in March 2026 to address cross-chapter questions and expand analyses to different class granularities.

## Subdirectories

### `ch5_4class/` — 4-Class Classification Extension
Extends Chapter 5's 3-class clinical classification to 4 IAPS subcategories (Threat, Mutilation, Cute, Erotic). Tests whether within-valence pairs carry distinct spatiotemporal signatures in the reservoir embedding space.

### `ch6_ch7_3class/` — 3-Class Dynamical and Coupling Pipeline
Consolidated pipeline for Chapters 6 and 7 at 3-class granularity (Negative, Neutral, Pleasant). The 3-class regime provides a 3.6x signal advantage over 4-class (8.7% vs 2.4% condition variance). Four scripts run sequentially: feature extraction, raw observations, Chapter 6 experiments, Chapter 7 experiments.

### `ablation/` — Layer Ablation (Keystone Experiment)
The dissertation's keystone experiment testing whether ARSPI-Net's three response layers (discriminative embedding, dynamical trajectory, spatial topology) are redundant or complementary. Directly tests the central thesis by ablating and combining feature blocks.

## Relationship to Chapter Directories

The `chapter*Experiments/` directories at the repository root contain the original 4-class subcategory experiments (Chapters 6-7) and the synthetic proof-of-concept (Chapter 4). This `experiments/` directory contains the newer consolidated pipelines and cross-chapter analyses. Both are part of the dissertation; the chapter directories are the primary source for subcategory-level findings, while these scripts address the 3-class signal regime and cross-chapter complementarity.

## Data Requirements

All scripts require the [SHAPE EEG dataset](https://lab-can.com/shape/) (not included in repository). See the root README for data format specifications.
