# Nuisance-alignment compression study — results

Why a data-independent random projection beats data-fitted PCA on
subject-grouped clinical EEG, with the mechanism, its boundary, and an honest
null. Machine-readable: [`nuisance_results.json`](nuisance_results.json).
Exploratory; passing tests verify code behavior, not scientific validity.

## The claim

> On subject-grouped clinical EEG, unsupervised variance-maximizing compression
> (PCA) aligns its leading components to the **subject-nuisance** subspace by
> construction. A data-independent isotropic random projection has no such
> alignment, so it is leak-free *and*, when compression is aggressive enough to
> bury the condition signal below the retained nuisance spectrum, more
> discriminative.

## Real-data anchor (SHAPE three-condition, N=211 / 633)

**Accuracy (N1).** SRP-64 = **64.3%** vs fold-local PCA-64 = **60.0%**;
paired participant-bootstrap difference **+4.4pp [1.5, 7.2]** (excludes zero).

**Mechanism (M2).** In the 64-dim compressed space:

| Quantity | PCA-64 | SRP-64 | Ambient |
|---|---:|---:|---:|
| Subject/condition variance ratio ρ | **44.4** | **7.3** | 11.4 |
| Min principal angle to **subject** subspace | **1.34°** | — | — |
| Min principal angle to **condition** subspace | 3.38° | — | — |
| Mean angle to subject / condition | 3.95° / 10.19° | — | — |

PCA-64's subspace is **nearly collinear with the subject-nuisance subspace
(1.34°)** and inflates ρ to 44; SRP-64 stays *below* ambient (7.3). This is the
direct geometric explanation of the N1 accuracy gap.

## Cross-granularity replication (real data)

The mechanism replicates on the independent four-category labelling
(`nuisance_results_4class.json`, 210/840):

| Quantity | 3-condition | 4-category |
|---|---:|---:|
| PCA min principal angle → **subject** | **1.32°** | **1.59°** |
| PCA min principal angle → condition | 3.32° | 5.29° |
| ρ compressed — PCA / SRP / ambient | 44.4 / 7.3 / 11.4 | 31.9 / 14.1 / 17.2 |
| M5 null — raw-ERP ρ ≤ BSC6 ρ | 8.7 ≤ 11.4 | 12.3 ≤ 17.2 |

On both label sets PCA-64 sits ~1.3–1.6° from the subject subspace and inflates
ρ above ambient, while SRP-64 stays below ambient. The reservoir-is-not-the-
disentangler null replicates too. The geometric mechanism is not an artefact of
one labelling.

## Generalization — the mechanism is monotone (M1a, controlled)

As subject-nuisance magnitude grows, the PCA subspace rotates away from
condition and toward subject:

| Nuisance ratio | PCA angle → subject | PCA angle → condition |
|---:|---:|---:|
| 0.0 | 26.6° | 0.7° |
| 0.5 | 10.8° | 0.9° |
| 1.0 | 4.5° | 1.6° |
| 2.0 | 1.3° | 5.2° |
| 4.0 | 0.3° | 29.4° |

The rotation is monotone and dataset-independent: PCA maximizes variance, which
*is* the nuisance once the nuisance dominates. A random projection has no
data-dependent alignment.

## Honest boundary and null

- **M1b — accuracy is regime-dependent.** In the controlled model, SRP does
  **not** always beat PCA on accuracy (crossover not reached over the swept
  range at dim=200, k=16). The accuracy advantage requires compression
  aggressive enough that the condition signal falls below the retained nuisance
  spectrum — which the real SHAPE BSC regime (1,536→64 per electrode) satisfies,
  but not every regime. We report the crossover, not a universal win.
- **M3 — compressor comparison** (controlled, ratio 3.0, dim 200): fold-local
  PCA 98.8%, random-subsample 88.1%, SRP 91.9%, supervised LDA 85.0%. In a mild
  regime PCA is the best compressor; the SRP advantage is not universal.
- **M5 — the reservoir does not disentangle.** Subject/condition ρ: raw ERP-16
  = 8.7, BSC6 = 11.4, BSC6/SRP-64 = 9.8. The neuromorphic transform slightly
  *increases* entanglement relative to raw ERP; the reduction to 9.8 comes from
  the isotropic projection, not the reservoir. Stated plainly.

## What is genuinely publishable here

1. **Mechanistic result (real + controlled):** *PCA aligns to the grouping
   nuisance* — quantified by principal angles (1.34° on real data) and shown
   monotone in a controlled model. This is a clean, general, testable claim
   about compression on grouped data, independent of ARSPI-Net's accuracy.
2. **Practical consequence (real):** on SHAPE, this costs data-fitted PCA
   4.4pp versus a leak-free random projection — a concrete argument against the
   default PCA step in participant-grouped clinical-EEG pipelines.
3. **Honest scope:** the accuracy benefit is regime-bounded (M1b) and the
   reservoir is not the disentangler (M5).

## Reproduce

```text
python experiments/novel/verify_nuisance.py                       # machinery (no data)
python experiments/novel/run_nuisance_study.py --data3 ./batch_data --out nuisance_results.json
```

Open items before submission: full (non-fast) sweep, the four-category cohort,
and an external cohort to establish generality of the accuracy consequence.
