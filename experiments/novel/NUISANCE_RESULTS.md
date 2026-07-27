# Nuisance-alignment compression study — results

When does a data-independent random projection beat data-fitted PCA on
subject-grouped clinical EEG, and *why*? Machine-readable:
[`nuisance_results.json`](nuisance_results.json). Exploratory; passing tests
verify code behavior, not scientific validity.

## The claim (as finally supported)

> On subject-grouped clinical EEG, **aggressive** variance-maximizing
> compression (PCA that reduces the whole flattened representation to a few
> dozen coordinates) aligns its leading components to the **subject-nuisance**
> subspace by construction. A data-independent isotropic random projection has
> no such alignment. This gross alignment is **operator-dependent**: it is
> dramatic under global compression but *does not appear* under the mild
> per-electrode compression that ARSPI-Net (N1) actually uses — so it is **not**
> the explanation for the N1 accuracy gap.

The honest scope below is the result of running the geometry with the *same*
operator N1 uses, which the earlier version of this study had left pending.

## Real-data anchor (SHAPE three-condition, N=211 / 633)

**Accuracy (N1).** SRP-64 = **64.329%** vs fold-local PCA-64 = **59.968%**;
paired participant-bootstrap difference **+4.36pp [1.5, 7.2]** (excludes zero).
These accuracies reproduce exactly across library versions.

**Mechanism (M2), two operators.** The geometry is computed both ways: the
**global** operator (flatten the full 52,224-dim BSC tensor, reduce to 64 total
coordinates) and the **blockwise** operator that N1 actually uses (reduce each
electrode's 1,536 BSC coordinates to 64 *independently*, concatenate to 2,176
coordinates).

| Quantity (compressed 64-space) | GLOBAL PCA | **BLOCKWISE PCA (N1)** | SRP | Ambient |
|---|---:|---:|---:|---:|
| Subject/condition variance ratio ρ | **43.55** | **7.16** | 9.82 | 11.41 |
| PCA min principal angle → **subject** | **1.33°** | **23.9°** | — | — |
| PCA min principal angle → condition | 3.21° | 28.83° | — | — |

Under the **global** operator PCA-64 is nearly collinear with the subject
subspace (1.33°) and inflates ρ to 43.6. Under the **blockwise** operator — the
one whose accuracy gap N1 reports — PCA-64 sits **23.9°** from the subject
subspace and its ρ (7.16) is *below* ambient (11.41). The gross nuisance
alignment does not survive the operator change.

> **Resolved (was: operator caveat).** The earlier version reported only the
> global geometry and withheld the causal "this explains N1" claim pending the
> matched-operator run. That run is now done (`M2_geometry_blockwise` in
> `nuisance_results.json`, produced by the exact N1 SRP/PCA operators). Its
> verdict is a **negative** one and is stated plainly: the strong subject
> alignment is a property of *aggressive global* compression, not of N1's
> per-electrode compression. Physically, subject identity is a coherent pattern
> *across* electrodes (a direction in the 52,224-dim flattened space); a shared
> per-electrode basis captures within-electrode variance shapes and structurally
> cannot reach that cross-electrode direction, so blockwise PCA does not
> concentrate the subject nuisance. **We therefore do not claim M2 explains the
> N1 accuracy gap.** The remaining +4.36pp is a mild-regime effect (consistent
> with fold-local PCA's estimation variance and a much weaker version of the
> alignment), not gross collinearity.

## Cross-granularity replication (real data)

The operator-dependence replicates on the four-category labelling
(`nuisance_results_4class.json`, 210/840) — a **cross-granularity consistency
check within SHAPE**, not an independent cohort:

| Quantity | 3-condition | 4-category |
|---|---:|---:|
| GLOBAL PCA min angle → subject | 1.33° | 1.58° |
| **BLOCKWISE PCA min angle → subject** | **23.9°** | **23.9°** |
| GLOBAL ρ_pca / ambient | 43.55 / 11.41 | 31.80 / 17.20 |
| **BLOCKWISE ρ_pca / ambient** | **7.16 / 11.41** | **15.53 / 17.20** |
| M5 null — raw-ERP ρ ≤ BSC6 ρ | 8.68 ≤ 11.41 | 12.32 ≤ 17.20 |

On both label sets the global operator aligns PCA to subject (~1.3–1.6°) and
inflates ρ above ambient, while the blockwise operator does not (23.9° from
subject, ρ below ambient). The operator-dependence is not an artefact of one
labelling.

## Generalization — the global mechanism is monotone (M1a, controlled)

As subject-nuisance magnitude grows, the *global* PCA subspace rotates away from
condition and toward subject:

| Nuisance ratio | PCA angle → subject | PCA angle → condition |
|---:|---:|---:|
| 0.0 | 26.6° | 0.7° |
| 0.5 | 10.8° | 0.9° |
| 1.0 | 4.5° | 1.6° |
| 2.0 | 1.3° | 5.2° |
| 4.0 | 0.3° | 29.4° |

The rotation is monotone and dataset-independent: variance-maximizing
compression follows the nuisance once the nuisance dominates. This is the
mechanism that makes an isotropic random projection preferable **under
aggressive compression**; it does not by itself predict the blockwise regime.

## Honest boundary and null

- **Operator scope (the key boundary).** The subject-alignment mechanism is
  demonstrated for aggressive global compression (M1a controlled + global M2).
  It is explicitly **absent** for N1's per-electrode operator (blockwise M2,
  both granularities). The N1 accuracy advantage is therefore *not* attributed
  to gross nuisance alignment.
- **M1b — the controlled model does NOT reproduce the accuracy ordering.** In
  the controlled model, fold-local PCA outperforms SRP at *every* tested nuisance
  level (crossover ratio: none). The defensible controlled result is only that
  global PCA increasingly aligns with the simulated subject subspace as subject
  variance grows (M1a).
- **M3 — compressor comparison** (controlled, ratio 3.0, dim 200): fold-local
  PCA 98.8%, random-subsample 88.1%, SRP 91.9%, supervised LDA 85.0%. In a mild
  regime PCA is the best compressor; the SRP advantage is not universal.
- **M5 — the reservoir does not disentangle.** Subject/condition ρ: raw ERP-16
  = 8.68, BSC6 = 11.41, BSC6/SRP-64 = 9.82. The neuromorphic transform slightly
  *increases* entanglement relative to raw ERP; the reduction to 9.82 comes from
  the isotropic projection, not the reservoir.

## What is genuinely publishable here

1. **Mechanistic result (real + controlled), scoped to the operator:**
   *aggressive variance-maximizing compression aligns to the grouping nuisance* —
   quantified by principal angles (1.33° on real data under global compression)
   and shown monotone in a controlled model. A clean, general, testable claim
   about aggressive compression on grouped data.
2. **A negative control that sharpens it:** the *same* alignment measurement
   under a mild per-electrode operator shows **no** gross alignment (23.9°),
   demonstrating the phenomenon is compression-ratio-dependent rather than an
   inevitable property of PCA on grouped data. This is itself a useful, honest
   contribution and guards against over-generalizing claim 1.
3. **Practical consequence (real, mild-regime):** on SHAPE, a leak-free random
   projection still edges data-fitted PCA by +4.36pp under the per-electrode
   operator — an argument for the default random-projection step in
   participant-grouped clinical-EEG pipelines on leak-avoidance grounds, without
   claiming the gross-alignment mechanism.
4. **Honest scope:** the accuracy benefit is regime-bounded (M1b), the reservoir
   is not the disentangler (M5), and the gross-alignment mechanism is
   operator-scoped (blockwise M2).

## Reproduce

```text
python experiments/novel/verify_nuisance.py                       # machinery (no data)
python experiments/novel/run_nuisance_study.py --data3 ./batch_data --out nuisance_results.json
python experiments/novel/run_nuisance_study.py --data4 ./categories \
    --granularity four_category --out nuisance_results_4class.json
```

Note on reproducibility: the classification accuracies (N1, +4.36pp gap)
reproduce exactly across library versions; the SVD-based descriptive ρ can drift
at the ~1-unit level with the BLAS/LAPACK build (e.g. global ρ_pca 44.4 → 43.55
under sklearn 1.9 / numpy 2.4 / scipy 1.17). The committed JSONs and manifest
reflect the values reproduced by the pinned committed pipeline in CI's
environment; the qualitative conclusions are invariant to that drift.

External replication: one independent cohort (TCRZEM, `experiments/external/`)
**is committed** and replicates the accuracy consequence (+5.0pp) and the global
nuisance-alignment. Open items before submission: the full (non-fast) sweep, and
one or more *additional* external cohorts to further establish generality of the
mild-regime accuracy consequence.
