# Short-paper outline — "Random projection beats PCA on subject-grouped EEG"

A concrete skeleton assembled from the committed results. Target: a methods
short paper / letter (e.g. *IEEE Signal Processing Letters*, an ML-for-health
workshop, or a NeurIPS/ICML workshop on trustworthy ML for health). This
outline reports only what the experiments actually show, including the external
replication (TCRZEM, N=228) and the honest scope of each claim.

**Working title:** *Unsupervised compression aligns to the grouping nuisance:
why a data-independent random projection outperforms PCA on subject-grouped
clinical EEG.*

## Claim (one sentence)
On participant-grouped EEG, unsupervised variance-maximizing compression (PCA)
aligns its leading components to the between-subject nuisance subspace by
construction; a fixed data-independent random projection avoids that alignment,
so it is leak-free **and**, under aggressive compression, more discriminative.

## 1. Motivation
- Grouped clinical EEG is dominated by between-subject variance (here ρ ≈ 8–17
  subject:condition). The field's default reduce-then-classify pipeline fits PCA
  — often globally, which also leaks — before a grouped readout.
- Question: is a *fitted* basis even the right choice when the fitted variance is
  the nuisance? Contribution: a mechanism + a practical consequence + honest
  bounds.

## 2. Setup
- Dataset: SHAPE affective EEG; two independent labellings (3-condition 211/633,
  4-category 210/840). Fixed LIF-reservoir BSC6 features (1,536/electrode).
- Estimator: 5×5 participant-grouped `StratifiedGroupKFold`, fold-local
  standardization, linear readout; participant-cluster bootstrap intervals.
- Compressors compared at matched output dimension: fixed sparse random
  projection (SRP), **fold-local** PCA (leak-free comparator), supervised in-fold
  LDA, random coordinate subsampling.

## 3. Result 1 — accuracy (three datasets)
- SRP-64 > fold-local PCA-64 on SHAPE: **+4.4pp [1.5, 7.2]** (3-cond), **+3.9pp
  [1.5, 6.4]** (4-cat); and on the **external TCRZEM IAPS cohort (N=228,
  different lab): +5.0pp [2.0, 8.0]** at dim 64 (+6.7/+5.6 at 16/32). Interval
  excludes zero in every case, at every dimension.
- Figure: dimensionality sweep, SRP vs PCA with bootstrap bands, all cohorts.
- Table: SRP-64 seed-stability (64.0±1.0 / 54.1±1.1) — not a lucky projection.

## 4. Result 2 — mechanism (real + controlled)
- **Real:** PCA-64 minimum principal angle to the subject subspace **1.3–1.6°**
  on SHAPE and **0.9°** on the external TCRZEM cohort (vs 3.3–21° to condition);
  compressed subject:condition ρ inflated by PCA (SHAPE 32–44 vs ambient 11–17;
  TCRZEM 133 vs 111) while SRP stays at/below ambient.
- **Controlled:** as nuisance grows, the PCA subspace rotates monotonically
  toward subject (26.6°→0.3°) and away from condition (0.7°→29.4°). Dataset-
  independent (PCA maximizes variance = nuisance once nuisance dominates).
- Figure: principal-angle rotation curve (controlled) + real-data angle/ρ bars.

## 5. Result 3 — boundary and honest nulls
- **Regime dependence (M1b/M3):** the *accuracy* win requires compression
  aggressive enough to bury condition below the retained nuisance spectrum; in
  mild regimes fold-local PCA is the best compressor. State the boundary; do not
  claim universality.
- **Reservoir is not the disentangler (M5):** raw-ERP ρ ≤ BSC6 ρ on both
  cohorts; any ρ reduction comes from the isotropic projection, not the
  reservoir.
- **Necessity map (N3):** temporal-bin order is the load-bearing factor
  (−26 to −28pp when destroyed); reported for interpretability, not as the paper's
  main claim.

## 6. Practical recommendation
- On subject-grouped clinical EEG, prefer a fixed data-independent projection to
  a fitted PCA front-end: it removes a transductive-leakage foot-gun and, under
  realistic aggressive compression, improves grouped accuracy.

## 7. Limitations (state plainly)
- **External replication done; absolute accuracy is dataset-specific.** The
  central claims replicate on an independent IAPS cohort (TCRZEM, N=228,
  different lab): SRP > PCA (+5–6.7 pp, CI excludes zero) and PCA→subject angle
  0.9°. But absolute valence-decoding accuracy there is near three-class chance
  (SRP 45.5%), because that cohort is far more subject-dominated (ambient ρ≈111
  vs SHAPE's ≈11). The *comparative* compression result and the *mechanism* are
  what generalize; absolute affect decoding is not the claim.
- Linear readout only; one feature family (reservoir BSC6). The mechanism
  argument extends to any high-dim grouped feature space but is not yet shown
  there.
- Reservoir operating point and arousal-pooling were fixed to match SHAPE, not
  tuned to the external cohort.

## 8. Reproducibility
- All estimators verified on synthetic data without the restricted dataset
  (`verify_novel.py` 13/13, `verify_nuisance.py` 10/10, in CI); every number is
  in `results_manifest.json` with its generating script; no participant-level
  data is distributed.

## Figures / tables checklist
- [x] F1 dimensionality sweep SRP vs PCA (both cohorts) — data in `novel_results_*.json`
- [x] F2 principal-angle rotation (controlled) — `nuisance_results.json` M1a
- [x] F3 real-data angle + ρ bars (both cohorts) — `nuisance_results*.json` M2
- [x] T1 compressor comparison — M3
- [x] F4 external-cohort replication (TCRZEM, N=228) — external_results.json

## What is NOT claimed
- Not that ARSPI-Net is an accuracy leader (it is not; raw ERP is stronger).
- Not that SRP always beats PCA (regime-bounded).
- Not any clinical or biomarker claim.
