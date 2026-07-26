# External-cohort replication — results

The two dataset-independent claims replicate on a **genuinely external** IAPS
affective-picture ERP cohort: **DataVerseNL `doi:10.34894/TCRZEM`** (Revers et
al.), 228 complete-panel participants, different lab, different acquisition
(BioSemi, 41 channels), same paradigm (passive IAPS viewing, negative / neutral
/ positive valence). Machine-readable:
[`external_results.json`](external_results.json). Exploratory; passing tests
verify code behavior, not scientific validity.

## N1 — a data-independent random projection beats data-fitted PCA (replicated)

Balanced accuracy, 5×5 participant-grouped fold-local CV; paired
participant-bootstrap 95% interval on the difference.

| Dim | SRP | fold-local PCA | SRP − PCA | 95% interval |
|---:|---:|---:|---:|---:|
| 16 | 43.7% | 37.0% | **+6.7 pp** | [3.3, 10.2] |
| 32 | 43.7% | 38.1% | **+5.6 pp** | [2.3, 8.9] |
| 64 | 45.5% | 40.5% | **+5.0 pp** | [2.0, 8.0] |

The interval excludes zero at every dimension — the same direction and a
**larger** effect than on SHAPE (+3.9–4.4 pp).

## M2 — PCA aligns to the subject-nuisance subspace (replicated, starker)

| Quantity | SHAPE (211) | TCRZEM external (228) |
|---|---:|---:|
| PCA-64 min principal angle → **subject** | 1.32° | **0.89°** |
| PCA-64 min principal angle → condition | 3.32° | 21.15° |
| ρ compressed: PCA / SRP / ambient | 44 / 7 / 11 | 133 / 111 / 111 |

On the external cohort PCA-64 is **within 0.9° of the subject subspace** and
21° from the condition subspace — even more collinear with the nuisance than on
SHAPE.

## Honest reading of the absolute accuracies

Absolute balanced accuracies here are **lower and near three-class chance**
(33.3%): SRP-64 reaches 45.5%. This is expected and consistent with the theory,
not a contradiction:

- The external cohort is **far more subject-dominated** (ambient ρ ≈ **111** vs
  SHAPE's ≈ 11). The condition (valence) signal is a much smaller fraction of
  the variance, so *every* linear decoder is closer to chance.
- That extreme nuisance dominance is exactly the regime in which the
  nuisance-alignment mechanism predicts the **largest** PCA penalty — and indeed
  the SRP-over-PCA gap is *bigger* here (+5–6.7 pp) and the PCA→subject angle is
  *smaller* (0.89°).
- The reservoir operating point and the arousal-pooling were fixed to match
  SHAPE, not tuned to this cohort; the point of the external run is the
  **relative** compression comparison and the mechanism, both of which hold.

So the *comparative* claims (SRP > data-fitted PCA; PCA aligns to nuisance)
replicate and strengthen on an independent cohort, while the *absolute* affect-
decoding accuracy is dataset-specific and low here — reported plainly.

## What this establishes

The paper's single-dataset limitation is resolved for its central claims:

> Across two independent IAPS affective-picture EEG cohorts (SHAPE, N=211;
> Revers/TCRZEM, N=228) — different labs, montages, and acquisition systems —
> a fixed data-independent random projection significantly outperforms a
> fold-local data-fitted PCA under participant-grouped evaluation (+4–7 pp,
> intervals excluding zero), because PCA's leading components align to the
> between-subject nuisance subspace (principal angle 0.9–1.3°). The effect is
> strongest where the subject nuisance is most dominant.

## Reproduce

```text
python experiments/external/verify_external.py            # machinery, no data
python experiments/external/run_tcrzem_replication.py \
    --data ./external_data/tcrzem --out external_results.json
```

The external EEG is restricted-download (CC BY-NC 4.0) and gitignored; only the
summary statistics above are committed.
