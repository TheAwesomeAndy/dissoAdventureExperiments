# External-cohort replication

Replicates the two dataset-independent claims from `experiments/novel/` on an
**external** affective-picture ERP EEG dataset — a genuinely different cohort in
the same paradigm (IAPS passive viewing, negative/neutral/positive valence):

- **N1** — does a fixed data-independent SRP-64 still beat a fold-local
  data-fitted PCA-64 under participant-grouped CV?
- **M2** — does PCA-64 still align to the subject-nuisance subspace (small
  principal angle)?

The reservoir, BSC6, SRP, grouped-CV, and geometry operators are the committed
ones; only the dataset loader is new. `external_replication.prepare_epochs()`
turns single-trial epochs into the canonical trial-averaged, 256-sample,
per-channel z-scored ERP-per-(subject,condition) form the pipeline consumes —
matching how the SHAPE ERPs were made. The pipeline is channel-count agnostic,
so no montage matching is needed for these claims.

## Candidate dataset (identified, pending download)

**`doi:10.34894/TCRZEM`** (DataVerseNL) — "EEG data of 239 participants during
passive viewing of IAPS images": 239 subjects, 150 IAPS images across
negative/neutral/positive valence, 32-channel BioSemi, raw + preprocessed
epoched EEG (−1000..+2500 ms), EEGLAB `.mat`, CC BY-NC 4.0. Same paradigm as
SHAPE, different (larger) cohort — the right external replication set.

## Files

| File | Purpose |
|---|---|
| `external_replication.py` | dataset-agnostic core: `prepare_epochs()` + `replicate()` (N1 + M2) |
| `verify_external.py` | synthetic verification of the harness (no download) — CI-safe |
| `load_tcrzem.py` | dataset-specific loader for `doi:10.34894/TCRZEM` (added at download time) |

## Running

```text
python experiments/external/verify_external.py            # machinery, no data
# once the external epochs are staged (gitignored):
python experiments/external/run_tcrzem_replication.py --data ./external_data/tcrzem \
    --out external_results.json
```

## Scope and data handling

The external EEG is downloaded to a **gitignored** directory and never
committed; only summary statistics (accuracies, angles, ρ) are recorded. Like
the rest of the repo, `verify_external.py` establishes code behavior, not the
empirical finding, which requires the downloaded cohort. This is an
**exploratory** replication — a positive result strengthens the generality of
the compression-nuisance claim; a null would bound it, and either is reported
honestly.
