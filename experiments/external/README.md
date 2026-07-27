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

## Dataset (downloaded and used)

**`doi:10.34894/TCRZEM`** (DataVerseNL) — IAPS passive viewing,
negative/neutral/positive valence, CC BY-NC 4.0. The **preprocessed** release is
BrainVision segmented files, one per subject × valence × arousal
(`S<id>_Mastoid_<Neg|Neu|Pos>_<High|Low>.{vhdr,dat}`); each `.dat` is INT_16,
MULTIPLEXED, `NumberOfChannels` channels, 1792-point segments at 512 Hz. The
channel count is **41** (including reference/EOG), different from SHAPE's 34; the
pipeline is channel-count agnostic, so no montage matching is done. Of the ~239
participants, the **228** with a complete negative/neutral/positive panel are
used (684 subject × condition ERPs). This is a genuinely different cohort in the
same paradigm — the external replication set.

## Files

| File | Purpose |
|---|---|
| `external_replication.py` | dataset-agnostic core: `prepare_epochs()` + `replicate()` (N1 + M2) |
| `verify_external.py` | synthetic verification of the harness (no download) — CI-safe |
| `load_tcrzem.py` | committed dataset-specific loader for `doi:10.34894/TCRZEM` (BrainVision `.vhdr/.dat`, memory-safe trial-averaging) |
| `run_tcrzem_replication.py` | committed runner that produces `external_results.json` |
| `external_results.json` | committed summary outputs (accuracies, angles, ρ) — no participant-level data |
| `EXTERNAL_RESULTS.md` | committed results narrative |

## Running

```text
python experiments/external/verify_external.py            # machinery, no data
# once the external epochs are staged (gitignored):
python experiments/external/run_tcrzem_replication.py --data ./external_data/tcrzem \
    --out external_results.json
```

## Scope and data handling

The external EEG lives in a **gitignored** directory and is never committed; only
the summary statistics in `external_results.json` are. `verify_external.py`
establishes code behavior; the empirical finding comes from the downloaded
cohort. This is an **exploratory** replication: on the 228-participant cohort the
committed run reproduces the N1 accuracy ordering (SRP-64 > fold-local PCA-64,
+5.0 pp [2.0, 8.0]) and exhibits the global-PCA subject alignment (see
`EXTERNAL_RESULTS.md`).
