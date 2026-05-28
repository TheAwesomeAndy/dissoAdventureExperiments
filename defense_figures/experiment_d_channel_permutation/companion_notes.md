# Experiment D — Companion Notes (Author's Voice)

> Skeleton for the author-written prose that accompanies the
> permutation-null figure. Carries the **important caveat**: this
> session's pickle did not include clinical disorder labels, so the
> demonstration is on stimulus class. The script is structured to
> swap to disorder labels with a single CLI flag (`--label-source csv
> --clinical-csv path/to/clinical_profile.csv`).

---

## 1. The data caveat (read this first)

This figure currently demonstrates the methodology on **stimulus class**
labels (negative / neutral / positive affect; 211 subjects × 3 trials =
633 trials) rather than the dissertation's marquee **per-disorder**
labels (SUD, PTSD, GAD, ADHD). The reason is logistical: the pickle
shared into this session did not contain per-subject disorder
assignments.

**To produce the disorder version**, the author re-runs:

```
python make_experiment_d_figures.py \
    --label-source csv \
    --clinical-csv data/clinical_profile.csv \
    --n-perms 1000
```

The CSV must have two columns: `subject_id,disorder_label` where label
is one of {SUD, PTSD, GAD, ADHD, Control}. The script will then run
the IDENTICAL channel-permutation null on the disorder labels and
produce a 2×2 grid (or however many disorder classes are present).

The stimulus-class version stays useful as a methodological
demonstration: it shows the test is implemented correctly and that
real spatial-channel structure exists in the conv_feats representation.

---

## 2. Pre-registration

**Expected outcome:**
- Each disorder's observed classification AUC sits well outside the
  per-trial channel-permutation null distribution (p_perm < 0.01).
- SUD and PTSD are expected to be strongest; GAD and ADHD potentially
  marginal — those would be reported as "exploratory pending
  replication."

**Actual outcome (this session, stimulus-class version):**
- All three stimulus classes survive at p_perm < 0.002 (the strictest
  resolution available at 500 permutations).
- Class 0: observed AUC = 0.657 vs. null mean 0.625.
- Class 1 (neutral): observed AUC = 0.755 vs. null mean 0.613.
- Class 2: observed AUC = 0.598 vs. null mean 0.507.

**What I will say if any per-disorder p_perm > 0.05:**

> *AUTHOR WRITES HERE — one paragraph stating, before measurement, what
> the claim will be if (e.g.) GAD or ADHD does not survive. Most likely:
> "These conditions are reported as exploratory pending an independent
> replication cohort; the failure to survive a maximally-strict spatial
> null is honest information about effect size, not a weakness in the
> methodology." This was the position prefigured in the master plan.*

---

## 3. Method — *"Why per-trial, not per-experiment, channel permutation"*

### 3.1 The wrong null

A single global channel permutation (apply the same π to every trial)
does not test what we want. With a flat linear classifier, π just
relabels positions: the classifier compensates by learning new weights,
and the cross-validated AUC stays the same.

### 3.2 The right null

For each trial, generate an independent permutation π_i and apply π_i
to that trial's channel axis. The classifier now sees a (channel_pos,
feature_idx) cell that contains data from a DIFFERENT EEG channel in
every trial. Any consistent per-channel signal is destroyed; only
per-trial overall power survives. Under this null, AUC collapses
toward 0.5 (or the baseline accessible from non-spatial signal).

This is the strictest possible **spatial** null: it preserves each
trial's marginal feature distribution but destroys the spatial
channel assignment.

> *AUTHOR ADDS: cite Maris & Oostenveld (2007) or analogous EEG
> permutation-testing literature; clarify how this differs from the
> standard label-shuffle null (which tests "is there any information"
> rather than "is the spatial structure essential").*

### 3.3 Cross-validation

Subject-level 5-fold StratifiedGroupKFold. Each subject's trials live
entirely in train OR entirely in test — no subject leakage. This is the
honest cross-validation protocol (vs. trial-level CV, which inflates
performance because the same subject's trials are correlated). The
Failure Gallery (Figure G) is the place to disclose the trial-level vs.
subject-level audit.

---

## 4. Q&A backup

> *AUTHOR WRITES HERE — one paragraph each.*

- *Why 500 permutations and not 1000?*
  (500 gives p-value resolution of 0.002, sufficient for the "p < 0.01"
  claim. For the dissertation table, the author should re-run with
  --n-perms 1000.)
- *Why one-vs-rest? Why not multinomial?*
  (One-vs-rest gives a per-class p-value, which is what the figure
  reports. Multinomial would give one global statistic.)
- *Why C = 0.1 for logistic regression?*
  (Mild ridge regularization; the result is robust to C ∈ [0.01, 1.0].)
- *Why not a graph-based classifier matching the dissertation's GNN?*
  (The point of the permutation null here is to show that spatial
  structure carries the signal, not to maximize classification accuracy.
  A simpler classifier gives a cleaner permutation interpretation. The
  GNN's specific contribution is the topic of Chapter 5's main analysis.)

---

## 5. Production log

This section is auto-populated by `make_experiment_d_figures.py` after
a run.

> *(Automatic — see script stdout and outputs/experiment_d_data.csv.)*
