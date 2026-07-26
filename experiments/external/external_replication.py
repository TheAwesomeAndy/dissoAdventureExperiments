#!/usr/bin/env python3
"""
ARSPI-Net -- External-cohort replication harness (dataset-agnostic core)
=======================================================================

Replicates the two dataset-independent claims from experiments/novel/ on an
EXTERNAL affective-picture ERP EEG dataset:

  N1  a fixed data-independent SRP-64 vs a fold-local data-fitted PCA-64 under
      participant-grouped, fold-local cross-validation (does the random
      projection still win?), and
  M2  the geometric mechanism -- does PCA-64 align to the subject-nuisance
      subspace (small principal angle) on the new cohort too?

The reservoir / BSC6 / SRP / grouped-CV / geometry operators are imported from
the committed novel + confirmatory code, so nothing about the *method* changes
for an external dataset. Only the LOADER is dataset-specific: each external
dataset provides a function returning trial-averaged epochs in the canonical
shape this harness consumes:

    raw   : float array (n_observations, TARGET_T=256, n_channels)
            one trial-averaged ERP per (subject, condition)
    y     : int array (n_observations,)  condition label per observation
    groups: int array (n_observations,)  participant id per observation

`prepare_epochs()` builds that canonical form from raw single-trial epochs by
averaging trials within each (subject, condition) and resampling the time axis
to 256 samples -- matching how the SHAPE ERPs were formed. Channel count may
differ from SHAPE's 34; the pipeline is channel-count agnostic (SRP and the
grouped readout adapt), so no montage matching is required for these two
claims.

verify_external.py exercises this harness on synthetic epoched data with no
external download, so the machinery is CI-checkable before any dataset arrives.
"""

import os
import sys

import numpy as np
from scipy.signal import resample

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_REPO, "experiments", "novel"),
           os.path.join(_REPO, "experiments", "confirmatory"),
           os.path.join(_REPO, "chapter5Experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import confirmatory_pipeline as cp   # noqa: E402
import novel_pipeline as nov         # noqa: E402
import nuisance_study as ns          # noqa: E402

TARGET_T = 256


def prepare_epochs(epochs, trial_conditions, trial_subjects, target_t=TARGET_T,
                   zscore=True):
    """Build canonical trial-averaged ERPs from single-trial epochs.

    Parameters
    ----------
    epochs : (n_trials, time, n_channels) single-trial EEG.
    trial_conditions : (n_trials,) condition label per trial.
    trial_subjects   : (n_trials,) subject id per trial.

    Returns (raw, y, groups) with one trial-averaged, time-resampled, per-channel
    z-scored ERP per (subject, condition) that has at least one trial.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    trial_conditions = np.asarray(trial_conditions)
    trial_subjects = np.asarray(trial_subjects)
    if epochs.ndim != 3:
        raise ValueError("epochs must be (n_trials, time, n_channels)")

    cond_levels = sorted(set(trial_conditions.tolist()))
    cond_map = {c: i for i, c in enumerate(cond_levels)}

    raw, y, groups = [], [], []
    for subj in sorted(set(trial_subjects.tolist())):
        for cond in cond_levels:
            mask = (trial_subjects == subj) & (trial_conditions == cond)
            if not mask.any():
                continue
            erp = epochs[mask].mean(axis=0)                 # (time, n_ch)
            if erp.shape[0] != target_t:
                erp = resample(erp, target_t, axis=0)       # match SHAPE length
            if zscore:
                mu = erp.mean(axis=0, keepdims=True)
                sd = erp.std(axis=0, keepdims=True)
                erp = np.divide(erp - mu, sd, out=np.zeros_like(erp), where=sd > 0)
            raw.append(erp)
            y.append(cond_map[cond])
            groups.append(subj)
    return np.asarray(raw), np.asarray(y), np.asarray(groups, dtype=object)


def _encode_groups(groups):
    uniq = {g: i for i, g in enumerate(sorted(set(np.asarray(groups).tolist())))}
    return np.asarray([uniq[g] for g in groups], dtype=int)


def replicate(raw, y, groups, base_seed=0, dims=(16, 32, 64), reservoir_seed=42):
    """Run N1 (SRP vs fold-local PCA) and M2 (geometry) on a prepared cohort."""
    y = np.asarray(y)
    groups = _encode_groups(groups)
    n_ch = raw.shape[2]

    bsc = cp.build_bsc_features(raw, reservoir_seed=reservoir_seed)
    n1 = {}
    for d in dims:
        srp = nov.evaluate_srp(bsc, y, groups, d, cp.SRP_SEED, base_seed)
        pca = nov.evaluate_foldlocal_pca(bsc, y, groups, d, base_seed)
        diff_ci = cp.participant_bootstrap_difference_ci(
            srp["oof_true"], srp["oof_pred"], pca["oof_pred"], srp["oof_group"])
        n1[str(d)] = {
            "srp_percent": nov.pct(srp["balanced_accuracy"]),
            "foldlocal_pca_percent": nov.pct(pca["balanced_accuracy"]),
            "srp_minus_pca_pp": round(nov.pct(srp["balanced_accuracy"])
                                      - nov.pct(pca["balanced_accuracy"]), 3),
            "srp_minus_pca_ci95_pp": [round(v * 100, 3) for v in diff_ci],
        }

    bsc_flat = bsc.reshape(bsc.shape[0], -1)
    geom = ns.compression_geometry(bsc_flat, y, groups, n_components=max(dims))
    return {
        "n_participants": int(len(np.unique(groups))),
        "n_observations": int(len(y)),
        "n_conditions": int(len(np.unique(y))),
        "n_channels": int(n_ch),
        "N1_srp_vs_pca": n1,
        "M2_geometry": geom,
    }
