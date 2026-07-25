#!/usr/bin/env python3
"""
ARSPI-Net canonical confirmatory pipeline.

The module implements the corrected participant-grouped protocol used by the
confirmatory runner:

* five repeats of five-fold participant-grouped cross-validation;
* fold-local feature standardization and linear readout fitting;
* a fixed, data-independent SRP-64 projection shared across electrodes;
* participant-cluster bootstrap intervals over all repeated out-of-fold
  predictions;
* paired participant-bootstrap intervals for representation contrasts;
* within-participant permutation testing; and
* temporal and electrode destruction controls.

The SHAPE data are restricted and are not distributed with the repository.
Synthetic verification is provided by verify_confirmatory.py. Exact numerical
reproduction requires authorized data supplied to run_confirmatory_validation.py.
"""

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _path in (
    os.path.join(_REPO, "chapter5Experiments"),
    os.path.join(_REPO, "experiments", "ch5_4class"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

N_RES = 256
N_BINS = 6
N_SRP = 64
N_REPEATS = 5
N_FOLDS = 5
RESERVOIR_SEEDS = (7, 42, 99)
SRP_SEED = 20240517
BOOTSTRAP_SEED = 12345
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 100


def make_srp(n_features_per_electrode, seed=SRP_SEED, n_components=N_SRP):
    """Construct the fixed data-independent sparse random projection."""
    if n_features_per_electrode <= 0:
        raise ValueError("n_features_per_electrode must be positive")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    srp = SparseRandomProjection(n_components=n_components, random_state=seed)
    marker = np.ones((2, n_features_per_electrode), dtype=np.float64)
    srp.fit(marker)
    return srp


def project_bsc(bsc_per_electrode, srp):
    """Apply one shared SRP to each electrode and concatenate the blocks."""
    bsc = np.asarray(bsc_per_electrode, dtype=np.float64)
    if bsc.ndim != 3:
        raise ValueError("bsc_per_electrode must have shape (N, channels, features)")
    n_obs, n_ch, _ = bsc.shape
    out = np.empty((n_obs, n_ch, srp.n_components), dtype=np.float64)
    for ch in range(n_ch):
        out[:, ch, :] = srp.transform(bsc[:, ch, :])
    return out.reshape(n_obs, n_ch * srp.n_components)


def build_bsc_features(raw_data, reservoir_seed):
    """Generate electrode-resolved BSC6 features with the validated LIF operator."""
    from experiment_zero import LIFReservoir, bsc6_encode

    raw = np.asarray(raw_data, dtype=np.float64)
    if raw.ndim != 3:
        raise ValueError("raw_data must have shape (observations, time, channels)")
    n_obs, _, n_ch = raw.shape
    reservoirs = {
        ch: LIFReservoir(seed=reservoir_seed + ch * 17) for ch in range(n_ch)
    }
    bsc = np.empty((n_obs, n_ch, N_BINS * N_RES), dtype=np.float64)
    for i in range(n_obs):
        for ch in range(n_ch):
            spikes = reservoirs[ch].forward(raw[i, :, ch])
            bsc[i, ch, :] = bsc6_encode(spikes, n_bins=N_BINS)
    return bsc


def _validate_evaluation_inputs(X, y, groups, n_folds):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must contain the same number of rows")
    if len(np.unique(groups)) < n_folds:
        raise ValueError("number of participant groups is smaller than n_folds")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values")
    return X, y, groups


def evaluate_features(
    X,
    y,
    groups,
    n_repeats=N_REPEATS,
    n_folds=N_FOLDS,
    base_seed=0,
):
    """Evaluate features with repeated participant-grouped fold-local CV.

    The returned out-of-fold arrays contain every observation once per repeat.
    This implements the dissertation's pooled repeated-prediction estimand rather
    than retaining a single representative repeat.
    """
    X, y, groups = _validate_evaluation_inputs(X, y, groups, n_folds)
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")

    per_repeat = []
    pooled_pred = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=base_seed + repeat,
        )
        fold_pred = np.full(len(y), -1, dtype=int)
        for train_idx, test_idx in splitter.split(X, y, groups):
            scaler = StandardScaler().fit(X[train_idx])
            classifier = LogisticRegression(
                max_iter=2000,
                C=1.0,
                random_state=base_seed + repeat,
            )
            classifier.fit(scaler.transform(X[train_idx]), y[train_idx])
            fold_pred[test_idx] = classifier.predict(scaler.transform(X[test_idx]))
        if np.any(fold_pred < 0):
            raise RuntimeError("at least one observation did not receive an OOF prediction")
        per_repeat.append(float(balanced_accuracy_score(y, fold_pred)))
        pooled_pred.append(fold_pred)

    oof_pred = np.concatenate(pooled_pred)
    oof_true = np.tile(y, n_repeats)
    oof_group = np.tile(groups, n_repeats)
    return {
        "balanced_accuracy": float(np.mean(per_repeat)),
        "per_repeat": per_repeat,
        "oof_pred": oof_pred,
        "oof_true": oof_true,
        "oof_group": oof_group,
        "n_repeats": int(n_repeats),
    }


def _participant_bootstrap_indices(groups, n_boot, seed):
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.size == 0:
        raise ValueError("groups cannot be empty")
    indices_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        yield np.concatenate([indices_by_group[g] for g in sampled])


def participant_bootstrap_ci(
    oof_true,
    oof_pred,
    oof_group,
    n_boot=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
    alpha=0.05,
):
    """Participant-cluster bootstrap interval for balanced accuracy."""
    true = np.asarray(oof_true)
    pred = np.asarray(oof_pred)
    groups = np.asarray(oof_group)
    if not (len(true) == len(pred) == len(groups)):
        raise ValueError("bootstrap arrays must be aligned")
    stats = [
        balanced_accuracy_score(true[idx], pred[idx])
        for idx in _participant_bootstrap_indices(groups, n_boot, seed)
    ]
    return [
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    ]


def participant_bootstrap_difference_ci(
    oof_true,
    oof_pred_a,
    oof_pred_b,
    oof_group,
    n_boot=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
    alpha=0.05,
):
    """Paired participant-bootstrap interval for BA(A) minus BA(B)."""
    true = np.asarray(oof_true)
    pred_a = np.asarray(oof_pred_a)
    pred_b = np.asarray(oof_pred_b)
    groups = np.asarray(oof_group)
    if not (len(true) == len(pred_a) == len(pred_b) == len(groups)):
        raise ValueError("paired bootstrap arrays must be aligned")
    stats = []
    for idx in _participant_bootstrap_indices(groups, n_boot, seed):
        stats.append(
            balanced_accuracy_score(true[idx], pred_a[idx])
            - balanced_accuracy_score(true[idx], pred_b[idx])
        )
    return [
        float(np.percentile(stats, 100 * alpha / 2)),
        float(np.percentile(stats, 100 * (1 - alpha / 2))),
    ]


def permutation_test(
    X,
    y,
    groups,
    n_perm=N_PERMUTATIONS,
    n_repeats=2,
    n_folds=N_FOLDS,
    seed=0,
):
    """Within-participant label permutation test with a deterministic ridge readout."""
    X, y, groups = _validate_evaluation_inputs(X, y, groups, n_folds)

    def ridge_ba(labels):
        repeat_scores = []
        for repeat in range(n_repeats):
            splitter = StratifiedGroupKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=seed + repeat,
            )
            pred = np.full(len(labels), -1, dtype=int)
            for train_idx, test_idx in splitter.split(X, labels, groups):
                scaler = StandardScaler().fit(X[train_idx])
                classifier = RidgeClassifier()
                classifier.fit(scaler.transform(X[train_idx]), labels[train_idx])
                pred[test_idx] = classifier.predict(scaler.transform(X[test_idx]))
            repeat_scores.append(balanced_accuracy_score(labels, pred))
        return float(np.mean(repeat_scores))

    observed = ridge_ba(y)
    rng = np.random.RandomState(seed + 777)
    unique_groups = np.unique(groups)
    exceedances = 0
    for _ in range(n_perm):
        permuted = y.copy()
        for group in unique_groups:
            idx = np.flatnonzero(groups == group)
            permuted[idx] = rng.permutation(y[idx])
        if ridge_ba(permuted) >= observed:
            exceedances += 1
    return {
        "observed_balanced_accuracy": observed,
        "p_value": float((1 + exceedances) / (1 + n_perm)),
        "n_permutations": int(n_perm),
    }


def destroy_temporal_bin_order(bsc, rng):
    """Shuffle the six temporal-bin blocks within each observation and electrode."""
    bsc = np.asarray(bsc)
    n_obs, n_ch, feature_dim = bsc.shape
    out = bsc.reshape(n_obs, n_ch, N_BINS, N_RES).copy()
    for i in range(n_obs):
        for ch in range(n_ch):
            out[i, ch] = out[i, ch][rng.permutation(N_BINS)]
    return out.reshape(n_obs, n_ch, feature_dim)


def destroy_temporal_collapse(bsc):
    """Replace each temporal bin by the per-neuron mean across bins."""
    bsc = np.asarray(bsc)
    n_obs, n_ch, feature_dim = bsc.shape
    blocks = bsc.reshape(n_obs, n_ch, N_BINS, N_RES)
    mean_block = blocks.mean(axis=2, keepdims=True)
    return np.repeat(mean_block, N_BINS, axis=2).reshape(n_obs, n_ch, feature_dim)


def destroy_electrode_identity(bsc, rng):
    """Shuffle electrode blocks independently within each observation."""
    bsc = np.asarray(bsc)
    n_obs, n_ch, _ = bsc.shape
    out = bsc.copy()
    for i in range(n_obs):
        out[i] = out[i][rng.permutation(n_ch)]
    return out


TEMPORAL_WINDOWS = {
    "39-273ms": (0.00, 0.25),
    "273-508ms": (0.25, 0.50),
    "508-742ms": (0.50, 0.75),
    "742-977ms": (0.75, 1.00),
    "39-977ms_full": (0.00, 1.00),
}


def window_slice(raw_data, frac_lo, frac_hi):
    """Extract a fractional temporal window while retaining at least six samples."""
    raw = np.asarray(raw_data)
    if not (0.0 <= frac_lo < frac_hi <= 1.0):
        raise ValueError("window fractions must satisfy 0 <= lo < hi <= 1")
    time_steps = raw.shape[1]
    lo = int(round(frac_lo * time_steps))
    hi = int(round(frac_hi * time_steps))
    hi = max(hi, lo + N_BINS)
    return raw[:, lo:hi, :]
