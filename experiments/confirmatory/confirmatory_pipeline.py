#!/usr/bin/env python3
"""
ARSPI-Net -- Canonical Confirmatory Pipeline (library)
======================================================

This module is the single source of truth for the corrected *confirmatory*
classification results reported in the dissertation (chgraph.tex, main.tex
abstract). It implements exactly the protocol the dissertation describes and
nothing that would leak held-out information:

  * Five repeats of five-fold PARTICIPANT-GROUPED cross-validation. Every
    participant's complete condition panel stays inside one fold.
  * FOLD-LOCAL preprocessing: per-feature standardization is fitted on the
    training participants of each fold only.
  * A fixed, DATA-INDEPENDENT sparse random projection (SRP-64) shared across
    electrodes, generated from a prespecified seed. It never estimates a basis
    from the observations, so held-out participants cannot influence it. This
    is the correction that replaces the archived globally fitted PCA-64.
  * A linear logistic-regression readout for accuracy.
  * A separate deterministic ridge readout for within-participant label
    permutation tests.
  * Participant-bootstrap 95% intervals over the repeated out-of-fold
    predictions.
  * Mechanistic destruction controls (temporal-bin shuffle, temporal collapse,
    electrode-identity shuffle).
  * Robustness across three reservoir seeds and temporal-window ablations.

The reservoir, BSC6 encoder, and data loaders are imported from the already
validated scripts so this pipeline uses the exact same operators the rest of
the repository is verified against -- it does not re-implement them.

The numeric outputs are written as machine-readable JSON and can be compared
automatically against results_manifest.json by check_against_manifest.py.

NOTE ON DATA. The SHAPE dataset is restricted and is not part of this
repository. The real-data entry point is run_confirmatory_validation.py. The
methodological correctness of everything in this module (SRP data-independence,
fold-local fitting, participant grouping, determinism, output schema) is
verified WITHOUT the dataset by verify_confirmatory.py on synthetic data, so
the machinery is testable in CI. Reproducing the exact headline numbers
requires feeding the authorized SHAPE data to the runner.
"""

import os
import sys
import numpy as np

from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

# --- Canonical operators reused from the validated scripts -----------------
# We add the sibling experiment directories to sys.path and import the exact
# reservoir + BSC6 + loaders used elsewhere in the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (
    os.path.join(_REPO, "chapter5Experiments"),
    os.path.join(_REPO, "experiments", "ch5_4class"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# Protocol constants (match the dissertation and the rest of the repo)
# ============================================================================
N_RES = 256          # reservoir size (Ch4 validated operating point)
N_BINS = 6           # BSC6 temporal bins
N_SRP = 64           # SRP output dimension per electrode
N_REPEATS = 5        # 5 repeats ...
N_FOLDS = 5          # ... of 5-fold participant-grouped CV
RESERVOIR_SEEDS = (7, 42, 99)   # robustness seed set
SRP_SEED = 20240517  # prespecified SRP seed (data-independent)
BOOTSTRAP_SEED = 12345
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 100
CENTER_WINDOW = None  # temporal-window ablation handled explicitly


# ============================================================================
# Fixed, data-independent SRP-64 (shared across electrodes)
# ============================================================================
def make_srp(n_features_per_electrode, seed=SRP_SEED, n_components=N_SRP):
    """Return a *fixed* SparseRandomProjection.

    The projection is fitted on a synthetic all-ones marker of the correct
    width so scikit-learn allocates the sparse matrix, but its entries depend
    ONLY on `seed` and the dimensionality -- never on the observations. The
    same projection is applied to every electrode. Because it is seeded and
    data-independent, held-out participants cannot influence it, which is the
    property the corrected protocol requires.
    """
    srp = SparseRandomProjection(n_components=n_components, random_state=seed)
    # `fit` on a shape-only marker: SparseRandomProjection.fit uses only
    # n_features and random_state to build the matrix; the values of X are
    # ignored. We pass a 2-row marker to satisfy the API.
    marker = np.ones((2, n_features_per_electrode), dtype=np.float64)
    srp.fit(marker)
    return srp


def project_bsc(bsc_per_electrode, srp):
    """Project (N, n_ch, 1536) BSC6 features to (N, n_ch*64) via shared SRP.

    Applies the SAME fixed projection to each electrode, then concatenates,
    preserving electrode identity (so the electrode-shuffle control is
    meaningful).
    """
    N, n_ch, d = bsc_per_electrode.shape
    out = np.zeros((N, n_ch, srp.n_components), dtype=np.float64)
    for ch in range(n_ch):
        out[:, ch, :] = srp.transform(bsc_per_electrode[:, ch, :])
    return out.reshape(N, n_ch * srp.n_components)


# ============================================================================
# Reservoir + BSC6 feature construction
# ============================================================================
def build_bsc_features(raw_data, reservoir_seed):
    """Drive the validated LIF reservoir on (N, T, n_ch) EEG and return
    per-electrode BSC6 features of shape (N, n_ch, N_BINS*N_RES).

    Uses the exact LIFReservoir + bsc6_encode from experiment_zero.py so the
    operators are identical to the verified ones. One reservoir per channel,
    seeded deterministically from `reservoir_seed`.
    """
    from experiment_zero import LIFReservoir, bsc6_encode  # canonical operators

    N, T, n_ch = raw_data.shape
    reservoirs = {ch: LIFReservoir(seed=reservoir_seed + ch * 17)
                  for ch in range(n_ch)}
    bsc = np.zeros((N, n_ch, N_BINS * N_RES), dtype=np.float64)
    for i in range(N):
        for ch in range(n_ch):
            spikes = reservoirs[ch].forward(raw_data[i, :, ch])  # (T, N_RES)
            bsc[i, ch, :] = bsc6_encode(spikes, n_bins=N_BINS)
    return bsc


# ============================================================================
# Cross-validated evaluation (fold-local, participant-grouped)
# ============================================================================
def evaluate_features(X, y, groups, n_repeats=N_REPEATS, n_folds=N_FOLDS,
                      base_seed=0):
    """Return pooled out-of-fold (balanced accuracy, per-repeat list, oof
    predictions/labels/groups) under fold-local standardization + logistic
    readout, with 5x5 participant-grouped CV.

    Standardization is fitted on the training split ONLY inside each fold,
    which is the fold-local preprocessing boundary the corrected protocol
    requires. `groups` are participant ids; StratifiedGroupKFold keeps a
    participant entirely on one side of every split.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    per_repeat = []
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_true = y.copy()
    oof_group = groups.copy()
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                    random_state=base_seed + r)
        fold_pred = np.full(len(y), -1, dtype=int)
        for tr, te in sgkf.split(X, y, groups):
            scaler = StandardScaler().fit(X[tr])           # fold-local fit
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(scaler.transform(X[tr]), y[tr])
            fold_pred[te] = clf.predict(scaler.transform(X[te]))
        per_repeat.append(balanced_accuracy_score(y, fold_pred))
        if r == 0:
            oof_pred = fold_pred  # representative repeat for bootstrap pooling
    mean_ba = float(np.mean(per_repeat))
    return {
        "balanced_accuracy": mean_ba,
        "per_repeat": [float(v) for v in per_repeat],
        "oof_pred": oof_pred,
        "oof_true": oof_true,
        "oof_group": oof_group,
    }


def participant_bootstrap_ci(oof_true, oof_pred, oof_group,
                             n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
                             alpha=0.05):
    """Participant-bootstrap 95% interval for balanced accuracy: resample
    PARTICIPANTS (not observations) with replacement and recompute."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(oof_group)
    # index lists per participant
    idx_by_group = {g: np.where(oof_group == g)[0] for g in uniq}
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in chosen])
        try:
            stats.append(balanced_accuracy_score(oof_true[idx], oof_pred[idx]))
        except ValueError:
            continue
    lo = float(np.percentile(stats, 100 * (alpha / 2)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]


def permutation_test(X, y, groups, n_perm=N_PERMUTATIONS,
                     n_repeats=2, n_folds=N_FOLDS, seed=0):
    """Within-participant label-permutation test using a DETERMINISTIC ridge
    readout (separate from the logistic readout used for point estimates).

    Returns the one-sided p-value that observed balanced accuracy exceeds the
    permuted null. Labels are permuted WITHIN each participant, preserving the
    participant grouping.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)

    def _ridge_ba(labels):
        per = []
        for r in range(n_repeats):
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed + r)
            pred = np.full(len(labels), -1, dtype=int)
            for tr, te in sgkf.split(X, labels, groups):
                scaler = StandardScaler().fit(X[tr])
                clf = RidgeClassifier()
                clf.fit(scaler.transform(X[tr]), labels[tr])
                pred[te] = clf.predict(scaler.transform(X[te]))
            per.append(balanced_accuracy_score(labels, pred))
        return float(np.mean(per))

    observed = _ridge_ba(y)
    rng = np.random.RandomState(seed + 777)
    uniq = np.unique(groups)
    count = 0
    for _ in range(n_perm):
        perm = y.copy()
        for g in uniq:
            gi = np.where(groups == g)[0]
            perm[gi] = rng.permutation(y[gi])
        if _ridge_ba(perm) >= observed:
            count += 1
    p = (1 + count) / (1 + n_perm)
    return {"observed_balanced_accuracy": observed, "p_value": float(p),
            "n_permutations": n_perm}


# ============================================================================
# Destruction controls
# ============================================================================
def destroy_temporal_bin_order(bsc, rng):
    """Independently shuffle the six temporal bins within each observation and
    electrode. bsc: (N, n_ch, N_BINS*N_RES) laid out as [bin0..bin5] blocks of
    N_RES."""
    N, n_ch, d = bsc.shape
    out = bsc.reshape(N, n_ch, N_BINS, N_RES).copy()
    for i in range(N):
        for ch in range(n_ch):
            out[i, ch] = out[i, ch][rng.permutation(N_BINS)]
    return out.reshape(N, n_ch, d)


def destroy_temporal_collapse(bsc):
    """Collapse time: replace all six bins with the per-neuron mean across
    bins (removes temporal ordering AND resolution, keeps totals)."""
    N, n_ch, d = bsc.shape
    blk = bsc.reshape(N, n_ch, N_BINS, N_RES)
    mean = blk.mean(axis=2, keepdims=True)
    return np.repeat(mean, N_BINS, axis=2).reshape(N, n_ch, d)


def destroy_electrode_identity(bsc, rng):
    """Independently shuffle electrode order within each observation."""
    N, n_ch, d = bsc.shape
    out = bsc.copy()
    for i in range(N):
        out[i] = out[i][rng.permutation(n_ch)]
    return out


# ============================================================================
# Temporal-window ablation
# ============================================================================
TEMPORAL_WINDOWS = {
    "39-273ms": (0.00, 0.25),
    "273-508ms": (0.25, 0.50),
    "508-742ms": (0.50, 0.75),
    "742-977ms": (0.75, 1.00),
    "39-977ms_full": (0.00, 1.00),
}


def window_slice(raw_data, frac_lo, frac_hi):
    """Return the time-window slice of (N, T, n_ch) between fractional bounds."""
    T = raw_data.shape[1]
    lo, hi = int(round(frac_lo * T)), int(round(frac_hi * T))
    hi = max(hi, lo + N_BINS)  # ensure at least one sample per bin
    return raw_data[:, lo:hi, :]
