#!/usr/bin/env python3
"""
ARSPI-Net -- Novel experiments library
======================================

Three experiments that answer the central reviewer question for an
interpretability-first neuromorphic model that is deliberately *not* an
accuracy leader: if the raw ERP is more accurate, what does the fixed,
inspectable measurement chain buy you?

  N1  Leak-free compression is free. A fixed, data-independent sparse random
      projection (SRP-64) matches a data-fitted, fold-local PCA-64 under
      participant-grouped cross-validation, with a Johnson-Lindenstrauss
      dimensionality law and an SRP-seed-stability distribution. The projection
      the confirmatory pipeline already uses is therefore not a lucky draw and
      pays no accuracy cost for being leak-free by construction.

  N2  The reservoir behaves as a calibrated instrument. Its BSC6 features are
      reproducible across reservoir seeds (intraclass correlation), its
      input-amplitude transfer function is monotonic with a quantified dynamic
      range, and the downstream readout degrades gracefully under channel
      dropout and additive noise.

  N3  Representational necessity and subject/condition information. A necessity
      map measures, for each structural factor (temporal-bin order, temporal
      resolution, electrode identity), how much held-out condition accuracy is
      lost and how the subject-to-condition variance ratio (rho) shifts. This
      quantifies which structure is necessary for condition decoding and how it
      relates to the subject-covariance entanglement rho ~ 7.2 reported in the
      dissertation.

Everything reuses the committed confirmatory operators (the validated LIF
reservoir, BSC6 encoder, SRP-64, participant-grouped fold-local evaluation, and
participant-cluster bootstrap) so the novel results sit on the same verified
machinery as the confirmatory stream. The restricted SHAPE data are not
required to verify the machinery; verify_novel.py exercises every estimator on
synthetic data.
"""

import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (
    os.path.join(_REPO, "experiments", "confirmatory"),
    os.path.join(_REPO, "chapter5Experiments"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import confirmatory_pipeline as cp  # noqa: E402  (validated operators)

N_BINS = cp.N_BINS
N_RES = cp.N_RES


def pct(x):
    return round(float(x) * 100.0, 3)


# ===========================================================================
# N1 -- leak-free compression
# ===========================================================================
def evaluate_srp(bsc, y, groups, n_components, seed, base_seed=0):
    """Balanced accuracy for a fixed data-independent SRP-`n_components`."""
    srp = cp.make_srp(bsc.shape[2], seed=seed, n_components=n_components)
    X = cp.project_bsc(bsc, srp)
    return cp.evaluate_features(X, y, groups, base_seed=base_seed)


def evaluate_foldlocal_pca(bsc, y, groups, n_components, base_seed=0,
                           n_repeats=None, n_folds=None):
    """Balanced accuracy for a *fold-local* PCA-`n_components`.

    PCA is fitted on the training electrodes-concatenation of each fold only, so
    held-out participants never influence the basis. This is the leak-free way
    to run the data-fitted comparator against which SRP is measured; a global
    PCA would leak and is exactly what the archived pipeline did wrong.
    """
    n_repeats = cp.N_REPEATS if n_repeats is None else n_repeats
    n_folds = cp.N_FOLDS if n_folds is None else n_folds
    n_obs, n_ch, n_feat = bsc.shape
    flat = bsc.reshape(n_obs, n_ch * n_feat)
    y = np.asarray(y)
    groups = np.asarray(groups)

    per_repeat, pooled = [], []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=base_seed + repeat)
        pred = np.full(n_obs, -1, dtype=int)
        for tr, te in splitter.split(flat, y, groups):
            # fold-local PCA on the per-electrode blocks, fitted on train only
            pca = PCA(n_components=n_components, random_state=base_seed + repeat)
            # fit on stacked electrodes of training observations
            train_blocks = bsc[tr].reshape(len(tr) * n_ch, n_feat)
            pca.fit(train_blocks)
            Xtr = pca.transform(bsc[tr].reshape(len(tr) * n_ch, n_feat)
                                ).reshape(len(tr), n_ch * n_components)
            Xte = pca.transform(bsc[te].reshape(len(te) * n_ch, n_feat)
                                ).reshape(len(te), n_ch * n_components)
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     random_state=base_seed + repeat)
            clf.fit(scaler.transform(Xtr), y[tr])
            pred[te] = clf.predict(scaler.transform(Xte))
        per_repeat.append(float(balanced_accuracy_score(y, pred)))
        pooled.append(pred)
    return {
        "balanced_accuracy": float(np.mean(per_repeat)),
        "per_repeat": per_repeat,
        "oof_pred": np.concatenate(pooled),
        "oof_true": np.tile(y, n_repeats),
        "oof_group": np.tile(groups, n_repeats),
    }


def srp_seed_stability(bsc, y, groups, seeds, n_components=None, base_seed=0):
    """Distribution of balanced accuracy across many SRP seeds."""
    n_components = cp.N_SRP if n_components is None else n_components
    accs = [evaluate_srp(bsc, y, groups, n_components, s, base_seed)["balanced_accuracy"]
            for s in seeds]
    accs = np.asarray(accs)
    return {
        "n_seeds": len(seeds),
        "mean": pct(accs.mean()),
        "std": pct(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "min": pct(accs.min()),
        "max": pct(accs.max()),
        "range_pp": round(pct(accs.max()) - pct(accs.min()), 3),
    }


# ===========================================================================
# N2 -- reservoir as a calibrated instrument
# ===========================================================================
def icc_2_1(measurements):
    """Two-way random single-measure ICC(2,1) for an (n_targets, n_raters) array.

    Targets = features (or observations); raters = reservoir seeds. Returns the
    reliability of a single rater's measurement.
    """
    x = np.asarray(measurements, dtype=np.float64)
    n, k = x.shape
    grand = x.mean()
    row = x.mean(axis=1, keepdims=True)
    col = x.mean(axis=0, keepdims=True)
    ss_total = ((x - grand) ** 2).sum()
    ss_row = k * ((row - grand) ** 2).sum()
    ss_col = n * ((col - grand) ** 2).sum()
    ss_err = ss_total - ss_row - ss_col
    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_row + (k - 1) * ms_err + k * (ms_col - ms_err) / n
    if denom <= 0:
        return 0.0
    return float((ms_row - ms_err) / denom)


def cross_seed_feature_reliability(bsc_by_seed, srp):
    """Median ICC(2,1) of SRP-projected features across reservoir seeds.

    bsc_by_seed: list of (n_obs, n_ch, feat) BSC arrays, one per seed.
    For each projected feature, reliability is computed over observations with
    the seeds as raters, then summarized by the median across features.
    """
    projected = [cp.project_bsc(b, srp) for b in bsc_by_seed]  # each (n_obs, D)
    stack = np.stack(projected, axis=2)  # (n_obs, D, n_seeds)
    n_obs, D, _ = stack.shape
    iccs = []
    for f in range(D):
        col = stack[:, f, :]
        if col.std() < 1e-9:
            continue
        iccs.append(icc_2_1(col))
    iccs = np.asarray(iccs)
    return {
        "median_icc": round(float(np.median(iccs)), 4),
        "q1_icc": round(float(np.percentile(iccs, 25)), 4),
        "q3_icc": round(float(np.percentile(iccs, 75)), 4),
        "n_features": int(iccs.size),
    }


def transfer_function(template, amplitudes, reservoir_seed=42):
    """Input-amplitude -> total-spike transfer function for the fixed reservoir.

    Drives the reservoir with `template` scaled by each amplitude and records
    the total spike count. Returns monotonicity (Spearman rho) and dynamic
    range. `template` is a (time,) drive; a real centered ERP channel or a
    synthetic pulse both work.
    """
    from experiment_zero import LIFReservoir
    from scipy.stats import spearmanr

    reservoir = LIFReservoir(seed=reservoir_seed)
    totals = []
    for a in amplitudes:
        spikes = reservoir.forward(np.asarray(template, dtype=np.float64) * a)
        totals.append(float(np.asarray(spikes).sum()))
    totals = np.asarray(totals)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    rho = spearmanr(amplitudes, totals).correlation
    nonzero = totals[totals > 0]
    dyn = float(nonzero.max() / nonzero.min()) if nonzero.size and nonzero.min() > 0 else 0.0
    return {
        "monotonicity_spearman": round(float(rho), 4),
        "dynamic_range_ratio": round(dyn, 3),
        "totals": [round(t, 2) for t in totals.tolist()],
    }


def perturb_channel_dropout(raw, fraction, seed):
    """Zero a random `fraction` of electrodes per observation."""
    rng = np.random.RandomState(seed)
    out = np.array(raw, dtype=np.float64, copy=True)
    n_obs, _, n_ch = out.shape
    k = int(round(fraction * n_ch))
    for i in range(n_obs):
        drop = rng.choice(n_ch, size=k, replace=False)
        out[i, :, drop] = 0.0
    return out


def perturb_additive_noise(raw, sigma, seed):
    """Add zero-mean Gaussian noise (std `sigma`, in z-scored units)."""
    rng = np.random.RandomState(seed)
    return np.asarray(raw, dtype=np.float64) + rng.normal(0.0, sigma, size=np.shape(raw))


def robustness_curve(raw, y, groups, perturb, levels, feature_builder,
                     base_seed=0, seed=777):
    """Balanced accuracy of a representation across perturbation levels.

    feature_builder(perturbed_raw) -> feature matrix X for evaluate_features.
    """
    curve = []
    for level in levels:
        pert = raw if level == 0 else perturb(raw, level, seed)
        X = feature_builder(pert)
        ba = cp.evaluate_features(X, y, groups, base_seed=base_seed)["balanced_accuracy"]
        curve.append({"level": level, "balanced_accuracy_percent": pct(ba)})
    return curve


# ===========================================================================
# N3 -- necessity map and subject/condition variance decomposition
# ===========================================================================
def subject_condition_variance_ratio(X, y, groups):
    """Subject-to-condition variance ratio rho via a per-feature ANOVA split.

    For each feature, decompose total variance into between-subject and
    between-condition components (one-way sums of squares for each factor),
    average across features, and return rho = subject / condition together with
    the normalized fractions. This is the estimand behind the reported rho~7.2
    subject-covariance entanglement.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    grand = X.mean(axis=0, keepdims=True)
    total = ((X - grand) ** 2).sum(axis=0)

    def factor_ss(labels):
        ss = np.zeros(X.shape[1])
        for lab in np.unique(labels):
            mask = labels == lab
            grp_mean = X[mask].mean(axis=0, keepdims=True)
            ss += mask.sum() * ((grp_mean - grand) ** 2).ravel()
        return ss

    ss_subject = factor_ss(groups)
    ss_condition = factor_ss(y)
    total = np.where(total <= 0, np.nan, total)
    subj_frac = np.nanmean(ss_subject / total)
    cond_frac = np.nanmean(ss_condition / total)
    resid_frac = max(0.0, 1.0 - subj_frac - cond_frac)
    rho = float(subj_frac / cond_frac) if cond_frac > 0 else float("inf")
    return {
        "rho_subject_to_condition": round(rho, 3),
        "subject_fraction": round(float(subj_frac), 4),
        "condition_fraction": round(float(cond_frac), 4),
        "residual_fraction": round(float(resid_frac), 4),
    }


def necessity_map(bsc, y, groups, srp, base_seed=0):
    """Condition accuracy and subject/condition rho for the intact SRP features
    and under each structural destruction. Returns one row per condition."""
    rng = np.random.RandomState(base_seed + 4242)
    variants = {
        "intact": bsc,
        "temporal_bin_shuffle": cp.destroy_temporal_bin_order(bsc, rng),
        "temporal_collapse": cp.destroy_temporal_collapse(bsc),
        "electrode_shuffle": cp.destroy_electrode_identity(bsc, rng),
    }
    rows = {}
    intact_ba = None
    for name, variant in variants.items():
        X = cp.project_bsc(variant, srp)
        ba = cp.evaluate_features(X, y, groups, base_seed=base_seed)["balanced_accuracy"]
        rho = subject_condition_variance_ratio(X, y, groups)
        if name == "intact":
            intact_ba = pct(ba)
        rows[name] = {
            "condition_balanced_accuracy_percent": pct(ba),
            "condition_accuracy_drop_pp": round(pct(ba) - intact_ba, 3)
            if intact_ba is not None else 0.0,
            "subject_to_condition_rho": rho["rho_subject_to_condition"],
            "subject_fraction": rho["subject_fraction"],
            "condition_fraction": rho["condition_fraction"],
        }
    return rows
