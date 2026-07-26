#!/usr/bin/env python3
"""
ARSPI-Net -- Nuisance-alignment compression study (library)
===========================================================

Turns the single-dataset N1 observation ("a fixed random projection beats a
data-fitted PCA on grouped clinical EEG") into a general, mechanistic claim:

  Unsupervised variance-maximizing compression (PCA) is actively harmful when a
  grouping nuisance (subject identity) dominates the signal of interest,
  because its leading components align to the nuisance subspace. A
  data-independent isotropic random projection has no such bias, so it is both
  leak-free by construction and more discriminative.

Three parts:

  M1  Controlled causal demonstration (semi-synthetic, no reservoir). Fixed
      condition signal, tunable subject-nuisance magnitude. Sweep the
      nuisance-to-signal ratio and watch PCA cross below SRP. This makes the
      claim general -- a phenomenon with a knob, not a dataset curiosity.

  M2  Geometric mechanism (real data). In the k-dim compressed space, the
      subject-to-condition variance ratio rho_PCA >> rho_SRP, and the PCA-k
      subspace makes small principal angles with the subject subspace and large
      angles with the condition subspace. Explains WHY PCA fails.

  M3  Compressor comparison (real or synthetic). SRP vs fold-local PCA vs
      supervised in-fold LDA vs random coordinate subsampling under identical
      participant-grouped fold-local CV.

  M5  Reservoir disentanglement. Subject/condition rho for raw ERP-16 vs BSC6
      vs BSC6/SRP-64: does the neuromorphic transform lower entanglement?

Everything reuses the committed operators and the participant-grouped,
fold-local, leak-free evaluation discipline. verify_nuisance.py exercises the
estimators on synthetic data without the restricted dataset.
"""

import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.linalg import subspace_angles

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_REPO, "experiments", "confirmatory"),
           os.path.join(_REPO, "chapter5Experiments"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import confirmatory_pipeline as cp   # noqa: E402
import novel_pipeline as nov         # noqa: E402


def pct(x):
    return round(float(x) * 100.0, 3)


# ===========================================================================
# Generic participant-grouped, fold-local evaluation with a pluggable
# compressor. `fit_compressor(Xtr, ytr) -> transform(X)` is refit per fold on
# training participants only, so every compressor is compared under the same
# leak-free discipline. SRP ignores the data; PCA/LDA fit it.
# ===========================================================================
def _srp_fitter(n_components, seed):
    def fit(Xtr, ytr):
        srp = SparseRandomProjection(n_components=n_components, random_state=seed)
        srp.fit(np.ones((2, Xtr.shape[1])))  # shape-only; data-independent
        return srp.transform
    return fit


def _pca_fitter(n_components):
    def fit(Xtr, ytr):
        pca = PCA(n_components=min(n_components, Xtr.shape[1], len(Xtr) - 1))
        pca.fit(Xtr)
        return pca.transform
    return fit


def _lda_fitter():
    def fit(Xtr, ytr):
        lda = LinearDiscriminantAnalysis()
        lda.fit(Xtr, ytr)
        return lda.transform  # supervised, n_classes-1 dims
    return fit


def _subsample_fitter(n_components, seed):
    def fit(Xtr, ytr):
        rng = np.random.RandomState(seed)
        cols = rng.choice(Xtr.shape[1], size=min(n_components, Xtr.shape[1]),
                          replace=False)
        return lambda X: X[:, cols]
    return fit


COMPRESSORS = {
    "srp": _srp_fitter,
    "pca": _pca_fitter,
    "lda": _lda_fitter,
    "subsample": _subsample_fitter,
}


def evaluate_compressor(X, y, groups, fit_compressor, base_seed=0,
                        n_repeats=None, n_folds=None):
    """Balanced accuracy under repeated participant-grouped fold-local CV with a
    pluggable, per-fold-refit compressor."""
    n_repeats = cp.N_REPEATS if n_repeats is None else n_repeats
    n_folds = cp.N_FOLDS if n_folds is None else n_folds
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    per_repeat = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=base_seed + repeat)
        pred = np.full(len(y), -1, dtype=int)
        for tr, te in splitter.split(X, y, groups):
            transform = fit_compressor(X[tr], y[tr])
            Xtr, Xte = transform(X[tr]), transform(X[te])
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     random_state=base_seed + repeat)
            clf.fit(scaler.transform(Xtr), y[tr])
            pred[te] = clf.predict(scaler.transform(Xte))
        per_repeat.append(float(balanced_accuracy_score(y, pred)))
    return {"balanced_accuracy": float(np.mean(per_repeat)),
            "per_repeat": per_repeat}


# ===========================================================================
# M1 -- controlled nuisance-ratio sweep (semi-synthetic, no reservoir)
# ===========================================================================
def make_controlled(n_participants, n_cond, dim, sigma_subj, sigma_noise,
                    seed, signal=1.0):
    """Feature vectors = condition mean + subject offset + observation noise.

    The condition means are fixed shared templates; each participant adds a
    random offset (the grouping nuisance) applied to all of its observations.
    sigma_subj tunes the nuisance magnitude relative to `signal`.
    """
    rng = np.random.RandomState(seed)
    cond_means = rng.randn(n_cond, dim) * signal
    X, y, groups = [], [], []
    for p in range(n_participants):
        offset = rng.randn(dim) * sigma_subj
        for c in range(n_cond):
            X.append(cond_means[c] + offset + rng.randn(dim) * sigma_noise)
            y.append(c)
            groups.append(p)
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def angle_rotation_sweep(ratios, n_components=16, dim=200, n_participants=60,
                         n_cond=3, sigma_noise=1.0, subj_rank=20, seed=0):
    """The generalizable geometric mechanism: as the subject-nuisance magnitude
    grows, the PCA-k subspace rotates away from the condition subspace and
    toward the subject subspace. Reports, per nuisance ratio, the PCA minimum
    principal angle (degrees) to the subject and condition subspaces. A random
    projection has no data-dependent alignment and is not swept here.

    This is monotone and dataset-independent (PCA maximizes variance, which is
    the nuisance when the nuisance dominates), and is the mechanism that makes
    an isotropic random projection preferable under aggressive compression.
    """
    rows = []
    for r in ratios:
        X, y, g = make_controlled(n_participants, n_cond, dim,
                                  sigma_subj=r, sigma_noise=sigma_noise, seed=seed)
        geo = compression_geometry(X, y, g, n_components=n_components,
                                   subj_rank=subj_rank, cond_rank=n_cond - 1)
        rows.append({
            "nuisance_ratio": r,
            "pca_angle_to_subject_deg": geo["pca_min_angle_to_subject_deg"],
            "pca_angle_to_condition_deg": geo["pca_min_angle_to_condition_deg"],
            "rho_pca_compressed": geo["rho_pca_compressed"],
            "rho_srp_compressed": geo["rho_srp_compressed"],
        })
    return {"curve": rows, "n_components": n_components, "dim": dim,
            "subject_rank": subj_rank}


def accuracy_sweep(ratios, n_components=16, dim=200, n_participants=60,
                   n_cond=3, sigma_noise=1.0, seed=0, base_seed=0):
    """SRP vs fold-local PCA balanced accuracy across nuisance ratios. The
    accuracy *consequence* of the geometric rotation is regime-dependent: SRP
    overtakes PCA only once the condition signal falls below the retained
    nuisance spectrum. Reported honestly, including the crossover ratio (or
    None if PCA remains ahead over the swept range)."""
    srp_fit = _srp_fitter(n_components, cp.SRP_SEED)
    pca_fit = _pca_fitter(n_components)
    rows, crossover = [], None
    for r in ratios:
        X, y, g = make_controlled(n_participants, n_cond, dim,
                                  sigma_subj=r, sigma_noise=sigma_noise, seed=seed)
        srp = evaluate_compressor(X, y, g, srp_fit, base_seed)["balanced_accuracy"]
        pca = evaluate_compressor(X, y, g, pca_fit, base_seed)["balanced_accuracy"]
        rows.append({"nuisance_ratio": r, "srp_percent": pct(srp),
                     "pca_percent": pct(pca),
                     "srp_minus_pca_pp": round(pct(srp) - pct(pca), 3)})
        if crossover is None and srp > pca:
            crossover = r
    return {"curve": rows, "crossover_ratio": crossover,
            "n_components": n_components, "dim": dim}


# ===========================================================================
# M2 -- geometric mechanism (real or synthetic)
# ===========================================================================
def _factor_subspace(X, labels, rank):
    """Top-`rank` principal directions of the between-group mean matrix."""
    grand = X.mean(axis=0, keepdims=True)
    rows = []
    for lab in np.unique(labels):
        m = X[labels == lab].mean(axis=0)
        rows.append(m - grand.ravel())
    M = np.asarray(rows)
    # right singular vectors span the between-group direction subspace
    _, _, vt = np.linalg.svd(M, full_matrices=False)
    r = min(rank, vt.shape[0])
    return vt[:r].T  # (dim, r) orthonormal columns


def compression_geometry(X, y, groups, n_components=64, subj_rank=10,
                         cond_rank=2):
    """Compare where PCA vs SRP put their variance, and how the PCA subspace
    aligns to the subject vs condition subspaces (principal angles, degrees).

    Descriptive geometry on the full feature matrix (not a performance
    estimate): it characterizes the ambient structure that drives M1/N1.
    """
    X = np.asarray(X, dtype=np.float64)
    # compressed-space subject/condition variance ratio
    pca = PCA(n_components=min(n_components, X.shape[1], len(X) - 1)).fit(X)
    Xpca = pca.transform(X)
    srp = cp.make_srp(1, n_components=1)  # placeholder to reuse rng-free path
    srp = SparseRandomProjection(n_components=n_components,
                                 random_state=cp.SRP_SEED)
    srp.fit(np.ones((2, X.shape[1])))
    Xsrp = srp.transform(X)

    rho_amb = nov.subject_condition_variance_ratio(X, y, groups)["rho_subject_to_condition"]
    rho_pca = nov.subject_condition_variance_ratio(Xpca, y, groups)["rho_subject_to_condition"]
    rho_srp = nov.subject_condition_variance_ratio(Xsrp, y, groups)["rho_subject_to_condition"]

    U_subj = _factor_subspace(X, groups, subj_rank)
    U_cond = _factor_subspace(X, y, cond_rank)
    U_pca = pca.components_[:n_components].T
    ang_subj = np.degrees(subspace_angles(U_pca, U_subj))
    ang_cond = np.degrees(subspace_angles(U_pca, U_cond))
    return {
        "rho_ambient": rho_amb,
        "rho_pca_compressed": rho_pca,
        "rho_srp_compressed": rho_srp,
        "pca_min_angle_to_subject_deg": round(float(ang_subj.min()), 2),
        "pca_min_angle_to_condition_deg": round(float(ang_cond.min()), 2),
        "pca_mean_angle_to_subject_deg": round(float(ang_subj.mean()), 2),
        "pca_mean_angle_to_condition_deg": round(float(ang_cond.mean()), 2),
    }


def _blockwise_pca_angles(P, U_factor, n_ch, n_feat):
    """Principal angles (degrees) between the block-diagonal blockwise-PCA read
    subspace and an ambient factor subspace, without materialising the full
    (n_ch*n_feat, n_ch*k) basis.

    The blockwise operator places the SAME orthonormal PCA basis `P`
    (n_feat, k) on each electrode block and concatenates. Its ambient read
    subspace therefore has orthonormal columns (P orthonormal on disjoint row
    blocks). For an orthonormal ambient factor basis `U_factor`
    (n_ch*n_feat, r), the cosines of the principal angles are the singular
    values of Q_pca^T U_factor, which is block-computable as
    einsum('fk,cfr->ckr', P, U_factor.reshape(n_ch, n_feat, r)).
    """
    k = P.shape[1]
    r = U_factor.shape[1]
    Uf = U_factor.reshape(n_ch, n_feat, r)
    M = np.einsum("fk,cfr->ckr", P, Uf).reshape(n_ch * k, r)
    sv = np.linalg.svd(M, compute_uv=False)
    cos = np.clip(sv, 0.0, 1.0)
    return np.degrees(np.arccos(cos))


def compression_geometry_blockwise(bsc, y, groups, n_components=64,
                                   subj_rank=10, cond_rank=2, srp_seed=None):
    """M2 geometry computed with the *exact N1 operator*: a shared per-electrode
    compression (1,536 -> `n_components` on each of the `n_ch` electrode blocks,
    concatenated to `n_ch * n_components` coordinates), not the global
    52,224 -> 64 flatten used by `compression_geometry`.

    This closes the operator-mismatch gap the audit flagged: N1's accuracy gap
    uses the blockwise operator, so the geometry that would *explain* that gap
    must use the same operator. Reports the compressed-space subject/condition
    variance ratio rho and the PCA read-subspace principal angles to the subject
    and condition subspaces, directly comparable to the global M2 numbers.

    `bsc` is the electrode-resolved (n_obs, n_ch, n_feat) BSC tensor -- the same
    input N1 consumes. Descriptive geometry on the full matrix (not a CV
    estimate), matching `compression_geometry`.
    """
    bsc = np.asarray(bsc, dtype=np.float64)
    if bsc.ndim != 3:
        raise ValueError("bsc must have shape (n_obs, n_ch, n_feat)")
    n_obs, n_ch, n_feat = bsc.shape
    srp_seed = cp.SRP_SEED if srp_seed is None else srp_seed

    # Shared per-electrode PCA basis, fitted on all electrode blocks stacked --
    # identical construction to N1's fold-local PCA, here on the full matrix.
    k = min(n_components, n_feat, n_obs * n_ch - 1)
    pca = PCA(n_components=k).fit(bsc.reshape(n_obs * n_ch, n_feat))
    P = pca.components_.T  # (n_feat, k), orthonormal columns
    Xpca = (bsc @ P).reshape(n_obs, n_ch * k)

    # Shared per-electrode SRP -- exactly N1's evaluate_srp operator.
    srp = cp.make_srp(n_feat, seed=srp_seed, n_components=n_components)
    Xsrp = cp.project_bsc(bsc, srp)

    flat = bsc.reshape(n_obs, n_ch * n_feat)
    rho_amb = nov.subject_condition_variance_ratio(flat, y, groups)["rho_subject_to_condition"]
    rho_pca = nov.subject_condition_variance_ratio(Xpca, y, groups)["rho_subject_to_condition"]
    rho_srp = nov.subject_condition_variance_ratio(Xsrp, y, groups)["rho_subject_to_condition"]

    U_subj = _factor_subspace(flat, groups, subj_rank)
    U_cond = _factor_subspace(flat, y, cond_rank)
    ang_subj = _blockwise_pca_angles(P, U_subj, n_ch, n_feat)
    ang_cond = _blockwise_pca_angles(P, U_cond, n_ch, n_feat)
    return {
        "operator": "blockwise_per_electrode",
        "n_components_per_electrode": int(k),
        "total_compressed_dims": int(n_ch * k),
        "rho_ambient": rho_amb,
        "rho_pca_compressed": rho_pca,
        "rho_srp_compressed": rho_srp,
        "pca_min_angle_to_subject_deg": round(float(ang_subj.min()), 2),
        "pca_min_angle_to_condition_deg": round(float(ang_cond.min()), 2),
        "pca_mean_angle_to_subject_deg": round(float(ang_subj.mean()), 2),
        "pca_mean_angle_to_condition_deg": round(float(ang_cond.mean()), 2),
    }


# ===========================================================================
# M5 -- reservoir disentanglement across representations
# ===========================================================================
def disentanglement(raw_erp16, bsc_flat, srp_features, y, groups):
    """Subject/condition rho for raw ERP-16, BSC6 (flattened), and BSC6/SRP-64.
    Lower rho = less subject-condition entanglement in that representation."""
    return {
        "raw_erp16_rho": nov.subject_condition_variance_ratio(raw_erp16, y, groups)["rho_subject_to_condition"],
        "bsc6_rho": nov.subject_condition_variance_ratio(bsc_flat, y, groups)["rho_subject_to_condition"],
        "bsc6_srp64_rho": nov.subject_condition_variance_ratio(srp_features, y, groups)["rho_subject_to_condition"],
    }
