#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
Sklearn Baselines for ARSPI-Net Dissertation (Chapter 5, Table 5.4)
═══════════════════════════════════════════════════════════════════════

Produces the sklearn rows of the conventional baselines table:
  - BandPower + LinearSVM       → 47.7%
  - BandPower + MLP             → 51.0%
  - Raw EEG PCA-200 + MLP      → 60.8%
  - Raw EEG PCA-200 + LogReg   → 64.9%
  - Raw EEG (full) + LogReg    → 70.5%
  - Reservoir + LogReg          → 59.4%
  - Reservoir + LinearSVM       → 55.8%
  - Reservoir + MLP             → 54.7%

Plus subject-centered comparisons:
  - Raw EEG + centering + LogReg    → 88.4%
  - Reservoir + centering + LogReg  → 78.8%

Protocol: 10-fold StratifiedGroupKFold, subject-level splitting,
          balanced accuracy, StandardScaler per fold.

Input:  shape_features_211.pkl
Output: sklearn_baseline_results.pkl (per-fold data for all methods)

Usage:
  python sklearn_baselines.py
  python sklearn_baselines.py --features /path/to/shape_features_211.pkl
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pickle
import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.decomposition import PCA


def load_data(path):
    """Load shape_features_211.pkl."""
    for p in [path, 'shape_features_211.pkl', 'data/shape_features_211.pkl',
              '../shape_features_211.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                d = pickle.load(f)
            print(f"  Loaded from {p}")
            return d
    raise FileNotFoundError("Cannot find shape_features_211.pkl")


def subject_center(X, subjects):
    """Subtract per-subject mean."""
    Xc = X.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        Xc[mask] -= Xc[mask].mean(axis=0)
    return Xc


def evaluate(X, y, subjects, model_fn, cv, name, use_pca=None):
    """Run subject-level CV, return per-fold balanced accuracy and MCC."""
    accs, mccs = [], []
    for train_idx, test_idx in cv.split(X, y, groups=subjects):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        if use_pca:
            pca = PCA(n_components=use_pca, random_state=42)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
        model = model_fn()
        model.fit(Xtr, y[train_idx])
        yp = model.predict(Xte)
        accs.append(balanced_accuracy_score(y[test_idx], yp))
        mccs.append(matthews_corrcoef(y[test_idx], yp))
    acc_m, acc_s, mcc_m = np.mean(accs), np.std(accs), np.mean(mccs)
    print(f"    {name:45s}: {acc_m:.1%} ± {acc_s:.1%}  MCC={mcc_m:.3f}")
    return {'fold_accs': accs, 'fold_mccs': mccs,
            'acc_mean': acc_m, 'acc_std': acc_s, 'mcc_mean': mcc_m}


def main():
    parser = argparse.ArgumentParser(description='Sklearn baselines for Ch5 Table 5.4')
    parser.add_argument('--features', type=str, default='shape_features_211.pkl',
                        help='Path to shape_features_211.pkl')
    parser.add_argument('--output', type=str, default='sklearn_baseline_results.pkl')
    args = parser.parse_args()

    print("=" * 70)
    print("SKLEARN BASELINES — Chapter 5, Table 5.4")
    print("=" * 70)

    d = load_data(args.features)
    X_raw = d['X_ds'].reshape(d['X_ds'].shape[0], -1)   # (633, 8704)
    y = d['y']
    subjects = d['subjects']
    X_res = d['lsm_bsc6_pca'].reshape(d['lsm_bsc6_pca'].shape[0], -1)  # (633, 2176)
    X_bp = d['conv_feats'].reshape(d['conv_feats'].shape[0], -1)         # (633, 170)
    N = X_raw.shape[0]

    print(f"\n  Observations: {N}")
    print(f"  Subjects: {len(np.unique(subjects))}")
    print(f"  Raw EEG dims: {X_raw.shape[1]}")
    print(f"  Reservoir dims: {X_res.shape[1]}")
    print(f"  BandPower dims: {X_bp.shape[1]}")

    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    # --- Spectral features ---
    print("\n  ── Spectral Features ──")
    lr = lambda: LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
    svm = lambda: LinearSVC(C=1.0, max_iter=5000, dual=True, random_state=42)
    mlp_small = lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                       random_state=42, early_stopping=True,
                                       validation_fraction=0.15, batch_size=32, alpha=0.01)

    results['BandPower + LinearSVM'] = evaluate(X_bp, y, subjects, svm, cv, "BandPower (170d) + LinearSVM")
    results['BandPower + MLP'] = evaluate(X_bp, y, subjects, mlp_small, cv, "BandPower (170d) + MLP(128,64)")

    # --- Raw EEG ---
    print("\n  ── Raw EEG ──")
    results['Raw EEG PCA-200 + MLP'] = evaluate(X_raw, y, subjects, mlp_small, cv,
                                                  "Raw EEG PCA-200 + MLP(128,64)", use_pca=200)
    results['Raw EEG PCA-200 + LogReg'] = evaluate(X_raw, y, subjects, lr, cv,
                                                     "Raw EEG PCA-200 + LogReg", use_pca=200)
    results['Raw EEG (full) + LogReg'] = evaluate(X_raw, y, subjects, lr, cv,
                                                    "Raw EEG (8704d) + LogReg")

    # --- Reservoir ---
    print("\n  ── ARSPI-Net Reservoir ──")
    results['Reservoir + LogReg'] = evaluate(X_res, y, subjects, lr, cv, "Reservoir (2176d) + LogReg")
    results['Reservoir + LinearSVM'] = evaluate(X_res, y, subjects, svm, cv, "Reservoir (2176d) + LinearSVM")
    results['Reservoir + MLP'] = evaluate(X_res, y, subjects, mlp_small, cv, "Reservoir (2176d) + MLP(128,64)")

    # --- Subject-centered ---
    print("\n  ── Subject-Centered ──")
    X_raw_c = subject_center(X_raw, subjects)
    X_res_c = subject_center(X_res, subjects)
    results['Raw EEG centered + LogReg'] = evaluate(X_raw_c, y, subjects, lr, cv,
                                                      "Raw EEG centered (8704d) + LogReg")
    results['Reservoir centered + LogReg'] = evaluate(X_res_c, y, subjects, lr, cv,
                                                        "Reservoir centered (2176d) + LogReg")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (sorted by accuracy)")
    print("=" * 70)
    sorted_r = sorted(results.items(), key=lambda x: x[1]['acc_mean'])
    print(f"  {'Method':<45s} {'Acc':>8s} {'MCC':>6s}")
    print("  " + "-" * 62)
    for name, r in sorted_r:
        print(f"  {name:<45s} {r['acc_mean']:.1%}±{r['acc_std']:.1%} {r['mcc_mean']:.3f}")

    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
