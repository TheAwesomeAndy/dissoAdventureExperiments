#!/usr/bin/env python3
"""
ARSPI-Net Chapter 5: Conventional sklearn Baselines (Table 5.4)
================================================================

Provides 8 conventional machine learning classifiers as baselines for
the 3-class affective EEG classification task. These baselines use
BandPower + Hjorth features (the same conventional feature set as
Row 1 and Row 2 of the 7-row baseline table in run_chapter5_experiments.py).

Classifiers:
  1. Logistic Regression (L2-regularized)
  2. Linear SVM
  3. RBF SVM
  4. K-Nearest Neighbors (k=5)
  5. Random Forest (100 trees)
  6. Gradient Boosting (100 estimators)
  7. AdaBoost (100 estimators)
  8. MLP (128-64, early stopping)

All classifiers use subject-stratified 10-fold CV (StratifiedGroupKFold)
with StandardScaler normalization. PCA is NOT applied to conventional
features (they are already low-dimensional: 34 channels x 8 features = 272).

Usage:
  python sklearn_baselines.py --data_dir /path/to/batch_data/

Output:
  Prints Table 5.4 to stdout.
  Saves sklearn_baseline_results.pkl with fold-level data.
"""

import numpy as np
import os
import re
import argparse
import pickle
import time
from pathlib import Path
from scipy.signal import decimate, welch

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, f1_score


# ================================================================
# FEATURE EXTRACTION (conventional EEG features only)
# ================================================================

def extract_bandpower(X_ds, fs=256):
    """Extract 5-band spectral power per channel.

    Bands: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz),
           beta (13-30 Hz), gamma (30-100 Hz)

    Args:
        X_ds: (N, T, 34) downsampled EEG
        fs: sampling rate after downsampling

    Returns:
        (N, 34, 5) band power features
    """
    N, T, n_ch = X_ds.shape
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, fs // 2 - 1)]
    bp = np.zeros((N, n_ch, len(bands)))

    for i in range(N):
        for ch in range(n_ch):
            freqs, psd = welch(X_ds[i, :, ch], fs=fs, nperseg=min(T, 128))
            for b_idx, (lo, hi) in enumerate(bands):
                mask = (freqs >= lo) & (freqs <= hi)
                if mask.any():
                    bp[i, ch, b_idx] = np.trapezoid(psd[mask], freqs[mask])
    return bp


def extract_hjorth(X_ds):
    """Extract Hjorth parameters (activity, mobility, complexity) per channel.

    Returns:
        (N, 34, 3) Hjorth parameters
    """
    N, T, n_ch = X_ds.shape
    hj = np.zeros((N, n_ch, 3))

    for i in range(N):
        for ch in range(n_ch):
            x = X_ds[i, :, ch]
            dx = np.diff(x)
            ddx = np.diff(dx)

            var_x = np.var(x)
            var_dx = np.var(dx)
            var_ddx = np.var(ddx)

            activity = var_x
            mobility = np.sqrt(var_dx / (var_x + 1e-15))
            complexity = np.sqrt(var_ddx / (var_dx + 1e-15)) / (mobility + 1e-15)

            hj[i, ch, :] = [activity, mobility, complexity]
    return hj


def load_and_preprocess(data_dir):
    """Load SHAPE 3-class EEG data and preprocess."""
    data_dir = Path(data_dir)
    pattern = re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt')

    files = {}
    for f in sorted(os.listdir(data_dir)):
        m = pattern.match(f)
        if m:
            subj = int(m.group(1))
            cond = m.group(2)
            files[(subj, cond)] = str(data_dir / f)

    cond_map = {'Neg': 0, 'Neu': 1, 'Pos': 2}
    subjects_set = sorted(set(s for s, _ in files.keys()))

    raw_data, y_labels, subj_ids = [], [], []
    for subj in subjects_set:
        for cond in ['Neg', 'Neu', 'Pos']:
            if (subj, cond) not in files:
                continue
            X = np.loadtxt(files[(subj, cond)])
            X = X[205:]  # Remove baseline
            X_ds = np.zeros((256, X.shape[1]))
            for ch in range(X.shape[1]):
                X_ds[:, ch] = decimate(X[:, ch], 4)[:256]
            # Z-score per channel
            for ch in range(X_ds.shape[1]):
                mu, sigma = X_ds[:, ch].mean(), X_ds[:, ch].std()
                if sigma > 0:
                    X_ds[:, ch] = (X_ds[:, ch] - mu) / sigma
            raw_data.append(X_ds)
            y_labels.append(cond_map[cond])
            subj_ids.append(f"{subj:03d}")

    return np.array(raw_data), np.array(y_labels), np.array(subj_ids)


# ================================================================
# EVALUATION
# ================================================================

CLASSIFIERS = {
    'LogReg (L2)': lambda: LogisticRegression(C=0.1, max_iter=3000, random_state=42),
    'Linear SVM': lambda: LinearSVC(C=1.0, max_iter=5000, random_state=42, dual=True),
    'RBF SVM': lambda: SVC(C=1.0, kernel='rbf', random_state=42),
    'KNN (k=5)': lambda: KNeighborsClassifier(n_neighbors=5),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
    'MLP (128-64)': lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                           random_state=42, early_stopping=True,
                                           validation_fraction=0.15),
}


def evaluate_classifier(features, y, subjects, clf_factory, n_folds=10):
    """Run subject-stratified CV with a given classifier factory."""
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs, fold_f1s = [], []

    for train_idx, test_idx in gkf.split(features, y, subjects):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(features[train_idx])
        X_te = scaler.transform(features[test_idx])

        clf = clf_factory()
        clf.fit(X_tr, y[train_idx])
        y_pred = clf.predict(X_te)

        fold_accs.append(balanced_accuracy_score(y[test_idx], y_pred))
        fold_f1s.append(f1_score(y[test_idx], y_pred, average='macro'))

    return np.array(fold_accs), np.array(fold_f1s)


def main():
    parser = argparse.ArgumentParser(
        description='ARSPI-Net Chapter 5: sklearn Baseline Classifiers (Table 5.4)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ with SHAPE 3-class files')
    parser.add_argument('--output', type=str, default='sklearn_baseline_results.pkl',
                        help='Output pickle path')
    parser.add_argument('--n_folds', type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("ARSPI-Net — SKLEARN BASELINE CLASSIFIERS (Table 5.4)")
    print("=" * 70)

    # Load data
    print("\nLoading SHAPE 3-class data...")
    raw_data, y, subjects = load_and_preprocess(args.data_dir)
    print(f"  {raw_data.shape[0]} observations, {len(set(subjects))} subjects")

    # Extract conventional features
    print("\nExtracting conventional features...")
    bp = extract_bandpower(raw_data)
    hj = extract_hjorth(raw_data)
    conv_feats = np.concatenate([bp, hj], axis=2)  # (N, 34, 8)
    features = conv_feats.reshape(conv_feats.shape[0], -1)  # (N, 272)
    print(f"  Feature shape: {features.shape}")

    # Run all classifiers
    print(f"\n{'Classifier':<25s} {'BalAcc':>8s} {'± SD':>8s} {'F1':>8s}")
    print("-" * 55)

    results = {}
    for name, factory in CLASSIFIERS.items():
        t0 = time.time()
        accs, f1s = evaluate_classifier(features, y, subjects, factory, args.n_folds)
        elapsed = time.time() - t0
        print(f"  {name:<23s} {accs.mean()*100:>7.1f}% {accs.std()*100:>7.1f}% "
              f"{f1s.mean()*100:>7.1f}%  ({elapsed:.1f}s)")
        results[name] = {
            'fold_accs': accs, 'fold_f1s': f1s,
            'mean_acc': accs.mean(), 'std_acc': accs.std(),
            'mean_f1': f1s.mean(),
        }

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
