#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
ARSPI-Net — Experiment Zero: Baseline Disambiguation
═══════════════════════════════════════════════════════════════════════

SCIENTIFIC QUESTION
-------------------
The 70.5% raw-EEG LogReg result's centering status is unknown. This
factual ambiguity changes the interpretation of every subsequent result
in the v2 research plan. Experiment Zero resolves it by producing four
numbers under identical protocol:

  (1) Raw EEG (flattened) + LogReg, UNCENTERED
  (2) Raw EEG (flattened) + LogReg, SUBJECT-CENTERED
  (3) Reservoir Embedding (BSC6-PCA-64) + LogReg, UNCENTERED
  (4) Reservoir Embedding (BSC6-PCA-64) + LogReg, SUBJECT-CENTERED

All four use the same 10-fold subject-level stratified group CV on the
full 211-subject SHAPE dataset. No new interventions are introduced.
EA-preprocessed comparisons are deferred to Phase 2.

METHODOLOGY RULES APPLIED
--------------------------
  Rule 5: Linear readout (LogReg) as primary classifier
  Rule 12: Both centered and uncentered numbers reported together
  Rule 7: Summary table built immediately

USAGE
-----
  python experiment_zero.py --data_dir /path/to/batch_data/

  The data_dir should contain files named:
    SHAPE_Community_{SUBJ}_{COND}_BC.txt
  where COND ∈ {IAPSNeg, IAPSNeu, IAPSPos}

OUTPUT
------
  Prints the four-number disambiguation table to stdout.
  Saves experiment_zero_results.pkl with full fold-level data.

ESTIMATED TIME: <30 minutes on CPU
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import os
import re
import pickle
import time
import argparse
from pathlib import Path
from scipy.signal import decimate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════
# LIF RESERVOIR (identical to v1 dissertation pipeline)
# ══════════════════════════════════════════════════════════════════

class LIFReservoir:
    """Leaky Integrate-and-Fire reservoir: N=256, β=0.05, θ=0.5."""
    def __init__(self, n_res=256, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        self.n_res = n_res
        self.beta = beta
        self.threshold = threshold
        # Input weights: uniform ±√(6/(1+N))
        lim_in = np.sqrt(6.0 / (1 + n_res))
        self.W_in = rng.uniform(-lim_in, lim_in, (n_res, 1))
        # Recurrent weights: uniform ±√(6/(2N)), spectral radius 0.9
        lim_rec = np.sqrt(6.0 / (2 * n_res))
        self.W_rec = rng.uniform(-lim_rec, lim_rec, (n_res, n_res))
        eig = np.abs(np.linalg.eigvals(self.W_rec)).max()
        if eig > 0:
            self.W_rec *= 0.9 / eig

    def forward(self, x):
        """Process 1D input signal, return (T, N) binary spike matrix."""
        T = len(x)
        n = self.n_res
        mem = np.zeros(n)
        spike = np.zeros(n)
        spikes = np.zeros((T, n))
        for t in range(T):
            I = self.W_in[:, 0] * x[t] + self.W_rec @ spike
            mem = (1 - self.beta) * mem * (1 - spike) + I
            s = (mem >= self.threshold).astype(float)
            mem = mem - s * self.threshold
            mem = np.maximum(mem, 0)
            spikes[t] = s
            spike = s
        return spikes


def bsc6_encode(spikes, n_bins=6, t_start=10, t_end=70):
    """Encode the stated early window as six equal spike-count bins.

    The defended Chapter 5 run used six full-epoch bins.  This corrected
    rerun function follows the Chapter 4 specification instead.
    """
    if t_end > spikes.shape[0]:
        raise ValueError(
            f"BSC6 window [{t_start}, {t_end}) exceeds {spikes.shape[0]} steps")
    window = spikes[t_start:t_end]
    T, N = window.shape
    if T % n_bins != 0:
        raise ValueError("BSC6 window length must be divisible by n_bins")
    bin_size = T // n_bins
    features = np.zeros(n_bins * N)
    for b in range(n_bins):
        s = b * bin_size
        e = s + bin_size
        features[b * N:(b + 1) * N] = window[s:e, :].sum(axis=0)
    return features


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_shape_3class(data_dir):
    """Load SHAPE 3-class EEG data from batch_data directory."""
    data_dir = Path(data_dir)
    pattern = re.compile(
        r'SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt'
    )

    files = {}
    for f in sorted(os.listdir(data_dir)):
        m = pattern.match(f)
        if m:
            subj = int(m.group(1))
            cond = m.group(2)  # Neg, Neu, Pos
            files[(subj, cond)] = str(data_dir / f)

    subjects_set = sorted(set(s for s, c in files.keys()))
    conds = ['Neg', 'Neu', 'Pos']
    cond_map = {'Neg': 0, 'Neu': 1, 'Pos': 2}

    print(f"  Found {len(files)} files from {len(subjects_set)} subjects")

    # Load all data
    raw_data = []    # (N_obs, T_ds, 34) — downsampled raw EEG
    y_labels = []
    subj_ids = []

    for subj in subjects_set:
        for cond in conds:
            key = (subj, cond)
            if key not in files:
                continue
            X = np.loadtxt(files[key])  # (T, 34) at 1024 Hz or (1229, 34)
            # Downsample to 256 steps
            T_orig, n_ch = X.shape
            if T_orig > 300:
                # Decimate by factor to get ~256 steps
                factor = max(1, T_orig // 256)
                target_len = T_orig // factor
                X_ds = np.zeros((target_len, n_ch))
                for ch in range(n_ch):
                    decimated = decimate(X[:, ch], factor)
                    X_ds[:, ch] = decimated[:target_len]
                X_ds = X_ds[:256, :]  # Truncate to exactly 256
            else:
                X_ds = X[:256, :]

            # Z-score per channel
            for ch in range(n_ch):
                mu, sigma = X_ds[:, ch].mean(), X_ds[:, ch].std()
                if sigma > 0:
                    X_ds[:, ch] = (X_ds[:, ch] - mu) / sigma

            raw_data.append(X_ds)
            y_labels.append(cond_map[cond])
            subj_ids.append(f"{subj:03d}")

    raw_data = np.array(raw_data)  # (N, 256, 34)
    y = np.array(y_labels)
    subjects = np.array(subj_ids)

    print(f"  Loaded: {raw_data.shape[0]} observations, "
          f"{len(set(subj_ids))} subjects, {raw_data.shape[1]} timesteps, "
          f"{raw_data.shape[2]} channels")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return raw_data, y, subjects


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_reservoir_features(raw_data):
    """Run the LIF reservoir and return unreduced BSC6 node features."""
    N, T, n_ch = raw_data.shape
    print(f"\n  Running LIF reservoir on {N} observations × {n_ch} channels...")
    t0 = time.time()

    # One shared saved transformation keeps equal-numbered feature coordinates
    # commensurate across channels.
    reservoir = LIFReservoir(seed=42)

    # Early-window BSC6: (N, n_ch, 6*256) = (N, n_ch, 1536)
    bsc_all = np.zeros((N, n_ch, 6 * 256))
    for i in range(N):
        if (i + 1) % 10 == 0:  # Print every 10 subjects instead of 100
            print(f"    Processing subject {i+1}/{N}...")
        for ch in range(n_ch):
            spikes = reservoir.forward(raw_data[i, :, ch])
            bsc_all[i, ch, :] = bsc6_encode(spikes, n_bins=6)

    elapsed = time.time() - t0
    print(f"  Reservoir complete in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")

    print(f"  Unreduced BSC6 feature shape: {bsc_all.shape}")
    return bsc_all


def flatten_raw(raw_data):
    """Flatten raw EEG: (N, T, 34) → (N, T*34)."""
    N = raw_data.shape[0]
    return raw_data.reshape(N, -1)


def subject_center(features, subjects):
    """Subtract per-subject mean from features."""
    centered = features.copy()
    for s in np.unique(subjects):
        mask = subjects == s
        centered[mask] -= centered[mask].mean(axis=0)
    return centered


# ══════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════

def _fold_common_pca(train_features, test_features, n_components=64):
    """Fit a shared channel basis on training participants only."""
    n_train, n_channels, feature_dim = train_features.shape
    n_test = test_features.shape[0]
    train_pool = train_features.reshape(n_train * n_channels, feature_dim)
    test_pool = test_features.reshape(n_test * n_channels, feature_dim)
    n_comp = min(n_components, train_pool.shape[0] - 1, feature_dim)
    scaler = StandardScaler()
    train_pool = scaler.fit_transform(train_pool)
    test_pool = scaler.transform(test_pool)
    pca = PCA(n_components=n_comp, random_state=42)
    X_train = pca.fit_transform(train_pool).reshape(n_train, n_channels, n_comp)
    X_test = pca.transform(test_pool).reshape(n_test, n_channels, n_comp)
    return X_train.reshape(n_train, -1), X_test.reshape(n_test, -1)


def evaluate(features, y, subjects, name, n_folds=10,
             pca_components=None, classifier='logreg'):
    """Run subject-grouped CV with learned preprocessing fit in-fold."""
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr, te) in enumerate(gkf.split(features, y, subjects)):
        if pca_components is not None:
            if features.ndim != 3:
                raise ValueError("Fold-local PCA requires unreduced (sample, channel, feature) input")
            X_tr, X_te = _fold_common_pca(
                features[tr], features[te], n_components=pca_components)
        else:
            X_tr, X_te = features[tr], features[te]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        if classifier == 'svm':
            clf = SVC(C=1.0, kernel='rbf', random_state=42)
        else:
            clf = LogisticRegression(C=0.1, max_iter=3000, random_state=42)
        clf.fit(X_tr, y[tr])

        y_pred = clf.predict(X_te)
        acc = balanced_accuracy_score(y[te], y_pred)
        fold_accs.append(acc)

    fold_accs = np.array(fold_accs)
    mean_acc = fold_accs.mean()
    std_acc = fold_accs.std()
    print(f"    {name:45s}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    return fold_accs


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ARSPI-Net Experiment Zero: Baseline Disambiguation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Produces exactly four numbers under identical protocol:
  (1) Raw EEG + LogReg, uncentered
  (2) Raw EEG + LogReg, subject-centered
  (3) Reservoir BSC6-PCA-64 + LogReg, uncentered
  (4) Reservoir BSC6-PCA-64 + LogReg, subject-centered

Example:
  python experiment_zero.py --data_dir /path/to/batch_data/
  python experiment_zero.py --data_dir /workspaces/dissoAdventureExperiments/data/batch_data/
        """)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ directory with SHAPE 3-class files')
    parser.add_argument('--output', type=str, default='experiment_zero_results.pkl',
                        help='Output pickle path')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of CV folds (default: 10)')
    parser.add_argument('--skip_reservoir', action='store_true',
                        help='Skip reservoir (if features pickle exists)')
    parser.add_argument('--features_pkl', type=str, default=None,
                        help='Path to existing features pickle (skips reservoir)')

    args = parser.parse_args()

    print("=" * 70)
    print("ARSPI-Net — EXPERIMENT ZERO: BASELINE DISAMBIGUATION")
    print("=" * 70)
    print(f"\n  Data directory: {args.data_dir}")
    print(f"  CV folds: {args.n_folds}")
    print(f"  Output: {args.output}")

    # ── Load data ──
    print("\n" + "─" * 70)
    print("STEP 1: Load SHAPE 3-class data")
    print("─" * 70)
    raw_data, y, subjects = load_shape_3class(args.data_dir)
    N, T, n_ch = raw_data.shape

    # ── Extract features ──
    print("\n" + "─" * 70)
    print("STEP 2: Extract features")
    print("─" * 70)

    # Raw EEG flattened
    print("\n  Flattening raw EEG...")
    raw_flat = flatten_raw(raw_data)
    print(f"  Raw EEG shape: {raw_flat.shape}")

    # Reservoir embedding
    if args.features_pkl and os.path.exists(args.features_pkl):
        print(f"\n  Loading pre-computed features from {args.features_pkl}...")
        with open(args.features_pkl, 'rb') as f:
            d = pickle.load(f)
        bsc_features = d.get('lsm_bsc6_raw', d.get('bsc_all', None))
        if bsc_features is None:
            raise ValueError(
                "The feature file contains only globally reduced embeddings. "
                "Provide unreduced BSC6 features so PCA can be fitted in each fold.")
        print(f"  Unreduced BSC6 shape: {bsc_features.shape}")
    else:
        bsc_features = extract_reservoir_features(raw_data)

    # Subject-centered versions
    print("\n  Computing subject-centered versions...")
    raw_flat_c = subject_center(raw_flat, subjects)
    bsc_features_c = subject_center(bsc_features, subjects)

    # ── Run evaluations ──
    print("\n" + "─" * 70)
    print("STEP 3: Evaluate (10-fold subject-level stratified group CV, LogReg)")
    print("─" * 70)
    print()

    results = {}

    print("  ── Raw EEG ──")
    results['raw_uncentered'] = evaluate(
        raw_flat, y, subjects,
        "(1) Raw EEG + LogReg, UNCENTERED", args.n_folds)
    results['raw_centered'] = evaluate(
        raw_flat_c, y, subjects,
        "(2) Raw EEG + LogReg, SUBJECT-CENTERED", args.n_folds)

    print("\n  ── Reservoir Embedding (BSC6-PCA-64) ──")
    print("  Running 10-fold CV for reservoir uncentered...")
    results['res_uncentered'] = evaluate(
        bsc_features, y, subjects,
        "(3) Reservoir + LogReg, UNCENTERED", args.n_folds,
        pca_components=64)
    print("  Reservoir uncentered complete.")
    print("  Running 10-fold CV for reservoir centered...")
    results['res_centered'] = evaluate(
        bsc_features_c, y, subjects,
        "(4) Reservoir + LogReg, SUBJECT-CENTERED", args.n_folds,
        pca_components=64)
    print("  Reservoir centered complete.")

    # ── Also run SVM for comparison with existing Chapter 5 numbers ──
    print("\n  ── SVM cross-check (for consistency with Ch5 results) ──")
    for label, feats, pca_components in [
            ("Raw EEG uncentered + SVM", raw_flat, None),
            ("Reservoir uncentered + SVM", bsc_features, 64),
            ("Reservoir centered + SVM", bsc_features_c, 64)]:
        fold_accs = evaluate(
            feats, y, subjects, label, args.n_folds,
            pca_components=pca_components, classifier='svm')
        results[f"svm_{label.split()[0].lower()}_{label.split()[1].lower()}"] = fold_accs

    # ── Summary table ──
    print("\n" + "═" * 70)
    print("EXPERIMENT ZERO — DISAMBIGUATION TABLE")
    print("═" * 70)
    print()
    print(f"  {'Representation':<30s} {'Uncentered':>12s} {'Centered':>12s} {'Δ (centering)':>14s}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*14}")

    raw_u = results['raw_uncentered'].mean() * 100
    raw_c = results['raw_centered'].mean() * 100
    res_u = results['res_uncentered'].mean() * 100
    res_c = results['res_centered'].mean() * 100

    print(f"  {'Raw EEG + LogReg':<30s} {raw_u:>11.1f}% {raw_c:>11.1f}% {raw_c-raw_u:>+13.1f} pp")
    print(f"  {'Reservoir BSC6-PCA-64 + LogReg':<30s} {res_u:>11.1f}% {res_c:>11.1f}% {res_c-res_u:>+13.1f} pp")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*14}")
    print(f"  {'Reservoir advantage':<30s} {res_u-raw_u:>+11.1f}  {res_c-raw_c:>+11.1f}  ")
    print()

    # ── Diagnostic interpretation ──
    print("  DIAGNOSTIC INTERPRETATION:")
    print(f"  ─────────────────────────")
    if raw_u > 68:
        print(f"  → Raw EEG uncentered ({raw_u:.1f}%) confirms the 70.5% was likely uncentered.")
        print(f"    The reservoir ({res_u:.1f}%) is outperformed on pure classification.")
    else:
        print(f"  → Raw EEG uncentered ({raw_u:.1f}%) is substantially below 70.5%.")
        print(f"    The original 70.5% may have used centering or other preprocessing.")

    if res_c > raw_c:
        print(f"  → After centering, the reservoir ({res_c:.1f}%) exceeds raw EEG ({raw_c:.1f}%).")
        print(f"    The reservoir's temporal code captures structure that centering reveals.")
    else:
        print(f"  → After centering, raw EEG ({raw_c:.1f}%) still exceeds the reservoir ({res_c:.1f}%).")
        print(f"    The reservoir does not add discriminative value beyond what centering recovers from raw EEG.")

    res_advantage_uncent = res_u - raw_u
    res_advantage_cent = res_c - raw_c
    print(f"\n  → Reservoir advantage: {res_advantage_uncent:+.1f} pp (uncentered), "
          f"{res_advantage_cent:+.1f} pp (centered)")

    # ── Save results ──
    results_out = {
        'raw_uncentered': results['raw_uncentered'],
        'raw_centered': results['raw_centered'],
        'res_uncentered': results['res_uncentered'],
        'res_centered': results['res_centered'],
        'summary': {
            'raw_uncentered_mean': raw_u,
            'raw_centered_mean': raw_c,
            'res_uncentered_mean': res_u,
            'res_centered_mean': res_c,
        },
        'metadata': {
            'n_subjects': len(set(subjects)),
            'n_observations': N,
            'n_channels': n_ch,
            'n_folds': args.n_folds,
            'classifier': 'LogisticRegression(C=0.1)',
            'cv': 'StratifiedGroupKFold(shuffle=True, random_state=42)',
        }
    }

    with open(args.output, 'wb') as f:
        pickle.dump(results_out, f)
    print(f"\n  Results saved to {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
