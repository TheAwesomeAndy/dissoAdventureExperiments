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


def bsc6_encode(spikes, n_bins=6):
    """Binned Spike Count with n_bins temporal bins."""
    T, N = spikes.shape
    bin_size = T // n_bins
    features = np.zeros(n_bins * N)
    for b in range(n_bins):
        s = b * bin_size
        e = min(s + bin_size, T)
        features[b * N:(b + 1) * N] = spikes[s:e, :].sum(axis=0)
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

def extract_reservoir_features(raw_data, n_pca=64):
    """Run LIF reservoir on all data, extract BSC6-PCA features."""
    N, T, n_ch = raw_data.shape
    print(f"\n  Running LIF reservoir on {N} observations × {n_ch} channels...")
    t0 = time.time()

    # One reservoir per channel (seeded deterministically)
    reservoirs = {ch: LIFReservoir(seed=42 + ch * 17) for ch in range(n_ch)}

    # BSC6 features: (N, n_ch, 6*256) = (N, n_ch, 1536)
    bsc_all = np.zeros((N, n_ch, 6 * 256))
    for i in range(N):
        if (i + 1) % 10 == 0:  # Print every 10 subjects instead of 100
            print(f"    Processing subject {i+1}/{N}...")
        for ch in range(n_ch):
            spikes = reservoirs[ch].forward(raw_data[i, :, ch])
            bsc_all[i, ch, :] = bsc6_encode(spikes, n_bins=6)

    elapsed = time.time() - t0
    print(f"  Reservoir complete in {elapsed:.0f}s ({elapsed/60:.1f} minutes)")

    # PCA per channel → (N, n_ch, n_pca)
    print(f"  Applying PCA-{n_pca} per channel...")
    pca_all = np.zeros((N, n_ch, n_pca))
    for ch in range(n_ch):
        pca = PCA(n_components=n_pca)
        pca_all[:, ch, :] = pca.fit_transform(bsc_all[:, ch, :])

    # Flatten: (N, n_ch * n_pca)
    embedding = pca_all.reshape(N, -1)
    print(f"  Embedding shape: {embedding.shape}")

    return embedding


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

def evaluate(features, y, subjects, name, n_folds=10):
    """Run subject-level stratified group K-fold CV with LogReg."""
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (tr, te) in enumerate(gkf.split(features, y, subjects)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(features[tr])
        X_te = scaler.transform(features[te])

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
        embedding = d.get('embedding', d.get('lsm_bsc6_pca', None))
        if embedding is not None and embedding.ndim == 3:
            embedding = embedding.reshape(embedding.shape[0], -1)
        print(f"  Embedding shape: {embedding.shape}")
    else:
        embedding = extract_reservoir_features(raw_data, n_pca=64)

    # Subject-centered versions
    print("\n  Computing subject-centered versions...")
    raw_flat_c = subject_center(raw_flat, subjects)
    embedding_c = subject_center(embedding, subjects)

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
        embedding, y, subjects,
        "(3) Reservoir + LogReg, UNCENTERED", args.n_folds)
    print("  Reservoir uncentered complete.")
    print("  Running 10-fold CV for reservoir centered...")
    results['res_centered'] = evaluate(
        embedding_c, y, subjects,
        "(4) Reservoir + LogReg, SUBJECT-CENTERED", args.n_folds)
    print("  Reservoir centered complete.")

    # ── Also run SVM for comparison with existing Chapter 5 numbers ──
    print("\n  ── SVM cross-check (for consistency with Ch5 results) ──")
    gkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    for label, feats in [("Raw EEG uncentered + SVM", raw_flat),
                         ("Reservoir uncentered + SVM", embedding),
                         ("Reservoir centered + SVM", embedding_c)]:
        fold_accs = []
        for tr, te in gkf.split(feats, y, subjects):
            sc = StandardScaler()
            X_tr = sc.fit_transform(feats[tr])
            X_te = sc.transform(feats[te])
            clf = SVC(C=1.0, kernel='rbf', random_state=42)
            clf.fit(X_tr, y[tr])
            fold_accs.append(balanced_accuracy_score(y[te], clf.predict(X_te)))
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        print(f"    {label:45s}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
        results[f"svm_{label.split()[0].lower()}_{label.split()[1].lower()}"] = np.array(fold_accs)

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
