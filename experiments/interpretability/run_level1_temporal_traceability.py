#!/usr/bin/env python3
"""
ARSPI-Net Interpretability: Level 1 — Temporal Traceability
============================================================

Validates Level 1 of the four-level interpretability taxonomy:
every output variable traces back to specific input ERP transients
through the explicit BSC₆ temporal coding.

Key results:
  r = 0.82   Pearson correlation between input LPP amplitude and
             BSC₆-reconstructed temporal envelope (per-channel median)
  R² = 0.661 Ridge regression from 7 dynamical descriptors to
             per-trial LPP amplitude

These demonstrate that the reservoir's fixed-weight transformation
preserves — rather than destroys — the dominant temporal structure
of the input ERP.

Publication: Lane, A. A. (2026). Affective Reservoir-Spike Processing and
Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic
Framework for Clinical EEG Analysis. PhD Dissertation, Stony Brook University.

Usage:
  python run_level1_temporal_traceability.py --data_dir /path/to/batch_data/

Requires: numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
import argparse
import os
import re
import pickle
from pathlib import Path
from scipy import stats
from scipy.signal import decimate
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ── Constants (consistent with all chapters) ─────────────────────────
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
TARGET_SR = 0.9
FS = 1024
DS_FACTOR = 4
FS_DS = FS // DS_FACTOR      # 256 Hz
BASELINE_SAMPLES = 205       # 200 ms at 1024 Hz
POST_STIM_END = 1229         # 1000 ms post-stimulus at 1024 Hz
# After downsampling: ~256 samples for 1000 ms post-stimulus
# LPP window: 400–700 ms post-stimulus
LPP_START_MS = 400
LPP_END_MS = 700
LPP_START = int(LPP_START_MS * FS_DS / 1000)   # ~102
LPP_END = int(LPP_END_MS * FS_DS / 1000)       # ~179


# ============================================================
# LIF Reservoir (matches chapter4Experiments implementation)
# ============================================================
class LIFReservoir:
    def __init__(self, n_input, n_res=N_RES, beta=BETA,
                 threshold=THRESHOLD, seed=SEED):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigs = np.abs(np.linalg.eigvals(self.W_rec))
        if eigs.max() > 0:
            self.W_rec *= TARGET_SR / eigs.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
        T = X.shape[0]
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        for t in range(T):
            I_tot = self.W_in @ X[t] + self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_tot
            spk = (mem >= self.threshold).astype(float)
            mem = mem - spk * self.threshold
            mem = np.maximum(mem, 0.0)
            spikes[t] = spk
            spk_prev = spk
        return spikes


# ============================================================
# Feature Extraction
# ============================================================
def bsc6_bin_means(spikes, n_bins=6):
    """Per-bin mean firing rate (temporal envelope from BSC₆)."""
    T = spikes.shape[0]
    bin_size = T // n_bins
    envelope = np.zeros(n_bins)
    for b in range(n_bins):
        envelope[b] = spikes[b * bin_size:(b + 1) * bin_size].mean()
    return envelope


def dynamical_descriptors(spikes):
    """7 dynamical descriptors from a spike train (matches Ch6)."""
    T, N = spikes.shape
    pop_rate = spikes.sum(axis=1)

    # 1. Total spikes
    total = spikes.sum()
    # 2. Mean firing rate
    mfr = spikes.mean()
    # 3. Rate variance
    rv = spikes.mean(axis=0).var()
    # 4. Rate entropy
    r = spikes.mean(axis=0)
    rn = r / (r.sum() + 1e-12)
    re = -np.sum(rn * np.log(rn + 1e-12))
    # 5. Temporal sparsity
    ts = (pop_rate > 0).mean()
    # 6. Permutation entropy (order 3)
    order = 3
    patterns = {}
    for i in range(len(pop_rate) - order + 1):
        pat = tuple(np.argsort(pop_rate[i:i + order]))
        patterns[pat] = patterns.get(pat, 0) + 1
    tot = sum(patterns.values())
    import math
    pe = -sum((c / tot) * np.log(c / tot + 1e-12)
              for c in patterns.values()) / np.log(math.factorial(order))
    # 7. Autocorrelation decay (1/e crossing)
    pc = pop_rate - pop_rate.mean()
    acf = np.correlate(pc, pc, 'full')
    acf = acf[len(acf) // 2:]
    acf = acf / (acf[0] + 1e-12)
    tau = int(np.argmax(acf < 1 / np.e)) if np.any(acf < 1 / np.e) else len(acf)

    return np.array([total, mfr, rv, re, ts, pe, tau], dtype=float)


# ============================================================
# Analysis 1: LPP Recovery via BSC₆
# ============================================================
def analysis_lpp_recovery(eeg, subjects, outdir):
    """Correlate input LPP amplitude with BSC₆ temporal envelope."""
    print("=" * 60)
    print("ANALYSIS 1: LPP Recovery via BSC₆ Temporal Traceability")
    print("=" * 60)

    n_obs, T, n_ch = eeg.shape
    # Per-channel reservoir (1-input → N_RES neurons), same seed
    reservoir = LIFReservoir(1, N_RES, seed=SEED)

    per_ch_r = []
    for ch in range(n_ch):
        lpp_amps, bsc_lpps = [], []
        for i in range(n_obs):
            trial = eeg[i, :, ch:ch + 1]  # (T, 1)

            # Input LPP amplitude: mean voltage in 400–700 ms window
            lpp_amp = trial[LPP_START:LPP_END, 0].mean()
            lpp_amps.append(lpp_amp)

            # Reservoir → BSC₆ envelope → LPP bins
            spikes = reservoir.forward(trial)
            env = bsc6_bin_means(spikes)
            # Bins 2–4 roughly span the 333–833 ms window which covers LPP
            bsc_lpp = env[2:5].mean()
            bsc_lpps.append(bsc_lpp)

        r, p = stats.pearsonr(lpp_amps, bsc_lpps)
        per_ch_r.append(r)
        if ch % 10 == 0:
            print(f"  Channel {ch:2d}: r = {r:.3f} (p = {p:.2e})")

    corrs = np.array(per_ch_r)
    med_r = np.median(corrs)
    print(f"\n  Median per-channel r = {med_r:.3f}")
    print(f"  Range: [{corrs.min():.3f}, {corrs.max():.3f}]")
    print(f"  Channels with r > 0.7: {(corrs > 0.7).sum()}/{n_ch}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(n_ch), corrs, color='#4c72b0', alpha=0.8)
    ax1.axhline(med_r, color='#c44e52', ls='--', lw=2,
                label=f'Median r = {med_r:.2f}')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Pearson r')
    ax1.set_title('Level 1: LPP Recovery per Channel')
    ax1.legend()

    ax2.hist(corrs, bins=15, color='#55a868', alpha=0.8, edgecolor='white')
    ax2.axvline(med_r, color='#c44e52', ls='--', lw=2,
                label=f'Median = {med_r:.2f}')
    ax2.set_xlabel('Pearson r')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Per-Channel Correlations')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(outdir / 'level1_lpp_recovery.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved level1_lpp_recovery.pdf")
    return {'median_r': med_r, 'per_channel': corrs}


# ============================================================
# Analysis 2: LPP Prediction from Dynamical Descriptors
# ============================================================
def analysis_lpp_prediction(eeg, subjects, outdir):
    """Predict per-trial LPP amplitude from dynamical descriptors."""
    print("\n" + "=" * 60)
    print("ANALYSIS 2: LPP Prediction from Dynamical Descriptors")
    print("=" * 60)

    n_obs, T, n_ch = eeg.shape
    reservoir = LIFReservoir(n_ch, N_RES, seed=SEED)

    # Per-trial LPP amplitude (grand-average across channels)
    lpp = np.array([eeg[i, LPP_START:LPP_END, :].mean() for i in range(n_obs)])

    # Dynamical descriptors per trial
    descs = []
    for i in range(n_obs):
        spikes = reservoir.forward(eeg[i])
        descs.append(dynamical_descriptors(spikes))
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_obs} trials...")
    descs = np.array(descs)

    # Ridge regression with subject-aware GroupKFold
    unique_subj = np.unique(subjects)
    n_splits = min(10, len(unique_subj))
    gkf = GroupKFold(n_splits=n_splits)

    y_true_all, y_pred_all = [], []
    fold_r2s = []
    for train_idx, test_idx in gkf.split(descs, groups=subjects):
        sc = StandardScaler()
        X_tr = sc.fit_transform(descs[train_idx])
        X_te = sc.transform(descs[test_idx])
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_tr, lpp[train_idx])
        y_pred = ridge.predict(X_te)
        fold_r2s.append(r2_score(lpp[test_idx], y_pred))
        y_true_all.extend(lpp[test_idx])
        y_pred_all.extend(y_pred)

    overall_r2 = r2_score(y_true_all, y_pred_all)
    print(f"\n  Overall R² = {overall_r2:.3f}")
    print(f"  Mean fold R² = {np.mean(fold_r2s):.3f} ± {np.std(fold_r2s):.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_all, y_pred_all, s=8, alpha=0.3, color='#4c72b0')
    lims = [min(min(y_true_all), min(y_pred_all)),
            max(max(y_true_all), max(y_pred_all))]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5)
    ax.set_xlabel('Actual LPP Amplitude (µV)')
    ax.set_ylabel('Predicted LPP Amplitude (µV)')
    ax.set_title(f'LPP Prediction from Dynamical Descriptors\nR² = {overall_r2:.3f}')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(outdir / 'level1_lpp_prediction.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved level1_lpp_prediction.pdf")
    return {'r2': overall_r2, 'fold_r2s': fold_r2s}


# ============================================================
# Data Loading
# ============================================================
def load_eeg(data_dir):
    """Load raw EEG from batch_data/ directory.

    Expected format: .npy files with shape (N_trials, T_raw, N_channels)
    per subject, or a single pickle with all subjects.
    """
    data_dir = Path(data_dir)

    # Try single pickle first
    pkl = data_dir / 'shape_features_211.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        for key in ['raw_eeg', 'X_raw']:
            if key in d:
                return d[key], d['subjects']
        raise KeyError(f"Pickle lacks 'raw_eeg' or 'X_raw' key")

    # Per-subject .npy files
    files = sorted(data_dir.glob('*.npy'))
    if not files:
        raise FileNotFoundError(
            f"No .npy files in {data_dir}. "
            "SHAPE data: https://lab-can.com/shape/")

    eeg_list, subj_list = [], []
    for f in files:
        m = re.search(r'(\d+)', f.stem)
        if not m:
            continue
        subj_id = int(m.group(1))
        arr = np.load(f)
        # Preprocess: take post-stimulus, downsample, z-score
        if arr.shape[0] > BASELINE_SAMPLES:
            arr = arr[BASELINE_SAMPLES:POST_STIM_END]
        arr = decimate(arr, DS_FACTOR, axis=0)
        mu, sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
        arr = (arr - mu) / sd
        eeg_list.append(arr)
        subj_list.append(subj_id)

    return np.array(eeg_list), np.array(subj_list)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Level 1 Interpretability: Temporal Traceability')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ with EEG files')
    parser.add_argument('--outdir', type=str,
                        default='./interpretability_results',
                        help='Output directory for figures')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LEVEL 1 INTERPRETABILITY: Temporal Traceability")
    print("=" * 60)
    print(f"Targets: r = 0.82 (LPP recovery), R² = 0.661 (prediction)\n")

    eeg, subjects = load_eeg(args.data_dir)
    print(f"  Data: {eeg.shape[0]} obs, {eeg.shape[2]} channels, "
          f"{eeg.shape[1]} timepoints")
    print(f"  Subjects: {len(np.unique(subjects))}\n")

    r1 = analysis_lpp_recovery(eeg, subjects, outdir)
    r2 = analysis_lpp_prediction(eeg, subjects, outdir)

    print("\n" + "=" * 60)
    print("SUMMARY — Level 1 Temporal Traceability")
    print("=" * 60)
    print(f"  LPP recovery (median r): {r1['median_r']:.3f}  (target: 0.82)")
    print(f"  LPP prediction (R²):     {r2['r2']:.3f}  (target: 0.661)")


if __name__ == '__main__':
    main()
