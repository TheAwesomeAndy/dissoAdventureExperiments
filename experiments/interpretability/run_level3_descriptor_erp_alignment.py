#!/usr/bin/env python3
"""
ARSPI-Net Interpretability: Level 3 -- Descriptor-to-ERP Alignment
===================================================================

Validates Level 3 of the four-level interpretability taxonomy:
seven named dynamical descriptors computed from the LIF reservoir
align with classical ERP scalars at the per-channel level.

Key results (dissertation Table 6.1):
  Mean amplitude descriptor vs LPP amplitude: |r| = 0.82 (34-channel median)
  Per-channel peak:  r = 0.837 (Channel 31)
  Mean amplitude descriptor vs P300 amplitude: r = 0.647

This script computes per-channel Pearson correlations between each of
the seven dynamical descriptors (computed from reservoir spike trains)
and classical ERP scalars (P300 amplitude, LPP amplitude, P300 latency)
extracted directly from the input EEG.

The analysis is per-channel: for each channel, we correlate the descriptor
value across 633 observations with the ERP scalar across the same
observations.  This produces a 34-element correlation vector per
descriptor-scalar pair, from which we report the median and peak.

Publication: Lane, A. A. (2026). Affective Reservoir-Spike Processing and
Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic
Framework for Clinical EEG Analysis. PhD Dissertation, Stony Brook University.

Usage:
  python run_level3_descriptor_erp_alignment.py --data_dir /path/to/batch_data/

Requires: numpy, scipy, matplotlib
"""

import numpy as np
import argparse
import os
import re
import pickle
from pathlib import Path
from scipy import stats
from scipy.signal import decimate
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

# -- Constants (consistent with all chapters) --
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
TARGET_SR = 0.9
FS = 1024
DS_FACTOR = 4
FS_DS = FS // DS_FACTOR      # 256 Hz
BASELINE_SAMPLES = 205
POST_STIM_END = 1229

# ERP windows in downsampled samples (256 Hz)
P300_START = int(250 * FS_DS / 1000)   # ~64
P300_END   = int(500 * FS_DS / 1000)   # ~128
LPP_START  = int(400 * FS_DS / 1000)   # ~102
LPP_END    = int(700 * FS_DS / 1000)   # ~179


# ============================================================
# LIF Reservoir (matches all chapters)
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
# Dynamical Descriptors (7 metrics, consistent with Ch6)
# ============================================================
def compute_descriptors(spikes):
    """Seven named dynamical descriptors from a reservoir spike train.

    Returns dict with named keys for Table 6.1 alignment.
    """
    T, N = spikes.shape
    pop_rate = spikes.sum(axis=1) / N  # per-timestep population rate

    # 1. Mean amplitude (total spikes normalized)
    mean_amp = spikes.mean()

    # 2. Amplitude variance
    amp_var = spikes.mean(axis=0).var()

    # 3. Temporal autocorrelation (1/e crossing)
    pc = pop_rate - pop_rate.mean()
    acf = np.correlate(pc, pc, 'full')
    acf = acf[len(acf) // 2:]
    acf = acf / (acf[0] + 1e-12)
    tau_ac = int(np.argmax(acf < 1 / np.e)) if np.any(acf < 1 / np.e) else len(acf)

    # 4. Signal complexity (permutation entropy, order 3)
    order = 3
    patterns = {}
    for i in range(len(pop_rate) - order + 1):
        pat = tuple(np.argsort(pop_rate[i:i + order]))
        patterns[pat] = patterns.get(pat, 0) + 1
    tot = sum(patterns.values())
    import math
    pe = -sum((c / tot) * np.log(c / tot + 1e-12)
              for c in patterns.values()) / np.log(math.factorial(order))

    # 5. Peak latency (timestep of maximum population rate)
    peak_lat = np.argmax(pop_rate)

    # 6. Temporal asymmetry (skewness of population rate)
    temp_asym = float(stats.skew(pop_rate))

    return {
        'mean_amplitude': mean_amp,
        'amplitude_variance': amp_var,
        'temporal_autocorrelation': tau_ac,
        'signal_complexity': pe,
        'peak_latency': peak_lat,
        'temporal_asymmetry': temp_asym,
    }


def extract_erp_scalars(eeg_channel, fs=FS_DS):
    """Extract P300 amplitude, LPP amplitude, and P300 latency from raw EEG."""
    p300_amp = eeg_channel[P300_START:P300_END].mean()
    lpp_amp = eeg_channel[LPP_START:LPP_END].mean()
    p300_lat = P300_START + np.argmax(eeg_channel[P300_START:P300_END])
    return p300_amp, lpp_amp, p300_lat


# ============================================================
# Main Analysis
# ============================================================
def run_alignment(eeg, outdir):
    """Per-channel descriptor-to-ERP alignment analysis."""
    print("=" * 70)
    print("LEVEL 3: Descriptor-to-ERP Per-Channel Alignment (Table 6.1)")
    print("=" * 70)

    n_obs, T, n_ch = eeg.shape
    descriptor_names = [
        'mean_amplitude', 'amplitude_variance', 'temporal_autocorrelation',
        'signal_complexity', 'peak_latency', 'temporal_asymmetry',
    ]
    erp_names = ['P300_amp', 'LPP_amp', 'P300_lat']

    # Storage: (n_ch, n_desc, n_erp)
    corr_matrix = np.zeros((n_ch, len(descriptor_names), len(erp_names)))

    for ch in range(n_ch):
        # Build per-channel reservoir (seeded per channel, matching pipeline)
        reservoir = LIFReservoir(1, N_RES, seed=SEED + ch * 17)

        desc_vals = {d: [] for d in descriptor_names}
        erp_vals = {e: [] for e in erp_names}

        for i in range(n_obs):
            trial = eeg[i, :, ch:ch + 1]  # (T, 1)

            # ERP scalars from raw input
            p300_a, lpp_a, p300_l = extract_erp_scalars(eeg[i, :, ch])
            erp_vals['P300_amp'].append(p300_a)
            erp_vals['LPP_amp'].append(lpp_a)
            erp_vals['P300_lat'].append(p300_l)

            # Reservoir descriptors
            spikes = reservoir.forward(trial)
            desc = compute_descriptors(spikes)
            for d in descriptor_names:
                desc_vals[d].append(desc[d])

        # Correlations
        for di, d in enumerate(descriptor_names):
            for ei, e in enumerate(erp_names):
                r, _ = stats.pearsonr(desc_vals[d], erp_vals[e])
                corr_matrix[ch, di, ei] = r

        if ch % 10 == 0 or ch == n_ch - 1:
            print(f"  Channel {ch:2d}/{n_ch-1} done")

    # -- Report Table 6.1 --
    print("\n" + "=" * 70)
    print("TABLE 6.1: Per-Channel Descriptor-to-ERP Correlations (median across 34 channels)")
    print("=" * 70)
    print(f"{'Descriptor':<28s} {'r(P300 amp)':>12s} {'r(LPP amp)':>12s} {'r(P300 lat)':>12s}")
    print("-" * 70)
    for di, d in enumerate(descriptor_names):
        med_p300 = np.median(corr_matrix[:, di, 0])
        med_lpp  = np.median(corr_matrix[:, di, 1])
        med_lat  = np.median(corr_matrix[:, di, 2])
        print(f"  {d:<26s} {med_p300:>+10.3f}   {med_lpp:>+10.3f}   {med_lat:>+10.3f}")

    # Key result: mean amplitude vs LPP
    ma_lpp = corr_matrix[:, 0, 1]  # mean_amplitude vs LPP_amp
    med_ma_lpp = np.median(np.abs(ma_lpp))
    peak_ch = np.argmax(np.abs(ma_lpp))
    peak_r = ma_lpp[peak_ch]

    print(f"\n  KEY: Mean amplitude vs LPP:  median |r| = {med_ma_lpp:.3f}")
    print(f"       Peak channel: Ch{peak_ch} (r = {peak_r:.3f})")
    print(f"       Dissertation claims: |r| = 0.82, peak r = 0.837 at Ch31")

    # -- Figure --
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ei, (ename, ax) in enumerate(zip(erp_names, axes)):
        for di, d in enumerate(descriptor_names):
            ax.plot(range(n_ch), corr_matrix[:, di, ei],
                    marker='o', markersize=3, label=d, alpha=0.7)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Pearson r')
        ax.set_title(f'Descriptor vs {ename}')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.legend(fontsize=6, loc='best')

    fig.suptitle('Level 3: Per-Channel Descriptor-to-ERP Alignment', fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / 'level3_descriptor_erp_alignment.pdf', dpi=300)
    plt.close(fig)
    print(f"\n  -> Saved level3_descriptor_erp_alignment.pdf")

    return {
        'corr_matrix': corr_matrix,
        'descriptor_names': descriptor_names,
        'erp_names': erp_names,
        'median_ma_lpp': med_ma_lpp,
        'peak_channel': peak_ch,
        'peak_r': peak_r,
    }


# ============================================================
# Data Loading (same as other interpretability scripts)
# ============================================================
def load_eeg(data_dir):
    data_dir = Path(data_dir)
    pkl = data_dir / 'shape_features_211.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        for key in ['raw_eeg', 'X_raw']:
            if key in d:
                return d[key]
        raise KeyError("Pickle lacks 'raw_eeg' or 'X_raw'")

    files = sorted(data_dir.glob('SHAPE_Community_*_BC.txt'))
    if not files:
        files = sorted(data_dir.glob('*.npy'))
    if not files:
        raise FileNotFoundError(
            f"No data in {data_dir}. SHAPE: https://lab-can.com/shape/")

    eeg_list = []
    for f in files:
        if f.suffix == '.txt':
            arr = np.loadtxt(f)
        else:
            arr = np.load(f)
        if arr.shape[0] > BASELINE_SAMPLES:
            arr = arr[BASELINE_SAMPLES:POST_STIM_END]
        arr = decimate(arr, DS_FACTOR, axis=0)
        mu, sd = arr.mean(axis=0), arr.std(axis=0) + 1e-8
        arr = (arr - mu) / sd
        eeg_list.append(arr)

    return np.array(eeg_list)


def main():
    parser = argparse.ArgumentParser(
        description='Level 3 Interpretability: Descriptor-ERP Alignment')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ with EEG files')
    parser.add_argument('--outdir', type=str,
                        default='./interpretability_results')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eeg = load_eeg(args.data_dir)
    print(f"  Data: {eeg.shape[0]} obs, {eeg.shape[2]} channels, "
          f"{eeg.shape[1]} timepoints\n")

    results = run_alignment(eeg, outdir)

    print("\n" + "=" * 70)
    print("SUMMARY -- Level 3 Descriptor-to-ERP Alignment")
    print("=" * 70)
    print(f"  Mean amplitude vs LPP: median |r| = {results['median_ma_lpp']:.3f}")
    print(f"  Peak: Ch{results['peak_channel']} (r = {results['peak_r']:.3f})")


if __name__ == '__main__':
    main()
