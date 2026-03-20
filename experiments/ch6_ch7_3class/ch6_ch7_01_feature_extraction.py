#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
ARSPI-Net — Script 01: 3-Class Feature Extraction (Chapters 6 & 7)
═══════════════════════════════════════════════════════════════════════

PURPOSE
-------
Drive the validated LIF reservoir (N=256, β=0.05, θ=0.5, seed=42) on
211 subjects × 3 conditions × 34 channels = 21,522 reservoir runs.
Extract 7 core dynamical metrics + 4 extra metrics per channel per
observation, plus theta-band tPLV matrices for topological analysis.

This script produces the computational foundation for:
  Chapter 6 — Dynamical Characterization of Reservoir States
  Chapter 7 — Structure-Function Coupling

MATHEMATICAL MOTIVATION
-----------------------
The reservoir maps continuous EEG u(t) → spike train S ∈ {0,1}^{256×T}
and membrane trajectory M ∈ R^{256×T}. The dynamical metrics characterize
the DRIVEN TRAJECTORY, not the reservoir in isolation (Def. 6.5). Each
metric measures a specific property of the reservoir-as-instrument:

  Amplitude-tracking: total_spikes, MFR, rate_entropy, rate_variance
  Temporal-structure:  perm_entropy, tau_AC
  Sparsity:           temporal_sparsity
  Complexity:         CLZ, lambda_proxy
  Persistence:        tau_relax, T_RTB

The topological metrics (tPLV → strength, clustering) characterize
inter-channel phase synchronization in theta band (4–8 Hz), providing
the spatial descriptors for Chapter 7's coupling analysis.

INPUT
-----
  Raw 3-class EEG: batch_data/*/SHAPE_Community_XXX_IAPS{Neg,Neu,Pos}_BC.txt
  Each file: (1229, 34) float64 — baseline-corrected, trial-averaged μV

OUTPUT
------
  ch6_ch7_3class_features.pkl (~50–200 MB)

RESERVOIR PARAMETERS
--------------------
  Source: run_chapter4_experiments.py (validated operating point)
  Xavier uniform init: limit = sqrt(6/(n_in + n_res))
  Spectral radius scaled to 0.9
  Membrane: mem = (1-β)*mem*(1-spk_prev) + I_tot
  Threshold subtraction, floor at 0
  β=0.05, θ=0.5, N_res=256, seed=42, n_input=1 per channel

RUN COMMAND
-----------
  python ch6_ch7_01_feature_extraction.py

  Adjust DATA_DIR below if batch directories are elsewhere.
  Runtime: ~10–20 minutes depending on hardware.
"""

import numpy as np
import os
import re
import pickle
import time
import math
from scipy.signal import decimate, butter, filtfilt, hilbert

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — adjust paths here
# ══════════════════════════════════════════════════════════════════
DATA_DIR = './batch_data/'      # parent directory with 3-class EEG files
OUTPUT_FILE = './ch6_ch7_3class_features.pkl'
EXCLUDE_SUBJECTS = {127}        # recording anomaly (Ch5 QC)

# Reservoir parameters (Ch4 validated)
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42

# Signal processing
FS_RAW = 1024                   # original sampling rate (Hz)
DOWNSAMPLE = 4                  # decimation factor
FS_DS = FS_RAW // DOWNSAMPLE    # 256 Hz after decimation
TARGET_T = 256                  # timesteps after downsample + truncation
BASELINE_ROWS = 205             # pre-stimulus rows to discard

# Theta-band for tPLV (Chapter 7)
THETA_LOW = 4.0
THETA_HIGH = 8.0

# Condition mapping
COND_MAP = {'Neg': 0, 'Neu': 1, 'Pos': 2}
COND_NAMES = {0: 'Negative', 1: 'Neutral', 2: 'Pleasant'}


# ══════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ══════════════════════════════════════════════════════════════════
def discover_files(data_dir):
    """Find all 3-class EEG files and organize by subject × condition."""
    files = {}
    pattern = re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$')
    for root, dirs, fnames in os.walk(data_dir):
        for fn in fnames:
            m = pattern.search(fn)
            if m:
                sid = int(m.group(1))
                cond = m.group(2)
                if sid in EXCLUDE_SUBJECTS:
                    continue
                if sid not in files:
                    files[sid] = {}
                # Keep latest version if duplicates
                files[sid][cond] = os.path.join(root, fn)
    # Filter to subjects with all 3 conditions
    complete = {s: c for s, c in files.items() if len(c) == 3}
    return complete


# ══════════════════════════════════════════════════════════════════
# RESERVOIR
# ══════════════════════════════════════════════════════════════════
def init_reservoir(n_input, n_res, seed):
    """Initialize reservoir weights — exact replica of Ch4 implementation."""
    rng = np.random.RandomState(seed)
    # Xavier uniform initialization
    limit = np.sqrt(6.0 / (n_input + n_res))
    W_in = rng.uniform(-limit, limit, (n_res, n_input))
    # Recurrent weights with spectral radius 0.9
    W_rec = rng.randn(n_res, n_res) * (1.0 / np.sqrt(n_res))
    eigvals = np.linalg.eigvals(W_rec)
    sr = np.max(np.abs(eigvals))
    if sr > 0:
        W_rec = W_rec * (0.9 / sr)
    return W_in, W_rec


def run_reservoir(signal_1d, W_in, W_rec):
    """
    Drive reservoir with single-channel signal.
    Returns spike train (N_RES × T) and membrane potential (N_RES × T).
    """
    T = len(signal_1d)
    n_res = W_in.shape[0]
    mem = np.zeros(n_res)
    spk = np.zeros(n_res)
    spikes = np.zeros((n_res, T), dtype=np.int8)
    membrane = np.zeros((n_res, T), dtype=np.float32)

    for t in range(T):
        I_in = W_in[:, 0] * signal_1d[t]
        I_rec = W_rec @ spk
        I_tot = I_in + I_rec
        mem = (1 - BETA) * mem * (1 - spk) + I_tot
        mem = np.maximum(mem, 0.0)
        spk = (mem >= THRESHOLD).astype(np.float64)
        mem = mem - spk * THRESHOLD
        spikes[:, t] = spk.astype(np.int8)
        membrane[:, t] = mem.astype(np.float32)

    return spikes, membrane


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def preprocess_eeg(raw, downsample_factor, target_T):
    """
    Preprocess raw EEG: remove baseline, downsample 4×, z-score.
    Input: (1229, 34) raw EEG
    Output: (target_T, 34) z-scored EEG at 256 Hz
    """
    # Remove baseline (first 205 rows)
    post = raw[BASELINE_ROWS:, :]  # (1024, 34)

    n_ch = post.shape[1]
    # Decimate each channel
    ch0 = decimate(post[:, 0], downsample_factor)
    ds = np.zeros((len(ch0), n_ch))
    ds[:, 0] = ch0
    for ch in range(1, n_ch):
        dec_ch = decimate(post[:, ch], downsample_factor)
        n = min(len(dec_ch), ds.shape[0])
        ds[:n, ch] = dec_ch[:n]

    # Truncate to target_T
    if ds.shape[0] >= target_T:
        ds = ds[:target_T]
    else:
        pad = np.zeros((target_T - ds.shape[0], n_ch))
        ds = np.concatenate([ds, pad], axis=0)

    # Z-score per channel
    for ch in range(n_ch):
        mu = ds[:, ch].mean()
        sigma = ds[:, ch].std()
        if sigma > 0:
            ds[:, ch] = (ds[:, ch] - mu) / sigma
        else:
            ds[:, ch] = 0.0

    return ds


# ══════════════════════════════════════════════════════════════════
# tPLV COMPUTATION (full resolution, for Chapter 7)
# ══════════════════════════════════════════════════════════════════
def compute_tplv(raw_post, fs=FS_RAW, f_low=THETA_LOW, f_high=THETA_HIGH):
    """
    Compute 34×34 theta-band time-averaged Phase Locking Value matrix.
    Input: (T_raw, 34) post-stimulus EEG at original sampling rate
    Output: (34, 34) symmetric tPLV matrix
    """
    n_t, n_ch = raw_post.shape
    # Bandpass filter (theta: 4–8 Hz)
    nyq = fs / 2.0
    b, a = butter(3, [f_low / nyq, f_high / nyq], btype='band')
    filtered = np.zeros_like(raw_post)
    for ch in range(n_ch):
        filtered[:, ch] = filtfilt(b, a, raw_post[:, ch])

    # Extract instantaneous phase via Hilbert transform
    phases = np.zeros_like(filtered)
    for ch in range(n_ch):
        analytic = hilbert(filtered[:, ch])
        phases[:, ch] = np.angle(analytic)

    # Compute tPLV: |mean(exp(i*(φ_j - φ_k)))|
    plv = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            phase_diff = phases[:, i] - phases[:, j]
            plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv[j, i] = plv[i, j]
    # Diagonal = 1 (self-locking)
    np.fill_diagonal(plv, 1.0)
    return plv


def tplv_to_topo(plv):
    """
    Extract topological metrics from tPLV matrix.
    Output: (34, 2) — [weighted_strength, weighted_clustering]
    """
    n_ch = plv.shape[0]
    topo = np.zeros((n_ch, 2))

    # Metric 0: Weighted node strength (sum of edge weights)
    for i in range(n_ch):
        topo[i, 0] = np.sum(plv[i, :]) - 1.0  # exclude self-loop

    # Metric 1: Weighted clustering (Onnela formula)
    # C_i = (1/(k_i(k_i-1))) * Σ_{j,k} (w_ij * w_ik * w_jk)^{1/3}
    for i in range(n_ch):
        neighbors = np.where((plv[i, :] > 0) & (np.arange(n_ch) != i))[0]
        k_i = len(neighbors)
        if k_i < 2:
            topo[i, 1] = 0.0
            continue
        tri_sum = 0.0
        for ji, j in enumerate(neighbors):
            for ki_idx in range(ji + 1, len(neighbors)):
                k = neighbors[ki_idx]
                tri_sum += (plv[i, j] * plv[i, k] * plv[j, k]) ** (1.0 / 3.0)
        topo[i, 1] = 2.0 * tri_sum / (k_i * (k_i - 1))

    return topo


# ══════════════════════════════════════════════════════════════════
# DYNAMICAL METRICS — 7 core
# ══════════════════════════════════════════════════════════════════
def compute_core_metrics(spikes, membrane):
    """
    Compute 7 core dynamical metrics for one channel's reservoir response.

    Input:
      spikes:   (N_RES, T) binary spike matrix
      membrane: (N_RES, T) membrane potential matrix

    Output: (7,) array
      [total_spikes, mfr, rate_entropy, rate_variance,
       temporal_sparsity, perm_entropy, tau_ac]
    """
    n_res, T = spikes.shape

    # 0: Total spikes
    total_spikes = float(spikes.sum())

    # 1: Mean firing rate
    mfr = total_spikes / (n_res * T)

    # 2: Rate entropy (entropy of per-neuron firing rate distribution)
    neuron_rates = spikes.sum(axis=1).astype(float) / T
    if neuron_rates.max() > neuron_rates.min():
        hist, edges = np.histogram(neuron_rates, bins=20, density=False)
        hist = hist[hist > 0].astype(float)
        probs = hist / hist.sum()
        rate_entropy = -np.sum(probs * np.log2(probs))
    else:
        rate_entropy = 0.0

    # 3: Rate variance (variance of population firing rate over time)
    pop_rate = spikes.sum(axis=0).astype(float) / n_res
    rate_variance = float(pop_rate.var())

    # 4: Temporal sparsity (fraction of silent timesteps)
    temporal_sparsity = float((pop_rate < (1.0 / n_res)).mean())

    # 5: Permutation entropy of mean membrane potential
    mean_mem = membrane.mean(axis=0).astype(float)
    perm_entropy = permutation_entropy(mean_mem, d=4, tau=1)

    # 6: Autocorrelation decay of population firing rate
    tau_ac = autocorrelation_decay(pop_rate)

    return np.array([total_spikes, mfr, rate_entropy, rate_variance,
                     temporal_sparsity, perm_entropy, tau_ac])


# ══════════════════════════════════════════════════════════════════
# DYNAMICAL METRICS — 4 extra
# ══════════════════════════════════════════════════════════════════
def compute_extra_metrics(spikes, membrane):
    """
    Compute 4 additional dynamical metrics for Chapter 6 experiments.

    Output: (4,) array [CLZ, lambda_proxy, tau_relax, T_RTB]
    """
    n_res, T = spikes.shape
    pop_rate = spikes.sum(axis=0).astype(float) / n_res

    # 0: Lempel-Ziv complexity (population-averaged, sampled neurons)
    clz = lempel_ziv_population(spikes, n_res, T)

    # 1: Lyapunov proxy from membrane potential
    lambda_proxy = lyapunov_proxy(membrane)

    # 2: Relaxation time
    tau_relax = relaxation_time(pop_rate)

    # 3: Return-to-baseline time
    t_rtb = return_to_baseline(pop_rate)

    return np.array([clz, lambda_proxy, tau_relax, t_rtb])


# ══════════════════════════════════════════════════════════════════
# METRIC HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def permutation_entropy(x, d=4, tau=1):
    """Normalized permutation entropy (Bandt & Pompe 2002)."""
    N = len(x)
    n_patterns = N - (d - 1) * tau
    if n_patterns <= 0:
        return 0.0
    pattern_counts = {}
    for i in range(n_patterns):
        idx = list(range(i, i + d * tau, tau))
        if idx[-1] >= N:
            break
        pattern = tuple(np.argsort(x[idx]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    total = sum(pattern_counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(pattern_counts.values()), dtype=float) / total
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(math.factorial(d))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def autocorrelation_decay(x, max_lag=None):
    """Lag at which autocorrelation drops below 1/e (Def. 6.17–6.18)."""
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return 1.0
    N = len(x)
    if max_lag is None:
        max_lag = min(N // 4, 200)
    threshold = np.exp(-1)
    for k in range(1, max_lag):
        acf = np.sum(x[:N - k] * x[k:]) / ((N - k) * var)
        if acf < threshold:
            return float(k)
    return float(max_lag)


def lempel_ziv_population(spikes, n_res, T):
    """Population-averaged normalized LZ complexity (Def. 6.13)."""
    n_sample = min(32, n_res)
    indices = np.linspace(0, n_res - 1, n_sample, dtype=int)
    norm = T / np.log2(T) if T > 1 else 1.0
    complexities = []
    for idx in indices:
        c = lz76_count(spikes[idx, :])
        complexities.append(c / norm)
    return float(np.mean(complexities))


def lz76_count(seq):
    """Count distinct substrings via LZ76 sequential parsing."""
    n = len(seq)
    if n == 0:
        return 0
    complexity = 1
    i = 0
    l = 1
    while i + l <= n:
        # Check if seq[i:i+l] appeared as substring of seq[0:i+l-1]
        sub = seq[i:i + l]
        found = False
        # Search in the history seq[0:i+l-1]
        end = i + l - 1
        for j in range(end):
            if j + l > end:
                break
            if np.array_equal(seq[j:j + l], sub):
                found = True
                break
        if found:
            l += 1
        else:
            complexity += 1
            i += l
            l = 1
    return complexity


def lyapunov_proxy(membrane):
    """
    Approximate driven Lyapunov exponent from membrane potential.
    Uses finite-difference divergence rates on the mean trajectory.
    """
    mean_mem = membrane.mean(axis=0).astype(float)
    T = len(mean_mem)
    if T < 10:
        return 0.0
    diffs = np.abs(np.diff(mean_mem))
    diffs = diffs[diffs > 1e-12]
    if len(diffs) < 2:
        return 0.0
    ratios = diffs[1:] / diffs[:-1]
    ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(np.log(ratios)))


def relaxation_time(pop_rate):
    """Exponential relaxation time (Def. 6.16)."""
    T = len(pop_rate)
    if T < 20:
        return float(T)
    t_peak = np.argmax(pop_rate)
    if t_peak >= T - 10:
        return float(T - t_peak)
    r_inf = np.mean(pop_rate[-max(10, T // 10):])
    r_peak = pop_rate[t_peak]
    if r_peak - r_inf < 1e-8:
        return 1.0
    target = r_inf + (r_peak - r_inf) * np.exp(-1)
    decay = pop_rate[t_peak:]
    for t in range(len(decay)):
        if decay[t] <= target:
            return float(t + 1)
    return float(len(decay))


def return_to_baseline(pop_rate, alpha=0.1, sustain=5):
    """Return-to-baseline time (Def. 6.19)."""
    T = len(pop_rate)
    if T < 20:
        return float(T)
    # Baseline from first 10% of epoch
    n_bl = max(10, T // 10)
    r_baseline = np.mean(pop_rate[:n_bl])
    t_peak = np.argmax(pop_rate)
    r_peak = pop_rate[t_peak]
    if r_peak - r_baseline < 1e-8:
        return 1.0
    threshold = r_baseline + alpha * (r_peak - r_baseline)
    for t in range(t_peak, T - sustain):
        if np.all(pop_rate[t:t + sustain] <= threshold):
            return float(t - t_peak)
    return float(T - t_peak)


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("ARSPI-NET — 3-CLASS FEATURE EXTRACTION (CHAPTERS 6 & 7)")
    print("=" * 70)

    # ── Discover files ──
    file_map = discover_files(DATA_DIR)
    subjects_sorted = sorted(file_map.keys())
    N_subj = len(subjects_sorted)
    N_obs = N_subj * 3
    N_ch = 34

    print(f"\nData: {DATA_DIR}")
    print(f"  Subjects: {N_subj} (excluding {EXCLUDE_SUBJECTS})")
    print(f"  Conditions: 3 (Negative, Neutral, Pleasant)")
    print(f"  Total observations: {N_obs}")
    print(f"  Channels: {N_ch}")
    print(f"  Reservoir runs: {N_obs * N_ch}")

    # ── Initialize reservoir ──
    W_in, W_rec = init_reservoir(1, N_RES, SEED)
    print(f"\nReservoir: N={N_RES}, β={BETA}, θ={THRESHOLD}, seed={SEED}")
    print(f"  W_in: {W_in.shape}, W_rec: {W_rec.shape}")
    print(f"  Spectral radius: {np.max(np.abs(np.linalg.eigvals(W_rec))):.4f}")

    # ── Allocate output arrays ──
    D = np.zeros((N_obs, N_ch, 7), dtype=np.float32)
    D_extra = np.zeros((N_obs, N_ch, 4), dtype=np.float32)
    T_topo = np.zeros((N_obs, N_ch, 2), dtype=np.float32)
    tPLV_mats = np.zeros((N_obs, N_ch, N_ch), dtype=np.float32)
    pop_rate_ts = np.zeros((N_obs, N_ch, TARGET_T), dtype=np.float32)
    y = np.zeros(N_obs, dtype=np.int64)
    subjects = np.zeros(N_obs, dtype=np.int64)

    # ── Process each observation ──
    print(f"\nProcessing {N_obs} observations ({N_ch} channels each)...")
    print(f"  Reservoir: {N_RES} neurons × {TARGET_T} timesteps per channel")
    print(f"  tPLV: theta {THETA_LOW}–{THETA_HIGH} Hz at {FS_RAW} Hz\n")

    t_start = time.time()
    obs_i = 0
    cond_order = ['Neg', 'Neu', 'Pos']

    for si, sid in enumerate(subjects_sorted):
        for cond_key in cond_order:
            filepath = file_map[sid][cond_key]
            cond_label = COND_MAP[cond_key]

            # Load raw EEG
            raw = np.loadtxt(filepath)  # (1229, 34)

            # ── tPLV from full-resolution post-stimulus ──
            raw_post = raw[BASELINE_ROWS:, :]  # (1024, 34)
            plv = compute_tplv(raw_post)
            tPLV_mats[obs_i] = plv.astype(np.float32)
            T_topo[obs_i] = tplv_to_topo(plv).astype(np.float32)

            # ── Downsample for reservoir ──
            eeg = preprocess_eeg(raw, DOWNSAMPLE, TARGET_T)  # (256, 34)

            # ── Drive reservoir on each channel ──
            for ch in range(N_ch):
                spikes, membrane = run_reservoir(eeg[:, ch], W_in, W_rec)
                D[obs_i, ch] = compute_core_metrics(spikes, membrane)
                D_extra[obs_i, ch] = compute_extra_metrics(spikes, membrane)
                pop_rate_ts[obs_i, ch] = (
                    spikes.sum(axis=0).astype(np.float32) / N_RES
                )

            y[obs_i] = cond_label
            subjects[obs_i] = sid
            obs_i += 1

            # Progress
            if obs_i % 20 == 0 or obs_i == N_obs:
                elapsed = time.time() - t_start
                rate = obs_i / elapsed
                remaining = (N_obs - obs_i) / rate if rate > 0 else 0
                print(f"  [{obs_i:4d}/{N_obs}] Subject {sid:3d} "
                      f"{COND_NAMES[cond_label]:10s} | "
                      f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "
                      f"({rate:.1f} obs/s)")

    total_time = time.time() - t_start
    print(f"\n  Feature extraction complete: {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")

    # ── Data integrity check ──
    print(f"\n{'=' * 70}")
    print("DATA INTEGRITY CHECK")
    print(f"{'=' * 70}")
    print(f"  D shape:          {D.shape} (obs × channels × 7 metrics)")
    print(f"  D_extra shape:    {D_extra.shape} (obs × channels × 4 metrics)")
    print(f"  T_topo shape:     {T_topo.shape} (obs × channels × 2 metrics)")
    print(f"  tPLV_mats shape:  {tPLV_mats.shape} (obs × 34 × 34)")
    print(f"  pop_rate_ts:      {pop_rate_ts.shape} (obs × channels × T)")
    print(f"  y distribution:   {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  subjects unique:  {len(np.unique(subjects))}")

    # Metric summaries
    metric_names = ['total_spikes', 'MFR', 'rate_entropy', 'rate_variance',
                    'temporal_sparsity', 'perm_entropy', 'tau_AC']
    print(f"\n  Core metric ranges (across all observations × channels):")
    for k, name in enumerate(metric_names):
        vals = D[:, :, k].flatten()
        print(f"    {name:20s}: mean={vals.mean():.4f}, "
              f"std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    extra_names = ['CLZ', 'lambda_proxy', 'tau_relax', 'T_RTB']
    print(f"\n  Extra metric ranges:")
    for k, name in enumerate(extra_names):
        vals = D_extra[:, :, k].flatten()
        print(f"    {name:20s}: mean={vals.mean():.4f}, "
              f"std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    topo_names = ['weighted_strength', 'weighted_clustering']
    print(f"\n  Topological metric ranges:")
    for k, name in enumerate(topo_names):
        vals = T_topo[:, :, k].flatten()
        print(f"    {name:20s}: mean={vals.mean():.4f}, "
              f"std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # ── Save ──
    output = {
        'D': D,
        'D_extra': D_extra,
        'T_topo': T_topo,
        'tPLV_mats': tPLV_mats,
        'pop_rate_ts': pop_rate_ts,
        'y': y,
        'subjects': subjects,
        'cond_names': COND_NAMES,
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output, f, protocol=4)

    fsize = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n  Saved: {OUTPUT_FILE} ({fsize:.1f} MB)")

    print(f"\n{'=' * 70}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  {N_obs} observations × {N_ch} channels × 11 dynamical metrics")
    print(f"  {N_obs} observations × {N_ch} channels × 2 topological metrics")
    print(f"  {N_obs} tPLV matrices (34 × 34)")
    print(f"  3 classes: Negative ({(y == 0).sum()}), "
          f"Neutral ({(y == 1).sum()}), Pleasant ({(y == 2).sum()})")
    print(f"  Ready for: ch6_ch7_02_raw_observations.py")


if __name__ == '__main__':
    main()
