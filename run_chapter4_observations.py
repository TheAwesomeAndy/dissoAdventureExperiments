#!/usr/bin/env python3
"""
============================================================================
ARSPI-Net Chapter 4: Raw Data Observation Figures
============================================================================
Generates the six raw observation figures for Chapter 4 (Spike-to-Embedding
Pipeline) that visualize the data at every stage of the BSC6→PCA-64 pipeline.

Figures:
  obs01_raw_input_signals.pdf     — Input stimuli for both classes
  obs02_raw_spike_rasters.pdf     — LIF reservoir spike rasters
  obs03_raw_bsc6_features.pdf     — BSC6 feature matrices (256 neurons × 6 bins)
  obs04_raw_embedding_space.pdf   — PCA embedding geometry (BSC6 vs MFR)
  obs05_population_dynamics.pdf   — Population firing rates and sparsity
  obs06_membrane_dynamics.pdf     — Membrane potential heatmaps and traces

Task: Temporal Pattern Discrimination
  Class 0 ("Early Burst"): strong→medium→weak amplitude profile
  Class 1 ("Late Burst"):  weak→medium→strong amplitude profile
  Both classes have equal total energy — rate coding cannot distinguish them.
  Amplitude jitter (±30%), timing jitter (±5 steps), noise (σ=0.5).

Reservoir: 256-neuron LIF, β=0.05, M_th=0.5, feature window steps 10–70.

Usage:
  python run_chapter4_observations.py [--output_dir pictures/chLSMEmbeddings]

Author: Andrew Lane
Date: March 2026
============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
N_INPUT = 33          # Input dimensionality (33 features, first 5 driven)
N_RES = 256           # Reservoir size
BETA = 0.05           # Membrane leak rate
THRESHOLD = 0.5       # Firing threshold
T_TOTAL = 150         # Total simulation timesteps
T_START = 10          # Feature window start
T_END = 70            # Feature window end
N_BINS = 6            # BSC temporal bins
N_SAMPLES = 200       # Samples per class


# ══════════════════════════════════════════════════════════════════
# LIF RESERVOIR
# ══════════════════════════════════════════════════════════════════
class LIFReservoir:
    """Leaky Integrate-and-Fire reservoir with fixed random weights.
    
    Implements the discrete-time LIF dynamics:
        V_i(t+1) = (1-β) * V_i(t) * (1 - s_i(t)) + W_in @ x(t) + W_rec @ s(t)
        s_i(t) = 1 if V_i(t) >= M_th, else 0
    
    Parameters
    ----------
    n_input : int
        Input dimensionality.
    n_res : int
        Number of reservoir neurons.
    beta : float
        Membrane leak rate (1/tau_m).
    threshold : float
        Firing threshold.
    seed : int
        Random seed for weight initialization.
    """
    
    def __init__(self, n_input, n_res, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier uniform initialization for input weights
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        # Xavier uniform for recurrent weights with spectral radius control
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigenvalues = np.abs(np.linalg.eigvals(self.W_rec))
        if eigenvalues.max() > 0:
            self.W_rec *= 0.9 / eigenvalues.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res
    
    def forward(self, X):
        """Process input through the reservoir.
        
        Parameters
        ----------
        X : ndarray, shape (T, n_input)
            Input time series.
            
        Returns
        -------
        spikes : ndarray, shape (T, n_res)
            Binary spike output.
        membrane : ndarray, shape (T, n_res)
            Membrane potential traces.
        """
        T, _ = X.shape
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        membrane = np.zeros((T, self.n_res))
        
        for t in range(T):
            I_tot = self.W_in @ X[t] + self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_tot
            spk = (mem >= self.threshold).astype(float)
            mem = mem - spk * self.threshold
            mem = np.maximum(mem, 0.0)
            spikes[t] = spk
            membrane[t] = mem
            spk_prev = spk
        
        return spikes, membrane


# ══════════════════════════════════════════════════════════════════
# STIMULUS GENERATION
# ══════════════════════════════════════════════════════════════════
def generate_stimuli(n_samples_per_class, seed=42):
    """Generate temporal pattern discrimination stimuli.
    
    Class 0 ("Early Burst"): strong→medium→weak sub-pulse profile.
    Class 1 ("Late Burst"):  weak→medium→strong sub-pulse profile.
    Both classes have approximately equal total energy.
    
    Parameters
    ----------
    n_samples_per_class : int
        Number of trials per class.
    seed : int
        Random seed.
        
    Returns
    -------
    X : ndarray, shape (2*n_samples_per_class, T_TOTAL, N_INPUT)
    y : ndarray, shape (2*n_samples_per_class,)
    """
    rng = np.random.RandomState(seed)
    X_all, y_all = [], []
    
    for cls in [0, 1]:
        for i in range(n_samples_per_class):
            x = np.zeros((T_TOTAL, N_INPUT))
            amp_jitter = 1.0 + rng.uniform(-0.3, 0.3)
            time_jitter = rng.randint(-5, 6)
            
            if cls == 0:  # Early burst: strong → medium → weak
                amps = [1.5 * amp_jitter, 1.0 * amp_jitter, 0.5 * amp_jitter]
            else:         # Late burst:  weak → medium → strong
                amps = [0.5 * amp_jitter, 1.0 * amp_jitter, 1.5 * amp_jitter]
            
            for phase, (amp, start) in enumerate(zip(amps, [15, 30, 45])):
                s = max(0, start + time_jitter)
                e = min(T_TOTAL, s + 12)
                x[s:e, :5] = amp
            
            x += rng.randn(T_TOTAL, N_INPUT) * 0.5
            X_all.append(x)
            y_all.append(cls)
    
    return np.array(X_all), np.array(y_all)


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════
def extract_bsc6(spikes, t_start=T_START, t_end=T_END, n_bins=N_BINS):
    """Extract Binned Spike Count features (BSC6).
    
    Divides the feature window into n_bins equal temporal bins and
    sums spikes per neuron per bin.
    
    Parameters
    ----------
    spikes : ndarray, shape (T, N_res)
    t_start, t_end : int
        Feature window boundaries.
    n_bins : int
        Number of temporal bins.
        
    Returns
    -------
    features : ndarray, shape (N_res * n_bins,)
    """
    window = spikes[t_start:t_end]
    T_w, N = window.shape
    bin_size = T_w // n_bins
    bins = [window[b * bin_size:(b + 1) * bin_size].sum(axis=0) for b in range(n_bins)]
    return np.concatenate(bins)


# ══════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════════
def generate_all_figures(output_dir):
    """Generate all six raw observation figures."""
    
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    
    # ── Generate data ──
    print("Generating stimuli...")
    X_stim, y_stim = generate_stimuli(N_SAMPLES)
    
    print("Processing through reservoir...")
    res = LIFReservoir(N_INPUT, N_RES)
    
    all_spikes = []
    all_membrane = []
    all_bsc6 = []
    all_mfr = []
    
    for i in range(len(X_stim)):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(X_stim)}...")
        spk, mem = res.forward(X_stim[i])
        all_spikes.append(spk)
        all_membrane.append(mem)
        all_bsc6.append(extract_bsc6(spk))
        all_mfr.append(spk[T_START:T_END].mean(axis=0))
    
    all_spikes = np.array(all_spikes)
    all_membrane = np.array(all_membrane)
    bsc6_mat = np.array(all_bsc6)
    mfr_mat = np.array(all_mfr)
    
    c0 = y_stim == 0
    c1 = y_stim == 1
    
    print(f"Data: {len(X_stim)} trials, BSC6: {bsc6_mat.shape}")
    
    # ════════════════════════════════════════════════════════
    # FIGURE 1: Raw Input Signals
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    
    for row, cls in enumerate([0, 1]):
        color = '#e74c3c' if cls == 0 else '#3498db'
        name = 'Early Burst' if cls == 0 else 'Late Burst'
        for col in range(3):
            ax = axes[row, col]
            idx = cls * N_SAMPLES + col * 20
            for ch in range(5):
                ax.plot(X_stim[idx, :, ch] + ch * 3, color=color,
                        linewidth=0.8, alpha=0.8)
            ax.set_xlim(0, T_TOTAL)
            ax.set_ylabel('Input Channels' if col == 0 else '', fontsize=10)
            ax.set_xlabel('Time Step' if row == 1 else '', fontsize=10)
            ax.set_title(f'Class {cls} ({name}), Trial {col + 1}',
                         fontsize=10, fontweight='bold', color=color)
            # Shade sub-pulse phases
            alphas = [0.10, 0.07, 0.04] if cls == 0 else [0.04, 0.07, 0.10]
            for phase, (start, alpha) in enumerate(zip([15, 30, 45], alphas)):
                ax.axvspan(start, start + 12, alpha=alpha, color='red')
    
    fig.suptitle(
        'Raw Input Stimuli: Temporal Pattern Discrimination Task\n'
        'Classes differ in temporal profile (strong→weak vs weak→strong), '
        'not total energy.\n'
        'Amplitude jitter (±30%), timing jitter (±5 steps), '
        'additive noise (σ=0.5).',
        fontsize=12, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(OUT / 'obs01_raw_input_signals.pdf', bbox_inches='tight')
    plt.close()
    print("  obs01_raw_input_signals.pdf")
    
    # ════════════════════════════════════════════════════════
    # FIGURE 2: Raw Spike Rasters
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for row, cls in enumerate([0, 1]):
        color = '#e74c3c' if cls == 0 else '#3498db'
        name = 'Early Burst' if cls == 0 else 'Late Burst'
        for col in range(3):
            ax = axes[row, col]
            idx = cls * N_SAMPLES + col * 20
            spk = all_spikes[idx]
            for n in range(80):
                st = np.where(spk[:, n] > 0)[0]
                ax.scatter(st, np.full_like(st, n), s=0.3, color=color,
                           marker='|', linewidths=0.4)
            ax.axvspan(T_START, T_END, alpha=0.08, color='green',
                       label='Feature window' if col == 0 else '')
            ax.set_xlim(0, T_TOTAL)
            ax.set_ylim(-1, 80)
            ax.set_xlabel('Time Step' if row == 1 else '', fontsize=10)
            ax.set_ylabel('Neuron ID' if col == 0 else '', fontsize=10)
            ax.set_title(f'Class {cls} ({name}), Trial {col + 1}',
                         fontsize=10, fontweight='bold', color=color)
            if col == 0 and row == 0:
                ax.legend(fontsize=7, loc='upper right')
    
    fig.suptitle(
        'Raw Reservoir Spike Rasters (80 of 256 neurons)\n'
        'The LIF reservoir transforms continuous input into sparse binary '
        'spike trains.\n'
        'Green shading: feature extraction window (steps 10–70).',
        fontsize=12, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(OUT / 'obs02_raw_spike_rasters.pdf', bbox_inches='tight')
    plt.close()
    print("  obs02_raw_spike_rasters.pdf")
    
    # ════════════════════════════════════════════════════════
    # FIGURE 3: Raw BSC6 Feature Vectors
    # ════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.8],
                  wspace=0.35, hspace=0.4)
    
    for row, cls in enumerate([0, 1]):
        color = '#e74c3c' if cls == 0 else '#3498db'
        name = 'Early Burst' if cls == 0 else 'Late Burst'
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            idx = cls * N_SAMPLES + col * 20
            bsc = bsc6_mat[idx].reshape(N_RES, N_BINS)
            ax.imshow(bsc[:60, :], aspect='auto', cmap='hot',
                      interpolation='nearest', vmin=0, vmax=bsc.max() * 0.8)
            ax.set_xlabel('Temporal Bin' if row == 1 else '', fontsize=10)
            ax.set_ylabel('Neuron ID' if col == 0 else '', fontsize=10)
            ax.set_xticks(range(6))
            ax.set_xticklabels([f'B{i + 1}' for i in range(6)], fontsize=8)
            ax.set_title(f'Class {cls} ({name}), Trial {col + 1}',
                         fontsize=9, fontweight='bold', color=color)
    
    # Class mean
    mean_c0 = bsc6_mat[c0].mean(0).reshape(N_RES, N_BINS)
    mean_c1 = bsc6_mat[c1].mean(0).reshape(N_RES, N_BINS)
    
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(mean_c0[:60, :], aspect='auto', cmap='hot',
              interpolation='nearest')
    ax.set_title('Class 0 Mean', fontsize=9, fontweight='bold', color='#e74c3c')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'B{i + 1}' for i in range(6)], fontsize=8)
    ax.set_ylabel('Neuron ID', fontsize=10)
    
    # Difference
    ax = fig.add_subplot(gs[1, 3])
    diff = mean_c0 - mean_c1
    im = ax.imshow(diff[:60, :], aspect='auto', cmap='RdBu_r',
                   interpolation='nearest', vmin=-diff.max(), vmax=diff.max())
    ax.set_title('Class 0 − Class 1\nDifference', fontsize=9, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'B{i + 1}' for i in range(6)], fontsize=8)
    ax.set_xlabel('Temporal Bin', fontsize=10)
    ax.set_ylabel('Neuron ID', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7, label='ΔSpike Count')
    
    fig.suptitle(
        'Raw BSC₆ Feature Vectors (256 neurons × 6 temporal bins = '
        '1,536 dimensions)\n'
        "Each row is one neuron's binned spike count across 6 temporal "
        'windows.\n'
        'Class 0 concentrates activity in early bins; Class 1 in late bins.',
        fontsize=12, fontweight='bold', y=1.04)
    plt.savefig(OUT / 'obs03_raw_bsc6_features.pdf', bbox_inches='tight')
    plt.close()
    print("  obs03_raw_bsc6_features.pdf")
    
    # ════════════════════════════════════════════════════════
    # FIGURE 4: Embedding Space Geometry
    # ════════════════════════════════════════════════════════
    pca2_bsc = PCA(2, random_state=42).fit_transform(bsc6_mat)
    pca2_mfr = PCA(2, random_state=42).fit_transform(mfr_mat)
    pca_full = PCA(random_state=42).fit(bsc6_mat)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    ax = axes[0]
    ax.scatter(pca2_bsc[c0, 0], pca2_bsc[c0, 1], s=8, alpha=0.5,
               c='#e74c3c', label='Class 0 (Early)')
    ax.scatter(pca2_bsc[c1, 0], pca2_bsc[c1, 1], s=8, alpha=0.5,
               c='#3498db', label='Class 1 (Late)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA-2 Projection of BSC₆\n(clear linear separability)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    
    ax = axes[1]
    ax.scatter(pca2_mfr[c0, 0], pca2_mfr[c0, 1], s=8, alpha=0.5,
               c='#e74c3c', label='Class 0')
    ax.scatter(pca2_mfr[c1, 0], pca2_mfr[c1, 1], s=8, alpha=0.5,
               c='#3498db', label='Class 1')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('PCA-2 Projection of MFR\n(complete class overlap — '
                 'rate coding fails)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    
    ax = axes[2]
    ax.plot(range(1, len(cum_var) + 1), cum_var * 100, 'b-', linewidth=1.5)
    ax.axhline(50.5, color='gray', linestyle=':', label='50.5% (64 PCs)')
    ax.axvline(64, color='#e74c3c', linestyle='--', linewidth=1,
               label='PCA-64 cutoff')
    ax.axvline(5, color='#27ae60', linestyle='--', linewidth=1,
               label='5 PCs (99.5% acc)')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance (%)')
    ax.set_title('PCA Eigenspectrum\n(discriminative structure in first '
                 'few PCs)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 200)
    ax.grid(alpha=0.3)
    
    fig.suptitle(
        'Embedding Space Geometry: BSC₆ vs MFR\n'
        'BSC₆ produces linearly separable classes; MFR produces complete '
        'overlap (equal-energy task).',
        fontsize=12, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(OUT / 'obs04_raw_embedding_space.pdf', bbox_inches='tight')
    plt.close()
    print("  obs04_raw_embedding_space.pdf")
    
    # ════════════════════════════════════════════════════════
    # FIGURE 5: Population Dynamics
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Population firing rate traces
    ax = axes[0]
    for cls, color, label in [(0, '#e74c3c', 'Class 0 (Early)'),
                               (1, '#3498db', 'Class 1 (Late)')]:
        pop_rate = all_spikes[y_stim == cls].mean(axis=2).mean(axis=0)
        ax.plot(pop_rate, color=color, linewidth=1.5, label=label)
    ax.axvspan(T_START, T_END, alpha=0.08, color='green')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Population Firing Rate')
    ax.set_title('Population Firing Rate\n(class-averaged)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, T_TOTAL)
    
    # Sparsity histogram
    ax = axes[1]
    sparsity = 1 - all_spikes[:, T_START:T_END, :].mean(axis=(1, 2))
    ax.hist(sparsity[c0], bins=30, alpha=0.6, color='#e74c3c',
            label='Class 0', density=True)
    ax.hist(sparsity[c1], bins=30, alpha=0.6, color='#3498db',
            label='Class 1', density=True)
    ax.set_xlabel('Spike Sparsity (1 − mean rate)')
    ax.set_ylabel('Density')
    ax.set_title(f'Spike Sparsity Distribution\n'
                 f'(mean = {sparsity.mean():.3f})',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Per-neuron sorted rates
    ax = axes[2]
    mfr_c0 = all_spikes[c0, T_START:T_END, :].mean(axis=(0, 1))
    mfr_c1 = all_spikes[c1, T_START:T_END, :].mean(axis=(0, 1))
    sort_idx = np.argsort(mfr_c0)[::-1]
    ax.bar(range(N_RES), mfr_c0[sort_idx], width=1, alpha=0.6,
           color='#e74c3c', label='Class 0')
    ax.bar(range(N_RES), mfr_c1[sort_idx], width=1, alpha=0.6,
           color='#3498db', label='Class 1')
    ax.set_xlabel('Neuron (sorted by Class 0 rate)')
    ax.set_ylabel('Mean Firing Rate')
    ax.set_title('Per-Neuron Firing Rates\n(sorted; near-identical totals = '
                 'rate coding fails)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, N_RES)
    
    fig.suptitle(
        'Reservoir Population Dynamics: Why Rate Coding Fails\n'
        'Both classes produce similar population rates (equal energy) '
        'but different temporal profiles.',
        fontsize=12, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.savefig(OUT / 'obs05_population_dynamics.pdf', bbox_inches='tight')
    plt.close()
    print("  obs05_population_dynamics.pdf")
    
    # ════════════════════════════════════════════════════════
    # FIGURE 6: Membrane Potential Dynamics
    # ════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, cls in enumerate([0, 1]):
        color = '#e74c3c' if cls == 0 else '#3498db'
        name = 'Early Burst' if cls == 0 else 'Late Burst'
        idx = cls * N_SAMPLES + 10
        mem = all_membrane[idx]
        
        # Left: membrane potential heatmap
        ax = axes[row, 0]
        im = ax.imshow(mem[:, :80].T, aspect='auto', cmap='viridis',
                       interpolation='nearest')
        ax.set_xlabel('Time Step' if row == 1 else '', fontsize=10)
        ax.set_ylabel('Neuron ID', fontsize=10)
        ax.set_title(f'Class {cls} ({name}): Membrane Potential\n'
                     f'(80 neurons, color = voltage)',
                     fontsize=10, fontweight='bold', color=color)
        ax.axvline(T_START, color='white', linestyle='--', linewidth=0.8)
        ax.axvline(T_END, color='white', linestyle='--', linewidth=0.8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Right: selected neuron traces
        ax = axes[row, 1]
        for n_idx, n in enumerate([5, 25, 50, 75]):
            ax.plot(mem[:, n] + n_idx * 1.5, color=color,
                    linewidth=0.8, alpha=0.8)
            ax.axhline(THRESHOLD + n_idx * 1.5, color='gray',
                       linewidth=0.3, linestyle=':')
        ax.axvspan(T_START, T_END, alpha=0.08, color='green')
        ax.set_xlabel('Time Step' if row == 1 else '', fontsize=10)
        ax.set_ylabel('Membrane Potential (offset)', fontsize=10)
        ax.set_title(f'Class {cls}: Selected Neuron Traces\n'
                     f'(4 neurons, dotted = threshold)',
                     fontsize=10, fontweight='bold', color=color)
        ax.set_xlim(0, T_TOTAL)
    
    fig.suptitle(
        'Membrane Potential Dynamics During Stimulus Processing\n'
        'The LIF reservoir transforms input through integrate-and-fire '
        'dynamics.\n'
        'White dashed lines mark the feature extraction window '
        '(steps 10–70).',
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'obs06_membrane_dynamics.pdf', bbox_inches='tight')
    plt.close()
    print("  obs06_membrane_dynamics.pdf")
    
    print(f"\nAll 6 observation figures saved to {OUT}/")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ARSPI-Net Chapter 4: Raw Data Observation Figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generates 6 figures showing the data at every stage of the BSC6→PCA-64
embedding pipeline for the temporal pattern discrimination task.

Example:
  python run_chapter4_observations.py --output_dir pictures/chLSMEmbeddings
        """)
    parser.add_argument('--output_dir', type=str,
                        default='pictures/chLSMEmbeddings',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ARSPI-Net Chapter 4: Raw Data Observation Figures")
    print("=" * 70)
    generate_all_figures(args.output_dir)
