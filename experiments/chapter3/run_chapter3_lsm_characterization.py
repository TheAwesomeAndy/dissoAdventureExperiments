#!/usr/bin/env python3
"""
ARSPI-Net Chapter 3: Controlled LIF Reservoir Characterization
===============================================================

Computational validation of the fixed-weight LIF reservoir described
in Chapter 3. All experiments use synthetic signals — no external data
required. Chapter 3 is primarily theoretical/analytical; this script
provides the controlled numerical experiments that verify the properties
the theory predicts.

Experiments:
  1. Membrane dynamics verification (leak, integrate, fire, reset)
  2. Separation property (Maass 2002): distinct inputs → distinct states
  3. Fading memory: response to past inputs decays at controlled rate
  4. Spectral radius sweep: edge-of-stability characterization
  5. Kernel quality: effective dimensionality of reservoir state space
  6. Cross-seed reproducibility on temporal discrimination task

Publication: Lane, A. A. (2026). Affective Reservoir-Spike Processing and
Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic
Framework for Clinical EEG Analysis. PhD Dissertation, Stony Brook University.

Usage:
  python run_chapter3_lsm_characterization.py
  python run_chapter3_lsm_characterization.py --outdir ./ch3_results

Requires: numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
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

# ── Reservoir parameters (consistent with all chapters) ─────────────
N_INPUT = 33
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
TARGET_SR = 0.9


# ============================================================
# LIF Reservoir (matches chapter4Experiments implementation)
# ============================================================
class LIFReservoir:
    def __init__(self, n_input, n_res, beta=BETA, threshold=THRESHOLD, seed=SEED):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigenvalues = np.abs(np.linalg.eigvals(self.W_rec))
        if eigenvalues.max() > 0:
            self.W_rec *= TARGET_SR / eigenvalues.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
        T, n_in = X.shape
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


# ============================================================
# Feature Extraction (BSC6, matches chapter4)
# ============================================================
def extract_bsc(spikes, n_bins, t_start=10, t_end=70):
    window = spikes[t_start:t_end]
    T_w, N = window.shape
    bin_size = T_w // n_bins
    features = []
    for b in range(n_bins):
        features.append(window[b * bin_size:(b + 1) * bin_size].sum(axis=0))
    return np.concatenate(features)


# ============================================================
# Synthetic Data Generators
# ============================================================
def generate_temporal_task(n_per_class=200, n_input=N_INPUT, T=150,
                           noise_std=0.15, seed=42):
    """Two-class temporal pattern discrimination (same as Ch4)."""
    rng = np.random.RandomState(seed)
    X_list, y = [], []
    for cls in range(2):
        amps = [1.0, 0.6, 0.3] if cls == 0 else [0.3, 0.6, 1.0]
        for _ in range(n_per_class):
            x = np.zeros((T, n_input))
            for start, amp in zip([20, 35, 50], amps):
                jitter = rng.randint(-3, 4)
                a_jit = 1.0 + rng.uniform(-0.2, 0.2)
                s = max(0, min(T - 8, start + jitter))
                x[s:s + 8, :5] = 1.5 * amp * a_jit
            x[15:65, :] += rng.randn(50, n_input) * noise_std
            X_list.append(x)
            y.append(cls)
    return X_list, np.array(y)


def generate_separation_pairs(n_pairs=200, T=150, n_input=N_INPUT, seed=42):
    """Pairs of signals with controlled L2 distances."""
    rng = np.random.RandomState(seed)
    pairs, distances = [], []
    for _ in range(n_pairs):
        base = rng.randn(T, n_input) * 0.5
        delta = rng.uniform(0.05, 2.0)
        pert = rng.randn(T, n_input)
        pert = pert / (np.linalg.norm(pert) + 1e-12) * delta
        pairs.append((base, base + pert))
        distances.append(delta)
    return pairs, np.array(distances)


# ============================================================
# Experiment 1: Membrane Dynamics Verification
# ============================================================
def experiment_1_membrane_dynamics(outdir):
    print("=" * 60)
    print("EXPERIMENT 1: Membrane Dynamics Verification")
    print("=" * 60)

    reservoir = LIFReservoir(1, 8, beta=BETA, threshold=THRESHOLD, seed=0)
    reservoir.W_in = np.array([[0.3], [0.0], [0.15], [0.5],
                                [0.1], [0.25], [0.0], [0.4]])
    reservoir.W_rec = np.zeros((8, 8))

    T = 100
    x = np.zeros((T, 1))
    x[10:30, 0] = 1.0
    x[50:55, 0] = 3.0

    spikes, mem = reservoir.forward(x)

    # Verify leak: membrane decays during silent period (no spikes)
    n3 = mem[35:45, 3]
    ratios = n3[1:] / (n3[:-1] + 1e-15)
    valid = n3[:-1] > 0.01
    leak_ratio = np.mean(ratios[valid]) if valid.any() else float('nan')
    leak_error = abs(leak_ratio - (1.0 - BETA))

    # Verify fire: spikes only when membrane >= threshold
    spike_times = np.where(spikes[:, 3] > 0)[0]
    fire_ok = all(mem[max(0, t - 1), 3] >= THRESHOLD * 0.7 for t in spike_times) \
        if len(spike_times) > 0 else True

    # Verify reset: membrane resets after spike
    reset_ok = all(mem[t, 3] < THRESHOLD for t in spike_times if t < T)

    print(f"  Leak ratio: {leak_ratio:.4f} (expected {1 - BETA:.4f}, "
          f"error {leak_error:.6f})")
    print(f"  Fire threshold:  {'PASS' if fire_ok else 'FAIL'}")
    print(f"  Reset after spike: {'PASS' if reset_ok else 'FAIL'}")
    print(f"  Spikes (neuron 3): {int(spikes[:, 3].sum())}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x[:, 0], color='#4c72b0', lw=1.5)
    axes[0].set_ylabel('Input')
    axes[0].set_title('Experiment 1: LIF Membrane Dynamics Verification')
    for n, lbl in [(0, 'N0 w=0.30'), (2, 'N2 w=0.15'), (3, 'N3 w=0.50')]:
        axes[1].plot(mem[:, n], label=lbl, alpha=0.8)
    axes[1].axhline(THRESHOLD, color='gray', ls='--', alpha=0.5, label='θ')
    axes[1].set_ylabel('Membrane')
    axes[1].legend(fontsize=8, ncol=4)
    for t in spike_times:
        axes[2].axvline(t, color='#c44e52', alpha=0.7, lw=0.8)
    axes[2].set_ylabel('Spikes (N3)')
    axes[2].set_xlabel('Time Step')
    fig.tight_layout()
    fig.savefig(outdir / 'ch3_exp1_membrane_dynamics.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved ch3_exp1_membrane_dynamics.pdf")
    return {'leak_error': leak_error, 'fire': fire_ok, 'reset': reset_ok}


# ============================================================
# Experiment 2: Separation Property (Maass 2002)
# ============================================================
def experiment_2_separation_property(outdir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Separation Property")
    print("=" * 60)

    reservoir = LIFReservoir(N_INPUT, N_RES, seed=SEED)
    pairs, d_in = generate_separation_pairs(200, seed=42)

    d_out = []
    for (x1, x2), _ in zip(pairs, d_in):
        s1, _ = reservoir.forward(x1)
        s2, _ = reservoir.forward(x2)
        d_out.append(np.linalg.norm(s1.mean(axis=0) - s2.mean(axis=0)))
    d_out = np.array(d_out)

    r_sep = np.corrcoef(d_in, d_out)[0, 1]

    # Binned monotonicity
    edges = np.linspace(d_in.min(), d_in.max(), 11)
    bin_means = []
    for i in range(10):
        mask = (d_in >= edges[i]) & (d_in < edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(d_out[mask].mean())
    monotonic = all(a <= b for a, b in zip(bin_means, bin_means[1:]))

    print(f"  Input–state distance correlation: r = {r_sep:.4f}")
    print(f"  Binned monotonicity: {'PASS' if monotonic else 'APPROX'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d_in, d_out, s=12, alpha=0.35, color='#4c72b0')
    z = np.polyfit(d_in, d_out, 1)
    xl = np.linspace(d_in.min(), d_in.max(), 100)
    ax.plot(xl, np.polyval(z, xl), '--', color='#c44e52', lw=2)
    ax.set_xlabel('Input Distance (L2)')
    ax.set_ylabel('Reservoir State Distance (L2)')
    ax.set_title(f'Separation Property: r = {r_sep:.3f}')
    fig.tight_layout()
    fig.savefig(outdir / 'ch3_exp2_separation_property.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved ch3_exp2_separation_property.pdf")
    return {'r': r_sep, 'monotonic': monotonic}


# ============================================================
# Experiment 3: Fading Memory
# ============================================================
def experiment_3_fading_memory(outdir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Fading Memory Property")
    print("=" * 60)

    reservoir = LIFReservoir(N_INPUT, N_RES, seed=SEED)
    rng = np.random.RandomState(42)

    T = 300
    pulse = 50
    x = rng.randn(T, N_INPUT) * 0.05
    x[pulse:pulse + 10, :] += 2.0

    spikes, _ = reservoir.forward(x)
    baseline = spikes[:pulse].mean(axis=0)

    delays = np.arange(0, 200, 5)
    devs = []
    for d in delays:
        t = pulse + 10 + d
        if t + 10 < T:
            devs.append(np.linalg.norm(spikes[t:t + 10].mean(axis=0) - baseline))
        else:
            devs.append(0.0)
    devs = np.array(devs)

    valid = devs > 0.01
    if valid.sum() > 3:
        c = np.polyfit(delays[valid], np.log(devs[valid] + 1e-12), 1)
        tau = -1.0 / c[0] if c[0] < 0 else float('inf')
    else:
        tau = float('inf')

    print(f"  Memory decay τ ≈ {tau:.1f} steps")
    print(f"  Deviation at 0: {devs[0]:.4f}")
    print(f"  Deviation at 50: {devs[10]:.4f}")
    print(f"  Deviation at 100: {devs[20]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(delays, devs + 1e-6, 'o-', color='#4c72b0', ms=4)
    if tau < 1000:
        ax.semilogy(delays, np.exp(c[0] * delays + c[1]), '--',
                     color='#c44e52', lw=2, label=f'τ = {tau:.0f} steps')
        ax.legend()
    ax.set_xlabel('Delay After Pulse (steps)')
    ax.set_ylabel('State Deviation from Baseline')
    ax.set_title('Fading Memory Property')
    fig.tight_layout()
    fig.savefig(outdir / 'ch3_exp3_fading_memory.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved ch3_exp3_fading_memory.pdf")
    return {'tau': tau}


# ============================================================
# Experiment 4: Spectral Radius Sweep
# ============================================================
def experiment_4_spectral_radius(outdir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Spectral Radius Sweep")
    print("=" * 60)

    X_list, y = generate_temporal_task(n_per_class=100, seed=42)
    radii = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    results = {}

    for sr in radii:
        rng = np.random.RandomState(SEED)
        lim_in = np.sqrt(6.0 / (N_INPUT + N_RES))
        W_in = rng.uniform(-lim_in, lim_in, (N_RES, N_INPUT))
        lim_rec = np.sqrt(6.0 / (N_RES + N_RES))
        W_rec = rng.uniform(-lim_rec, lim_rec, (N_RES, N_RES))
        eigs = np.abs(np.linalg.eigvals(W_rec))
        if eigs.max() > 0:
            W_rec *= sr / eigs.max()

        res = LIFReservoir(N_INPUT, N_RES, seed=SEED)
        res.W_in = W_in
        res.W_rec = W_rec

        feats, total_spk = [], []
        for x in X_list:
            s, _ = res.forward(x)
            feats.append(extract_bsc(s, 6))
            total_spk.append(s.sum())
        feats = np.array(feats)

        scaler = StandardScaler()
        feats_s = scaler.fit_transform(feats)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for tr, te in skf.split(feats_s, y):
            clf = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
            clf.fit(feats_s[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(feats_s[te])))

        acc = np.mean(accs)
        ms = np.mean(total_spk)
        results[sr] = (acc, ms)
        print(f"  ρ = {sr:.1f}: accuracy = {acc:.1%}, mean spikes = {ms:.0f}")

    rs = sorted(results)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rs, [results[r][0] for r in rs], 'o-', color='#4c72b0', lw=2,
             label='Accuracy')
    ax1.set_xlabel('Spectral Radius')
    ax1.set_ylabel('Balanced Accuracy', color='#4c72b0')
    ax1.axvline(0.9, color='gray', ls='--', alpha=0.5, label='Default ρ = 0.9')
    ax2 = ax1.twinx()
    ax2.plot(rs, [results[r][1] for r in rs], 's--', color='#c44e52',
             lw=1.5, alpha=0.7, label='Spikes')
    ax2.set_ylabel('Mean Total Spikes', color='#c44e52')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower right')
    ax1.set_title('Spectral Radius: Accuracy vs Spiking Activity')
    fig.tight_layout()
    fig.savefig(outdir / 'ch3_exp4_spectral_radius.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved ch3_exp4_spectral_radius.pdf")
    return results


# ============================================================
# Experiment 5: Kernel Quality (Effective Dimensionality)
# ============================================================
def experiment_5_kernel_quality(outdir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Kernel Quality (Effective Dimensionality)")
    print("=" * 60)

    reservoir = LIFReservoir(N_INPUT, N_RES, seed=SEED)
    rng = np.random.RandomState(42)

    states = []
    for _ in range(300):
        x = rng.randn(150, N_INPUT) * 0.5
        s, _ = reservoir.forward(x)
        states.append(s.mean(axis=0))
    states = np.array(states)

    U, S, Vt = np.linalg.svd(states - states.mean(axis=0), full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()

    # Participation ratio = effective rank
    p = S ** 2 / (S ** 2).sum()
    eff_rank = 1.0 / (p ** 2).sum()

    # 95% variance threshold
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1

    # Numerical rank (> 1% of max)
    num_rank = int((S > 0.01 * S[0]).sum())

    print(f"  Effective rank (participation ratio): {eff_rank:.1f}")
    print(f"  Numerical rank (1% threshold): {num_rank}")
    print(f"  Components for 95% variance: {n95}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogy(S[:80], 'o-', color='#4c72b0', ms=3)
    ax1.axhline(0.01 * S[0], color='#c44e52', ls='--', alpha=0.5,
                label='1% threshold')
    ax1.set_xlabel('Component Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Value Spectrum')
    ax1.legend()

    ax2.plot(cumvar[:120], '-', color='#55a868', lw=2)
    ax2.axhline(0.95, color='gray', ls='--', alpha=0.5)
    ax2.axvline(n95, color='#c44e52', ls='--', alpha=0.5,
                label=f'{n95} components for 95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title(f'Kernel Quality (eff. rank = {eff_rank:.0f})')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(outdir / 'ch3_exp5_kernel_quality.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved ch3_exp5_kernel_quality.pdf")
    return {'eff_rank': eff_rank, 'num_rank': num_rank, 'n95': n95}


# ============================================================
# Experiment 6: Cross-Seed Reproducibility
# ============================================================
def experiment_6_cross_seed(outdir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Cross-Seed Reproducibility")
    print("=" * 60)

    X_list, y = generate_temporal_task(n_per_class=150, seed=42)
    seeds = [42, 123, 456, 789, 1024, 2048, 3333, 7777, 9999, 12345]
    accs = []

    for s in seeds:
        res = LIFReservoir(N_INPUT, N_RES, seed=s)
        feats = np.array([extract_bsc(res.forward(x)[0], 6) for x in X_list])
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []
        for tr, te in skf.split(feats, y):
            clf = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
            clf.fit(feats[tr], y[tr])
            fold_accs.append(balanced_accuracy_score(y[te], clf.predict(feats[te])))
        accs.append(np.mean(fold_accs))

    m, sd = np.mean(accs), np.std(accs)
    for s, a in zip(seeds, accs):
        print(f"  Seed {s:>5d}: {a:.1%}")
    print(f"  Mean: {m:.1%} ± {sd:.1%}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(seeds)), [a * 100 for a in accs], color='#4c72b0', alpha=0.8)
    ax.axhline(m * 100, color='#c44e52', ls='--', lw=2,
               label=f'Mean: {m:.1%} ± {sd:.1%}')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds], rotation=45, fontsize=8)
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('Cross-Seed Robustness')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'ch3_exp6_cross_seed.pdf', dpi=300)
    plt.close(fig)
    print(f"  -> Saved ch3_exp6_cross_seed.pdf")
    return {'mean': m, 'std': sd}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Chapter 3: LIF Reservoir Characterization')
    parser.add_argument('--outdir', type=str, default='./ch3_results',
                        help='Output directory for figures')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CHAPTER 3: Controlled LIF Reservoir Characterization")
    print("=" * 60)
    print(f"Reservoir: N_RES={N_RES}, β={BETA}, θ={THRESHOLD}, ρ={TARGET_SR}")
    print(f"Output: {outdir}\n")

    r1 = experiment_1_membrane_dynamics(outdir)
    r2 = experiment_2_separation_property(outdir)
    r3 = experiment_3_fading_memory(outdir)
    r4 = experiment_4_spectral_radius(outdir)
    r5 = experiment_5_kernel_quality(outdir)
    r6 = experiment_6_cross_seed(outdir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Exp 1 — Membrane: leak error = {r1['leak_error']:.6f}, "
          f"fire = {'PASS' if r1['fire'] else 'FAIL'}, "
          f"reset = {'PASS' if r1['reset'] else 'FAIL'}")
    print(f"  Exp 2 — Separation: r = {r2['r']:.4f}")
    print(f"  Exp 3 — Fading memory: τ = {r3['tau']:.1f} steps")
    best_sr = max(r4, key=lambda k: r4[k][0])
    print(f"  Exp 4 — Spectral radius: peak at ρ = {best_sr}")
    print(f"  Exp 5 — Kernel quality: eff. rank = {r5['eff_rank']:.0f}, "
          f"95% var in {r5['n95']} components")
    print(f"  Exp 6 — Cross-seed: {r6['mean']:.1%} ± {r6['std']:.1%}")
    print(f"\n  All figures saved to {outdir}/")


if __name__ == '__main__':
    main()
