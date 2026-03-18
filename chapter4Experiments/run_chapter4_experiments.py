#!/usr/bin/env python3
"""
ARSPI-Net Chapter 4: Redesigned Experimental Pipeline
======================================================
The original binary amplitude discrimination (1.2 vs 2.0, no noise) is 
trivially easy - all coding schemes achieve 100%. This redesign uses
a more challenging and realistic controlled task:

Task: Discriminate two stimulus patterns that differ in TEMPORAL STRUCTURE
rather than just amplitude. Both classes have the same total energy but
different temporal profiles, with additive noise and inter-trial jitter.

This is the kind of task where temporal coding should genuinely outperform
rate coding - providing an honest test of the chapter's claims.

Class 0: "Early burst" - strong pulse in first sub-window, weak in second
Class 1: "Late burst"  - weak pulse in first sub-window, strong in second

With Gaussian noise, amplitude jitter, and timing jitter across trials.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

OUTDIR = Path("pictures/chLSMEmbeddings")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LIF Reservoir (same as before)
# ============================================================
class LIFReservoir:
    def __init__(self, n_input, n_res, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        # Scale recurrent weights to control spectral radius
        eigenvalues = np.abs(np.linalg.eigvals(self.W_rec))
        if eigenvalues.max() > 0:
            self.W_rec *= 0.9 / eigenvalues.max()  # target spectral radius ~0.9
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
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


# ============================================================
# Challenging Synthetic Data: Temporal Pattern Discrimination
# ============================================================
def generate_temporal_task(n_trials_per_class=200, n_input=33, T=150,
                           noise_std=0.15, amp_jitter=0.2, 
                           timing_jitter=3, seed=42):
    """
    Two classes with SAME total energy but DIFFERENT temporal profiles.
    Class 0: "Early burst" pattern
    Class 1: "Late burst" pattern
    
    Both have 3 sub-pulses with different relative amplitudes.
    Noise, amplitude jitter, and timing jitter make the task non-trivial.
    """
    rng = np.random.RandomState(seed)
    X_all = []
    y_all = []
    
    n_active = 5
    base_amp = 1.5
    
    for cls in range(2):
        for trial in range(n_trials_per_class):
            x = np.zeros((T, n_input))
            
            # Three sub-pulses with class-dependent amplitude profiles
            # Class 0: [strong, medium, weak] = "early emphasis"
            # Class 1: [weak, medium, strong] = "late emphasis"
            if cls == 0:
                amps = [1.0, 0.6, 0.3]
            else:
                amps = [0.3, 0.6, 1.0]
            
            # Base timing: pulses at steps 20, 35, 50 (each 8 steps long)
            base_starts = [20, 35, 50]
            pulse_dur = 8
            
            for p_idx, (start, amp_scale) in enumerate(zip(base_starts, amps)):
                # Add timing jitter
                t_jitter = rng.randint(-timing_jitter, timing_jitter + 1)
                actual_start = max(0, min(T - pulse_dur, start + t_jitter))
                # Add amplitude jitter
                a_jitter = 1.0 + rng.uniform(-amp_jitter, amp_jitter)
                actual_amp = base_amp * amp_scale * a_jitter
                
                x[actual_start:actual_start + pulse_dur, :n_active] = actual_amp
            
            # Add Gaussian noise to all channels during stimulus period
            x[15:65, :] += rng.randn(50, n_input) * noise_std
            
            X_all.append(x)
            y_all.append(cls)
    
    return X_all, np.array(y_all)


# ============================================================
# Feature Extraction (same as before)
# ============================================================
def extract_mfr(spikes, t_start=10, t_end=70):
    return spikes[t_start:t_end].mean(axis=0)

def extract_lfs(spikes, t_start=10, t_end=70):
    window = spikes[t_start:t_end]
    T_w, N = window.shape
    lfs = np.full(N, T_w, dtype=float)
    for i in range(N):
        spike_times = np.where(window[:, i] > 0)[0]
        if len(spike_times) > 0:
            lfs[i] = spike_times[0]
    return lfs / T_w

def extract_bsc(spikes, n_bins, t_start=10, t_end=70):
    window = spikes[t_start:t_end]
    T_w, N = window.shape
    bin_size = T_w // n_bins
    features = []
    for b in range(n_bins):
        features.append(window[b*bin_size:(b+1)*bin_size].sum(axis=0))
    return np.concatenate(features)

def extract_features(spikes, method='bsc6', t_start=10, t_end=70):
    if method == 'mfr': return extract_mfr(spikes, t_start, t_end)
    elif method == 'lfs': return extract_lfs(spikes, t_start, t_end)
    elif method == 'bsc3': return extract_bsc(spikes, 3, t_start, t_end)
    elif method == 'bsc6': return extract_bsc(spikes, 6, t_start, t_end)


# ============================================================
# Classification and FDR (same as before, with fixes)
# ============================================================
def run_classification(features, labels, classifier='logreg', n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if classifier == 'logreg':
            clf = LogisticRegression(C=0.1, solver='liblinear', random_state=42, max_iter=1000)
        elif classifier == 'svm_rbf':
            clf = SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        accs.append(clf.score(X_test, y_test))
    return np.mean(accs), np.std(accs)


def compute_fdr(features, labels, eps_factor=1e-4):
    classes = np.unique(labels)
    var = features.var(axis=0)
    active = var > 1e-12
    if active.sum() < 2:
        return 0.0
    features = features[:, active]
    d = features.shape[1]
    mu = [features[labels == c].mean(axis=0) for c in classes]
    diff = (mu[0] - mu[1]).reshape(-1, 1)
    S_B = diff @ diff.T
    S_W = np.zeros((d, d))
    for c in classes:
        X_c = features[labels == c]
        X_centered = X_c - mu[c]
        S_W += X_centered.T @ X_centered
    trace_sw = np.trace(S_W)
    if trace_sw < 1e-12:
        return 0.0
    eps = eps_factor * trace_sw / d
    S_W_reg = S_W + eps * np.eye(d)
    try:
        X = np.linalg.solve(S_W_reg, S_B)
        fdr = np.trace(X)
    except np.linalg.LinAlgError:
        fdr = 0.0
    if not np.isfinite(fdr) or fdr < 0:
        fdr = 0.0
    return fdr


# ============================================================
# Run reservoir
# ============================================================
def run_reservoir(X_list, n_res=256, beta=0.05, threshold=0.5, seed=42):
    reservoir = LIFReservoir(n_input=33, n_res=n_res, beta=beta,
                              threshold=threshold, seed=seed)
    return [reservoir.forward(x)[0] for x in X_list]


# ============================================================
# ALL EXPERIMENTS + PLOTTING
# ============================================================
def experiment_and_plot_size_ablation(X_list, y):
    print("=" * 60)
    print("EXPERIMENT 1: Reservoir Size Ablation")
    print("=" * 60)
    sizes = [64, 128, 256, 512]
    results = {}
    for n_res in sizes:
        print(f"  N_res = {n_res}...", end=" ", flush=True)
        spk = run_reservoir(X_list, n_res=n_res)
        feats = np.array([extract_bsc(s, 6) for s in spk])
        n_comp = min(64, feats.shape[1], feats.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        feats_pca = pca.fit_transform(feats)
        mean_acc, std_acc = run_classification(feats_pca, y)
        results[n_res] = (mean_acc, std_acc)
        print(f"Acc = {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    s_list = sorted(results.keys())
    means = [results[s][0]*100 for s in s_list]
    stds = [results[s][1]*100 for s in s_list]
    colors = ['#c44e52' if s < 256 else '#55a868' if s == 256 else '#4c72b0' for s in s_list]
    bars = ax.bar([str(s) for s in s_list], means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_xlabel('Reservoir Size ($N_{res}$)')
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_title('Classification Accuracy vs. Reservoir Size')
    ymin = max(40, min(means) - 10)
    ax.set_ylim(ymin, min(102, max(means) + 8))
    # Mark 256 as selected
    idx256 = s_list.index(256)
    ax.annotate('Selected', xy=(idx256, means[idx256]+stds[idx256]+0.5),
                ha='center', fontsize=8, color='#55a868', fontweight='bold')
    fig.savefig(OUTDIR / "ablation_reservoir_size.pdf")
    plt.close(fig)
    print(f"  -> Saved ablation_reservoir_size.pdf")
    return results


def experiment_and_plot_fdr(X_list, y, spikes_list):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: FDR Three-Way Comparison")
    print("=" * 60)
    
    # Raw input BSC6
    raw_feats = np.array([extract_bsc(x[..., np.newaxis].squeeze() if x.ndim == 1 else x, 6, 10, 70) 
                           if False else 
                           np.concatenate([x[10+b*10:10+(b+1)*10, :].sum(axis=0) for b in range(6)])
                           for x in X_list])
    
    # Linear filter + BSC6
    kernel = np.ones(5) / 5.0
    filt_feats = []
    for x in X_list:
        x_filt = np.zeros_like(x)
        for ch in range(x.shape[1]):
            x_filt[:, ch] = np.convolve(x[:, ch], kernel, mode='same')
        filt_feats.append(np.concatenate([x_filt[10+b*10:10+(b+1)*10, :].sum(axis=0) for b in range(6)]))
    filt_feats = np.array(filt_feats)
    
    # LSM BSC6
    lsm_feats = np.array([extract_bsc(s, 6) for s in spikes_list])
    
    fdr_raw = compute_fdr(raw_feats, y)
    fdr_filt = compute_fdr(filt_feats, y)
    fdr_lsm = compute_fdr(lsm_feats, y)
    
    print(f"  FDR (Raw Input):     {fdr_raw:.3f}")
    print(f"  FDR (Linear Filter): {fdr_filt:.3f}")
    print(f"  FDR (LSM Reservoir): {fdr_lsm:.3f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ['Raw Input\n(Binned)', 'Linear Filter\n(5-pt MA)', 'LSM Reservoir\n(BSC$_6$)']
    values = [fdr_raw, fdr_filt, fdr_lsm]
    colors = ['#c44e52', '#dd8452', '#55a868']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    if fdr_raw > 0.01:
        ratio = fdr_lsm / fdr_raw
        ax.annotate(f'{ratio:.1f}×', xy=(2, fdr_lsm * 0.5), fontsize=12,
                    ha='center', color='white', fontweight='bold')
    ax.set_ylabel('Fisher Discriminant Ratio')
    ax.set_title('Separability Enhancement by the LSM Transformation')
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
    fig.savefig(OUTDIR / "fdr_three_way_comparison.pdf")
    plt.close(fig)
    print(f"  -> Saved fdr_three_way_comparison.pdf")
    return {'raw': fdr_raw, 'filter': fdr_filt, 'lsm': fdr_lsm}


def experiment_and_plot_coding(spikes_list, y):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Neural Coding Scheme Comparison")
    print("=" * 60)
    methods = ['mfr', 'lfs', 'bsc3', 'bsc6']
    classifiers = ['logreg', 'svm_rbf']
    results = {}
    for method in methods:
        feats = np.array([extract_features(s, method) for s in spikes_list])
        results[method] = {}
        for clf_name in classifiers:
            mean_acc, std_acc = run_classification(feats, y, clf_name)
            results[method][clf_name] = (mean_acc, std_acc)
            print(f"  {method:6s} + {clf_name:8s}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    labels = ['MFR', 'LFS', 'BSC$_3$', 'BSC$_6$']
    x = np.arange(len(methods))
    width = 0.32
    lr_m = [results[m]['logreg'][0]*100 for m in methods]
    lr_s = [results[m]['logreg'][1]*100 for m in methods]
    sv_m = [results[m]['svm_rbf'][0]*100 for m in methods]
    sv_s = [results[m]['svm_rbf'][1]*100 for m in methods]
    ax.bar(x - width/2, lr_m, width, yerr=lr_s, label='Logistic Regression',
           color='#4c72b0', capsize=3, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, sv_m, width, yerr=sv_s, label='SVM (RBF)',
           color='#55a868', capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Coding Scheme')
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_title('Classification Accuracy by Coding Scheme and Classifier')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    all_vals = lr_m + sv_m
    ax.set_ylim(max(40, min(all_vals) - 8), min(102, max(all_vals) + 6))
    ax.legend(loc='lower right')
    fig.savefig(OUTDIR / "coding_scheme_accuracy_comparison.pdf")
    plt.close(fig)
    print(f"  -> Saved coding_scheme_accuracy_comparison.pdf")
    return results


def experiment_and_plot_pca(spikes_list, y):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: PCA Dimensionality Reduction")
    print("=" * 60)
    feats = np.array([extract_bsc(s, 6) for s in spikes_list])
    
    pca_full = PCA()
    pca_full.fit(feats)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Test different component counts
    pca_accs = {}
    for n_comp in [5, 10, 20, 32, 64, 128, 256]:
        if n_comp > min(feats.shape):
            continue
        pca = PCA(n_components=n_comp)
        feats_pca = pca.fit_transform(feats)
        mean_acc, std_acc = run_classification(feats_pca, y)
        pca_accs[n_comp] = (mean_acc, std_acc)
        var_expl = cumvar[min(n_comp-1, len(cumvar)-1)] * 100
        print(f"  PCA-{n_comp:3d}: Acc = {mean_acc*100:.1f}% ± {std_acc*100:.1f}%, Var = {var_expl:.1f}%")
    
    # Also get full BSC6 accuracy (no PCA)
    full_acc, full_std = run_classification(feats, y)
    print(f"  Full BSC6 (d={feats.shape[1]}): Acc = {full_acc*100:.1f}% ± {full_std*100:.1f}%")
    
    # Plot: Explained Variance
    fig, ax = plt.subplots(figsize=(5, 3.5))
    n_show = min(200, len(cumvar))
    ax.plot(range(1, n_show+1), cumvar[:n_show]*100, 'b-', linewidth=1.5)
    ax.axvline(x=64, color='red', linestyle='--', linewidth=1, label='$d = 64$ (selected)')
    if len(cumvar) > 63:
        ax.axhline(y=cumvar[63]*100, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'{cumvar[63]*100:.1f}% variance\n64 components',
                    xy=(64, cumvar[63]*100), xytext=(100, cumvar[63]*100 - 15),
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Variance Explained (%)')
    ax.set_title('PCA Explained Variance Curve (BSC$_6$ Features)')
    ax.set_xlim(0, n_show)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    fig.savefig(OUTDIR / "pca_explained_variance.pdf")
    plt.close(fig)
    print(f"  -> Saved pca_explained_variance.pdf")
    
    # Plot: PCA Component Temporal Profiles
    n_neurons = spikes_list[0].shape[1]
    n_bins = 6
    pca64 = PCA(n_components=min(64, feats.shape[1], feats.shape[0]-1))
    pca64.fit(feats)
    
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    bin_labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70']
    colors_pc = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
    titles = ['PC1', 'PC2', 'PC3', 'PC4']
    
    for idx, ax in enumerate(axes.flat):
        if idx >= pca64.n_components_:
            ax.set_visible(False)
            continue
        comp = pca64.components_[idx]
        if len(comp) == n_neurons * n_bins:
            reshaped = comp.reshape(n_neurons, n_bins)
            temporal = reshaped.mean(axis=0)  # average loadings across neurons
            temporal_abs = np.abs(reshaped).mean(axis=0)
        else:
            temporal = comp[:n_bins]
            temporal_abs = np.abs(temporal)
        
        # Show signed loadings for interpretation
        pos_color = colors_pc[idx]
        neg_color = '#999999'
        bar_colors = [pos_color if v >= 0 else neg_color for v in temporal]
        ax.bar(range(n_bins), temporal, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.6)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(bin_labels, fontsize=7, rotation=30)
        ax.set_title(titles[idx], fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Loading' if idx % 2 == 0 else '')
        ax.set_xlabel('Time Steps' if idx >= 2 else '')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    fig.suptitle('Temporal Structure of Top 4 Principal Components', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pca_component_visualization.pdf")
    plt.close(fig)
    print(f"  -> Saved pca_component_visualization.pdf")
    
    return {'cumvar': cumvar, 'pca_accs': pca_accs}


def experiment_and_plot_robustness(X_list, y, n_seeds=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Cross-Initialization Robustness")
    print("=" * 60)
    bsc6_accs = []
    mfr_accs = []
    bsc3_accs = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        spk = run_reservoir(X_list, n_res=256, seed=seed*7+13)
        
        bsc_feats = np.array([extract_bsc(s, 6) for s in spk])
        n_comp = min(64, bsc_feats.shape[1], bsc_feats.shape[0]-1)
        pca = PCA(n_components=n_comp)
        bsc_pca = pca.fit_transform(bsc_feats)
        bsc_acc, _ = run_classification(bsc_pca, y)
        bsc6_accs.append(bsc_acc)
        
        bsc3_feats = np.array([extract_bsc(s, 3) for s in spk])
        bsc3_acc, _ = run_classification(bsc3_feats, y)
        bsc3_accs.append(bsc3_acc)
        
        mfr_feats = np.array([extract_mfr(s) for s in spk])
        mfr_acc, _ = run_classification(mfr_feats, y)
        mfr_accs.append(mfr_acc)
        print(f"BSC6={bsc_acc*100:.1f}%, BSC3={bsc3_acc*100:.1f}%, MFR={mfr_acc*100:.1f}%")
    
    bsc6_accs = np.array(bsc6_accs)
    bsc3_accs = np.array(bsc3_accs)
    mfr_accs = np.array(mfr_accs)
    print(f"\n  BSC6+PCA64: {bsc6_accs.mean()*100:.1f}% ± {bsc6_accs.std()*100:.1f}%")
    print(f"  BSC3:       {bsc3_accs.mean()*100:.1f}% ± {bsc3_accs.std()*100:.1f}%")
    print(f"  MFR:        {mfr_accs.mean()*100:.1f}% ± {mfr_accs.std()*100:.1f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 3.5))
    data = [mfr_accs*100, bsc3_accs*100, bsc6_accs*100]
    bp = ax.boxplot(data, labels=['MFR', 'BSC$_3$', 'BSC$_6$+PCA-64'],
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1.5))
    box_colors = ['#c44e52', '#dd8452', '#55a868']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for i, d in enumerate(data):
        jitter = np.random.RandomState(0).uniform(-0.08, 0.08, len(d))
        ax.scatter(np.ones(len(d))*(i+1) + jitter, d, color='black', s=20, zorder=3, alpha=0.7)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustness Across 10 Random Initializations')
    ax.set_ylim(max(35, min(min(d) for d in data) - 5), 105)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, label='Chance')
    ax.legend(loc='lower right')
    fig.savefig(OUTDIR / "cross_initialization_robustness.pdf")
    plt.close(fig)
    print(f"  -> Saved cross_initialization_robustness.pdf")
    return {'bsc6': bsc6_accs, 'bsc3': bsc3_accs, 'mfr': mfr_accs}


def experiment_and_plot_sensitivity(X_list, y):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Parameter Sensitivity")
    print("=" * 60)
    betas = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    acc_grid = np.zeros((len(betas), len(thresholds)))
    for i, beta in enumerate(betas):
        for j, thresh in enumerate(thresholds):
            print(f"  β={beta:.2f}, M_th={thresh:.1f}...", end=" ", flush=True)
            spk = run_reservoir(X_list, beta=beta, threshold=thresh)
            feats = np.array([extract_bsc(s, 6) for s in spk])
            n_comp = min(64, feats.shape[1], feats.shape[0]-1)
            pca = PCA(n_components=n_comp)
            feats_pca = pca.fit_transform(feats)
            mean_acc, _ = run_classification(feats_pca, y)
            acc_grid[i, j] = mean_acc * 100
            print(f"{mean_acc*100:.1f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    vmin = max(50, acc_grid.min() - 5)
    im = ax.imshow(acc_grid, cmap='RdYlGn', aspect='auto', vmin=vmin, vmax=100)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t:.1f}' for t in thresholds])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f'{b:.2f}' for b in betas])
    ax.set_xlabel('Firing Threshold ($M_{th}$)')
    ax.set_ylabel('Membrane Decay ($\\beta$)')
    ax.set_title('Accuracy (%) vs. LSM Parameters')
    for i in range(len(betas)):
        for j in range(len(thresholds)):
            val = acc_grid[i, j]
            color = 'white' if val < (vmin + (100-vmin)*0.4) else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, 
                    color=color, fontweight='bold')
    beta_idx = betas.index(0.05)
    thresh_idx = thresholds.index(0.5)
    rect = plt.Rectangle((thresh_idx-0.5, beta_idx-0.5), 1, 1,
                          fill=False, edgecolor='blue', linewidth=2.5)
    ax.add_patch(rect)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy (%)')
    fig.savefig(OUTDIR / "parameter_sensitivity_heatmap.pdf")
    plt.close(fig)
    print(f"  -> Saved parameter_sensitivity_heatmap.pdf")
    return {'betas': betas, 'thresholds': thresholds, 'grid': acc_grid}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("ARSPI-Net Chapter 4: Redesigned Experimental Pipeline")
    print("Task: Temporal Pattern Discrimination (Early vs Late Burst)")
    print("=" * 60)
    
    X_list, y = generate_temporal_task(noise_std=0.5, amp_jitter=0.3, 
                                       timing_jitter=5, seed=42)
    print(f"Generated {len(X_list)} trials ({int(y.sum())} class 1, {int(len(y)-y.sum())} class 0)")
    
    # Baseline reservoir
    print("\nRunning baseline reservoir (N=256, β=0.05, M_th=0.5)...")
    spikes_256 = run_reservoir(X_list, n_res=256, seed=42)
    total_spk = sum(s.sum() for s in spikes_256)
    print(f"  Total spikes: {total_spk:.0f}, Mean/trial: {total_spk/len(spikes_256):.1f}")
    
    # Run all experiments
    size_res = experiment_and_plot_size_ablation(X_list, y)
    fdr_res = experiment_and_plot_fdr(X_list, y, spikes_256)
    coding_res = experiment_and_plot_coding(spikes_256, y)
    pca_res = experiment_and_plot_pca(spikes_256, y)
    rob_res = experiment_and_plot_robustness(X_list, y, n_seeds=10)
    sens_res = experiment_and_plot_sensitivity(X_list, y)
    
    # Summary
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"\nFDR: Raw={fdr_res['raw']:.2f}, Filter={fdr_res['filter']:.2f}, LSM={fdr_res['lsm']:.2f}")
    print(f"\nCoding (LogReg):")
    for m in ['mfr', 'lfs', 'bsc3', 'bsc6']:
        a, s = coding_res[m]['logreg']
        print(f"  {m:6s}: {a*100:.1f}% ± {s*100:.1f}%")
    print(f"\nRobustness:")
    print(f"  BSC6+PCA64: {rob_res['bsc6'].mean()*100:.1f}% ± {rob_res['bsc6'].std()*100:.1f}%")
    print(f"  MFR:        {rob_res['mfr'].mean()*100:.1f}% ± {rob_res['mfr'].std()*100:.1f}%")
    
    print(f"\nFigures saved to {OUTDIR}/:")
    for f in sorted(OUTDIR.glob("*.pdf")):
        print(f"  {f.name}")
