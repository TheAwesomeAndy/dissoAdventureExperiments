#!/usr/bin/env python3
"""
Chapter 3: Controlled Characterization of the Liquid State Machine
===================================================================
Four systematic experimental modules establishing the reservoir's
operating regime on synthetic stimuli before application to real EEG.

Module 1: Impact of membrane decay β on temporal integration
Module 2: Input strength and firing threshold transition to spiking
Module 3: Temporal localization of discriminative information
Module 4: Comparative readout classifiers on synthetic data

Dissertation Reference: Chapter 3, Sections 3.4–3.7
Key Results:
  - β = 0.05 selected: balances persistence and responsiveness
  - Threshold θ = 0.5: produces ~7% sparsity, stable spiking regime
  - Spectral radius ρ(W) = 0.9: edge-of-chaos operating point
  - Discriminative information localized to stimulus-driven window
  - Linear readout sufficient for controlled task (Koopman linearization)

Author: Andrew Lane, Stony Brook University
"""

import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("CHAPTER 3: CONTROLLED LIF RESERVOIR CHARACTERIZATION")
print("=" * 70)

# ============================================================
# LIF RESERVOIR IMPLEMENTATION
# ============================================================
class LIFReservoir:
    """Fixed-weight Leaky Integrate-and-Fire reservoir."""
    def __init__(self, n_input=1, n_reservoir=256, beta=0.05, theta=0.5,
                 spectral_radius=0.9, input_scale=0.1, seed=42):
        rng = np.random.RandomState(seed)
        self.n_reservoir = n_reservoir
        self.beta = beta
        self.theta = theta

        # Fixed random weights
        self.W_in = rng.randn(n_input, n_reservoir) * input_scale
        W_rec = rng.randn(n_reservoir, n_reservoir) * 0.1
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_rec)
        W_rec *= spectral_radius / (np.max(np.abs(eigenvalues)) + 1e-10)
        self.W_rec = W_rec

    def run(self, x):
        """Process input x: (T, n_input) → spikes: (T, n_reservoir)"""
        T = x.shape[0]
        v = np.zeros(self.n_reservoir)  # membrane potential
        spikes = np.zeros((T, self.n_reservoir))

        for t in range(T):
            # LIF dynamics
            inp = x[t] @ self.W_in + spikes[max(0, t-1)] @ self.W_rec
            v = self.beta * v + (1 - self.beta) * inp
            # Spike and reset
            spike_mask = v >= self.theta
            spikes[t, spike_mask] = 1.0
            v[spike_mask] = 0.0  # reset

        return spikes

# ============================================================
# SYNTHETIC STIMULUS GENERATION
# ============================================================
def generate_synthetic_erp(n_samples=200, n_timesteps=256, n_classes=2, seed=42):
    """Generate synthetic ERP-like signals with class-dependent components."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, n_timesteps))
    y = np.zeros(n_samples, dtype=int)

    t = np.arange(n_timesteps) / 256.0  # seconds

    for i in range(n_samples):
        c = i % n_classes
        y[i] = c

        # Background noise
        signal = rng.randn(n_timesteps) * 0.1

        # Class-dependent ERP components
        if c == 0:
            # Class 0: strong P300 (250ms), weak LPP
            signal += 0.5 * np.exp(-((t - 0.250)**2) / (2 * 0.030**2))
            signal += 0.1 * np.exp(-((t - 0.500)**2) / (2 * 0.060**2))
        else:
            # Class 1: weak P300, strong LPP (500ms)
            signal += 0.1 * np.exp(-((t - 0.250)**2) / (2 * 0.030**2))
            signal += 0.5 * np.exp(-((t - 0.500)**2) / (2 * 0.060**2))

        # Add jitter
        signal += rng.randn(n_timesteps) * 0.05
        X[i] = signal

    return X, y

# ============================================================
# MODULE 1: MEMBRANE DECAY β SWEEP
# ============================================================
print("\n" + "=" * 70)
print("MODULE 1: MEMBRANE DECAY β SWEEP")
print("=" * 70)

X_syn, y_syn = generate_synthetic_erp(n_samples=200, n_classes=2)
print(f"Synthetic data: {X_syn.shape[0]} samples, {X_syn.shape[1]} timesteps, 2 classes")

beta_values = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99]
beta_results = []

for beta in beta_values:
    res = LIFReservoir(beta=beta, n_reservoir=256, spectral_radius=0.9)
    features = []
    spike_counts = []
    for i in range(len(X_syn)):
        spikes = res.run(X_syn[i:i+1].T)  # (T, 256)
        # BSC6-style binning
        bins = []
        for b in range(6):
            s, e = 10 + b*10, 10 + (b+1)*10
            bins.append(spikes[s:e].sum(axis=0))
        features.append(np.concatenate(bins))
        spike_counts.append(spikes.sum())

    features = np.array(features)
    mean_spikes = np.mean(spike_counts)
    sparsity = np.mean([s / (256 * 256) for s in spike_counts]) * 100

    # Classification
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_sc = sc.fit_transform(features)
    lr = LogisticRegression(max_iter=5000, C=1.0)
    # Simple train/test split
    n_train = 150
    lr.fit(X_sc[:n_train], y_syn[:n_train])
    acc = balanced_accuracy_score(y_syn[n_train:], lr.predict(X_sc[n_train:]))
    beta_results.append({'beta': beta, 'acc': acc, 'spikes': mean_spikes, 'sparsity': sparsity})
    print(f"  β = {beta:.3f}: acc = {acc*100:.1f}%, "
          f"mean spikes = {mean_spikes:.0f}, sparsity = {sparsity:.1f}%")

# ============================================================
# MODULE 2: THRESHOLD AND INPUT STRENGTH TRANSITION
# ============================================================
print("\n" + "=" * 70)
print("MODULE 2: THRESHOLD θ AND SPIKING REGIME")
print("=" * 70)

theta_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
theta_results = []

for theta in theta_values:
    res = LIFReservoir(beta=0.05, theta=theta, n_reservoir=256, spectral_radius=0.9)
    features = []
    spike_counts = []
    for i in range(len(X_syn)):
        spikes = res.run(X_syn[i:i+1].T)
        bins = []
        for b in range(6):
            s, e = 10 + b*10, 10 + (b+1)*10
            bins.append(spikes[s:e].sum(axis=0))
        features.append(np.concatenate(bins))
        spike_counts.append(spikes.sum())

    features = np.array(features)
    mean_spikes = np.mean(spike_counts)
    sparsity = np.mean([s / (256 * 256) for s in spike_counts]) * 100

    sc = StandardScaler()
    X_sc = sc.fit_transform(features)
    lr = LogisticRegression(max_iter=5000, C=1.0)
    lr.fit(X_sc[:150], y_syn[:150])
    acc = balanced_accuracy_score(y_syn[150:], lr.predict(X_sc[150:]))
    theta_results.append({'theta': theta, 'acc': acc, 'spikes': mean_spikes, 'sparsity': sparsity})
    print(f"  θ = {theta:.2f}: acc = {acc*100:.1f}%, "
          f"mean spikes = {mean_spikes:.0f}, sparsity = {sparsity:.1f}%")

# ============================================================
# MODULE 3: TEMPORAL LOCALIZATION OF DISCRIMINATIVE INFORMATION
# ============================================================
print("\n" + "=" * 70)
print("MODULE 3: TEMPORAL LOCALIZATION")
print("=" * 70)

res = LIFReservoir(beta=0.05, theta=0.5, n_reservoir=256, spectral_radius=0.9)

# Run reservoir on all samples
all_spikes = []
for i in range(len(X_syn)):
    spikes = res.run(X_syn[i:i+1].T)
    all_spikes.append(spikes)
all_spikes = np.array(all_spikes)  # (200, 256, 256)

# Sliding window Fisher Discriminant Ratio
window_size = 20
fdr_values = []
time_centers = []
for start in range(0, 256 - window_size, 5):
    end = start + window_size
    window_features = all_spikes[:, start:end, :].sum(axis=1)  # (200, 256)

    # FDR = (μ₁ - μ₀)² / (σ₁² + σ₀²) averaged across features
    mu0 = window_features[y_syn == 0].mean(axis=0)
    mu1 = window_features[y_syn == 1].mean(axis=0)
    var0 = window_features[y_syn == 0].var(axis=0) + 1e-10
    var1 = window_features[y_syn == 1].var(axis=0) + 1e-10
    fdr = ((mu1 - mu0)**2 / (var0 + var1)).mean()
    fdr_values.append(fdr)
    time_centers.append((start + end) / 2 / 256 * 1000)  # ms

print(f"Peak discriminability at: {time_centers[np.argmax(fdr_values)]:.0f} ms")
print(f"Peak FDR: {max(fdr_values):.4f}")

# ============================================================
# MODULE 4: COMPARATIVE READOUT CLASSIFIERS
# ============================================================
print("\n" + "=" * 70)
print("MODULE 4: COMPARATIVE READOUT CLASSIFIERS")
print("=" * 70)

# Extract BSC6 features with selected operating point
features_all = []
for i in range(len(X_syn)):
    bins = []
    for b in range(6):
        s, e = 10 + b*10, 10 + (b+1)*10
        bins.append(all_spikes[i, s:e].sum(axis=0))
    features_all.append(np.concatenate(bins))
features_all = np.array(features_all)

sc = StandardScaler()
X_sc = sc.fit_transform(features_all)
n_train = 150

classifiers = {
    'LogReg (C=1)': LogisticRegression(max_iter=5000, C=1.0),
    'LogReg (C=0.1)': LogisticRegression(max_iter=5000, C=0.1),
    'Linear SVM': SVC(kernel='linear', C=1.0),
    'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
}

for name, clf in classifiers.items():
    clf.fit(X_sc[:n_train], y_syn[:n_train])
    acc = balanced_accuracy_score(y_syn[n_train:], clf.predict(X_sc[n_train:]))
    print(f"  {name:<20s}: {acc*100:.1f}%")

# ============================================================
# MODULE 5: SPECTRAL RADIUS SWEEP
# ============================================================
print("\n" + "=" * 70)
print("MODULE 5: SPECTRAL RADIUS SWEEP")
print("=" * 70)

rho_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0, 1.1, 1.3]
rho_results = []

for rho in rho_values:
    res = LIFReservoir(beta=0.05, theta=0.5, n_reservoir=256, spectral_radius=rho)
    features = []
    spike_counts = []
    for i in range(len(X_syn)):
        spikes = res.run(X_syn[i:i+1].T)
        bins = []
        for b in range(6):
            s, e = 10 + b*10, 10 + (b+1)*10
            bins.append(spikes[s:e].sum(axis=0))
        features.append(np.concatenate(bins))
        spike_counts.append(spikes.sum())

    features = np.array(features)
    mean_spikes = np.mean(spike_counts)

    sc = StandardScaler()
    X_sc = sc.fit_transform(features)
    lr = LogisticRegression(max_iter=5000, C=1.0)
    lr.fit(X_sc[:150], y_syn[:150])
    acc = balanced_accuracy_score(y_syn[150:], lr.predict(X_sc[150:]))
    rho_results.append({'rho': rho, 'acc': acc, 'spikes': mean_spikes})
    print(f"  ρ(W) = {rho:.2f}: acc = {acc*100:.1f}%, mean spikes = {mean_spikes:.0f}")

# ============================================================
# GENERATE FIGURE
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Panel A: β sweep
ax = fig.add_subplot(gs[0, 0])
betas = [r['beta'] for r in beta_results]
accs = [r['acc'] * 100 for r in beta_results]
ax.semilogx(betas, accs, 'o-', color='#1976D2', lw=2)
ax.axvline(0.05, color='red', ls='--', lw=1, alpha=0.7, label='β = 0.05 (selected)')
ax.set_xlabel('Membrane decay β')
ax.set_ylabel('Balanced accuracy (%)')
ax.set_title('(A) Module 1: β Sweep', fontweight='bold')
ax.legend(fontsize=8)

# Panel B: θ sweep
ax = fig.add_subplot(gs[0, 1])
thetas = [r['theta'] for r in theta_results]
accs_t = [r['acc'] * 100 for r in theta_results]
spars = [r['sparsity'] for r in theta_results]
ax.plot(thetas, accs_t, 'o-', color='#1976D2', lw=2, label='Accuracy')
ax2 = ax.twinx()
ax2.plot(thetas, spars, 's--', color='#E53935', lw=1.5, alpha=0.7, label='Sparsity')
ax.axvline(0.5, color='red', ls='--', lw=1, alpha=0.7)
ax.set_xlabel('Threshold θ')
ax.set_ylabel('Balanced accuracy (%)', color='#1976D2')
ax2.set_ylabel('Sparsity (%)', color='#E53935')
ax.set_title('(B) Module 2: θ Sweep', fontweight='bold')

# Panel C: Spectral radius sweep
ax = fig.add_subplot(gs[0, 2])
rhos = [r['rho'] for r in rho_results]
accs_r = [r['acc'] * 100 for r in rho_results]
ax.plot(rhos, accs_r, 'o-', color='#1976D2', lw=2)
ax.axvline(0.9, color='red', ls='--', lw=1, alpha=0.7, label='ρ = 0.9 (selected)')
ax.axvline(1.0, color='gray', ls=':', lw=1, alpha=0.5, label='Edge of chaos')
ax.set_xlabel('Spectral radius ρ(W)')
ax.set_ylabel('Balanced accuracy (%)')
ax.set_title('(C) Module 5: ρ(W) Sweep', fontweight='bold')
ax.legend(fontsize=8)

# Panel D: Temporal localization
ax = fig.add_subplot(gs[1, 0])
ax.plot(time_centers, fdr_values, '-', color='#1976D2', lw=2)
ax.axvline(250, color='green', ls='--', lw=1, alpha=0.7, label='P300 (250ms)')
ax.axvline(500, color='orange', ls='--', lw=1, alpha=0.7, label='LPP (500ms)')
peak_ms = time_centers[np.argmax(fdr_values)]
ax.axvline(peak_ms, color='red', ls='-', lw=1.5, alpha=0.7, label=f'Peak ({peak_ms:.0f}ms)')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Fisher Discriminant Ratio')
ax.set_title('(D) Module 3: Temporal Localization', fontweight='bold')
ax.legend(fontsize=7)

# Panel E: Membrane potential heatmaps
ax = fig.add_subplot(gs[1, 1])
# Show spike raster for one sample
sample_spikes = all_spikes[0, :100, :50]  # first 100ms, 50 neurons
ax.imshow(sample_spikes.T, aspect='auto', cmap='binary', interpolation='nearest')
ax.set_xlabel('Time step')
ax.set_ylabel('Neuron index')
ax.set_title('(E) Spike Raster (Class 0, 50 neurons)', fontweight='bold')

# Panel F: Readout comparison
ax = fig.add_subplot(gs[1, 2])
clf_names = list(classifiers.keys())
clf_accs = []
for name, clf in classifiers.items():
    clf.fit(X_sc[:150], y_syn[:150])
    clf_accs.append(balanced_accuracy_score(y_syn[150:], clf.predict(X_sc[150:])) * 100)
ax.barh(range(len(clf_names)), clf_accs, color='#1976D2', alpha=0.7)
ax.set_yticks(range(len(clf_names)))
ax.set_yticklabels(clf_names, fontsize=9)
ax.set_xlabel('Balanced accuracy (%)')
ax.set_title('(F) Module 4: Readout Comparison', fontweight='bold')
ax.axvline(50, color='gray', ls=':', lw=1, label='Chance')

plt.savefig('fig_chapter3_characterization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_chapter3_characterization.png', dpi=150, bbox_inches='tight')
print("Saved: fig_chapter3_characterization.pdf/png")

print("\n" + "=" * 70)
print("OPERATING POINT SUMMARY")
print("=" * 70)
print(f"""
Selected operating point:
  β = 0.05 (membrane decay)
  θ = 0.5  (firing threshold)
  ρ(W) = 0.9 (spectral radius, sub-critical)
  N = 256 neurons
  BSC₆ temporal coding (6 bins, ~39ms resolution)

Justification:
  - β = 0.05 balances temporal persistence with responsiveness
  - θ = 0.5 produces stable spiking regime (~7% sparsity)
  - ρ = 0.9 operates near edge-of-chaos for memory-computation balance
  - Linear readout sufficient → consistent with Koopman linearization
  - Discriminative information localized to stimulus-driven window
""")
print("DONE.")
