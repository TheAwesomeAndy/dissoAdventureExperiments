#!/usr/bin/env python3
"""
Level 1 Interpretability Validation: Temporal Traceability
===========================================================
BSC6 Bin-to-ERP Correlation Analysis

Computes Pearson correlations between BSC6-aligned temporal bin activations
and classical ERP scalars (P300 amplitude, LPP amplitude) from centered
ERP data, validating Level 1 of the four-level interpretability taxonomy.

Dissertation Reference: Chapter 5, Section 5.13.3 / Chapter 6, Section 6.10
Key Results:
  - Grand-average: r = 0.65 (P300), |r| = 0.82 (LPP)
  - Per-channel peak: r = 0.837 (Channel 31, LPP)
  - Temporal resolution sweep: 6 bins optimal (67.0%), degrades at 12/24

Requires: shape_features_211.pkl (SHAPE dataset features)
Author: Andrew Lane, Stony Brook University
"""

import numpy as np
import pickle
import warnings
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("LEVEL 1 VALIDATION: TEMPORAL TRACEABILITY")
print("BSC6 Bin-to-ERP Correlation Analysis")
print("=" * 70)

with open('shape_features_211.pkl', 'rb') as f:
    data = pickle.load(f)

X_ds = data['X_ds'].astype(np.float32)  # (633, 256, 34) raw EEG
y = data['y']                            # (633,) conditions
subjects = data['subjects']              # (633,) subject IDs
del data

n_obs, n_time, n_chan = X_ds.shape
fs = 256
cond_names = ['Negative', 'Neutral', 'Pleasant']

# Per-subject centering
X_c = X_ds.copy()
for s in np.unique(subjects):
    idx = np.where(subjects == s)[0]
    sm = X_ds[idx].mean(axis=0)
    for i in idx:
        X_c[i] -= sm

print(f"Data: {n_obs} observations, {n_time} timesteps, {n_chan} channels")

# ============================================================
# DEFINE ERP SCALARS
# ============================================================
# P300 amplitude: mean amplitude in 250-500 ms window
p300_start, p300_end = int(0.250 * fs), int(0.500 * fs)  # steps 64-128
erp_p300 = X_c[:, p300_start:p300_end, :].mean(axis=1)   # (633, 34)
erp_p300_grand = erp_p300.mean(axis=1)                     # (633,)

# LPP amplitude: mean amplitude in 500-800 ms window
lpp_start, lpp_end = int(0.500 * fs), int(0.800 * fs)     # steps 128-205
erp_lpp = X_c[:, lpp_start:lpp_end, :].mean(axis=1)       # (633, 34)
erp_lpp_grand = erp_lpp.mean(axis=1)                       # (633,)

# P300 peak latency
p300_latency = np.argmax(np.abs(X_c[:, p300_start:p300_end, :].mean(axis=2)), axis=1)
p300_latency_ms = (p300_latency + p300_start) / fs * 1000

print(f"\nERP Scalars (centered, grand-averaged across channels):")
print(f"  P300 amplitude: {erp_p300_grand.mean():.4f} ± {erp_p300_grand.std():.4f}")
print(f"  LPP amplitude:  {erp_lpp_grand.mean():.4f} ± {erp_lpp_grand.std():.4f}")
print(f"  P300 latency:   {p300_latency_ms.mean():.1f} ± {p300_latency_ms.std():.1f} ms")

# ============================================================
# EXPERIMENT 1: BSC6 BIN-TO-ERP CORRELATION (GRAND AVERAGE)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: BSC6 BIN-TO-ERP CORRELATION (GRAND AVERAGE)")
print("=" * 70)

# BSC6 bin edges (in steps at 256 Hz)
bin_edges = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70)]
bin_names = ['Early\n39-78ms', 'N1\n78-117ms', 'P2\n117-156ms',
             'N2\n156-195ms', 'P300\n195-234ms', 'LPP\n234-273ms']

# Compute per-channel mean amplitude in each BSC6 window
bin_features = np.zeros((n_obs, n_chan, 6))
for b, (s, e) in enumerate(bin_edges):
    bin_features[:, :, b] = X_c[:, s:e, :].mean(axis=1)

# Grand-average bin features (average across channels)
bin_grand = bin_features.mean(axis=1)  # (633, 6)

# Correlations
print(f"\nGrand-average bin-to-ERP correlations:")
print(f"{'Bin':<20s} {'r(P300)':<12s} {'r(LPP)':<12s}")
print("-" * 44)
for b in range(6):
    r_p300, p_p300 = stats.pearsonr(bin_grand[:, b], erp_p300_grand)
    r_lpp, p_lpp = stats.pearsonr(bin_grand[:, b], erp_lpp_grand)
    print(f"  {bin_names[b].replace(chr(10), ' '):<18s} {r_p300:+.4f}       {r_lpp:+.4f}")

# Overall correlation (best bin)
r_p300_best = max(abs(stats.pearsonr(bin_grand[:, b], erp_p300_grand)[0]) for b in range(6))
r_lpp_best = max(abs(stats.pearsonr(bin_grand[:, b], erp_lpp_grand)[0]) for b in range(6))
print(f"\n  Best |r| for P300: {r_p300_best:.4f}")
print(f"  Best |r| for LPP:  {r_lpp_best:.4f}")

# Combined correlation using all 6 bins
from sklearn.linear_model import Ridge as RidgeReg
r_combined_p300 = np.corrcoef(bin_grand.mean(axis=1), erp_p300_grand)[0, 1]
r_combined_lpp = np.corrcoef(bin_grand.mean(axis=1), erp_lpp_grand)[0, 1]
print(f"\n  Combined mean-bin r(P300): {r_combined_p300:.4f}")
print(f"  Combined mean-bin r(LPP): {r_combined_lpp:.4f}")

# ============================================================
# EXPERIMENT 2: PER-CHANNEL BIN-TO-ERP CORRELATION
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: PER-CHANNEL BIN-TO-ERP CORRELATION")
print("=" * 70)

r_per_channel_p300 = np.zeros(n_chan)
r_per_channel_lpp = np.zeros(n_chan)

for ch in range(n_chan):
    # Use all 6 bins for this channel, correlate with that channel's ERP scalar
    ch_bins_mean = bin_features[:, ch, :].mean(axis=1)
    r_per_channel_p300[ch] = stats.pearsonr(ch_bins_mean, erp_p300[:, ch])[0]
    r_per_channel_lpp[ch] = stats.pearsonr(ch_bins_mean, erp_lpp[:, ch])[0]

print(f"\nPer-channel correlations (mean-bin vs channel-specific ERP):")
print(f"  P300: mean |r| = {np.abs(r_per_channel_p300).mean():.4f}, "
      f"max |r| = {np.abs(r_per_channel_p300).max():.4f} "
      f"(Channel {np.argmax(np.abs(r_per_channel_p300))})")
print(f"  LPP:  mean |r| = {np.abs(r_per_channel_lpp).mean():.4f}, "
      f"max |r| = {np.abs(r_per_channel_lpp).max():.4f} "
      f"(Channel {np.argmax(np.abs(r_per_channel_lpp))})")

# ============================================================
# EXPERIMENT 3: MULTIVARIATE RIDGE REGRESSION (R² for LPP/P300)
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: DESCRIPTOR-TO-ERP REGRESSION (R²)")
print("=" * 70)

# Six descriptors from the bin features (per-channel)
# Mean amplitude, amplitude variance, temporal autocorrelation,
# signal complexity, peak latency, temporal asymmetry
descriptors = np.zeros((n_obs, 6))
for i in range(n_obs):
    signal = X_c[i, 10:70, :].mean(axis=1)  # reservoir feature window, channel-avg
    descriptors[i, 0] = signal.mean()                          # mean amplitude
    descriptors[i, 1] = signal.var()                           # amplitude variance
    descriptors[i, 2] = np.corrcoef(signal[:-1], signal[1:])[0, 1]  # lag-1 autocorr
    descriptors[i, 3] = np.diff(signal).var()                  # signal complexity
    descriptors[i, 4] = np.argmax(np.abs(signal)) / len(signal)  # peak latency (normalized)
    descriptors[i, 5] = signal[:len(signal)//2].mean() - signal[len(signal)//2:].mean()  # asymmetry

desc_names = ['Mean Amplitude', 'Amplitude Variance', 'Temporal Autocorr.',
              'Signal Complexity', 'Peak Latency', 'Temporal Asymmetry']

# Individual descriptor correlations
print(f"\nIndividual descriptor-to-ERP correlations:")
print(f"{'Descriptor':<25s} {'r(P300)':<12s} {'r(LPP)':<12s}")
print("-" * 49)
for d in range(6):
    r_p, _ = stats.pearsonr(descriptors[:, d], erp_p300_grand)
    r_l, _ = stats.pearsonr(descriptors[:, d], erp_lpp_grand)
    print(f"  {desc_names[d]:<23s} {r_p:+.4f}       {r_l:+.4f}")

# Cross-validated Ridge regression
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

for target_name, target in [('P300', erp_p300_grand), ('LPP', erp_lpp_grand)]:
    r2_scores = []
    for tri, tei in sgkf.split(descriptors, y, subjects):
        sc = StandardScaler()
        ridge = Ridge(alpha=1.0)
        ridge.fit(sc.fit_transform(descriptors[tri]), target[tri])
        pred = ridge.predict(sc.transform(descriptors[tei]))
        ss_res = np.sum((target[tei] - pred) ** 2)
        ss_tot = np.sum((target[tei] - target[tei].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)
    print(f"\n  Cross-validated Ridge R² for {target_name}: "
          f"{np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# ============================================================
# EXPERIMENT 4: TEMPORAL RESOLUTION SWEEP
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: TEMPORAL RESOLUTION SWEEP")
print("=" * 70)

for n_bins in [3, 6, 12, 24]:
    window_size = 60 // n_bins  # steps in the 10-70 range
    bin_feats_sweep = np.zeros((n_obs, n_chan, n_bins))
    for b in range(n_bins):
        s_step = 10 + b * window_size
        e_step = s_step + window_size
        bin_feats_sweep[:, :, b] = X_c[:, s_step:e_step, :].mean(axis=1)

    X_sweep = bin_feats_sweep.reshape(n_obs, -1)
    accs = []
    for tri, tei in sgkf.split(X_sweep, y, subjects):
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=5000, C=1.0)
        lr.fit(sc.fit_transform(X_sweep[tri]), y[tri])
        accs.append(balanced_accuracy_score(y[tei], lr.predict(sc.transform(X_sweep[tei]))))
    print(f"  {n_bins:2d} bins: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")

# ============================================================
# GENERATE FIGURE
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Bin-to-ERP correlation bars
ax = axes[0]
r_p300_vals = [abs(stats.pearsonr(bin_grand[:, b], erp_p300_grand)[0]) for b in range(6)]
r_lpp_vals = [abs(stats.pearsonr(bin_grand[:, b], erp_lpp_grand)[0]) for b in range(6)]
x = np.arange(6); w = 0.35
ax.bar(x - w/2, r_p300_vals, w, color='#FF8A65', label='P300', edgecolor='#BF360C', alpha=0.8)
ax.bar(x + w/2, r_lpp_vals, w, color='#64B5F6', label='LPP', edgecolor='#0D47A1', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n.replace('\n', '\n') for n in bin_names], fontsize=7)
ax.set_ylabel('|Pearson r|')
ax.set_title('(A) BSC₆ Bin–ERP Correlation', fontweight='bold')
ax.legend()

# Panel B: Per-channel LPP correlation
ax = axes[1]
ax.bar(range(n_chan), np.abs(r_per_channel_lpp), color='#64B5F6', alpha=0.7)
ax.set_xlabel('Channel')
ax.set_ylabel('|Pearson r| with LPP')
ax.set_title('(B) Per-Channel LPP Recovery', fontweight='bold')
peak_ch = np.argmax(np.abs(r_per_channel_lpp))
ax.annotate(f'Ch{peak_ch}\nr={np.abs(r_per_channel_lpp[peak_ch]):.3f}',
            (peak_ch, np.abs(r_per_channel_lpp[peak_ch])),
            textcoords='offset points', xytext=(10, 5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red'))

# Panel C: Descriptor-to-ERP alignment
ax = axes[2]
r_desc_p300 = [abs(stats.pearsonr(descriptors[:, d], erp_p300_grand)[0]) for d in range(6)]
r_desc_lpp = [abs(stats.pearsonr(descriptors[:, d], erp_lpp_grand)[0]) for d in range(6)]
x = np.arange(6); w = 0.35
ax.bar(x - w/2, r_desc_p300, w, color='#FF8A65', label='→ P300', alpha=0.8)
ax.bar(x + w/2, r_desc_lpp, w, color='#64B5F6', label='→ LPP', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([n.split()[0] for n in desc_names], fontsize=7, rotation=30, ha='right')
ax.set_ylabel('|Pearson r|')
ax.set_title('(C) Descriptor–ERP Alignment\n(R² = 0.661 LPP)', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('fig_level1_validation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_level1_validation.png', dpi=150, bbox_inches='tight')
print("Saved: fig_level1_validation.pdf/png")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'bin_erp_correlations': {
        'r_p300_per_bin': r_p300_vals,
        'r_lpp_per_bin': r_lpp_vals,
        'r_p300_best': r_p300_best,
        'r_lpp_best': r_lpp_best,
    },
    'per_channel_correlations': {
        'r_p300': r_per_channel_p300,
        'r_lpp': r_per_channel_lpp,
    },
    'descriptor_correlations': {
        'r_p300': r_desc_p300,
        'r_lpp': r_desc_lpp,
        'names': desc_names,
    },
}
with open('level1_validation_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Results saved to level1_validation_results.pkl")
print("\nDONE.")
