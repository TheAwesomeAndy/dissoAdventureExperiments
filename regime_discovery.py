#!/usr/bin/env python3
"""
Pass 1, Deliverable 1.4: Unsupervised Regime Discovery
Tests whether subjects naturally group into dynamical response regimes
that cross-cut diagnostic categories.

Uses the ARSPI-Net reservoir to compute 7 dynamical metrics per channel,
then clusters the descriptor space and tests against clinical labels.
"""

import numpy as np
import pickle
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DELIVERABLE 1.4: UNSUPERVISED REGIME DISCOVERY")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
with open('/home/claude/shape_features_211.pkl', 'rb') as f:
    data = pickle.load(f)

X_ds = data['X_ds']        # (633, 256, 34) — downsampled EEG
y = data['y']              # (633,) — condition labels
subjects = data['subjects'] # (633,) — subject IDs

n_obs, n_time, n_channels = X_ds.shape
print(f"  EEG data: {n_obs} observations × {n_time} time steps × {n_channels} channels")
print(f"  Conditions: {np.unique(y)} (counts: {np.bincount(y)})")
print(f"  Unique subjects: {len(np.unique(subjects))}")

# ── Load clinical metadata ────────────────────────────────────────
import pandas as pd
try:
    clinical = pd.read_csv('/home/claude/data/clinical_profile.csv') if \
        __import__('os').path.exists('/home/claude/data/clinical_profile.csv') else None
except:
    clinical = None

# Also try the uploaded psychopathology file
if clinical is None:
    try:
        psych = pd.read_excel('/mnt/user-data/uploads/SHAPE_Community_Andrew_Psychopathology.xlsx')
        clinical = psych
        print(f"  Clinical data loaded: {clinical.shape}")
    except:
        print("  WARNING: No clinical data found. Will skip diagnosis-cluster analysis.")
        clinical = None

# ── Run LIF Reservoir ─────────────────────────────────────────────
print("\n[2/6] Running LIF reservoir on all observations...")
print("  Parameters: N=256, β=0.05, θ=0.5, spectral_radius=0.9")

# Fixed reservoir parameters
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42

np.random.seed(SEED)

# Generate fixed random weights (same as dissertation pipeline)
W_in = np.random.randn(N_RES, 1) * 0.1  # input weights (1 channel at a time)
W_rec = np.random.randn(N_RES, N_RES) * (1.0 / np.sqrt(N_RES))

# Scale to spectral radius 0.9
eigenvalues = np.linalg.eigvals(W_rec)
spectral_radius = np.max(np.abs(eigenvalues))
W_rec = W_rec * (0.9 / spectral_radius)

def run_lif_reservoir(signal_1d, W_in, W_rec, beta, threshold):
    """Run LIF reservoir on a single-channel signal. Returns spike matrix (N_RES × T)."""
    T = len(signal_1d)
    N = W_in.shape[0]
    mem = np.zeros(N)
    spikes = np.zeros((N, T), dtype=np.float32)
    
    for t in range(T):
        # Input current
        I = W_in[:, 0] * signal_1d[t]
        # Recurrent current from previous spikes
        if t > 0:
            I += W_rec @ spikes[:, t-1]
        # Membrane update with leak
        mem = beta * mem + I
        # Spike generation
        spike = (mem >= threshold).astype(np.float32)
        spikes[:, t] = spike
        # Reset
        mem = mem * (1 - spike)
    
    return spikes

def compute_dynamical_metrics(spike_matrix):
    """Compute 7 dynamical metrics from a spike matrix (N_neurons × T)."""
    N, T = spike_matrix.shape
    
    # 1. Mean Firing Rate
    mfr = np.mean(spike_matrix)
    
    # 2. Rate Entropy — entropy of per-neuron firing rates
    rates = np.mean(spike_matrix, axis=1)
    rates_nonzero = rates[rates > 0]
    if len(rates_nonzero) > 1:
        # Normalize to distribution
        p = rates_nonzero / rates_nonzero.sum()
        rate_entropy = -np.sum(p * np.log2(p + 1e-12))
    else:
        rate_entropy = 0.0
    
    # 3. Permutation Entropy (order 3) on population spike count time series
    pop_count = np.sum(spike_matrix, axis=0)  # T-length
    order = 3
    if T > order:
        # Extract ordinal patterns
        patterns = []
        for t in range(T - order + 1):
            window = pop_count[t:t+order]
            pattern = tuple(np.argsort(window))
            patterns.append(pattern)
        unique, counts = np.unique(patterns, axis=0, return_counts=True)
        p = counts / counts.sum()
        perm_entropy = -np.sum(p * np.log2(p + 1e-12))
        # Normalize by max possible entropy
        import math
        max_entropy = np.log2(math.factorial(order))
        perm_entropy = perm_entropy / max_entropy if max_entropy > 0 else 0
    else:
        perm_entropy = 0.0
    
    # 4. Autocorrelation Time of population spike count
    if np.std(pop_count) > 1e-10:
        pop_centered = pop_count - np.mean(pop_count)
        autocorr = np.correlate(pop_centered, pop_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-12)
        # Find first crossing below 1/e
        tau_ac = T  # default
        for lag in range(1, len(autocorr)):
            if autocorr[lag] < 1.0/np.e:
                tau_ac = lag
                break
    else:
        tau_ac = 0
    
    # 5. Spike-Count Variance (across neurons)
    neuron_counts = np.sum(spike_matrix, axis=1)
    sc_variance = np.var(neuron_counts)
    
    # 6. Active Ratio — fraction of neurons that spike at least once
    active_ratio = np.mean(np.any(spike_matrix > 0, axis=1))
    
    # 7. Mean ISI (across all neurons that spike)
    isis = []
    for n in range(N):
        spike_times = np.where(spike_matrix[n, :] > 0)[0]
        if len(spike_times) > 1:
            isis.extend(np.diff(spike_times).tolist())
    mean_isi = np.mean(isis) if isis else T  # default to full window if no spikes
    
    return np.array([mfr, rate_entropy, perm_entropy, tau_ac, sc_variance, active_ratio, mean_isi])

METRIC_NAMES = ['MFR', 'RateEntropy', 'PermEntropy', 'AutocorrTime', 
                'SpikeCountVar', 'ActiveRatio', 'MeanISI']

# Compute dynamical metrics for all observations × channels
# Shape: (633, 34, 7)
all_metrics = np.zeros((n_obs, n_channels, len(METRIC_NAMES)))

from datetime import datetime
t_start = datetime.now()

for obs_idx in range(n_obs):
    if obs_idx % 50 == 0:
        elapsed = (datetime.now() - t_start).total_seconds()
        eta = (elapsed / max(obs_idx, 1)) * (n_obs - obs_idx)
        print(f"  Processing observation {obs_idx}/{n_obs} "
              f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")
    
    for ch_idx in range(n_channels):
        signal = X_ds[obs_idx, :, ch_idx]
        # Z-score normalize the channel signal
        if np.std(signal) > 1e-10:
            signal = (signal - np.mean(signal)) / np.std(signal)
        
        # Run reservoir
        spikes = run_lif_reservoir(signal, W_in, W_rec, BETA, THRESHOLD)
        
        # Compute dynamical metrics
        all_metrics[obs_idx, ch_idx, :] = compute_dynamical_metrics(spikes)

elapsed_total = (datetime.now() - t_start).total_seconds()
print(f"  Reservoir + metrics complete in {elapsed_total:.0f}s")

# ── Aggregate to subject level ────────────────────────────────────
print("\n[3/6] Aggregating to subject level...")

unique_subjects = np.unique(subjects)
n_subjects = len(unique_subjects)

# Average metrics across conditions for each subject to get subject-level descriptor
# Shape: (n_subjects, 34, 7) → flatten to (n_subjects, 34*7)
subject_metrics = np.zeros((n_subjects, n_channels, len(METRIC_NAMES)))

for i, subj in enumerate(unique_subjects):
    mask = subjects == subj
    subject_metrics[i] = np.mean(all_metrics[mask], axis=0)  # avg across conditions

# Also keep per-observation metrics for condition analysis
# Flatten: (n_obs, 34*7) = (633, 238)
obs_flat = all_metrics.reshape(n_obs, -1)
subj_flat = subject_metrics.reshape(n_subjects, -1)

print(f"  Subject-level descriptors: {subj_flat.shape}")
print(f"  Observation-level descriptors: {obs_flat.shape}")

# ── Clustering ────────────────────────────────────────────────────
print("\n[4/6] Clustering descriptor space...")

# Standardize features
scaler = StandardScaler()
subj_scaled = scaler.fit_transform(subj_flat)

# Test k=2 through k=8
silhouette_scores = {}
cluster_labels = {}

for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
    labels = km.fit_predict(subj_scaled)
    sil = silhouette_score(subj_scaled, labels)
    silhouette_scores[k] = sil
    cluster_labels[k] = labels
    print(f"  k={k}: silhouette={sil:.3f}, sizes={np.bincount(labels)}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\n  Best k by silhouette: k={best_k} (silhouette={silhouette_scores[best_k]:.3f})")

# Also try spectral clustering at best k
try:
    sc = SpectralClustering(n_clusters=best_k, n_init=20, random_state=SEED,
                           affinity='nearest_neighbors', n_neighbors=15)
    spectral_labels = sc.fit_predict(subj_scaled)
    spectral_sil = silhouette_score(subj_scaled, spectral_labels)
    print(f"  Spectral clustering k={best_k}: silhouette={spectral_sil:.3f}")
except Exception as e:
    print(f"  Spectral clustering failed: {e}")
    spectral_labels = cluster_labels[best_k]

# ── UMAP Visualization ───────────────────────────────────────────
print("\n[5/6] UMAP visualization...")

import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED)
embedding_2d = reducer.fit_transform(subj_scaled)

# Create figure with 4 panels
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Colored by best-k cluster
ax = axes[0, 0]
for c in range(best_k):
    mask = cluster_labels[best_k] == c
    ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
              s=20, alpha=0.7, label=f'Regime {c+1} (n={mask.sum()})')
ax.set_title(f'Discovered Response Regimes (k={best_k})', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

# Panel 2: Colored by condition-averaged MFR (dominant metric)
mean_mfr = subject_metrics[:, :, 0].mean(axis=1)  # avg MFR across channels
ax = axes[0, 1]
sc_plot = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                     c=mean_mfr, cmap='viridis', s=20, alpha=0.7)
plt.colorbar(sc_plot, ax=ax, label='Mean Firing Rate')
ax.set_title('Colored by Mean Firing Rate', fontsize=13, fontweight='bold')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

# Panel 3: Colored by permutation entropy
mean_pe = subject_metrics[:, :, 2].mean(axis=1)
ax = axes[1, 0]
sc_plot = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                     c=mean_pe, cmap='plasma', s=20, alpha=0.7)
plt.colorbar(sc_plot, ax=ax, label='Permutation Entropy')
ax.set_title('Colored by Permutation Entropy', fontsize=13, fontweight='bold')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

# Panel 4: Silhouette score by k
ax = axes[1, 1]
ks = sorted(silhouette_scores.keys())
sils = [silhouette_scores[k] for k in ks]
ax.bar(ks, sils, color='steelblue', alpha=0.8)
ax.axhline(y=silhouette_scores[best_k], color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Cluster Quality by k', fontsize=13, fontweight='bold')
ax.set_xticks(ks)

plt.suptitle('Unsupervised Regime Discovery in Reservoir Dynamical Descriptor Space\n'
             '(211 subjects, 7 metrics × 34 channels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/regime_discovery_umap.pdf', dpi=200, bbox_inches='tight')
plt.savefig('/home/claude/regime_discovery_umap.png', dpi=150, bbox_inches='tight')
print("  Saved: regime_discovery_umap.pdf/png")

# ── Clinical Correspondence Analysis ─────────────────────────────
print("\n[6/6] Testing correspondence with clinical categories...")

if clinical is not None:
    # Map clinical data to subjects
    # Try to find subject ID column
    id_cols = [c for c in clinical.columns if 'id' in c.lower() or 'subject' in c.lower() or 'participant' in c.lower()]
    print(f"  Potential ID columns: {id_cols}")
    
    if id_cols:
        id_col = id_cols[0]
        # Find diagnostic columns (binary)
        diag_cols = []
        for col in clinical.columns:
            vals = clinical[col].dropna().unique()
            if len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0, True, False}):
                if clinical[col].sum() >= 5:  # at least 5 positive cases
                    diag_cols.append(col)
        
        print(f"  Found {len(diag_cols)} binary diagnostic columns with ≥5 cases")
        
        # Key diagnoses to test
        key_diag = ['MDD', 'SUD', 'PTSD', 'GAD', 'ADHD']
        
        # For each diagnosis, test whether cluster membership differs
        best_labels = cluster_labels[best_k]
        
        print(f"\n  Testing cluster-diagnosis correspondence (k={best_k}):")
        print(f"  {'Diagnosis':<20} {'Chi2':>8} {'p-value':>10} {'ARI':>8} {'NMI':>8}")
        print(f"  {'-'*56}")
        
        for diag in key_diag:
            # Find matching column
            matching = [c for c in clinical.columns if diag.lower() in c.lower()]
            if matching:
                col = matching[0]
                try:
                    # Get diagnosis values for our subjects
                    diag_values = []
                    for subj in unique_subjects:
                        row = clinical[clinical[id_col] == subj]
                        if len(row) > 0:
                            diag_values.append(int(row[col].values[0]))
                        else:
                            diag_values.append(-1)  # missing
                    
                    diag_arr = np.array(diag_values)
                    valid = diag_arr >= 0
                    
                    if valid.sum() > 20:
                        # Chi-squared test
                        contingency = pd.crosstab(best_labels[valid], diag_arr[valid])
                        chi2, p, dof, expected = stats.chi2_contingency(contingency)
                        
                        # ARI and NMI
                        ari = adjusted_rand_score(diag_arr[valid], best_labels[valid])
                        nmi = normalized_mutual_info_score(diag_arr[valid], best_labels[valid])
                        
                        marker = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
                        print(f"  {diag:<20} {chi2:8.2f} {p:10.4f} {ari:8.3f} {nmi:8.3f}{marker}")
                except Exception as e:
                    print(f"  {diag:<20} Error: {e}")
            else:
                print(f"  {diag:<20} Column not found")
    else:
        print("  Could not identify subject ID column")
else:
    print("  No clinical data available — skipping diagnosis analysis")
    print("  (Clinical correspondence can be tested when clinical_profile.csv is available)")

# ── Characterize discovered regimes ───────────────────────────────
print(f"\n  Regime characterization (k={best_k}):")
best_labels = cluster_labels[best_k]
for c in range(best_k):
    mask = best_labels == c
    regime_metrics = subject_metrics[mask]  # (n_in_cluster, 34, 7)
    mean_vals = regime_metrics.mean(axis=(0, 1))  # avg across subjects and channels
    print(f"\n  Regime {c+1} (n={mask.sum()}):")
    for m_idx, m_name in enumerate(METRIC_NAMES):
        all_mean = subject_metrics[:, :, m_idx].mean()
        regime_mean = mean_vals[m_idx]
        direction = "↑" if regime_mean > all_mean * 1.05 else "↓" if regime_mean < all_mean * 0.95 else "≈"
        print(f"    {m_name:<16} {regime_mean:.4f} (pop mean: {all_mean:.4f}) {direction}")

# ── Test condition sensitivity by regime ──────────────────────────
print(f"\n  Condition sensitivity by regime:")
conditions = ['Negative', 'Neutral', 'Pleasant']
for c in range(best_k):
    # Get subjects in this regime
    regime_subjects = unique_subjects[best_labels == c]
    # Get observations for these subjects
    obs_mask = np.isin(subjects, regime_subjects)
    regime_obs = all_metrics[obs_mask]  # (n_obs_in_regime, 34, 7)
    regime_y = y[obs_mask]
    
    # Test if metrics differ by condition within this regime
    # Average across channels first
    regime_avg = regime_obs.mean(axis=1)  # (n_obs, 7)
    
    print(f"\n  Regime {c+1} (n_obs={obs_mask.sum()}):")
    for m_idx, m_name in enumerate(METRIC_NAMES):
        groups = [regime_avg[regime_y == cond, m_idx] for cond in range(3)]
        if all(len(g) > 2 for g in groups):
            stat, p = stats.kruskal(*groups) if len(groups[0]) > 0 else (0, 1)
            marker = " *" if p < 0.05 else ""
            print(f"    {m_name:<16} H={stat:.2f}, p={p:.4f}{marker}")

# ── Save results ──────────────────────────────────────────────────
results = {
    'all_metrics': all_metrics,           # (633, 34, 7)
    'subject_metrics': subject_metrics,   # (211, 34, 7)
    'metric_names': METRIC_NAMES,
    'cluster_labels': cluster_labels,     # dict: k -> labels
    'best_k': best_k,
    'silhouette_scores': silhouette_scores,
    'embedding_2d': embedding_2d,
    'unique_subjects': unique_subjects,
    'subjects': subjects,
    'y': y,
}
with open('/home/claude/regime_discovery_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\n  Saved: regime_discovery_results.pkl")

print("\n" + "=" * 70)
print("REGIME DISCOVERY COMPLETE")
print("=" * 70)
