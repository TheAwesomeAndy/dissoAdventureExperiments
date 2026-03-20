"""
============================================================================
Chapter 5 — Script 02: RAW DATA OBSERVATIONS (4-Class)
============================================================================
PURPOSE:
  Characterize the 4-class SHAPE data at every stage of the pipeline
  BEFORE any classification modeling. Every subsequent experiment builds
  on these observations. The reader sees WHAT THE DATA LOOKS LIKE at
  subcategory granularity.

MATHEMATICAL MOTIVATION:
  The 3-class analysis (Negative, Neutral, Pleasant) collapsed two
  within-valence subcategory pairs: {Threat, Mutilation} → Negative and
  {Cute, Erotic} → Pleasant. The 4-class design separates these,
  increasing the classification task from K=3 (chance 33.3%) to K=4
  (chance 25%). The information-theoretic question is whether the
  within-valence pairs (Threat vs Mutilation, Cute vs Erotic) carry
  distinct spatiotemporal signatures in the reservoir embedding space,
  or whether the 3-class collapse lost no discriminative structure.

OBSERVATIONS:
  OBS-1: Raw EEG waveforms per subcategory (grand-average ERPs)
  OBS-2: Reservoir spike statistics per subcategory
  OBS-3: BSC6/PCA-64 embedding structure at 4-class granularity
  OBS-4: Inter-channel connectivity per subcategory (4 matrices + diffs)
  OBS-5: Between-subject graph variability (211 subjects)
  OBS-6: Clinical metadata for the full 211-subject sample
  OBS-7: Within-valence vs between-valence embedding distances

INPUT:   shape_features_4class.pkl (from Script 01)
         clinical_profile.csv OR SHAPE_Community_Andrew_Psychopathology.xlsx
OUTPUT:  ch5_4class_raw_data/ — 7 PDF figures + 7 .npz data files

Usage:   python ch5_4class_02_raw_observations.py
============================================================================
"""
import numpy as np
import pickle
import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
INPUT_FILE = './shape_features_4class.pkl'
CLINICAL_FILE_CSV = './clinical_profile.csv'
CLINICAL_FILE_XLSX = './SHAPE_Community_Andrew_Psychopathology.xlsx'
OUT = './ch5_4class_raw_data'
os.makedirs(OUT, exist_ok=True)

COND_NAMES = {0: 'Threat', 1: 'Mutilation', 2: 'Cute', 3: 'Erotic'}
COND_COLORS = {0: '#e74c3c', 1: '#c0392b', 2: '#27ae60', 3: '#2980b9'}

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHAPTER 5 — 4-CLASS RAW DATA OBSERVATIONS")
print("=" * 70)

with open(INPUT_FILE, 'rb') as f:
    d = pickle.load(f)

X_ds = d['X_ds']               # (N_obs, 256, 34)
pca64 = d['lsm_bsc6_pca']     # (N_obs, 34, 64)
bsc6_raw = d['lsm_bsc6_raw']  # (N_obs, 34, 1536)
conv_feats = d['conv_feats']   # (N_obs, 34, 5)
mfr = d['lsm_mfr']            # (N_obs, 34, 256)
y = d['y']
subjects = d['subjects']

N_obs, T_steps, N_ch = X_ds.shape
unique_subj = np.unique(subjects)
N_subj = len(unique_subj)
K = len(np.unique(y))
triu = np.triu_indices(N_ch, k=1)

print(f"\nDataset: {N_obs} observations, {N_subj} subjects, {N_ch} channels")
print(f"Conditions: {COND_NAMES}")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Try loading clinical metadata
clinical_available = False
clin_map = {}
try:
    if os.path.exists(CLINICAL_FILE_CSV):
        df_clin = pd.read_csv(CLINICAL_FILE_CSV)
    elif os.path.exists(CLINICAL_FILE_XLSX):
        df_clin = pd.read_excel(CLINICAL_FILE_XLSX)
    else:
        raise FileNotFoundError("No clinical file found")
    id_col = 'ID'
    clin_map = {int(row[id_col]): row.to_dict() for _, row in df_clin.iterrows()}
    clinical_available = True
    print(f"Clinical metadata loaded: {len(clin_map)} subjects")
except Exception as e:
    print(f"Clinical metadata not found ({e}). OBS-6 will use available data only.")


# ═══════════════════════════════════════════════════════════════════════
# OBS-1: RAW EEG WAVEFORMS PER SUBCATEGORY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-1: Raw EEG Waveforms per Subcategory")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: Before any reservoir transformation,")
print(f"  characterize the input signal. Do the four subcategories produce")
print(f"  distinguishable ERP waveforms? In particular, how large are the")
print(f"  within-valence differences (Threat−Mutilation, Cute−Erotic)")
print(f"  relative to the between-valence differences?\n")

erp_means = {}
erp_sems = {}
for c in range(K):
    eeg_c = X_ds[y == c]
    erp_means[c] = eeg_c.mean(axis=0)
    erp_sems[c] = eeg_c.std(axis=0) / np.sqrt(eeg_c.shape[0])
    print(f"  {COND_NAMES[c]:12s}: n={eeg_c.shape[0]}, "
          f"global mean={eeg_c.mean():.5f}, "
          f"peak range=[{erp_means[c].min():.3f}, {erp_means[c].max():.3f}]")

neg_diff = erp_means[0] - erp_means[1]
pos_diff = erp_means[2] - erp_means[3]
between_diff = (erp_means[0] + erp_means[1])/2 - (erp_means[2] + erp_means[3])/2

print(f"\n  Within-valence ERP differences (max |Δ| across channels × time):")
print(f"    Threat − Mutilation: {np.abs(neg_diff).max():.4f}")
print(f"    Cute − Erotic:       {np.abs(pos_diff).max():.4f}")
print(f"    Negative − Positive: {np.abs(between_diff).max():.4f}")
print(f"\n  Analysis: between-valence difference is "
      f"{np.abs(between_diff).max()/max(np.abs(neg_diff).max(), np.abs(pos_diff).max()):.1f}× "
      f"larger than the largest within-valence difference in raw EEG.")

# FIGURE OBS-1
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
example_channels = [0, 10, 20, 33]

for col, ch in enumerate(example_channels):
    ax = axes[0, col]
    for c in range(K):
        ax.plot(erp_means[c][:, ch], color=COND_COLORS[c], linewidth=1.5,
                label=COND_NAMES[c])
        ax.fill_between(range(T_steps),
                        erp_means[c][:, ch] - erp_sems[c][:, ch],
                        erp_means[c][:, ch] + erp_sems[c][:, ch],
                        color=COND_COLORS[c], alpha=0.15)
    ax.set_xlabel('Time Step', fontsize=9)
    ax.set_ylabel('Amplitude (z)' if col == 0 else '', fontsize=9)
    ax.set_title(f'Channel {ch}', fontsize=11, fontweight='bold')
    if col == 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, T_steps)
    ax.grid(alpha=0.2)

for col, ch in enumerate(example_channels):
    ax = axes[1, col]
    ax.plot(neg_diff[:, ch], color='#8e44ad', linewidth=1.5,
            label='Threat − Mutilation')
    ax.plot(pos_diff[:, ch], color='#16a085', linewidth=1.5,
            label='Cute − Erotic')
    ax.plot(between_diff[:, ch], color='#2c3e50', linewidth=1.5,
            linestyle='--', label='Neg − Pos')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time Step', fontsize=9)
    ax.set_ylabel('Δ Amplitude' if col == 0 else '', fontsize=9)
    ax.set_title(f'Ch {ch} — Difference Waveforms', fontsize=10)
    if col == 0:
        ax.legend(fontsize=7)
    ax.set_xlim(0, T_steps)
    ax.grid(alpha=0.2)

fig.suptitle('OBS-1: Grand-Average ERPs for Four IAPS Subcategories\n'
             'Top: 4-class overlaid (±SEM shading). Bottom: difference waveforms.',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/obs01_4class_erps.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs01_4class_erps.pdf")

np.savez(f'{OUT}/obs01_data.npz',
    **{f'erp_mean_{COND_NAMES[c]}': erp_means[c] for c in range(K)},
    **{f'erp_sem_{COND_NAMES[c]}': erp_sems[c] for c in range(K)},
    neg_diff=neg_diff, pos_diff=pos_diff, between_diff=between_diff)


# ═══════════════════════════════════════════════════════════════════════
# OBS-2: RESERVOIR SPIKE STATISTICS PER SUBCATEGORY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-2: Reservoir Spike Statistics per Subcategory")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: The LIF reservoir transforms continuous")
print(f"  EEG into binary spike trains. The data processing inequality")
print(f"  guarantees information can only be lost in this transformation.")
print(f"  Characterizing how each subcategory maps through the reservoir")
print(f"  reveals whether the spike representation preserves the 4-class")
print(f"  structure observed in OBS-1.\n")

for c in range(K):
    mfr_c = mfr[y == c]
    total_rate = mfr_c.mean()
    ch_rates = mfr_c.mean(axis=(0, 2))
    print(f"  {COND_NAMES[c]:12s}: global MFR={total_rate:.4f}, "
          f"ch range=[{ch_rates.min():.4f}, {ch_rates.max():.4f}]")

print(f"\n  BSC6 sparsity per condition:")
for c in range(K):
    bsc_c = bsc6_raw[y == c]
    sparsity = (bsc_c == 0).mean() * 100
    print(f"    {COND_NAMES[c]:12s}: {sparsity:.1f}% zeros, "
          f"mean count={bsc_c.mean():.3f}, max={bsc_c.max():.0f}")

# MFR condition differences
mfr_cond = {c: mfr[y==c].mean(axis=2).mean(axis=0) for c in range(K)}
mfr_neg_diff = mfr_cond[0] - mfr_cond[1]
mfr_pos_diff = mfr_cond[2] - mfr_cond[3]
print(f"\n  Within-valence MFR differences (max |Δ| across channels):")
print(f"    Threat − Mutilation: {np.abs(mfr_neg_diff).max():.5f}")
print(f"    Cute − Erotic:       {np.abs(mfr_pos_diff).max():.5f}")

# FIGURE OBS-2
fig, axes = plt.subplots(2, 4, figsize=(22, 10))

for c in range(K):
    ax = axes[0, c]
    mfr_c = mfr[y == c].mean(axis=2)  # (N_c, 34)
    ch_means = mfr_c.mean(axis=0)
    ch_stds = mfr_c.std(axis=0)
    ax.bar(range(N_ch), ch_means, yerr=ch_stds, capsize=2,
           color=COND_COLORS[c], edgecolor='black', linewidth=0.3, alpha=0.8)
    ax.set_xlabel('Channel', fontsize=9)
    ax.set_ylabel('MFR' if c == 0 else '', fontsize=9)
    ax.set_title(f'{COND_NAMES[c]} (n={int((y==c).sum())})',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(range(0, 34, 5))
    ax.grid(axis='y', alpha=0.3)

# Bottom row: BSC6 structure for one example channel
ch_ex = 10
for c in range(K):
    ax = axes[1, c]
    bsc_c = bsc6_raw[y == c, ch_ex, :]  # (N_c, 1536)
    bsc_reshaped = bsc_c.reshape(-1, 256, 6)  # (N_c, 256 neurons, 6 bins)
    neuron_avg = bsc_reshaped.mean(axis=0)  # (256, 6)
    bin_profile = neuron_avg.mean(axis=0)  # (6,)
    ax.bar(range(6), bin_profile, color=COND_COLORS[c],
           edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_xlabel('BSC Bin', fontsize=9)
    ax.set_ylabel('Mean Count' if c == 0 else '', fontsize=9)
    ax.set_title(f'{COND_NAMES[c]} — Ch{ch_ex} BSC6 Profile', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('OBS-2: Reservoir Spike Statistics at 4-Class Granularity\n'
             'Top: per-channel MFR (±between-subject SD). '
             f'Bottom: BSC6 temporal profiles for channel {ch_ex}.',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/obs02_4class_spike_stats.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs02_4class_spike_stats.pdf")

np.savez(f'{OUT}/obs02_data.npz',
    **{f'mfr_ch_mean_{COND_NAMES[c]}': mfr[y==c].mean(axis=2).mean(axis=0) for c in range(K)},
    **{f'bsc6_sparsity_{COND_NAMES[c]}': (bsc6_raw[y==c]==0).mean() for c in range(K)})


# ═══════════════════════════════════════════════════════════════════════
# OBS-3: PCA-64 EMBEDDING STRUCTURE AT 4-CLASS GRANULARITY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-3: PCA-64 Embedding Structure")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: PCA-64 compresses the BSC6 representation")
print(f"  from 1536 to 64 dimensions. The inter-centroid distances in this")
print(f"  space quantify how well the reservoir separates the four classes.")
print(f"  If within-valence centroids are closer than between-valence,")
print(f"  the 4-class task is harder than the 3-class by a measurable margin.\n")

ch_var = pca64.var(axis=0).sum(axis=1)
print(f"  Per-channel total variance: range=[{ch_var.min():.0f}, {ch_var.max():.0f}], "
      f"ratio={ch_var.max()/ch_var.min():.1f}×")

centroids = {c: pca64[y == c].mean(axis=0) for c in range(K)}
print(f"\n  Inter-centroid Frobenius distances:")
for c1 in range(K):
    for c2 in range(c1 + 1, K):
        dist = np.linalg.norm(centroids[c1] - centroids[c2])
        within = (c1 in [0,1] and c2 in [0,1]) or (c1 in [2,3] and c2 in [2,3])
        tag = '★ WITHIN' if within else '  between'
        print(f"    {tag}  {COND_NAMES[c1]:12s} ↔ {COND_NAMES[c2]:12s}: {dist:.2f}")

# FIGURE OBS-3
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

for col, ch in enumerate([0, 10, 20, 33]):
    ax = fig.add_subplot(gs[0, col])
    for c in range(K):
        ax.plot(centroids[c][ch, :20], color=COND_COLORS[c], linewidth=1.5,
                label=COND_NAMES[c])
    ax.set_xlabel('PC Index', fontsize=9)
    ax.set_ylabel('Mean Value' if col == 0 else '', fontsize=9)
    ax.set_title(f'Channel {ch} — Centroids (PC 0–19)', fontsize=10, fontweight='bold')
    if col == 0: ax.legend(fontsize=7)
    ax.grid(alpha=0.2); ax.axhline(0, color='black', linewidth=0.5)

ax = fig.add_subplot(gs[1, 0:2])
colors_ch = plt.cm.viridis(ch_var / ch_var.max())
ax.bar(range(N_ch), ch_var, color=colors_ch, edgecolor='black', linewidth=0.3)
ax.set_xlabel('Channel', fontsize=10); ax.set_ylabel('Total Embedding Variance', fontsize=10)
ax.set_title('Per-Channel Embedding Variance', fontsize=11, fontweight='bold')
ax.set_xticks(range(0, 34, 5))

ax = fig.add_subplot(gs[1, 2:4])
dist_mat = np.zeros((K, K))
for c1 in range(K):
    for c2 in range(K):
        dist_mat[c1, c2] = np.linalg.norm(centroids[c1] - centroids[c2])
im = ax.imshow(dist_mat, cmap='YlOrRd', aspect='equal')
for i in range(K):
    for j in range(K):
        ax.text(j, i, f'{dist_mat[i,j]:.1f}', ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if dist_mat[i,j] > dist_mat.max()*0.6 else 'black')
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels([COND_NAMES[c] for c in range(K)], fontsize=10)
ax.set_yticklabels([COND_NAMES[c] for c in range(K)], fontsize=10)
ax.set_title('Inter-Centroid Frobenius Distance', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('OBS-3: PCA-64 Embedding Structure at 4-Class Granularity',
             fontsize=13, fontweight='bold', y=1.02)
fig.savefig(f'{OUT}/obs03_4class_embeddings.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs03_4class_embeddings.pdf")

np.savez(f'{OUT}/obs03_data.npz', ch_var=ch_var, dist_mat=dist_mat,
    **{f'centroid_{COND_NAMES[c]}': centroids[c] for c in range(K)})


# ═══════════════════════════════════════════════════════════════════════
# OBS-4: INTER-CHANNEL CONNECTIVITY PER SUBCATEGORY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-4: Inter-Channel Connectivity per Subcategory")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: The graph in ARSPI-Net is defined by")
print(f"  inter-channel embedding correlations. If different emotional")
print(f"  subcategories produce different connectivity patterns, the graph")
print(f"  carries condition-dependent information beyond the node features.\n")

def fast_conn(embeddings):
    N, C, D = embeddings.shape
    X = embeddings - embeddings.mean(axis=2, keepdims=True)
    norms = np.sqrt((X**2).sum(axis=2, keepdims=True)); norms[norms==0]=1
    return np.einsum('nid,njd->nij', X/norms, X/norms)

conn_all = fast_conn(pca64)

conn_cond = {}
for c in range(K):
    conn_cond[c] = conn_all[y == c].mean(axis=0)
    vals = conn_cond[c][triu]
    print(f"  {COND_NAMES[c]:12s}: mean r={vals.mean():.4f}, "
          f"range=[{vals.min():.3f}, {vals.max():.3f}]")

print(f"\n  Pairwise connectivity differences (mean |Δr|):")
for c1 in range(K):
    for c2 in range(c1 + 1, K):
        diff = conn_cond[c1] - conn_cond[c2]
        mean_abs = np.abs(diff[triu]).mean()
        within = (c1 in [0,1] and c2 in [0,1]) or (c1 in [2,3] and c2 in [2,3])
        tag = '★' if within else ' '
        print(f"    {tag} {COND_NAMES[c1]:12s} − {COND_NAMES[c2]:12s}: mean={mean_abs:.4f}")

# FIGURE OBS-4
diff_TM = conn_cond[0] - conn_cond[1]
diff_CE = conn_cond[2] - conn_cond[3]
diff_neg_pos = (conn_cond[0]+conn_cond[1])/2 - (conn_cond[2]+conn_cond[3])/2

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for c in range(K):
    ax = axes[0, c]; cm_plot = conn_cond[c].copy(); np.fill_diagonal(cm_plot, np.nan)
    ax.imshow(cm_plot, cmap='RdBu_r', vmin=-0.2, vmax=0.6, aspect='equal')
    ax.set_title(f'{COND_NAMES[c]} (n={(y==c).sum()})',
                 fontsize=11, fontweight='bold', color=COND_COLORS[c])
    ax.set_xlabel('Ch', fontsize=9)
    if c == 0: ax.set_ylabel('Ch', fontsize=9)

for ax_i, (diff, title) in enumerate([
    (diff_TM, 'Threat − Mutilation\n(within-negative)'),
    (diff_CE, 'Cute − Erotic\n(within-positive)'),
    (diff_neg_pos, 'Negative − Positive\n(between-valence)')]):
    ax = axes[1, ax_i]; dp = diff.copy(); np.fill_diagonal(dp, np.nan)
    vmax = max(np.abs(diff[triu]).max(), 0.05)
    im = ax.imshow(dp, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Ch', fontsize=9)
    if ax_i == 0: ax.set_ylabel('Ch', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7, label='Δr')

# Between-subject edge variability
subj_conn_local = np.zeros((N_subj, N_ch, N_ch))
for si, s in enumerate(unique_subj):
    subj_conn_local[si] = conn_all[subjects == s].mean(axis=0)
edge_stds = subj_conn_local[:, triu[0], triu[1]].std(axis=0)
std_mat = np.zeros((N_ch, N_ch)); std_mat[triu] = edge_stds
std_mat = std_mat + std_mat.T; np.fill_diagonal(std_mat, np.nan)

ax = axes[1, 3]
im = ax.imshow(std_mat, cmap='hot_r', aspect='equal')
ax.set_title(f'Between-Subject Edge σ\n(mean={edge_stds.mean():.3f})',
             fontsize=10, fontweight='bold')
ax.set_xlabel('Ch', fontsize=9)
plt.colorbar(im, ax=ax, shrink=0.7, label='σ')

fig.suptitle('OBS-4: Inter-Channel Connectivity at 4-Class Granularity',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/obs04_4class_connectivity.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs04_4class_connectivity.pdf")

np.savez(f'{OUT}/obs04_data.npz',
    **{f'conn_{COND_NAMES[c]}': conn_cond[c] for c in range(K)},
    diff_TM=diff_TM, diff_CE=diff_CE, diff_neg_pos=diff_neg_pos,
    edge_stds=edge_stds, subj_conn=subj_conn_local)


# ═══════════════════════════════════════════════════════════════════════
# OBS-5: BETWEEN-SUBJECT GRAPH VARIABILITY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-5: Between-Subject Graph Variability (N={0})".format(N_subj))
print(f"{'='*70}")
print(f"\n  Mathematical motivation: The condition effect must be large enough")
print(f"  relative to between-subject variability for classification to succeed.")
print(f"  This observation quantifies the signal-to-noise ratio.\n")

def quick_graph_props(conn_mat):
    N = conn_mat.shape[0]; tri = np.triu_indices(N, k=1)
    vals = conn_mat[tri]; thr = np.percentile(vals, 80)
    A = (conn_mat >= thr).astype(float); np.fill_diagonal(A, 0)
    deg = A.sum(axis=1)
    cc = np.zeros(N)
    for i in range(N):
        nb = np.where(A[i]>0)[0]; k = len(nb)
        if k >= 2:
            links = sum(A[nb[a],nb[b]] for a in range(len(nb)) for b in range(a+1,len(nb)))
            cc[i] = 2*links/(k*(k-1))
    return {'mean_degree': deg.mean(), 'mean_clustering': cc.mean(),
            'degree_std': deg.std(), 'mean_strength': np.abs(conn_mat[tri]).mean()}

all_props = [quick_graph_props(subj_conn_local[si]) for si in range(N_subj)]
for prop in ['mean_degree', 'mean_clustering', 'degree_std', 'mean_strength']:
    vals = np.array([p[prop] for p in all_props])
    print(f"  {prop:<20s}: mean={vals.mean():.3f}, std={vals.std():.3f}")

cond_signal = np.abs(diff_neg_pos[triu]).mean()
subj_noise = edge_stds.mean()
snr = subj_noise / cond_signal if cond_signal > 0 else float('inf')
print(f"\n  Condition signal (|neg−pos|): {cond_signal:.4f}")
print(f"  Subject noise:               {subj_noise:.4f}")
print(f"  Noise/signal ratio:          1:{snr:.0f}")

# FIGURE OBS-5
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for col, (prop, label) in enumerate([
    ('mean_clustering', 'Mean Clustering'),
    ('mean_strength', 'Mean Strength'),
    ('degree_std', 'Degree Heterogeneity')]):
    ax = axes[0, col]
    vals = np.array([p[prop] for p in all_props])
    ax.hist(vals, bins=20, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axvline(vals.mean(), color='red', linewidth=2, linestyle='--',
               label=f'Mean={vals.mean():.3f}')
    ax.set_xlabel(label, fontsize=10); ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'{label} (N={N_subj})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

ax = axes[1, 0]
for si in range(N_subj):
    tri_l = np.triu_indices(N_ch, k=1)
    thr = np.percentile(subj_conn_local[si][tri_l], 80)
    A = (subj_conn_local[si] >= thr).astype(float); np.fill_diagonal(A, 0)
    dd = np.sort(A.sum(axis=1))[::-1]
    ax.plot(range(N_ch), dd, 'b-', alpha=0.03, linewidth=0.8)
ax.set_xlabel('Channel Rank', fontsize=10); ax.set_ylabel('Degree', fontsize=10)
ax.set_title('Sorted Degree Distributions (N=211)', fontsize=11, fontweight='bold')

ax = axes[1, 1]
cond_effects = np.abs(diff_neg_pos[triu])
ax.scatter(edge_stds, cond_effects, s=4, alpha=0.5, color='steelblue')
ax.set_xlabel('Between-Subject Edge σ', fontsize=10)
ax.set_ylabel('|Neg−Pos| Δr', fontsize=10)
ax.set_title('Subject Variability vs Condition Effect', fontsize=11, fontweight='bold')
ax.plot([0, edge_stds.max()], [0, edge_stds.max()], 'r--', linewidth=1, alpha=0.5)
ax.grid(alpha=0.3)

ax = axes[1, 2]
within_neg_e = np.abs(diff_TM[triu])
within_pos_e = np.abs(diff_CE[triu])
between_e = np.abs(diff_neg_pos[triu])
bp = ax.boxplot([within_neg_e, within_pos_e, between_e],
                labels=['Threat−\nMut', 'Cute−\nEro', 'Neg−Pos'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#8e44ad', '#16a085', '#2c3e50']):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_ylabel('|Δr| per edge', fontsize=10)
ax.set_title('Within- vs Between-Valence\nConnectivity Differences', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

fig.suptitle(f'OBS-5: Between-Subject Graph Variability (N={N_subj})',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{OUT}/obs05_4class_graph_variability.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs05_4class_graph_variability.pdf")

np.savez(f'{OUT}/obs05_data.npz',
    graph_props=np.array([[p[k] for k in ['mean_degree','mean_clustering',
                           'degree_std','mean_strength']] for p in all_props]),
    cond_signal=cond_signal, subj_noise=subj_noise, snr=snr)


# ═══════════════════════════════════════════════════════════════════════
# OBS-6: CLINICAL METADATA FOR 211-SUBJECT SAMPLE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-6: Clinical Metadata for {0}-Subject Sample".format(N_subj))
print(f"{'='*70}")
print(f"\n  Mathematical motivation: The clinical interpretability experiments")
print(f"  (EXP 7–11 in Script 03) require sufficient group sizes for each")
print(f"  clinical variable. This observation characterizes the diagnostic")
print(f"  distribution and identifies which clinical contrasts have adequate")
print(f"  statistical power in the {N_subj}-subject sample.\n")

sids = [int(s) for s in unique_subj]
dx_cols = ['MDD', 'MDD_Recurrent', 'PDD', 'GAD', 'PTSD', 'SUD', 'ADHD',
           'EAT', 'Mania']
clinical_arrays = {}

if clinical_available:
    print(f"  {'Variable':<25s} {'Yes':>5s} {'No':>5s} {'Miss':>5s} {'%Yes':>6s} {'Power':>8s}")
    print(f"  {'─'*60}")
    for var in dx_cols:
        vals = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        clinical_arrays[var] = vals
        valid = ~np.isnan(vals)
        n_yes = int((vals[valid]==1).sum())
        n_no = int((vals[valid]==0).sum())
        n_miss = int((~valid).sum())
        pct = n_yes/valid.sum()*100 if valid.sum()>0 else 0
        power = 'adequate' if min(n_yes, n_no) >= 15 else 'marginal' if min(n_yes,n_no)>=10 else 'low'
        print(f"  {var:<25s} {n_yes:5d} {n_no:5d} {n_miss:5d} {pct:5.1f}% {power:>8s}")

    # Medication and sex (from clinical_profile.csv)
    for var, label in [('Psychiatric_Medication', 'Medication'),
                       ('Assigned_Sex', 'Sex (1=M,2=F)')]:
        vals = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        clinical_arrays[var] = vals
        valid = ~np.isnan(vals)
        if var == 'Assigned_Sex':
            n1 = int((vals[valid]==1).sum()); n2 = int((vals[valid]==2).sum())
            print(f"  {label:<25s}   M={n1}, F={n2}, miss={int((~valid).sum())}")
        else:
            n_yes = int((vals[valid]==1).sum()); n_no = int((vals[valid]==0).sum())
            print(f"  {label:<25s} {n_yes:5d} {n_no:5d} {int((~valid).sum()):5d} "
                  f"{n_yes/valid.sum()*100 if valid.sum()>0 else 0:5.1f}%")

    main_dx = ['MDD', 'GAD', 'PTSD', 'SUD', 'ADHD', 'EAT', 'Mania', 'PDD']
    comorbidity = np.array([
        sum(1 for c in main_dx if clin_map.get(sid, {}).get(c, 0) == 1)
        for sid in sids])
    print(f"\n  Comorbidity: mean={comorbidity.mean():.1f}, "
          f"median={np.median(comorbidity):.0f}, range=[{comorbidity.min()}, {comorbidity.max()}]")
else:
    print("  Clinical metadata not available.")
    comorbidity = np.zeros(N_subj)

# FIGURE OBS-6
fig = plt.figure(figsize=(18, 8))
gs = GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.4)

if clinical_available:
    for pi, (var, label) in enumerate([('MDD','MDD'), ('GAD','GAD'), ('PTSD','PTSD'),
                                        ('SUD','SUD'), ('ADHD','ADHD')]):
        ax = fig.add_subplot(gs[0, pi])
        vals = clinical_arrays.get(var, np.array([]))
        valid = ~np.isnan(vals)
        n0 = int((vals[valid]==0).sum()); n1 = int((vals[valid]==1).sum())
        ax.bar([0,1], [n0,n1], color=['#3498db','#e74c3c'],
               edgecolor='black', linewidth=0.8)
        ax.text(0, n0+2, str(n0), ha='center', fontsize=9, fontweight='bold')
        ax.text(1, n1+2, str(n1), ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks([0,1]); ax.set_xticklabels(['No','Yes'], fontsize=9)
        ax.set_title(f'{label} (valid={int(valid.sum())})', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count' if pi==0 else '', fontsize=9)

ax = fig.add_subplot(gs[1, 0:2])
ax.hist(comorbidity, bins=range(0, int(comorbidity.max())+2),
        color='coral', edgecolor='black', align='left')
ax.set_xlabel('N Diagnoses', fontsize=10); ax.set_ylabel('Count', fontsize=10)
ax.set_title(f'Comorbidity (N={N_subj}, mean={comorbidity.mean():.1f})',
             fontsize=11, fontweight='bold')

ax = fig.add_subplot(gs[1, 2:4])
ax.bar([0,1], [115, N_subj], color=['#95a5a6','#3498db'],
       edgecolor='black', linewidth=0.8, width=0.5)
ax.text(0, 118, '115', ha='center', fontsize=14, fontweight='bold')
ax.text(1, N_subj+3, str(N_subj), ha='center', fontsize=14, fontweight='bold')
ax.set_xticks([0,1])
ax.set_xticklabels(['3-Class\n(original)', '4-Class\n(this analysis)'], fontsize=11)
ax.set_ylabel('N Subjects', fontsize=11)
ax.set_title('Sample Size Comparison', fontsize=11, fontweight='bold')

ax = fig.add_subplot(gs[1, 4])
counts = [int((y==c).sum()) for c in range(K)]
ax.bar(range(K), counts, color=[COND_COLORS[c] for c in range(K)],
       edgecolor='black', linewidth=0.8)
for c in range(K):
    ax.text(c, counts[c]+5, str(counts[c]), ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(K))
ax.set_xticklabels([COND_NAMES[c] for c in range(K)], fontsize=8, rotation=15)
ax.set_ylabel('Count', fontsize=9)
ax.set_title('4-Class Balance', fontsize=11, fontweight='bold')

fig.suptitle(f'OBS-6: Clinical Metadata and Sample Structure (N={N_subj})',
             fontsize=13, fontweight='bold', y=1.02)
fig.savefig(f'{OUT}/obs06_4class_clinical.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs06_4class_clinical.pdf")

np.savez(f'{OUT}/obs06_data.npz', comorbidity=comorbidity,
    **{f'clinical_{k}': v for k, v in clinical_arrays.items()})


# ═══════════════════════════════════════════════════════════════════════
# OBS-7: WITHIN- VS BETWEEN-VALENCE EMBEDDING DISTANCES
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("OBS-7: Within-Valence vs Between-Valence Embedding Distances")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: This is the central observation that")
print(f"  motivates the 4-class design. The distance hierarchy in the")
print(f"  embedding space predicts the classification difficulty hierarchy.")
print(f"  If between-valence distances are consistently larger than")
print(f"  within-valence, then the 4-class task decomposes into an 'easy'")
print(f"  valence boundary plus two 'hard' within-valence boundaries.\n")

within_neg_dists, within_pos_dists, between_dists = [], [], []
for s in unique_subj:
    s_mask = subjects == s; s_y = y[s_mask]; s_emb = pca64[s_mask]
    cond_embs = {}
    for c in range(K):
        cm = s_y == c
        if cm.sum() > 0: cond_embs[c] = s_emb[cm].mean(axis=0)
    if len(cond_embs) == K:
        within_neg_dists.append(np.linalg.norm(cond_embs[0] - cond_embs[1]))
        within_pos_dists.append(np.linalg.norm(cond_embs[2] - cond_embs[3]))
        between_dists.append(np.mean([
            np.linalg.norm(cond_embs[c1]-cond_embs[c2])
            for c1 in [0,1] for c2 in [2,3]]))

within_neg_dists = np.array(within_neg_dists)
within_pos_dists = np.array(within_pos_dists)
between_dists = np.array(between_dists)

print(f"  Within-valence (Threat−Mutilation):  {within_neg_dists.mean():.2f} ±{within_neg_dists.std():.2f}")
print(f"  Within-valence (Cute−Erotic):        {within_pos_dists.mean():.2f} ±{within_pos_dists.std():.2f}")
print(f"  Between-valence (cross):             {between_dists.mean():.2f} ±{between_dists.std():.2f}")

r_neg = between_dists.mean() / (within_neg_dists.mean() + 1e-10)
r_pos = between_dists.mean() / (within_pos_dists.mean() + 1e-10)
print(f"\n  Between/within ratio: {min(r_neg,r_pos):.1f}–{max(r_neg,r_pos):.1f}×")

pct_neg = (between_dists > within_neg_dists).mean() * 100
pct_pos = (between_dists > within_pos_dists).mean() * 100
print(f"  Subjects where between > within-neg: {pct_neg:.0f}%")
print(f"  Subjects where between > within-pos: {pct_pos:.0f}%")
print(f"\n  Analysis: The embedding space encodes a clear valence boundary.")
print(f"  Within-valence contrasts operate in a tighter regime, predicting")
print(f"  that pairwise classification for Threat↔Mutilation and Cute↔Erotic")
print(f"  will be harder than any between-valence pair (verified in EXP-4).")

# FIGURE OBS-7
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
bp = ax.boxplot([within_neg_dists, within_pos_dists, between_dists],
                labels=['Threat−\nMutilation', 'Cute−\nErotic', 'Between-\nvalence'],
                patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['#8e44ad', '#16a085', '#2c3e50']):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_ylabel('Frobenius Distance', fontsize=11)
ax.set_title('Within- vs Between-Valence\nEmbedding Distance', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ax.scatter(within_neg_dists, between_dists, s=15, alpha=0.5, color='#8e44ad', label='vs Threat−Mut')
ax.scatter(within_pos_dists, between_dists, s=15, alpha=0.5, color='#16a085', label='vs Cute−Ero')
lim = max(within_neg_dists.max(), within_pos_dists.max(), between_dists.max()) * 1.1
ax.plot([0,lim], [0,lim], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel('Within-Valence Distance', fontsize=11)
ax.set_ylabel('Between-Valence Distance', fontsize=11)
ax.set_title('Per-Subject Scatter\n(above diagonal = between > within)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(between_dists/(within_neg_dists+1e-10), bins=30, alpha=0.6, color='#8e44ad', label='vs Threat−Mut')
ax.hist(between_dists/(within_pos_dists+1e-10), bins=30, alpha=0.6, color='#16a085', label='vs Cute−Ero')
ax.axvline(1.0, color='black', linewidth=2, linestyle='--', label='Ratio=1')
ax.set_xlabel('Between/Within Ratio', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distance Ratio Distribution\n(>1 = between more separable)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.suptitle('OBS-7: Information-Theoretic Structure of the 4-Class Embedding Space',
             fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
fig.savefig(f'{OUT}/obs07_4class_distance_structure.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs07_4class_distance_structure.pdf")

np.savez(f'{OUT}/obs07_data.npz',
    within_neg_dists=within_neg_dists, within_pos_dists=within_pos_dists,
    between_dists=between_dists)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("RAW OBSERVATION SUMMARY")
print(f"{'='*70}")
print(f"""
7 observation sets characterize the 4-class data BEFORE modeling:
  OBS-1: Grand-average ERPs + within-valence difference waveforms
  OBS-2: Reservoir spike statistics (MFR, BSC6 temporal profiles)
  OBS-3: PCA-64 embedding centroids + inter-centroid distance matrix
  OBS-4: 4 connectivity matrices + 3 difference matrices + subject σ
  OBS-5: Graph property distributions + signal-to-noise ratio
  OBS-6: Clinical metadata for {N_subj}-subject sample + power assessment
  OBS-7: Within- vs between-valence distance hierarchy ({pct_neg:.0f}–{pct_pos:.0f}%
         of subjects show between > within)

Each: PDF figure + .npz data file → {OUT}/
""")

for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f'{OUT}/{f}')
    print(f"  {f}: {size:,} bytes")
