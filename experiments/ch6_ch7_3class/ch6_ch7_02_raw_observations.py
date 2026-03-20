#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
ARSPI-Net — Script 02: Raw Observations (Chapters 6 & 7)
═══════════════════════════════════════════════════════════════════════

PURPOSE
-------
Characterize the dynamical and topological features BEFORE any clinical
or statistical analysis. Following the Scientific Voice Directive:
observe before you analyze.

Eight observation sets characterize the 3-class data:

  OBS-1: Dynamical metric distributions across all observations
         Mathematical motivation: What ranges do the 7 core metrics span?
         Are there degenerate channels or metrics? This is the prerequisite
         for every downstream analysis — if a metric has zero variance, it
         carries no information.

  OBS-2: Condition-dependent dynamical profiles
         Mathematical motivation: Do the reservoir's internal dynamics
         differ between Negative, Neutral, and Pleasant stimuli? If the
         dynamical metrics are condition-blind, Chapter 6's clinical
         analyses cannot succeed. The observable: per-condition means and
         distributions for each metric.

  OBS-3: Per-channel metric heterogeneity
         Mathematical motivation: The 34 channels receive different EEG
         signals. Do they produce different reservoir dynamics? Channel
         heterogeneity is the spatial structure that Chapter 7's coupling
         analysis exploits. If all channels produce identical dynamics,
         coupling is trivially zero.

  OBS-4: Population firing rate timecourses
         Mathematical motivation: The temporal profile of population
         activity reveals the reservoir's transient and steady-state
         response to the driving EEG. Relaxation time (τ_relax) and
         return-to-baseline (T_RTB) are extracted from these timecourses.
         This observation verifies that the timecourses have interpretable
         temporal structure.

  OBS-5: tPLV connectivity structure
         Mathematical motivation: The theta-band phase-locking matrices
         define the spatial graph for Chapter 7. This observation
         characterizes the connectivity structure: mean PLV, edge density,
         and whether condition-dependent differences exist in the
         phase-locking pattern.

  OBS-6: Topological metric distributions (strength, clustering)
         Mathematical motivation: Weighted node strength and clustering
         are the spatial descriptors in Chapter 7's coupling analysis.
         Their distributions and per-channel profiles determine the
         statistical power of the coupling tests.

  OBS-7: Between-subject variability in dynamical metrics
         Mathematical motivation: Chapter 5 found that subject identity
         dominates the BSC6 embedding space (62.6% of variance). Before
         any formal variance decomposition (EXP-6.1), this observation
         displays the raw subject × condition structure: heatmaps of
         individual subjects, scatter plots of between versus within
         variability, and individual trajectories across conditions.
         The question: visually, does subject identity or emotional
         condition dominate?

  OBS-8: Clinical metadata coverage
         Mathematical motivation: The clinical experiments (Script 03)
         require sufficient group sizes. This observation characterizes
         the diagnostic distribution and statistical power in the
         211-subject sample.

INPUT
-----
  ch6_ch7_3class_features.pkl (from Script 01)
  clinical_profile.csv (optional, for OBS-8)

OUTPUT
------
  8 PDF figures in ./ch6_ch7_obs/
  8 .npz data files in ./ch6_ch7_obs/
  Terminal output with quantitative statistics for each observation

SCIENTIFIC QUESTION
-------------------
What do the dynamical and topological features look like before any
statistical analysis? The five-step cycle requires observation before
analysis. Each observation set answers a specific structural question
about the data that constrains and informs the downstream experiments.

RUN COMMAND
-----------
  python ch6_ch7_02_raw_observations.py
"""

import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
INPUT_FILE = './ch6_ch7_3class_features.pkl'
CLINICAL_FILE = './clinical_profile.csv'
OUT = './ch6_ch7_obs'
os.makedirs(OUT, exist_ok=True)

METRIC_NAMES = ['total_spikes', 'MFR', 'rate_entropy', 'rate_variance',
                'temporal_sparsity', 'perm_entropy', 'tau_AC']
EXTRA_NAMES = ['CLZ', 'lambda_proxy', 'tau_relax', 'T_RTB']
TOPO_NAMES = ['weighted_strength', 'weighted_clustering']
COND_COLORS = {'Negative': '#d63031', 'Neutral': '#636e72', 'Pleasant': '#00b894'}

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ARSPI-NET — 3-CLASS RAW DATA OBSERVATIONS (CHAPTERS 6 & 7)")
print("=" * 70)

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

D = data['D']                   # (633, 34, 7)
D_extra = data['D_extra']       # (633, 34, 4)
T_topo = data['T_topo']         # (633, 34, 2)
tPLV_mats = data['tPLV_mats']  # (633, 34, 34)
pop_rate_ts = data['pop_rate_ts']  # (633, 34, 256)
y = data['y']                   # (633,)
subjects = data['subjects']     # (633,)
COND_NAMES = data['cond_names']

N_obs, N_ch, N_metrics = D.shape
N_subj = len(np.unique(subjects))

print(f"\nDataset: {N_obs} observations, {N_subj} subjects, {N_ch} channels")
print(f"Conditions: {COND_NAMES}")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Load clinical metadata
try:
    import pandas as pd
    df_clin = pd.read_csv(CLINICAL_FILE)
    df_clin = df_clin.drop_duplicates(subset='ID', keep='first')
    clin_subjects = set(df_clin['ID'].values)
    n_clin = len(clin_subjects & set(np.unique(subjects)))
    print(f"Clinical metadata loaded: {n_clin} subjects")
    HAS_CLINICAL = True
except Exception as e:
    print(f"Clinical metadata not found ({e}). OBS-8 will use available data only.")
    HAS_CLINICAL = False


# ══════════════════════════════════════════════════════════════════
# OBS-1: Dynamical Metric Distributions
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-1: Dynamical Metric Distributions Across All Observations")
print("=" * 70)

print(f"\n  Mathematical motivation: Before any clinical comparison, verify")
print(f"  that each dynamical metric spans a meaningful range with non-zero")
print(f"  variance. Degenerate metrics (zero variance) carry no information")
print(f"  and must be excluded from downstream analysis.\n")

for k, name in enumerate(METRIC_NAMES):
    vals = D[:, :, k].flatten()
    print(f"  {name:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"range=[{vals.min():.4f}, {vals.max():.4f}], "
          f"CV={vals.std()/vals.mean():.3f}" if vals.mean() != 0 else
          f"  {name:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"range=[{vals.min():.4f}, {vals.max():.4f}]")

print(f"\n  Extra metrics:")
for k, name in enumerate(EXTRA_NAMES):
    vals = D_extra[:, :, k].flatten()
    print(f"  {name:20s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"range=[{vals.min():.4f}, {vals.max():.4f}]")

# Figure: 7 histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for k in range(7):
    ax = axes[k]
    vals = D[:, :, k].flatten()
    ax.hist(vals, bins=50, color='#2d3436', alpha=0.7, edgecolor='white')
    ax.set_title(METRIC_NAMES[k], fontsize=11, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.axvline(vals.mean(), color='red', linestyle='--', alpha=0.7, label=f'μ={vals.mean():.3f}')
    ax.legend(fontsize=8)
axes[7].axis('off')
fig.suptitle('OBS-1: Dynamical Metric Distributions (N=633 obs × 34 ch)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs01_metric_distributions.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs01_metric_distributions.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-2: Condition-Dependent Dynamical Profiles
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-2: Condition-Dependent Dynamical Profiles")
print("=" * 70)

print(f"\n  Mathematical motivation: If the reservoir's internal dynamics do")
print(f"  not differ between emotional conditions, the dynamical metrics")
print(f"  carry no affective information. The observable: per-condition")
print(f"  means ± SEM for each metric, channel-averaged.\n")

for k, name in enumerate(METRIC_NAMES):
    print(f"  {name:20s}:", end='')
    for c in range(3):
        vals = D[y == c, :, k].mean(axis=1)  # average across channels
        print(f"  {COND_NAMES[c]}={vals.mean():.4f}±{vals.std()/np.sqrt(len(vals)):.4f}", end='')
    # Within-subject paired difference (Neg - Pos)
    neg_means = []
    pos_means = []
    for sid in np.unique(subjects):
        neg_mask = (subjects == sid) & (y == 0)
        pos_mask = (subjects == sid) & (y == 2)
        if neg_mask.sum() > 0 and pos_mask.sum() > 0:
            neg_means.append(D[neg_mask, :, k].mean())
            pos_means.append(D[pos_mask, :, k].mean())
    neg_arr = np.array(neg_means)
    pos_arr = np.array(pos_means)
    diff = neg_arr - pos_arr
    print(f"  Δ(Neg-Pos)={diff.mean():.4f}")

# Figure: 7 violin plots by condition
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for k in range(7):
    ax = axes[k]
    data_by_cond = []
    for c in range(3):
        vals = D[y == c, :, k].mean(axis=1)  # channel-averaged per observation
        data_by_cond.append(vals)
    parts = ax.violinplot(data_by_cond, positions=[0, 1, 2], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        color = list(COND_COLORS.values())[i]
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Neg', 'Neu', 'Pos'])
    ax.set_title(METRIC_NAMES[k], fontsize=11, fontweight='bold')
axes[7].axis('off')
fig.suptitle('OBS-2: Condition-Dependent Dynamical Profiles (channel-averaged)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs02_condition_profiles.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs02_condition_profiles.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-3: Per-Channel Metric Heterogeneity
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-3: Per-Channel Dynamical Metric Heterogeneity")
print("=" * 70)

print(f"\n  Mathematical motivation: Chapter 7 correlates dynamical metrics")
print(f"  with topological metrics ACROSS electrodes. This requires that")
print(f"  different channels produce different dynamical profiles. If all")
print(f"  channels are identical, the within-observation correlation is")
print(f"  undefined.\n")

# Channel-mean for each metric
for k, name in enumerate(METRIC_NAMES):
    ch_means = D[:, :, k].mean(axis=0)  # (34,) mean across all obs
    ch_std = D[:, :, k].std(axis=0)
    cv = ch_std / (ch_means + 1e-12)
    print(f"  {name:20s}: ch range=[{ch_means.min():.4f}, {ch_means.max():.4f}], "
          f"ratio={ch_means.max()/(ch_means.min()+1e-12):.2f}×, "
          f"mean CV={cv.mean():.3f}")

# Figure: heatmap of channel × metric (grand-mean)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ch_metric_matrix = np.zeros((N_ch, 7))
for k in range(7):
    vals = D[:, :, k].mean(axis=0)
    # Normalize to [0,1] for visualization
    if vals.max() > vals.min():
        ch_metric_matrix[:, k] = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        ch_metric_matrix[:, k] = 0.5

im = ax.imshow(ch_metric_matrix, aspect='auto', cmap='viridis')
ax.set_xlabel('Metric')
ax.set_ylabel('Channel')
ax.set_xticks(range(7))
ax.set_xticklabels([n[:10] for n in METRIC_NAMES], rotation=45, ha='right')
ax.set_yticks(range(0, 34, 2))
plt.colorbar(im, ax=ax, label='Normalized value')
ax.set_title('OBS-3: Channel × Metric Grand-Mean (normalized)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs03_channel_heterogeneity.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs03_channel_heterogeneity.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-4: Population Firing Rate Timecourses
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-4: Population Firing Rate Timecourses")
print("=" * 70)

print(f"\n  Mathematical motivation: The temporal profile of population")
print(f"  activity is the observable from which τ_relax and T_RTB are")
print(f"  extracted. This observation verifies interpretable temporal")
print(f"  structure: onset transient, steady state, and decay.\n")

# Grand-average timecourses by condition (averaged across subjects + channels)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for c in range(3):
    ax = axes[c]
    mask = y == c
    # Average across observations and channels
    grand_avg = pop_rate_ts[mask].mean(axis=(0, 1))  # (256,)
    grand_sem = pop_rate_ts[mask].mean(axis=1).std(axis=0) / np.sqrt(mask.sum())  # SEM across obs
    t = np.arange(len(grand_avg))
    color = list(COND_COLORS.values())[c]
    ax.plot(t, grand_avg, color=color, linewidth=1.5)
    ax.fill_between(t, grand_avg - grand_sem, grand_avg + grand_sem, color=color, alpha=0.2)
    ax.set_xlabel('Timestep')
    ax.set_title(COND_NAMES[c], fontsize=12, fontweight='bold', color=color)
    # Mark peak
    t_peak = np.argmax(grand_avg)
    ax.axvline(t_peak, color='gray', linestyle='--', alpha=0.5, label=f'peak t={t_peak}')
    ax.legend(fontsize=8)
    
    print(f"  {COND_NAMES[c]:10s}: peak rate={grand_avg.max():.4f} at t={t_peak}, "
          f"final rate={grand_avg[-10:].mean():.4f}, "
          f"peak/final ratio={grand_avg.max()/(grand_avg[-10:].mean()+1e-8):.2f}")

axes[0].set_ylabel('Population firing rate')
fig.suptitle('OBS-4: Grand-Average Population Firing Rate by Condition', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs04_firing_rate_timecourses.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs04_firing_rate_timecourses.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-5: tPLV Connectivity Structure
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-5: Theta-Band Phase-Locking Connectivity Structure")
print("=" * 70)

print(f"\n  Mathematical motivation: The tPLV matrices define the spatial")
print(f"  graph for Chapter 7. Characterize the connectivity structure:")
print(f"  mean PLV, condition differences, and edge density.\n")

for c in range(3):
    mask = y == c
    plvs = tPLV_mats[mask]
    # Off-diagonal mean
    offdiag = plvs[:, np.triu_indices(N_ch, k=1)[0], np.triu_indices(N_ch, k=1)[1]]
    print(f"  {COND_NAMES[c]:10s}: mean PLV={offdiag.mean():.4f}, "
          f"std={offdiag.std():.4f}, "
          f"range=[{offdiag.min():.4f}, {offdiag.max():.4f}]")

# Condition differences
for c1, c2 in [(0, 1), (0, 2), (1, 2)]:
    m1 = tPLV_mats[y == c1].mean(axis=0)
    m2 = tPLV_mats[y == c2].mean(axis=0)
    diff = np.abs(m1 - m2)
    idx = np.triu_indices(N_ch, k=1)
    print(f"  |Δ PLV| {COND_NAMES[c1][:3]}−{COND_NAMES[c2][:3]}: "
          f"mean={diff[idx].mean():.4f}, max={diff[idx].max():.4f}")

# Figure: 3 condition-averaged tPLV matrices + difference
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for c in range(3):
    ax = axes[c]
    mean_plv = tPLV_mats[y == c].mean(axis=0)
    im = ax.imshow(mean_plv, vmin=0, vmax=1, cmap='hot')
    ax.set_title(f'{COND_NAMES[c]}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Channel')
# Difference: Neg - Pos
ax = axes[3]
diff = tPLV_mats[y == 0].mean(axis=0) - tPLV_mats[y == 2].mean(axis=0)
vmax = np.abs(diff[np.triu_indices(N_ch, k=1)]).max()
im2 = ax.imshow(diff, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
ax.set_title('Neg − Pos', fontsize=11, fontweight='bold')
plt.colorbar(im2, ax=ax, label='ΔPLV')
fig.suptitle('OBS-5: Theta-Band tPLV Matrices by Condition', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs05_tplv_connectivity.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs05_tplv_connectivity.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-6: Topological Metric Distributions
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-6: Topological Metric Distributions (Strength, Clustering)")
print("=" * 70)

print(f"\n  Mathematical motivation: Weighted node strength and clustering")
print(f"  are the spatial descriptors in Chapter 7's coupling matrix.")
print(f"  Their distributions determine statistical power.\n")

for k, name in enumerate(TOPO_NAMES):
    vals = T_topo[:, :, k].flatten()
    print(f"  {name:25s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
          f"range=[{vals.min():.4f}, {vals.max():.4f}]")
    for c in range(3):
        cv = T_topo[y == c, :, k].flatten()
        print(f"    {COND_NAMES[c]:12s}: mean={cv.mean():.4f}")

# Figure: topological distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for k in range(2):
    ax = axes[k]
    for c in range(3):
        vals = T_topo[y == c, :, k].mean(axis=1)
        color = list(COND_COLORS.values())[c]
        ax.hist(vals, bins=30, alpha=0.5, color=color, label=COND_NAMES[c], edgecolor='white')
    ax.set_title(TOPO_NAMES[k], fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend()
fig.suptitle('OBS-6: Topological Metric Distributions by Condition', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs06_topological_distributions.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs06_topological_distributions.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-7: Between-Subject Variability in Dynamical Metrics
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-7: Between-Subject Variability in Dynamical Metrics")
print("=" * 70)

print(f"\n  Mathematical motivation: Chapter 5 found that subject identity")
print(f"  dominates the BSC6 embedding space (62.6% of variance). Before")
print(f"  any formal variance decomposition (EXP-6.1 in Script 03), this")
print(f"  observation displays the raw subject × condition structure of")
print(f"  each dynamical metric. The question: visually, does subject")
print(f"  identity or emotional condition dominate the variability?\n")

unique_subs = np.unique(subjects)

# For two representative metrics (one amplitude, one temporal-structure),
# show the subject × condition matrix and individual trajectories
rep_metrics = [(1, 'MFR'), (6, 'tau_AC')]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for row, (k, name) in enumerate(rep_metrics):
    # Build subject × condition matrix
    scm = np.zeros((N_subj, 3))
    for si, sid in enumerate(unique_subs):
        for c in range(3):
            mask = (subjects == sid) & (y == c)
            if mask.sum() > 0:
                scm[si, c] = D[:, :, k].mean(axis=1)[mask].mean()

    # Panel 1: Subject × condition heatmap (first 50 subjects)
    ax = axes[row, 0]
    n_show = min(50, N_subj)
    im = ax.imshow(scm[:n_show, :], aspect='auto', cmap='viridis')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Subject')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Neg', 'Neu', 'Pos'])
    ax.set_title(f'{name}: Subject × Condition', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Panel 2: Subject means vs within-subject std
    ax = axes[row, 1]
    sub_means = scm.mean(axis=1)
    sub_stds = scm.std(axis=1)
    ax.scatter(sub_means, sub_stds, alpha=0.4, s=15, color='#2d3436')
    ax.set_xlabel(f'Subject mean {name}')
    ax.set_ylabel(f'Within-subject std')
    ax.set_title(f'Between vs within variability', fontsize=11, fontweight='bold')
    # Report the ratio
    between_std = sub_means.std()
    within_std_mean = sub_stds.mean()
    ax.text(0.05, 0.95, f'Between σ = {between_std:.4f}\nWithin σ = {within_std_mean:.4f}\n'
            f'Ratio = {between_std/(within_std_mean+1e-8):.2f}×',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    print(f"  {name:20s}: between-subject σ = {between_std:.4f}, "
          f"mean within-subject σ = {within_std_mean:.4f}, "
          f"ratio = {between_std/(within_std_mean+1e-8):.2f}×")

    # Panel 3: Individual trajectories (20 random subjects)
    ax = axes[row, 2]
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(N_subj, size=min(20, N_subj), replace=False)
    for si in sample_idx:
        ax.plot([0, 1, 2], scm[si, :], color='gray', alpha=0.3, linewidth=0.8)
    # Condition means
    cond_means = scm.mean(axis=0)
    ax.plot([0, 1, 2], cond_means, color='red', linewidth=2.5,
            marker='o', markersize=8, label='Condition mean', zorder=10)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Neg', 'Neu', 'Pos'])
    ax.set_ylabel(name)
    ax.set_title(f'Individual trajectories (gray) vs condition mean (red)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

# Additional metrics: print summary for all 7
print(f"\n  All 7 metrics (between-subject σ / within-subject σ):")
for k, name in enumerate(METRIC_NAMES):
    scm = np.zeros((N_subj, 3))
    for si, sid in enumerate(unique_subs):
        for c in range(3):
            mask = (subjects == sid) & (y == c)
            if mask.sum() > 0:
                scm[si, c] = D[:, :, k].mean(axis=1)[mask].mean()
    between = scm.mean(axis=1).std()
    within = scm.std(axis=1).mean()
    print(f"    {name:20s}: between={between:.4f}, within={within:.4f}, "
          f"ratio={between/(within+1e-8):.2f}×")

fig.suptitle('OBS-7: Between-Subject Variability (raw visualization)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/obs07_subject_variability.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → obs07_subject_variability.pdf")


# ══════════════════════════════════════════════════════════════════
# OBS-8: Clinical Metadata Coverage
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OBS-8: Clinical Metadata Coverage for 211-Subject Sample")
print("=" * 70)

print(f"\n  Mathematical motivation: The transdiagnostic clinical analyses")
print(f"  (Script 03) require sufficient group sizes for each diagnostic")
print(f"  variable. This observation characterizes the diagnostic distribution.\n")

if HAS_CLINICAL:
    eeg_subs = np.unique(subjects)
    df_eeg = df_clin[df_clin['ID'].isin(eeg_subs)]
    
    clinical_vars = {
        'MDD': 'Major Depressive Disorder',
        'PTSD': 'PTSD',
        'GAD': 'Generalized Anxiety',
        'SUD': 'Substance Use Disorder',
        'ADHD': 'ADHD',
        'EAT': 'Eating Disorder',
        'Mania': 'Mania',
    }
    
    print(f"  {'Variable':<28s} {'Yes':>5s} {'No':>5s} {'Miss':>5s} {'%Yes':>6s}  Power")
    print(f"  {'─' * 60}")
    
    for var, label in clinical_vars.items():
        if var in df_eeg.columns:
            vals = df_eeg[var].dropna()
            n_yes = (vals == 1).sum()
            n_no = (vals == 0).sum()
            n_miss = len(df_eeg) - len(vals)
            pct = 100 * n_yes / len(vals) if len(vals) > 0 else 0
            power = 'adequate' if min(n_yes, n_no) >= 20 else 'LOW'
            print(f"  {label:<28s} {n_yes:5d} {n_no:5d} {n_miss:5d} {pct:5.1f}% {power}")
    
    # Medication and sex
    for var, label in [('Psychiatric_Medication', 'Medication'),
                       ('Assigned_Sex', 'Sex (1=M,2=F)')]:
        if var in df_eeg.columns:
            vals = df_eeg[var].dropna()
            if var == 'Assigned_Sex':
                n_m = (vals == 1).sum()
                n_f = (vals == 2).sum()
                print(f"  {label:<28s} M={n_m}, F={n_f}, miss={len(df_eeg)-len(vals)}")
            else:
                n_yes = (vals == 1).sum()
                n_no = (vals == 0).sum()
                n_miss = len(df_eeg) - len(vals)
                pct = 100 * n_yes / len(vals) if len(vals) > 0 else 0
                print(f"  {label:<28s} {n_yes:5d} {n_no:5d} {n_miss:5d} {pct:5.1f}%")
    
    # Comorbidity
    diag_cols = [c for c in ['MDD', 'PTSD', 'GAD', 'SUD', 'ADHD', 'EAT', 'Mania', 'OCD', 'PDA', 'SAD']
                 if c in df_eeg.columns]
    comorbidity = df_eeg[diag_cols].apply(lambda r: r.dropna().astype(int).sum(), axis=1)
    print(f"\n  Comorbidity: mean={comorbidity.mean():.1f}, median={comorbidity.median():.0f}, "
          f"range=[{comorbidity.min()}, {comorbidity.max()}]")
    
    # Healthy controls (0 diagnoses on primary 5)
    primary = [c for c in ['MDD', 'PTSD', 'GAD', 'SUD', 'ADHD'] if c in df_eeg.columns]
    hc_mask = df_eeg[primary].apply(lambda r: r.dropna().astype(int).sum() == 0, axis=1)
    n_hc = hc_mask.sum()
    print(f"  Healthy controls (0 of 5 primary dx): {n_hc}")
    
    # Figure: clinical bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    labels = []
    yes_counts = []
    no_counts = []
    for var, label in clinical_vars.items():
        if var in df_eeg.columns:
            vals = df_eeg[var].dropna()
            labels.append(label[:15])
            yes_counts.append((vals == 1).sum())
            no_counts.append((vals == 0).sum())
    x = np.arange(len(labels))
    ax.bar(x - 0.15, yes_counts, 0.3, label='Yes', color='#e17055')
    ax.bar(x + 0.15, no_counts, 0.3, label='No', color='#74b9ff')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title(f'OBS-8: Clinical Distribution (N={len(df_eeg)}, HC={n_hc})', fontsize=13, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/obs08_clinical_distribution.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → obs08_clinical_distribution.pdf")
else:
    print("  No clinical metadata available. Skipping OBS-8 figure.")


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RAW OBSERVATION SUMMARY")
print("=" * 70)

print(f"""
8 observation sets characterize the 3-class data BEFORE modeling:
  OBS-1: Dynamical metric distributions (7 core + 4 extra)
  OBS-2: Condition-dependent profiles (Neg/Neu/Pos per metric)
  OBS-3: Per-channel heterogeneity (34 channels × 7 metrics)
  OBS-4: Population firing rate timecourses (temporal structure)
  OBS-5: Theta-band tPLV connectivity (34×34 matrices × 3 conditions)
  OBS-6: Topological metric distributions (strength, clustering)
  OBS-4: Population firing rate timecourses (temporal structure)
  OBS-5: Theta-band tPLV connectivity (34×34 matrices × 3 conditions)
  OBS-6: Topological metric distributions (strength, clustering)
  OBS-7: Between-subject variability (raw subject × condition structure)
  OBS-8: Clinical metadata coverage ({n_clin if HAS_CLINICAL else '?'} subjects)

Each: PDF figure + terminal statistics → ./{OUT}/
""")

# Save data summaries as npz
np.savez(f'{OUT}/obs_summary.npz',
         metric_names=np.array(METRIC_NAMES))

# List output files
for fn in sorted(os.listdir(OUT)):
    fpath = os.path.join(OUT, fn)
    print(f"  {fn}: {os.path.getsize(fpath):,} bytes")
