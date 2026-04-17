#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
ARSPI-Net — Layer Ablation and Complementarity Analysis
═══════════════════════════════════════════════════════════════════════

SCIENTIFIC QUESTION
-------------------
Are the three response layers of ARSPI-Net (discriminative embedding,
dynamical trajectory, spatial topology) redundant, complementary, or
does one subsume the others?

This is the keystone experiment of the dissertation. The three-layer
decomposition is the central thesis:

    "ARSPI-Net reveals three operationally distinct response layers
     in affective EEG — discriminative representation, dynamical
     response, and topology/coupling — each sensitive to different
     aspects of the signal."

That thesis is a claim. This experiment is the test.

METHODOLOGY RULES APPLIED
-------------------------
  Rule 1 (Horizontal before vertical): This experiment cuts across
  Chapters 5, 6, and 7 using features from all three on the same
  task with the same protocol.

  Rule 2 (Claims require direct tests): The complementarity claim
  is tested by direct ablation/combination, not by narrative
  consistency across separate chapters.

  Rule 5 (Linear readouts for content comparison): Primary readout
  is L2-regularized logistic regression so geometric separability
  is exposed, not rescued by nonlinear kernels. RBF-SVM is run as
  an appendix sensitivity check.

FEATURE BLOCKS
--------------
  E = Chapter 5 embedding: BSC6 → PCA-64 per channel, flattened.
      Shape: (633, 2176) = 34 channels × 64 PCA dimensions.
      This is the primary discriminative representation.

  D = Chapter 6 dynamical: 7 core dynamical metrics per channel.
      Shape: (633, 238) = 34 channels × 7 metrics.
      These are the reservoir trajectory descriptors.

  T = Chapter 5/7 topological: weighted strength + clustering per channel.
      Shape: (633, 68) = 34 channels × 2 tPLV-derived metrics.
      These are the spatial graph descriptors.

  C = Chapter 7 coupling: scalar κ + mean signed coupling (strength
      column, clustering column) per observation.
      Shape: (633, 3).
      This is the structure-function coupling summary.

  BandPower = Conventional baseline: 5-band spectral power per channel.
      Shape: (633, 170) = 34 channels × 5 bands.
      This is the non-neuromorphic reference.

ABLATION MATRIX
---------------
  ID   Features          Scientific question
  ─────────────────────────────────────────────────────────────────
  A1   E                 Can the embedding alone carry the signal?
  A2   D                 Do dynamics alone classify above chance?
  A3   T                 Does topology alone classify emotion?
  A4   C                 Does coupling alone classify emotion?
  A5   D + T             Are temporal and spatial summaries complementary?
  A6   E + D             Do dynamics add to the embedding?
  A7   E + T             Does topology add to the embedding?
  A8   E + D + T         Are all three layers jointly complementary?
  A9   E + D + T + C     Does coupling add after everything else?
  A0   BandPower         Conventional non-neuromorphic reference

CLINICAL ABLATION (C1–C6)
-------------------------
  C1   E                 Does the embedding carry diagnosis signal?
  C2   D                 Are dynamics clinically informative alone?
  C3   T                 Does topology carry the strongest diagnosis signal?
  C4   D + T             Are temporal and spatial clinical signals complementary?
  C5   E + D + T         Does the full staged representation help diagnosis?
  C6   C                 Does coupling carry diagnosis signal independently?

PREDEFINED DECISION RULES
--------------------------
  If A1 remains best:
    The embedding layer is the primary discriminative substrate. The
    dynamical and topological layers are explanatory and organizational
    measurements rather than additive classification features.

  If A6, A7, or A8 exceeds A1:
    The later layers contain nonredundant information beyond the
    embedding, supporting ARSPI-Net as a genuinely multi-layer
    representation framework.

  If A3 or T-containing rows help diagnosis more than emotion:
    This supports layer-specific sensitivity — coarse affective
    discrimination is driven by reservoir embeddings, while clinical
    variation is more strongly reflected in spatial organization.

  If A4/A9 show no gain from coupling:
    Coupling is an organizational systems descriptor, not a primary
    discriminative feature family.

INPUT
-----
  shape_features_211.pkl (Chapter 5 features)
  ch6_ch7_3class_features.pkl (Chapters 6-7 features)
  clinical_profile.csv

OUTPUT
------
  layer_ablation_results.pkl
  3 PDF figures in ./ablation_figures/
  Terminal output with complete ablation tables

RUN COMMAND
-----------
  python layer_ablation.py
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
CH5_FILE = './shape_features_211.pkl'
CH67_FILE = './ch6_ch7_3class_features.pkl'
CLINICAL_FILE = './clinical_profile.csv'
RESULTS_FILE = './layer_ablation_results.pkl'
FIG_DIR = './ablation_figures'
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 10

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ARSPI-NET — LAYER ABLATION AND COMPLEMENTARITY ANALYSIS")
print("=" * 70)

with open(CH5_FILE, 'rb') as f:
    ch5 = pickle.load(f)
with open(CH67_FILE, 'rb') as f:
    ch67 = pickle.load(f)

y = ch5['y']
subjects = ch5['subjects']
unique_subjects = np.unique(subjects)
N_obs = len(y)
N_subj = len(unique_subjects)

# Verify alignment
assert np.array_equal(ch5['subjects'], ch67['subjects']), "Subject mismatch"
assert np.array_equal(ch5['y'], ch67['y']), "Label mismatch"

print(f"\nData: {N_obs} observations, {N_subj} subjects, 3 classes")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ══════════════════════════════════════════════════════════════════
# BUILD FEATURE BLOCKS
# ══════════════════════════════════════════════════════════════════
print(f"\nBuilding feature blocks...")

# E: BSC6-PCA64 embedding (Chapter 5)
E = ch5['lsm_bsc6_pca'].reshape(N_obs, -1)  # (633, 2176)
print(f"  E (embedding):   {E.shape}")

# D: 7 core dynamical metrics per channel (Chapter 6)
D = ch67['D'].reshape(N_obs, -1)  # (633, 238)
print(f"  D (dynamical):   {D.shape}")

# T: topological descriptors per channel (Chapter 7)
T = ch67['T_topo'].reshape(N_obs, -1)  # (633, 68)
print(f"  T (topological): {T.shape}")

# C: coupling summary per observation
# Compute coupling matrix for each observation, extract κ + signed means
print(f"  Computing coupling summaries...")
D_perchannel = ch67['D']       # (633, 34, 7)
T_perchannel = ch67['T_topo']  # (633, 34, 2)


def _rankdata(x):
    sorter = np.argsort(x)
    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def compute_coupling(D_obs, T_obs):
    """7×2 Spearman coupling matrix across 34 electrodes."""
    D_ranks = np.zeros_like(D_obs)
    T_ranks = np.zeros_like(T_obs)
    for j in range(D_obs.shape[1]):
        D_ranks[:, j] = _rankdata(D_obs[:, j])
    for k in range(T_obs.shape[1]):
        T_ranks[:, k] = _rankdata(T_obs[:, k])
    Dc = D_ranks - D_ranks.mean(axis=0)
    Tc = T_ranks - T_ranks.mean(axis=0)
    Dn = np.sqrt((Dc ** 2).sum(axis=0))
    Tn = np.sqrt((Tc ** 2).sum(axis=0))
    C_mat = (Dc.T @ Tc) / (Dn[:, None] * Tn[None, :] + 1e-12)
    return np.where(np.isfinite(C_mat), C_mat, 0.0)


C_block = np.zeros((N_obs, 3))
for i in range(N_obs):
    C_mat = compute_coupling(D_perchannel[i], T_perchannel[i])
    p, q = C_mat.shape
    C_block[i, 0] = np.linalg.norm(C_mat, 'fro') / np.sqrt(p * q)  # κ (Frobenius-normalized)
    C_block[i, 1] = C_mat[:, 0].mean()               # mean signed strength coupling
    C_block[i, 2] = C_mat[:, 1].mean()               # mean signed clustering coupling
print(f"  C (coupling):    {C_block.shape}")

# BandPower: conventional baseline (Chapter 5)
BP = ch5['conv_feats'].reshape(N_obs, -1)  # (633, 170)
print(f"  BandPower:       {BP.shape}")

# Load clinical metadata
try:
    df_clin = pd.read_csv(CLINICAL_FILE)
    df_clin = df_clin.drop_duplicates(subset='ID', keep='first')
    HAS_CLINICAL = True
    print(f"  Clinical metadata: {len(df_clin)} subjects")
except:
    HAS_CLINICAL = False
    print(f"  Clinical metadata: not found")


# ══════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════
def evaluate_3class(X, y, subjects, classifier='logistic'):
    """
    10-fold subject-level stratified CV for 3-class emotion classification.
    Returns dict with balanced accuracy stats per fold.
    """
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    accs = []
    for tr, te in cv.split(X, y, groups=subjects):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        if classifier == 'logistic':
            clf = LogisticRegression(C=1.0, class_weight='balanced',
                                     max_iter=2000, random_state=RANDOM_STATE,
                                     solver='lbfgs', multi_class='multinomial')
        elif classifier == 'rbf_svm':
            clf = SVC(kernel='rbf', class_weight='balanced',
                      random_state=RANDOM_STATE)
        else:
            clf = SVC(kernel='linear', class_weight='balanced',
                      random_state=RANDOM_STATE)
        clf.fit(Xtr, y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    accs = np.array(accs)
    return {'mean': float(accs.mean()), 'std': float(accs.std()),
            'folds': accs.tolist()}


def evaluate_clinical(X, y_clin, subjects_clin, classifier='logistic'):
    """
    5-fold subject-level stratified CV for binary clinical detection.
    Returns dict with balanced accuracy and AUC.
    """
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE)
    accs, aucs = [], []
    for tr, te in cv.split(X, y_clin, groups=subjects_clin):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        if classifier == 'logistic':
            clf = LogisticRegression(C=1.0, class_weight='balanced',
                                     max_iter=2000, random_state=RANDOM_STATE)
        else:
            clf = SVC(kernel='rbf', class_weight='balanced',
                      random_state=RANDOM_STATE, probability=True)
        clf.fit(Xtr, y_clin[tr])
        pred = clf.predict(Xte)
        accs.append(balanced_accuracy_score(y_clin[te], pred))
        try:
            if hasattr(clf, 'predict_proba'):
                prob = clf.predict_proba(Xte)[:, 1]
            else:
                prob = clf.decision_function(Xte)
            aucs.append(roc_auc_score(y_clin[te], prob))
        except:
            aucs.append(0.5)
    return {'acc': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'auc': float(np.mean(aucs)), 'auc_std': float(np.std(aucs))}


def build_clinical_data(var_name):
    """Build subject-level feature arrays for binary clinical detection."""
    if not HAS_CLINICAL or var_name not in df_clin.columns:
        return None, None, None, None, None
    if var_name == 'Assigned_Sex':
        yes_ids = set(df_clin[df_clin[var_name] == 1]['ID'].values)
        no_ids = set(df_clin[df_clin[var_name] == 2]['ID'].values)
    else:
        yes_ids = set(df_clin[df_clin[var_name] == 1]['ID'].values)
        no_ids = set(df_clin[df_clin[var_name] == 0]['ID'].values)

    # Subject-level features: average across conditions
    X_dict = {}
    y_dict = {}
    for si, sid in enumerate(unique_subjects):
        mask = subjects == sid
        if sid in yes_ids:
            X_dict[sid] = mask
            y_dict[sid] = 1
        elif sid in no_ids:
            X_dict[sid] = mask
            y_dict[sid] = 0

    sids = sorted(X_dict.keys())
    y_c = np.array([y_dict[s] for s in sids])
    s_c = np.array(sids)

    n_yes = (y_c == 1).sum()
    n_no = (y_c == 0).sum()
    if n_yes < 15 or n_no < 15:
        return None, None, None, None, None

    return sids, y_c, s_c, n_yes, n_no


def get_subject_features(feature_matrix, sids):
    """Average feature matrix across conditions for each subject."""
    feats = []
    for sid in sids:
        mask = subjects == sid
        feats.append(feature_matrix[mask].mean(axis=0))
    return np.array(feats)


# ══════════════════════════════════════════════════════════════════
# PART 1: EMOTION CLASSIFICATION ABLATION (A0–A9)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: 3-CLASS EMOTION CLASSIFICATION ABLATION")
print("=" * 70)
print(f"\n  Mathematical motivation: The dissertation claims ARSPI-Net")
print(f"  reveals three operationally distinct response layers. This")
print(f"  experiment directly tests that claim by measuring the")
print(f"  discriminative content of each layer and their combinations")
print(f"  using a linear readout that exposes geometric separability.\n")

ablation_defs = [
    ('A0', 'BandPower',       BP),
    ('A1', 'E (embedding)',    E),
    ('A2', 'D (dynamical)',    D),
    ('A3', 'T (topological)',  T),
    ('A4', 'C (coupling)',     C_block),
    ('A5', 'D + T',           np.hstack([D, T])),
    ('A6', 'E + D',           np.hstack([E, D])),
    ('A7', 'E + T',           np.hstack([E, T])),
    ('A8', 'E + D + T',       np.hstack([E, D, T])),
    ('A9', 'E + D + T + C',   np.hstack([E, D, T, C_block])),
]

# ── Step 3: Raw observation — run all combinations ──
print(f"  Raw observation — Logistic regression, {N_FOLDS}-fold subject-level CV:\n")
print(f"  {'ID':4s}  {'Features':20s}  {'Dim':>6s}  {'Accuracy':>12s}  {'Chance':>8s}")
print(f"  {'─' * 58}")

emotion_results = {}
for aid, label, X in ablation_defs:
    res = evaluate_3class(X, y, subjects, classifier='logistic')
    emotion_results[aid] = {'label': label, 'dim': X.shape[1], **res}
    print(f"  {aid:4s}  {label:20s}  {X.shape[1]:6d}  "
          f"{res['mean']*100:5.1f}% ± {res['std']*100:4.1f}%  {'33.3%':>8s}")

# ── Step 4: Analysis — interpret against decision rules ──
print(f"\n  Analysis — Decision rule evaluation:\n")

a1_acc = emotion_results['A1']['mean']
best_id = max(emotion_results, key=lambda k: emotion_results[k]['mean'])
best_acc = emotion_results[best_id]['mean']

print(f"  E alone (A1): {a1_acc*100:.1f}%")
print(f"  Best overall: {best_id} ({emotion_results[best_id]['label']}) "
      f"at {best_acc*100:.1f}%")

# Check if later layers add to embedding
for aid in ['A6', 'A7', 'A8', 'A9']:
    gain = emotion_results[aid]['mean'] - a1_acc
    print(f"  {aid} vs A1: {gain*100:+.1f}pp "
          f"({'gain' if gain > 0 else 'no gain'})")

# Decision rule interpretation
if best_id == 'A1':
    print(f"\n  → DECISION: A1 (E alone) is best. The embedding is the primary")
    print(f"    discriminative substrate. Later layers are explanatory and")
    print(f"    organizational, not additive for emotion classification.")
elif best_id in ['A6', 'A7', 'A8', 'A9']:
    gain = best_acc - a1_acc
    print(f"\n  → DECISION: {best_id} exceeds A1 by {gain*100:.1f}pp. The later")
    print(f"    layers contain nonredundant discriminative information beyond")
    print(f"    the embedding, supporting ARSPI-Net as a multi-layer framework.")
elif best_id == 'A0':
    print(f"\n  → OBSERVATION: BandPower (A0) is best at {best_acc*100:.1f}%.")
    print(f"    The linear readout favors the lower-dimensional spectral")
    print(f"    representation. The embedding's advantage may require")
    print(f"    nonlinear decoding (see RBF sensitivity check).")
else:
    print(f"\n  → OBSERVATION: {best_id} is best. Unexpected — further analysis needed.")

# BandPower vs reservoir comparison
bp_acc = emotion_results['A0']['mean']
print(f"\n  BandPower (A0) vs Embedding (A1): {bp_acc*100:.1f}% vs {a1_acc*100:.1f}% "
      f"(Δ = {(a1_acc-bp_acc)*100:+.1f}pp)")

emotion_results['decision'] = {
    'best_id': best_id,
    'best_acc': best_acc,
    'a1_acc': a1_acc,
    'bp_acc': bp_acc,
}

# Figure 1: Ablation bar chart
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ids = [r[0] for r in ablation_defs]
accs = [emotion_results[aid]['mean'] * 100 for aid in ids]
stds = [emotion_results[aid]['std'] * 100 for aid in ids]
labels = [emotion_results[aid]['label'] for aid in ids]
dims = [emotion_results[aid]['dim'] for aid in ids]

colors = ['#95a5a6',  # A0 bandpower (gray)
          '#0984e3',  # A1 E (blue)
          '#e17055',  # A2 D (red)
          '#00b894',  # A3 T (green)
          '#6c5ce7',  # A4 C (purple)
          '#fd79a8',  # A5 D+T (pink)
          '#74b9ff',  # A6 E+D (light blue)
          '#55efc4',  # A7 E+T (light green)
          '#2d3436',  # A8 E+D+T (dark)
          '#636e72',  # A9 E+D+T+C (darker gray)
          ]

bars = ax.bar(range(len(ids)), accs, yerr=stds, capsize=4,
              color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(33.3, color='red', linestyle='--', alpha=0.5, label='Chance (33.3%)')
ax.axhline(accs[1], color='blue', linestyle=':', alpha=0.3, label=f'E alone ({accs[1]:.1f}%)')
ax.set_xticks(range(len(ids)))
ax.set_xticklabels([f'{aid}\n{lab}\n({dim}d)' for aid, lab, dim in zip(ids, labels, dims)],
                    fontsize=8, rotation=0)
ax.set_ylabel('Balanced Accuracy (%)')
ax.set_title('Layer Ablation: 3-Class Emotion Classification (Logistic Regression)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
# Annotate bars
for i, (acc, std) in enumerate(zip(accs, stds)):
    ax.text(i, acc + std + 0.5, f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig01_emotion_ablation.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"\n  → fig01_emotion_ablation.pdf")


# ══════════════════════════════════════════════════════════════════
# PART 2: CLINICAL DETECTION ABLATION (C1–C6)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: CLINICAL DETECTION ABLATION")
print("=" * 70)
print(f"\n  Mathematical motivation: The dissertation claims different layers")
print(f"  are sensitive to different aspects of the signal. If topology (T)")
print(f"  carries stronger clinical signal than dynamics (D) or embedding (E),")
print(f"  that directly supports layer-specific sensitivity.\n")

clinical_ablation_defs = [
    ('C1', 'E (embedding)',    E),
    ('C2', 'D (dynamical)',    D),
    ('C3', 'T (topological)',  T),
    ('C4', 'D + T',           np.hstack([D, T])),
    ('C5', 'E + D + T',       np.hstack([E, D, T])),
    ('C6', 'C (coupling)',     C_block),
]

clinical_tasks = ['SUD', 'MDD', 'PTSD', 'GAD', 'ADHD']
clinical_results = {}

if not HAS_CLINICAL:
    print("  SKIPPED: No clinical metadata.")
else:
    for var in clinical_tasks:
        sids, y_c, s_c, n_yes, n_no = build_clinical_data(var)
        if sids is None:
            continue

        print(f"  {var} (n={n_no} vs {n_yes}):")

        # ── Raw observation: group accuracies for each block ──
        print(f"    {'ID':4s}  {'Features':16s}  {'Accuracy':>12s}  {'AUC':>10s}")
        print(f"    {'─' * 48}")

        clinical_results[var] = {'n_yes': n_yes, 'n_no': n_no, 'blocks': {}}

        for cid, label, X_full in clinical_ablation_defs:
            X_subj = get_subject_features(X_full, sids)
            res = evaluate_clinical(X_subj, y_c, s_c, classifier='logistic')
            clinical_results[var]['blocks'][cid] = {'label': label, **res}
            print(f"    {cid:4s}  {label:16s}  "
                  f"{res['acc']*100:5.1f}% ± {res['acc_std']*100:4.1f}%  "
                  f"{res['auc']:.3f} ± {res['auc_std']:.3f}")

        # ── Analysis ──
        e_acc = clinical_results[var]['blocks']['C1']['acc']
        t_acc = clinical_results[var]['blocks']['C3']['acc']
        d_acc = clinical_results[var]['blocks']['C2']['acc']
        if t_acc > e_acc and t_acc > d_acc:
            print(f"    → Topology (T) carries strongest clinical signal for {var}")
        elif e_acc > t_acc and e_acc > d_acc:
            print(f"    → Embedding (E) carries strongest clinical signal for {var}")
        elif d_acc > t_acc and d_acc > e_acc:
            print(f"    → Dynamics (D) carries strongest clinical signal for {var}")
        print()

    # Figure 2: Clinical ablation heatmap
    if clinical_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy heatmap
        tasks_present = [v for v in clinical_tasks if v in clinical_results]
        cids = [c[0] for c in clinical_ablation_defs]
        clabels = [c[1] for c in clinical_ablation_defs]

        acc_matrix = np.zeros((len(tasks_present), len(cids)))
        auc_matrix = np.zeros((len(tasks_present), len(cids)))
        for ti, var in enumerate(tasks_present):
            for ci, cid in enumerate(cids):
                if cid in clinical_results[var]['blocks']:
                    acc_matrix[ti, ci] = clinical_results[var]['blocks'][cid]['acc']
                    auc_matrix[ti, ci] = clinical_results[var]['blocks'][cid]['auc']

        ax = axes[0]
        im = ax.imshow(acc_matrix * 100, aspect='auto', cmap='YlOrRd', vmin=45, vmax=65)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels([f'{c}\n{l[:10]}' for c, l in zip(cids, clabels)], fontsize=8)
        ax.set_yticks(range(len(tasks_present)))
        ax.set_yticklabels(tasks_present)
        for ti in range(len(tasks_present)):
            for ci in range(len(cids)):
                ax.text(ci, ti, f'{acc_matrix[ti, ci]*100:.1f}',
                        ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label='Balanced Accuracy (%)')
        ax.set_title('Clinical Detection: Accuracy', fontweight='bold')

        ax = axes[1]
        im2 = ax.imshow(auc_matrix, aspect='auto', cmap='YlOrRd', vmin=0.45, vmax=0.65)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels([f'{c}\n{l[:10]}' for c, l in zip(cids, clabels)], fontsize=8)
        ax.set_yticks(range(len(tasks_present)))
        ax.set_yticklabels(tasks_present)
        for ti in range(len(tasks_present)):
            for ci in range(len(cids)):
                ax.text(ci, ti, f'{auc_matrix[ti, ci]:.3f}',
                        ha='center', va='center', fontsize=8)
        plt.colorbar(im2, ax=ax, label='AUC')
        ax.set_title('Clinical Detection: AUC', fontweight='bold')

        fig.suptitle('Clinical Ablation: Layer-Specific Sensitivity (Logistic Regression)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{FIG_DIR}/fig02_clinical_ablation.pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  → fig02_clinical_ablation.pdf")


# ══════════════════════════════════════════════════════════════════
# PART 3: RBF-SVM SENSITIVITY CHECK (APPENDIX)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: RBF-SVM SENSITIVITY CHECK (APPENDIX)")
print("=" * 70)
print(f"\n  This repeats the top rows with RBF-SVM to verify that the linear")
print(f"  readout results are not an artifact of classifier choice.\n")

rbf_rows = ['A0', 'A1', 'A2', 'A3', 'A5', 'A8']
rbf_features = {aid: X for aid, _, X in ablation_defs if aid in rbf_rows}

print(f"  {'ID':4s}  {'Features':20s}  {'Logistic':>12s}  {'RBF-SVM':>12s}  {'Δ':>8s}")
print(f"  {'─' * 62}")

rbf_results = {}
for aid in rbf_rows:
    X = rbf_features[aid]
    res_rbf = evaluate_3class(X, y, subjects, classifier='rbf_svm')
    rbf_results[aid] = res_rbf
    log_acc = emotion_results[aid]['mean']
    rbf_acc = res_rbf['mean']
    delta = rbf_acc - log_acc
    print(f"  {aid:4s}  {emotion_results[aid]['label']:20s}  "
          f"{log_acc*100:5.1f}% ± {emotion_results[aid]['std']*100:4.1f}%  "
          f"{rbf_acc*100:5.1f}% ± {res_rbf['std']*100:4.1f}%  "
          f"{delta*100:+5.1f}pp")

# Figure 3: Linear vs RBF comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
x = np.arange(len(rbf_rows))
w = 0.35
log_vals = [emotion_results[aid]['mean'] * 100 for aid in rbf_rows]
rbf_vals = [rbf_results[aid]['mean'] * 100 for aid in rbf_rows]
ax.bar(x - w/2, log_vals, w, label='Logistic Regression', color='#74b9ff', edgecolor='white')
ax.bar(x + w/2, rbf_vals, w, label='RBF-SVM', color='#e17055', edgecolor='white')
ax.axhline(33.3, color='red', linestyle='--', alpha=0.3, label='Chance')
ax.set_xticks(x)
ax.set_xticklabels([f'{aid}\n{emotion_results[aid]["label"][:12]}' for aid in rbf_rows],
                    fontsize=8)
ax.set_ylabel('Balanced Accuracy (%)')
ax.set_title('Appendix: Linear vs Nonlinear Readout Comparison', fontsize=12, fontweight='bold')
ax.legend()
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/fig03_rbf_sensitivity.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"\n  → fig03_rbf_sensitivity.pdf")


# ══════════════════════════════════════════════════════════════════
# PART 4: 3-CLASS vs 4-CLASS REGIME SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: 3-CLASS vs 4-CLASS REGIME SUMMARY TABLE")
print("=" * 70)
print(f"\n  This table assembles all cross-granularity comparisons into")
print(f"  a single quantitative display.\n")

# 3-class values (from this session and Chapter 5)
# 4-class values (from earlier session)
regime_table = {
    'Number of classes':              ('3', '4'),
    'Observations':                   ('633', '844'),
    'Best raw classifier':            ('PCA-64 + SVM', 'BandPower + SVM'),
    'Best raw balanced accuracy':     ('63.4%', '40.4%'),
    'Best centered accuracy':         ('79.4%', '52.0%'),
    'Subject-centering gain':         ('+16.0pp', '+10.8pp'),
    'BandPower accuracy':             ('50.9%', '40.4%'),
    'Reservoir > BandPower?':         ('Yes (+12.5pp)', 'No (-6.2pp)'),
    'Best GNN beats non-propagated?': ('No', 'No'),
    'Subject variance fraction':      ('62.6%', '47.0%'),
    'Condition variance fraction':    ('8.7%', '2.4%'),
    'Subject/condition ratio':        ('7.2×', '19.6×'),
    'Strongest clinical finding':     ('SUD p=0.0004', 'PTSD p=0.036'),
    'Clinical signal hierarchy':      ('SUD dominant', 'PTSD dominant'),
}

print(f"  {'Quantity':<35s}  {'3-class':>15s}  {'4-class':>15s}")
print(f"  {'─' * 70}")
for qty, (v3, v4) in regime_table.items():
    print(f"  {qty:<35s}  {v3:>15s}  {v4:>15s}")


# ══════════════════════════════════════════════════════════════════
# SAVE & SUMMARY
# ══════════════════════════════════════════════════════════════════
all_results = {
    'emotion_ablation': emotion_results,
    'clinical_ablation': clinical_results,
    'rbf_sensitivity': rbf_results,
    'regime_table': regime_table,
}
with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(all_results, f, protocol=4)

print(f"\n{'=' * 70}")
print("COMPLETE ABLATION SUMMARY")
print(f"{'=' * 70}")

print(f"\n  ── EMOTION CLASSIFICATION (logistic regression) ──")
for aid in ['A0', 'A1', 'A2', 'A3', 'A4']:
    r = emotion_results[aid]
    print(f"  {aid} {r['label']:20s}: {r['mean']*100:.1f}%")
print(f"  ---")
for aid in ['A5', 'A6', 'A7', 'A8', 'A9']:
    r = emotion_results[aid]
    gain = r['mean'] - emotion_results['A1']['mean']
    print(f"  {aid} {r['label']:20s}: {r['mean']*100:.1f}% "
          f"({gain*100:+.1f}pp vs E alone)")

if HAS_CLINICAL and clinical_results:
    print(f"\n  ── CLINICAL DETECTION (logistic regression) ──")
    for var in clinical_results:
        best_cid = max(clinical_results[var]['blocks'],
                       key=lambda k: clinical_results[var]['blocks'][k]['acc'])
        best = clinical_results[var]['blocks'][best_cid]
        print(f"  {var}: best = {best_cid} ({best['label'][:15]}) "
              f"at {best['acc']*100:.1f}% (AUC={best['auc']:.3f})")

print(f"\n  Figures: {len(os.listdir(FIG_DIR))} PDFs in {FIG_DIR}/")
print(f"  Results: {RESULTS_FILE}")
print(f"\n  ── ABLATION COMPLETE ──")
print("=" * 70)
