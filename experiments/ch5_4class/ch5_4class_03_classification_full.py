"""
============================================================================
Chapter 5 — Script 03: CLASSIFICATION + CLINICAL INTERPRETABILITY (4-Class)
============================================================================
PURPOSE:
  Complete experimental program for the 4-class ARSPI-Net analysis.
  Every experiment follows the scientific cycle from the directive:
    1. Mathematical motivation (why this experiment)
    2. Experimental design (what is measured)
    3. Observation (quantitative results)
    4. Analysis (what the data reveals)
    5. Next question (what motivates the following experiment)

EXPERIMENTS:
  ── CLASSIFICATION TIER ──
  EXP-1:  Baseline classification (M1 BandPower, M2 PCA-64, M3 Relational)
  EXP-2:  GNN architecture comparison (4 archs × 3 adjacencies)
  EXP-3:  Confusion matrix and per-class recall analysis
  EXP-4:  Pairwise contrasts (6 pairs: within- vs between-valence)
  EXP-5:  Variance decomposition (subject / condition / residual)
  EXP-6:  Per-channel discrimination analysis

  ── CLINICAL INTERPRETABILITY TIER ──
  EXP-7:  Channel-level clinical biomarkers (MDD, PTSD, GAD, SUD, sex, med)
  EXP-8:  Condition × clinical status interactions
  EXP-9:  Edge-level biomarkers
  EXP-10: Comorbidity burden vs network complexity
  EXP-11: Within-valence × clinical interactions (new for 4-class)

INPUT:   shape_features_4class.pkl, clinical_profile.csv
OUTPUT:  ch5_4class_results.pkl, ch5_4class_figures/*.pdf

Usage:   python ch5_4class_classification_full.py
============================================================================
"""
import numpy as np
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                              confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon, mannwhitneyu
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
INPUT_FILE = './shape_features_4class.pkl'
CLINICAL_FILE = './clinical_profile.csv'  # from prior session; adjust path
OUTPUT_DIR = './ch5_4class_figures'
RESULTS_FILE = './ch5_4class_results.pkl'
N_FOLDS = 10
RANDOM_STATE = 42

COND_NAMES = {0: 'Threat', 1: 'Mutilation', 2: 'Cute', 3: 'Erotic'}
COND_COLORS = {0: '#e74c3c', 1: '#c0392b', 2: '#27ae60', 3: '#2980b9'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("CHAPTER 5 — 4-CLASS FULL EXPERIMENTAL PROGRAM")
print("=" * 70)

with open(INPUT_FILE, 'rb') as f:
    d = pickle.load(f)

pca64 = d['lsm_bsc6_pca']     # (N_obs, 34, 64)
conv_feats = d['conv_feats']   # (N_obs, 34, 5)
bsc6_raw = d['lsm_bsc6_raw']  # (N_obs, 34, 1536)
mfr = d['lsm_mfr']            # (N_obs, 34, 256)
y = d['y']
subjects = d['subjects']
X_ds = d['X_ds']

N_obs, N_ch, D = pca64.shape
unique_subjects = np.unique(subjects)
N_subj = len(unique_subjects)
K = len(np.unique(y))
triu = np.triu_indices(N_ch, k=1)

print(f"\nData: {N_obs} obs, {N_subj} subjects, {N_ch} channels, {K} classes")
print(f"Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Load clinical metadata
try:
    df_clin = pd.read_csv(CLINICAL_FILE)
    clin_map = {int(row['ID']): row.to_dict() for _, row in df_clin.iterrows()}
    print(f"Clinical metadata: {len(clin_map)} subjects, {len(df_clin.columns)} variables")
    clinical_available = True
except:
    print("WARNING: Clinical metadata not found. EXP-7 through EXP-11 will be skipped.")
    clinical_available = False
    clin_map = {}

# Connectivity matrices (precompute once)
def fast_conn(emb):
    N, C, Dim = emb.shape
    X = emb - emb.mean(axis=2, keepdims=True)
    norms = np.sqrt((X**2).sum(axis=2, keepdims=True))
    norms[norms == 0] = 1
    return np.einsum('nid,njd->nij', X/norms, X/norms)

print("Computing connectivity matrices...")
conn_all = fast_conn(pca64)

# Per-subject connectivity
subj_conn = np.zeros((N_subj, N_ch, N_ch))
for si, s in enumerate(unique_subjects):
    subj_conn[si] = conn_all[subjects == s].mean(axis=0)

# ═══════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION SETUP
# ═══════════════════════════════════════════════════════════════════════
subj_labels = np.array([int(np.median(y[subjects == s])) for s in unique_subjects])
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(unique_subjects, subj_labels))

def evaluate_method(X, name, verbose=True):
    accs, mccs = [], []
    all_yt, all_yp = [], []
    for train_si, test_si in folds:
        tr_s = unique_subjects[train_si]
        te_s = unique_subjects[test_si]
        tr = np.isin(subjects, tr_s)
        te = np.isin(subjects, te_s)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(Xtr, y[tr])
        yp = clf.predict(Xte)
        accs.append(balanced_accuracy_score(y[te], yp))
        mccs.append(matthews_corrcoef(y[te], yp))
        all_yt.extend(y[te]); all_yp.extend(yp)
    ba = np.array(accs); mc = np.array(mccs)
    if verbose:
        print(f"  {name:<40s} {ba.mean()*100:5.1f}% ±{ba.std()*100:4.1f}  MCC={mc.mean():.3f}")
    return {'bal_accs': ba, 'mccs': mc, 'mean_acc': ba.mean(), 'std_acc': ba.std(),
            'mean_mcc': mc.mean(), 'y_true': np.array(all_yt), 'y_pred': np.array(all_yp)}


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                    CLASSIFICATION TIER (EXP 1–6)                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ═══════════════════════════════════════════════════════════════════════
# EXP-1: BASELINE CLASSIFICATION
# Mathematical motivation: The data processing inequality guarantees
# that each processing stage can only preserve or lose information
# about the class label. The three baselines isolate the contribution
# of each pipeline stage: conventional features (M1), reservoir
# temporal encoding (M2), and inter-channel relational structure (M3).
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP-1: BASELINE CLASSIFICATION")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: isolate each pipeline stage's contribution")
print(f"  to discriminative information preservation.\n")

F_band = conv_feats.reshape(N_obs, -1)
res_m1 = evaluate_method(F_band, 'M1: BandPower + SVM')

F_pca = pca64.reshape(N_obs, -1)
res_m2 = evaluate_method(F_pca, 'M2: PCA-64 Concat + SVM')

n_pairs = len(triu[0])
F_rel_raw = np.zeros((N_obs, n_pairs * D))
for obs in range(N_obs):
    for pi, (i, j) in enumerate(zip(*triu)):
        F_rel_raw[obs, pi*D:(pi+1)*D] = pca64[obs, i] - pca64[obs, j]
n_rel_comp = min(200, N_obs - 1, F_rel_raw.shape[1])
pca_rel = PCA(n_components=n_rel_comp, random_state=RANDOM_STATE)
F_rel = pca_rel.fit_transform(F_rel_raw)
res_m3 = evaluate_method(F_rel, 'M3: Relational + SVM')

print(f"\n  Analysis: Reservoir adds +{(res_m2['mean_acc']-res_m1['mean_acc'])*100:.1f}pp "
      f"over conventional features.")
print(f"  Relational structure adds +{(res_m3['mean_acc']-res_m2['mean_acc'])*100:.1f}pp "
      f"over concatenation.")
print(f"  → The next question: does message-passing improve on relational features?")


# ═══════════════════════════════════════════════════════════════════════
# EXP-2: GNN ARCHITECTURE COMPARISON
# Mathematical motivation: Message-passing replaces h_v with a weighted
# combination of neighbor embeddings. On a graph with C nodes and
# density ρ, the spectral decomposition of Â^K shows that high-frequency
# (inter-node contrast) components decay exponentially with K.
# This predicts that GNNs should degrade performance on small dense graphs.
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP-2: GNN ARCHITECTURE COMPARISON")
print(f"{'='*70}")
print(f"\n  Mathematical motivation: spectral analysis predicts variance")
print(f"  destruction at K≥2 layers on C={N_ch}-node graphs.\n")

def build_adj(emb, method='correlation'):
    X = emb - emb.mean(axis=1, keepdims=True)
    n = np.sqrt((X**2).sum(axis=1, keepdims=True)); n[n==0]=1
    corr = (X/n) @ (X/n).T
    if method == 'correlation':
        v = corr[triu]; thr = np.percentile(v, 80)
        A = (corr >= thr).astype(float); np.fill_diagonal(A, 0)
    elif method == 'knn':
        A = np.zeros_like(corr)
        for i in range(N_ch):
            nb = np.argsort(-corr[i]); nb = nb[nb!=i][:7]
            A[i,nb]=1; A[nb,i]=1
    elif method == 'full_weighted':
        A = np.maximum(corr, 0); np.fill_diagonal(A, 0)
    return A

def gin_prop(H, A, W, eps=0.1):
    return np.maximum(((1+eps)*H + A@H) @ W, 0)

def cheb_prop(H, A, W):
    Ah = A + np.eye(A.shape[0])
    Dh = np.diag(1.0/np.sqrt(np.maximum(Ah.sum(1), 1e-10)))
    L = np.eye(A.shape[0]) - Dh@Ah@Dh
    return np.maximum((H + L@H) @ W, 0)

def edge_prop(H, A, W):
    N, Din = H.shape; Dout = W.shape[1]; out = np.zeros((N, Dout))
    for i in range(N):
        nb = np.where(A[i]>0)[0]
        if len(nb)==0:
            ef = np.concatenate([H[i], np.zeros(Din)])
            out[i] = np.maximum(ef@W, 0)
        else:
            ef = np.zeros((len(nb), 2*Din))
            for ji,j in enumerate(nb): ef[ji] = np.concatenate([H[i], H[j]-H[i]])
            out[i] = np.maximum(ef@W, 0).max(axis=0)
    return out

def trans_prop(H, A, WQ, WK, WV):
    Q=H@WQ; Km=H@WK; V=H@WV
    sc = (Q@Km.T)/np.sqrt(H.shape[1]) + A*2
    sc -= sc.max(1, keepdims=True); e=np.exp(sc)
    return np.maximum((e/(e.sum(1,keepdims=True)+1e-10))@V, 0)

def run_gnn(arch, adj_method, n_seeds=5, n_layers=2):
    accs = []
    for train_si, test_si in folds:
        tr_s = unique_subjects[train_si]; te_s = unique_subjects[test_si]
        tr = np.isin(subjects, tr_s); te = np.isin(subjects, te_s)
        all_tr, all_te = [], []
        for seed_i in range(n_seeds):
            rng = np.random.RandomState(seed_i*7+13)
            def proc(obs_indices):
                feats = []
                for oi in obs_indices:
                    emb = pca64[oi]; A = build_adj(emb, adj_method)
                    H = emb.copy(); layers = [H]
                    rng2 = np.random.RandomState(seed_i*7+13)
                    for _ in range(n_layers):
                        Di, Do = H.shape[1], 32
                        if arch=='gin': W=rng2.randn(Di,Do)*np.sqrt(2/Di); H=gin_prop(H,A,W)
                        elif arch=='chebnet': W=rng2.randn(Di,Do)*np.sqrt(2/Di); H=cheb_prop(H,A,W)
                        elif arch=='edgeconv': W=rng2.randn(2*Di,Do)*np.sqrt(2/(2*Di)); H=edge_prop(H,A,W)
                        elif arch=='transformer':
                            WQ=rng2.randn(Di,Di)*np.sqrt(2/Di)
                            WK=rng2.randn(Di,Di)*np.sqrt(2/Di)
                            WV=rng2.randn(Di,Do)*np.sqrt(2/Di)
                            H=trans_prop(H,A,WQ,WK,WV)
                        layers.append(H)
                    rd = []
                    for lo in layers: rd.extend([lo.mean(0), lo.max(0)])
                    feats.append(np.concatenate(rd))
                return np.array(feats)
            all_tr.append(proc(np.where(tr)[0]))
            all_te.append(proc(np.where(te)[0]))
        Xtr = np.concatenate(all_tr, axis=1)
        Xte = np.concatenate(all_te, axis=1)
        sc = StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(Xtr, y[tr]); accs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return np.array(accs)

gnn_results = {}
for arch in ['gin', 'chebnet', 'edgeconv', 'transformer']:
    best_acc, best_adj, best_accs = 0, None, None
    for adj in ['correlation', 'knn', 'full_weighted']:
        print(f"  {arch}+{adj}...", end=' ', flush=True)
        t0 = time.time()
        fa = run_gnn(arch, adj)
        print(f"{fa.mean()*100:.1f}% ({time.time()-t0:.0f}s)")
        if fa.mean() > best_acc: best_acc=fa.mean(); best_adj=adj; best_accs=fa
    try: _, p = wilcoxon(res_m3['bal_accs'], best_accs)
    except: p = 1.0
    gnn_results[arch] = {'best_adj':best_adj, 'bal_accs':best_accs,
                          'mean_acc':best_accs.mean(), 'std_acc':best_accs.std(), 'p_vs_m3':p}
    print(f"  → {arch}: {best_accs.mean()*100:.1f}% (p={p:.4f})")

print(f"\n  Analysis: {'Every' if all(r['p_vs_m3']<0.05 for r in gnn_results.values()) else 'Some'} "
      f"GNN architecture degrades performance vs relational SVM.")
print(f"  → The next question: where does the classifier confuse classes?")


# ═══════════════════════════════════════════════════════════════════════
# EXP-3: CONFUSION MATRIX ANALYSIS
# Mathematical motivation: The confusion matrix reveals the geometry
# of class separability. Within-valence pairs (Threat↔Mutilation,
# Cute↔Erotic) are predicted to produce more confusions than
# between-valence pairs, reflecting smaller inter-centroid distances.
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP-3: CONFUSION MATRIX ANALYSIS")
print(f"{'='*70}")

cm = confusion_matrix(res_m3['y_true'], res_m3['y_pred'])
print(f"\n  Relational SVM confusion matrix:")
labels = [COND_NAMES[c] for c in range(K)]
print(f"  {'':>12s}", end='')
for c in labels: print(f"  {c:>10s}", end='')
print()
for i in range(K):
    print(f"  {labels[i]:>12s}", end='')
    for j in range(K): print(f"  {cm[i,j]:10d}", end='')
    recall = cm[i,i]/cm[i].sum() if cm[i].sum()>0 else 0
    print(f"  (recall={recall*100:.1f}%)")

# Within-valence confusions
within_neg = cm[0,1] + cm[1,0]
within_pos = cm[2,3] + cm[3,2]
between_total = cm.sum() - np.trace(cm) - within_neg - within_pos
print(f"\n  Within-valence confusions: {within_neg} (neg) + {within_pos} (pos) = {within_neg+within_pos}")
print(f"  Between-valence confusions: {between_total}")
print(f"  → Within-valence pairs account for "
      f"{(within_neg+within_pos)/(cm.sum()-np.trace(cm))*100:.0f}% of all errors")


# ═══════════════════════════════════════════════════════════════════════
# EXP-4: PAIRWISE CONTRAST ANALYSIS
# Mathematical motivation: The 4-class design enables 6 pairwise
# contrasts, organized into a 2×2 hierarchy: 2 within-valence pairs
# and 4 between-valence pairs. Information theory predicts that
# between-valence contrasts should be easier (larger embedding
# distances from OBS-7) than within-valence contrasts.
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP-4: PAIRWISE CONTRAST ANALYSIS (6 pairs)")
print(f"{'='*70}")

pairwise_results = {}
for c1 in range(K):
    for c2 in range(c1+1, K):
        mask = (y==c1)|(y==c2)
        yp = (y[mask]==c2).astype(int)
        sp = subjects[mask]; Fp = F_rel[mask]
        us = np.unique(sp)
        sl = np.array([int(np.median(yp[sp==s])) for s in us])
        if len(np.unique(sl))<2: sl=np.zeros(len(us),dtype=int); sl[len(sl)//2:]=1
        pa = []
        skfp = StratifiedKFold(n_splits=min(N_FOLDS,len(us)//2), shuffle=True, random_state=RANDOM_STATE)
        for tri, tei in skfp.split(us, sl):
            trs=us[tri]; tes=us[tei]
            trm=np.isin(sp,trs); tem=np.isin(sp,tes)
            sc=StandardScaler(); Xtr=sc.fit_transform(Fp[trm]); Xte=sc.transform(Fp[tem])
            clf=SVC(kernel='rbf',class_weight='balanced',random_state=RANDOM_STATE)
            clf.fit(Xtr,yp[trm]); pa.append(balanced_accuracy_score(yp[tem],clf.predict(Xte)))
        pa = np.array(pa)
        key = f"{COND_NAMES[c1]} vs {COND_NAMES[c2]}"
        within = (c1 in [0,1] and c2 in [0,1]) or (c1 in [2,3] and c2 in [2,3])
        pairwise_results[key] = {'mean_acc':pa.mean(),'std_acc':pa.std(),
                                  'classes':(c1,c2),'within_valence':within}
        tag = '★ WITHIN' if within else '  between'
        print(f"  {tag}  {key:30s}: {pa.mean()*100:.1f}% ±{pa.std()*100:.1f}")

within_mean = np.mean([r['mean_acc'] for r in pairwise_results.values() if r['within_valence']])
between_mean = np.mean([r['mean_acc'] for r in pairwise_results.values() if not r['within_valence']])
print(f"\n  Within-valence mean:  {within_mean*100:.1f}%")
print(f"  Between-valence mean: {between_mean*100:.1f}%")
print(f"  Gap: {(between_mean-within_mean)*100:.1f}pp — quantifying the valence boundary")


# ═══════════════════════════════════════════════════════════════════════
# EXP-5: VARIANCE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP-5: VARIANCE DECOMPOSITION")
print(f"{'='*70}")

emb_flat = pca64.reshape(N_obs, -1)
gm = emb_flat.mean(axis=0)
SS_subj = sum(((subjects==s).sum() * np.sum((emb_flat[subjects==s].mean(0)-gm)**2))
              for s in unique_subjects)
SS_cond = sum(((y==c).sum() * np.sum((emb_flat[y==c].mean(0)-gm)**2)) for c in range(K))
SS_tot = np.sum((emb_flat - gm)**2)
SS_res = SS_tot - SS_subj - SS_cond

var_subj = SS_subj/SS_tot*100; var_cond = SS_cond/SS_tot*100; var_res = SS_res/SS_tot*100
print(f"  Subject:   {var_subj:.1f}%")
print(f"  Condition: {var_cond:.1f}%")
print(f"  Residual:  {var_res:.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# EXP-6: PER-CHANNEL DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXP-6: PER-CHANNEL DISCRIMINATION")
print(f"{'='*70}")

ch_accs = np.zeros(N_ch)
for ch in range(N_ch):
    r = evaluate_method(pca64[:,ch,:], f'Ch{ch}', verbose=False)
    ch_accs[ch] = r['mean_acc']
print(f"  Best:  Ch{np.argmax(ch_accs)} ({ch_accs.max()*100:.1f}%)")
print(f"  Worst: Ch{np.argmin(ch_accs)} ({ch_accs.min()*100:.1f}%)")
print(f"  Mean:  {ch_accs.mean()*100:.1f}%, Range: [{ch_accs.min()*100:.1f}, {ch_accs.max()*100:.1f}]")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║              CLINICAL INTERPRETABILITY TIER (EXP 7–11)               ║
# ╚═══════════════════════════════════════════════════════════════════════╝

if not clinical_available:
    print("\n  Clinical metadata not available. Skipping EXP-7 through EXP-11.")
else:
    sids = [int(s) for s in unique_subjects]

    # ═══════════════════════════════════════════════════════════════════
    # GRAPH-TOPOLOGICAL DESCRIPTORS (precompute for all subjects)
    # ═══════════════════════════════════════════════════════════════════
    def compute_graph_props(conn_mat):
        N = conn_mat.shape[0]
        tri = np.triu_indices(N, k=1)
        v = conn_mat[tri]; thr = np.percentile(v, 80)
        A = (conn_mat >= thr).astype(float); np.fill_diagonal(A, 0)
        deg = A.sum(axis=1)
        strength = np.abs(conn_mat[tri]).mean()
        # Clustering
        cc = np.zeros(N)
        for i in range(N):
            nb = np.where(A[i]>0)[0]; k=len(nb)
            if k>=2:
                links = sum(A[nb[a],nb[b]] for a in range(len(nb)) for b in range(a+1,len(nb)))
                cc[i] = 2*links/(k*(k-1))
        # Eigenvector centrality
        try:
            evals, evecs = np.linalg.eigh(A)
            ec = np.abs(evecs[:, -1])
        except:
            ec = np.zeros(N)
        # Global efficiency
        from scipy.sparse.csgraph import shortest_path
        dist = shortest_path(A, directed=False, unweighted=True)
        dist[dist == 0] = np.inf; np.fill_diagonal(dist, 0)
        inv_dist = 1.0 / dist; inv_dist[np.isinf(inv_dist)] = 0
        glob_eff = inv_dist.sum() / (N * (N-1))

        return {'degree': deg, 'clustering': cc, 'eigenvec_centrality': ec,
                'strength_per_ch': np.abs(conn_mat).sum(axis=1)/(N-1),
                'global_efficiency': glob_eff, 'mean_clustering': cc.mean(),
                'mean_strength': strength, 'degree_heterogeneity': deg.std()/max(deg.mean(),1e-10)}

    print("\nComputing graph properties for all subjects...")
    subj_graph_props = [compute_graph_props(subj_conn[si]) for si in range(N_subj)]

    # ═══════════════════════════════════════════════════════════════════
    # EXP-7: CHANNEL-LEVEL CLINICAL BIOMARKERS
    # Mathematical motivation: If a clinical condition alters the
    # functional organization of the brain, then the graph-topological
    # properties of specific channels should differ between clinical
    # groups. Each channel's degree, clustering, eigenvector centrality,
    # and strength provide distinct views of its network role.
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("EXP-7: CHANNEL-LEVEL CLINICAL BIOMARKERS")
    print(f"{'='*70}")

    clinical_vars = {
        'MDD': 'Major Depressive Disorder',
        'PTSD': 'PTSD',
        'GAD': 'Generalized Anxiety',
        'SUD': 'Substance Use Disorder',
        'ADHD': 'ADHD',
    }
    # Add sex and medication if available
    if 'Assigned_Sex' in df_clin.columns:
        clinical_vars['Assigned_Sex'] = 'Biological Sex'
    if 'Psychiatric_Medication' in df_clin.columns:
        clinical_vars['Psychiatric_Medication'] = 'Medication Status'

    graph_metrics = ['degree', 'clustering', 'eigenvec_centrality', 'strength_per_ch']
    channel_biomarkers = {}

    for var, label in clinical_vars.items():
        vals = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        valid = ~np.isnan(vals)
        if valid.sum() < 20:
            print(f"  {label}: too few valid ({valid.sum()}), skipping")
            continue

        if var == 'Assigned_Sex':
            g1 = vals[valid] == 1  # Male
            g2 = vals[valid] == 2  # Female
        else:
            g1 = vals[valid] == 0
            g2 = vals[valid] == 1

        n1, n2 = g1.sum(), g2.sum()
        if min(n1, n2) < 10:
            print(f"  {label}: group too small ({n1} vs {n2}), skipping")
            continue

        var_results = {}
        sig_count = 0
        for metric in graph_metrics:
            metric_vals = np.array([subj_graph_props[si][metric] for si in range(N_subj)])
            metric_valid = metric_vals[valid]

            for ch in range(N_ch):
                if metric_vals.ndim == 1:
                    continue  # graph-level metric, skip channel loop
                v1 = metric_valid[g1, ch]
                v2 = metric_valid[g2, ch]
                try:
                    _, p = mannwhitneyu(v1, v2, alternative='two-sided')
                except:
                    p = 1.0
                d_cohen = (v2.mean() - v1.mean()) / np.sqrt((v1.std()**2 + v2.std()**2)/2 + 1e-10)
                if p < 0.05:
                    sig_count += 1
                    var_results[(metric, ch)] = {'d': d_cohen, 'p': p,
                                                  'mean_g1': v1.mean(), 'mean_g2': v2.mean()}

        channel_biomarkers[var] = var_results
        print(f"  {label:<25s}: n={n1}vs{n2}, {sig_count} significant channel-metric pairs")

        # Report top findings
        if var_results:
            top = sorted(var_results.items(), key=lambda x: x[1]['p'])[:3]
            for (metric, ch), res in top:
                print(f"    Ch{ch:2d} {metric:20s}: d={res['d']:+.2f}, p={res['p']:.4f}")


    # ═══════════════════════════════════════════════════════════════════
    # EXP-8: CONDITION × CLINICAL STATUS INTERACTIONS
    # Mathematical motivation: The most clinically informative signal
    # is not a baseline group difference but a condition-dependent
    # interaction: how does emotional processing CHANGE the network
    # differently in clinical vs control groups?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("EXP-8: CONDITION × CLINICAL STATUS INTERACTIONS")
    print(f"{'='*70}")

    # Per-subject, per-condition graph properties
    subj_cond_props = {}
    for si, s in enumerate(unique_subjects):
        for c in range(K):
            mask = (subjects == s) & (y == c)
            if mask.sum() > 0:
                cm_sc = conn_all[mask].mean(axis=0)
                subj_cond_props[(si, c)] = compute_graph_props(cm_sc)

    interaction_results = {}
    test_vars = ['MDD', 'PTSD', 'GAD', 'SUD']
    if 'Psychiatric_Medication' in clinical_vars:
        test_vars.append('Psychiatric_Medication')

    graph_level_metrics = ['global_efficiency', 'mean_clustering', 'mean_strength']

    for var in test_vars:
        vals = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        valid = ~np.isnan(vals)
        if var == 'Assigned_Sex':
            g0 = (vals == 1) & valid; g1_mask = (vals == 2) & valid
        else:
            g0 = (vals == 0) & valid; g1_mask = (vals == 1) & valid

        for metric in graph_level_metrics:
            # Compute reactivity: negative conditions minus positive conditions
            # (Threat+Mutilation)/2 - (Cute+Erotic)/2
            react_g0, react_g1 = [], []
            for si in range(N_subj):
                if not valid[si]:
                    continue
                neg_props = []
                pos_props = []
                for c in [0, 1]:  # Threat, Mutilation
                    if (si, c) in subj_cond_props:
                        neg_props.append(subj_cond_props[(si, c)][metric])
                for c in [2, 3]:  # Cute, Erotic
                    if (si, c) in subj_cond_props:
                        pos_props.append(subj_cond_props[(si, c)][metric])
                if neg_props and pos_props:
                    reactivity = np.mean(neg_props) - np.mean(pos_props)
                    if g0[si]:
                        react_g0.append(reactivity)
                    elif g1_mask[si]:
                        react_g1.append(reactivity)

            react_g0 = np.array(react_g0)
            react_g1 = np.array(react_g1)

            if len(react_g0) >= 10 and len(react_g1) >= 10:
                try:
                    _, p = mannwhitneyu(react_g0, react_g1, alternative='two-sided')
                except:
                    p = 1.0
                d_cohen = (react_g1.mean()-react_g0.mean()) / \
                          np.sqrt((react_g0.std()**2+react_g1.std()**2)/2+1e-10)
                sig = '*' if p < 0.05 else ''
                print(f"  {var:20s} × {metric:20s}: Δ_g0={react_g0.mean():+.4f}, "
                      f"Δ_g1={react_g1.mean():+.4f}, d={d_cohen:+.2f}, p={p:.4f} {sig}")
                interaction_results[(var, metric)] = {
                    'd': d_cohen, 'p': p,
                    'react_g0': react_g0.mean(), 'react_g1': react_g1.mean(),
                    'n_g0': len(react_g0), 'n_g1': len(react_g1)
                }


    # ═══════════════════════════════════════════════════════════════════
    # EXP-9: EDGE-LEVEL BIOMARKERS
    # Mathematical motivation: Graph-level and channel-level descriptors
    # aggregate over many connections. Edge-level testing asks whether
    # specific channel pairs are disproportionately affected by clinical
    # conditions — testing 561 edges per variable.
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("EXP-9: EDGE-LEVEL BIOMARKERS")
    print(f"{'='*70}")

    edge_results = {}
    for var in ['MDD', 'PTSD', 'GAD', 'SUD']:
        vals = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        valid = ~np.isnan(vals)
        g0 = (vals == 0) & valid
        g1_mask = (vals == 1) & valid

        if g0.sum() < 10 or g1_mask.sum() < 10:
            continue

        sig_edges = 0
        max_d, max_edge, min_p = 0, None, 1
        for ei in range(len(triu[0])):
            i, j = triu[0][ei], triu[1][ei]
            e0 = subj_conn[g0, i, j]
            e1 = subj_conn[g1_mask, i, j]
            try:
                _, p = mannwhitneyu(e0, e1, alternative='two-sided')
            except:
                p = 1.0
            d = (e1.mean()-e0.mean()) / np.sqrt((e0.std()**2+e1.std()**2)/2+1e-10)
            if p < 0.05:
                sig_edges += 1
            if np.abs(d) > np.abs(max_d):
                max_d = d; max_edge = (i,j); min_p = p

        expected = 561 * 0.05
        edge_results[var] = {'sig_edges': sig_edges, 'expected': expected,
                              'max_d': max_d, 'max_edge': max_edge, 'min_p': min_p}
        ratio = sig_edges / expected if expected > 0 else 0
        print(f"  {var:10s}: {sig_edges}/{561} significant edges "
              f"({ratio:.1f}× expected={expected:.0f}), "
              f"max |d|={np.abs(max_d):.2f} at Ch{max_edge[0]}↔Ch{max_edge[1]} "
              f"(p={min_p:.4f})")


    # ═══════════════════════════════════════════════════════════════════
    # EXP-10: COMORBIDITY BURDEN VS NETWORK COMPLEXITY
    # Mathematical motivation: Transdiagnostic models predict that
    # increasing diagnostic burden should be associated with systematic
    # changes in network architecture — either increased uniformity
    # (loss of functional differentiation) or increased segregation.
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("EXP-10: COMORBIDITY BURDEN VS NETWORK COMPLEXITY")
    print(f"{'='*70}")

    dx_cols = ['MDD', 'GAD', 'PTSD', 'SUD', 'ADHD', 'EAT', 'Mania', 'PDD']
    comorbidity = np.array([
        sum(1 for c in dx_cols if clin_map.get(sid, {}).get(c, 0) == 1)
        for sid in sids])

    from scipy.stats import spearmanr
    comorbidity_results = {}
    for prop in ['global_efficiency', 'mean_clustering', 'mean_strength', 'degree_heterogeneity']:
        vals = np.array([subj_graph_props[si][prop] for si in range(N_subj)])
        rho, p = spearmanr(comorbidity, vals)
        comorbidity_results[prop] = {'rho': rho, 'p': p}
        sig = '*' if p < 0.05 else ''
        print(f"  {prop:25s}: ρ={rho:+.3f}, p={p:.4f} {sig}")


    # ═══════════════════════════════════════════════════════════════════
    # EXP-11: WITHIN-VALENCE × CLINICAL INTERACTIONS (4-CLASS SPECIFIC)
    # Mathematical motivation: The 4-class design enables a question
    # impossible in the 3-class: do clinical conditions modulate the
    # WITHIN-valence contrast? E.g., does MDD alter the brain's
    # differential response to threat vs mutilation specifically?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("EXP-11: WITHIN-VALENCE × CLINICAL INTERACTIONS (4-CLASS SPECIFIC)")
    print(f"{'='*70}")

    within_interaction_results = {}
    for var in ['MDD', 'PTSD', 'GAD', 'SUD']:
        vals = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        valid = ~np.isnan(vals)
        g0 = (vals == 0) & valid
        g1_mask = (vals == 1) & valid

        if g0.sum() < 10 or g1_mask.sum() < 10:
            continue

        for pair_name, c_a, c_b in [('Threat−Mutilation', 0, 1), ('Cute−Erotic', 2, 3)]:
            for metric in graph_level_metrics:
                delta_g0, delta_g1 = [], []
                for si in range(N_subj):
                    if not valid[si]: continue
                    if (si, c_a) in subj_cond_props and (si, c_b) in subj_cond_props:
                        d = subj_cond_props[(si, c_a)][metric] - subj_cond_props[(si, c_b)][metric]
                        if g0[si]: delta_g0.append(d)
                        elif g1_mask[si]: delta_g1.append(d)

                delta_g0 = np.array(delta_g0)
                delta_g1 = np.array(delta_g1)
                if len(delta_g0) >= 10 and len(delta_g1) >= 10:
                    try:
                        _, p = mannwhitneyu(delta_g0, delta_g1, alternative='two-sided')
                    except:
                        p = 1.0
                    d_cohen = (delta_g1.mean()-delta_g0.mean()) / \
                              np.sqrt((delta_g0.std()**2+delta_g1.std()**2)/2+1e-10)
                    sig = '*' if p < 0.05 else ''
                    if p < 0.10:  # report trends
                        print(f"  {var:8s} × {pair_name:18s} × {metric:20s}: "
                              f"d={d_cohen:+.2f}, p={p:.4f} {sig}")
                    within_interaction_results[(var, pair_name, metric)] = {
                        'd': d_cohen, 'p': p}


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                         FIGURE GENERATION                            ║
# ╚═══════════════════════════════════════════════════════════════════════╝
print(f"\n{'='*70}")
print("GENERATING FIGURES")
print(f"{'='*70}")

# ── Figure 1: Main comparison bar chart ──
fig, ax = plt.subplots(figsize=(14, 6))
methods = ['BandPower\n+SVM', 'PCA-64\n+SVM', 'Relational\n+SVM',
           'GIN', 'ChebNet', 'EdgeConv', 'Transformer']
accs_p = [res_m1['mean_acc']*100, res_m2['mean_acc']*100, res_m3['mean_acc']*100]
stds_p = [res_m1['std_acc']*100, res_m2['std_acc']*100, res_m3['std_acc']*100]
for a in ['gin','chebnet','edgeconv','transformer']:
    accs_p.append(gnn_results[a]['mean_acc']*100)
    stds_p.append(gnn_results[a]['std_acc']*100)
colors = ['#7f8c8d','#7f8c8d','#27ae60','#e74c3c','#e74c3c','#e74c3c','#e74c3c']
ax.bar(range(len(methods)), accs_p, yerr=stds_p, capsize=4,
       color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
ax.axhline(25, color='gray', linestyle=':', linewidth=1, label='Chance (25%)')
ax.axhline(res_m3['mean_acc']*100, color='#27ae60', linestyle='--', linewidth=1.5,
           alpha=0.5, label=f'Relational ({res_m3["mean_acc"]*100:.1f}%)')
ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Balanced Accuracy (%)', fontsize=12)
ax.set_title(f'4-Class Classification: {N_subj} subjects, {N_FOLDS}-fold subject-level CV\n'
             f'Threat / Mutilation / Cute / Erotic', fontsize=13, fontweight='bold')
ax.set_ylim(15, max(accs_p)+15); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
for i in range(3,7):
    a = ['gin','chebnet','edgeconv','transformer'][i-3]
    if gnn_results[a]['p_vs_m3'] < 0.05:
        ax.annotate('*', (i, accs_p[i]+stds_p[i]+1.5), ha='center', fontsize=14, color='red')
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig01_gnn_comparison.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig01_gnn_comparison.pdf")

# ── Figure 2: Confusion matrix ──
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues', aspect='equal')
for i in range(K):
    for j in range(K):
        ax.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > cm.max()*0.5 else 'black')
ax.set_xticks(range(K)); ax.set_yticks(range(K))
ax.set_xticklabels(labels, fontsize=11); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
ax.set_title(f'Confusion Matrix — Relational SVM ({res_m3["mean_acc"]*100:.1f}%)',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig02_confusion_matrix.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig02_confusion_matrix.pdf")

# ── Figure 3: Pairwise contrasts ──
fig, ax = plt.subplots(figsize=(10, 5))
pk = sorted(pairwise_results.keys(), key=lambda k: pairwise_results[k]['mean_acc'])
pa_p = [pairwise_results[k]['mean_acc']*100 for k in pk]
ps_p = [pairwise_results[k]['std_acc']*100 for k in pk]
pc = ['#e74c3c' if pairwise_results[k]['within_valence'] else '#2980b9' for k in pk]
ax.barh(range(len(pk)), pa_p, xerr=ps_p, color=pc, edgecolor='black',
        linewidth=0.5, capsize=3)
ax.axvline(50, color='gray', linestyle=':', linewidth=1)
ax.set_yticks(range(len(pk))); ax.set_yticklabels(pk, fontsize=10)
ax.set_xlabel('Balanced Accuracy (%)', fontsize=11)
ax.set_title('Pairwise Contrasts: Red=within-valence, Blue=between-valence',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig03_pairwise_contrasts.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig03_pairwise_contrasts.pdf")

# ── Figure 4: Variance decomposition + per-channel ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
sizes = [var_subj, var_cond, var_res]
ax.pie(sizes, labels=[f'Subject\n{var_subj:.1f}%', f'Condition\n{var_cond:.1f}%',
       f'Residual\n{var_res:.1f}%'], colors=['#3498db','#e74c3c','#95a5a6'],
       autopct='%1.1f%%', startangle=90, textprops={'fontsize':11})
ax.set_title('Variance Decomposition', fontsize=12, fontweight='bold')

ax = axes[1]
ax.bar(range(N_ch), ch_accs*100, color=plt.cm.viridis((ch_accs-ch_accs.min())/(ch_accs.max()-ch_accs.min()+1e-10)),
       edgecolor='black', linewidth=0.3)
ax.axhline(25, color='gray', linestyle=':', linewidth=1, label='Chance')
ax.axhline(ch_accs.mean()*100, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean ({ch_accs.mean()*100:.1f}%)')
ax.set_xlabel('Channel', fontsize=11); ax.set_ylabel('Bal. Acc. (%)', fontsize=11)
ax.set_title('Per-Channel 4-Class Discrimination', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_xticks(range(0,34,5)); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig04_variance_channels.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig04_variance_channels.pdf")

# ── Figure 5: Clinical biomarker summary (if available) ──
if clinical_available and channel_biomarkers:
    n_vars = min(len(channel_biomarkers), 6)
    fig, axes = plt.subplots(1, n_vars, figsize=(4*n_vars, 4))
    if n_vars == 1: axes = [axes]
    for ax_i, (var, results) in enumerate(list(channel_biomarkers.items())[:n_vars]):
        ax = axes[ax_i]
        # Summarize: for each channel, take the max |d| across metrics
        ch_max_d = np.zeros(N_ch)
        ch_sig = np.zeros(N_ch, dtype=bool)
        for (metric, ch), res in results.items():
            if np.abs(res['d']) > np.abs(ch_max_d[ch]):
                ch_max_d[ch] = res['d']
            if res['p'] < 0.05:
                ch_sig[ch] = True

        theta = np.linspace(0, 2*np.pi, N_ch, endpoint=False)
        xp, yp = np.cos(theta), np.sin(theta)
        sizes = np.where(ch_sig, 120, 30)
        sc = ax.scatter(xp, yp, c=ch_max_d, cmap='RdBu_r', s=sizes,
                        edgecolors='black', linewidths=0.8, vmin=-0.8, vmax=0.8)
        for ch in range(N_ch):
            if ch_sig[ch]:
                ax.annotate(f'{ch}', (xp[ch], yp[ch]), fontsize=6,
                            ha='center', va='center')
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect('equal'); ax.axis('off')
        label = clinical_vars.get(var, var)
        ax.set_title(f'{label}\n({ch_sig.sum()} sig channels)', fontsize=10, fontweight='bold')
    fig.suptitle('Channel-Level Network Biomarkers\nNode color=Cohen d, size=significance',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig05_channel_biomarkers.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → fig05_channel_biomarkers.pdf")

# ── Figure 6: Interaction effects (if significant) ──
if clinical_available and interaction_results:
    sig_interactions = {k: v for k, v in interaction_results.items() if v['p'] < 0.10}
    if sig_interactions:
        n_int = min(len(sig_interactions), 4)
        fig, axes = plt.subplots(1, n_int, figsize=(5*n_int, 4.5))
        if n_int == 1: axes = [axes]
        for ax_i, ((var, metric), res) in enumerate(list(sig_interactions.items())[:n_int]):
            ax = axes[ax_i]
            ax.bar([0, 1], [res['react_g0'], res['react_g1']],
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=0.8, width=0.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'No {var}\n(n={res["n_g0"]})',
                                f'{var}\n(n={res["n_g1"]})'], fontsize=9)
            ax.set_ylabel('Neg−Pos Reactivity', fontsize=10)
            ax.axhline(0, color='black', linewidth=0.5)
            sig = '*' if res['p'] < 0.05 else '†'
            ax.set_title(f'{var} × {metric}\nd={res["d"]:+.2f}, p={res["p"]:.3f}{sig}',
                         fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        fig.suptitle('Condition × Clinical Status Interaction Effects',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/fig06_interactions.pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  → fig06_interactions.pdf")

# ── Figure 7: Edge biomarkers ──
if clinical_available and edge_results:
    n_vars = min(len(edge_results), 4)
    fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4))
    if n_vars == 1: axes = [axes]
    for ax_i, (var, res) in enumerate(list(edge_results.items())[:n_vars]):
        ax = axes[ax_i]
        vals_v = np.array([clin_map.get(sid, {}).get(var, np.nan) for sid in sids])
        valid = ~np.isnan(vals_v)
        g0 = (vals_v == 0) & valid; g1m = (vals_v == 1) & valid
        diff_mat = subj_conn[g1m].mean(axis=0) - subj_conn[g0].mean(axis=0)
        np.fill_diagonal(diff_mat, np.nan)
        vmax = max(np.nanmax(np.abs(diff_mat)), 0.05)
        ax.imshow(diff_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title(f'{var}\n({res["sig_edges"]} sig edges, '
                     f'{res["sig_edges"]/res["expected"]:.1f}× expected)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Ch'); ax.set_ylabel('Ch' if ax_i==0 else '')
    fig.suptitle('Edge-Level Connectivity Differences by Clinical Variable',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig07_edge_biomarkers.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → fig07_edge_biomarkers.pdf")

# ── Figure 8: Comorbidity ──
if clinical_available:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax_i, (prop, label) in enumerate([
        ('global_efficiency', 'Global Efficiency'),
        ('mean_clustering', 'Mean Clustering'),
        ('mean_strength', 'Mean Strength'),
        ('degree_heterogeneity', 'Degree Heterogeneity')]):
        ax = axes[ax_i]
        vals = np.array([subj_graph_props[si][prop] for si in range(N_subj)])
        ax.scatter(comorbidity + np.random.randn(N_subj)*0.1, vals,
                   s=15, alpha=0.5, color='steelblue')
        z = np.polyfit(comorbidity, vals, 1)
        xr = np.linspace(0, comorbidity.max(), 100)
        ax.plot(xr, np.polyval(z, xr), 'r-', linewidth=2)
        res = comorbidity_results.get(prop, {})
        ax.set_xlabel('N Diagnoses', fontsize=10)
        ax.set_ylabel(label if ax_i==0 else '', fontsize=10)
        ax.set_title(f'{label}\nρ={res.get("rho",0):+.3f}, p={res.get("p",1):.3f}',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
    fig.suptitle('Network Complexity vs Comorbidity Burden',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig08_comorbidity.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  → fig08_comorbidity.pdf")

# ── Figure 9: Fold comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
fx = np.arange(N_FOLDS); w = 0.18
ax.bar(fx-1.5*w, res_m1['bal_accs']*100, w, label='BandPower', color='#7f8c8d')
ax.bar(fx-0.5*w, res_m2['bal_accs']*100, w, label='PCA-64', color='#3498db')
ax.bar(fx+0.5*w, res_m3['bal_accs']*100, w, label='Relational', color='#27ae60')
best_gnn = max(gnn_results, key=lambda k: gnn_results[k]['mean_acc'])
ax.bar(fx+1.5*w, gnn_results[best_gnn]['bal_accs']*100, w,
       label=f'Best GNN ({best_gnn})', color='#e74c3c')
ax.axhline(25, color='gray', linestyle=':')
ax.set_xlabel('Fold', fontsize=11); ax.set_ylabel('Bal. Acc. (%)', fontsize=11)
ax.set_title('Per-Fold Accuracy', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_xticks(fx)
ax.set_xticklabels([f'F{i+1}' for i in range(N_FOLDS)]); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/fig09_fold_comparison.pdf', bbox_inches='tight', dpi=150)
plt.close()
print(f"  → fig09_fold_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════
all_results = {
    'baselines': {'M1': res_m1, 'M2': res_m2, 'M3': res_m3},
    'gnn_results': gnn_results,
    'confusion_matrix': cm,
    'pairwise_results': pairwise_results,
    'variance_decomposition': {'subject': var_subj, 'condition': var_cond, 'residual': var_res},
    'per_channel_accuracy': ch_accs,
    'channel_biomarkers': channel_biomarkers if clinical_available else {},
    'interaction_results': interaction_results if clinical_available else {},
    'edge_results': edge_results if clinical_available else {},
    'comorbidity_results': comorbidity_results if clinical_available else {},
    'within_interaction_results': within_interaction_results if clinical_available else {},
    'config': {'n_folds': N_FOLDS, 'n_subjects': N_subj, 'n_obs': N_obs,
               'n_classes': K, 'chance': 1/K},
}

with open(RESULTS_FILE, 'wb') as f:
    pickle.dump(all_results, f, protocol=4)


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("COMPLETE EXPERIMENTAL SUMMARY")
print(f"{'='*70}")
print(f"\n  Dataset: {N_subj} subjects × 4 categories = {N_obs} observations")
print(f"  Chance: {1/K*100:.1f}%")

print(f"\n  ── CLASSIFICATION ──")
print(f"  M1 BandPower:   {res_m1['mean_acc']*100:.1f}% ±{res_m1['std_acc']*100:.1f}")
print(f"  M2 PCA-64:      {res_m2['mean_acc']*100:.1f}% ±{res_m2['std_acc']*100:.1f}")
print(f"  M3 Relational:  {res_m3['mean_acc']*100:.1f}% ±{res_m3['std_acc']*100:.1f}")
for a in ['gin','chebnet','edgeconv','transformer']:
    r = gnn_results[a]
    print(f"  {a.upper():12s}:  {r['mean_acc']*100:.1f}% (p vs M3 = {r['p_vs_m3']:.4f})")

print(f"\n  ── PAIRWISE CONTRASTS ──")
print(f"  Within-valence mean:  {within_mean*100:.1f}%")
print(f"  Between-valence mean: {between_mean*100:.1f}%")
print(f"  Valence boundary gap: {(between_mean-within_mean)*100:.1f}pp")

print(f"\n  ── VARIANCE ──")
print(f"  Subject: {var_subj:.1f}%  Condition: {var_cond:.1f}%  Residual: {var_res:.1f}%")

if clinical_available:
    print(f"\n  ── CLINICAL INTERPRETABILITY ──")
    total_sig = sum(len(v) for v in channel_biomarkers.values())
    print(f"  Channel biomarkers: {total_sig} significant channel-metric pairs")
    sig_int = sum(1 for v in interaction_results.values() if v['p'] < 0.05)
    print(f"  Condition × clinical interactions: {sig_int} significant")
    sig_within = sum(1 for v in within_interaction_results.values() if v['p'] < 0.05)
    print(f"  Within-valence × clinical (4-class specific): {sig_within} significant")

n_figs = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pdf')])
print(f"\n  Figures: {n_figs} PDFs in {OUTPUT_DIR}/")
print(f"  Results: {RESULTS_FILE}")

print(f"\n  ── 11 EXPERIMENTS COMPLETE ──")
print(f"  EXP 1-6:  Classification tier")
print(f"  EXP 7-11: Clinical interpretability tier")
print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
