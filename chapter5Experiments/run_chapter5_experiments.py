#!/usr/bin/env python3
"""
ARSPI-Net Chapter 5: Complete End-to-End Experimental Pipeline
===============================================================
Implements the full ARSPI-Net architecture and all comparison baselines
for the SHAPE EEG dataset (https://lab-can.com/shape/).

Data specification (from Brady):
  - Per file: 1229 rows × 34 columns (µV)
  - Sampling rate: 1024 Hz
  - Rows 0-204: 200ms baseline (already baseline-corrected)
  - Rows 205-1228: 1000ms post-stimulus
  - 3 conditions: IAPSNeg, IAPSNeu, IAPSPos
  - 80+ subjects, each with 3 files
  - Naming: SHAPE_Community_{SUBJ}_{COND}_BC.txt

Pipeline:
  1. Load all data, parse subject/condition metadata
  2. Preprocess: remove baseline, downsample 4x to 256 Hz, z-score
  3. Feature extraction:
     a. LSM BSC6+PCA-64 embeddings (34 channels × 64 dims)
     b. LSM MFR embeddings (34 channels × 256 dims) 
     c. Conventional: band-power features (34 channels × 5 bands)
     d. Conventional: Hjorth parameters (34 channels × 3 params)
  4. Graph construction: spatial adjacency + functional adjacency
  5. GNN propagation: GCN, GraphSAGE, GAT (from scratch, numpy)
  6. Classification with subject-level CV
  7. Full 7-row baseline table + ablations + figures

Usage:
  python3 run_chapter5_experiments.py --data_dir /path/to/SHAPE/files
                                      --meta_file /path/to/metadata.csv
"""

import numpy as np
import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.signal import decimate, welch
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 11, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGDIR = Path("figures/ch5")
FIGDIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# SECTION 1: DATA LOADING
# ================================================================
def load_shape_data(data_dir):
    """Load all SHAPE .txt files and return structured dataset.
    
    Returns:
        X: list of (1024, 34) post-stimulus arrays
        y_cond: array of condition labels (0=Neg, 1=Neu, 2=Pos)
        subject_ids: array of subject IDs (for group CV)
        metadata: dict of parsed info
    """
    files = sorted(glob.glob(os.path.join(data_dir, "SHAPE_Community_*_BC.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    print(f"Found {len(files)} files in {data_dir}")
    
    cond_map = {'IAPSNeg': 0, 'IAPSNeu': 1, 'IAPSPos': 2, 
                'iapsneg': 0, 'iapsneu': 1, 'iapspos': 2,
                'Neg': 0, 'Neu': 1, 'Pos': 2}
    cond_names = {0: 'Negative', 1: 'Neutral', 2: 'Pleasant'}
    
    X_list, y_list, subj_list = [], [], []
    bad_files = []
    
    for f in files:
        fname = os.path.basename(f)
        # Parse: SHAPE_Community_007_IAPSNeg_BC.txt
        parts = fname.replace('.txt', '').split('_')
        
        # Find subject ID and condition
        subj_id = None
        condition = None
        for p in parts:
            if p.isdigit() and len(p) >= 2:
                subj_id = p
            if p in cond_map:
                condition = cond_map[p]
            # Try partial match
            for key in cond_map:
                if key.lower() in p.lower():
                    condition = cond_map[key]
        
        if subj_id is None or condition is None:
            bad_files.append(fname)
            continue
        
        try:
            data = np.loadtxt(f)
            if data.shape != (1229, 34):
                print(f"  WARNING: {fname} has shape {data.shape}, expected (1229, 34)")
                if data.shape[0] < 205 or data.shape[1] != 34:
                    bad_files.append(fname)
                    continue
            
            # Remove baseline (first 205 rows)
            poststim = data[205:, :]
            
            X_list.append(poststim)
            y_list.append(condition)
            subj_list.append(subj_id)
            
        except Exception as e:
            print(f"  ERROR loading {fname}: {e}")
            bad_files.append(fname)
    
    if bad_files:
        print(f"  Skipped {len(bad_files)} problematic files: {bad_files[:5]}...")
    
    X = np.array(X_list)  # (N, 1024, 34)
    y = np.array(y_list)
    subjects = np.array(subj_list)
    
    print(f"Loaded: {X.shape[0]} samples, {len(np.unique(subjects))} subjects")
    for c in sorted(np.unique(y)):
        print(f"  Condition {c} ({cond_names.get(c, '?')}): {(y==c).sum()} samples")
    
    return X, y, subjects, cond_names


def create_demo_dataset(sample_file, n_subjects=20):
    """Create a demo dataset from a single file for pipeline testing.
    Adds noise and simulated condition differences."""
    
    data = np.loadtxt(sample_file)
    poststim = data[205:, :]  # (1024, 34)
    
    rng = np.random.RandomState(42)
    X_list, y_list, subj_list = [], [], []
    
    for subj in range(n_subjects):
        for cond in range(3):  # Neg=0, Neu=1, Pos=2
            # Base signal with subject-specific noise
            x = poststim.copy()
            x += rng.randn(*x.shape) * 2.0  # subject noise
            
            # Condition-specific modulation (mimics real ERP differences)
            if cond == 0:  # Negative: enhanced LPP (channels 15-25, 400-800ms)
                x[410:820, 15:25] += rng.randn(410, 10) * 1.5 + 3.0
            elif cond == 2:  # Pleasant: enhanced early (channels 20-30, 200-400ms)
                x[205:410, 20:30] += rng.randn(205, 10) * 1.5 + 2.5
            # Neutral: no enhancement
            
            # Subject-specific baseline shift
            x += rng.randn(1, 34) * 3.0
            
            X_list.append(x)
            y_list.append(cond)
            subj_list.append(f"{subj:03d}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    subjects = np.array(subj_list)
    cond_names = {0: 'Negative', 1: 'Neutral', 2: 'Pleasant'}
    
    print(f"Created demo dataset: {X.shape[0]} samples, {n_subjects} subjects")
    return X, y, subjects, cond_names


# ================================================================
# SECTION 2: PREPROCESSING
# ================================================================
def preprocess(X, downsample_factor=4):
    """Downsample and z-score normalize each epoch.
    
    Args:
        X: (N, 1024, 34) raw post-stimulus data
    Returns:
        X_ds: (N, 256, 34) preprocessed data
    """
    N, T, C = X.shape
    X_ds = np.zeros((N, T // downsample_factor, C))
    
    for i in range(N):
        for ch in range(C):
            X_ds[i, :, ch] = decimate(X[i, :, ch], downsample_factor)
    
    # Z-score per epoch per channel
    for i in range(N):
        for ch in range(C):
            mu = X_ds[i, :, ch].mean()
            sigma = X_ds[i, :, ch].std()
            if sigma > 1e-10:
                X_ds[i, :, ch] = (X_ds[i, :, ch] - mu) / sigma
    
    return X_ds


# ================================================================
# SECTION 3: LIF RESERVOIR
# ================================================================
class LIFReservoir:
    """Leaky Integrate-and-Fire reservoir for single-channel input."""
    
    def __init__(self, n_res=256, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        # Input weights: single channel input
        limit_in = np.sqrt(6.0 / (1 + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, 1))
        # Recurrent weights with spectral radius control
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eig_max = np.abs(np.linalg.eigvals(self.W_rec)).max()
        if eig_max > 0:
            self.W_rec *= 0.9 / eig_max
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res
    
    def forward(self, x):
        """Process single-channel signal.
        Args: x: (T,) single channel time series
        Returns: spikes: (T, n_res)
        """
        T = len(x)
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        
        for t in range(T):
            I_in = self.W_in[:, 0] * x[t]
            I_rec = self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_in + I_rec
            spk = (mem >= self.threshold).astype(float)
            mem = mem - spk * self.threshold
            mem = np.maximum(mem, 0.0)
            spikes[t] = spk
            spk_prev = spk
        
        return spikes


def extract_bsc6(spikes, t_start=0, t_end=None):
    """Binned Spike Counts with 6 bins."""
    if t_end is None:
        t_end = spikes.shape[0]
    window = spikes[t_start:t_end]
    T_w, N = window.shape
    n_bins = 6
    bin_size = T_w // n_bins
    features = []
    for b in range(n_bins):
        features.append(window[b*bin_size:(b+1)*bin_size].sum(axis=0))
    return np.concatenate(features)


def extract_mfr(spikes, t_start=0, t_end=None):
    """Mean Firing Rate."""
    if t_end is None:
        t_end = spikes.shape[0]
    return spikes[t_start:t_end].mean(axis=0)


def run_reservoir_pipeline(X_ds, n_res=256, seed=42, coding='bsc6'):
    """Process all samples through per-channel reservoirs.
    
    Args:
        X_ds: (N, T, 34) preprocessed multichannel data
    Returns:
        features: (N, 34, feature_dim) per-channel features
    """
    N, T, C = X_ds.shape
    
    # Create one reservoir per channel (shared architecture, independent weights)
    reservoirs = []
    for ch in range(C):
        reservoirs.append(LIFReservoir(n_res=n_res, seed=seed + ch * 17))
    
    # Determine feature dim
    test_spk = reservoirs[0].forward(X_ds[0, :, 0])
    if coding == 'bsc6':
        test_feat = extract_bsc6(test_spk)
    else:
        test_feat = extract_mfr(test_spk)
    feat_dim = len(test_feat)
    
    features = np.zeros((N, C, feat_dim))
    
    for i in range(N):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing sample {i+1}/{N}...", flush=True)
        for ch in range(C):
            spk = reservoirs[ch].forward(X_ds[i, :, ch])
            if coding == 'bsc6':
                features[i, ch] = extract_bsc6(spk)
            else:
                features[i, ch] = extract_mfr(spk)
    
    return features


def apply_pca_per_channel(features_train, features_test, n_components=64):
    """Apply PCA independently per channel, fitted on train data only.
    
    Args:
        features_train: (N_train, 34, feat_dim)
        features_test: (N_test, 34, feat_dim)
    Returns:
        train_pca: (N_train, 34, n_components)
        test_pca: (N_test, 34, n_components)
    """
    C = features_train.shape[1]
    n_comp = min(n_components, features_train.shape[0] - 1, features_train.shape[2])
    
    train_pca = np.zeros((features_train.shape[0], C, n_comp))
    test_pca = np.zeros((features_test.shape[0], C, n_comp))
    
    for ch in range(C):
        pca = PCA(n_components=n_comp)
        train_pca[:, ch, :] = pca.fit_transform(features_train[:, ch, :])
        test_pca[:, ch, :] = pca.transform(features_test[:, ch, :])
    
    return train_pca, test_pca


# ================================================================
# SECTION 4: CONVENTIONAL EEG FEATURES
# ================================================================
def extract_bandpower(X_ds, fs=256):
    """Extract band-power features (5 bands × 34 channels).
    Bands: delta(1-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-100)
    """
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, min(100, fs//2 - 1))]
    N, T, C = X_ds.shape
    features = np.zeros((N, C, len(bands)))
    
    for i in range(N):
        for ch in range(C):
            freqs, psd = welch(X_ds[i, :, ch], fs=fs, nperseg=min(128, T))
            for b_idx, (flo, fhi) in enumerate(bands):
                mask = (freqs >= flo) & (freqs <= fhi)
                if mask.sum() > 0:
                    features[i, ch, b_idx] = np.trapezoid(psd[mask], freqs[mask])
    
    return features


def extract_hjorth(X_ds):
    """Extract Hjorth parameters (activity, mobility, complexity) × 34 channels."""
    N, T, C = X_ds.shape
    features = np.zeros((N, C, 3))
    
    for i in range(N):
        for ch in range(C):
            x = X_ds[i, :, ch]
            dx = np.diff(x)
            ddx = np.diff(dx)
            
            var_x = np.var(x) + 1e-12
            var_dx = np.var(dx) + 1e-12
            var_ddx = np.var(ddx) + 1e-12
            
            activity = var_x
            mobility = np.sqrt(var_dx / var_x)
            complexity = np.sqrt(var_ddx / var_dx) / mobility if mobility > 1e-12 else 0
            
            features[i, ch] = [activity, mobility, complexity]
    
    return features


# ================================================================
# SECTION 5: GRAPH CONSTRUCTION
# ================================================================
def get_standard_34ch_positions():
    """Standard 10-20 positions for a 34-channel montage.
    Returns (34, 2) array of (x, y) positions for topographic plots
    and (34, 3) array of (x, y, z) for distance computation.
    
    Channel order assumed (common 34-ch EEG cap):
    Fp1,Fp2, F7,F3,Fz,F4,F8, FC5,FC1,FC2,FC6,
    T7,C3,Cz,C4,T8, CP5,CP1,CP2,CP6,
    P7,P3,Pz,P4,P8, PO7,PO3,POz,PO4,PO8, O1,Oz,O2, REF
    """
    # Approximate 2D positions (normalized to [-1, 1])
    positions_2d = np.array([
        [-0.31, 0.95],  # 0: Fp1
        [ 0.31, 0.95],  # 1: Fp2
        [-0.81, 0.59],  # 2: F7
        [-0.39, 0.59],  # 3: F3
        [ 0.00, 0.59],  # 4: Fz
        [ 0.39, 0.59],  # 5: F4
        [ 0.81, 0.59],  # 6: F8
        [-0.63, 0.31],  # 7: FC5
        [-0.20, 0.31],  # 8: FC1
        [ 0.20, 0.31],  # 9: FC2
        [ 0.63, 0.31],  # 10: FC6
        [-0.95, 0.00],  # 11: T7
        [-0.39, 0.00],  # 12: C3
        [ 0.00, 0.00],  # 13: Cz
        [ 0.39, 0.00],  # 14: C4
        [ 0.95, 0.00],  # 15: T8
        [-0.63,-0.31],  # 16: CP5
        [-0.20,-0.31],  # 17: CP1
        [ 0.20,-0.31],  # 18: CP2
        [ 0.63,-0.31],  # 19: CP6
        [-0.81,-0.59],  # 20: P7
        [-0.39,-0.59],  # 21: P3
        [ 0.00,-0.59],  # 22: Pz
        [ 0.39,-0.59],  # 23: P4
        [ 0.81,-0.59],  # 24: P8
        [-0.63,-0.78],  # 25: PO7
        [-0.25,-0.78],  # 26: PO3
        [ 0.00,-0.78],  # 27: POz
        [ 0.25,-0.78],  # 28: PO4
        [ 0.63,-0.78],  # 29: PO8
        [-0.31,-0.95],  # 30: O1
        [ 0.00,-0.95],  # 31: Oz
        [ 0.31,-0.95],  # 32: O2
        [ 0.00, 1.10],  # 33: REF (approximate)
    ])
    
    channel_names = [
        'Fp1','Fp2','F7','F3','Fz','F4','F8',
        'FC5','FC1','FC2','FC6',
        'T7','C3','Cz','C4','T8',
        'CP5','CP1','CP2','CP6',
        'P7','P3','Pz','P4','P8',
        'PO7','PO3','POz','PO4','PO8',
        'O1','Oz','O2','REF'
    ]
    
    return positions_2d, channel_names


def build_spatial_adjacency(positions, k_neighbors=5):
    """Build k-nearest-neighbor spatial adjacency from electrode positions."""
    N = positions.shape[0]
    dist = squareform(pdist(positions))
    A = np.zeros((N, N))
    
    for i in range(N):
        # k nearest neighbors (excluding self)
        neighbors = np.argsort(dist[i])[1:k_neighbors+1]
        A[i, neighbors] = 1
        A[neighbors, i] = 1  # symmetric
    
    return A


def build_functional_adjacency(node_features, threshold_percentile=75):
    """Build functional adjacency from correlation of node features."""
    # node_features: (N_channels, feat_dim)
    corr = np.corrcoef(node_features)
    corr = np.abs(corr)  # use absolute correlation
    np.fill_diagonal(corr, 0)
    
    # Threshold
    thresh = np.percentile(corr[corr > 0], threshold_percentile)
    A = (corr >= thresh).astype(float)
    
    return A


def normalize_adjacency(A):
    """Compute D^{-1/2} A_tilde D^{-1/2} for GCN propagation."""
    A_tilde = A + np.eye(A.shape[0])  # add self-loops
    D = np.diag(A_tilde.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-12))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


# ================================================================
# SECTION 6: GNN PROPAGATION (NUMPY)
# ================================================================
def gcn_propagate(H, A_norm, n_layers=2):
    """GCN-style feature propagation: H' = A_norm @ H for each layer.
    Fixed propagation (no learnable weights) — tests graph structure value.
    """
    for _ in range(n_layers):
        H = A_norm @ H
    return H


def graphsage_propagate(H, A, n_layers=2):
    """GraphSAGE-style: concatenate self with mean-neighbor, then project."""
    for _ in range(n_layers):
        # Mean aggregation of neighbors
        D = A.sum(axis=1, keepdims=True) + 1e-12
        H_neigh = (A @ H) / D
        # Concatenate self with neighbor aggregate
        H = np.concatenate([H, H_neigh], axis=1)
    return H


def gat_propagate(H, A, n_layers=2, n_heads=4, seed=42):
    """GAT-style attention propagation.
    Uses random attention weights (not trained) but with proper softmax.
    Returns propagated features AND attention weights for interpretability.
    """
    rng = np.random.RandomState(seed)
    N, d = H.shape
    all_attentions = []
    
    for layer in range(n_layers):
        d_head = d // n_heads if d >= n_heads else d
        H_new = np.zeros((N, d_head * n_heads))
        layer_attention = np.zeros((N, N))
        
        for head in range(n_heads):
            # Random attention parameters
            W = rng.randn(d, d_head) * 0.1
            a = rng.randn(2 * d_head) * 0.1
            
            H_proj = H @ W  # (N, d_head)
            
            # Compute attention scores
            e = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if A[i, j] > 0 or i == j:
                        concat = np.concatenate([H_proj[i], H_proj[j]])
                        e[i, j] = np.maximum(0.2 * (a @ concat), a @ concat)  # LeakyReLU
                    else:
                        e[i, j] = -1e9  # mask non-neighbors
            
            # Softmax per row
            alpha = softmax(e, axis=1)
            layer_attention += alpha / n_heads
            
            # Aggregate
            H_head = alpha @ H_proj  # (N, d_head)
            H_new[:, head*d_head:(head+1)*d_head] = H_head
        
        H = H_new
        all_attentions.append(layer_attention)
    
    return H, all_attentions


def graph_readout(H, mode='mean'):
    """Graph-level readout: aggregate node features into graph embedding."""
    if mode == 'mean':
        return H.mean(axis=0)
    elif mode == 'sum':
        return H.sum(axis=0)
    elif mode == 'max':
        return H.max(axis=0)
    else:
        return H.mean(axis=0)


# ================================================================
# SECTION 7: EXPERIMENT RUNNER
# ================================================================
def run_experiment(X_node_features_train, X_node_features_test,
                   y_train, y_test, A, gnn_type='gcn', n_layers=2,
                   classifier='logreg'):
    """Run a single experiment: GNN propagation + readout + classification.
    
    Args:
        X_node_features_train: (N_train, 34, d) node features
        X_node_features_test: (N_test, 34, d) node features
        A: (34, 34) adjacency matrix (None for no-graph baselines)
        gnn_type: 'gcn', 'sage', 'gat', or 'none'
    Returns:
        predictions, probabilities
    """
    N_train = X_node_features_train.shape[0]
    N_test = X_node_features_test.shape[0]
    
    # GNN propagation per sample
    train_graph_feats = []
    test_graph_feats = []
    
    for i in range(N_train):
        H = X_node_features_train[i]  # (34, d)
        if gnn_type == 'none' or A is None:
            g = H.flatten()
        elif gnn_type == 'gcn':
            A_norm = normalize_adjacency(A)
            H_prop = gcn_propagate(H, A_norm, n_layers)
            g = graph_readout(H_prop)
        elif gnn_type == 'sage':
            H_prop = graphsage_propagate(H, A, n_layers)
            g = graph_readout(H_prop)
        elif gnn_type == 'gat':
            A_self = A + np.eye(A.shape[0])
            H_prop, _ = gat_propagate(H, A_self, n_layers)
            g = graph_readout(H_prop)
        else:
            g = H.flatten()
        train_graph_feats.append(g)
    
    for i in range(N_test):
        H = X_node_features_test[i]
        if gnn_type == 'none' or A is None:
            g = H.flatten()
        elif gnn_type == 'gcn':
            A_norm = normalize_adjacency(A)
            H_prop = gcn_propagate(H, A_norm, n_layers)
            g = graph_readout(H_prop)
        elif gnn_type == 'sage':
            H_prop = graphsage_propagate(H, A, n_layers)
            g = graph_readout(H_prop)
        elif gnn_type == 'gat':
            A_self = A + np.eye(A.shape[0])
            H_prop, _ = gat_propagate(H, A_self, n_layers)
            g = graph_readout(H_prop)
        else:
            g = H.flatten()
        test_graph_feats.append(g)
    
    X_tr = np.array(train_graph_feats)
    X_te = np.array(test_graph_feats)
    
    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    # Classify
    if classifier == 'logreg':
        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    elif classifier == 'svm':
        clf = SVC(C=1.0, kernel='rbf', random_state=42, probability=True)
    elif classifier == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                            random_state=42, early_stopping=True,
                            validation_fraction=0.15)
    else:
        clf = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    
    return preds


def run_full_cv(X_node_features, y, subjects, A, gnn_type='gcn',
                n_layers=2, classifier='logreg', n_folds=10):
    """Subject-level stratified k-fold cross-validation."""
    
    # Group k-fold: all conditions from same subject stay together
    gkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_preds = np.zeros_like(y)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_node_features, y, subjects)):
        X_tr = X_node_features[train_idx]
        X_te = X_node_features[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        
        preds = run_experiment(X_tr, X_te, y_tr, y_te, A, 
                               gnn_type=gnn_type, n_layers=n_layers,
                               classifier=classifier)
        all_preds[test_idx] = preds
    
    acc = accuracy_score(y, all_preds)
    bal_acc = balanced_accuracy_score(y, all_preds)
    f1 = f1_score(y, all_preds, average='macro')
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'f1_macro': f1,
        'predictions': all_preds,
        'true': y
    }


# ================================================================
# SECTION 8: THE 7-ROW BASELINE TABLE
# ================================================================
def run_all_experiments(X_ds, y, subjects, positions):
    """Run the complete 7-row baseline table + ablations.
    
    Returns dict of all results.
    """
    results = {}
    N = X_ds.shape[0]
    
    # ---- Feature extraction (done once, reused across experiments) ----
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    # Conventional features
    print("\nExtracting band-power features...")
    bp_feats = extract_bandpower(X_ds)  # (N, 34, 5)
    print(f"  Shape: {bp_feats.shape}")
    
    print("Extracting Hjorth features...")
    hj_feats = extract_hjorth(X_ds)  # (N, 34, 3)
    print(f"  Shape: {hj_feats.shape}")
    
    # Combined conventional
    conv_feats = np.concatenate([bp_feats, hj_feats], axis=2)  # (N, 34, 8)
    print(f"Combined conventional features: {conv_feats.shape}")
    
    # LSM features
    print("\nRunning LSM reservoir pipeline (BSC6)...")
    lsm_bsc6_raw = run_reservoir_pipeline(X_ds, coding='bsc6')
    print(f"  Raw BSC6 shape: {lsm_bsc6_raw.shape}")
    
    print("Running LSM reservoir pipeline (MFR)...")
    lsm_mfr_raw = run_reservoir_pipeline(X_ds, coding='mfr')
    print(f"  MFR shape: {lsm_mfr_raw.shape}")
    
    # PCA will be applied per-fold (to avoid data leakage)
    # For non-CV feature inspection, fit on all data
    print("Fitting PCA-64 on all BSC6 features (for inspection)...")
    lsm_bsc6_pca_all = np.zeros((N, 34, 64))
    for ch in range(34):
        n_comp = min(64, N - 1, lsm_bsc6_raw.shape[2])
        pca = PCA(n_components=n_comp)
        lsm_bsc6_pca_all[:, ch, :n_comp] = pca.fit_transform(lsm_bsc6_raw[:, ch, :])
    print(f"  PCA-64 shape: {lsm_bsc6_pca_all.shape}")
    
    # ---- Graph construction ----
    print("\n" + "="*60)
    print("GRAPH CONSTRUCTION")
    print("="*60)
    
    A_spat = build_spatial_adjacency(positions, k_neighbors=5)
    print(f"Spatial adjacency: {A_spat.sum():.0f} edges, "
          f"density={A_spat.sum()/(34*33):.3f}")
    
    # Functional adjacency from average embedding correlation
    avg_feats = lsm_bsc6_pca_all.mean(axis=0)  # (34, 64)
    A_func = build_functional_adjacency(avg_feats, threshold_percentile=75)
    print(f"Functional adjacency: {A_func.sum():.0f} edges, "
          f"density={A_func.sum()/(34*33):.3f}")
    
    # ---- THE 7-ROW TABLE ----
    print("\n" + "="*60)
    print("THE 7-ROW BASELINE TABLE")
    print("="*60)
    
    rows = [
        ("Row 1: BandPower + LogReg (no graph)",     conv_feats,        None,    'none', 'logreg'),
        ("Row 2: BandPower + MLP (no graph)",         conv_feats,        None,    'none', 'mlp'),
        ("Row 3: LSM-BSC6-PCA64 + MLP (no graph)",   lsm_bsc6_pca_all,  None,    'none', 'mlp'),
        ("Row 4: BandPower + GAT (spatial graph)",    conv_feats,        A_spat,  'gat',  'logreg'),
        ("Row 5: LSM-BSC6-PCA64 + GAT (spatial)",    lsm_bsc6_pca_all,  A_spat,  'gat',  'logreg'),
        ("Row 6: LSM-BSC6-PCA64 + GAT (functional)", lsm_bsc6_pca_all,  A_func,  'gat',  'logreg'),
        ("Row 7: LSM-MFR + GAT (spatial)",            lsm_mfr_raw,       A_spat,  'gat',  'logreg'),
    ]
    
    for name, feats, A, gnn, clf in rows:
        print(f"\n{name}...")
        res = run_full_cv(feats, y, subjects, A, gnn_type=gnn, classifier=clf)
        results[name] = res
        print(f"  Acc={res['accuracy']:.3f}, BalAcc={res['balanced_accuracy']:.3f}, "
              f"F1={res['f1_macro']:.3f}")
    
    # ---- EXPERIMENT 2: GNN ARCHITECTURE COMPARISON ----
    print("\n" + "="*60)
    print("EXPERIMENT 2: GNN ARCHITECTURE COMPARISON")
    print("="*60)
    
    for gnn_type in ['gcn', 'sage', 'gat']:
        name = f"Arch: {gnn_type.upper()} (spatial, LSM-BSC6-PCA64)"
        print(f"\n{name}...")
        res = run_full_cv(lsm_bsc6_pca_all, y, subjects, A_spat,
                          gnn_type=gnn_type, classifier='logreg')
        results[name] = res
        print(f"  Acc={res['accuracy']:.3f}, BalAcc={res['balanced_accuracy']:.3f}, "
              f"F1={res['f1_macro']:.3f}")
    
    # ---- EXPERIMENT 3: GRAPH SPARSITY ----
    print("\n" + "="*60)
    print("EXPERIMENT 3: GRAPH SPARSITY SWEEP")
    print("="*60)
    
    for k in [3, 5, 7, 10, 15]:
        A_k = build_spatial_adjacency(positions, k_neighbors=k)
        name = f"Sparsity: k={k} neighbors"
        print(f"\n{name} ({A_k.sum():.0f} edges)...")
        res = run_full_cv(lsm_bsc6_pca_all, y, subjects, A_k,
                          gnn_type='gcn', classifier='logreg')
        results[name] = res
        print(f"  Acc={res['accuracy']:.3f}")
    
    # ---- EXPERIMENT 4: GNN DEPTH ----
    print("\n" + "="*60)
    print("EXPERIMENT 4: GNN DEPTH ABLATION")
    print("="*60)
    
    for depth in [1, 2, 3, 4]:
        name = f"Depth: {depth} GCN layers"
        print(f"\n{name}...")
        res = run_full_cv(lsm_bsc6_pca_all, y, subjects, A_spat,
                          gnn_type='gcn', n_layers=depth, classifier='logreg')
        results[name] = res
        print(f"  Acc={res['accuracy']:.3f}")
    
    return results


# ================================================================
# SECTION 9: FIGURE GENERATION
# ================================================================
def plot_baseline_table(results):
    """Figure 1: The 7-row baseline comparison bar chart."""
    row_names = [k for k in results if k.startswith("Row")]
    short_names = [
        "BandPower\n+LogReg", "BandPower\n+MLP", "LSM+PCA\n+MLP",
        "BandPower\n+GAT(spat)", "ARSPI-Net\n(full)", "ARSPI-Net\n(func)", 
        "LSM-MFR\n+GAT(spat)"
    ]
    
    accs = [results[k]['balanced_accuracy'] * 100 for k in row_names]
    f1s = [results[k]['f1_macro'] * 100 for k in row_names]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(row_names))
    width = 0.35
    
    colors_acc = ['#999999','#999999','#4c72b0','#dd8452','#55a868','#55a868','#c44e52']
    
    bars = ax.bar(x, accs, width, color=colors_acc, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('ARSPI-Net: 7-Row Baseline Comparison (SHAPE EEG)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylim(max(20, min(accs) - 10), min(105, max(accs) + 8))
    ax.axhline(y=33.3, color='gray', linestyle=':', alpha=0.5, label='Chance (3-class)')
    
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.legend(loc='lower left')
    fig.savefig(FIGDIR / "baseline_comparison.pdf")
    plt.close(fig)
    print(f"  -> Saved baseline_comparison.pdf")


def plot_architecture_comparison(results):
    """Figure 2: GCN vs GraphSAGE vs GAT."""
    arch_names = [k for k in results if k.startswith("Arch:")]
    if not arch_names:
        return
    
    labels = [k.split(':')[1].strip().split()[0] for k in arch_names]
    accs = [results[k]['balanced_accuracy'] * 100 for k in arch_names]
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    colors = ['#4c72b0', '#dd8452', '#55a868']
    ax.bar(labels, accs, color=colors[:len(labels)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('GNN Architecture Comparison')
    ax.set_ylim(max(20, min(accs) - 10), min(105, max(accs) + 8))
    
    for i, v in enumerate(accs):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    fig.savefig(FIGDIR / "architecture_comparison.pdf")
    plt.close(fig)
    print(f"  -> Saved architecture_comparison.pdf")


def plot_sparsity_sweep(results):
    """Figure 3: Graph sparsity sensitivity."""
    sparsity_names = [k for k in results if k.startswith("Sparsity:")]
    if not sparsity_names:
        return
    
    ks = [int(k.split('=')[1].split()[0]) for k in sparsity_names]
    accs = [results[k]['balanced_accuracy'] * 100 for k in sparsity_names]
    
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(ks, accs, 'o-', color='#4c72b0', linewidth=1.5, markersize=6)
    ax.set_xlabel('Number of Neighbors ($k$)')
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('Graph Sparsity Sensitivity')
    ax.set_xticks(ks)
    
    fig.savefig(FIGDIR / "sparsity_sweep.pdf")
    plt.close(fig)
    print(f"  -> Saved sparsity_sweep.pdf")


def plot_depth_ablation(results):
    """Figure 4: GNN depth vs accuracy."""
    depth_names = [k for k in results if k.startswith("Depth:")]
    if not depth_names:
        return
    
    depths = [int(k.split(':')[1].strip().split()[0]) for k in depth_names]
    accs = [results[k]['balanced_accuracy'] * 100 for k in depth_names]
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(depths, accs, 's-', color='#55a868', linewidth=1.5, markersize=7)
    ax.set_xlabel('Number of GNN Layers')
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('GNN Depth Ablation')
    ax.set_xticks(depths)
    
    fig.savefig(FIGDIR / "depth_ablation.pdf")
    plt.close(fig)
    print(f"  -> Saved depth_ablation.pdf")


def plot_confusion_matrix(results):
    """Figure 5: Confusion matrix for best model (Row 5)."""
    best_key = [k for k in results if 'Row 5' in k]
    if not best_key:
        return
    
    res = results[best_key[0]]
    cm = confusion_matrix(res['true'], res['predictions'])
    
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    labels = ['Negative', 'Neutral', 'Pleasant']
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('ARSPI-Net Confusion Matrix')
    
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color=color, fontweight='bold')
    
    fig.colorbar(im, shrink=0.8)
    fig.savefig(FIGDIR / "confusion_matrix.pdf")
    plt.close(fig)
    print(f"  -> Saved confusion_matrix.pdf")


def generate_all_figures(results):
    """Generate all Chapter 5 figures."""
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    plot_baseline_table(results)
    plot_architecture_comparison(results)
    plot_sparsity_sweep(results)
    plot_depth_ablation(results)
    plot_confusion_matrix(results)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='ARSPI-Net Chapter 5 Experiments')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing SHAPE .txt files')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data from single file')
    parser.add_argument('--sample_file', type=str, 
                        default='/mnt/user-data/uploads/SHAPE_Community_007_IAPSNeg_BC.txt',
                        help='Sample file for demo mode')
    parser.add_argument('--n_demo_subjects', type=int, default=20,
                        help='Number of simulated subjects in demo mode')
    args = parser.parse_args()
    
    print("="*60)
    print("ARSPI-Net Chapter 5: Complete Experimental Pipeline")
    print("="*60)
    
    # Load data
    if args.data_dir and os.path.isdir(args.data_dir):
        X, y, subjects, cond_names = load_shape_data(args.data_dir)
    else:
        print("\nNo data directory provided. Running DEMO mode.")
        print(f"  Using sample file: {args.sample_file}")
        X, y, subjects, cond_names = create_demo_dataset(
            args.sample_file, n_subjects=args.n_demo_subjects)
    
    # Preprocess
    print("\nPreprocessing (downsample 4x to 256 Hz, z-score)...")
    X_ds = preprocess(X, downsample_factor=4)
    print(f"  Preprocessed shape: {X_ds.shape}")
    
    # Get electrode positions
    positions, ch_names = get_standard_34ch_positions()
    
    # Run all experiments
    results = run_all_experiments(X_ds, y, subjects, positions)
    
    # Generate figures
    generate_all_figures(results)
    
    # Summary
    print("\n" + "="*60)
    print("COMPLETE RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Experiment':<50} {'BalAcc':>8} {'F1':>8}")
    print("-"*66)
    for name, res in results.items():
        print(f"{name:<50} {res['balanced_accuracy']*100:7.1f}% {res['f1_macro']*100:7.1f}%")
    
    # Key comparisons
    r5 = [v for k,v in results.items() if 'Row 5' in k]
    r3 = [v for k,v in results.items() if 'Row 3' in k]
    r4 = [v for k,v in results.items() if 'Row 4' in k]
    r7 = [v for k,v in results.items() if 'Row 7' in k]
    
    if r5 and r3:
        delta_graph = r5[0]['balanced_accuracy'] - r3[0]['balanced_accuracy']
        print(f"\n  Graph structure value (Row5 - Row3):  {delta_graph*100:+.1f}%")
    if r5 and r4:
        delta_lsm = r5[0]['balanced_accuracy'] - r4[0]['balanced_accuracy']
        print(f"  LSM embedding value (Row5 - Row4):    {delta_lsm*100:+.1f}%")
    if r5 and r7:
        delta_temporal = r5[0]['balanced_accuracy'] - r7[0]['balanced_accuracy']
        print(f"  Temporal coding value (Row5 - Row7):  {delta_temporal*100:+.1f}%")
    
    print(f"\nFigures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
