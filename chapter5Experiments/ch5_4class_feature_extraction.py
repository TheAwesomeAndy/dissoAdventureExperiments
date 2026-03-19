"""
============================================================================
Chapter 5 — 4-Class Feature Extraction
============================================================================
Purpose: Extract BSC6-PCA64 and band-power features from all 4 IAPS
         subcategories (Threat, Mutilation, Cute, Erotic) for the 4-class
         re-run of Chapter 5 classification experiments.
Data:    SHAPE EEG epochs from categoriesbatch{1-4}/
         4 categories x 212 subjects (211 after excluding Subject 127)
         Each file: (1229, 34) -- 1229 timepoints x 34 channels
Pipeline:
  1. Load raw EEG (1229 x 34)
  2. Downsample 4x -> truncate to 256 timesteps
  3. Z-score per channel
  4. Run LIF reservoir (N=256, beta=0.05, theta=0.5) on each channel
  5. Extract BSC6 (6 bins, t=10-70) -> 256x6 = 1536 per channel
  6. Extract 5-band power (delta, theta, alpha, beta, gamma)
  7. Fit PCA-64 on pooled BSC6, transform all observations
  8. Save shape_features_4class.pkl
Output:  shape_features_4class.pkl with keys:
           X_ds:          (844, 256, 34) -- downsampled z-scored EEG
           y:             (844,) -- class labels (0=Threat, 1=Mutilation,
                                                 2=Cute, 3=Erotic)
           subjects:      (844,) -- subject IDs
           categories:    (844,) -- category name strings
           lsm_bsc6_raw: (844, 34, 1536) -- raw BSC6 features
           lsm_bsc6_pca: (844, 34, 64) -- PCA-64 embeddings
           conv_feats:    (844, 34, 5) -- band-power features
           lsm_mfr:       (844, 34, 256) -- mean firing rate vectors
           cond_names:    {0:'Threat', 1:'Mutilation', 2:'Cute', 3:'Erotic'}
Usage:   python ch5_4class_feature_extraction.py
         Adjust DATA_DIR below if your batch directories are elsewhere.
Runtime: ~20-40 minutes depending on hardware (844 obs x 34 channels each)
============================================================================
"""
import numpy as np
import os
import pickle
import time
from scipy.signal import welch, decimate
from sklearn.decomposition import PCA
# ===================================================================
# CONFIGURATION -- adjust these paths for your machine
# ===================================================================
DATA_DIR = './CategoryFiles'  # parent directory containing categoriesbatch{1-4}/
OUTPUT_FILE = './shape_features_4class.pkl'
EXCLUDE_SUBJECTS = {127}  # subjects to exclude
# Reservoir parameters (must match Ch3/Ch4 exactly)
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
# Feature extraction parameters (must match Ch4 exactly)
BSC_N_BINS = 6
BSC_T_START = 10
BSC_T_END = 70
PCA_N_COMPONENTS = 64
# Preprocessing
DOWNSAMPLE_FACTOR = 4
TARGET_TIMESTEPS = 256
# Sampling rate (original, before downsampling)
FS_ORIGINAL = 1000  # Hz -- assumed; adjust if different
FS_DS = FS_ORIGINAL // DOWNSAMPLE_FACTOR
# Category mapping
CATEGORY_MAP = {
    'IAPSNeg_Threat_BC': ('Threat', 0),
    'IAPSNeg_Mutilation_BC': ('Mutilation', 1),
    'IAPSPos_Cute_BC': ('Cute', 2),
    'IAPSPos_Erotic_BC': ('Erotic', 3),
}
# ===================================================================
# LIF RESERVOIR (exact replication of Ch4 LIFReservoir class)
# ===================================================================
class LIFReservoir:
    """Leaky Integrate-and-Fire reservoir.

    Replicates the exact implementation from run_chapter4_experiments.py:
    - Xavier uniform initialization for W_in and W_rec
    - Spectral radius scaling to 0.9
    - Multiplicative membrane reset on spike
    - Hard threshold with subtraction and floor at zero
    """
    def __init__(self, n_input, n_res, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigenvalues = np.abs(np.linalg.eigvals(self.W_rec))
        if eigenvalues.max() > 0:
            self.W_rec *= 0.9 / eigenvalues.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res
    def forward(self, X):
        """Process input X of shape (T, n_input). Returns (spikes, membrane)."""
        T = X.shape[0]
        n_input = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
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
# ===================================================================
# FEATURE EXTRACTION FUNCTIONS
# ===================================================================
def extract_bsc(spikes, n_bins=6, t_start=10, t_end=70):
    """Binned Spike Count: partition spike window into n_bins temporal bins.

    Args:
        spikes: (T, N_res) spike matrix
        n_bins: number of temporal bins
        t_start, t_end: time window boundaries (in timesteps)

    Returns:
        (N_res * n_bins,) flattened BSC vector
    """
    window = spikes[t_start:t_end]  # (t_end-t_start, N_res)
    T_w = window.shape[0]
    bin_size = T_w // n_bins
    bsc = np.zeros((spikes.shape[1], n_bins))
    for b in range(n_bins):
        bsc[:, b] = window[b*bin_size:(b+1)*bin_size].sum(axis=0)
    return bsc.flatten()  # (N_res * n_bins,)
def extract_mfr(spikes, t_start=10, t_end=70):
    """Mean Firing Rate in the specified window.

    Returns:
        (N_res,) vector of per-neuron firing rates
    """
    return spikes[t_start:t_end].mean(axis=0)
def extract_band_power(signal, fs):
    """Extract 5-band power features from a 1D signal.

    Bands: delta (1-4 Hz), theta (4-8), alpha (8-13), beta (13-30), gamma (30-45)

    Args:
        signal: (T,) 1D signal
        fs: sampling frequency in Hz

    Returns:
        (5,) array of band powers
    """
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, min(45, fs/2 - 1))]
    nperseg = min(len(signal), 256)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    powers = np.zeros(5)
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.any():
            powers[i] = np.trapezoid(psd[mask], freqs[mask])
    return powers
# ===================================================================
# DATA LOADING
# ===================================================================
def discover_files(data_dir):
    """Scan batch directories and build a list of (filepath, subject_id, category_key)."""
    files = []
    for batch in ['categoriesbatch1', 'categoriesbatch2', 'categoriesbatch3', 'categoriesbatch4']:
        bdir = os.path.join(data_dir, batch)
        if not os.path.exists(bdir):
            print(f"  WARNING: {bdir} not found, skipping")
            continue
        for fn in sorted(os.listdir(bdir)):
            if not fn.endswith('.txt'):
                continue
            # Parse: SHAPE_Community_006_IAPSNeg_Threat_BC.txt
            parts = fn.replace('.txt', '').split('_')
            # parts = ['SHAPE', 'Community', '006', 'IAPSNeg', 'Threat', 'BC']
            # or      ['SHAPE', 'Community', '006', 'IAPSPos', 'Cute', 'BC']
            subject_id = int(parts[2])
            cat_key = '_'.join(parts[3:])  # e.g. 'IAPSNeg_Threat_BC'

            if cat_key not in CATEGORY_MAP:
                print(f"  WARNING: Unknown category {cat_key} in {fn}")
                continue
            if subject_id in EXCLUDE_SUBJECTS:
                continue

            files.append((os.path.join(bdir, fn), subject_id, cat_key))

    return files
def preprocess_eeg(raw, downsample_factor=4, target_T=256):
    """Downsample and z-score raw EEG.

    Args:
        raw: (T_raw, 34) raw EEG array
        downsample_factor: decimation factor
        target_T: target number of timesteps after downsampling

    Returns:
        (target_T, 34) preprocessed EEG
    """
    n_ch = raw.shape[1]
    # Decimate each channel — let scipy determine output length
    ch0 = decimate(raw[:, 0], downsample_factor)
    ds = np.zeros((len(ch0), n_ch))
    ds[:, 0] = ch0
    for ch in range(1, n_ch):
        ds[:, ch] = decimate(raw[:, ch], downsample_factor)

    # Truncate or pad to target_T
    if ds.shape[0] >= target_T:
        ds = ds[:target_T]
    else:
        # Pad with zeros if shorter (should not happen with 1229/4 = 307)
        pad = np.zeros((target_T - ds.shape[0], n_ch))
        ds = np.concatenate([ds, pad], axis=0)

    # Z-score per channel
    for ch in range(n_ch):
        mu = ds[:, ch].mean()
        sigma = ds[:, ch].std()
        if sigma > 0:
            ds[:, ch] = (ds[:, ch] - mu) / sigma
        else:
            ds[:, ch] = 0.0

    return ds
# ===================================================================
# MAIN PIPELINE
# ===================================================================
def main():
    print("=" * 70)
    print("CHAPTER 5 -- 4-CLASS FEATURE EXTRACTION")
    print("=" * 70)

    # --- Discover files -----------------------------------------------
    print(f"\nScanning {DATA_DIR} for EEG files...")
    files = discover_files(DATA_DIR)
    print(f"  Found {len(files)} files")

    # Organize by (subject, category)
    subjects_all = sorted(set(sid for _, sid, _ in files))
    print(f"  {len(subjects_all)} unique subjects (after excluding {EXCLUDE_SUBJECTS})")

    # Verify completeness: every subject should have all 4 categories
    from collections import Counter
    subj_counts = Counter(sid for _, sid, _ in files)
    incomplete = {sid: cnt for sid, cnt in subj_counts.items() if cnt != 4}
    if incomplete:
        print(f"  WARNING: {len(incomplete)} subjects have incomplete data:")
        for sid, cnt in sorted(incomplete.items()):
            print(f"    Subject {sid}: {cnt}/4 categories")

    # Sort files by subject then category for deterministic ordering
    files.sort(key=lambda x: (x[1], CATEGORY_MAP[x[2]][1]))

    N_obs = len(files)
    N_ch = 34
    print(f"\n  Total observations: {N_obs}")
    print(f"  Expected: {len(subjects_all)} subjects x 4 categories = {len(subjects_all)*4}")

    # --- Initialize reservoir -----------------------------------------
    print(f"\nInitializing LIF reservoir: N={N_RES}, beta={BETA}, theta={THRESHOLD}, seed={SEED}")
    reservoir = LIFReservoir(n_input=1, n_res=N_RES, beta=BETA,
                              threshold=THRESHOLD, seed=SEED)

    # --- Allocate arrays ----------------------------------------------
    X_ds = np.zeros((N_obs, TARGET_TIMESTEPS, N_ch))
    y = np.zeros(N_obs, dtype=np.int64)
    subjects = np.zeros(N_obs, dtype=np.int64)
    categories = []
    bsc6_raw = np.zeros((N_obs, N_ch, N_RES * BSC_N_BINS))
    mfr_all = np.zeros((N_obs, N_ch, N_RES))
    conv_feats = np.zeros((N_obs, N_ch, 5))

    # --- Process all files --------------------------------------------
    print(f"\nProcessing {N_obs} observations ({N_ch} channels each)...")
    print(f"  Reservoir: {N_RES} neurons x {TARGET_TIMESTEPS} timesteps per channel")
    print(f"  BSC6: bins [{BSC_T_START}-{BSC_T_END}], {BSC_N_BINS} bins")
    print()

    t0 = time.time()
    for obs_i, (filepath, sid, cat_key) in enumerate(files):
        cat_name, cat_label = CATEGORY_MAP[cat_key]

        # Load raw EEG
        raw = np.loadtxt(filepath)  # (1229, 34)

        # Preprocess
        eeg = preprocess_eeg(raw, DOWNSAMPLE_FACTOR, TARGET_TIMESTEPS)
        X_ds[obs_i] = eeg
        y[obs_i] = cat_label
        subjects[obs_i] = sid
        categories.append(cat_name)

        # Process each channel through the reservoir
        for ch in range(N_ch):
            signal = eeg[:, ch].reshape(-1, 1)  # (T, 1)
            spikes, _ = reservoir.forward(signal)  # (T, N_RES)

            # BSC6 features
            bsc6_raw[obs_i, ch] = extract_bsc(spikes, BSC_N_BINS,
                                               BSC_T_START, BSC_T_END)

            # Mean firing rate
            mfr_all[obs_i, ch] = extract_mfr(spikes, BSC_T_START, BSC_T_END)

            # Band-power features
            conv_feats[obs_i, ch] = extract_band_power(eeg[:, ch], FS_DS)

        # Progress reporting
        elapsed = time.time() - t0
        rate = (obs_i + 1) / elapsed
        eta = (N_obs - obs_i - 1) / rate if rate > 0 else 0
        if (obs_i + 1) % 20 == 0 or obs_i == N_obs - 1:
            print(f"  [{obs_i+1:4d}/{N_obs}] Subject {sid:3d} {cat_name:12s} | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining "
                  f"({rate:.1f} obs/s)")

    total_time = time.time() - t0
    print(f"\n  Feature extraction complete: {total_time:.1f}s ({total_time/60:.1f} min)")

    # --- BSC6 statistics ----------------------------------------------
    print(f"\n  BSC6 raw: shape={bsc6_raw.shape}, mean={bsc6_raw.mean():.3f}, "
          f"zeros={100*(bsc6_raw==0).mean():.1f}%")

    # --- PCA-64 reduction ---------------------------------------------
    print(f"\nFitting PCA-{PCA_N_COMPONENTS} on pooled BSC6 features...")
    # Pool: (N_obs x N_ch, 1536) -> fit PCA
    pooled = bsc6_raw.reshape(-1, N_RES * BSC_N_BINS)  # (N_obs*34, 1536)
    n_comp = min(PCA_N_COMPONENTS, pooled.shape[0] - 1, pooled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pooled_pca = pca.fit_transform(pooled)

    # Reshape back to (N_obs, N_ch, n_comp)
    bsc6_pca = pooled_pca.reshape(N_obs, N_ch, n_comp)

    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA-{n_comp}: {var_explained:.1f}% variance explained")
    print(f"  Embeddings: shape={bsc6_pca.shape}")

    # --- Verify data integrity ----------------------------------------
    print(f"\n{'='*70}")
    print("DATA INTEGRITY CHECK")
    print(f"{'='*70}")
    print(f"  X_ds shape:          {X_ds.shape}")
    print(f"  X_ds mean:           {X_ds.mean():.6f} (expected ~0)")
    print(f"  X_ds std:            {X_ds.std():.6f} (expected ~1)")
    print(f"  y distribution:      {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  subjects unique:     {len(np.unique(subjects))}")
    print(f"  lsm_bsc6_pca shape:  {bsc6_pca.shape}")
    print(f"  lsm_bsc6_raw shape:  {bsc6_raw.shape}")
    print(f"  conv_feats shape:    {conv_feats.shape}")
    print(f"  lsm_mfr shape:       {mfr_all.shape}")

    # --- Save ---------------------------------------------------------
    output = {
        'X_ds': X_ds,
        'y': y,
        'subjects': subjects,
        'categories': np.array(categories),
        'lsm_bsc6_pca': bsc6_pca,
        'lsm_bsc6_raw': bsc6_raw,
        'conv_feats': conv_feats,
        'lsm_mfr': mfr_all,
        'cond_names': {0: 'Threat', 1: 'Mutilation', 2: 'Cute', 3: 'Erotic'},
        'pca_variance_explained': pca.explained_variance_ratio_,
    }

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(output, f, protocol=4)

    fsize = os.path.getsize(OUTPUT_FILE) / (1024**2)
    print(f"\n  Saved: {OUTPUT_FILE} ({fsize:.1f} MB)")
    print(f"\n{'='*70}")
    print("FEATURE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  {N_obs} observations x {N_ch} channels x {n_comp} PCA dimensions")
    print(f"  4 classes: Threat ({(y==0).sum()}), Mutilation ({(y==1).sum()}), "
          f"Cute ({(y==2).sum()}), Erotic ({(y==3).sum()})")
    print(f"  Ready for: ch5_4class_classification.py")
if __name__ == '__main__':
    main()
