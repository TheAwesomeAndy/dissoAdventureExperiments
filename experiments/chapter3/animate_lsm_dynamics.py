#!/usr/bin/env python3
"""
animate_lsm_dynamics.py
================================================================
LSM-in-action animations for the ARSPI-Net dissertation defense.

The defense deck presents the liquid state machine (LSM) and graph layer
through static PDF figures. Three core claims are inherently *dynamical*
and undersold by static images:

  1. Graph diffusion / over-smoothing  -> message passing is discrete
     diffusion that collapses node-feature contrast (the regime boundary,
     slide "Why the boundary exists: graph diffusion").
  2. The LIF reservoir as a driven point process -> ERP transients become
     sparse spike *timing* (temporal-coding claim).
  3. Reservoir lifting -> trajectories become linearly separable, so a
     linear readout suffices (Koopman lens).

This script renders one animation per claim (GIF + MP4 + poster PDF),
reusing the EXACT operators already validated in the repo so the
animations show the same system the dissertation characterizes:

  - single-channel LIFReservoir / extract_bsc6 / functional-adjacency /
    GCN-propagation operators are verbatim from
    chapter5Experiments/graph_diffusion_oversmoothing.py
  - multi-channel LIFReservoir (returns spikes + membrane) and
    generate_temporal_task are verbatim from
    experiments/chapter3/run_chapter3_lsm_characterization.py

Animation 1 (graph diffusion) runs on the REAL SHAPE node features when
shape_features_211.pkl (or an X_ds .npz extract) is supplied via --pkl;
otherwise it falls back to a synthetic spatially-correlated 34-channel
example and prints a clear notice. Animations 2 & 3 use the synthetic
controlled temporal task, faithful to how Chapters 3 & 4 are designed.

Usage:
  python animate_lsm_dynamics.py --only all
  python animate_lsm_dynamics.py --only diffusion --pkl X_ds_small.npz
  python animate_lsm_dynamics.py --format mp4 --fps 18 --dpi 100

Author: Andrew Lane | Stony Brook University ECE
================================================================
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------
# Defense / static-figure palette
# ----------------------------------------------------------------
NAVY, TEAL, ACCENT, GREY = '#1A2A4F', '#1C7293', '#B5651D', '#8A94A6'
C0, C1 = NAVY, ACCENT          # fixed condition colour pair (class 0 / class 1)
NODE_CMAP = LinearSegmentedColormap.from_list(
    'arspi', ['#E7E2D6', TEAL, NAVY])

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 11, 'figure.facecolor': 'white',
})

# ----------------------------------------------------------------
# Reservoir parameters (consistent with all chapters)
# ----------------------------------------------------------------
N_INPUT = 33
N_RES = 256
BETA = 0.05
THRESHOLD = 0.5
SEED = 42
TARGET_SR = 0.9


# ================================================================
# Operators -- verbatim from run_chapter3_lsm_characterization.py
# ================================================================
class LIFReservoirMC:
    """Multi-channel leaky integrate-and-fire reservoir; returns spikes
    AND membrane. Verbatim from run_chapter3_lsm_characterization.py."""
    def __init__(self, n_input, n_res, beta=BETA, threshold=THRESHOLD, seed=SEED):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eigenvalues = np.abs(np.linalg.eigvals(self.W_rec))
        if eigenvalues.max() > 0:
            self.W_rec *= TARGET_SR / eigenvalues.max()
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
        T, n_in = X.shape
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


def generate_temporal_task(n_per_class=200, n_input=N_INPUT, T=150,
                           noise_std=0.15, seed=42):
    """Two-class temporal pattern discrimination (verbatim from Ch3/Ch4).
    Class 0 = early burst, class 1 = late burst."""
    rng = np.random.RandomState(seed)
    X_list, y = [], []
    for cls in range(2):
        amps = [1.0, 0.6, 0.3] if cls == 0 else [0.3, 0.6, 1.0]
        for _ in range(n_per_class):
            x = np.zeros((T, n_input))
            for start, amp in zip([20, 35, 50], amps):
                jitter = rng.randint(-3, 4)
                a_jit = 1.0 + rng.uniform(-0.2, 0.2)
                s = max(0, min(T - 8, start + jitter))
                x[s:s + 8, :5] = 1.5 * amp * a_jit
            x[15:65, :] += rng.randn(50, n_input) * noise_std
            X_list.append(x)
            y.append(cls)
    return X_list, np.array(y)


# ================================================================
# Operators -- verbatim from graph_diffusion_oversmoothing.py
# ================================================================
class LIFReservoirSC:
    """Single-channel LIF reservoir. Verbatim from
    graph_diffusion_oversmoothing.py."""
    def __init__(self, n_res=256, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (1 + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, 1))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eig_max = np.abs(np.linalg.eigvals(self.W_rec)).max()
        if eig_max > 0:
            self.W_rec *= 0.9 / eig_max
        self.beta, self.threshold, self.n_res = beta, threshold, n_res

    def forward(self, x):
        T = len(x)
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        for t in range(T):
            I_in = self.W_in[:, 0] * x[t]
            I_rec = self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_in + I_rec
            spk = (mem >= self.threshold).astype(float)
            mem = np.maximum(mem - spk * self.threshold, 0.0)
            spikes[t] = spk
            spk_prev = spk
        return spikes


def extract_bsc6(spikes):
    """Binned spike counts, 6 temporal bins. Verbatim."""
    T_w, _ = spikes.shape
    bs = T_w // 6
    return np.concatenate([spikes[b * bs:(b + 1) * bs].sum(axis=0)
                           for b in range(6)])


def build_functional_adjacency(node_features, threshold_percentile=75):
    """Verbatim from graph_diffusion_oversmoothing.py."""
    corr = np.nan_to_num(np.corrcoef(node_features), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    pos = corr[corr > 0]
    if pos.size == 0:
        return None
    thresh = np.percentile(pos, threshold_percentile)
    return (corr >= thresh).astype(float)


def normalize_adjacency(A):
    """Verbatim from graph_diffusion_oversmoothing.py."""
    A_tilde = A + np.eye(A.shape[0])
    D = np.diag(A_tilde.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-12))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


def mean_pairwise_cosine(H):
    """Verbatim from graph_diffusion_oversmoothing.py."""
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    S = Hn @ Hn.T
    iu = np.triu_indices(H.shape[0], k=1)
    return S[iu].mean()


def dirichlet_energy(H, A_norm):
    """Verbatim from graph_diffusion_oversmoothing.py."""
    diff2 = ((H[:, None, :] - H[None, :, :]) ** 2).sum(axis=2)
    return 0.5 * np.sum(A_norm * diff2) / (np.sum(H ** 2) + 1e-12)


# ================================================================
# Helpers
# ================================================================
def circular_layout(n):
    """Evenly spaced nodes on a circle: every node visible, no overlap,
    edges drawn as chords. Clean and readable for a 34-electrode graph."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 2.0
    pos = np.column_stack([np.cos(ang), np.sin(ang)])
    return (pos + 1.12) / 2.24


def add_progress_bar(fig, label):
    """Slim progress bar + persistent label drawn in figure coords."""
    ax = fig.add_axes([0.08, 0.014, 0.84, 0.013])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='#E9E9EC', edgecolor='none'))
    fill = ax.add_patch(Rectangle((0, 0), 0, 1, facecolor=TEAL, edgecolor='none'))
    fig.text(0.08, 0.034, label, fontsize=8.5, color=GREY, ha='left')

    def _set(frac):
        fill.set_width(max(0.0, min(1.0, frac)))
    return _set


def synth_X_ds(n=130, T=170, C=34, seed=0):
    """Synthetic spatially-correlated 34-channel ERP-like fallback for
    animation 1 when the real SHAPE pkl is unavailable."""
    rng = np.random.RandomState(seed)
    groups = rng.randint(0, 6, size=C)
    tt = np.arange(T)
    X = np.zeros((n, T, C), dtype=np.float32)
    for i in range(n):
        latent = {}
        for g in range(6):
            sig = np.zeros(T)
            for _ in range(rng.randint(2, 5)):
                c = rng.uniform(0.2, 0.85) * T
                w = rng.uniform(0.03, 0.09) * T
                a = rng.uniform(0.5, 1.6) * (1.0 if rng.rand() < 0.5 else -1.0)
                sig += a * np.exp(-0.5 * ((tt - c) / w) ** 2)
            latent[g] = sig
        for c in range(C):
            X[i, :, c] = latent[groups[c]] + rng.randn(T) * 0.18
    return X


def load_X_ds(pkl_path):
    """Return (X_ds, source_label). Accepts a .npz with key 'X_ds' or a
    pickle dict with key 'X_ds'; otherwise synthesizes a fallback."""
    if pkl_path and os.path.exists(pkl_path):
        if pkl_path.endswith('.npz'):
            X = np.asarray(np.load(pkl_path)['X_ds'], dtype=np.float64)
        else:
            with open(pkl_path, 'rb') as f:
                X = np.asarray(pickle.load(f)['X_ds'], dtype=np.float64)
        return X, f'real SHAPE data ({os.path.basename(pkl_path)})'
    if pkl_path:
        print(f"  [notice] --pkl '{pkl_path}' not found; using synthetic "
              f"fallback.", flush=True)
    else:
        print("  [notice] no --pkl supplied; using synthetic spatially-"
              "correlated fallback.", flush=True)
    return synth_X_ds(), 'SYNTHETIC fallback (no real data supplied)'


# ================================================================
# Rendering driver
# ================================================================
def render(fig, update, n_frames, poster_idx, base, fmt, fps, dpi):
    """Save GIF / MP4 / poster PDF for one animation."""
    written = []
    if fmt in ('gif', 'both'):
        try:
            anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
            anim.save(base + '.gif', writer=PillowWriter(fps=fps), dpi=dpi)
            written.append(base + '.gif')
        except Exception as e:                                  # noqa: BLE001
            print(f"  [warn] GIF render failed: {e}", flush=True)
    if fmt in ('mp4', 'both'):
        try:
            anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
            writer = FFMpegWriter(
                fps=fps, codec='libx264', bitrate=2400,
                extra_args=['-pix_fmt', 'yuv420p',
                            '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2'])
            anim.save(base + '.mp4', writer=writer, dpi=dpi)
            written.append(base + '.mp4')
        except Exception as e:                                  # noqa: BLE001
            print(f"  [warn] MP4 render failed ({e}); GIF still available.",
                  flush=True)
    update(poster_idx)
    fig.savefig(base + '_poster.pdf', bbox_inches='tight')
    written.append(base + '_poster.pdf')
    plt.close(fig)
    for w in written:
        print(f"  -> {w}  ({os.path.getsize(w) / 1e3:.0f} kB)", flush=True)
    return written


# ================================================================
# Animation 1 -- Graph diffusion / over-smoothing
# ================================================================
def animate_diffusion(outdir, pkl_path, n_obs, fmt, fps, dpi):
    print("\n[1/3] Graph diffusion (over-smoothing)", flush=True)
    X_ds, source = load_X_ds(pkl_path)
    N, T, C = X_ds.shape
    print(f"  X_ds: {X_ds.shape}  source: {source}", flush=True)

    rng = np.random.RandomState(0)
    n_use = int(min(n_obs, N))
    idx = rng.choice(N, size=n_use, replace=False)
    reservoirs = [LIFReservoirSC(n_res=256, seed=42 + ch * 17) for ch in range(C)]

    t0 = time.time()
    bsc = np.zeros((n_use, C, 6 * 256))
    for j, oi in enumerate(idx):
        for ch in range(C):
            bsc[j, ch] = extract_bsc6(reservoirs[ch].forward(X_ds[oi, :, ch]))
        if (j + 1) % 40 == 0:
            print(f"    reservoir {j + 1}/{n_use} ({time.time() - t0:.0f}s)",
                  flush=True)

    n_comp = int(min(64, n_use - 1, bsc.shape[2]))
    node = np.zeros((n_use, C, n_comp))
    for ch in range(C):
        node[:, ch, :] = PCA(n_components=n_comp).fit_transform(bsc[:, ch, :])

    # GCN propagation = diffusion on the normalized graph Laplacian
    Ks = list(range(0, 9))
    valid = []
    for j in range(n_use):
        A = build_functional_adjacency(node[j], 75)
        if A is None or A.sum() == 0:
            continue
        A_norm = normalize_adjacency(A)
        H = node[j].copy()
        dr, cs, states = [], [], []
        for _ in Ks:
            states.append(H.copy())
            cs.append(mean_pairwise_cosine(H))
            dr.append(dirichlet_energy(H, A_norm))
            H = A_norm @ H
        valid.append(dict(j=j, A=A, dr=np.array(dr), cos=np.array(cs),
                          states=states))

    dr_all = np.array([v['dr'] for v in valid])
    cos_all = np.array([v['cos'] for v in valid])
    dr_mean, cos_mean = dr_all.mean(0), cos_all.mean(0)
    # representative graph = closest to the aggregate Dirichlet trajectory
    rep = valid[int(np.argmin(((dr_all - dr_mean) ** 2).sum(1)))]
    dr, cos, states = rep['dr'], rep['cos'], rep['states']
    agg_drop = 100 * (dr_mean[2] - dr_mean[0]) / abs(dr_mean[0])
    print(f"  valid graphs: {len(valid)}/{n_use};  aggregate Dirichlet "
          f"K0->K2 drop {abs(agg_drop):.0f}%", flush=True)

    # Frobenius-normalised states for scale-invariant display
    disp = [s / (np.linalg.norm(s) + 1e-12) for s in states]
    norms = [np.linalg.norm(d, axis=1) for d in disp]
    vmin, vmax = norms[0].min(), norms[0].max()
    pos = circular_layout(C)
    A = rep['A']
    edges = [(i, k) for i in range(C) for k in range(i + 1, C) if A[i, k] > 0]

    # ---- frame schedule -------------------------------------------------
    HOLD0, TWEEN, HOLD, ENDFREEZE = 12, 14, 5, 32
    sched = []  # (Hdisp_norms, int_K, reveal)
    sched += [(norms[0], 0, 1)] * HOLD0
    for k in range(1, 9):
        for s in range(1, TWEEN + 1):
            a = s / TWEEN
            blended = (1 - a) * disp[k - 1] + a * disp[k]
            sched.append((np.linalg.norm(blended, axis=1), k - 1, k))
        sched += [(norms[k], k, k + 1)] * HOLD
    sched += [(norms[8], 8, 9)] * ENDFREEZE
    n_frames = len(sched)

    # ---- figure ---------------------------------------------------------
    fig = plt.figure(figsize=(12.4, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                          left=0.04, right=0.95, top=0.89, bottom=0.21,
                          wspace=0.22)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1])
    fig.suptitle('Graph diffusion drives node features toward uniformity',
                 fontsize=14, color=NAVY, weight='bold', y=0.97)

    # left: graph (circular layout -- every node visible, edges as chords)
    axL.set_xlim(-0.10, 1.10)
    axL.set_ylim(-0.10, 1.10)
    axL.set_aspect('equal')
    axL.axis('off')
    for (i, k) in edges:
        axL.plot([pos[i, 0], pos[k, 0]], [pos[i, 1], pos[k, 1]],
                 color=GREY, lw=0.5, alpha=0.22, zorder=1)
    scat = axL.scatter(pos[:, 0], pos[:, 1], c=norms[0], cmap=NODE_CMAP,
                       vmin=vmin, vmax=vmax, s=300, edgecolors=NAVY,
                       linewidths=1.1, zorder=3)
    cb = fig.colorbar(scat, ax=axL, location='bottom', fraction=0.045, pad=0.04)
    cb.set_label('node-feature norm (scale-normalised)', fontsize=9)
    klabel = axL.text(0.04, 0.97, '', fontsize=16, color=NAVY, weight='bold',
                      transform=axL.transAxes)

    # right: over-smoothing proxies vs K (representative graph)
    axR.set_xlabel('GCN propagation depth  $K$', fontsize=12)
    axR.set_ylabel('Dirichlet energy', color=NAVY, fontsize=12)
    axR.set_xticks(Ks)
    axR.set_xlim(-0.3, 8.3)
    axR.set_ylim(0, max(dr) * 1.12)
    axR.tick_params(axis='y', labelcolor=NAVY)
    axR.axvspan(0.0, 2.0, color=TEAL, alpha=0.10)
    axR2 = axR.twinx()
    axR2.set_ylabel('Mean pairwise cosine similarity', color=ACCENT, fontsize=12)
    axR2.set_ylim(0, max(cos) * 1.18)
    axR2.tick_params(axis='y', labelcolor=ACCENT)
    lineD, = axR.plot([], [], 'o-', color=NAVY, lw=2.4, ms=6)
    lineC, = axR2.plot([], [], 's--', color=ACCENT, lw=2.4, ms=6)
    rep_drop = 100 * (dr[2] - dr[0]) / abs(dr[0])
    ann = axR.annotate(
        f'exp03 classification accuracy\nfalls fastest here\n'
        f'(Dirichlet energy $-${abs(rep_drop):.0f}% by $K=2$)',
        xy=(1.0, dr[1]), xytext=(2.85, max(dr) * 0.52),
        fontsize=9.5, color=TEAL, alpha=0.0,
        arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3))

    set_prog = add_progress_bar(
        fig, 'Animation 1 / 3  -  message passing as discrete diffusion')
    fig.text(0.5, 0.085,
             f'34-node SHAPE functional graph  -  GCN operator = (normalised '
             f'adjacency)$^K$  -  representative of {len(valid)} valid graphs; '
             f'aggregate Dirichlet drop {abs(agg_drop):.0f}%  -  {source}',
             ha='center', fontsize=8, color=GREY)

    def update(frame):
        frame = int(np.clip(frame, 0, n_frames - 1))
        node_norms, kint, reveal = sched[frame]
        scat.set_array(node_norms)
        klabel.set_text(f'$K = {kint}$')
        lineD.set_data(Ks[:reveal], dr[:reveal])
        lineC.set_data(Ks[:reveal], cos[:reveal])
        ann.set_alpha(float(np.clip((reveal - 3) / 2.0, 0.0, 1.0)))
        set_prog(frame / (n_frames - 1))
        return ()

    base = os.path.join(outdir, 'lsm_graph_diffusion')
    return render(fig, update, n_frames, n_frames - 1, base, fmt, fps, dpi)


# ================================================================
# Animation 2 -- Membrane potential + spike raster (two conditions)
# ================================================================
def animate_raster(outdir, fmt, fps, dpi):
    print("\n[2/3] Membrane potential + spike raster", flush=True)
    T = 150
    X_list, y = generate_temporal_task(n_per_class=60, seed=42)
    res = LIFReservoirMC(N_INPUT, N_RES, seed=SEED)        # identical reservoir
    x0, x1 = X_list[0], X_list[60]                         # one trial per class
    sp0, mem0 = res.forward(x0)
    sp1, mem1 = res.forward(x1)
    erp0, erp1 = x0[:, :5].mean(1), x1[:, :5].mean(1)

    # example neuron: a few spikes, concentrated near the burst window, so
    # integrate->fire->reset is legible
    tot = sp0.sum(0) + sp1.sum(0)
    cand = np.where((tot >= 3) & (tot <= 9))[0]
    if cand.size:
        burst = (sp0[15:85] + sp1[15:85]).sum(0)[cand]
        ex = int(cand[int(np.argmax(burst))])
    else:
        ex = int(np.argmin(np.abs(tot - 5)))
    mvmax = float(np.percentile(np.concatenate([mem0, mem1]), 99)) or 1.0

    def smooth(v, w=5):
        k = np.ones(w) / w
        return np.convolve(v, k, mode='same')
    fr0, fr1 = smooth(sp0.sum(1)), smooth(sp1.sum(1))
    fr_max = max(fr0.max(), fr1.max()) * 1.15 + 1e-6
    bin_edges = [10, 20, 30, 40, 50, 60, 70]               # BSC6 window 10..70

    HOLD0, ENDFREEZE = 8, 30
    n_frames = HOLD0 + T + ENDFREEZE

    fig = plt.figure(figsize=(12.6, 9.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.55, 1.65, 1.0],
                          left=0.07, right=0.94, top=0.91, bottom=0.085,
                          hspace=0.40, wspace=0.18)
    fig.suptitle('The LIF reservoir converts ERP timing into sparse spikes',
                 fontsize=14, color=NAVY, weight='bold', y=0.975)

    cols = [
        dict(name='Class 0  -  early burst', col=C0, erp=erp0, mem=mem0,
             sp=sp0, fr=fr0),
        dict(name='Class 1  -  late burst', col=C1, erp=erp1, mem=mem1,
             sp=sp1, fr=fr1),
    ]
    art = []
    for c, d in enumerate(cols):
        # row 0: input ERP + example-neuron membrane (twin axis)
        ax_in = fig.add_subplot(gs[0, c])
        ax_in.set_title(d['name'], color=d['col'], fontsize=12, weight='bold')
        ax_in.set_xlim(0, T)
        ax_in.set_ylim(min(d['erp'].min(), 0) - 0.1, d['erp'].max() * 1.15 + 0.1)
        ax_in.set_ylabel('input ERP', color=d['col'], fontsize=9)
        ax_in.tick_params(labelbottom=False)
        l_erp, = ax_in.plot([], [], color=d['col'], lw=1.6)
        ax_mem = ax_in.twinx()
        ax_mem.set_ylim(0, mvmax * 1.1)
        ax_mem.set_ylabel(f'neuron {ex}\nmembrane', color=TEAL, fontsize=8)
        ax_mem.tick_params(axis='y', labelcolor=TEAL, labelsize=7)
        ax_mem.axhline(THRESHOLD, color=GREY, ls='--', lw=1.0)
        ax_mem.text(2, THRESHOLD + mvmax * 0.04, r'$\theta$', color=GREY,
                    fontsize=9)
        l_mv, = ax_mem.plot([], [], color=TEAL, lw=1.3)
        l_msp, = ax_mem.plot([], [], 'o', color='#C0392B', ms=4)
        cur_in = ax_in.axvline(0, color=GREY, lw=1.0, alpha=0.7)

        # row 1: membrane heatmap (256 neurons)
        ax_h = fig.add_subplot(gs[1, c])
        ax_h.imshow(d['mem'].T, aspect='auto', origin='lower',
                    extent=[0, T, 0, N_RES], cmap='magma', vmin=0, vmax=mvmax)
        ax_h.set_ylabel('reservoir neuron', fontsize=9)
        ax_h.tick_params(labelbottom=False)
        if c == 0:
            ax_h.text(0.015, 0.92, 'membrane potential', transform=ax_h.transAxes,
                      fontsize=8.5, color='white')
        cover_h = ax_h.add_patch(Rectangle((0, 0), T, N_RES, facecolor='white',
                                           alpha=0.92, zorder=4))
        cur_h = ax_h.axvline(0, color='white', lw=1.2, zorder=5)

        # row 2: spike raster with BSC6 bins
        ax_r = fig.add_subplot(gs[2, c])
        ax_r.imshow(d['sp'].T, aspect='auto', origin='lower',
                    extent=[0, T, 0, N_RES], cmap='binary', vmin=0, vmax=1)
        ax_r.set_ylabel('reservoir neuron', fontsize=9)
        ax_r.tick_params(labelbottom=True)
        for be in bin_edges:
            ax_r.axvline(be, color=TEAL, lw=0.9, alpha=0.55)
        binfo = []
        for b in range(6):
            tx = ax_r.text((bin_edges[b] + bin_edges[b + 1]) / 2, N_RES * 1.04,
                           f'bin{b + 1}\n0', ha='center', va='bottom',
                           fontsize=7.5, color=TEAL)
            binfo.append(tx)
        cover_r = ax_r.add_patch(Rectangle((0, 0), T, N_RES, facecolor='white',
                                           alpha=0.92, zorder=4))
        cur_r = ax_r.axvline(0, color='#C0392B', lw=1.2, zorder=5)

        art.append(dict(d=d, l_erp=l_erp, l_mv=l_mv, l_msp=l_msp, cur_in=cur_in,
                        cover_h=cover_h, cur_h=cur_h, cover_r=cover_r,
                        cur_r=cur_r, binfo=binfo))

    # row 3: shared population firing rate
    ax_fr = fig.add_subplot(gs[3, :])
    ax_fr.set_xlim(0, T)
    ax_fr.set_ylim(0, fr_max)
    ax_fr.set_xlabel('time step', fontsize=9)
    ax_fr.set_ylabel('population\nfiring rate', fontsize=9)
    ax_fr.set_title('Instantaneous population firing rate (both conditions)',
                    fontsize=10, color=NAVY)
    l_fr0, = ax_fr.plot([], [], color=C0, lw=1.8, label='class 0 (early)')
    l_fr1, = ax_fr.plot([], [], color=C1, lw=1.8, label='class 1 (late)')
    ax_fr.legend(fontsize=8, loc='upper right')
    cur_fr = ax_fr.axvline(0, color=GREY, lw=1.0, alpha=0.7)

    tt = np.arange(T)
    set_prog = add_progress_bar(
        fig, 'Animation 2 / 3  -  membrane integration, firing, BSC6 binning')

    def update(frame):
        frame = int(np.clip(frame, 0, n_frames - 1))
        t = int(np.clip(frame - HOLD0 + 1, 1, T))
        for a in art:
            d = a['d']
            a['l_erp'].set_data(tt[:t], d['erp'][:t])
            a['l_mv'].set_data(tt[:t], d['mem'][:t, ex])
            stimes = np.where(d['sp'][:t, ex] > 0)[0]
            a['l_msp'].set_data(stimes, np.full(stimes.size, THRESHOLD))
            for ln in ('cur_in', 'cur_h', 'cur_r'):
                a[ln].set_xdata([t, t])
            a['cover_h'].set_x(t)
            a['cover_h'].set_width(T - t)
            a['cover_r'].set_x(t)
            a['cover_r'].set_width(T - t)
            for b in range(6):
                lo, hi = bin_edges[b], min(t, bin_edges[b + 1])
                cnt = int(d['sp'][lo:hi].sum()) if hi > lo else 0
                a['binfo'][b].set_text(f'bin{b + 1}\n{cnt}')
        l_fr0.set_data(tt[:t], fr0[:t])
        l_fr1.set_data(tt[:t], fr1[:t])
        cur_fr.set_xdata([t, t])
        set_prog(frame / (n_frames - 1))
        return ()

    base = os.path.join(outdir, 'lsm_membrane_raster')
    return render(fig, update, n_frames, n_frames - 1, base, fmt, fps, dpi)


# ================================================================
# Animation 3 -- BSC6 feature trajectory + linear readout (Koopman lens)
# ================================================================
def animate_trajectory(outdir, fmt, fps, dpi):
    print("\n[3/3] BSC6 feature trajectory + linear readout", flush=True)
    X_list, y = generate_temporal_task(n_per_class=100, seed=42)
    res = LIFReservoirMC(N_INPUT, N_RES, seed=SEED)
    spk = np.array([res.forward(x)[0] for x in X_list])    # (200, T, 256)

    # BSC6 over the window [10:70] -- the actual ARSPI-Net readout feature.
    # The discriminative information is temporal (WHEN spikes occur), so the
    # readout space must keep the 6 bins; a time-average would destroy it.
    W0, W1, NB = 10, 70, 6
    bw = (W1 - W0) // NB                                   # bin width = 10

    def partial_bsc6(spk_T256, t):
        """Cumulative BSC6 with bins filled only up to time t."""
        feats = []
        for b in range(NB):
            lo = W0 + b * bw
            hi = min(max(t, lo), lo + bw)
            feats.append(spk_T256[lo:hi].sum(axis=0))
        return np.concatenate(feats)                       # (1536,)

    bsc = np.array([partial_bsc6(s, W1) for s in spk])     # (200, 1536) full
    scaler = StandardScaler().fit(bsc)
    Xs = scaler.transform(bsc)
    clf = LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
    clf.fit(Xs, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = float(np.mean([
        balanced_accuracy_score(
            y[te],
            LogisticRegression(C=0.1, solver='liblinear', max_iter=1000)
            .fit(Xs[tr], y[tr]).predict(Xs[te]))
        for tr, te in skf.split(Xs, y)]))

    # projection plane: axis 1 = the readout's own decision direction,
    # axis 2 = leading residual (PCA) direction orthogonal to it
    coef = clf.coef_[0]
    cnorm = np.linalg.norm(coef) + 1e-12
    u = coef / cnorm
    resid = Xs - np.outer(Xs @ u, u)
    v = PCA(n_components=1).fit(resid).components_[0]
    v = v / (np.linalg.norm(v) + 1e-12)

    def project(raw_bsc):
        Z = scaler.transform(np.atleast_2d(raw_bsc))
        return np.column_stack([Z @ u, Z @ v])

    feat = project(bsc)                                    # (200, 2)
    mean0, mean1 = spk[y == 0].mean(0), spk[y == 1].mean(0)
    ts = np.arange(W0, W1 + 1)
    Z0 = project(np.array([partial_bsc6(mean0, t) for t in ts]))   # (61, 2)
    Z1 = project(np.array([partial_bsc6(mean1, t) for t in ts]))

    # one-step linear (DMD / Koopman) operator on the readout-plane trajectory
    Zin = np.vstack([Z0[:-1], Z1[:-1]])
    Zout = np.vstack([Z0[1:], Z1[1:]])
    Amat, *_ = np.linalg.lstsq(Zin, Zout, rcond=None)      # z_next ~ z @ Amat
    b_x = -clf.intercept_[0] / cnorm                       # boundary: x = b_x

    allZ = np.vstack([Z0, Z1, feat, [[b_x, 0.0]]])
    xlo, xhi = allZ[:, 0].min(), allZ[:, 0].max()
    ylo, yhi = allZ[:, 1].min(), allZ[:, 1].max()
    mx = 0.14 * (xhi - xlo) + 1e-6
    my = 0.18 * (yhi - ylo) + 1e-6

    nidx = len(ts)                                         # 61
    HOLD0, PERSTEP, HOLD, BOUND, ARROW, ENDFREEZE = 6, 2, 12, 24, 24, 32
    n_frames = HOLD0 + nidx * PERSTEP + HOLD + BOUND + ARROW + ENDFREEZE

    fig = plt.figure(figsize=(10.6, 8.0))
    ax = fig.add_axes([0.115, 0.13, 0.83, 0.76])
    ax.set_xlim(xlo - mx, xhi + mx)
    ax.set_ylim(ylo - my, yhi + my)
    ax.set_xlabel('linear-readout axis  (logistic decision direction)',
                  fontsize=11)
    ax.set_ylabel('leading residual axis  (orthogonal PC)', fontsize=11)
    ax.set_title('Reservoir lifting: the BSC6 feature becomes linearly '
                 'separable', fontsize=13, color=NAVY, weight='bold', pad=12)

    # decision half-planes (faded in with the boundary)
    span0 = ax.axvspan(xlo - mx, b_x, color=C0, alpha=0.0)
    span1 = ax.axvspan(b_x, xhi + mx, color=C1, alpha=0.0)

    # per-trial full-window BSC6 clouds
    ax.scatter(feat[y == 0, 0], feat[y == 0, 1], s=22, color=C0, alpha=0.35,
               edgecolors='none')
    ax.scatter(feat[y == 1, 0], feat[y == 1, 1], s=22, color=C1, alpha=0.35,
               edgecolors='none')

    lineT0, = ax.plot([], [], '-', color=C0, lw=2.8,
                      label='class 0 (early burst)')
    lineT1, = ax.plot([], [], '-', color=C1, lw=2.8,
                      label='class 1 (late burst)')
    headT0, = ax.plot([], [], 'o', color=C0, ms=13, mec='white', mew=1.5)
    headT1, = ax.plot([], [], 'o', color=C1, ms=13, mec='white', mew=1.5)
    ax.plot(Z0[0, 0], Z0[0, 1], '*', color=NAVY, ms=17, mec='white', mew=1.0,
            zorder=5)
    ax.annotate('empty feature\n(both classes identical)',
                xy=(Z0[0, 0], Z0[0, 1]),
                xytext=(Z0[0, 0], Z0[0, 1] + 0.11 * (yhi - ylo)),
                fontsize=8.5, color=GREY, ha='center', va='bottom')

    bound, = ax.plot([b_x, b_x], [ylo - my, yhi + my], color=NAVY, lw=2.4,
                     ls='--', alpha=0.0)
    btxt = ax.text(0.985, 0.035,
                   f'linear readout boundary  -  5-fold CV balanced '
                   f'accuracy {acc:.0%}',
                   transform=ax.transAxes, ha='right', fontsize=9.5,
                   color=NAVY, alpha=0.0)

    GAIN = 5.0
    sel = np.arange(2, nidx - 1, 6)

    def arrows(Z):
        d = GAIN * (Z[sel] @ Amat - Z[sel])
        return Z[sel, 0], Z[sel, 1], d[:, 0], d[:, 1]
    q0 = ax.quiver(*arrows(Z0), color=C0, alpha=0.0, width=0.004,
                   scale=1.0, scale_units='xy', angles='xy')
    q1 = ax.quiver(*arrows(Z1), color=C1, alpha=0.0, width=0.004,
                   scale=1.0, scale_units='xy', angles='xy')
    ktxt = ax.text(0.015, 0.035,
                   f'one-step linear operator  $z_{{t+1}}\\approx\\hat A z_t$  '
                   f'(DMD, arrows x{GAIN:.0f})', transform=ax.transAxes,
                   fontsize=9.5, color=GREY, alpha=0.0)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    set_prog = add_progress_bar(
        fig, 'Animation 3 / 3  -  why a linear readout suffices (Koopman lens)')

    def update(frame):
        frame = int(np.clip(frame, 0, n_frames - 1))
        i = int(np.clip((frame - HOLD0) // PERSTEP, 0, nidx - 1))
        lineT0.set_data(Z0[:i + 1, 0], Z0[:i + 1, 1])
        lineT1.set_data(Z1[:i + 1, 0], Z1[:i + 1, 1])
        headT0.set_data([Z0[i, 0]], [Z0[i, 1]])
        headT1.set_data([Z1[i, 0]], [Z1[i, 1]])
        f_b = frame - HOLD0 - nidx * PERSTEP - HOLD
        ba = float(np.clip(f_b / BOUND, 0.0, 1.0))
        bound.set_alpha(ba)
        btxt.set_alpha(ba)
        span0.set_alpha(0.07 * ba)
        span1.set_alpha(0.07 * ba)
        aa = float(np.clip((f_b - BOUND) / ARROW, 0.0, 1.0))
        q0.set_alpha(0.75 * aa)
        q1.set_alpha(0.75 * aa)
        ktxt.set_alpha(aa)
        set_prog(frame / (n_frames - 1))
        return ()

    base = os.path.join(outdir, 'lsm_state_trajectory')
    return render(fig, update, n_frames, n_frames - 1, base, fmt, fps, dpi)


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser(
        description='LSM-in-action animations for the ARSPI-Net defense.')
    ap.add_argument('--outdir', default='pictures/animations')
    ap.add_argument('--only', default='all',
                    choices=['diffusion', 'raster', 'trajectory', 'all'])
    ap.add_argument('--format', default='both', choices=['gif', 'mp4', 'both'])
    ap.add_argument('--fps', type=int, default=18)
    ap.add_argument('--dpi', type=int, default=100)
    ap.add_argument('--pkl', default=None,
                    help='real shape_features_211.pkl or X_ds .npz extract '
                         '(animation 1); synthetic fallback if omitted')
    ap.add_argument('--n', type=int, default=96,
                    help='observations used to fit per-channel PCA (anim 1)')
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    outdir = args.outdir if os.path.isabs(args.outdir) \
        else os.path.join(repo_root, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    try:
        ffmpeg = __import__('imageio_ffmpeg').get_ffmpeg_exe()
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    except Exception as e:                                      # noqa: BLE001
        print(f"[warn] imageio-ffmpeg unavailable ({e}); MP4 may fail.")

    print("=" * 64)
    print("ARSPI-Net defense animations")
    print(f"  outdir={outdir}  only={args.only}  format={args.format}  "
          f"fps={args.fps}  dpi={args.dpi}")
    print("=" * 64)
    t0 = time.time()

    if args.only in ('diffusion', 'all'):
        animate_diffusion(outdir, args.pkl, args.n, args.format,
                          args.fps, args.dpi)
    if args.only in ('raster', 'all'):
        animate_raster(outdir, args.format, args.fps, args.dpi)
    if args.only in ('trajectory', 'all'):
        animate_trajectory(outdir, args.format, args.fps, args.dpi)

    print("\n" + "=" * 64)
    print(f"done in {time.time() - t0:.0f}s  ->  {outdir}")
    print("=" * 64)


if __name__ == '__main__':
    sys.exit(main())
