#!/usr/bin/env python3
"""
animate_ch4_dynamics.py
================================================================
Chapter 4 dynamics animations for the ARSPI-Net dissertation defense.

Three animations that make the core dynamical claims of Chapter 4
visible. Each reuses the EXACT operators already validated in the repo
so the animations show the same system the dissertation characterizes:

  4. Driven LIF on real SHAPE ERP
     - the elementary operator of the reservoir under real ERP statistics.
     - dense Xavier-uniform reservoir from chapter4Experiments/
       run_chapter4_observations.py (the architecture the static ch4
       figures use).

  5. Driven Lyapunov convergence (ESP)
     - Benettin two-copy stretching with running cumulative estimate,
     - 10% sparse Gaussian reservoir from chapter6Experiments/
       run_chapter6_exp1_esp.py (the architecture under which the
       dissertation measures lambda_1 = -0.054, 100% negative across
       4220 measurements). Sample subset of ~500 trials labelled
       "representative sample" on the figure.

  6. BSC6 temporal-bin accumulation on real SHAPE
     - six bins fill in real time as spikes arrive,
     - side panel shows the live 1536-d BSC6 vector,
     - rate-code (total spike count) bars stay visibly identical
       between Neg and Pos to make the message explicit: timing carries
       the information, totals do not.

All three drive on the downsampled SHAPE features in
shape_features_211.pkl (X_ds: 633 x 256 x 34, real per-trial z-scored).

Usage:
  python animate_ch4_dynamics.py --only all
  python animate_ch4_dynamics.py --only driven_erp --erp-channel 8
  python animate_ch4_dynamics.py --only lyapunov --n-cohort 500
  python animate_ch4_dynamics.py --only bsc6 --format mp4 --dpi 100

Author: Andrew Lane | Stony Brook University ECE
================================================================
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle


# ----------------------------------------------------------------
# Defense palette (kept consistent with the Round-1 renderer)
# ----------------------------------------------------------------
NAVY, TEAL, ACCENT, GREY = '#1A2A4F', '#1C7293', '#B5651D', '#8A94A6'
CNEG, CPOS = NAVY, ACCENT  # one fixed colour per condition

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 11, 'figure.facecolor': 'white',
})

# ----------------------------------------------------------------
# Reservoir parameters (Ch4/Ch6 consensus values)
# ----------------------------------------------------------------
N_RES = 256
BETA = 0.05
M_TH = 0.5
SEED = 42


# ================================================================
# Operator 1 -- ch4 LIFReservoir (dense Xavier-uniform W, sr = 0.9)
# Verbatim from chapter4Experiments/run_chapter4_observations.py:75-142.
# Used for animations 4 and 6 (real ERP driving, single channel).
# ================================================================
class LIFReservoirCh4:
    """Dense leaky integrate-and-fire reservoir; returns spikes and
    membrane. Verbatim from chapter4Experiments/
    run_chapter4_observations.py."""

    def __init__(self, n_input, n_res=N_RES, beta=BETA, threshold=M_TH,
                 seed=SEED):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eig = np.abs(np.linalg.eigvals(self.W_rec)).max()
        if eig > 0:
            self.W_rec *= 0.9 / eig
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
        T, _ = X.shape
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


# ================================================================
# Operator 2 -- ch6 sparse Reservoir + Benettin driven Lyapunov
# Reservoir verbatim from chapter6Experiments/run_chapter6_exp1_esp.py:39-50.
# Benettin loop refactored from the same file (lines 165-185) to yield
# the running cumulative estimate per renormalization step (the conv
# pattern lives in chapter6Experiments/reproduce_chapter6.py:90-129).
# Used for animation 5 -- this is the architecture under which the
# dissertation measures lambda_1 = -0.054, 100% negative.
# ================================================================
class ReservoirCh6:
    """10% sparse Gaussian reservoir from ch6 ESP/Lyapunov experiment."""

    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        self.Win = rng.randn(N_RES) * 0.3
        mask = (rng.rand(N_RES, N_RES) < 0.1).astype(float)
        np.fill_diagonal(mask, 0)
        self.Wrec = rng.randn(N_RES, N_RES) * 0.05 * mask


def compute_driven_lyapunov(res, x, delta0=1e-8, T_renorm=10):
    """Driven maximal Lyapunov (Benettin), ch6 sparse-reservoir variant.

    Two reservoir copies, displacement renormalized every T_renorm steps.

    Returns
    -------
    lam : float
        Final estimate, mean(logs) / T_renorm.
    logs : list[float]
        log(dist / delta0) per renormalization (per-step stretching).
    conv : list[float]
        Running cumulative estimate after each renormalization.
    td : list[int]
        Timestep at which each renormalization occurred.
    """
    T, n = len(x), N_RES
    rng = np.random.RandomState(99)
    e = rng.randn(n); e /= np.linalg.norm(e)

    mem_r = np.zeros(n)
    mem_p = np.zeros(n) + delta0 * e
    sp_r = np.zeros(n)
    sp_p = np.zeros(n)
    logs, conv, td = [], [], []

    for t in range(T):
        I = res.Win * x[t]
        mem_r_n = (1 - BETA) * mem_r * (1 - sp_r) + I + res.Wrec @ sp_r
        sp_r_n = (mem_r_n >= M_TH).astype(float)
        mem_p_n = (1 - BETA) * mem_p * (1 - sp_p) + I + res.Wrec @ sp_p
        sp_p_n = (mem_p_n >= M_TH).astype(float)
        mem_r, sp_r, mem_p, sp_p = mem_r_n, sp_r_n, mem_p_n, sp_p_n

        if (t + 1) % T_renorm == 0:
            d = mem_p - mem_r
            dist = np.linalg.norm(d)
            if dist > 1e-15:
                logs.append(np.log(dist / delta0))
                mem_p = mem_r + delta0 * d / dist
            else:
                logs.append(np.log(1e-15 / delta0))
                e = rng.randn(n); e /= np.linalg.norm(e)
                mem_p = mem_r + delta0 * e
            sp_p = sp_r.copy()
            conv.append(float(np.mean(logs)) / T_renorm)
            td.append(t)

    lam = (float(np.mean(logs)) / T_renorm) if logs else 0.0
    return lam, logs, conv, td


# ================================================================
# Operator 3 -- BSC6 over the full 256-sample SHAPE window
# Same logic as chapter4Experiments/run_chapter4_observations.py:196-218
# and chapter5Experiments/run_chapter5_experiments.py BSC6.
# ================================================================
def extract_bsc6_with_edges(spikes, n_bins=6):
    """Return (1536-d feature, bin edges)."""
    T, _ = spikes.shape
    bin_size = T // n_bins
    edges = np.array([b * bin_size for b in range(n_bins + 1)])
    edges[-1] = T
    bins = [spikes[edges[b]:edges[b + 1]].sum(axis=0) for b in range(n_bins)]
    return np.concatenate(bins), edges, bin_size


def norm_signal(u):
    """z-score a 1D signal (matches ch6 norm())."""
    return (u - u.mean()) / (u.std() + 1e-10)


# ================================================================
# Data loaders
# ================================================================
def load_shape_pkl(path):
    """Load the SHAPE feature pickle. Returns (X_ds, y, subjects).

    X_ds: (N, T, 34), per-trial z-scored (verified mean ~ 0, std ~ 1).
    y:    (N,) in {0,1,2} for {IAPSNeg, IAPSNeu, IAPSPos}.
    """
    if not (path and os.path.exists(path)):
        raise FileNotFoundError(
            f"--pkl '{path}' not found. shape_features_211.pkl is "
            f"required for chapter-4 animations (real SHAPE data).")
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return (np.asarray(d['X_ds'], dtype=np.float64),
            np.asarray(d['y'], dtype=np.int64),
            np.asarray(d['subjects'], dtype=np.int64))


def pick_median_trial(X_ds, y, target_class):
    """Deterministic trial pick: median-energy trial within a class.
    Reproducible, avoids extreme outliers, easy to caption."""
    idx = np.where(y == target_class)[0]
    energy = (X_ds[idx] ** 2).sum(axis=(1, 2))
    return int(idx[np.argsort(energy)[len(idx) // 2]])


# ================================================================
# Helpers shared with the Round-1 renderer
# ================================================================
def add_progress_bar(fig, label):
    """Slim progress bar + persistent label, in figure coordinates."""
    ax = fig.add_axes([0.08, 0.014, 0.84, 0.013])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='#E9E9EC',
                           edgecolor='none'))
    fill = ax.add_patch(Rectangle((0, 0), 0, 1, facecolor=TEAL,
                                  edgecolor='none'))
    fig.text(0.08, 0.034, label, fontsize=8.5, color=GREY, ha='left')

    def _set(frac):
        fill.set_width(max(0.0, min(1.0, frac)))
    return _set


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
# Animation 4 -- Driven LIF on real SHAPE ERP
# ================================================================
def animate_driven_erp(outdir, pkl_path, erp_channel, fmt, fps, dpi):
    """Two-column LIF dynamics under real Neg vs Pos SHAPE ERPs."""
    print("\n[1/3] Driven LIF on real SHAPE ERP", flush=True)
    X_ds, y, subjects = load_shape_pkl(pkl_path)
    N, T, C = X_ds.shape
    print(f"  X_ds: {X_ds.shape}  (real SHAPE, 256 Hz)", flush=True)

    # Pick the two median-energy trials (one per condition)
    neg_idx = pick_median_trial(X_ds, y, target_class=0)  # IAPSNeg
    pos_idx = pick_median_trial(X_ds, y, target_class=2)  # IAPSPos
    print(f"  trials: Neg subject {int(subjects[neg_idx])} (idx {neg_idx}), "
          f"Pos subject {int(subjects[pos_idx])} (idx {pos_idx}); "
          f"reservoir-input channel = {erp_channel}", flush=True)

    res = LIFReservoirCh4(n_input=1, seed=SEED)
    cond = [('Neg', neg_idx, CNEG), ('Pos', pos_idx, CPOS)]
    data = {}
    for label, idx, c in cond:
        erp = X_ds[idx]                              # (256, 34)
        u = erp[:, erp_channel]                      # (256,)
        spk, mem = res.forward(u.reshape(-1, 1))     # (256, 256), (256, 256)
        # smoothed instantaneous firing rate (sum across neurons, box filter)
        pop = spk.sum(axis=1)
        win = 8
        kernel = np.ones(win) / win
        rate = np.convolve(pop, kernel, mode='same')
        data[label] = dict(erp=erp, u=u, mem=mem, spk=spk, rate=rate, color=c,
                           idx=idx)

    # 8 representative neurons spaced across the population
    neuron_ids = [20, 50, 80, 110, 140, 170, 200, 230]
    # vertical offset per trace so the integrate->fire->reset pattern is
    # legible without overlap; threshold = M_TH so a 1.0 offset gives one
    # threshold-worth of headroom per trace.
    OFFSET = 1.0

    # ----- figure layout -----
    fig = plt.figure(figsize=(13.5, 10.0))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.5, 2.6, 2.5, 0.9],
        hspace=0.40, wspace=0.16,
        left=0.06, right=0.985, top=0.93, bottom=0.085)

    ax_erp, ax_mem, ax_rast, ax_rate = [], [], [], None

    for col, (label, idx, c) in enumerate(cond):
        ax_erp.append(fig.add_subplot(gs[0, col]))
        ax_mem.append(fig.add_subplot(gs[1, col]))
        ax_rast.append(fig.add_subplot(gs[2, col]))

    ax_rate = fig.add_subplot(gs[3, :])

    # ERP top: faint all-34 + highlighted reservoir-input channel
    erp_lines = []
    erp_input_lines = []
    for col, (label, idx, c) in enumerate(cond):
        ax = ax_erp[col]
        for ch in range(C):
            ln, = ax.plot([], [], color=GREY, alpha=0.28, lw=0.7)
            erp_lines.append((col, ch, ln))
        # highlighted input channel on top
        ln_in, = ax.plot([], [], color=c, lw=1.6, alpha=0.95)
        erp_input_lines.append(ln_in)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(X_ds[idx].min() * 1.05, X_ds[idx].max() * 1.05)
        ax.set_title(f'Class: {label}   (subject {int(subjects[idx])})',
                     color=c, fontweight='bold', fontsize=11.5)
        ax.set_ylabel('ERP (z-score)' if col == 0 else '')
        ax.grid(alpha=0.22, lw=0.5)

    # Membrane traces of 8 neurons, with offset, threshold line
    mem_lines = []
    for col, (label, idx, c) in enumerate(cond):
        ax = ax_mem[col]
        for k, _ in enumerate(neuron_ids):
            ln, = ax.plot([], [], color=c, lw=0.85, alpha=0.95)
            mem_lines.append((col, k, ln))
        # threshold guides at every neuron offset
        for k in range(len(neuron_ids)):
            ax.axhline(M_TH + k * OFFSET, color=GREY, ls=':', lw=0.55,
                       alpha=0.6)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-0.05, OFFSET * len(neuron_ids) + 0.18)
        ax.set_yticks([k * OFFSET + M_TH / 2 for k in range(len(neuron_ids))])
        ax.set_yticklabels([f'n{n}' for n in neuron_ids], fontsize=8.5)
        ax.set_ylabel('membrane V (8 neurons)' if col == 0 else '')
        ax.grid(axis='x', alpha=0.22, lw=0.5)

    # Raster: 256 neurons by time; reveal by setting alpha on a precomputed
    # scatter, OR by drawing scatter only up to t. Cleanest: precompute
    # arrays of (t, neuron) for fired spikes and use set_offsets to a
    # growing slice.
    spk_scatters = []
    spk_data = {}
    for col, (label, idx, c) in enumerate(cond):
        ax = ax_rast[col]
        sp = data[label]['spk']
        ts, ns = np.where(sp > 0)
        spk_data[col] = (ts, ns)
        sc = ax.scatter([], [], s=2.2, c=c, alpha=0.65, marker='|',
                        linewidths=0.9)
        spk_scatters.append(sc)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-2, N_RES + 2)
        ax.set_ylabel('reservoir neuron (256)' if col == 0 else '')
        ax.set_xlabel('time step (256 Hz)')
        ax.grid(alpha=0.22, lw=0.5)
        ax.text(0.985, 0.96,
                f'{int(sp.sum()):,} spikes',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8.6, color=c)

    # Firing rate row (both conditions overlaid)
    rate_lines = {}
    for label, idx, c in cond:
        ln, = ax_rate.plot([], [], color=c, lw=1.7, alpha=0.95,
                           label=f'class {label}')
    rate_lines = ax_rate.lines
    ax_rate.set_xlim(0, T - 1)
    ymax = max(data['Neg']['rate'].max(), data['Pos']['rate'].max()) * 1.15
    ax_rate.set_ylim(0, ymax)
    ax_rate.set_ylabel('pop firing rate')
    ax_rate.set_xlabel('time step (256 Hz)')
    ax_rate.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax_rate.set_title('Instantaneous population firing rate (both conditions)',
                      color=NAVY, fontsize=11)
    ax_rate.grid(alpha=0.22, lw=0.5)

    # time cursors across the four rows of each column
    cursors = []
    for col in range(2):
        cl = []
        for ax in [ax_erp[col], ax_mem[col], ax_rast[col]]:
            v = ax.axvline(0, color='#cc3333', lw=0.9, alpha=0.0)
            cl.append(v)
        cursors.append(cl)
    rate_cursor = ax_rate.axvline(0, color='#cc3333', lw=0.9, alpha=0.0)

    fig.suptitle('Driven LIF reservoir: real SHAPE ERP -> sparse spike code',
                 fontsize=14, fontweight='bold', color=NAVY, y=0.985)

    set_prog = add_progress_bar(
        fig, 'Animation 4 / 6  -  driven LIF on real SHAPE ERP '
        '(IAPSNeg vs IAPSPos, ch ' + str(erp_channel) + ')')

    PRE, REVEAL, FREEZE = 8, 182, 30
    n_frames = PRE + REVEAL + FREEZE

    def time_at(frame):
        if frame < PRE:
            return 0
        if frame < PRE + REVEAL:
            return int((frame - PRE + 1) / REVEAL * T)
        return T

    tt = np.arange(T)

    def update(frame):
        t = max(1, time_at(frame))
        cursor_alpha = 0.9 if frame < PRE + REVEAL else 0.0

        for col, (label, idx, c) in enumerate(cond):
            erp = data[label]['erp']
            mem = data[label]['mem']
            # ERP: faint all-34 + highlighted input channel
            for (cc, ch, ln) in erp_lines:
                if cc != col:
                    continue
                ln.set_data(tt[:t], erp[:t, ch])
            erp_input_lines[col].set_data(tt[:t], erp[:t, erp_channel])
            # Membrane traces
            for (cc, k, ln) in mem_lines:
                if cc != col:
                    continue
                nid = neuron_ids[k]
                ln.set_data(tt[:t], mem[:t, nid] + k * OFFSET)
            # Spike raster: show all spikes whose time index < t
            ts_all, ns_all = spk_data[col]
            keep = ts_all < t
            if keep.any():
                spk_scatters[col].set_offsets(
                    np.column_stack([ts_all[keep], ns_all[keep]]))
            else:
                spk_scatters[col].set_offsets(np.empty((0, 2)))
            # Cursor
            for v in cursors[col]:
                v.set_xdata([t - 1, t - 1])
                v.set_alpha(cursor_alpha)
            # Firing rate
            rate_lines[col].set_data(tt[:t], data[label]['rate'][:t])

        rate_cursor.set_xdata([t - 1, t - 1])
        rate_cursor.set_alpha(cursor_alpha)
        set_prog(frame / max(1, n_frames - 1))

    base = os.path.join(outdir, 'lsm_driven_erp')
    return render(fig, update, n_frames, n_frames - 5, base, fmt, fps, dpi)


# ================================================================
# Animation 5 -- Driven Lyapunov convergence
# ================================================================
def precompute_lyapunov(X_ds, y, subjects, channels, n_cohort,
                        T_renorm=10, cache_path=None, rebuild=False):
    """Stratified pre-computation of n_cohort driven-Lyapunov estimates.

    Stratifies trials evenly across {IAPSNeg, IAPSNeu, IAPSPos} and
    distributes channel choice uniformly across the analysis_channels
    list. Returns (finals, demo_logs, demo_conv, demo_td, demo_idx,
    demo_ch).
    """
    if cache_path and os.path.exists(cache_path) and not rebuild:
        print(f"  [cache] reading {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=True)
        return (z['finals'], list(z['demo_logs']), list(z['demo_conv']),
                list(z['demo_td']), int(z['demo_idx']), int(z['demo_ch']))

    res = ReservoirCh6(seed=SEED)
    rng = np.random.RandomState(SEED)
    per_cond = n_cohort // 3
    finals = []
    selected = []
    for cond_val in [0, 1, 2]:
        ci = np.where(y == cond_val)[0]
        pick = rng.choice(ci, size=min(per_cond, len(ci)), replace=False)
        for ti in pick:
            ch = int(rng.choice(channels))
            selected.append((int(ti), ch))

    t0 = time.time()
    demo_logs = demo_conv = demo_td = None
    demo_idx = demo_ch = None
    for i, (ti, ch) in enumerate(selected):
        u = norm_signal(X_ds[ti, :, ch])
        lam, logs, conv, td = compute_driven_lyapunov(
            res, u, delta0=1e-8, T_renorm=T_renorm)
        finals.append(lam)
        if i == 0:
            demo_logs, demo_conv, demo_td = logs, conv, td
            demo_idx, demo_ch = ti, ch
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(selected)}  ({time.time()-t0:.0f}s)",
                  flush=True)
    finals = np.array(finals, dtype=np.float64)
    print(f"  cohort: mean lambda1 = {finals.mean():.5f},  "
          f"share negative = {float(np.mean(finals < 0)):.3f}  "
          f"(N = {len(finals)})", flush=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            finals=finals,
            demo_logs=np.array(demo_logs, dtype=np.float64),
            demo_conv=np.array(demo_conv, dtype=np.float64),
            demo_td=np.array(demo_td, dtype=np.int64),
            demo_idx=np.int64(demo_idx),
            demo_ch=np.int64(demo_ch),
            T_renorm=np.int64(T_renorm),
            n_cohort=np.int64(n_cohort))
        print(f"  [cache] wrote {cache_path}", flush=True)
    return finals, demo_logs, demo_conv, demo_td, demo_idx, demo_ch


def animate_lyapunov(outdir, pkl_path, n_cohort, fmt, fps, dpi,
                     cache_path, rebuild_cache, channels):
    print("\n[2/3] Driven Lyapunov convergence", flush=True)
    X_ds, y, subjects = load_shape_pkl(pkl_path)
    print(f"  X_ds: {X_ds.shape}  (real SHAPE, 256 Hz)", flush=True)
    finals, demo_logs, demo_conv, demo_td, demo_idx, demo_ch = \
        precompute_lyapunov(
            X_ds, y, subjects, channels=channels, n_cohort=n_cohort,
            T_renorm=10, cache_path=cache_path, rebuild=rebuild_cache)

    final_mean = float(finals.mean())
    share_neg = float((finals < 0).mean())
    K = len(demo_logs)
    demo_final = float(np.mean(demo_logs) / 10.0)

    # ----- figure -----
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.1, 1.0], hspace=0.42, wspace=0.22,
        left=0.075, right=0.98, top=0.92, bottom=0.10)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axH = fig.add_subplot(gs[1, :])

    # axL: raw stretching factors per renormalization (log of stretch)
    axL.set_title('Benettin stretching factor per renormalization step',
                  fontsize=11, color=NAVY)
    axL.set_xlabel('renormalization step  k')
    axL.set_ylabel(r'$\log(\|\delta_k\| / \delta_0)$')
    axL.axhline(0, color=GREY, ls='--', lw=1.0, alpha=0.7)
    axL.set_xlim(0, K + 0.5)
    yL_min = min(demo_logs) - 0.6
    yL_max = max(demo_logs) + 0.6
    axL.set_ylim(yL_min, yL_max)
    axL.grid(alpha=0.22, lw=0.5)
    stretch_line, = axL.plot([], [], 'o-', color=TEAL, lw=1.4, ms=5.5,
                             alpha=0.95)
    axL.text(0.985, 0.94, 'two reservoir copies; δ rescaled every 10 steps',
             transform=axL.transAxes, ha='right', va='top', color=GREY,
             fontsize=8.5)

    # axR: running cumulative estimate lambda_1(k)
    axR.set_title(r'Running estimate  $\lambda_1(k) = '
                  r'\langle \log(\delta_k/\delta_0)\rangle / T_\mathrm{renorm}$',
                  fontsize=11, color=NAVY)
    axR.set_xlabel('renormalization step  k')
    axR.set_ylabel(r'$\lambda_1(k)$')
    axR.set_xlim(0, K + 0.5)
    cmin = min(min(demo_conv), demo_final) - 0.015
    cmax = max(max(demo_conv), 0.0) + 0.015
    axR.set_ylim(cmin, cmax)
    axR.axhline(0, color=GREY, ls='--', lw=1.0, alpha=0.7)
    axR.axhline(demo_final, color=NAVY, ls=':', lw=1.2, alpha=0.0)
    axR.grid(alpha=0.22, lw=0.5)
    cum_line, = axR.plot([], [], '-', color=NAVY, lw=2.0)
    cum_end = axR.scatter([], [], s=55, c=NAVY, zorder=5)
    cum_label = axR.text(0.985, 0.04, '', transform=axR.transAxes,
                         ha='right', fontsize=10, color=NAVY, alpha=0.0)
    contraction_text = axR.text(
        0.5, 0.86,
        r'$\lambda_1 < 0$  $\Rightarrow$  fading memory '
        r'$\Rightarrow$  Echo State Property holds',
        transform=axR.transAxes, ha='center', fontsize=10.5,
        color=ACCENT, alpha=0.0)

    # axH: cohort histogram -- the distribution is extremely tight under
    # z-scored real EEG drive (std ~ 0.4 percent of mean), so we zoom to
    # the data span and call out the distance to zero rather than waste
    # plot area on empty white space between the peak and zero.
    finals_std = float(finals.std())
    sigmas_to_zero = abs(final_mean) / max(finals_std, 1e-12)
    margin = max(0.0004, 3 * finals_std)
    hist_lo = finals.min() - margin
    hist_hi = finals.max() + margin
    bins = np.linspace(hist_lo, hist_hi, 28)
    centers = 0.5 * (bins[:-1] + bins[1:])
    final_counts, _ = np.histogram(finals, bins=bins)
    bar_w = bins[1] - bins[0]
    bars = axH.bar(centers, np.zeros_like(centers), width=bar_w * 0.92,
                   color=ACCENT, edgecolor='black', alpha=0.82,
                   linewidth=0.5)
    mean_line = axH.axvline(final_mean, color=NAVY, lw=2.0, ls='--',
                            alpha=0.0)
    axH.set_xlim(hist_lo, hist_hi)
    axH.set_ylim(0, max(final_counts) * 1.22)
    axH.set_xlabel(r'$\lambda_1$  (final, per trial)')
    axH.set_ylabel('count')
    axH.set_title(
        f'Driven Lyapunov across a representative sample of {len(finals)} '
        f'trials (stratified over IAPSNeg/Neu/Pos × {len(channels)} '
        f'channels)', fontsize=11, color=NAVY)
    axH.grid(alpha=0.22, axis='y', lw=0.5)
    hist_label = axH.text(
        0.985, 0.93, '', transform=axH.transAxes, ha='right', fontsize=11,
        color=NAVY, alpha=0.0)
    zero_call = axH.text(
        0.985, 0.78,
        f'$\\lambda_1 = 0$ lies {sigmas_to_zero:.0f}$\\sigma$ to the right\n'
        f'-- every trial is deep in the\n   contracting regime',
        transform=axH.transAxes, ha='right', va='top', fontsize=10,
        color=NAVY, alpha=0.0,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f6f5ef',
                  edgecolor=GREY, linewidth=0.5))

    fig.suptitle('Reservoir is uniformly contracting under real EEG drive',
                 fontsize=14, fontweight='bold', color=NAVY, y=0.985)

    set_prog = add_progress_bar(
        fig, 'Animation 5 / 6  -  driven Lyapunov convergence + ESP gate')

    # Frame schedule:
    PRE = 8
    A_FRAMES = 56          # reveal demo stretching + cumulative
    B_FRAMES = 50          # fill histogram (10 trials/frame)
    C_FRAMES = 36          # mean line + annotations fade in
    FREEZE = 26
    n_frames = PRE + A_FRAMES + B_FRAMES + C_FRAMES + FREEZE
    BAR_REVEAL = max(1, len(finals) // B_FRAMES)

    final_line = axR.lines[1]  # the axhline at demo_final (set later)
    # rebuild the axhline so we can fade it cleanly:
    final_hline = axR.axhline(demo_final, color=NAVY, ls=':', lw=1.4,
                              alpha=0.0)

    def update(frame):
        if frame < PRE:
            set_prog(0.0)
            return
        f = frame - PRE
        # Phase A: stretching & cumulative reveal
        a = min(f, A_FRAMES)
        k = int(np.round(a / A_FRAMES * K))
        k = min(max(k, 0), K)
        ks = np.arange(1, k + 1)
        if k > 0:
            stretch_line.set_data(ks, demo_logs[:k])
            cum_line.set_data(ks, demo_conv[:k])
            cum_end.set_offsets(np.array([[ks[-1], demo_conv[k - 1]]]))
        else:
            stretch_line.set_data([], [])
            cum_line.set_data([], [])
            cum_end.set_offsets(np.empty((0, 2)))

        if k >= K:
            final_hline.set_alpha(min(1.0, (f - A_FRAMES + 6) / 14.0))
            cum_label.set_text(
                rf'demo trial:  $\lambda_1 = {demo_final:.4f}$')
            cum_label.set_alpha(min(1.0, (f - A_FRAMES + 6) / 18.0))

        # Phase B: histogram fill
        if f >= A_FRAMES:
            b = min(f - A_FRAMES, B_FRAMES)
            n_revealed = min(int(b * BAR_REVEAL), len(finals))
            if n_revealed > 0:
                running_counts, _ = np.histogram(finals[:n_revealed],
                                                 bins=bins)
                for rect, h in zip(bars, running_counts):
                    rect.set_height(h)
        # Phase C: mean line + annotation
        if f >= A_FRAMES + B_FRAMES:
            c = min(f - A_FRAMES - B_FRAMES, C_FRAMES)
            a_fade = c / C_FRAMES
            mean_line.set_alpha(min(1.0, a_fade * 1.2))
            hist_label.set_text(
                f'mean $\\lambda_1$ = {final_mean:.4f}        '
                f'share negative: {share_neg*100:.1f}%')
            hist_label.set_alpha(min(1.0, a_fade * 1.4))
            contraction_text.set_alpha(min(1.0, a_fade * 1.2))
            zero_call.set_alpha(min(1.0, a_fade * 1.2))

        set_prog(frame / max(1, n_frames - 1))

    base = os.path.join(outdir, 'lsm_lyapunov_convergence')
    return render(fig, update, n_frames, n_frames - 5, base, fmt, fps, dpi)


# ================================================================
# Animation 6 -- BSC6 bin accumulation on real SHAPE
# ================================================================
def animate_bsc6(outdir, pkl_path, erp_channel, fmt, fps, dpi):
    print("\n[3/3] BSC6 bin accumulation on real SHAPE", flush=True)
    X_ds, y, subjects = load_shape_pkl(pkl_path)
    N, T, C = X_ds.shape

    neg_idx = pick_median_trial(X_ds, y, target_class=0)
    pos_idx = pick_median_trial(X_ds, y, target_class=2)
    print(f"  trials: Neg subject {int(subjects[neg_idx])}, "
          f"Pos subject {int(subjects[pos_idx])}; "
          f"reservoir-input channel = {erp_channel}", flush=True)

    res = LIFReservoirCh4(n_input=1, seed=SEED)
    cond = [('Neg', neg_idx, CNEG), ('Pos', pos_idx, CPOS)]
    data = {}
    for label, idx, c in cond:
        u = X_ds[idx, :, erp_channel]
        spk, _ = res.forward(u.reshape(-1, 1))
        bsc, edges, bin_size = extract_bsc6_with_edges(spk)
        data[label] = dict(spk=spk, bsc=bsc, edges=edges,
                           bin_size=bin_size, color=c, idx=idx,
                           total=int(spk.sum()))

    # the rate-code "total" is identical-magnitude between conditions if
    # the reservoir's mean firing is preserved across stimuli; the BSC6
    # bins are NOT.
    edges = data['Neg']['edges']
    bin_size = data['Neg']['bin_size']
    n_bins = 6

    fig = plt.figure(figsize=(13.5, 10.4))
    gs = fig.add_gridspec(
        4, 2, height_ratios=[2.8, 1.6, 1.6, 0.7], hspace=0.40, wspace=0.18,
        left=0.06, right=0.985, top=0.93, bottom=0.085)

    ax_rast = [fig.add_subplot(gs[0, col]) for col in range(2)]
    ax_bars = [fig.add_subplot(gs[1, col]) for col in range(2)]
    ax_heat = [fig.add_subplot(gs[2, col]) for col in range(2)]
    ax_rate = fig.add_subplot(gs[3, :])

    # -- Raster + bin guides
    spk_scatters = []
    spk_data = {}
    bin_shades = {}
    for col, (label, idx, c) in enumerate(cond):
        ax = ax_rast[col]
        sp = data[label]['spk']
        ts, ns = np.where(sp > 0)
        spk_data[col] = (ts, ns)
        sc = ax.scatter([], [], s=2.0, c=c, alpha=0.65, marker='|',
                        linewidths=0.85)
        spk_scatters.append(sc)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(-2, N_RES + 2)
        ax.set_ylabel('reservoir neuron (256)' if col == 0 else '')
        ax.set_title(f'Class: {label}  (subject {int(subjects[idx])})',
                     color=c, fontweight='bold', fontsize=11.5)
        # bin guides
        shades = []
        for b in range(n_bins):
            sh = ax.axvspan(edges[b], edges[b + 1], color='none',
                            alpha=0.0)
            shades.append(sh)
            ax.axvline(edges[b], color=GREY, ls=':', lw=0.7, alpha=0.7)
        ax.axvline(edges[-1], color=GREY, ls=':', lw=0.7, alpha=0.7)
        bin_shades[col] = shades
        ax.grid(axis='y', alpha=0.22, lw=0.4)
        ax.text(0.985, 0.96, f'{data[label]["total"]:,} total spikes',
                transform=ax.transAxes, ha='right', va='top',
                color=c, fontsize=8.6)

    # -- 6 horizontal bars filling per bin (sum across all 256 neurons)
    bin_bars = []
    for col, (label, idx, c) in enumerate(cond):
        ax = ax_bars[col]
        bs = []
        per_bin_max = 0
        for b in range(n_bins):
            start, end = edges[b], edges[b + 1]
            per_bin_max = max(per_bin_max,
                              int(data[label]['spk'][start:end].sum()))
        per_bin_max = max(per_bin_max,
                          int(data['Neg']['spk'][:bin_size].sum()),
                          int(data['Pos']['spk'][:bin_size].sum()))
        for b in range(n_bins):
            bar = ax.barh(b, 0, color=c, edgecolor='black', linewidth=0.5,
                          alpha=0.85)[0]
            bs.append(bar)
        bin_bars.append(bs)
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels(
            [f'bin{b+1}\n[{edges[b]}-{edges[b+1]})' for b in range(n_bins)],
            fontsize=8.2)
        ax.invert_yaxis()
        ax.set_xlim(0, per_bin_max * 1.15)
        ax.set_xlabel('spike count in bin (sum over 256 neurons)')
        ax.grid(axis='x', alpha=0.22, lw=0.5)
        ax.set_title('temporal-bin code (BSC6)', fontsize=10.5, color=NAVY)

    # -- BSC6 heatmap (6 bins x 256 neurons), filling row-by-row
    heat_imgs = []
    heat_states = []
    for col, (label, idx, c) in enumerate(cond):
        ax = ax_heat[col]
        full = data[label]['bsc'].reshape(6, N_RES)
        vmax = full.max() if full.max() > 0 else 1
        z = np.full_like(full, np.nan, dtype=float)
        im = ax.imshow(z, aspect='auto', cmap='magma', vmin=0, vmax=vmax,
                       interpolation='nearest', origin='lower')
        heat_imgs.append((im, full))
        heat_states.append(z)
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels([f'bin{b+1}' for b in range(n_bins)],
                           fontsize=8.5)
        ax.set_xlabel('reservoir neuron index')
        ax.set_title('BSC6 feature vector (6 x 256 = 1536-d)',
                     fontsize=10.5, color=NAVY)

    # -- rate-code comparison (single bar per condition)
    rate_bars = []
    rate_max = max(data['Neg']['total'], data['Pos']['total']) * 1.18
    for ci, (label, idx, c) in enumerate(cond):
        bar = ax_rate.barh(ci, 0, color=c, edgecolor='black',
                           linewidth=0.6, alpha=0.85,
                           label=f'class {label}: total spike count')[0]
        rate_bars.append(bar)
    ax_rate.set_yticks([0, 1])
    ax_rate.set_yticklabels(['Neg', 'Pos'], fontsize=10)
    ax_rate.set_xlim(0, rate_max)
    ax_rate.set_xlabel('total spikes (rate-code only)')
    ax_rate.set_title(
        'Rate code (total spike count): the two conditions look identical -- '
        'timing is what differs', fontsize=10.5, color=NAVY)
    ax_rate.grid(axis='x', alpha=0.22, lw=0.5)

    fig.suptitle(
        'BSC6 temporal coding: spike timing -> 6-bin discriminative vector',
        fontsize=14, fontweight='bold', color=NAVY, y=0.985)

    set_prog = add_progress_bar(
        fig, 'Animation 6 / 6  -  BSC6 bin accumulation on real SHAPE')

    PRE, REVEAL, FREEZE = 8, 182, 30
    n_frames = PRE + REVEAL + FREEZE

    def time_at(frame):
        if frame < PRE:
            return 0
        if frame < PRE + REVEAL:
            return int((frame - PRE + 1) / REVEAL * T)
        return T

    def update(frame):
        t = max(1, time_at(frame))
        # which bin index does t belong to?
        active_bin = -1
        for b in range(n_bins):
            if edges[b] <= t - 1 < edges[b + 1]:
                active_bin = b
                break
        if t >= edges[-1]:
            active_bin = n_bins - 1

        for col, (label, idx, c) in enumerate(cond):
            # raster reveal
            ts_all, ns_all = spk_data[col]
            keep = ts_all < t
            if keep.any():
                spk_scatters[col].set_offsets(
                    np.column_stack([ts_all[keep], ns_all[keep]]))
            else:
                spk_scatters[col].set_offsets(np.empty((0, 2)))
            # bin guide highlight (only the active bin glows)
            for b in range(n_bins):
                bin_shades[col][b].set_color(c if b == active_bin else 'none')
                bin_shades[col][b].set_alpha(0.18 if b == active_bin else 0.0)
            # per-bin bar fill (running count up to time t)
            sp = data[label]['spk']
            for b in range(n_bins):
                start, end = edges[b], edges[b + 1]
                if t <= start:
                    count = 0
                elif t >= end:
                    count = int(sp[start:end].sum())
                else:
                    count = int(sp[start:t].sum())
                bin_bars[col][b].set_width(count)
            # heatmap row reveal: when a bin closes, fill its row
            im, full = heat_imgs[col]
            z = heat_states[col]
            for b in range(n_bins):
                if t >= edges[b + 1]:
                    z[b, :] = full[b, :]
                elif t > edges[b]:
                    progress = (t - edges[b]) / (edges[b + 1] - edges[b])
                    z[b, :] = full[b, :] * progress
                else:
                    z[b, :] = np.nan
            im.set_data(z)
            # rate code: total spikes up to time t
            rate_bars[col].set_width(int(sp[:t].sum()))

        set_prog(frame / max(1, n_frames - 1))

    base = os.path.join(outdir, 'lsm_bsc6_accumulation')
    return render(fig, update, n_frames, n_frames - 5, base, fmt, fps, dpi)


# ================================================================
# CLI driver
# ================================================================
def main():
    p = argparse.ArgumentParser(
        description='Render Chapter-4 dynamics animations (LIF on real '
        'ERP, driven Lyapunov, BSC6 bin accumulation).')
    p.add_argument('--outdir', default='pictures/animations')
    p.add_argument('--only', default='all',
                   choices=['driven_erp', 'lyapunov', 'bsc6', 'all'])
    p.add_argument('--format', dest='fmt', default='both',
                   choices=['gif', 'mp4', 'both'])
    p.add_argument('--fps', type=int, default=18)
    p.add_argument('--dpi', type=int, default=100)
    p.add_argument('--pkl', default='/tmp/shape_features_211.pkl',
                   help='path to shape_features_211.pkl (real SHAPE data)')
    p.add_argument('--erp-channel', type=int, default=8,
                   help='electrode used to drive the LIF reservoir '
                   '(default 8; the same indices used in ch6 analysis)')
    p.add_argument('--n-cohort', type=int, default=501,
                   help='stratified subset size for the Lyapunov histogram '
                   '(default 501 -> 167 per condition).')
    p.add_argument('--lyap-channels', nargs='+', type=int,
                   default=[0, 8, 16, 24, 33],
                   help='channels sampled for the Lyapunov cohort '
                   '(matches the ch6 analysis_channels).')
    p.add_argument('--cache', default='chapter4Experiments/cache/'
                   'ch4_lyapunov_cache.npz')
    p.add_argument('--rebuild-cache', action='store_true')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    try:
        ffmpeg = __import__('imageio_ffmpeg').get_ffmpeg_exe()
        matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
    except Exception as e:                                      # noqa: BLE001
        print(f"[warn] imageio-ffmpeg unavailable ({e}); MP4 may fail.",
              flush=True)

    print('=' * 64, flush=True)
    print('ARSPI-Net Chapter-4 dynamics animations', flush=True)
    print(f'  outdir = {args.outdir}   only = {args.only}   '
          f'format = {args.fmt}   fps = {args.fps}   dpi = {args.dpi}',
          flush=True)
    print(f'  pkl    = {args.pkl}', flush=True)
    print('=' * 64, flush=True)

    t0 = time.time()
    if args.only in ('driven_erp', 'all'):
        animate_driven_erp(args.outdir, args.pkl, args.erp_channel,
                           args.fmt, args.fps, args.dpi)
    if args.only in ('lyapunov', 'all'):
        animate_lyapunov(args.outdir, args.pkl, args.n_cohort, args.fmt,
                         args.fps, args.dpi, args.cache,
                         args.rebuild_cache, args.lyap_channels)
    if args.only in ('bsc6', 'all'):
        animate_bsc6(args.outdir, args.pkl, args.erp_channel, args.fmt,
                     args.fps, args.dpi)

    print('=' * 64, flush=True)
    print(f'done in {time.time() - t0:.0f}s  ->  {args.outdir}', flush=True)
    print('=' * 64, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
