"""
defense_figures/experiment_a_autonomous_vs_driven/make_experiment_a_figures.py

Experiment A — Autonomous Spectral Radius vs. Driven Lyapunov Exponent.

Produces five PDFs and one CSV that together form the central defense
figure for the Neuromorphic Operator (Dive 1). See the plan file
`/root/.claude/plans/experiment-a-autonomous-foamy-bear.md` for the
philosophical context and slide-by-slide intent.

OUTPUTS
-------
  outputs/rawA_1a_weight_matrix.pdf
      W_rec heatmap + nonzero-entry distribution. Raw observation of the
      recurrent matrix itself, before any analysis.

  outputs/rawA_1b_eigenspectrum.pdf
      All 256 complex eigenvalues of W_rec in the unit-circle frame.
      The autonomous criterion, made visible: ρ(W) is the radius of the
      outermost eigenvalue.

  outputs/rawA_1c_benettin_sample_trajectories.pdf
      Twelve sampled per-trial ln(separation) trajectories under the
      Benettin two-trajectory algorithm. Shows where each λ₁ value
      comes from at the level of an individual trial.

  outputs/rawA_1d_per_trial_lambda_scatter.pdf
      All measurements unpooled, plotted by subject index and colored by
      condition class. Demonstrates the contraction holds uniformly,
      not as an artifact of pooling.

  outputs/analysisA_1e_autonomous_vs_driven.pdf
      THE SLIDE. Two-panel comparison: autonomous criterion (single
      ρ(W) line in red/green stability zones) vs. driven criterion
      (population histogram of λ₁ values).

  outputs/experiment_a_data.csv
      Audit trail: every λ₁ value, plus a header recording ρ(W) and
      all reservoir parameters. The "show me the number" reference.

USAGE
-----
  python make_experiment_a_figures.py
  python make_experiment_a_figures.py --pickle path/to/features.pkl --outdir outputs

THE TWO RESERVOIRS — CRITICAL
-----------------------------
The dissertation's source contains two distinct reservoir classes. This
script uses the **chapter-6 Reservoir** (sparse Gaussian, no spectral-radius
rescaling) — the same reservoir that computed the existing λ₁ population
results in the dissertation text. Mixing this with the chapter-3/4/5
LIFReservoir (which IS rescaled to ρ=0.9) would be a category error.

The chapter-6 Reservoir class is defined inline (verbatim copy from
`chapter6Experiments/run_chapter6_exp1_esp.py:39-50`) to make the
reservoir provenance fully legible in this single file.

PRE-REGISTRATION
----------------
Expected: ρ(W) ≈ 0.20–0.30 from random-matrix theory σ√(Np) ≈ 0.25,
landing well inside the unit circle (green zone). The driven population
λ₁ distribution is expected to be entirely negative with median near
−0.054. If the measured values land differently, the figure presents
them faithfully — see §1.8 of the plan file.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle

# Repo-local style module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import (
    apply_style, save_pdf, figtext_footer,
    PALETTE, FIGSIZE, CATEGORY_COLORS,
)

# ──────────────────────────────────────────────────────────────────────────
# Chapter-6 Reservoir — verbatim copy from
# chapter6Experiments/run_chapter6_exp1_esp.py:39-50, preserving conventions:
#   • W_in: 1D Gaussian × 0.3 (single scalar input per neuron)
#   • W_rec: dense Gaussian × 0.05, 10% Bernoulli mask, zero diagonal
#   • NO spectral-radius rescaling
#   • Leak β = 0.05, threshold θ = 0.5
# ──────────────────────────────────────────────────────────────────────────
N_RES = 256
BETA = 0.05
M_TH = 0.5
DELTA_0 = 1e-8
T_RENORM = 50


class Reservoir:
    """Chapter-6 LIF reservoir (sparse Gaussian, no SR rescaling)."""

    def __init__(self, seed: int = 42, n_res: int = N_RES) -> None:
        rng = np.random.RandomState(seed)
        self.Win = rng.randn(n_res) * 0.3
        mask = (rng.rand(n_res, n_res) < 0.1).astype(float)
        np.fill_diagonal(mask, 0)
        self.Wrec = rng.randn(n_res, n_res) * 0.05 * mask
        self.fanout = mask.sum(1)
        self.n_res = n_res

    def run(self, u, M0=None):
        T = len(u)
        m = M0.copy() if M0 is not None else np.zeros(self.n_res)
        s = np.zeros(self.n_res)
        for t in range(T):
            I = self.Win * u[t] + self.Wrec @ s
            m = (1 - BETA) * m * (1 - s) + I
            s = (m >= M_TH).astype(float)
        return m, s


def normalize(u):
    return (u - u.mean()) / (u.std() + 1e-10)


# ──────────────────────────────────────────────────────────────────────────
# Benettin two-trajectory Lyapunov estimator (driven)
# Adapted verbatim from run_chapter6_exp1_esp.py:174-183, but generalized
# to optionally capture the (time, ln_sep) trajectory for sample plotting.
# ──────────────────────────────────────────────────────────────────────────
def benettin_lambda1(res: Reservoir, u: np.ndarray,
                     capture_trajectory: bool = False,
                     perturbation_seed: int = 42):
    """
    Compute driven λ₁ for input `u` (length T) on reservoir `res`.

    Returns (lambda1, td, ld) where:
      lambda1 — float, mean log-growth rate over the renormalization cycles
      td      — list of timestep indices at which renormalization occurred
      ld      — list of ln(‖δ‖/δ₀) values at each renormalization
    """
    rng = np.random.RandomState(perturbation_seed)
    e = rng.randn(res.n_res)
    e /= np.linalg.norm(e)

    mr = np.zeros(res.n_res)
    sr = np.zeros(res.n_res)
    mp = mr + DELTA_0 * e
    sp = np.zeros(res.n_res)

    td = []
    ld = []

    T = len(u)
    for t in range(T):
        I = res.Win * u[t]
        mr_n = (1 - BETA) * mr * (1 - sr) + I + res.Wrec @ sr
        sr_n = (mr_n >= M_TH).astype(float)
        mp_n = (1 - BETA) * mp * (1 - sp) + I + res.Wrec @ sp
        sp_n = (mp_n >= M_TH).astype(float)
        mr, sr, mp, sp = mr_n, sr_n, mp_n, sp_n
        if (t + 1) % T_RENORM == 0:
            d = mp - mr
            dist = np.linalg.norm(d)
            if dist > 0:
                ld.append(float(np.log(dist / DELTA_0)))
                td.append(t)
                mp = mr + DELTA_0 * (d / dist)
                sp = sr.copy()

    if not ld:
        return float("nan"), td, ld
    lam = float(np.mean(ld) / T_RENORM)
    if capture_trajectory:
        return lam, td, ld
    return lam, None, None


# ──────────────────────────────────────────────────────────────────────────
# Population sweep: run Benettin for every (trial, channel) pair
# ──────────────────────────────────────────────────────────────────────────
def compute_population_lambdas(
    res: Reservoir,
    X_ds: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    channels: list[int],
    sample_indices_for_trajectory: list[tuple[int, int]] | None = None,
    progress_every: int = 200,
):
    """
    Run Benettin over (trial × channel). Returns:
      records: list of dicts {trial, subject, class, channel, lambda1}
      trajectories: dict {(trial_idx, channel): (td, ld)} for sampled cases
    """
    n_trials = X_ds.shape[0]
    sample_set = set(sample_indices_for_trajectory or [])
    records = []
    trajectories: dict[tuple[int, int], tuple[list, list]] = {}

    print(f"  Running Benettin on {n_trials} trials × {len(channels)} channels "
          f"= {n_trials * len(channels)} measurements...")
    t0 = time.time()
    done = 0
    for i in range(n_trials):
        sid = int(subjects[i])
        cls = int(y[i])
        for ch in channels:
            u = normalize(X_ds[i, :, ch])
            capture = (i, ch) in sample_set
            lam, td, ld = benettin_lambda1(res, u, capture_trajectory=capture)
            records.append({
                "trial": i,
                "subject": sid,
                "class": cls,
                "channel": ch,
                "lambda1": lam,
            })
            if capture and td is not None:
                trajectories[(i, ch)] = (td, ld)
            done += 1
            if done % progress_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (n_trials * len(channels) - done) / rate if rate > 0 else 0.0
                print(f"    {done}/{n_trials * len(channels)} ({100*done/(n_trials*len(channels)):.0f}%)  "
                      f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s")
    print(f"  Benettin complete in {time.time()-t0:.0f}s.")
    return records, trajectories


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1A — Recurrent weight matrix
# ──────────────────────────────────────────────────────────────────────────
def make_rawA_1a(res: Reservoir, outdir: Path) -> Path:
    W = res.Wrec
    fig, (ax_heat, ax_hist) = plt.subplots(1, 2, figsize=FIGSIZE["raw_two_panel"])

    vmax = float(np.abs(W).max())
    im = ax_heat.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax_heat.set_title("W_rec (256 × 256), sparse Gaussian init")
    ax_heat.set_xlabel("pre-synaptic index")
    ax_heat.set_ylabel("post-synaptic index")
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="weight")

    nz = W[W != 0]
    density = (W != 0).mean()
    ax_hist.hist(nz, bins=50, color=PALETTE["histogram_charcoal"],
                 edgecolor="white", alpha=0.85)
    ax_hist.axvline(0, color=PALETTE["unstable_red"], lw=1.0, alpha=0.6)
    ax_hist.set_xlabel("nonzero weight value")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Distribution of nonzero entries")
    ax_hist.text(
        0.98, 0.95,
        f"density: {density:.3f}\n"
        f"#nonzero: {(W != 0).sum():,}\n"
        f"σ_nonzero: {nz.std():.4f}\n"
        f"max|W_ij|: {vmax:.4f}",
        transform=ax_hist.transAxes,
        ha="right", va="top",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                  edgecolor=PALETTE["neutral_gray"]),
    )

    fig.suptitle("Raw Observation A.1a — Recurrent weight matrix W_rec",
                 fontsize=12, fontweight="bold")
    return save_pdf(fig, outdir / "rawA_1a_weight_matrix.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1B — Eigenspectrum
# ──────────────────────────────────────────────────────────────────────────
def make_rawA_1b(res: Reservoir, rho: float, outdir: Path) -> Path:
    eigs = np.linalg.eigvals(res.Wrec)
    max_idx = int(np.argmax(np.abs(eigs)))

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Stability zones (shaded annuli)
    outer_R = 1.5
    ax.add_patch(Rectangle((-outer_R, -outer_R), 2 * outer_R, 2 * outer_R,
                           facecolor=PALETTE["unstable_red"], alpha=0.07,
                           edgecolor="none", zorder=0))
    ax.add_patch(Circle((0, 0), 1.0,
                        facecolor=PALETTE["stable_green"], alpha=0.13,
                        edgecolor="none", zorder=1))

    # Unit circle
    ax.add_patch(Circle((0, 0), 1.0, fill=False,
                        edgecolor=PALETTE["unstable_red"], lw=2.0, zorder=3,
                        linestyle="--"))

    # Eigenvalues
    ax.scatter(eigs.real, eigs.imag, s=18,
               color=PALETTE["histogram_charcoal"],
               alpha=0.75, edgecolors="white", linewidths=0.4, zorder=5)

    # Highlight max-modulus eigenvalue
    ax.scatter([eigs[max_idx].real], [eigs[max_idx].imag],
               s=120, marker="o", facecolor="none",
               edgecolor=PALETTE["unstable_red"], lw=2.0, zorder=6)
    # Place annotation outside the unit circle so the arrow doesn't cross it
    ax.annotate(
        f"ρ(W) = {rho:.4f}",
        xy=(eigs[max_idx].real, eigs[max_idx].imag),
        xytext=(1.10, 0.55),
        fontsize=12, fontweight="bold",
        color=PALETTE["unstable_red"],
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["unstable_red"],
                        lw=1.2, connectionstyle="arc3,rad=0.15"),
    )

    # Axes and reference lines
    ax.axhline(0, color=PALETTE["neutral_gray"], lw=0.5, zorder=2)
    ax.axvline(0, color=PALETTE["neutral_gray"], lw=0.5, zorder=2)
    ax.set_xlim(-outer_R, outer_R)
    ax.set_ylim(-outer_R, outer_R)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_aspect("equal")
    ax.set_title(
        "Raw Observation A.1b — Eigenspectrum of W_rec in the complex plane",
        fontsize=12, fontweight="bold",
    )

    ax.text(0.02, 0.98,
            "Field-standard\n'stable' (ρ<1)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=PALETTE["stable_green"], fontweight="bold")
    ax.text(0.98, 0.98,
            "Field-standard\n'unstable' (ρ>1)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=PALETTE["unstable_red"], fontweight="bold")

    return save_pdf(fig, outdir / "rawA_1b_eigenspectrum.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1C — Sample Benettin trajectories
# ──────────────────────────────────────────────────────────────────────────
def make_rawA_1c(trajectories: dict, lookup_by_idx, outdir: Path) -> Path | None:
    items = sorted(trajectories.items())
    if not items:
        print("  rawA_1c: no trajectories captured; skipping.")
        return None

    n = min(12, len(items))
    items = items[:n]
    nrows = 3
    ncols = (n + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE["raw_grid_3x4"],
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    class_colors = [PALETTE["category_threat"], PALETTE["neutral_gray"],
                    PALETTE["category_cute"]]

    for i, ((trial_idx, ch), (td, ld)) in enumerate(items):
        ax = axes[i // ncols, i % ncols]
        meta = lookup_by_idx(trial_idx)
        cls = meta["class"]
        sid = meta["subject"]
        ld_arr = np.asarray(ld, dtype=float)
        td_arr = np.asarray(td, dtype=float)
        ax.plot(td_arr, ld_arr, marker="o", ms=4, lw=1.1,
                color=class_colors[cls % len(class_colors)],
                label="ln(‖δ‖/δ₀) per renorm cycle")
        # Reference line at zero (would indicate no stretching/contraction)
        ax.axhline(0, color=PALETTE["unstable_red"], lw=1.0, linestyle="--",
                   alpha=0.6, label="no contraction" if i == 0 else None)
        # The Benettin estimator: λ̂₁ = mean(ld) / T_renorm
        # Show the mean as a horizontal line — this is what gets divided by T_renorm
        if len(ld_arr) >= 1:
            mean_ld = float(np.mean(ld_arr))
            ax.axhline(mean_ld, color=PALETTE["annotation_blue"], lw=1.4,
                       linestyle="-", alpha=0.85,
                       label="mean ln(‖δ‖/δ₀)" if i == 0 else None)
            lam = mean_ld / T_RENORM
            ax.set_title(f"S{sid} ch{ch} cls{cls}\nλ̂₁ = mean/{T_RENORM} = {lam:.4f}",
                         fontsize=9, fontweight="bold")
        if i // ncols == nrows - 1:
            ax.set_xlabel("time step")
        if i % ncols == 0:
            ax.set_ylabel("ln(‖δ‖/δ₀)")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.92)

    # Hide any unused axes
    for j in range(len(items), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(
        "Raw Observation A.1c — Sample Benettin trajectories (where each λ₁ comes from)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    plt.tight_layout()
    return save_pdf(fig, outdir / "rawA_1c_benettin_sample_trajectories.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1D — Per-trial scatter
# ──────────────────────────────────────────────────────────────────────────
def make_rawA_1d(records: list[dict], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE["raw_strip"])

    subjects = np.array([r["subject"] for r in records])
    lams = np.array([r["lambda1"] for r in records])
    cls = np.array([r["class"] for r in records])
    chans = np.array([r["channel"] for r in records])

    # Map subjects to a compact x index
    unique_sids = np.unique(subjects)
    sid_to_x = {sid: i for i, sid in enumerate(unique_sids)}
    x_idx = np.array([sid_to_x[s] for s in subjects], dtype=float)
    # Add small jitter per (subject, channel) so they don't overplot
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.35, 0.35, size=x_idx.shape)
    x_plot = x_idx + jitter

    class_colors = [PALETTE["category_threat"], PALETTE["neutral_gray"],
                    PALETTE["category_cute"]]
    class_labels = ["class 0", "class 1 (neutral)", "class 2"]
    for c in sorted(np.unique(cls)):
        mask = (cls == c)
        ax.scatter(x_plot[mask], lams[mask],
                   s=8, alpha=0.45, color=class_colors[c % len(class_colors)],
                   edgecolors="none",
                   label=class_labels[c % len(class_labels)],
                   rasterized=True)

    ax.axhline(0, color=PALETTE["unstable_red"], lw=1.5, linestyle="-", alpha=0.9)
    median = float(np.median(lams))
    mean = float(np.mean(lams))
    ax.axhline(median, color=PALETTE["annotation_blue"],
               lw=1.2, linestyle="--",
               label=f"median = {median:.4f}")
    ax.axhline(mean, color=PALETTE["stable_green"],
               lw=1.2, linestyle=":",
               label=f"mean = {mean:.4f}")

    ax.set_xlabel(f"subject index (n={len(unique_sids)})")
    ax.set_ylabel(r"driven Lyapunov exponent $\lambda_1$")
    ax.set_title(
        f"Raw Observation A.1d — All {len(records):,} per-trial measurements, "
        f"colored by stimulus class",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.15)

    return save_pdf(fig, outdir / "rawA_1d_per_trial_lambda_scatter.pdf")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1E — THE SLIDE
# ──────────────────────────────────────────────────────────────────────────
def make_analysisA_1e(rho: float, records: list[dict], outdir: Path,
                      meta_note: str | None = None) -> Path:
    fig = plt.figure(figsize=FIGSIZE["slide_landscape_tall"])
    gs = GridSpec(20, 2, figure=fig, hspace=0.45, wspace=0.30)
    ax_l = fig.add_subplot(gs[:17, 0])
    ax_r = fig.add_subplot(gs[:17, 1])

    # LEFT — autonomous criterion
    ax_l.set_xlim(0, 1.5)
    ax_l.set_ylim(0, 1)
    # Stability zones
    ax_l.axvspan(0.0, 1.0, color=PALETTE["stable_green"], alpha=0.15)
    ax_l.axvspan(1.0, 1.5, color=PALETTE["unstable_red"], alpha=0.15)
    ax_l.axvline(1.0, color=PALETTE["unstable_red"], lw=1.5, linestyle="--", alpha=0.8)
    # Measured ρ
    ax_l.axvline(rho, color=PALETTE["histogram_charcoal"], lw=2.5)
    ax_l.annotate(
        f"ρ(W) = {rho:.4f}",
        xy=(rho, 0.55), xytext=(rho + 0.10, 0.78),
        fontsize=12, fontweight="bold",
        color=PALETTE["histogram_charcoal"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["histogram_charcoal"], lw=1.2),
    )
    # Zone labels — placed at bottom of each zone, well separated, no overlap
    # Green zone spans [0, 1.0] → label centered at axes-x ≈ 0.333
    ax_l.text(0.333, 0.08, "Field-standard 'stable'  (ρ < 1)",
              transform=ax_l.transAxes, ha="center", va="bottom",
              fontsize=10, fontweight="bold",
              color=PALETTE["stable_green"])
    # Red zone spans [1.0, 1.5] → label centered at axes-x ≈ 0.833
    ax_l.text(0.833, 0.08, "'unstable'  (ρ > 1)",
              transform=ax_l.transAxes, ha="center", va="bottom",
              fontsize=10, fontweight="bold",
              color=PALETTE["unstable_red"])
    ax_l.set_xlabel(r"spectral radius $\rho(\mathbf{W})$")
    ax_l.set_yticks([])
    ax_l.set_title("Autonomous criterion (field standard)", fontsize=11, fontweight="bold")
    ax_l.text(0.5, -0.18,
              "A single static number, input-agnostic.",
              transform=ax_l.transAxes, ha="center", va="top",
              fontsize=9.5, fontstyle="italic",
              color=PALETTE["neutral_gray"])
    # Subscript provenance note — under the ρ marker, slightly raised above x-axis
    ax_l.text(rho, 0.30,
              "emergent from\nsparse Gaussian init\n(σ=0.05, p=0.10)",
              ha="center", va="center",
              fontsize=8, color=PALETTE["neutral_gray"], fontstyle="italic")

    # RIGHT — driven criterion
    lams = np.array([r["lambda1"] for r in records])
    n = len(lams)
    median = float(np.median(lams))
    mean = float(np.mean(lams))
    frac_neg = float((lams < 0).mean())
    q25, q75 = float(np.quantile(lams, 0.25)), float(np.quantile(lams, 0.75))

    # Ensure the histogram is not clipped — set x-range from below the data
    # minimum to slightly above 0 so the red 0-reference line is fully visible.
    xmin = float(lams.min()) - 0.005
    xmax = 0.005
    ax_r.set_xlim(xmin, xmax)
    ax_r.hist(lams, bins=50, color=PALETTE["histogram_charcoal"],
              edgecolor="white", alpha=0.88)
    ax_r.axvline(0, color=PALETTE["unstable_red"], lw=2.0,
                 label="λ₁ = 0  (contraction boundary)")
    ax_r.axvline(median, color=PALETTE["annotation_blue"], lw=1.6,
                 linestyle="--", label=f"median = {median:.4f}")
    ax_r.axvline(mean, color=PALETTE["stable_green"], lw=1.4,
                 linestyle=":", label=f"mean = {mean:.4f}")
    ax_r.legend(loc="center right", framealpha=0.92, fontsize=9)
    ax_r.set_xlabel(r"driven Lyapunov exponent $\lambda_1^{(\mathrm{driven})}$")
    ax_r.set_ylabel("count")
    ax_r.set_title("Driven criterion (this dissertation)", fontsize=11, fontweight="bold")
    ax_r.text(0.5, -0.18,
              "A quantitative contraction rate, measured along the actual trajectory.",
              transform=ax_r.transAxes, ha="center", va="top",
              fontsize=9.5, fontstyle="italic",
              color=PALETTE["neutral_gray"])
    ax_r.text(
        0.98, 0.95,
        f"n = {n:,}\n"
        f"{100*frac_neg:.1f}% < 0\n"
        f"median = {median:.4f}\n"
        f"mean = {mean:.4f}\n"
        f"IQR = [{q25:.4f}, {q75:.4f}]",
        transform=ax_r.transAxes, ha="right", va="top",
        family="monospace", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92,
                  edgecolor=PALETTE["neutral_gray"]),
    )

    # Bottom strip
    figtext_footer(
        fig,
        "Autonomous theory describes the reservoir at rest.  "
        "Driven theory describes the reservoir doing its job.",
        y=0.03,
    )

    title = (
        f"The reservoir contracts under the input it actually receives  "
        f"(λ₁ = {median:.4f},  n = {n:,},  {100*frac_neg:.1f}% negative)"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    if meta_note:
        fig.text(0.99, 0.01, meta_note, ha="right", va="bottom",
                 fontsize=7, color=PALETTE["neutral_gray"], style="italic")
    return save_pdf(fig, outdir / "analysisA_1e_autonomous_vs_driven.pdf")


# ──────────────────────────────────────────────────────────────────────────
# CSV audit trail
# ──────────────────────────────────────────────────────────────────────────
def write_csv(rho: float, records: list[dict], args, outdir: Path) -> Path:
    p = outdir / "experiment_a_data.csv"
    with open(p, "w") as f:
        f.write("# Experiment A — audit trail\n")
        f.write(f"# script: defense_figures/experiment_a_autonomous_vs_driven/make_experiment_a_figures.py\n")
        f.write(f"# pickle: {args.pickle}\n")
        f.write(f"# reservoir_class: chapter6Experiments/run_chapter6_exp1_esp.py:Reservoir (sparse Gaussian, no SR rescaling)\n")
        f.write(f"# reservoir_seed: {args.seed}\n")
        f.write(f"# n_res: {N_RES}\n")
        f.write(f"# beta: {BETA}\n")
        f.write(f"# threshold: {M_TH}\n")
        f.write(f"# density: 0.10\n")
        f.write(f"# sigma_w_rec: 0.05\n")
        f.write(f"# sigma_w_in: 0.30\n")
        f.write(f"# rho_W_rec: {rho:.10f}\n")
        f.write(f"# T_renorm: {T_RENORM}\n")
        f.write(f"# delta_0: {DELTA_0}\n")
        f.write(f"# n_measurements: {len(records)}\n")
        f.write(f"# columns: trial,subject,class,channel,lambda1\n")
        f.write("trial,subject,class,channel,lambda1\n")
        for r in records:
            f.write(f"{r['trial']},{r['subject']},{r['class']},{r['channel']},{r['lambda1']:.10f}\n")
    print(f"  wrote {p} ({len(records):,} rows)")
    return p


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pickle", default="chapter6Experiments/results/ch6_exp1_full.pkl",
                   help="Pickle with X_ds, y, subjects (and optionally lyapunov_results).")
    p.add_argument("--outdir", default=str(Path(__file__).parent / "outputs"),
                   help="Directory for PDFs and CSV.")
    p.add_argument("--seed", type=int, default=42, help="Reservoir RNG seed.")
    p.add_argument("--channels", type=int, nargs="+", default=[0, 8, 16, 24, 33],
                   help="Analysis channel indices (defaults match ch6 convention).")
    p.add_argument("--n-sample-trajectories", type=int, default=12,
                   help="Number of (trial, channel) Benettin trajectories to plot in 1c.")
    p.add_argument("--cache", default=None,
                   help="Optional .npz cache for the Benettin sweep (skip recompute if exists).")
    p.add_argument("--recompute", action="store_true",
                   help="Force Benettin recomputation even if cache exists.")
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Cache path
    cache_path = Path(args.cache).resolve() if args.cache else outdir / "_benettin_cache.npz"

    # ── Build reservoir, compute ρ(W) ───────────────────────────────────
    res = Reservoir(seed=args.seed)
    eigs = np.linalg.eigvals(res.Wrec)
    rho = float(np.max(np.abs(eigs)))
    print(f"  Reservoir (seed={args.seed}): ρ(W) = {rho:.6f}")

    # ── Load data ───────────────────────────────────────────────────────
    pickle_path = Path(args.pickle).resolve()
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")
    print(f"  Loading {pickle_path}...")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if "X_ds" not in data or "subjects" not in data or "y" not in data:
        raise KeyError(f"Pickle missing required keys (X_ds, y, subjects). Got: {list(data.keys())}")
    X_ds = np.asarray(data["X_ds"])
    y = np.asarray(data["y"])
    subjects = np.asarray(data["subjects"])
    print(f"  X_ds shape: {X_ds.shape}, y shape: {y.shape}, n_subjects: {len(np.unique(subjects))}")

    # ── Pick sample (trial, channel) pairs for trajectory capture ───────
    rng = np.random.default_rng(args.seed)
    n_trajectories = args.n_sample_trajectories
    classes_present = np.unique(y)
    sample_indices = []
    per_class = max(1, n_trajectories // len(classes_present))
    for c in classes_present:
        trial_pool = np.where(y == c)[0]
        chosen = rng.choice(trial_pool, size=min(per_class, len(trial_pool)), replace=False)
        for t in chosen:
            sample_indices.append((int(t), int(args.channels[len(sample_indices) % len(args.channels)])))
    sample_indices = sample_indices[:n_trajectories]
    print(f"  Sampled {len(sample_indices)} (trial, channel) pairs for trajectory capture.")

    # ── Run Benettin (or load cache) ────────────────────────────────────
    if cache_path.exists() and not args.recompute:
        print(f"  Loading Benettin cache: {cache_path}")
        cz = np.load(cache_path, allow_pickle=True)
        records = list(cz["records"])
        # Trajectories don't survive npz cleanly; just re-run capture quickly
        # for the small sample set.
        print(f"  Re-capturing {len(sample_indices)} trajectories from cached reservoir state...")
        trajectories = {}
        for (trial_i, ch) in sample_indices:
            u = normalize(X_ds[trial_i, :, ch])
            lam, td, ld = benettin_lambda1(res, u, capture_trajectory=True)
            trajectories[(trial_i, ch)] = (td, ld)
    else:
        records, trajectories = compute_population_lambdas(
            res, X_ds, y, subjects, args.channels,
            sample_indices_for_trajectory=sample_indices,
        )
        # Save cache
        np.savez(cache_path,
                 records=np.array(records, dtype=object),
                 rho=rho, seed=args.seed)
        print(f"  Cached records → {cache_path}")

    # ── Sanity stats ────────────────────────────────────────────────────
    lams = np.array([r["lambda1"] for r in records])
    valid = ~np.isnan(lams)
    print(f"  λ₁ summary (n={valid.sum():,}/{len(lams):,} valid):")
    print(f"    mean   = {lams[valid].mean():.6f}")
    print(f"    median = {np.median(lams[valid]):.6f}")
    print(f"    min    = {lams[valid].min():.6f}")
    print(f"    max    = {lams[valid].max():.6f}")
    print(f"    fraction <0: {(lams[valid] < 0).mean():.4f}")

    # ── Produce figures ─────────────────────────────────────────────────
    print("  Generating figures...")
    p1a = make_rawA_1a(res, outdir)
    print(f"    {p1a.name}")
    p1b = make_rawA_1b(res, rho, outdir)
    print(f"    {p1b.name}")

    # Lookup helper for 1c
    lookup = {r["trial"]: r for r in records}
    def lookup_by_idx(trial_idx):
        return lookup.get(trial_idx, {"subject": "?", "class": 0})
    p1c = make_rawA_1c(trajectories, lookup_by_idx, outdir)
    if p1c is not None:
        print(f"    {p1c.name}")

    p1d = make_rawA_1d([r for r in records if not np.isnan(r["lambda1"])], outdir)
    print(f"    {p1d.name}")

    n_cls = len(np.unique(y))
    meta_note = (
        f"Source: {pickle_path.name}  ·  "
        f"n_trials={X_ds.shape[0]}  ·  "
        f"n_classes={n_cls}  ·  "
        f"channels={args.channels}  ·  "
        f"n_measurements={valid.sum():,}"
    )
    p1e = make_analysisA_1e(rho, [r for r in records if not np.isnan(r["lambda1"])],
                            outdir, meta_note=meta_note)
    print(f"    {p1e.name}")

    write_csv(rho, records, args, outdir)

    print("\n  Done. Outputs in:", outdir)


if __name__ == "__main__":
    main()
