"""
defense_figures/experiment_a_autonomous_vs_driven/make_animation.py

Reservoir Contraction Animation — Echo State Property made visceral.

Five trajectories from different initial conditions, driven by the SAME
real ERP, converge onto a single input-driven manifold. Makes Exp A's
abstract Lyapunov claim viscerally legible.

Outputs:
  outputs/rawA_1f_contraction_animation.mp4
  outputs/rawA_1f_contraction_animation.gif
  outputs/rawA_1f_contraction_animation_poster.pdf

Imports the chapter-6 Reservoir from make_experiment_a_figures.py so the
animation is aligned with Exp A's reservoir provenance (single source of
truth).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

# Import the reservoir from Exp A — single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_experiment_a_figures import BETA, M_TH, N_RES, Reservoir, normalize  # noqa: E402

# Style consistent with the rest of the deck.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import PALETTE, apply_style, save_pdf  # noqa: E402


# ESP is "established" when the current max pairwise distance is less than
# this fraction of the initial-frame max pairwise distance.
ESP_RELATIVE_THRESHOLD = 0.05


def simulate_with_state_capture(res: Reservoir, u: np.ndarray, M0: np.ndarray) -> np.ndarray:
    """Run the reservoir, capturing the membrane state at every timestep.

    Returns array of shape (T, n_res).
    """
    T = len(u)
    m = M0.copy()
    s = np.zeros(res.n_res)
    history = np.empty((T, res.n_res))
    for t in range(T):
        I = res.Win * u[t] + res.Wrec @ s
        m = (1 - BETA) * m * (1 - s) + I
        s = (m >= M_TH).astype(float)
        history[t] = m
    return history


def make_animation(
    outdir: Path,
    pickle_path: Path,
    subject_idx: int = 0,
    channel: int = 16,
    n_initial_conditions: int = 5,
    init_std: float = 2.0,
    tail_frames: int = 35,
    fps: int = 18,
    dpi: int = 110,
) -> dict:
    apply_style()

    # ─── Load drive: a single real ERP channel ─────────────────────────────
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    X_ds = np.asarray(data["X_ds"])
    print(f"  X_ds shape: {X_ds.shape}")
    subjects = np.asarray(data.get("subjects", np.zeros(X_ds.shape[0])))
    # pick the first trial of the chosen subject
    matching = np.where(subjects == subject_idx)[0]
    trial_idx = int(matching[0]) if len(matching) else 0
    u = normalize(X_ds[trial_idx, :, channel])
    T = len(u)
    print(f"  drive: subject={subject_idx}, trial={trial_idx}, channel={channel}, T={T}")

    # ─── Build reservoir, run all initial conditions ───────────────────────
    res = Reservoir(seed=42)
    init_rng = np.random.RandomState(7)
    init_states = [np.zeros(N_RES)]  # reference (zero init)
    for _ in range(n_initial_conditions - 1):
        init_states.append(init_rng.randn(N_RES) * init_std)
    histories = np.stack([simulate_with_state_capture(res, u, M0) for M0 in init_states], axis=0)
    print(f"  histories shape: {histories.shape}  ({n_initial_conditions} ICs × {T} steps × {N_RES} units)")

    # ─── Fit PCA on a concatenation of initial-state perturbations + driven
    # trajectory, so both the IC spread AND the driven manifold are captured
    # in the 2-D projection.  Pure trajectory-only PCA projects random ICs
    # near zero (they're orthogonal to the driven principal axes), which
    # hides the convergence the animation is meant to show.
    init_block = np.stack(init_states, axis=0)  # (n_IC, n_res)
    fit_data = np.concatenate([init_block, histories[0]], axis=0)
    pca = PCA(n_components=2).fit(fit_data)
    proj = np.stack([pca.transform(h) for h in histories], axis=0)  # (IC, T, 2)
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # ─── Pre-compute per-frame inter-trajectory distance in NATIVE 256-d
    # space (the right metric for ESP — PCA-2D underestimates because it
    # projects random-direction perturbations onto driven principal axes).
    pairwise_max = np.array(
        [np.max([np.linalg.norm(histories[i, t] - histories[j, t])
                 for i in range(n_initial_conditions)
                 for j in range(i + 1, n_initial_conditions)])
         for t in range(T)]
    )
    initial_pairwise = pairwise_max[0] if pairwise_max[0] > 1e-9 else 1.0
    esp_threshold = ESP_RELATIVE_THRESHOLD * initial_pairwise
    print(f"  initial pairwise spread (256-d) = {initial_pairwise:.3f}; "
          f"ESP threshold = {esp_threshold:.4f} ({ESP_RELATIVE_THRESHOLD * 100:.0f}% of initial)")

    # ─── Build figure: PCA view (top) · 256-d distance log plot (middle) · drive (bottom)
    fig, (ax_pca, ax_dist, ax_input) = plt.subplots(
        3, 1, figsize=(8.5, 8.5),
        gridspec_kw=dict(height_ratios=[4, 1.5, 1]),
    )
    fig.suptitle(
        "Reservoir Contraction — Echo State Property in motion",
        fontsize=13, fontweight="bold", y=0.985,
    )

    # PCA panel
    pad_x = 0.08 * (proj[..., 0].max() - proj[..., 0].min())
    pad_y = 0.08 * (proj[..., 1].max() - proj[..., 1].min())
    ax_pca.set_xlim(proj[..., 0].min() - pad_x, proj[..., 0].max() + pad_x)
    ax_pca.set_ylim(proj[..., 1].min() - pad_y, proj[..., 1].max() + pad_y)
    ax_pca.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0] * 100:.1f}% var)", fontsize=10)
    ax_pca.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1] * 100:.1f}% var)", fontsize=10)
    ax_pca.set_title(
        "5 trajectories from different initial conditions, driven by the same real ERP.",
        fontsize=10, pad=8,
    )

    # Color per trajectory: reference in green, perturbations in shades of blue/red
    traj_colors = [PALETTE["stable_green"]] + [
        plt.cm.viridis(0.25 + 0.6 * i / max(1, n_initial_conditions - 2))
        for i in range(n_initial_conditions - 1)
    ]

    # Persistent matplotlib artists per trajectory (head dot + tail line)
    head_artists = []
    tail_artists = []
    for i in range(n_initial_conditions):
        (tail,) = ax_pca.plot([], [], lw=1.5, color=traj_colors[i], alpha=0.6, zorder=2)
        (head,) = ax_pca.plot([], [], "o", ms=10, color=traj_colors[i],
                              mec="white", mew=1.0, zorder=4)
        head_artists.append(head)
        tail_artists.append(tail)
    # Legend label for the reference vs perturbed
    ax_pca.legend(
        [tail_artists[0], tail_artists[1]],
        ["Reference (zero init)", "Perturbed initial states"],
        loc="upper right", fontsize=9, framealpha=0.9,
    )

    # Status text (ESP established / not yet) + frame counter
    status_text = ax_pca.text(
        0.02, 0.97, "", transform=ax_pca.transAxes, fontsize=10,
        fontweight="bold", va="top", ha="left",
    )
    frame_text = ax_pca.text(
        0.98, 0.04, "", transform=ax_pca.transAxes, fontsize=9,
        fontfamily="monospace", color=PALETTE["histogram_charcoal"],
        va="bottom", ha="right",
    )

    # Distance panel: 256-d max pairwise distance, log scale
    ax_dist.semilogy(np.arange(T), pairwise_max,
                     color=PALETTE["annotation_blue"], lw=1.5,
                     label="max pairwise ‖m_i − m_j‖  (256-d)")
    ax_dist.axhline(esp_threshold, color=PALETTE["stable_green"], lw=1.0, ls="--",
                    label=f"ESP threshold ({ESP_RELATIVE_THRESHOLD * 100:.0f}% of initial)")
    ax_dist.set_xlim(0, T - 1)
    ax_dist.set_ylim(max(pairwise_max.min() * 0.5, 1e-3),
                     pairwise_max.max() * 2)
    ax_dist.set_xlabel("t  (samples)", fontsize=9)
    ax_dist.set_ylabel("max pairwise\ndistance  (log)", fontsize=9)
    ax_dist.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_dist.set_title(
        f"Native 256-d distance:  initial = {initial_pairwise:.1f} → final = {pairwise_max[-1]:.2f}  "
        f"({pairwise_max[-1] / initial_pairwise * 100:.1f}% of initial)",
        fontsize=9, pad=4, loc="left",
    )
    dist_marker = ax_dist.axvline(0, color=PALETTE["unstable_red"], lw=1.5, alpha=0.7)

    # Input panel: real ERP drive
    ax_input.plot(np.arange(T), u, color=PALETTE["histogram_charcoal"], lw=1.0)
    ax_input.set_xlim(0, T - 1)
    pad_u = 0.1 * (u.max() - u.min())
    ax_input.set_ylim(u.min() - pad_u, u.max() + pad_u)
    ax_input.set_xlabel("t  (samples, 256 Hz)", fontsize=9)
    ax_input.set_ylabel("ERP drive\n(normalized)", fontsize=9)
    drive_marker = ax_input.axvline(0, color=PALETTE["annotation_blue"], lw=1.5)
    ax_input.set_title(
        f"Real ERP drive · subject {subject_idx}, channel {channel}",
        fontsize=9, pad=4, loc="left",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    # Footer note
    fig.text(
        0.5, 0.012,
        "Driven contraction visualized in PCA-2D of the 256-d membrane state.  "
        "All trajectories converge onto a single input-driven manifold (Echo State Property).",
        ha="center", va="bottom", fontsize=8, style="italic",
        color=PALETTE["neutral_gray"],
    )

    # ─── Animation update function ─────────────────────────────────────────
    def update(frame: int):
        t_start = max(0, frame - tail_frames)
        for i in range(n_initial_conditions):
            tail_artists[i].set_data(proj[i, t_start:frame + 1, 0],
                                     proj[i, t_start:frame + 1, 1])
            head_artists[i].set_data([proj[i, frame, 0]], [proj[i, frame, 1]])

        drive_marker.set_xdata([frame, frame])
        dist_marker.set_xdata([frame, frame])

        dist = pairwise_max[frame]
        ratio = dist / initial_pairwise * 100
        if dist < esp_threshold:
            status_text.set_text(f"ESP established  ·  max pairwise d = {dist:.2f}  ({ratio:.1f}% of initial)")
            status_text.set_color(PALETTE["stable_green"])
        else:
            status_text.set_text(f"converging...   max pairwise d = {dist:.2f}  ({ratio:.1f}% of initial)")
            status_text.set_color(PALETTE["unstable_red"])

        ms = frame * 1000 / 256
        frame_text.set_text(f"frame {frame:3d}/{T - 1} · {ms:5.1f} ms")

        return [*head_artists, *tail_artists, drive_marker, dist_marker,
                status_text, frame_text]

    base = outdir / "rawA_1f_contraction_animation"
    written = {}

    # MP4
    try:
        anim = FuncAnimation(fig, update, frames=T, blit=False)
        writer = FFMpegWriter(
            fps=fps, codec="libx264", bitrate=2400,
            extra_args=["-pix_fmt", "yuv420p",
                        "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"],
        )
        anim.save(str(base) + ".mp4", writer=writer, dpi=dpi)
        written["mp4"] = str(base) + ".mp4"
        print(f"  wrote {written['mp4']}")
    except Exception as e:
        print(f"  [warn] MP4 render failed: {e}")

    # GIF (slightly lower DPI to keep file size reasonable)
    try:
        anim = FuncAnimation(fig, update, frames=T, blit=False)
        anim.save(str(base) + ".gif", writer=PillowWriter(fps=fps), dpi=int(dpi * 0.7))
        written["gif"] = str(base) + ".gif"
        print(f"  wrote {written['gif']}")
    except Exception as e:
        print(f"  [warn] GIF render failed: {e}")

    # Poster PDF: hold at the converged frame so the static slide makes the point.
    poster_frame = int(min(T - 1, T - 30))
    update(poster_frame)
    poster_path = save_pdf(fig, base.with_suffix(".pdf"))
    written["poster"] = str(poster_path)
    print(f"  wrote {written['poster']}")

    plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pickle", default="chapter6Experiments/results/ch6_exp1_full.pkl",
                        help="Path to ch6 features pickle (relative to repo root).")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--channel", type=int, default=16)
    parser.add_argument("--n_initial_conditions", type=int, default=5)
    parser.add_argument("--init_std", type=float, default=2.0,
                        help="Std-dev of initial-state perturbations (in 256-d "
                             "membrane space).  Needs to be large enough that the "
                             "perturbations are visible in PCA-2D of the driven "
                             "trajectory; the driven contraction then pulls them in.")
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    written = make_animation(
        outdir=outdir,
        pickle_path=(repo / args.pickle).resolve(),
        subject_idx=args.subject,
        channel=args.channel,
        n_initial_conditions=args.n_initial_conditions,
        init_std=args.init_std,
        fps=args.fps,
        dpi=args.dpi,
    )
    print(f"\n  artifacts: {list(written.keys())}")


if __name__ == "__main__":
    main()
