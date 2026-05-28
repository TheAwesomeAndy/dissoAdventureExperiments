"""
defense_figures/figure_G_failure_gallery/make_figure_G.py

Figure G — Failure Gallery.

Four pivots from default to measured operator. Each panel names what
was tried (the field's default or the author's first guess), how it
failed (the specific contradiction the data delivered), and what was
changed and why.

CONTENT
-------
Anchored in the dissertation's documented record (companion notes,
commits, script docstrings) and confirmed by the author against
those sources. The four panels are:

  1. Exp A — autonomous ρ(W) → driven Lyapunov λ₁.
  2. Exp C — a-priori β = 0.05 → measured MC operating regime.
  3. Exp D — single global channel permutation → per-trial channel
     permutation.
  4. Exp D — fake clinical labels → stimulus-class demonstration
     with a CLI flag that swaps to disorder labels when a clinical
     CSV is supplied.

The figure carries the deck's strongest PhD claim: that the
dissertation repeatedly REPLACED default assumptions with measured
operators. Without this slide, the deck has theory and measurement
but no visible failure narrative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE, FIGSIZE


PIVOTS = [
    {
        "tag": "Exp A · Ch. 6",
        "title": "Autonomous spectral radius → driven Lyapunov exponent",
        "tried": (
            "Cite ρ(W) < 1 (the field's default for ESN stability) as evidence\n"
            "the reservoir is in the usable regime."
        ),
        "failed": (
            "ρ(W) is an autonomous, undriven property of W. It says nothing\n"
            "about how the reservoir behaves under real ERP input. ρ(W) < 1\n"
            "does NOT imply the reservoir contracts along driven trajectories."
        ),
        "changed": (
            "Compute the driven Lyapunov exponent λ₁ via the Benettin\n"
            "algorithm along 3,165 real ERP trajectories. Stability is\n"
            "now a MEASURED property of the system in operation, not a\n"
            "cited property of W in isolation."
        ),
        "evidence": "ρ(W) = 0.2647549015 (autonomous); driven λ₁ measured on 3,165 trials.",
    },
    {
        "tag": "Exp C · Ch. 6",
        "title": "Hyperparameter chosen a priori → operating-regime measurement",
        "tried": (
            "Set leak rate β = 0.05 from a biological time-constant argument\n"
            "and use it without measuring the reservoir's memory-capacity\n"
            "response across β."
        ),
        "failed": (
            "Measured MC peak is at β* ≈ 0.012 (MC ≈ 0.835), NOT at the\n"
            "dissertation's chosen β = 0.05 (MC ≈ 0.763 ≈ 91% of peak).\n"
            "The 'a priori' choice does not maximize measured capacity."
        ),
        "changed": (
            "Build the Propagation Operating Characteristic — a measured\n"
            "MC curve over β. Re-anchor β = 0.05 inside the measured\n"
            "plateau (0.010 ≤ β ≤ 0.118 all give MC ≥ 0.75) under a JOINT\n"
            "constraint: MC ≥ 90% of peak AND the time constant 1/β is\n"
            "matched to within-trial ERP dynamics — β* = 0.012 gives\n"
            "1/β ≈ 83 steps, a longer effective integration horizon\n"
            "within the 256-step ERP; β = 0.05 gives 1/β ≈ 20 steps,\n"
            "preserving a shorter temporal horizon.  β is not chosen\n"
            "by validation accuracy."
        ),
        "evidence": "β = 0.05 → MC ≈ 0.763 (91% of measured peak). Plateau: β ∈ [0.010, 0.118].",
    },
    {
        "tag": "Exp D · Ch. 5",
        "title": "Single global channel permutation → per-trial channel permutation",
        "tried": (
            "Test the spatial-feature claim with a single global channel\n"
            "permutation π applied identically to every trial."
        ),
        "failed": (
            "A flat linear classifier compensates for a single π by learning\n"
            "new weights — cross-validated AUC stays unchanged. The 'wrong\n"
            "null' does not actually test what the dissertation claims about\n"
            "spatial structure."
        ),
        "changed": (
            "Generate an independent π_i PER TRIAL. The classifier now sees a\n"
            "(channel_pos, feature_idx) cell that contains data from a\n"
            "DIFFERENT EEG channel on every trial. Per-channel signal is\n"
            "destroyed; only per-trial overall power survives. The\n"
            "STRICTEST possible spatial null."
        ),
        "evidence": "500 perms × 5 CV folds. Observed AUC sits well outside the null distribution per class.",
    },
    {
        "tag": "Exp D · Ch. 5",
        "title": "Unavailable clinical labels → stimulus-class methodological demonstration",
        "tried": (
            "Demonstrate the channel-permutation methodology on the\n"
            "dissertation's marquee per-disorder labels (SUD / PTSD /\n"
            "GAD / ADHD) as if the labels were already in hand."
        ),
        "failed": (
            "The session-level pickle that ships with the codebase does\n"
            "NOT carry per-subject disorder assignments. Fabricating,\n"
            "approximating, or proxy-labelling them would have invented\n"
            "a clinical effect rather than measured one."
        ),
        "changed": (
            "Refuse to fake labels. Demonstrate the methodology on\n"
            "stimulus class (negative / neutral / positive affect) —\n"
            "explicitly framed as stimulus-class, NOT a clinical-disorder\n"
            "claim. CLI flag (--label-source csv --clinical-csv ...)\n"
            "swaps to disorder labels the instant the CSV is provided.\n"
            "Methodology is the deliverable; the data swap is one flag."
        ),
        "evidence": "Flag wired at make_experiment_d_figures.py:389–394; documented in Exp D §1 'The data caveat'.",
    },
]


def make_figure_G(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "What was tried, what failed, what changed.",
        fontsize=15, fontweight="bold", y=0.99,
    )
    ax.text(0.5, 0.94,
            "Four pivots from default assumption to measured operator.  "
            "Each pivot replaced an unverified habit with a measurement.",
            ha="center", va="center",
            fontsize=10.5, style="italic",
            color=PALETTE["neutral_gray"])

    # 2x2 grid
    cols = 2
    rows = 2
    x_gap = 0.025
    y_gap = 0.04
    top_y = 0.90
    bot_y = 0.07
    left_x = 0.02
    right_x = 0.98

    panel_w = (right_x - left_x - (cols - 1) * x_gap) / cols
    panel_h = (top_y - bot_y - (rows - 1) * y_gap) / rows

    for i, p in enumerate(PIVOTS):
        row, col = divmod(i, cols)
        px = left_x + col * (panel_w + x_gap)
        py = top_y - (row + 1) * panel_h - row * y_gap

        # Container box
        ax.add_patch(FancyBboxPatch(
            (px, py), panel_w, panel_h,
            boxstyle="round,pad=0.006",
            facecolor="white",
            edgecolor=PALETTE["histogram_charcoal"],
            lw=1.2, zorder=2,
        ))

        # Tag (top-left)
        ax.text(px + 0.008, py + panel_h - 0.012,
                p["tag"],
                ha="left", va="top",
                fontsize=9, fontweight="bold", style="italic",
                color=PALETTE["annotation_blue"], zorder=3)

        # Title (top-center, larger)
        ax.text(px + panel_w / 2, py + panel_h - 0.028,
                p["title"],
                ha="center", va="top",
                fontsize=10.5, fontweight="bold",
                color=PALETTE["histogram_charcoal"], zorder=3)

        # Vertical layout for tried / failed / changed
        # Three rows inside the panel
        sec_y_top = py + panel_h - 0.055
        sec_y_bot = py + 0.030
        sec_h = (sec_y_top - sec_y_bot) / 3
        labels = [
            ("Tried", p["tried"], PALETTE["unstable_red"], (0.99, 0.97, 0.96)),
            ("Failed", p["failed"], PALETTE["histogram_charcoal"], (0.99, 0.99, 0.95)),
            ("Changed", p["changed"], PALETTE["stable_green"], (0.96, 0.99, 0.97)),
        ]
        for j, (lab, text, edge, fc) in enumerate(labels):
            sy_top = sec_y_top - j * sec_h
            sy_bot = sy_top - sec_h + 0.004
            inner_x = px + 0.014
            inner_w = panel_w - 0.028
            ax.add_patch(FancyBboxPatch(
                (inner_x, sy_bot), inner_w, sec_h - 0.008,
                boxstyle="round,pad=0.003",
                facecolor=fc,
                edgecolor=edge,
                lw=0.9, zorder=3,
            ))
            # Section label (top-left of section)
            ax.text(inner_x + 0.008, sy_top - 0.005,
                    lab,
                    ha="left", va="top",
                    fontsize=8.5, fontweight="bold", style="italic",
                    color=edge, zorder=4)
            # Text body
            ax.text(inner_x + 0.013, sy_top - 0.022,
                    text,
                    ha="left", va="top",
                    fontsize=8.5,
                    color=PALETTE["histogram_charcoal"], zorder=4)

        # Evidence line at bottom of panel
        ax.text(px + panel_w / 2, py + 0.012,
                p["evidence"],
                ha="center", va="bottom",
                fontsize=7.5, style="italic",
                color=PALETTE["neutral_gray"], zorder=3)

    ax.text(
        0.5, 0.035,
        "Each pivot replaced an unverified assumption with a measurement.  "
        "The deck's defensibility lives here.",
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", style="italic",
        color=PALETTE["histogram_charcoal"],
    )

    return save_pdf(fig, outdir / "figG_failure_gallery.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_G(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
