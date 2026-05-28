"""
defense_figures/figure_MR_methodological_refusals/make_figure_MR.py

Figure MR — Methodological Refusals.

Two-column slide naming specific methodological commitments the author
refused to make, paired with what they did instead and why. Pre-empts
the "did you optimize for accuracy?" line of questioning.

Each row IS a rigor commitment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE, FIGSIZE


ROWS = [
    {
        "refused": (
            "Report trial-level cross-validation\n"
            "(typically inflates clinical-EEG accuracy by\n"
            "10–15 percentage points by leaking\n"
            "same-subject correlation across train and test)."
        ),
        "did_instead": (
            "Subject-level StratifiedGroupKFold across every classification\n"
            "claim in the dissertation. Trial-level numbers were never\n"
            "the headline."
        ),
        "experiment": "→ Chapter 5; Exp D",
    },
    {
        "refused": (
            "Use parametric p-values that assume\n"
            "noise-structure models EEG channel selection\n"
            "does not satisfy."
        ),
        "did_instead": (
            "Per-trial channel-permutation null (Exp D) that assumes\n"
            "nothing about channels and tests the spatial claim the\n"
            "dissertation actually makes."
        ),
        "experiment": "→ Exp D",
    },
    {
        "refused": (
            "Select β by maximizing validation accuracy."
        ),
        "did_instead": (
            "Select β within the measured memory-capacity plateau\n"
            "(Exp C), anchored to a biological time-constant argument\n"
            "(1/β ≈ 20 steps matches the α-period). The choice does\n"
            "not move when the downstream task changes."
        ),
        "experiment": "→ Exp C",
    },
    {
        "refused": (
            "Treat autonomous ρ(W) < 1 as sufficient\n"
            "evidence that the reservoir is in the\n"
            "usable regime."
        ),
        "did_instead": (
            "Measure driven λ₁ along 3,165 real ERP trajectories\n"
            "(Exp A). The reservoir is in the usable regime BECAUSE\n"
            "we measured it there."
        ),
        "experiment": "→ Exp A",
    },
    {
        "refused": (
            "Justify the reservoir architecturally through\n"
            "Maass–Markram alone (a non-constructive\n"
            "existence theorem)."
        ),
        "did_instead": (
            "Measure the data's Takens dimension and verify the\n"
            "reservoir's state space exceeds it by 16–64× (Exp B).\n"
            "The justification is data-specific, not generic."
        ),
        "experiment": "→ Exp B",
    },
]


def make_figure_MR(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Five methodological refusals.  Each names a default the field accepts and that this dissertation chose to refuse.",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Column headers
    ax.text(0.22, 0.93, "What I refused to do",
            ha="center", va="center",
            fontsize=12.5, fontweight="bold",
            color=PALETTE["unstable_red"])
    ax.text(0.66, 0.93, "What I did instead, and why",
            ha="center", va="center",
            fontsize=12.5, fontweight="bold",
            color=PALETTE["stable_green"])

    n = len(ROWS)
    top_y = 0.87
    bot_y = 0.10
    row_step = (top_y - bot_y) / n
    LEFT_X, LEFT_W = 0.03, 0.38
    RIGHT_X, RIGHT_W = 0.45, 0.52

    for i, r in enumerate(ROWS):
        y_top = top_y - i * row_step
        y_bot = y_top - row_step + 0.015
        # Left (refused)
        box_l = FancyBboxPatch(
            (LEFT_X, y_bot), LEFT_W, row_step - 0.03,
            boxstyle="round,pad=0.005",
            facecolor=(0.99, 0.96, 0.96),
            edgecolor=PALETTE["unstable_red"],
            lw=1.0,
            zorder=2,
        )
        ax.add_patch(box_l)
        ax.text(LEFT_X + LEFT_W / 2, (y_top + y_bot) / 2,
                r["refused"],
                ha="center", va="center",
                fontsize=10,
                color=PALETTE["histogram_charcoal"], zorder=3)

        # Right (did instead)
        box_r = FancyBboxPatch(
            (RIGHT_X, y_bot), RIGHT_W, row_step - 0.03,
            boxstyle="round,pad=0.005",
            facecolor=(0.96, 0.99, 0.97),
            edgecolor=PALETTE["stable_green"],
            lw=1.2,
            zorder=2,
        )
        ax.add_patch(box_r)
        ax.text(RIGHT_X + RIGHT_W / 2, (y_top + y_bot) / 2 + 0.012,
                r["did_instead"],
                ha="center", va="center",
                fontsize=10,
                color=PALETTE["histogram_charcoal"], zorder=3)
        ax.text(RIGHT_X + RIGHT_W - 0.012, y_bot + 0.012,
                r["experiment"],
                ha="right", va="bottom",
                fontsize=9, fontweight="bold", style="italic",
                color=PALETTE["stable_green"], zorder=3)

    ax.text(
        0.5, 0.04,
        "Each refusal is a rigor commitment.  Each commitment is in the experiments above.",
        ha="center", va="center",
        fontsize=11, fontweight="bold", style="italic",
        color=PALETTE["histogram_charcoal"],
    )

    return save_pdf(fig, outdir / "figMR_methodological_refusals.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_MR(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
