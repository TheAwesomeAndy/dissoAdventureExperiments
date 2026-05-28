"""
defense_figures/figure_K_opening_questions/make_figure_K.py

Figure K — Opening Questions Slide.

Slide 1 of the defense deck (after title). Names the 3–5 deep,
research-motivated questions the dissertation answers, **each paired with
what the standard answer could not give the author for this problem**.
Sets the philosophical frame before any data is shown.

VOICE
-----
The right column is NOT "the dissatisfaction with the field" (which risks
reading as critique). It is "what the standard answer was structurally
underspecified to recover for THIS problem". That framing puts the author
in the position of a scholar inquiring, not an outsider scolding.

CONTENT
-------
The questions and structural-insufficiency statements below are AUTHOR
PLACEHOLDERS — the author should refine each in their own voice. The
script provides the typesetting; the content is the author's intellectual
signature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE, FIGSIZE


# ──────────────────────────────────────────────────────────────────────────
# Rows — author replaces these with their own voice
# ──────────────────────────────────────────────────────────────────────────
ROWS = [
    {
        "question": (
            "Is the field's autonomous stability criterion the right object\n"
            "to measure for a driven reservoir?"
        ),
        "structural": (
            "ρ(W) < 1 verifies a static property of the recurrent matrix; it does not\n"
            "characterize how the reservoir contracts under the input it actually receives.\n"
            "For a reservoir whose job IS the driven response, the autonomous criterion\n"
            "is structurally underspecified."
        ),
        "experiment_pointer": "→ Experiment A",
    },
    {
        "question": (
            "Can the reservoir's choice be justified from a theorem about\n"
            "the data, not from architectural fashion?"
        ),
        "structural": (
            "The Maass–Markram licensing argument is non-constructive for a specific\n"
            "dataset — it tells you reservoirs can separate, not whether mine does for\n"
            "these ERPs. Takens' theorem makes a measurable claim about the data the\n"
            "reservoir must accommodate; I wanted that measurement."
        ),
        "experiment_pointer": "→ Experiment B",
    },
    {
        "question": (
            "Can the operating hyperparameter be derived rather than tuned?"
        ),
        "structural": (
            "Validation-accuracy-driven selection optimizes for the metric, not for\n"
            "the underlying information-theoretic property. A measured memory-capacity\n"
            "peak provides a stable, theoretically-anchored operating point that does\n"
            "not move when the downstream task changes."
        ),
        "experiment_pointer": "→ Experiment C",
    },
    {
        "question": (
            "Do clinical effects survive the strictest possible null —\n"
            "channel permutation — at the per-disorder level?"
        ),
        "structural": (
            "Parametric p-values assume a model of the noise structure that EEG\n"
            "channel selection does not satisfy. A channel-permutation null assumes\n"
            "nothing about the channels; it tests the claim the author actually wants\n"
            "to make."
        ),
        "experiment_pointer": "→ Experiment D",
    },
]


def make_figure_K(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top title
    fig.suptitle(
        "Four questions this dissertation refused to outsource",
        fontsize=15, fontweight="bold", y=0.98,
    )
    # Subtitle giving the framing for the right column
    ax.text(
        0.5, 0.94,
        "left — the question;   "
        "right — what the field's standard answer was structurally underspecified to recover for this problem",
        ha="center", va="center", fontsize=10, style="italic",
        color=PALETTE["neutral_gray"],
    )

    n = len(ROWS)
    top_y = 0.86
    bottom_y = 0.08
    row_step = (top_y - bottom_y) / n
    row_top = top_y

    LEFT_X = 0.03
    LEFT_W = 0.36
    RIGHT_X = 0.45
    RIGHT_W = 0.52

    for i, r in enumerate(ROWS):
        y_lo = row_top - row_step + 0.02  # small inter-row gap
        # Left box (question)
        box_l = FancyBboxPatch(
            (LEFT_X, y_lo), LEFT_W, row_step - 0.04,
            boxstyle="round,pad=0.005",
            facecolor="white",
            edgecolor=PALETTE["histogram_charcoal"],
            lw=1.0,
            zorder=2,
        )
        ax.add_patch(box_l)
        cx_l = LEFT_X + LEFT_W / 2
        cy = (y_lo + row_top) / 2
        ax.text(cx_l, cy, r["question"],
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=PALETTE["histogram_charcoal"], zorder=3)

        # Right box (structural insufficiency)
        box_r = FancyBboxPatch(
            (RIGHT_X, y_lo), RIGHT_W, row_step - 0.04,
            boxstyle="round,pad=0.005",
            facecolor=(0.99, 0.99, 0.99),
            edgecolor=PALETTE["neutral_gray"],
            lw=0.8,
            zorder=2,
        )
        ax.add_patch(box_r)
        ax.text(RIGHT_X + RIGHT_W / 2, cy + 0.025,
                r["structural"],
                ha="center", va="center", fontsize=10,
                color=PALETTE["histogram_charcoal"], zorder=3)
        ax.text(RIGHT_X + RIGHT_W - 0.012, y_lo + 0.013,
                r["experiment_pointer"],
                ha="right", va="bottom", fontsize=9.5, fontweight="bold",
                color=PALETTE["stable_green"], zorder=3)
        row_top -= row_step

    # Bottom note
    ax.text(
        0.5, 0.03,
        "Each question is answered, with measurement, in the experiments that follow.",
        ha="center", va="center", fontsize=10.5, fontweight="bold", style="italic",
        color=PALETTE["histogram_charcoal"],
    )

    return save_pdf(fig, outdir / "figK_opening_questions.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_K(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
