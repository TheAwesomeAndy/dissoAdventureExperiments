"""
defense_figures/figure_OQ_open_questions/make_figure_OQ.py

Figure OQ — Open Questions (closing slide).

A PhD is a beginning, not an end. This slide names five unresolved
questions the dissertation *raises* but does not answer — the single
most doctoral move possible.

CONTENT
-------
Five entries drawn from the experiments' implicit boundaries. Each open
question paired with the experiment that raises it and a brief note on
what would resolve it. Author should refine wording as appropriate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE, FIGSIZE


QUESTIONS = [
    {
        "tag": "Exp B",
        "question": "Does a much smaller reservoir suffice?",
        "context": (
            "The Takens margin is 16–64×. The reservoir is far larger than the data\n"
            "requires. A minimal-N study (N = 64, post-PCA = 16, or smaller) would test\n"
            "whether the dissertation's reservoir is over-provisioned for its actual\n"
            "measurement task."
        ),
        "what_would_resolve": "Follow-up: minimal-N ablation.",
    },
    {
        "tag": "Exp C",
        "question": "Can the Propagation Operating Characteristic be derived analytically?",
        "context": (
            "The MC vs β curve was measured empirically. For known reservoir families\n"
            "(LIFReservoir at target ρ, sparse-Gaussian at given σ and density), does\n"
            "the MC peak position admit a closed-form expression? This is a tractable\n"
            "analytical follow-up question."
        ),
        "what_would_resolve": "Open theoretical question.",
    },
    {
        "tag": "Exp D",
        "question": "Does the channel-permutation finding replicate on an independent clinical cohort?",
        "context": (
            "The methodology survives the strictest spatial null at the stimulus-class\n"
            "level. The per-disorder version (SUD, PTSD, GAD, ADHD) awaits an\n"
            "independent replication cohort. The dissertation does not yet have one."
        ),
        "what_would_resolve": "Pending external data.",
    },
    {
        "tag": "Framework",
        "question": "Does the Measurement-Instrument Paradigm generalize to MEG and fMRI?",
        "context": (
            "The reservoir was treated as a measurement instrument with characterized\n"
            "ρ, λ₁, MC, m*. Other neuroimaging modalities have different noise structures\n"
            "and timescales. Whether the paradigm transfers — and what it predicts in\n"
            "those modalities — is an open architectural question."
        ),
        "what_would_resolve": "Cross-modality validation study.",
    },
    {
        "tag": "POC",
        "question": "Is there a single theoretical object that unifies driven contraction (λ₁) and memory capacity (MC)?",
        "context": (
            "The Propagation Operating Characteristic frames these as related, but treats\n"
            "them as two independent measurements. Whether they are two faces of one\n"
            "analytical object — say, a Lyapunov-weighted memory functional — is an open\n"
            "theoretical question the dissertation poses but cannot answer."
        ),
        "what_would_resolve": "Open theoretical question.",
    },
]


def make_figure_OQ(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Five questions this dissertation opens.  Each is the start of work that someone, someday, will do.",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # Subtitle
    ax.text(0.5, 0.93,
            "A PhD is a beginning, not an end.",
            ha="center", va="center",
            fontsize=11, style="italic",
            color=PALETTE["neutral_gray"])

    # Layout: 5 rows
    n = len(QUESTIONS)
    top_y = 0.88
    bot_y = 0.13
    row_step = (top_y - bot_y) / n

    for i, q in enumerate(QUESTIONS):
        y_top = top_y - i * row_step
        y_bot = y_top - row_step + 0.02
        # Container box
        box = FancyBboxPatch(
            (0.03, y_bot), 0.94, row_step - 0.04,
            boxstyle="round,pad=0.005",
            facecolor="white",
            edgecolor=PALETTE["histogram_charcoal"],
            lw=1.0,
            zorder=2,
        )
        ax.add_patch(box)

        # Tag on the left
        ax.text(0.05, y_top - row_step / 2,
                q["tag"],
                ha="left", va="center",
                fontsize=10, fontweight="bold", style="italic",
                color=PALETTE["stable_green"], zorder=3)

        # Question (large)
        ax.text(0.18, y_top - 0.018,
                q["question"],
                ha="left", va="top",
                fontsize=12, fontweight="bold",
                color=PALETTE["histogram_charcoal"], zorder=3)

        # Context (smaller)
        ax.text(0.18, y_top - 0.045,
                q["context"],
                ha="left", va="top",
                fontsize=9.5,
                color=PALETTE["histogram_charcoal"], zorder=3)

        # What would resolve (italic, right side)
        ax.text(0.95, y_bot + 0.012,
                q["what_would_resolve"],
                ha="right", va="bottom",
                fontsize=9, style="italic", fontweight="bold",
                color=PALETTE["annotation_blue"], zorder=3)

    # Closing line at very bottom
    ax.text(
        0.5, 0.04,
        '"This dissertation answers the questions it asks; it also opens five it cannot answer.\n'
        'Asking better questions than I can answer is what a doctorate of philosophy is for."',
        ha="center", va="center",
        fontsize=11, fontweight="bold", style="italic",
        color=PALETTE["histogram_charcoal"],
    )

    return save_pdf(fig, outdir / "figOQ_open_questions.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_OQ(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
