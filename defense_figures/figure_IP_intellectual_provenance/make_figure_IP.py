"""
defense_figures/figure_IP_intellectual_provenance/make_figure_IP.py

Figure IP — Intellectual Provenance.

A single page naming the thinkers who shaped HOW the dissertation asks
its questions. Intellectual lineage, not citation dump. Each entry is
"from X, I learned to ask Y" — demonstrates the work is part of a
scholarly conversation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE, FIGSIZE


ENTRIES = [
    {
        "thinkers": "Takens, Sauer, Yorke",
        "domain": "embedding theory",
        "citation": "Takens 1981 · Sauer–Yorke–Casdagli 1991",
        "learned": (
            "…whether the data's latent dynamics could be reconstructed from a\n"
            "sufficiently rich observation function — and whether that reconstruction\n"
            "is measurable on real ERPs."
        ),
        "appears_in": "Exp B",
    },
    {
        "thinkers": "Maass, Natschläger, Markram",
        "domain": "liquid state machines",
        "citation": "Maass, Natschläger, Markram 2002",
        "learned": (
            "…what kind of dynamical state space a useful reservoir must possess,\n"
            "and what separation property it must exhibit for distinct inputs."
        ),
        "appears_in": "Exp B premise · Ch. 5 architecture",
    },
    {
        "thinkers": "Jaeger, Lukoševičius",
        "domain": "Echo State Network theory",
        "citation": "Jaeger 2001 · Lukoševičius–Jaeger 2009",
        "learned": (
            "…whether the reservoir's response to input was consistent enough to be\n"
            "readable — and how to measure that consistency rather than assume it."
        ),
        "appears_in": "Exp A",
    },
    {
        "thinkers": "Oseledec, Benettin, Wolf",
        "domain": "Lyapunov spectra & numerical methods",
        "citation": "Oseledec 1968 · Benettin et al. 1980 · Wolf et al. 1985",
        "learned": (
            "…what stability MEANS along a driven trajectory — and how to compute it\n"
            "when the analytic Jacobian is undefined at threshold events."
        ),
        "appears_in": "Exp A method",
    },
    {
        "thinkers": "Bullmore, Sporns",
        "domain": "graph-theoretic neuroscience",
        "citation": "Bullmore & Sporns 2009",
        "learned": (
            "…whether the brain's spatial structure could be operationalized as a\n"
            "measurable graph — and whether that graph carries clinical information\n"
            "that survives nullification."
        ),
        "appears_in": "Ch. 5 graph construction · Exp D",
    },
    {
        "thinkers": "Marder, Krakauer et al.",
        "domain": "philosophy of neuroscience",
        "citation": "Marder 2015 · Krakauer, Ghazanfar, Gomez-Marin, MacIver, Poeppel 2017",
        "learned": (
            "…whether benchmark accuracy is what we should optimize for in clinical\n"
            "neuroscience — and what we lose when interpretability becomes the\n"
            "trade variable."
        ),
        "appears_in": "Methodological Refusals · J contributions",
    },
]


def make_figure_IP(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "From these thinkers I learned how to ask — not just what to cite.",
        fontsize=14, fontweight="bold", y=0.99,
    )
    ax.text(0.5, 0.93,
            "Intellectual provenance, not citation dump.  Each entry names a question I would not have asked without them.",
            ha="center", va="center",
            fontsize=10, style="italic",
            color=PALETTE["neutral_gray"])

    n = len(ENTRIES)
    top_y = 0.88
    bot_y = 0.07
    row_step = (top_y - bot_y) / n

    LEFT_X, LEFT_W = 0.03, 0.28
    RIGHT_X, RIGHT_W = 0.34, 0.64

    for i, e in enumerate(ENTRIES):
        y_top = top_y - i * row_step
        y_bot = y_top - row_step + 0.012
        # Left: thinker name + domain
        box_l = FancyBboxPatch(
            (LEFT_X, y_bot), LEFT_W, row_step - 0.024,
            boxstyle="round,pad=0.005",
            facecolor="white",
            edgecolor=PALETTE["histogram_charcoal"],
            lw=1.0,
            zorder=2,
        )
        ax.add_patch(box_l)
        ax.text(LEFT_X + LEFT_W / 2, (y_top + y_bot) / 2 + 0.022,
                e["thinkers"],
                ha="center", va="center",
                fontsize=11.5, fontweight="bold",
                color=PALETTE["histogram_charcoal"], zorder=3)
        ax.text(LEFT_X + LEFT_W / 2, (y_top + y_bot) / 2 + 0.005,
                e["domain"],
                ha="center", va="center",
                fontsize=9.5, style="italic",
                color=PALETTE["neutral_gray"], zorder=3)
        ax.text(LEFT_X + LEFT_W / 2, y_bot + 0.014,
                e["citation"],
                ha="center", va="bottom",
                fontsize=7.5, style="italic",
                color=PALETTE["neutral_gray"], zorder=3)

        # Right: "I learned to ask…" + appears_in
        box_r = FancyBboxPatch(
            (RIGHT_X, y_bot), RIGHT_W, row_step - 0.024,
            boxstyle="round,pad=0.005",
            facecolor=(0.96, 0.99, 0.97),
            edgecolor=PALETTE["stable_green"],
            lw=1.1,
            zorder=2,
        )
        ax.add_patch(box_r)
        # Lead-in label
        ax.text(RIGHT_X + 0.012, y_top - 0.018,
                'From them, I learned to ask…',
                ha="left", va="top",
                fontsize=9, style="italic",
                color=PALETTE["stable_green"], zorder=3)
        ax.text(RIGHT_X + RIGHT_W / 2, (y_top + y_bot) / 2 + 0.005,
                e["learned"],
                ha="center", va="center",
                fontsize=10,
                color=PALETTE["histogram_charcoal"], zorder=3)
        ax.text(RIGHT_X + RIGHT_W - 0.012, y_bot + 0.012,
                e["appears_in"],
                ha="right", va="bottom",
                fontsize=9, fontweight="bold", style="italic",
                color=PALETTE["stable_green"], zorder=3)

    return save_pdf(fig, outdir / "figIP_intellectual_provenance.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_IP(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
