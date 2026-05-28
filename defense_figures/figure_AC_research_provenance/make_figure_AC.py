"""
defense_figures/figure_AC_research_provenance/make_figure_AC.py

Figure AC — Research Provenance and Assistance.

One global disclosure slide. Three blocks (Author-owned / AI-assisted
/ Verification) above a single author-authorized paragraph. Placement
in the deck: after OQ — Open Questions and before final
acknowledgments / Q&A.

The slide's purpose is provenance, not confession. It transparently
attributes scope to the author and to AI-assisted tooling without
defensive or apologetic framing. There are no per-figure AI labels
anywhere else in the deck; this single slide carries the entire
disclosure.

Content blocks and paragraph wording are author-authorized verbatim.
Do not paraphrase without explicit author re-authorization.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE


BLOCKS = [
    {
        "title": "Author-owned",
        "items": [
            "Research questions",
            "Dissertation claims",
            "Mathematical interpretation",
            "Experimental decisions",
            "Final inclusion decisions",
            "Defense argument",
        ],
        "accent": PALETTE["annotation_blue"],
        "fill":   (0.95, 0.97, 0.99),
    },
    {
        "title": "AI-assisted",
        "items": [
            "Defense-figure typesetting",
            "Repository organization",
            "Figure-generation script drafting",
            "Handoff documentation",
            "Audit organization",
        ],
        "accent": PALETTE["neutral_gray"],
        "fill":   (0.97, 0.97, 0.97),
    },
    {
        "title": "Verification",
        "items": [
            "Numerical claims tied to scripts / audit CSVs",
            "Source data and labels not invented by AI",
            "Author reviewed and accepted or revised each artifact",
        ],
        "accent": PALETTE["stable_green"],
        "fill":   (0.94, 0.99, 0.96),
    },
]

PARAGRAPH = (
    "The research questions, dissertation claims, mathematical interpretation, "
    "experimental decisions, and final defense argument are author-owned. "
    "AI-assisted coding and formatting tools were used to help organize repository "
    "artifacts, draft figure-generation scripts, typeset defense figures, and "
    "prepare handoff documentation from author-provided analyses and verified "
    "outputs. Numerical claims in the deck are traceable to scripts, audit CSVs, "
    "committed figures, or dissertation source material. AI tools did not supply "
    "clinical labels, invent results, determine conclusions, or replace author "
    "review."
)


def make_figure_AC(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Research Provenance and Assistance",
        fontsize=16, fontweight="bold", y=0.97,
    )

    # Three blocks across the top two-thirds.
    n = len(BLOCKS)
    gap = 0.025
    left = 0.04
    right = 0.96
    panel_w = (right - left - (n - 1) * gap) / n
    panel_top = 0.88
    panel_bot = 0.38
    panel_h = panel_top - panel_bot

    for i, b in enumerate(BLOCKS):
        x = left + i * (panel_w + gap)
        # Card
        ax.add_patch(FancyBboxPatch(
            (x, panel_bot), panel_w, panel_h,
            boxstyle="round,pad=0.008",
            facecolor=b["fill"],
            edgecolor=b["accent"],
            lw=1.4, zorder=2,
        ))
        # Header band
        header_h = 0.055
        ax.add_patch(FancyBboxPatch(
            (x, panel_top - header_h), panel_w, header_h,
            boxstyle="round,pad=0.002",
            facecolor=b["accent"],
            edgecolor=b["accent"],
            lw=0, zorder=3,
        ))
        ax.text(x + panel_w / 2, panel_top - header_h / 2,
                b["title"],
                ha="center", va="center",
                fontsize=12.5, fontweight="bold",
                color="white", zorder=4)
        # Items (pre-wrap to fit card width).
        wrapped_items = [textwrap.fill(it, width=28) for it in b["items"]]
        # Step accounts for the tallest wrapped item; size items uniformly.
        n_items = len(wrapped_items)
        item_top = panel_top - header_h - 0.030
        item_bot = panel_bot + 0.020
        step = (item_top - item_bot) / max(n_items, 1)
        for j, item in enumerate(wrapped_items):
            iy = item_top - j * step
            ax.text(x + 0.014, iy,
                    "•",
                    ha="left", va="top",
                    fontsize=12, fontweight="bold",
                    color=b["accent"], zorder=4)
            ax.text(x + 0.034, iy,
                    item,
                    ha="left", va="top",
                    fontsize=10.5,
                    color=PALETTE["histogram_charcoal"], zorder=4)

    # Paragraph box at bottom.
    box_top = 0.34
    box_bot = 0.06
    ax.add_patch(FancyBboxPatch(
        (left, box_bot), right - left, box_top - box_bot,
        boxstyle="round,pad=0.008",
        facecolor="white",
        edgecolor=PALETTE["histogram_charcoal"],
        lw=1.0, zorder=2,
    ))
    ax.text(left + 0.013, box_top - 0.025,
            "Provenance statement",
            ha="left", va="top",
            fontsize=9, fontweight="bold", style="italic",
            color=PALETTE["neutral_gray"], zorder=3)
    # matplotlib's wrap=True is unreliable; pre-wrap to a fixed width.
    wrapped = textwrap.fill(PARAGRAPH, width=120)
    ax.text(
        0.5, (box_top + box_bot) / 2 - 0.022,
        wrapped,
        ha="center", va="center",
        fontsize=10.5,
        color=PALETTE["histogram_charcoal"], zorder=3,
    )

    return save_pdf(fig, outdir / "figAC_research_provenance.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_AC(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
