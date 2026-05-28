"""
defense_figures/figure_J_contributions/make_figure_J.py

Figure J — Contributions Slide.

The closing slide of the defense deck. Three columns name **what this
dissertation gives back to the field**, by NAME — not by category.

A category-labeled slide ("Theory / Measurement / Clinical insight")
gives the committee three buckets; a named-contribution slide gives
them five things that travel back into the field with the author's
name on them.

STRUCTURAL IDENTITY: 3 theoretical + 1 methodological + 1 empirical-discovery
                   = 5 named contributions.

The names below are PLACEHOLDERS, drawn from the author's own
characterization in the master plan. The author should validate the
exact mapping (which name goes in which column) and refine each
"what it is" / "where it is established" statement in their own voice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE, FIGSIZE


# Each contribution: name, what-it-is, where-it-is-established
THEORETICAL = [
    {
        "name": "Propagation Operating Characteristic",
        "what": ("A measured curve relating driven contraction (λ₁) to memory\n"
                 "capacity. Identifies the operating region in which a driven\n"
                 "reservoir is jointly stable and information-preserving."),
        "where": "Established: Chapter 6.  Used to derive β = 0.05 in Exp C.",
    },
    {
        "name": "Measurement-Instrument Paradigm",
        "what": ("A methodological commitment to treat the reservoir as a\n"
                 "measurement instrument whose properties (ρ, λ₁, MC, Takens-\n"
                 "sufficiency) are characterized before use — not as a black-\n"
                 "box learner."),
        "where": "Established: Chapters 3 and 6.  Operationalized in Exp A / B / C.",
    },
    {
        "name": "Spike-to-Embedding Pipeline",
        "what": ("A reproducible, interpretable transformation from raw ERP\n"
                 "to a low-dimensional embedding via BSC₆ and PCA-64, with\n"
                 "measured sufficiency for downstream classification."),
        "where": "Established: Chapter 4.",
    },
]

METHODOLOGICAL = [
    {
        "name": "Layer Ablation Methodology",
        "what": ("A per-disorder, per-layer ablation protocol that isolates\n"
                 "the contribution of each architectural level to clinical\n"
                 "separability, tested against a channel-permutation null."),
        "where": "Established: Chapter 5.  Tested in Exp D.",
    },
]

EMPIRICAL = [
    {
        "name": "Centered Baseline Comparison",
        "what": ("A clinical-cohort comparison against a centered,\n"
                 "demographically-matched baseline (not an arbitrary control),\n"
                 "producing the SUD and PTSD findings that survive permutation\n"
                 "testing."),
        "where": "Established: Chapter 5.  Tested in Exp D.",
    },
]


def _draw_column_header(ax, x: float, w: float, y: float, text: str, color: str):
    ax.text(x + w / 2, y, text,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=color)


def _draw_card(ax, x: float, y: float, w: float, h: float, c: dict,
               border_color: str):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005",
        facecolor="white",
        edgecolor=border_color,
        lw=1.4,
        zorder=2,
    )
    ax.add_patch(box)
    # Name (top, larger)
    ax.text(x + w / 2, y + h - 0.025,
            c["name"],
            ha="center", va="top",
            fontsize=11.2, fontweight="bold",
            color=border_color, zorder=3)
    # What (middle)
    ax.text(x + w / 2, y + h / 2 - 0.005,
            c["what"],
            ha="center", va="center",
            fontsize=9, color=PALETTE["histogram_charcoal"], zorder=3)
    # Where (bottom, italic)
    ax.text(x + w / 2, y + 0.012,
            c["where"],
            ha="center", va="bottom",
            fontsize=8.5, style="italic",
            color=PALETTE["neutral_gray"], zorder=3)


def make_figure_J(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Contributions: five named things that travel back into the field",
        fontsize=15, fontweight="bold", y=0.98,
    )
    ax.text(0.5, 0.94,
            "Three theoretical · one methodological · one empirical-discovery",
            ha="center", va="center",
            fontsize=10.5, style="italic",
            color=PALETTE["neutral_gray"])

    # Column geometry
    COL1_X, COL1_W = 0.03, 0.30
    COL2_X, COL2_W = 0.36, 0.30
    COL3_X, COL3_W = 0.69, 0.28

    THEORY_COLOR = PALETTE["stable_green"]
    METHOD_COLOR = PALETTE["annotation_blue"]
    EMPIRICAL_COLOR = PALETTE["highlight_purple"]

    # Headers
    _draw_column_header(ax, COL1_X, COL1_W, 0.88, "THEORETICAL (×3)", THEORY_COLOR)
    _draw_column_header(ax, COL2_X, COL2_W, 0.88, "METHODOLOGICAL (×1)", METHOD_COLOR)
    _draw_column_header(ax, COL3_X, COL3_W, 0.88, "EMPIRICAL DISCOVERY (×1)", EMPIRICAL_COLOR)

    # Layout: all three columns span y ∈ [0.13, 0.85].
    # Left column: 3 stacked cards of height 0.22 with 0.02 gap.
    # Middle / right columns: 1 large card spanning the same vertical extent
    # as the three left cards combined.
    card_h_T = 0.22
    spacing_T = 0.02
    base_top = 0.85
    for i, c in enumerate(THEORETICAL):
        y = base_top - (i + 1) * card_h_T - i * spacing_T
        _draw_card(ax, COL1_X, y, COL1_W, card_h_T, c, THEORY_COLOR)

    full_h = 3 * card_h_T + 2 * spacing_T
    full_y = base_top - full_h  # = 0.13
    _draw_card(ax, COL2_X, full_y, COL2_W, full_h, METHODOLOGICAL[0], METHOD_COLOR)
    _draw_card(ax, COL3_X, full_y, COL3_W, full_h, EMPIRICAL[0], EMPIRICAL_COLOR)

    # Closing sentence at the bottom
    ax.text(
        0.5, 0.03,
        '"These five names — the POC, the MIP, the spike-to-embedding pipeline, the layer ablation methodology,\n'
        'the centered baseline comparison — are this dissertation\'s contributions.  '
        'They travel back into the field with my name on them."',
        ha="center", va="center", fontsize=10.5, fontweight="bold", style="italic",
        color=PALETTE["histogram_charcoal"],
    )

    return save_pdf(fig, outdir / "figJ_contributions.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_J(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
