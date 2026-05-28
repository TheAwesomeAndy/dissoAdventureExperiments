"""
defense_figures/_style.py

Shared style module for the dissertation defense deck. Every
`make_figure_*.py` script in `defense_figures/` imports from here. This
module is the single source of truth for fonts, palette, figure sizes,
and PDF save conventions. Any figure that bypasses this module will fail
the deck-assembly visual-consistency check.

Import contract:
    from defense_figures._style import apply_style, PALETTE, FIGSIZE, save_pdf

Conventions match `experiments/chapter3/run_chapter3_lsm_characterization.py:44-49`
(serif, dpi=300) and category palette in
`chapter6Experiments/run_chapter6_exp1_esp.py:37`.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "unstable_red": "#d63031",
    "stable_green": "#00b894",
    "neutral_gray": "#636e72",
    "histogram_charcoal": "#2d3436",
    "annotation_blue": "#0984e3",
    "highlight_purple": "#8e44ad",
    "category_threat": "#e74c3c",
    "category_mutilation": "#c0392b",
    "category_cute": "#27ae60",
    "category_erotic": "#2ecc71",
}

FIGSIZE = {
    "slide_landscape": (12.0, 5.5),
    "slide_landscape_tall": (12.0, 6.5),
    "slide_portrait": (8.0, 10.0),
    "slide_square": (8.0, 8.0),
    "raw_two_panel": (12.0, 5.0),
    "raw_grid_3x4": (16.0, 11.0),
    "raw_strip": (14.0, 5.0),
    "scaffold": (12.0, 7.0),
    "questions": (12.0, 7.0),
    "contributions": (14.0, 7.0),
}

CATEGORY_COLORS = {
    "Threat": PALETTE["category_threat"],
    "Mutilation": PALETTE["category_mutilation"],
    "Cute": PALETTE["category_cute"],
    "Erotic": PALETTE["category_erotic"],
}


def apply_style() -> None:
    """Set matplotlib rcParams for the defense deck. Call once per script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def save_pdf(fig, path: str | os.PathLike) -> Path:
    """
    Save `fig` as PDF at `path`, enforcing deck conventions.
    Creates parent directories if needed. Closes the figure.
    Returns the absolute Path written.
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)
    return p


def figtext_footer(fig, text: str, *, y: float = 0.02) -> None:
    """
    Add a bottom-spanning footer to the figure. Used for the bold
    "Autonomous theory describes the reservoir at rest. Driven theory
    describes the reservoir doing its job." style strip.
    """
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=11, fontweight="bold",
             family="serif", color=PALETTE["histogram_charcoal"])
