"""
defense_figures/figure_QA/make_figure_QA.py

Figure QA — Questions.

Minimal closing slide for the defense deck. Title "Questions",
subtitle pointing at supplementary material. No sentimental
language, no "thank you for listening" billboard, no new scientific
claims. Formal closing only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE


def make_figure_QA(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.58,
            "Questions",
            ha="center", va="center",
            fontsize=44, fontweight="bold",
            color=PALETTE["histogram_charcoal"])

    ax.text(0.5, 0.42,
            "Supplementary figures and source-index slides are available for discussion.",
            ha="center", va="center",
            fontsize=12.5, style="italic",
            color=PALETTE["neutral_gray"])

    return save_pdf(fig, outdir / "figQA_questions.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_QA(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
