"""
defense_figures/figure_F_theorem_scaffold/make_figure_F.py

Figure F — Theorem Scaffold.

A directed graph that names every theorem the dissertation rests on and
the architectural choice each theorem licenses. Lands the
"theory-grounded, not citation-grounded" requirement in one image.

OUTPUT
------
  outputs/figF_theorem_scaffold.pdf

PHILOSOPHICAL WORK
------------------
Citing a theorem is decoration. Naming the architectural decision a
theorem licenses is doctoral. This figure makes the licensing relation
explicit: each theorem → the choice it justifies → where in the
dissertation that choice is concretized.

Pesin's identity is deliberately excluded (see master plan §3). A
theorem scaffold should contain only theorems whose claims are measured
in this work. Five strongly-licensed beats six with one decorative.

THE THEOREMS (left column)
--------------------------
1. Takens' embedding theorem (Takens 1981) — a delay reconstruction of
   sufficient dimension preserves the latent attractor's topology.
2. Maass-Markram separation property (Maass et al. 2002) — a
   high-dimensional reservoir maps distinct inputs to distinct
   trajectories.
3. Oseledec multiplicative ergodic theorem (Oseledec 1968) — the
   Lyapunov spectrum is well-defined along an ergodic trajectory.
4. Echo State Property (Jaeger 2001) — for any input, the reservoir
   converges to an input-driven manifold.
5. Jaeger memory capacity (Jaeger 2001) — total recall capacity is
   bounded above by N and peaks at an interior leak rate.

THE CHOICES (right column)
--------------------------
A. Use a reservoir, sized to exceed the measured Takens dimension
   of the data. (Experiment B)
B. Use a high-N, sparse, non-trivial recurrent matrix.
C. Driven λ₁, not autonomous ρ(W), is the correct stability object.
   (Experiment A)
D. λ₁ < 0 verifies the ESP under real input.
   (Experiment A)
E. β is chosen at the measured MC peak/plateau, not validation accuracy.
   (Experiment C)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, figtext_footer, PALETTE, FIGSIZE


# ──────────────────────────────────────────────────────────────────────────
# Theorem → architectural choice mappings
# Author can refine the wording but the structural identity (5 nodes ↔
# 5 nodes, with one or more arrows per theorem-choice pair) is fixed.
# ──────────────────────────────────────────────────────────────────────────
THEOREMS = [
    {
        "id": "T",
        "name": "Takens' embedding theorem",
        "citation": "Takens (1981)",
        "claim": "A delay reconstruction of\nsufficient dimension preserves\nthe latent attractor's topology.",
    },
    {
        "id": "MM",
        "name": "Maass-Markram separation",
        "citation": "Maass, Natschläger, Markram (2002)",
        "claim": "A high-dimensional recurrent\nnetwork maps distinct inputs\nto distinct trajectories.",
    },
    {
        "id": "O",
        "name": "Oseledec multiplicative\nergodic theorem",
        "citation": "Oseledec (1968)",
        "claim": "Along an ergodic trajectory,\nthe Lyapunov spectrum is\nwell-defined.",
    },
    {
        "id": "ESP",
        "name": "Echo State Property",
        "citation": "Jaeger (2001)",
        "claim": "For any input, the reservoir\nconverges to an input-driven\nmanifold.",
    },
    {
        "id": "MC",
        "name": "Jaeger memory capacity",
        "citation": "Jaeger (2001)",
        "claim": "Total recall MC is bounded\nabove by N; peaks at an\ninterior leak rate.",
    },
]

CHOICES = [
    {
        "id": "A",
        "name": "Use a reservoir, sized to exceed\nthe latent attractor's Takens dimension.",
        "experiment": "Exp B — measured m* = 4, far below N=256.",
    },
    {
        "id": "B",
        "name": "Use a high-N, sparse, non-trivial\nrecurrent matrix.",
        "experiment": "N=256, σ=0.05, density 0.10\n(established in Ch. 3/6).",
    },
    {
        "id": "C",
        "name": "Driven λ₁⁽driven⁾, not autonomous\nρ(W), is the correct stability object.",
        "experiment": "Exp A — autonomous vs. driven\ncomparison.",
    },
    {
        "id": "D",
        "name": "λ₁ < 0 verifies the ESP\nunder real ERP drive.",
        "experiment": "Exp A — n=3,165, median λ₁ = −0.062,\n100% negative.",
    },
    {
        "id": "E",
        "name": "β chosen at the measured MC\nplateau, not validation accuracy.",
        "experiment": "Exp C — β=0.05 within plateau\n(0.010 ≤ β ≤ 0.118, MC ≥ 0.75).",
    },
]

# Edges: which theorem licenses which choice. Many-to-many.
EDGES = [
    ("T", "A"),
    ("MM", "B"),
    ("O", "C"),
    ("ESP", "D"),
    ("ESP", "C"),     # ESP also informs the stability-object discussion
    ("MC", "E"),
]


# ──────────────────────────────────────────────────────────────────────────
# Layout: theorems on left, choices on right, arrows between
# ──────────────────────────────────────────────────────────────────────────
def _node_y(i: int, n: int, top: float = 0.85, bottom: float = 0.20) -> float:
    if n == 1:
        return (top + bottom) / 2.0
    return top - i * (top - bottom) / (n - 1)


def make_figure_F(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    LEFT_X = 0.04
    LEFT_W = 0.32
    RIGHT_X = 0.60
    RIGHT_W = 0.36

    # Section headers — above the first row of boxes
    ax.text(LEFT_X + LEFT_W / 2, 0.93, "Theorems",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=PALETTE["histogram_charcoal"])
    ax.text(RIGHT_X + RIGHT_W / 2, 0.93, "Architectural choices they license",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=PALETTE["stable_green"])

    n_t = len(THEOREMS)
    n_c = len(CHOICES)

    theorem_anchors = {}
    for i, t in enumerate(THEOREMS):
        y = _node_y(i, n_t)
        # Box
        rect = FancyBboxPatch(
            (LEFT_X, y - 0.06), LEFT_W, 0.12,
            boxstyle="round,pad=0.01",
            facecolor="white",
            edgecolor=PALETTE["histogram_charcoal"],
            lw=1.2,
            zorder=2,
        )
        ax.add_patch(rect)
        # Title and citation
        ax.text(LEFT_X + LEFT_W / 2, y + 0.038, t["name"],
                ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=PALETTE["histogram_charcoal"], zorder=3)
        ax.text(LEFT_X + LEFT_W / 2, y + 0.000, t["citation"],
                ha="center", va="center", fontsize=8.5, style="italic",
                color=PALETTE["neutral_gray"], zorder=3)
        ax.text(LEFT_X + LEFT_W / 2, y - 0.036, t["claim"],
                ha="center", va="center", fontsize=8,
                color=PALETTE["histogram_charcoal"], zorder=3)
        theorem_anchors[t["id"]] = (LEFT_X + LEFT_W, y)

    choice_anchors = {}
    for i, c in enumerate(CHOICES):
        y = _node_y(i, n_c)
        rect = FancyBboxPatch(
            (RIGHT_X, y - 0.06), RIGHT_W, 0.12,
            boxstyle="round,pad=0.01",
            facecolor=(0.96, 0.99, 0.97),
            edgecolor=PALETTE["stable_green"],
            lw=1.5,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(RIGHT_X + RIGHT_W / 2, y + 0.030, c["name"],
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=PALETTE["stable_green"], zorder=3)
        ax.text(RIGHT_X + RIGHT_W / 2, y - 0.030, c["experiment"],
                ha="center", va="center", fontsize=8.5, style="italic",
                color=PALETTE["histogram_charcoal"], zorder=3)
        choice_anchors[c["id"]] = (RIGHT_X, y)

    # Arrows: theorem → choice
    for t_id, c_id in EDGES:
        x1, y1 = theorem_anchors[t_id]
        x2, y2 = choice_anchors[c_id]
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->",
            mutation_scale=15,
            color=PALETTE["neutral_gray"],
            lw=1.3,
            alpha=0.75,
            connectionstyle="arc3,rad=0.08",
            zorder=1,
        )
        ax.add_patch(arrow)

    # Pesin exclusion note (bottom footnote) — wrapped manually
    ax.text(
        0.5, 0.04,
        "Pesin's identity is deliberately excluded — it licenses claims about systems with positive\n"
        "Lyapunov exponents, which this dissertation neither measures nor uses (operating regime: λ₁ < 0).",
        ha="center", va="bottom",
        fontsize=8.5, style="italic",
        color=PALETTE["neutral_gray"],
    )

    fig.suptitle(
        "Theorem scaffold: every architectural choice traces to a named theorem the dissertation measures",
        fontsize=13, fontweight="bold", y=0.99,
    )

    return save_pdf(fig, outdir / "figF_theorem_scaffold.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p = make_figure_F(outdir)
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
