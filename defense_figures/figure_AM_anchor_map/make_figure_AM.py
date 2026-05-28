"""
defense_figures/figure_AM_anchor_map/make_figure_AM.py

Figure AM — Defense Claims Are Anchored to Dissertation Chapters.

Produces two PDFs:
  1. figAM_anchor_map.pdf      — main-deck slide (spoken; 30 seconds).
  2. figAM_source_index.pdf    — backup appendix (committee due
                                  diligence; printed-only).

Both slides intentionally omit page numbers. Per author direction,
page-level anchors are brittle during LaTeX edits and will be added
by the author from a stable compiled dissertation PDF if needed. The
backup index lists section as "—" for the same reason; finer
granularity will be filled in by the author.

The main slide proves the defense is a compressed traversal of the
dissertation's theoretical and experimental spine — not a detached
sequence of slides. The backup is for due-diligence reference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _style import apply_style, save_pdf, PALETTE


MAIN_ROWS = [
    ("K / IP / F",  "Ch. 1–3",                "Research questions, theory, and prior-work positioning"),
    ("Exp A",       "Ch. 6",                  "Driven Lyapunov measurement of reservoir operation"),
    ("Exp B",       "Ch. 4 / Ch. 6",          "FNN-estimated embedding dimension relative to reservoir state capacity"),
    ("Exp C",       "Ch. 6",                  "Memory-capacity regime analysis for β"),
    ("Exp D",       "Ch. 5",                  "Spatial-null test for stimulus-class channel structure"),
    ("G",           "Ch. 5–6 + audit notes",  "Failure-to-pivot narrative"),
    ("J / OQ",      "Ch. 7–8",                "Contributions, limits, and open problems"),
]


BACKUP_ROWS = [
    # (Figure, Script / output path, Chapter, Notes)
    # Section and page columns intentionally omitted — see footer.
    ("K",        "figure_K_opening_questions/",        "Ch. 1",                 "Opening questions"),
    ("IP",       "figure_IP_intellectual_provenance/", "Ch. 2",                 "Scholarly-lineage map"),
    ("F",        "figure_F_theorem_scaffold/",         "Ch. 3",                 "Theorem scaffold (Takens / Jaeger / Maass / Kennel–Brown)"),
    ("MR",       "figure_MR_methodological_refusals/", "Ch. 5–6 (transversal)", "Methodological refusals — rides on Exp D / C anchors"),
    ("Exp A",    "experiment_a_autonomous_vs_driven/", "Ch. 6",                 "Autonomous ρ(W) → driven Lyapunov λ₁ measured on 3,165 trials"),
    ("Exp B",    "experiment_b_takens_dimension/",     "Ch. 4 / Ch. 6",         "Takens-motivated question, FNN-measured embedding dimension"),
    ("Exp C",    "experiment_c_memory_capacity/",      "Ch. 6",                 "Propagation Operating Characteristic; β = 0.05 vs measured β*"),
    ("TB",       "figure_TB_theoretical_bounds/",      "Ch. 6 (reads A / C)",   "Analytic bounds vs measured operating points"),
    ("Exp D",    "experiment_d_channel_permutation/",  "Ch. 5",                 "Per-trial channel-permutation null; stimulus-class demonstration"),
    ("G",        "figure_G_failure_gallery/",          "Ch. 5–6 + audit",       "Failure-to-pivot narrative across A, C, D"),
    ("J",        "figure_J_contributions/",            "Ch. 7",                 "Five named contributions"),
    ("AM",       "figure_AM_anchor_map/",              "—",                     "This slide (self-reference; no anchor)"),
    ("OQ",       "figure_OQ_open_questions/",          "Ch. 8",                 "Open questions / closing"),
    ("AC",       "figure_AC_research_provenance/",     "—",                     "Research-provenance disclosure (transversal)"),
]


def _draw_main(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Defense Claims Are Anchored to Dissertation Chapters",
        fontsize=16, fontweight="bold", y=0.97,
    )
    ax.text(
        0.5, 0.91,
        "Each defense object is a compressed traversal of a region of the dissertation's theoretical and experimental spine.",
        ha="center", va="center",
        fontsize=10.5, style="italic",
        color=PALETTE["neutral_gray"],
    )

    cols = ["Defense object", "Dissertation location", "Claim function"]
    col_x = [0.04, 0.22, 0.46]
    col_w = [0.18, 0.24, 0.50]
    body_top = 0.85
    body_bot = 0.16
    n_rows = len(MAIN_ROWS)
    row_h = (body_top - body_bot) / (n_rows + 1)

    # Header band
    ax.add_patch(FancyBboxPatch(
        (col_x[0], body_top - row_h), sum(col_w), row_h - 0.008,
        boxstyle="round,pad=0.002",
        facecolor=PALETTE["annotation_blue"],
        edgecolor=PALETTE["annotation_blue"],
        lw=0, zorder=2,
    ))
    for c, (title, x, w) in enumerate(zip(cols, col_x, col_w)):
        ax.text(x + 0.010, body_top - row_h / 2,
                title,
                ha="left", va="center",
                fontsize=11.5, fontweight="bold",
                color="white", zorder=3)

    # Body rows
    for i, row in enumerate(MAIN_ROWS):
        y_top = body_top - (i + 1) * row_h
        # Alternate-row subtle banding
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch(
                (col_x[0], y_top - row_h + 0.008), sum(col_w), row_h - 0.008,
                boxstyle="round,pad=0.001",
                facecolor=(0.97, 0.97, 0.98),
                edgecolor="none",
                lw=0, zorder=1,
            ))
        for c, (cell, x, w) in enumerate(zip(row, col_x, col_w)):
            ax.text(x + 0.010, y_top - row_h / 2,
                    cell,
                    ha="left", va="center",
                    fontsize=11 if c == 0 else 10.5,
                    fontweight="bold" if c == 0 else "normal",
                    color=PALETTE["histogram_charcoal"], zorder=3)

    ax.text(
        0.5, 0.08,
        "Page-level anchors are intentionally omitted; they are brittle during LaTeX edits.  "
        "Per-figure source paths and chapter-level anchors live in the backup index.",
        ha="center", va="center",
        fontsize=9, style="italic",
        color=PALETTE["neutral_gray"],
    )

    return save_pdf(fig, outdir / "figAM_anchor_map.pdf")


def _draw_backup(outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Figure-to-Source Index  (Backup Appendix)",
        fontsize=15, fontweight="bold", y=0.975,
    )
    ax.text(
        0.5, 0.94,
        "Committee due-diligence reference. Printed-only; not presented in the spoken deck.",
        ha="center", va="center",
        fontsize=10, style="italic",
        color=PALETTE["neutral_gray"],
    )

    cols = ["Figure", "Script / output path", "Chapter", "Notes"]
    col_x = [0.025, 0.105, 0.405, 0.580]
    col_w = [0.080, 0.300, 0.175, 0.390]
    body_top = 0.90
    body_bot = 0.07
    n_rows = len(BACKUP_ROWS)
    row_h = (body_top - body_bot) / (n_rows + 1)

    # Header band
    ax.add_patch(FancyBboxPatch(
        (col_x[0], body_top - row_h), sum(col_w), row_h - 0.005,
        boxstyle="round,pad=0.002",
        facecolor=PALETTE["histogram_charcoal"],
        edgecolor=PALETTE["histogram_charcoal"],
        lw=0, zorder=2,
    ))
    for title, x in zip(cols, col_x):
        ax.text(x + 0.006, body_top - row_h / 2,
                title,
                ha="left", va="center",
                fontsize=10.5, fontweight="bold",
                color="white", zorder=3)

    # Body rows
    for i, row in enumerate(BACKUP_ROWS):
        y_top = body_top - (i + 1) * row_h
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch(
                (col_x[0], y_top - row_h + 0.005), sum(col_w), row_h - 0.005,
                boxstyle="round,pad=0.001",
                facecolor=(0.97, 0.97, 0.98),
                edgecolor="none",
                lw=0, zorder=1,
            ))
        # Use monospace for the path column
        for c, (cell, x) in enumerate(zip(row, col_x)):
            family = "monospace" if c == 1 else "serif"
            weight = "bold" if c == 0 else "normal"
            ax.text(x + 0.006, y_top - row_h / 2,
                    cell,
                    ha="left", va="center",
                    fontsize=9.5,
                    fontweight=weight,
                    family=family,
                    color=PALETTE["histogram_charcoal"], zorder=3)

    ax.text(
        0.5, 0.035,
        "Section and page anchors are intentionally omitted: they are brittle across LaTeX edits.  "
        "Author fills in finer granularity from the stable compiled dissertation PDF as required.",
        ha="center", va="center",
        fontsize=8.5, style="italic",
        color=PALETTE["neutral_gray"],
    )

    return save_pdf(fig, outdir / "figAM_source_index.pdf")


def main():
    apply_style()
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    p1 = _draw_main(outdir)
    p2 = _draw_backup(outdir)
    print(f"  wrote {p1}")
    print(f"  wrote {p2}")


if __name__ == "__main__":
    main()
