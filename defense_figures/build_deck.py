"""
defense_figures/build_deck.py

Assemble the dissertation defense deck.

Concatenates pre-rendered defense-figure PDFs into two artifacts:

  outputs/ARSPI_Net_Defense_Main.pdf       — the 15-slide spoken deck.
  outputs/ARSPI_Net_Defense_Appendix.pdf   — backup PDFs (committee due
                                              diligence; not spoken).

Also writes outputs/deck_manifest.json describing the exact slide
order and source paths for each artifact.

Hard rules (this script enforces them; do not soften):
  - No silent skips. Every entry in MAIN_SLIDES must exist on disk;
    if any is missing the script aborts before writing anything.
  - No regeneration. This script never invokes a make_figure_*.py.
  - No figure modification. Source PDFs are concatenated as-is.
  - No new content. No title pages, no section dividers, no page
    numbers added by this script.
  - Main and appendix are written to separate output files.

For Exp A–D and TB, the main deck uses the analysis/summary PDF
(analysis*.pdf), not the raw diagnostic strip. Raw diagnostics live
in the appendix only.

Run from repo root:
    python3 defense_figures/build_deck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


HERE = Path(__file__).resolve().parent


# Main deck: ordered 1..15, each is (id, title, relative path under defense_figures/).
MAIN_SLIDES: list[tuple[str, str, str]] = [
    ("K",     "Opening Questions",                 "figure_K_opening_questions/outputs/figK_opening_questions.pdf"),
    ("MR",    "Methodological Refusals",           "figure_MR_methodological_refusals/outputs/figMR_methodological_refusals.pdf"),
    ("IP",    "Intellectual Provenance",           "figure_IP_intellectual_provenance/outputs/figIP_intellectual_provenance.pdf"),
    ("F",     "Theorem Scaffold",                  "figure_F_theorem_scaffold/outputs/figF_theorem_scaffold.pdf"),
    ("A",     "Exp A — Driven Lyapunov",           "experiment_a_autonomous_vs_driven/outputs/analysisA_1e_autonomous_vs_driven.pdf"),
    ("B",     "Exp B — FNN Embedding Dimension",   "experiment_b_takens_dimension/outputs/analysisB_2e_takens_dimension.pdf"),
    ("C",     "Exp C — Memory-Capacity Regime",    "experiment_c_memory_capacity/outputs/analysisC_3d_memory_capacity_peak.pdf"),
    ("TB",    "Bounds vs Measurement",             "figure_TB_theoretical_bounds/outputs/analysisTB_5a_bounds_vs_measurement.pdf"),
    ("D",     "Exp D — Channel-Permutation Null",  "experiment_d_channel_permutation/outputs/analysisD_4d_permutation_nulls.pdf"),
    ("G",     "Failure Gallery",                   "figure_G_failure_gallery/outputs/figG_failure_gallery.pdf"),
    ("J",     "Contributions",                     "figure_J_contributions/outputs/figJ_contributions.pdf"),
    ("AM",    "Dissertation Anchor Map",           "figure_AM_anchor_map/outputs/figAM_anchor_map.pdf"),
    ("OQ",    "Open Questions",                    "figure_OQ_open_questions/outputs/figOQ_open_questions.pdf"),
    ("AC",    "Research Provenance and Assistance","figure_AC_research_provenance/outputs/figAC_research_provenance.pdf"),
    ("QA",    "Questions",                         "figure_QA/outputs/figQA_questions.pdf"),
]


# Appendix: backup material in main-deck order. Each entry is
# (id, label, relative path). Raw diagnostic strips for Exp A–D
# follow each experiment's position in the spoken arc.
APPENDIX_PDFS: list[tuple[str, str, str]] = [
    ("A_raw_1a", "Exp A raw — weight matrix",                "experiment_a_autonomous_vs_driven/outputs/rawA_1a_weight_matrix.pdf"),
    ("A_raw_1b", "Exp A raw — eigenspectrum",                "experiment_a_autonomous_vs_driven/outputs/rawA_1b_eigenspectrum.pdf"),
    ("A_raw_1c", "Exp A raw — Benettin sample trajectories", "experiment_a_autonomous_vs_driven/outputs/rawA_1c_benettin_sample_trajectories.pdf"),
    ("A_raw_1d", "Exp A raw — per-trial λ scatter",          "experiment_a_autonomous_vs_driven/outputs/rawA_1d_per_trial_lambda_scatter.pdf"),
    ("A_raw_1f", "Exp A raw — contraction-animation poster", "experiment_a_autonomous_vs_driven/outputs/rawA_1f_contraction_animation.pdf"),
    ("B_raw_2a", "Exp B raw — ERP samples",                  "experiment_b_takens_dimension/outputs/rawB_2a_erp_samples.pdf"),
    ("B_raw_2b", "Exp B raw — delay embedding",              "experiment_b_takens_dimension/outputs/rawB_2b_delay_embedding.pdf"),
    ("B_raw_2c", "Exp B raw — τ sweep",                      "experiment_b_takens_dimension/outputs/rawB_2c_tau_sweep.pdf"),
    ("B_raw_2d", "Exp B raw — per-trial FNN",                "experiment_b_takens_dimension/outputs/rawB_2d_per_trial_fnn.pdf"),
    ("C_raw_3a", "Exp C raw — input drive",                  "experiment_c_memory_capacity/outputs/rawC_3a_input_drive.pdf"),
    ("C_raw_3b", "Exp C raw — state response per β",         "experiment_c_memory_capacity/outputs/rawC_3b_state_response_per_beta.pdf"),
    ("C_raw_3c", "Exp C raw — MC vs τ",                      "experiment_c_memory_capacity/outputs/rawC_3c_mc_vs_tau.pdf"),
    ("D_raw_4a", "Exp D raw — feature layout",               "experiment_d_channel_permutation/outputs/rawD_4a_feature_layout.pdf"),
    ("D_raw_4b", "Exp D raw — class-label distribution",     "experiment_d_channel_permutation/outputs/rawD_4b_class_label_distribution.pdf"),
    ("D_raw_4c", "Exp D raw — observed vs null example",     "experiment_d_channel_permutation/outputs/rawD_4c_observed_vs_null_example.pdf"),
    ("AM_idx",   "AM — Figure-to-Source Index (backup)",     "figure_AM_anchor_map/outputs/figAM_source_index.pdf"),
]


MAIN_OUT = HERE / "outputs" / "ARSPI_Net_Defense_Main.pdf"
APPX_OUT = HERE / "outputs" / "ARSPI_Net_Defense_Appendix.pdf"
MANIFEST = HERE / "outputs" / "deck_manifest.json"


def _validate(entries: list[tuple[str, str, str]], section: str) -> list[Path]:
    """Resolve relative paths to absolute and abort if any are missing."""
    resolved = []
    missing = []
    for sid, _, rel in entries:
        p = (HERE / rel).resolve()
        if not p.is_file():
            missing.append((sid, rel))
        resolved.append(p)
    if missing:
        msg_lines = [f"ABORT: missing {section} PDFs:"]
        for sid, rel in missing:
            msg_lines.append(f"  [{sid}]  {rel}")
        msg_lines.append("")
        msg_lines.append(
            "build_deck.py does not regenerate figures. Run the appropriate"
        )
        msg_lines.append(
            "  make_*.py script to produce the missing PDF, then re-run this."
        )
        sys.stderr.write("\n".join(msg_lines) + "\n")
        sys.exit(1)
    return resolved


def _concat(entries: list[tuple[str, str, str]], paths: list[Path], out: Path,
            section: str) -> int:
    """Concatenate PDFs in order into `out`. Returns total page count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    deck = fitz.open()
    try:
        for (sid, label, _), src in zip(entries, paths):
            src_doc = fitz.open(src)
            n_before = deck.page_count
            deck.insert_pdf(src_doc)
            n_added = deck.page_count - n_before
            src_doc.close()
            print(f"  [{section}] +{n_added}p  {sid:<8}  {label}  ←  {src.relative_to(HERE.parent)}")
        deck.save(out, garbage=4, deflate=True)
        return deck.page_count
    finally:
        deck.close()


def _write_manifest(main_paths: list[Path], appx_paths: list[Path]) -> None:
    def _row(entries, paths):
        return [
            {"id": sid, "label": label,
             "path": str(p.relative_to(HERE.parent))}
            for (sid, label, _), p in zip(entries, paths)
        ]
    manifest = {
        "deck": "ARSPI_Net_Defense",
        "main": {
            "output": str(MAIN_OUT.relative_to(HERE.parent)),
            "slides": _row(MAIN_SLIDES, main_paths),
        },
        "appendix": {
            "output": str(APPX_OUT.relative_to(HERE.parent)),
            "slides": _row(APPENDIX_PDFS, appx_paths),
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  manifest → {MANIFEST.relative_to(HERE.parent)}")


def main() -> None:
    print("Validating main-deck PDFs...")
    main_paths = _validate(MAIN_SLIDES, "main-deck")
    print("Validating appendix PDFs...")
    appx_paths = _validate(APPENDIX_PDFS, "appendix")

    print(f"\nAssembling main deck ({len(MAIN_SLIDES)} slides) → "
          f"{MAIN_OUT.relative_to(HERE.parent)}")
    main_pages = _concat(MAIN_SLIDES, main_paths, MAIN_OUT, "MAIN")

    print(f"\nAssembling appendix ({len(APPENDIX_PDFS)} backup PDFs) → "
          f"{APPX_OUT.relative_to(HERE.parent)}")
    appx_pages = _concat(APPENDIX_PDFS, appx_paths, APPX_OUT, "APPX")

    _write_manifest(main_paths, appx_paths)

    print(f"\nDone.")
    print(f"  main deck:  {main_pages} pages  →  {MAIN_OUT.relative_to(HERE.parent)}")
    print(f"  appendix:   {appx_pages} pages  →  {APPX_OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
