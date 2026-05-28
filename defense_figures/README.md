# Defense Figures

Figures and supporting artifacts for the dissertation defense deck. Each
figure is a standalone subdirectory containing:

- A `make_*.py` script that produces the PDFs and any data dumps.
- A `companion_notes.md` carrying the talking-points and derivations.
- An `outputs/` directory with the PDFs, CSV/TSV audit trails, and any
  intermediate caches.

All scripts import from the shared `_style.py` module — the **single
source of truth** for fonts, palette, figure sizes, and PDF save
conventions.

> **Branch state.** All defense-figure work in this README lives on the
> branch `claude/magical-volta-82g4t`. As of this writing, `main` does
> **not** carry these artifacts. A committee-facing workflow must
> either merge this branch to `main` or be told explicitly which ref
> to inspect. See `HANDOFF_TO_NEXT_AGENT.md` for details.

> **PhD argument framing** (per author audit). The dissertation's
> argument is **not** "I pursued truth." It is:
>
> *"The dissertation repeatedly replaced default assumptions with
> measured operators: autonomous stability with driven Lyapunov
> contraction (Exp A), architectural fashion with FNN-measured
> embedding-dimension sufficiency relative to reservoir capacity
> (Exp B), nominal parameter choice with memory-capacity regime
> analysis (Exp C), and performance claims with spatial null testing
> (Exp D)."*
>
> Every figure must serve that argument. Slides that read as therapy,
> manifesto, or AI-generated self-justification do not.

## What is built

### Measurement-driven experiments (the technical core)

| Exp | Defensible claim (precisely worded) | Path |
|---|---|---|
| **A** | Autonomous spectral-radius criteria are insufficient unless paired with a measurement under input drive. ρ(W) = 0.2647549015; driven λ₁ measured on 3,165 trajectories. | `experiment_a_autonomous_vs_driven/` |
| **B** | The Kennel–Brown FNN estimate provides an empirical bound on the embedding dimension required to reconstruct measured ERP trajectories via delay coordinates. The measured m* sits well below reservoir post-PCA dim (64) and far below raw reservoir state (256). **Takens-motivated question; FNN-measured answer.** The figure does not claim Takens "guarantees" reconstruction. | `experiment_b_takens_dimension/` |
| **C** | β = 0.05 sits inside the measured memory-capacity plateau (MC ≈ 0.763 at β = 0.05) but is **not** at the measured peak (MC ≈ 0.835 at β ≈ 0.012). Operating-regime mismatch is real, documented, and defended by a biological time-constant argument. | `experiment_c_memory_capacity/` |
| **D** | **Stimulus-class** (negative / neutral / positive affect) classification survives per-trial channel-permutation null at 500 perms × 5 CV folds. **Not** a clinical-disorder validation — disorder labels were not in the session pickle. CLI flag swaps to disorder labels when CSV is provided. | `experiment_d_channel_permutation/` |

### Scaffold figures (philosophical frame)

| Figure | Status | Path |
|---|---|---|
| **K** — Opening Questions | built | `figure_K_opening_questions/` |
| **F** — Theorem Scaffold | built | `figure_F_theorem_scaffold/` |
| **J** — Five Named Contributions | built | `figure_J_contributions/` |
| **G** — Failure Gallery | **NOT BUILT** — see HANDOFF §3.1 | — |

### Round-2 figures (deepen the doctoral signal)

| Figure | Status | Path |
|---|---|---|
| **TB** — Theoretical Bounds (analytic vs measured) | built | `figure_TB_theoretical_bounds/` |
| **MR** — Methodological Refusals | built | `figure_MR_methodological_refusals/` |
| **IP** — Intellectual Provenance | built | `figure_IP_intellectual_provenance/` |
| **OQ** — Open Questions (closing) | built | `figure_OQ_open_questions/` |
| **RC** — Reservoir contraction animation (ESP made visceral) | built (MP4 + GIF + PDF) | `experiment_a_autonomous_vs_driven/make_animation.py` → `outputs/rawA_1f_contraction_animation.{mp4,gif,pdf}` |

### What is not yet built (priority order)

1. **Figure G — Failure Gallery** (highest priority — the core "embraced failed results" claim has no visual carrier).
2. **Acknowledgments / AI-assistance disclosure slide** (integrity protection — see HANDOFF §3.2).
3. **Dissertation anchor map** (figure → chapter/section/page).
4. **Assembled deck PDF** (`build_deck.py` to concatenate in slide order).
5. **Author-voiced companion notes** (currently skeletons with `AUTHOR WRITES HERE` markers).

See `HANDOFF_TO_NEXT_AGENT.md` for detail on each.

## Running a figure script

Each `make_*.py` is runnable from the repo root:

```bash
python defense_figures/experiment_a_autonomous_vs_driven/make_experiment_a_figures.py
python defense_figures/experiment_b_takens_dimension/make_experiment_b_figures.py
python defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py
python defense_figures/experiment_d_channel_permutation/make_experiment_d_figures.py
python defense_figures/figure_K_opening_questions/make_figure_K.py
# ... etc.
```

Default arguments expect `chapter6Experiments/results/ch6_exp1_full.pkl`
for experiments A and B. (Gitignored; provided separately to the
dissertation environment.)

Use `--help` on any script for options.

## Reproducibility

- **Reservoir seed:** 42 throughout (matches `run_chapter6_exp1_esp.py`).
- **Reservoir provenance:** all Experiment A figures use the **chapter-6
  Reservoir** (sparse Gaussian, σ=0.05, density 0.10, no spectral-radius
  rescaling). Same reservoir family that produced the driven-Lyapunov
  results in the dissertation text.
- **Data audit:** every analysis figure has an accompanying
  `*_data.csv` recording every plotted value plus a comment-header with
  source paths, seeds, and parameters.
- **Caches:** intermediate caches (`_benettin_cache.npz`,
  `_fnn_cache.npz`, `_mc_cache.npz`, `_perm_cache.npz`) are committed so
  re-rendering figures after wording fixes does not require
  recomputation.

## Calibration on claims

The deck claims **artifact existence** for the figures in the tables
above. It does **not** claim **argument closure** — that requires
Figure G, the assembled deck, the author-voiced companion notes, and an
end-to-end read-through. See `HANDOFF_TO_NEXT_AGENT.md`.
