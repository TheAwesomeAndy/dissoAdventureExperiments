# Defense Figures

Figures and supporting artifacts for the dissertation defense deck. Each
figure is a standalone subdirectory containing:

- A `make_*.py` script that produces the PDFs and any data dumps.
- A `companion_notes.md` (author-written) carrying the talking-points
  and derivations that accompany the figure but do NOT appear on the
  figure itself.
- An `outputs/` directory with the PDFs, CSV/TSV audit trails, and any
  intermediate caches.

All scripts import from the shared `_style.py` module — the **single
source of truth** for fonts, palette, figure sizes, and PDF save
conventions. Any figure that bypasses `_style.py` will fail the deck's
visual-consistency requirement.

## Structure

```
defense_figures/
├── _style.py                                          # shared style — hard constraint
├── experiment_a_autonomous_vs_driven/
│   ├── make_experiment_a_figures.py
│   ├── companion_notes.md                             # author writes
│   └── outputs/
│       ├── rawA_1a_weight_matrix.pdf
│       ├── rawA_1b_eigenspectrum.pdf
│       ├── rawA_1c_benettin_sample_trajectories.pdf
│       ├── rawA_1d_per_trial_lambda_scatter.pdf
│       ├── analysisA_1e_autonomous_vs_driven.pdf      # the slide
│       └── experiment_a_data.csv                      # 3,165-row audit trail
└── (figures K, F, G, J, and remaining experiments B/C/D/E to be added)
```

## Planned figures

Per `/root/.claude/plans/experiment-a-autonomous-foamy-bear.md` (the
master defense-deck plan):

- **K — Opening Questions** (philosophical frame on slide 1)
- **F — Theorem Scaffold** (theorems → architectural choices)
- **Experiment A — Autonomous Spectral Radius vs. Driven Lyapunov** ✓
- **Experiment B — Takens embedding dimension** (TBD)
- **Experiment C — Memory-capacity curve** (TBD)
- **Experiment D — Channel-permutation null on clinical findings** (TBD)
- **Experiment E — Independent replication on held-out subjects** (if data permits)
- **G — Failure Gallery** (audit, not recovery; includes required GAT/oversmoothing case)
- **J — Contributions** (five named contributions)

## Running a figure script

Each `make_*.py` is runnable from the repo root:

```bash
python defense_figures/experiment_a_autonomous_vs_driven/make_experiment_a_figures.py
```

Default arguments expect:

- `chapter6Experiments/results/ch6_exp1_full.pkl` for Experiment A.
  (Not committed — see root `.gitignore`. Provided separately to the
  dissertation environment.)

Use `--help` on any script for options.

## Reproducibility

- **Reservoir seed:** 42 throughout (matches `run_chapter6_exp1_esp.py`).
- **Reservoir provenance:** all Experiment A figures use the **chapter-6
  Reservoir** (sparse Gaussian, σ=0.05, density 0.10, no spectral-radius
  rescaling). This is the same reservoir family that produced the
  driven-Lyapunov results in the dissertation text. The chapter-3/4/5
  `LIFReservoir` (rescaled to ρ=0.9) is intentionally NOT used — the
  plan file documents the reason.
- **Data audit:** every analysis figure has an accompanying
  `*_data.csv` recording every plotted value plus a comment-header with
  source paths, seeds, and parameters.
