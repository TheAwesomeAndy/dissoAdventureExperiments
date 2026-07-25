# ARSPI-Net: An Interpretable Neuromorphic Framework for Clinical EEG Analysis

**Dissertation:** Lane, A. A. (2026). *Affective Reservoir-Spike Processing and Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic Framework for Clinical EEG Analysis.* PhD Dissertation, Department of Electrical and Computer Engineering, Stony Brook University.

ARSPI-Net is a neuromorphic EEG framework that transforms multichannel EEG into an explicit hierarchy of temporal, dynamical, and graph-organized representations, keeping the intermediate quantities available as named, independently inspectable variables. A fixed-weight leaky integrate-and-fire (LIF) reservoir converts continuous EEG into spike trains; a six-bin spike-count code (BSC₆) discretizes the response; a fixed data-independent projection compresses it; and electrode-resolved embeddings, dynamical descriptors, and structure–function coupling summarize the spatial organization. Membrane trajectories, spike rasters, temporal codes, graph measures, and cross-scale coupling remain available as named intermediate quantities throughout the analysis.

The framework's primary contribution is an experimentally characterized measurement chain — every stage exposes an inspectable quantity — rather than a state-of-the-art accuracy result. On the SHAPE Community dataset, raw ERP summaries classify affective condition more accurately than the reservoir representation, so ARSPI-Net is **not** presented as an accuracy leader on this dataset. What it provides is a decomposable representation whose intermediate variables are open to inspection, together with mechanistic controls that identify which structure the representation actually uses.

### Scope and what this repository does not claim

This README states results at the same scope as the corrected dissertation, no stronger:

- The four-level interpretability taxonomy is an **audit framework proposed within this dissertation**, not an externally standardized or community-adopted taxonomy. Comparisons to NeuCube, ProtoPMed-EEG, and EEGNet+SHAP are **conceptual**, based on the specific analyses reviewed here, not a common-data benchmark. No priority claim (e.g. "first," "only," "no other architecture") is made or implied.
- The observation that different clinical labels attain their largest measured effects in different layers is a **cohort-level, exploratory, hypothesis-generating** pattern. It does **not** establish diagnostic subtypes, patient-level clinical interpretation, or validated biomarkers. Clinical readouts are near chance and were not corrected for multiple comparisons.
- The headline classification numbers are the **corrected confirmatory** estimates: five repeats of five-fold participant-grouped cross-validation, fold-local preprocessing, and a data-independent SRP-64 projection. Older PCA-64, subject-centered, and 10-fold results are retained only as **archived/descriptive legacy estimates** and are clearly labeled as such — they are not prospective performance and are not used for the primary inference.
- The generalization of every result is bounded to this cohort and requires external validation.

---

## Publications

1. **Lane, A., Nelson, B. D., & Tang, K. W.** (2023). "Towards ARSPI-Net: Development of an Efficient Hybrid Deep Learning Framework." *Proc. IEEE Long Island Systems, Applications and Technology Conference (LISAT)*. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10179592)

2. **Lane, A., Nelson, B. D., & Tang, K. W.** (2024). "Towards ARSPI-NET: Advancing EEG Feature Extraction with Neuromorphic Algorithms." *Proc. IEEE Long Island Systems, Applications and Technology Conference (LISAT)*.

3. **Lane, A. A., Tang, K. W., & Nelson, B. D.** (2026). "Subject-Covariance Entanglement in Affective EEG Embeddings." *IEEE Signal Processing Letters* — **submitted, under review**.

---

## Dissertation Source

The corrected LaTeX source for the written dissertation lives in this repo under `dissertation/` (`main.tex`, chapter files, `Appendixa.tex`, and the bibliography `Disso.bib`). The language-audit record is `dissertation/AUDIT_REPORT.md`. An earlier packaged source (`ArspiNetFinalVer_.zip`) is retained at the root for provenance; where the two differ, **the `dissertation/` source is authoritative**. Each experimental chapter has a corresponding folder of code and/or figures in this repository — the mapping is given below.

---

## Dissertation Map

The dissertation has eight chapters plus Appendix A. Theoretical chapters (1, 2, 3, 8) live primarily in the LaTeX source and the `pictures/` figure tree; experimental chapters (4, 5, 6, 7) each have a dedicated code folder.

| # | Chapter | .tex source | Code / experiments | Figures |
|---|---|---|---|---|
| 1 | Introduction | `dissertation/ch1.tex` | — (theoretical) | `pictures/ch1/` |
| 2 | Background & Related Work | `dissertation/ch2.tex` | — (theoretical) | `pictures/ch2/` |
| 3 | LIF Spiking Reservoir Theory & Characterization | `dissertation/chLSM.tex` | `experiments/chapter3/run_chapter3_lsm_characterization.py` | `pictures/chLSM/` |
| 4 | Spike-to-Embedding Pipeline (BSC₆ + compression) | `dissertation/chLSMEmbeddings.tex` | `chapter4Experiments/` | `pictures/chLSMEmbeddings/` |
| 5 | ARSPI-Net Graph Classification (GNN + baselines) | `dissertation/chgraph.tex` | `chapter5Experiments/` + `experiments/ch5_4class/` | `pictures/chGraphNeuralNetworks/` |
| 6 | Dynamical Characterization (7 descriptors, Lyapunov, PE) | `dissertation/chDynamics.tex` | `chapter6Experiments/` + `experiments/ch6_ch7_3class/` | `pictures/chDynamics/` |
| 7 | Structure-Function Coupling (κ) — Synthesis | `dissertation/chSynthesis.tex` | `chapter7Experiments/` + `experiments/ch6_ch7_3class/` + `experiments/ablation/` | `pictures/chSynthesis/` |
| 8 | Conclusion & Clinical Translation | `dissertation/chConclusion.tex` | — | `pictures/conclusion/fig_defense_summary.pdf` |
| A | Appendix A — Supplementary Analyses | `dissertation/Appendixa.tex` | (figures regenerated by the chapter scripts above) | `pictures/appendixA/` |

Interpretability validation scripts that span chapters 4–7 live under `experiments/interpretability/`. Data-quality validators live under `validation/`. Every reported number is indexed, with its provenance, in the machine-readable manifest [`results_manifest.json`](results_manifest.json).

---

## Confirmatory Results (headline)

**These are the corrected estimates the dissertation reports as its primary inference.**

**Protocol.** Five repeats of five-fold **participant-grouped** cross-validation on the SHAPE Community dataset. Every participant's complete condition panel stays in a single fold. Per-feature standardization and the linear logistic readout are **fitted on training participants only (fold-local)**. BSC₆ is compressed from 1,536 to 64 coordinates with a **fixed sparse random projection (SRP-64)** generated from a prespecified seed; because it does not estimate a basis from the observations, held-out participants cannot influence it. Reported brackets are participant-bootstrap 95% intervals over the repeated out-of-fold predictions. A separate deterministic ridge readout is used for 100 within-participant label permutations. Dataset: [https://lab-can.com/shape/](https://lab-can.com/shape/)

**Samples.** 211 participants / 633 observations for three-condition discrimination (Negative, Neutral, Pleasant); 210 participants / 840 observations for four-category discrimination (Threat, Mutilation, Cute, Erotic).

### Corrected participant-grouped validation

| Representation | Three conditions | Four categories |
|---|---|---|
| Raw ERP, 16 temporal bins | 68.1% [65.1, 71.1] | 60.5% [57.9, 63.0] |
| Conventional spectral / Hjorth | 49.4% [46.8, 51.9] | 36.7% [34.6, 38.9] |
| **BSC₆ / SRP-64** | **60.6% [57.8, 63.4]** | **53.2% [50.6, 55.8]** |
| BSC₆ / SRP-64 + conventional (fusion) | 62.0% [59.1, 64.8] | 53.5% [50.8, 56.3] |
| Chance | 33.3% | 25.0% |

- **BSC₆/SRP-64 exceeds the conventional band-power/Hjorth baseline by +11.3 pp** (bootstrap [7.9, 14.7]) at three conditions and **+16.5 pp** ([13.3, 19.8]) at four categories. Participant-label permutation gives **p = 0.0099** at both granularities.
- **Fusion** with conventional features adds 1.3 pp at three conditions ([0.0, 2.6]) and no stable gain at four categories.
- **Raw ERP-16 outperforms the reservoir** at both granularities (68.1% and 60.5%). This is an intentional boundary result: ARSPI-Net is not an accuracy leader on this dataset; its value is the inspectable intermediate representation, not peak accuracy.

### Mechanistic (destruction) controls

These test *what the representation uses*, holding the protocol fixed.

| Control | Three conditions | Four categories |
|---|---|---|
| Shuffle temporal-bin order (within observation × electrode) | −7.4 pp [3.7, 11.1] | −12.9 pp [9.9, 16.0] |
| Collapse time | −4.9 pp | −8.6 pp |
| Shuffle electrode identity | −13.9 pp | −20.1 pp |

Both ordered temporal evolution and stable electrode identity contribute to held-out classification; the result does not require an anatomical interpretation of unlabeled channels. Across reservoir seeds 7, 42, and 99, three-condition accuracy ranges from 60.5% to 62.1%; the full-epoch six-bin window (39–977 ms) reaches 63.3% in the three-seed robustness experiment.

---

## Interpretability Audit Framework (internally proposed)

The four levels below are the **operational audit levels defined for this dissertation**. They are not an external standard, and the accompanying comparison table summarizes the specific analyses reviewed here rather than every possible implementation of each architecture. Every quantity is directly inspectable; **its patient-level reliability and clinical actionability are not established by this dissertation.**

| Level | Property | Key Metric (as reported) | Chapter |
|---|---|---|---|
| 1 | Temporal traceability | pre-reservoir r = 0.82 (LPP), post-reservoir \|r\| = 0.23 | Ch4 |
| 2 | Geometric transparency | ρ = 7.2 subject-to-condition variance ratio | Ch5 |
| 3 | Dynamical characterization | amplitude-domain \|r\| = 0.82, spike-domain \|r\| = 0.23 (LPP) | Ch6 |
| 4 | Systems-level correspondence | κ (two estimands, see below), p < 0.001 | Ch7 |

The **Level 4 coupling** value has two distinct estimands, which the manifest records as separate entries rather than one blended number: a **per-observation median κ = 0.273** across the 844 four-category observations (`exp_4class_per_observation_median_kappa`), and a **subject-level median κ ≈ 0.22** on the corrected three-class stream over 211 subject-averaged values (`exp_3class_group_coupling_kappa`). Both use the normalized Frobenius statistic κ = ‖C‖_F/√(p·q) against a 5,000-permutation electrode-permutation null, and both are significant (p < 0.001); they differ in sample unit (per-observation vs subject-level) and granularity (4-category vs 3-class), not in a discrepancy.

The post-reservoir ERP correspondence is **weak** (\|r\| = 0.23); the strong \|r\| = 0.82 correspondence is an amplitude-domain analog on the centered ERP within the reservoir feature window, i.e. an input-side reference bound, not a post-reservoir property. Both facts are stated together so the correspondence is not overstated.

---

## Exploratory & Archived Results (not the headline)

The dissertation separates its corrected confirmatory findings from a set of **hypothesis-generating archived analyses**. These are retained for transparency and to define prespecified replication targets. They use the archived input pickle, globally fitted PCA, and — in the centered columns — the complete held-out participant panel, so they are **descriptive legacy estimates, not prospective performance**, and they are **not** used for the primary inference.

### Archived baseline / centered table (descriptive only)

The centered columns below use **transductive subject-mean centering** (the subject mean is computed using held-out data), which inflates apparent accuracy relative to a prospective setting. They are kept to motivate research on subject nuisance, not as performance claims.

| Method | Input | Uncentered | Centered (transductive) | Δ |
|---|---|---|---|---|
| Band Power + SVM | Spectral | 47.7% ± 5.1% | 61.0% ± 6.0% | +13.3 |
| Raw EEG PCA-200 + LogReg | Raw EEG | 64.9% ± 5.0% | 86.4% ± 4.1% | +21.5 |
| Raw EEG (full) + LogReg | Raw EEG | 70.5% ± 3.0% | 88.4% ± 2.1% | +17.9 |
| LSTM (bidir, 64h) | Raw EEG | 58.0% ± 5.5% | 71.1% ± 6.8% | +13.1 |
| GRU (bidir, 64h) | Raw EEG | 59.9% ± 6.4% | 78.4% ± 3.5% | +18.5 |
| EEGNet (Lawhern 2018) | Raw EEG | 72.0% ± 4.9% | 89.1% ± 3.1% | +17.1 |
| ARSPI-Net reservoir + LogReg | BSC₆/PCA | 59.4% ± 3.6% | 78.8% ± 3.4% | +19.4 |

Variance decomposition (archived): 62.6% subject, 8.7% condition, 28.7% residual (ρ = 7.2×). Per-subject centering increases accuracy by +13 to +21 pp across all seven tested representations. **The centering estimand is transductive and not prospective**; generalization beyond this cohort requires external validation.

### Archived message-passing GNN sweep (descriptive only)

PCA-64 concatenation + SVM reaches 63.4% ± 4.4% (MCC 0.457); GIN/GCN/EdgeConv five-seed ensembles reach 49.8% / 56.4% / 48.5%. The displayed non-propagated value is higher, but global preprocessing and dependent cross-validation folds mean this sweep **cannot** establish a universal graph operating boundary.

### Archived 4-class exploration and clinical screens

The original 4-class experiments (`chapter6Experiments/`, `chapter7Experiments/`) and the layer-specific clinical screens are exploratory. They yield nominal candidates but **no validated biomarkers**; near-chance clinical readouts and global preprocessing preclude any claim of disorder detection or superiority to single-stage classifiers.

Full provenance for every number above — confirmatory, exploratory, and archived — is in [`results_manifest.json`](results_manifest.json) and [`docs/REPRODUCTION_MAP.md`](docs/REPRODUCTION_MAP.md).

---

## Repository Structure

```
dissoAdventureExperiments/
├── README.md                                # This file (dissertation-wide overview)
├── results_manifest.json                    # Machine-readable provenance for every reported number
├── CITATION.cff                             # Citation metadata for the frozen publication release
├── RELEASE.md                              # Frozen-release + Zenodo DOI procedure
├── dissertation/                            # Corrected LaTeX source (authoritative) + AUDIT_REPORT.md
├── ArspiNetFinalVer_.zip                    # Earlier packaged source (provenance only)
│
├── chapter4Experiments/                     # Ch4: Spike-to-embedding pipeline (synthetic + EEG)
├── chapter5Experiments/                     # Ch5: Classification + baselines (archived/legacy stream)
├── chapter6Experiments/                     # Ch6: 4-class dynamical exploration (archived)
├── chapter7Experiments/                     # Ch7: 4-class coupling exploration (archived)
│
├── experiments/                             # Extended experiments + theoretical Ch3 + interpretability
│   ├── chapter3/                            #   Ch3: Controlled LIF reservoir characterization (synthetic)
│   ├── ch5_4class/                          #   4-class classification extension (exploratory)
│   ├── ch6_ch7_3class/                      #   3-class dynamical + coupling pipeline
│   ├── ablation/                            #   Layer ablation experiment (A0–A9, C1–C6) — archived
│   └── interpretability/                    #   Four-level interpretability validation scripts
│
├── validation/                              # Data quality control + dissertation cross-checker
├── data/clinical_profile.csv                # 211-subject clinical profiles (gitignored, private)
├── docs/
│   ├── methodology_rules.md                 #   12 governing methodology rules
│   ├── VERIFICATION_METHODOLOGY.md          #   What the tests do and do not establish
│   └── REPRODUCTION_MAP.md                  #   Script → dissertation table/figure mapping (tiered)
└── pictures/                                # Figures, organized by dissertation chapter
```

> **Confirmatory pipeline.** The corrected confirmatory numbers (SRP-64, five repeats of five-fold participant-grouped validation, fold-local preprocessing) are regenerated by the committed runner **`experiments/confirmatory/run_confirmatory_validation.py`** (library `confirmatory_pipeline.py`), which reuses the validated LIF reservoir and BSC₆ encoder and replaces the archived globally fitted PCA-64 with a fixed, data-independent `SparseRandomProjection`. Because the SHAPE dataset is restricted and not shipped here, the exact headline values are produced by feeding the authorized data to the runner; the *machinery* is verified without the dataset by `experiments/confirmatory/verify_confirmatory.py` (18 checks) and `check_against_manifest.py` confirms produced numbers match `results_manifest.json`. The archived scripts under `chapter5Experiments/` reproduce the descriptive legacy estimates, not the confirmatory ones.

---

## Reproduction Map

See [`docs/REPRODUCTION_MAP.md`](docs/REPRODUCTION_MAP.md), reorganized into three tiers:

- **Confirmatory** — corrected participant-grouped, fold-local, SRP-64 estimates that back the primary inference.
- **Exploratory** — 3-class dynamical/coupling and 4-class extensions that generate hypotheses.
- **Archived** — PCA-64, transductive-centered, GNN-sweep, and original 4-class pipelines retained for provenance and replication targets only.

---

## Architecture Overview

The ARSPI-Net pipeline implements a Koopman-theoretic signal processing chain:

1. **LIF Spiking Reservoir** (Ch3) — Fixed-weight LIF reservoir converts continuous EEG into spike trains. The separation property (Maass 2002) is derived analytically in Chapter 3; computational validation (reservoir size ablation, coding comparison, cross-seed robustness, parameter sensitivity) is in `chapter4Experiments/`.
2. **BSC₆ Temporal Coding** (Ch4) — Binned Spike Count with 6 temporal bins discretizes the continuous spike response into a structured, independently analyzable temporal representation.
3. **Fixed data-independent compression** (Ch5) — For the confirmatory stream, BSC₆ is compressed 1,536→64 with a fixed sparse random projection (SRP-64) that is generated from a seed and never fitted to observations. (The archived stream instead used a globally fitted PCA-64, which is why those estimates are descriptive rather than prospective.)
4. **Graph-Organized Spatial Analysis** (Ch5, Ch7) — Electrode-level features organized by channel topology support graph metrics and structure–function coupling (κ).

### Interpretability Signal Chain

Every output variable traces to input through: membrane voltage → spike times → BSC₆ bins → projected components → named descriptors → graph topology → coupling κ. No learned recurrent weights. This is intrinsic decomposability of the representation, not a post-hoc explanation of a black box — and it is a property of the *machinery*, not a claim about clinical validity.

---

## Dependencies

```
numpy>=2.0           # Code uses np.trapezoid (numpy 2.0 API); saved pickles use the 2.0 format
scipy>=1.13          # For numpy 2.0 ABI compatibility
scikit-learn>=1.5    # For numpy 2.0 ABI compatibility (SparseRandomProjection, PCA, SVM, LogReg)
pandas>=2.2.3        # For numpy 2.0 ABI compatibility (used by reproduce_chapter5.py)
matplotlib>=3.9
torch>=1.12          # For canonical_pytorch_baselines.py only
```

**Windows note:** verifier scripts use Unicode box-drawing characters in stdout and read source files as UTF-8. If your console defaults to cp1252, run with `PYTHONUTF8=1` (Python 3.7+) or `set PYTHONIOENCODING=utf-8`.

## Input Data Format

The SHAPE Community dataset is available at [https://lab-can.com/shape/](https://lab-can.com/shape/). Raw EEG data (not included in this repository) should be placed in `batch_data/` (3-class) or `categories/` (4-class) following the structure described in the validation scripts.

**3-class data** (`batch_data/`): Baseline-corrected text files, one per subject per condition.
- Format: `SHAPE_Community_{SUBJECT_ID}_IAPS{Neg,Neu,Pos}_BC.txt`
- Dimensions: 1229 rows (timepoints at 1024 Hz) × 34 columns (EEG channels)
- Units: microvolts, loadable with `np.loadtxt(filepath)`

**4-class data** (`categories/`): `.mat` files organized by affective subcategory (Threat / Mutilation / Cute / Erotic), loadable with `scipy.io.loadmat()`.

---

## Verification Summary

| Suite | Script | Tests | Status |
|---|---|---|---|
| Ch4 Core | `verify_chapter4.py` | 31 | PASS |
| Ch5 Core | `verify_chapter5.py` | 32 | PASS |
| Ch5 Exp Zero | `verify_experiment_zero.py` | 37 | PASS |
| Ch5 Reproduce | `verify_reproduce_chapter5.py` | 33 | PASS |
| Ch5 Baselines | `verify_baselines.py` | 55 | PASS |
| Ch6 Core | `verify_chapter6.py` | 31 | PASS |
| Ch6 Reproduce | `verify_reproduce_chapter6.py` | 42 | PASS |
| Ch7 Experiments | `verify_chapter7.py` | 38 | PASS |
| Ch7 Utilities | `verify_extract_utilities.py` | 40 | PASS |
| Validation | `verify_validators.py` | 20 | PASS |
| Ch5 4-class | `verify_ch5_4class.py` | 25 | PASS |
| Ch6/7 3-class | `verify_ch6_ch7_3class.py` | 28 | PASS |
| Ablation | `verify_ablation.py` | 23 | PASS |
| Confirmatory pipeline | `experiments/confirmatory/verify_confirmatory.py` | 18 | PASS |
| **Total** | | **453** | **453/453 PASS** |

> **What these tests establish — and what they do not.** The 453 passing tests verify **code behavior**: that scripts run, that named functions compute what their names say, that components integrate, and that fixed seeds produce deterministic output. **They do not establish scientific validity.** Passing tests say nothing about whether the statistical estimand is the right one, whether the preprocessing boundary is leak-free, or whether a generalization claim holds on unseen participants. Those properties depend on the study design and the data, and are argued — at explicitly bounded scope — in the dissertation itself, not proven by the test suite. Treat a green test run as evidence the machinery is correct, not as evidence a finding is true.

Methodology: [docs/VERIFICATION_METHODOLOGY.md](docs/VERIFICATION_METHODOLOGY.md)
Reproduction map: [docs/REPRODUCTION_MAP.md](docs/REPRODUCTION_MAP.md)
Results manifest: [results_manifest.json](results_manifest.json)
