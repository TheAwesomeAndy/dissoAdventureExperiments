# ARSPI-Net Repository Update Guide
## What to add to bring the repo up to date (March 20, 2026)

---

## Current Repo Contents (what you already have)

```
├── main_final.tex
├── ch1_final.tex
├── ch2_final.tex
├── chLSM.tex                        # Ch3 (revised)
├── chLSM_final.tex                  # Ch3 (older version)
├── chLSMEmbeddings_final.tex        # Ch4
├── chgraph_final.tex                # Ch5 (OUTDATED — 377 lines, 3-class only)
├── chDynamics_final.tex             # Ch6 (theoretical spec only, no results)
├── chSynthesis_final.tex            # Ch7 (OUTDATED — 147 lines, early spec)
├── Disso.bib
├── Dissertationv2.pdf
├── run_chapter4_experiments.py       # Ch4 script
├── *.pdf                            # Ch4 figures
└── ARSPI_Net_Architecture.pdf
```

---

## New Directory Structure to Create

```
dissoAdventureExperiments/
│
├── README.md                         # UPDATE with current project status
│
├── latex/                            # All chapter LaTeX files
│   ├── main_final.tex
│   ├── ch1_final.tex
│   ├── ch2_final.tex
│   ├── chLSM.tex                     # Ch3 (keep current)
│   ├── chLSMEmbeddings_final.tex     # Ch4 (keep current)
│   ├── chgraph_final.tex             # Ch5 — REPLACE with 680-line version
│   ├── chDynamics_final.tex          # Ch6 (keep spec; results in experiments/)
│   ├── chSynthesis_complete.tex      # Ch7 — REPLACE with 418-line version
│   └── Disso.bib                     # UPDATE with new entries
│
├── data/
│   └── clinical_profile.csv          # NEW — corrected 211-subject clinical data
│
├── experiments/
│   ├── ch4/
│   │   └── run_chapter4_experiments.py
│   │
│   ├── ch5_3class/                   # Existing 3-class experiments
│   │   ├── README.md                 # (document what exists)
│   │   └── (scripts from prior sessions if you have them locally)
│   │
│   ├── ch5_4class/                   # NEW — 4-class experimental program
│   │   ├── ch5_4class_01_feature_extraction.py
│   │   ├── ch5_4class_02_raw_observations.py
│   │   └── ch5_4class_03_classification_full.py
│   │
│   ├── ch6_ch7_3class/               # NEW — 3-class dynamical + coupling
│   │   ├── ch6_ch7_3class_README.md
│   │   ├── ch6_ch7_01_feature_extraction.py
│   │   ├── ch6_ch7_02_raw_observations.py
│   │   ├── ch6_03_experiments.py
│   │   └── ch7_04_experiments.py
│   │
│   └── ablation/                     # NEW — keystone experiment
│       └── layer_ablation.py
│
├── results/                          # Terminal outputs and result pickles
│   ├── ch5_4class/
│   │   ├── script01_output.txt       # (from your local runs)
│   │   ├── script02_output.txt
│   │   └── script03_output.txt
│   ├── ch6_ch7_3class/
│   │   ├── script01_output.txt
│   │   ├── script02_output.txt
│   │   ├── script03_ch6_output.txt
│   │   └── script04_ch7_output.txt
│   └── ablation/
│       └── ablation_output.txt
│
├── figures/                          # All generated figures
│   ├── ch5_4class_obs/               # 7 raw observation PDFs
│   ├── ch5_4class_figures/           # 9 experiment PDFs
│   ├── ch6_ch7_obs/                  # 8 raw observation PDFs
│   ├── ch6_figures/                  # 7 experiment PDFs
│   ├── ch7_figures/                  # 5 experiment PDFs
│   └── ablation_figures/             # 3 ablation PDFs
│
└── docs/
    ├── ch5_4class_output_specification.md
    ├── ch6_ch7_3class_README.md
    └── methodology_rules.md           # NEW — the 7 methodology rules
```

---

## Files to ADD (new since repo was last updated)

### Priority 1: Core Scripts (the experimental pipeline)

These are the scripts that produce all new results. Add them first.

| File | Source | Destination in repo |
|------|--------|---------------------|
| ch5_4class_01_feature_extraction.py | /mnt/user-data/outputs/ | experiments/ch5_4class/ |
| ch5_4class_02_raw_observations.py | /mnt/user-data/outputs/ | experiments/ch5_4class/ |
| ch5_4class_03_classification_full.py | /mnt/user-data/outputs/ | experiments/ch5_4class/ |
| ch6_ch7_01_feature_extraction.py | /mnt/user-data/outputs/ | experiments/ch6_ch7_3class/ |
| ch6_ch7_02_raw_observations.py | /mnt/user-data/outputs/ | experiments/ch6_ch7_3class/ |
| ch6_03_experiments.py | /mnt/user-data/outputs/ | experiments/ch6_ch7_3class/ |
| ch7_04_experiments.py | /mnt/user-data/outputs/ | experiments/ch6_ch7_3class/ |
| layer_ablation.py | /mnt/user-data/outputs/ | experiments/ablation/ |

### Priority 2: Updated LaTeX Chapters

| File | Source | Destination | Notes |
|------|--------|-------------|-------|
| chgraph_final.tex | /mnt/user-data/outputs/ | latex/ | REPLACE — 680 lines, includes 3-class results, clinical biomarkers, regime analysis |
| chSynthesis_complete.tex | /mnt/user-data/outputs/ | latex/ | REPLACE — 418 lines, includes 4-class coupling results |

### Priority 3: Clinical Data

| File | Source | Destination | Notes |
|------|--------|-------------|-------|
| clinical_profile.csv | /mnt/user-data/outputs/ | data/ | CRITICAL — corrected 211-subject file, replaces the old 116-subject version |

### Priority 4: Documentation

| File | Source | Destination |
|------|--------|-------------|
| ch6_ch7_3class_README.md | /mnt/user-data/outputs/ | docs/ or experiments/ch6_ch7_3class/ |
| ch5_4class_output_specification.md | /mnt/user-data/outputs/ | docs/ |

### Priority 5: Result Pickles (if you want reproducibility)

These are large files. Consider adding them to .gitignore and storing
separately, or use Git LFS.

| File | Size | Contents |
|------|------|----------|
| shape_features_211.pkl | ~350 MB | 3-class BSC6/PCA features (211 subj) |
| shape_features_4class.pkl | ~464 MB | 4-class BSC6/PCA features (211 subj) |
| ch6_ch7_3class_features.pkl | ~25 MB | Dynamical + topological features |
| ch5_4class_results.pkl | ~51 KB | 4-class classification results |
| ch6_results.pkl | small | Ch6 experiment results |
| ch7_3class_results.pkl | small | Ch7 experiment results |
| layer_ablation_results.pkl | small | Ablation matrix results |

Recommendation: Add the small pickles (<1 MB) directly. Add
the large pickles (>10 MB) to .gitignore with a note that they
are regenerated by running Scripts 01.

---

## Files to REPLACE (outdated versions in repo)

| Current in repo | Replace with | Why |
|-----------------|-------------|-----|
| chgraph_final.tex (377 lines) | /mnt/user-data/outputs/chgraph_final.tex (680 lines) | Includes full 3-class results, clinical biomarkers, SUD finding, regime analysis |
| chSynthesis_final.tex (147 lines) | /mnt/user-data/outputs/chSynthesis_complete.tex (418 lines) | Includes 4-class coupling results, 5 experiments with data |
| (no clinical data) | clinical_profile.csv | Old CSV had 116 subjects; new one has 211 |

---

## Files to KEEP (unchanged)

| File | Why |
|------|-----|
| ch1_final.tex | Chapter 1 is complete |
| ch2_final.tex | Chapter 2 is complete |
| chLSM.tex | Chapter 3 is complete (revised version) |
| chLSMEmbeddings_final.tex | Chapter 4 is complete |
| chDynamics_final.tex | Chapter 6 spec — keep as theoretical framework; results are in experiments/ |
| run_chapter4_experiments.py | Chapter 4 validated script |
| Disso.bib | Keep and UPDATE with new entries |
| main_final.tex | Keep |
| Ch4 figure PDFs | Keep |

---

## .gitignore Additions

Add these to prevent large binary files from bloating the repo:

```
# Large feature pickles (regenerate by running Script 01)
shape_features_211.pkl
shape_features_4class.pkl
ch6_ch7_3class_features.pkl

# EEG data (not stored in repo)
batch_data/
categories/
CategoryFiles/
shape_data/
*.zip

# Python cache
__pycache__/
*.pyc
```

---

## NEW FILE: methodology_rules.md

Create this file in docs/ to document the 7 methodology rules
that govern all future work on this project:

```markdown
# ARSPI-Net Methodology Rules

Established March 2026 from advisor feedback.
These rules govern all experimental design and writing.

## Rule 1 — Horizontal Before Vertical
When a cross-chapter claim emerges, immediately design the single
experiment that tests it directly using features from all chapters.

## Rule 2 — Claims Require Direct Tests
Every dissociation or complementarity claim must have a direct
ablation/combination experiment. Narrative consistency is not proof.

## Rule 3 — No Unearned Terminology
Do not use a mathematical framework's vocabulary unless its quantities
have been computed. "Operationally distinct" not "information-theoretic
decomposition."

## Rule 4 — Adversarial Committee Test
Before finalizing any program, list the 3 most likely committee
objections and verify each has an experimental answer.

## Rule 5 — Linear Readouts for Content Comparison
Use linear classifiers as primary when comparing feature families.
Nonlinear classifiers in appendix sensitivity checks only.

## Rule 6 — Interrogate Null Results
Test at finer resolution before concluding absence of effect.

## Rule 7 — Build Summary Tables Immediately
When a comparison is central to the argument, build the table as soon
as the numbers exist.
```

---

## Suggested Git Commit Sequence

Do this in order so the commit history is clean:

```bash
# 1. Create directory structure
mkdir -p experiments/ch5_4class experiments/ch6_ch7_3class experiments/ablation
mkdir -p results/ch5_4class results/ch6_ch7_3class results/ablation
mkdir -p figures docs data

# 2. Add clinical data
cp clinical_profile.csv data/
git add data/clinical_profile.csv
git commit -m "Add corrected 211-subject clinical profile"

# 3. Add Chapter 5 4-class scripts
cp ch5_4class_*.py experiments/ch5_4class/
git add experiments/ch5_4class/
git commit -m "Add Ch5 4-class experimental program (3 scripts, 18 analyses)"

# 4. Add Chapter 6/7 3-class scripts
cp ch6_ch7_*.py ch6_03_experiments.py ch7_04_experiments.py experiments/ch6_ch7_3class/
git add experiments/ch6_ch7_3class/
git commit -m "Add Ch6/7 3-class experimental program (4 scripts, 12 analyses)"

# 5. Add layer ablation (keystone experiment)
cp layer_ablation.py experiments/ablation/
git add experiments/ablation/
git commit -m "Add layer ablation matrix (A1-A9 emotion + C1-C6 clinical)"

# 6. Update LaTeX chapters
cp chgraph_final.tex latex/          # the 680-line version
cp chSynthesis_complete.tex latex/
git add latex/chgraph_final.tex latex/chSynthesis_complete.tex
git commit -m "Update Ch5 (680 lines, full results) and Ch7 (418 lines, coupling)"

# 7. Add documentation
cp ch6_ch7_3class_README.md docs/
# Create methodology_rules.md manually from the template above
git add docs/
git commit -m "Add pipeline documentation and methodology rules"

# 8. Update .gitignore
echo "shape_features*.pkl" >> .gitignore
echo "ch6_ch7_3class_features.pkl" >> .gitignore
echo "batch_data/" >> .gitignore
echo "categories/" >> .gitignore
echo "__pycache__/" >> .gitignore
git add .gitignore
git commit -m "Update .gitignore for large data files"
```

---

## What Is Still Missing (TODO after repo update)

These items are identified but not yet built:

1. **EEGNet + GRU/LSTM baselines** — must-do for committee positioning
2. **Chapter 8 (Conclusion)** — not yet written
3. **Revised Chapter 5 LaTeX** — incorporating 4-class results into
   the existing 3-class chapter as paired-granularity design
4. **Revised Chapter 6 LaTeX** — converting spec into results chapter
   using Script 03 data
5. **Revised Chapter 7 LaTeX** — integrating 3-class results alongside
   existing 4-class text
6. **Bibliography update** — ch7_new_bib_entries.bib needs to be
   appended to Disso.bib
7. **Cross-chapter reference audit** — ensure all \ref{} and \cite{}
   resolve correctly
