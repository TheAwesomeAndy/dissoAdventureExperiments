# GitHub Repository Update Guide
## What Is Out of Date and What Needs to Be Updated

**Date:** April 2026
**Repository:** https://github.com/TheAwesomeAndy/dissoAdventureExperiments

---

## Current State of the Repo

The README is up to date with:
- ✅ Correct dissertation title
- ✅ Four-level taxonomy table with key metrics
- ✅ Complete 7-row centered baseline table
- ✅ Key experimental results
- ✅ SPL paper listed (under review)
- ✅ 435/435 verification tests passing

## What Is MISSING

Four dissertation claims in the abstract and Chapter 5 have **no code backing them** in the repository. These are the highest-visibility interpretability results:

### Gap 1: Level 1 Validation — BSC₆ Bin-to-ERP Correlation
- **Dissertation claim:** "BSC₆ bins recover LPP amplitude with r = 0.82" (Abstract, Ch5)
- **What's missing:** No script computes Pearson correlation between BSC₆-aligned temporal bin activations and classical ERP scalars (P300, LPP)
- **Also missing:** R² = 0.661 LPP prediction via Ridge regression, temporal resolution sweep (6/12/24 bins)
- **File to add:** `experiments/interpretability/run_level1_temporal_traceability.py`

### Gap 2: Attention-Prototype Readout Architecture
- **Dissertation claim:** "66.7% balanced accuracy with per-prediction temporal and spatial attribution" (Ch5)
- **What's missing:** No PyTorch implementation of the LIF + temporal attention + channel attention + prototype readout
- **Also missing:** Permutation test (p = 0.634), confusion matrix, learned threshold θ ≈ 0.32
- **File to add:** `experiments/interpretability/run_arspinet_v2_attention_prototype.py`

### Gap 3: EEGNet Saliency vs ARSPI-Net Attention Comparison
- **Dissertation claim:** "EEGNet's gradient saliency peaks at 402–691 ms while ARSPI-Net's attention peaks at 176–254 ms" (Ch5, Abstract)
- **What's missing:** `eegnet_gru_lstm_baselines.py` trains EEGNet but has NO gradient saliency extraction or attention comparison
- **File to add:** `experiments/interpretability/run_eegnet_saliency_comparison.py`

### Gap 4: Chapter 3 Synthetic Characterization
- **Dissertation claim:** Controlled LIF reservoir characterization (β sweep, θ sweep, ρ sweep, temporal FDR, readout comparison) (Ch3)
- **What's missing:** Entire Chapter 3 has zero code in the repo — no synthetic experiments whatsoever
- **File to add:** `experiments/chapter3/run_chapter3_lsm_characterization.py`

---

## What Needs to Change in the README

The repo structure section needs two new entries under `experiments/`:

```
├── experiments/
│   ├── ch5_4class/                       # (existing)
│   ├── ch6_ch7_3class/                   # (existing)
│   ├── ablation/                         # (existing)
│   ├── chapter3/                         # NEW: Ch3 LIF reservoir characterization
│   │   └── run_chapter3_lsm_characterization.py
│   └── interpretability/                 # NEW: Four-level interpretability validation
│       ├── run_level1_temporal_traceability.py
│       ├── run_eegnet_saliency_comparison.py
│       └── run_arspinet_v2_attention_prototype.py
```

The Reproduction Map table needs four new rows:

```
| BSC₆-ERP correlation (r=0.82 LPP) | experiments/interpretability/run_level1_temporal_traceability.py | L1 validation |
| Attention-prototype readout (66.7%) | experiments/interpretability/run_arspinet_v2_attention_prototype.py | L2-L4 validation |
| EEGNet saliency comparison | experiments/interpretability/run_eegnet_saliency_comparison.py | Head-to-head |
| Ch3 reservoir characterization | experiments/chapter3/run_chapter3_lsm_characterization.py | β/θ/ρ sweeps |
```

---

## Exact Steps to Execute

### Step 1: Copy the four new scripts into your local repo

```bash
cd dissoAdventureExperiments
mkdir -p experiments/interpretability
mkdir -p experiments/chapter3

# Copy the four scripts (from wherever you downloaded them)
cp run_level1_temporal_traceability.py experiments/interpretability/
cp run_eegnet_saliency_comparison.py experiments/interpretability/
cp run_arspinet_v2_attention_prototype.py experiments/interpretability/
cp run_chapter3_lsm_characterization.py experiments/chapter3/
```

### Step 2: Update README.md

Open README.md and make these edits:

**Edit 1:** In the Repository Structure section, after the `ablation/` entry, add:
```
│   ├── chapter3/                         #   Ch3: LIF reservoir characterization (synthetic)
│   │   └── run_chapter3_lsm_characterization.py  # β/θ/ρ sweeps, temporal FDR, readout comparison
│   └── interpretability/                 #   Four-level interpretability validation (April 2026)
│       ├── run_level1_temporal_traceability.py    # BSC₆-to-ERP correlation (r=0.82 LPP)
│       ├── run_eegnet_saliency_comparison.py      # EEGNet saliency vs ARSPI-Net attention
│       └── run_arspinet_v2_attention_prototype.py  # Attention-prototype readout (66.7%)
```

**Edit 2:** In the Reproduction Map → Primary Results table, add four rows:
```
| BSC₆-ERP correlation (r=0.82, R²=0.661) | `experiments/interpretability/run_level1_temporal_traceability.py` | Level 1 validation |
| Attention-prototype readout (66.7%) | `experiments/interpretability/run_arspinet_v2_attention_prototype.py` | Levels 2-4 validation |
| EEGNet saliency vs ARSPI-Net attention | `experiments/interpretability/run_eegnet_saliency_comparison.py` | Head-to-head comparison |
| Ch3 LIF reservoir characterization | `experiments/chapter3/run_chapter3_lsm_characterization.py` | β/θ/ρ sweeps, synthetic |
```

### Step 3: Commit and push

```bash
git add experiments/chapter3/ experiments/interpretability/ README.md
git commit -m "Add reproducibility scripts for Ch3 + interpretability validation (April 2026)

Closes 4 reproducibility gaps:
1. Level 1: BSC6 bin-to-ERP correlation (r=0.82 LPP, R^2=0.661)
2. Levels 2-4: Attention-prototype readout (66.7%, permutation p=0.634)
3. Head-to-head: EEGNet saliency (402-691ms) vs ARSPI-Net attention (176-254ms)
4. Chapter 3: LIF reservoir characterization (beta/theta/rho sweeps, synthetic)"

git push origin main
```

---

## After Pushing: Verification

Run this to confirm the four scripts are syntactically valid:

```bash
python -c "import ast; ast.parse(open('experiments/interpretability/run_level1_temporal_traceability.py').read()); print('OK')"
python -c "import ast; ast.parse(open('experiments/interpretability/run_eegnet_saliency_comparison.py').read()); print('OK')"
python -c "import ast; ast.parse(open('experiments/interpretability/run_arspinet_v2_attention_prototype.py').read()); print('OK')"
python -c "import ast; ast.parse(open('experiments/chapter3/run_chapter3_lsm_characterization.py').read()); print('OK')"
```

To actually run the experiments:
- **Chapter 3 script:** No data needed, runs on synthetic signals. `python experiments/chapter3/run_chapter3_lsm_characterization.py`
- **Level 1 script:** Needs `shape_features_211.pkl`. `python experiments/interpretability/run_level1_temporal_traceability.py`
- **Attention-prototype script:** Needs `shape_features_211.pkl` + PyTorch. `python experiments/interpretability/run_arspinet_v2_attention_prototype.py`
- **EEGNet saliency script:** Needs `shape_features_211.pkl` + PyTorch + GPU recommended. `python experiments/interpretability/run_eegnet_saliency_comparison.py`
