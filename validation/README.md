# Data Validation Scripts

This directory contains validation and quality-control scripts for the SHAPE EEG dataset used throughout the ARSPI-Net dissertation.

## Scripts

### `validate_shape_data.py` — Broad-Condition EEG Validation (10 Checks)

Validates the 3-class (Negative / Neutral / Pleasant) SHAPE EEG files against Brady's specification.

| Check | Description | Expected |
|-------|-------------|----------|
| 1 | File dimensions | (1229, 34) per file |
| 2 | Sampling rate consistency | 1024 Hz |
| 3 | Channel count | 34 EEG channels |
| 4 | Baseline period | First 205 samples (~200 ms) near zero |
| 5 | NaN / Inf detection | No missing or infinite values |
| 6 | Amplitude range | Within ±500 µV (microvolts) |
| 7 | Flat channel detection | No channels with zero variance |
| 8 | File naming pattern | `SHAPE_Community_{ID}_{Condition}_BC.txt` |
| 9 | Subject completeness | 3 files per subject (Neg, Neu, Pos) |
| 10 | Cross-subject consistency | Consistent dimensions across all subjects |

### `validate_subcategory_data.py` — 4-Class Subcategory Validation (12 Checks)

Validates the 4-class (Threat / Mutilation / Cute / Erotic) SHAPE EEG subcategory files.

| Check | Description | Expected |
|-------|-------------|----------|
| 1–10 | Same as broad-condition checks | Same thresholds |
| 11 | Subcategory file pattern | `SHAPE_Community_{ID}_IAPS{Valence}_{Category}_BC.txt` |
| 12 | Subcategory completeness | 4 files per subject |

### `verify_validators.py` — Meta-Verification (20 Tests)

Verifies that the validation scripts themselves are correct by running them against mock data with known properties.

**Result: 20/20 PASS.**

## Usage

```bash
# Validate broad-condition data (requires SHAPE EEG files in batch_data/)
python validation/validate_shape_data.py

# Validate subcategory data (requires files in categoriesbatch{1-4}/)
python validation/validate_subcategory_data.py

# Verify the validators themselves (no external data required)
python validation/verify_validators.py
```

## Data Requirements

The SHAPE EEG dataset is not included in this repository. See the [SHAPE study page](https://lab-can.com/shape/) for data access information.
