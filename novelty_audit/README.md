# Corrected SHAPE novelty experiments

`run_corrected_shape_novelty.py` reconstructs three-condition and four-category analyses from the raw SHAPE text files. `run_reservoir_robustness.py` evaluates reservoir-seed and temporal-window robustness. `make_novelty_figure.py` builds the validation figure from the JSON results.

Large raw data, extracted files, and cached feature matrices are intentionally excluded from the deliverable. The scripts expect the extracted three-condition and four-category directories configured at the top of each file.

Primary outputs are in `results/corrected_novelty_results.json`, `results/reservoir_robustness.json`, and `results/corrected_novelty_validation.pdf`.
