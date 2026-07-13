# Final Dissertation Audit Changelog

Audit completed July 13, 2026. The attached dissertation source was treated as
authoritative.

## Submission and presentation

- Conformed the front matter to the current Stony Brook sequence and retained
  the required unsigned committee page as page ii.
- Matched committee names to the signed form and verified the current Graduate
  School dean.
- Placed the bibliography before the appendix.
- Set one-inch margins, double-spaced body text, single-spaced captions, Roman
  preliminary pagination, and Arabic body pagination.
- Added PDF/A-2b metadata and removed build warnings, unresolved citations,
  unresolved references, and overfull content.
- Removed all external source-hosting links and development-platform language
  from the dissertation.

## Writing and scholarship

- Removed drafting residue, defense-oriented instructions, formulaic booster
  language, and unsupported claims of optimality, mechanism, biomarkers,
  physiological connectivity, and clinical utility.
- Reworked the background review to distinguish cited theory from what the
  finite implementation and supplied data actually establish.
- Added or corrected foundational references for compression, attribution,
  attention, interpretability, reservoir computing, and the author's 2023 and
  2024 publications.
- Standardized the study name as Stress, Health, and the Psychophysiology of
  Emotion (SHAPE) and cited the official Laboratory for Clinical Affective
  Neuroscience page.

## Mathematics, methods, and results

- Removed the invalid mutual-information and Fano-derived claims.
- Replaced theorem-like statements that lacked sufficient assumptions with
  appropriately limited empirical or analogical interpretations.
- Corrected the reservoir update description to match the custom NumPy model,
  including reset and nonnegative membrane flooring.
- Distinguished the within-epoch phase-difference concentration statistic from
  classical cross-trial phase-locking value.
- Recast subject centering as transductive panel normalization rather than
  prospective single-record evaluation.
- Identified the invalid five-diagnosis control count and retained the related
  figure only as an explicitly labeled legacy display.
- Replaced dependent fold-level significance claims with descriptive reporting
  or subject-level/permutation requirements.
- Regenerated the controlled Chapter 4 figures with preprocessing fitted inside
  each training fold.

## Reproducibility corrections

- Updated the maintained analysis path to use one shared reservoir transform
  across channels.
- Standardized on training data and fitted one common PCA basis within each
  fold before applying it to held-out observations.
- Built data-derived graphs from training-fold information only.
- Standardized the corrected BSC6 readout window to steps 10–70.
- Disabled legacy entry points that would reproduce known global-preprocessing
  or held-out-selection defects without an explicit archival override.
- Added verification checks for shared channel coordinates and reran the
  available automated tests successfully.

## Unresolved data limitation

The raw SHAPE inputs were not supplied. Corrected real-data PCA, graph, and
classification estimates could not be regenerated. Rather than substitute
unverifiable numbers, the dissertation preserves the defended values as
clearly labeled descriptive history and states the exact corrections required
for a future rerun.
