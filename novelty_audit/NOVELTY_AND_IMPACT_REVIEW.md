# ARSPI-Net novelty and impact review

## Defensible novelty boundary

Liquid-state and spiking models for EEG emotion recognition predate this dissertation. ARSPI-Net therefore should not be described as the first use of a liquid-state machine, reservoir, or spiking network for EEG emotion classification. The publishable contribution is narrower and stronger:

1. a fixed, fully specified LIF reservoir evaluated on a transdiagnostic affective-ERP cohort;
2. an inspectable six-bin temporal spike representation evaluated at two label granularities;
3. participant-grouped, fold-safe evaluation reconstructed from raw files after a provenance audit;
4. targeted destruction controls showing that temporal-bin order and electrode identity contribute to decoding;
5. seed and temporal-window robustness, including disclosure that raw ERP summaries remain the stronger classifier; and
6. transparent separation of corrected evidence from archived exploratory graph and clinical analyses.

This positions the work as a reproducible representation-and-audit paper, not a state-of-the-art accuracy paper or a validated clinical-biomarker paper.

## Corrected headline results

| Measure | Three conditions | Four categories |
|---|---:|---:|
| Participants / observations | 211 / 633 | 210 / 840 |
| Chance | 33.3% | 25.0% |
| Raw ERP-16 | 68.1% [65.1, 71.1] | 60.5% [57.9, 63.0] |
| Conventional band-power/Hjorth | 49.4% [46.8, 51.9] | 36.7% [34.6, 38.9] |
| BSC6/SRP-64 | 60.6% [57.8, 63.4] | 53.2% [50.6, 55.8] |
| Reservoir minus conventional | +11.3 pp [7.9, 14.7] | +16.5 pp [13.3, 19.8] |
| Conventional + reservoir | 62.0% [59.1, 64.8] | 53.5% [50.8, 56.3] |
| Participant-label permutation p | 0.0099 | 0.0099 |

Intervals are participant-bootstrap 95% intervals around predictions pooled from five repeats of five-fold participant-grouped cross-validation.

## Mechanistic controls

Relative to intact BSC6/SRP-64, balanced accuracy falls after temporal-bin shuffle by 7.4 points (three conditions) and 12.9 points (four categories); after temporal collapse by 4.9 and 8.6 points; and after electrode shuffle by 13.9 and 20.1 points. Total-by-bin population counts perform substantially worse than intact channel-resolved codes. These controls support the statement that ordered temporal bins and electrode identity contribute to the readout.

Three fixed reservoir seeds yield 60.6%, 60.5%, and 62.1% in the three-condition robustness run. Early, middle, late, and latest windows yield 60.5%, 56.9%, 56.7%, and 48.5%; the full verified post-baseline epoch yields 63.3%. This supports distributed information across the epoch and argues against selecting only a favorable early interval.

## Claims to avoid

- Do not claim that ARSPI-Net is the first LSM or SNN for EEG emotion recognition.
- Do not claim state-of-the-art classification; raw ERP summaries are stronger here.
- Do not retain the earlier three-to-four-class “regime inversion.” It does not survive corrected preprocessing.
- Do not call uncorrected clinical screens biomarkers, disorder-specific signatures, network reorganization, or replications.
- Do not present complete-panel centering as prospective performance.
- Do not generalize the archived graph-propagation sweep into a universal node-count or density law.
- Do not infer arousal sensitivity without directly modeling stimulus-level arousal ratings.

## Recommended publication angle

**Working title:** *Auditable Temporal Spike Codes for Affective EEG: Participant-Grouped Validation, Destruction Controls, and a Post-Hoc Provenance Audit*

The central question is not whether a spiking model wins an accuracy benchmark. It is whether a fixed reservoir preserves useful, identifiable temporal and spatial structure under leakage-resistant evaluation. The raw-ERP result should appear prominently as a boundary condition. That candor strengthens the paper: the reservoir offers a structured intermediate representation with measurable information loss, and the controls identify what the representation uses.

## Reproducibility record

The supplied legacy feature pickle was verified against the raw files. Its input equals the first 1,024 samples selected at stride four and z-scored by channel; it includes baseline samples, omits the end of the post-stimulus epoch, and does not implement antialias decimation. Corrected experiments use raw rows 205--1228, antialias decimation by four, participant-grouped folds, fold-local scaling, and a shared data-independent sparse random projection. Subject 127 is excluded for a constant recording; Subject 195 is additionally excluded from the four-category task for a constant Cute recording.

The experiment scripts, machine-readable results, and validation figure accompany this report.
