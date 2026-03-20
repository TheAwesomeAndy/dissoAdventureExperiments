# ARSPI-Net Methodology Rules

Established March 2026 from advisor feedback.
These rules govern all experimental design and writing in this dissertation.

## Rule 1 — Horizontal Before Vertical

When a cross-chapter claim emerges (e.g. "three operationally distinct
layers"), immediately design the single experiment that tests it directly
using features from all chapters on the same task with the same protocol.
Do not finalize chapter-level experiments without the system-level
integration test. Every cross-chapter narrative needs a cross-chapter
experiment.

## Rule 2 — Claims Require Direct Tests

Every dissociation or complementarity claim must be accompanied by a
direct ablation/combination experiment. Narrative consistency across
separate analyses is not proof. If the direct test cannot be designed,
do not make the claim.

## Rule 3 — No Unearned Terminology

Do not use a mathematical framework's vocabulary (e.g. "information-
theoretic decomposition") unless the quantities that framework defines
have actually been computed. Use operationally precise language
("operationally distinct response layers") when only operational
evidence exists.

## Rule 4 — Adversarial Committee Test

Before finalizing any experimental program, explicitly list the 3 most
likely committee objections and verify each has a corresponding
EXPERIMENT, not just a narrative response. Conventional baselines
(EEGNet, GRU/LSTM) are strategically non-negotiable for positioning.

## Rule 5 — Linear Readouts for Content Comparison

When comparing information content across feature families, use linear
classifiers (linear SVM, logistic regression) as primary readout so
geometric separability is exposed, not rescued by nonlinear kernels.
RBF-SVM belongs in appendix sensitivity checks only.

## Rule 6 — Interrogate Null Results

When a null clinical or condition result emerges, ask: is this absence
of effect or absence of measurement resolution? Test at finer resolution
(per-channel instead of channel-averaged) before concluding absence.

## Rule 7 — Build Summary Tables Immediately

When a cross-condition or cross-granularity comparison is central to the
argument, build the quantitative summary table as soon as the numbers
exist. Do not distribute key comparisons across prose.

---

## Central Thesis (corrected March 2026)

"ARSPI-Net reveals three operationally distinct response layers in
affective EEG — discriminative representation, dynamical response,
and topology/coupling — each sensitive to different aspects of the
signal."

This replaces any prior framing as "information-theoretic decomposition."
The thesis is earned by the layer ablation matrix (A1-A9).
Frame as measurement framework, not classifier.

## Overclaim Corrections

Replace "information-theoretic decomposition" with "operationally
distinct response layers."

Replace "the spatial layer is where clinical biomarkers live" with
"the strongest clinical sensitivity appears in the spatial/topological
analyses."

Replace "the temporal layer is where condition encoding lives" with
"the temporal dynamical analyses are more strongly condition-sensitive."
