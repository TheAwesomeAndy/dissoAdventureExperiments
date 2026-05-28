# Q&A Index

A committee-question pointer index for the canonical deck
`ARSPI-Net_Defense.pdf`. Each entry is an anticipated question (or
question family), the single-sentence answer to give, and the PDF
page(s) to navigate to while answering.

Questions are grouped by topic. Within each topic, the **landing
slide** is the main-deck slide to put up while answering; the
**deeper appendix slides** are listed for follow-ups.

Approximate appendix page numbers are given to 1--2 pages because
the metropolis section dividers can shift counts by ±1 between
revisions; verify exact pages on the live PDF before defense.

---

## 1. Reservoir stability and contraction (Exp A)

### Q. "Your spectral radius ρ(W) = 0.265 is well below 1. Isn't that under-tuned for echo-state behaviour?"

**Land.** PDF 18 (audit slide).
**Answer.** ρ(W) is the wrong number. The driven Lyapunov exponent λ₁
is uniformly negative across N = 3,165 ERP trajectories; the
reservoir is in the contraction regime under real drive. Echo-state
behaviour is measured, not tuned.
**Deeper.** PDF ≈ 95 (eigenspectrum of W); PDF ≈ 96 (Benettin sample
trajectories); PDF ≈ 97 (per-trial λ₁ scatter).

### Q. "Why is N = 3,165 different from N = 4,220 on the previous slide?"

**Land.** PDF 18 (third bullet on the audit slide).
**Answer.** N = 4,220 is the original chDynamics Lyapunov pool;
N = 3,165 is the strict clean-ERP subset used for the defense-audit
Benettin recompute. Both populations are 100% stable. The subset is
a stricter trial-cleaning gate, not a different analysis.

### Q. "Could you have a non-contracting reservoir with the same ρ(W)?"

**Land.** PDF 18.
**Answer.** In principle yes for arbitrary drive; in practice on this
data, no --- the driven λ₁ distribution is on the stable side of zero
with no overlap. The bound on ρ(W) is necessary but not sufficient;
the measurement is sufficient.

### Q. "What if the EEG drive were larger amplitude --- would λ₁ flip?"

**Land.** PDF 17 (contraction visualised) → PDF 18.
**Answer.** Driven λ₁ is a function of the drive distribution. On
in-scope ERP amplitudes the result holds; out-of-scope amplitudes
are by definition not in the validation set. (Pointer to scope:
PDF 56.)

---

## 2. Embedding dimension and Takens (Exp B)

### Q. "Are you claiming Takens-equivalent reconstruction?"

**Land.** PDF 23 (audit slide).
**Answer.** No. The claim is operational: the FNN-measured embedding
dimension m\* is well under the 64-dim PCA projection, so the
embedding capacity exceeds the requirement on this data. Takens is
the motivation, not the guarantee.
**Deeper.** PDF ≈ 98 (delay embedding); PDF ≈ 99 (per-trial FNN
distribution).

### Q. "Why 64 PCA components? Why not 32 or 128?"

**Land.** PDF 23 → PDF 21 (spike-to-embedding pipeline).
**Answer.** 64 is above the FNN-measured m\* across all trials with
margin. 32 risks clipping; 128 doesn't add capacity beyond what FNN
needs. Variance retained at 64 is reported in the chapter.

### Q. "Could the FNN measurement be biased by the BSC₆ binning?"

**Land.** PDF 23 → App "Why six temporal bins" (PDF ≈ 64).
**Answer.** BSC₆ is six bins aligned to ERP windows (N1/P200/P300/LPP);
m\* is measured **after** BSC₆ in the pipeline, so any binning bias
is folded into the measurement. The FNN result is what the
downstream embedding actually sees.

---

## 3. Memory capacity and leak β (Exp C)

### Q. "Why β = 0.05 and not the MC peak at β ≈ 0.012?"

**Land.** PDF 19 (audit slide).
**Answer.** β = 0.05 is at 91% of MC peak --- inside the plateau,
not at the peak. It was selected by matching the LIF time constant
to the 256-step ERP window, then confirmed post-hoc by the MC sweep.
β\* would over-integrate relative to the ERP window.
**Deeper.** PDF ≈ 100 (input drive); PDF ≈ 101 (state response per
β); PDF ≈ 102 (MC vs. τ).

### Q. "How sensitive is the downstream accuracy to β within the plateau?"

**Land.** PDF 19.
**Answer.** The reservoir initialisation robustness (mentioned on
PDF 21: "<2% accuracy variance across reservoir seeds") generalises
to small β perturbations within the plateau. Exact accuracy-vs-β
curves are not on the main deck; the MC plateau is the
methodologically defensible answer.

### Q. "Why is MC on this slide and not in the graph block?"

**Land.** PDF 19 → PDF 14 (LIF, where β is introduced).
**Answer.** MC is a reservoir-side property of the LIF dynamics --- it
defends a hyperparameter introduced in the LIF block. It was moved
out of the graph block per the defense-audit feedback because it
has no bearing on the propagation operator.

---

## 4. Graph propagation and the POC (Contribution 1)

### Q. "Your POC says message passing underperforms the non-propagated embedding. Doesn't that contradict the EEG-GNN literature?"

**Land.** PDF 37 (POC) → PDF 38 (robust across graph construction).
**Answer.** The POC is a measured operating characteristic in the
regime relevant to clinical EEG: ≤ 64 channels, > 10% density,
< 1,000 samples. The literature reports gains in different regimes.
The POC contributes the regime boundary that the literature lacks.

### Q. "What if you use a learned propagation operator instead of A_tilde?"

**Land.** PDF 38.
**Answer.** The sweep on PDF 38 covers correlation, distance, and
functional adjacencies. A learned operator can in principle escape
the oversmoothing regime; the boundary remains as a design
criterion for fixed operators, which are the operationally relevant
ones for clinical-scale data.

### Q. "How tight are the theoretical bounds on the graph claim?"

**Land.** PDF 40 (TB audit).
**Answer.** Lyapunov: loose (operating point far from boundary).
Takens: ~50% loose. Memory capacity: tighter, ~91% of peak. All
three operating points sit inside the safe envelope.

### Q. "Is the Dirichlet-energy drop a sufficient mechanistic explanation?"

**Land.** PDF 39 (diffusion mechanism).
**Answer.** It's a mechanistic **correlate** of the observed
boundary, not a sufficient proof. The 84% energy drop by K = 2
co-occurs with the accuracy fall in exp03; both are measurements,
no causal claim is made beyond co-occurrence.

---

## 5. Null results and methodology (Exp D)

### Q. "Did the channel-averaged null preclude the layer-specific finding?"

**Land.** PDF 49 (null forced methodology) → PDF 50 (per-trial null).
**Answer.** The channel-averaged null returned 0/49 significant
comparisons --- under the standard reading, that would close the
chapter. The interpretability framework forced a reanalysis at
per-channel resolution, which yielded the SUD 53.6% and PTSD 52.6%
results. The layer-specific finding does not exist without that
null.

### Q. "Why per-trial channel permutation instead of a global permutation?"

**Land.** PDF 50.
**Answer.** Per-trial is the strictest spatial null that preserves
stimulus-class balance. A global permutation only tests dataset
shuffling; per-trial tests whether channel identity carries the
signal. The classifier cannot compensate.

### Q. "How many permutations, what folding?"

**Land.** PDF 50.
**Answer.** 500 permutations × 5-fold CV at the subject level. CLI
flag allows swapping to disorder-label permutations when the CSV
provides the disorder column.

---

## 6. Failure-driven rigor

### Q. "Walk me through the pivots."

**Land.** PDF 51 (failure gallery).
**Answer.** Four pivots: (i) autonomous ρ(W) → driven λ₁; (ii) MC
peak optimisation → time-constant matching; (iii) channel-averaged
null → per-trial permutation; (iv) channel-averaged null → per-
channel reanalysis. Each failure forced a methodology change that
ended up in a contribution.

### Q. "Are there other failures you've omitted?"

**Land.** PDF 51 → PDF 24 (path to silicon).
**Answer.** The path-to-silicon SNN training attempt is an honest
omission --- it didn't change the methodology, so it isn't on the
pivot gallery. It's a future direction.

---

## 7. Three-layer framework, ablation, accuracy trade

### Q. "What's the cost of intrinsic interpretability vs. a black-box model?"

**Land.** PDF 54 (accuracy--interpretability trade).
**Answer.** The trade is reported; the framework explicitly does not
maximise accuracy. The contribution is that the lost accuracy can
be quantified and the interpretability gained is named.

### Q. "Could you achieve the same disorder-layer mapping with a black box?"

**Land.** PDF 52 (different disorders, different layers) → PDF 53
(Layer ablation).
**Answer.** A black box could fit the four-disorder mapping by
chance under enough hyperparameter search, but it cannot **attribute**
disorders to layers because it has no layers to ablate. Layer
ablation is what makes the mapping a finding rather than a fit.

---

## 8. Scope, future, replication

### Q. "What's the validation scope?"

**Land.** PDF 56 (scope of validation).
**Answer.** Read the scope frame verbatim. It enumerates the
in-scope datasets and the out-of-scope claims that the work
deliberately does not make.

### Q. "What's your top future direction?"

**Land.** PDF 57 (future directions).
**Answer.** Read the top of the future-directions list. Pick one;
do not enumerate.

### Q. "Is the code replicable?"

**Land.** PDF 58 (contributions restated) → PDF 60 (anchor map).
**Answer.** Repository public; every defense claim maps to a
chapter and chapter scripts. PR #25 added the audit-slide source
figures to the same repo for traceability.

---

## 9. Provenance and AI assistance

### Q. "What AI tools did you use?"

**Land.** PDF 61 (AC slide).
**Answer.** Read the AC slide verbatim. Code assistants for
software, figure-generation scaffolding, document drafting.
Every output author-reviewed; nothing accepted unverified.

### Q. "How was assisted code verified?"

**Land.** PDF 61.
**Answer.** All assisted code was executed, all assisted claims
re-derived or ground-truth-matched, all assisted figures inspected
and accepted, retained, or discarded.

### Q. "Are the scientific claims yours?"

**Land.** PDF 61 → PDF 60 (anchor map).
**Answer.** Yes. The AC slide names "research questions, scientific
decisions, mathematical interpretation, experimental and scientific
decisions, final scientific claims" as author-owned. The anchor
map shows each claim is in a dissertation chapter.

---

## Quick navigation table

| Topic                                   | Land at PDF | Appendix range  |
|----------------------------------------|-------------|-----------------|
| Reservoir stability (Exp A)            | 18          | 95--97          |
| Embedding dimension (Exp B)            | 23          | 98--99          |
| Memory capacity (Exp C)                | 19          | 100--102        |
| Graph POC                              | 37, 38      | 70--72 (existing) |
| Bounds vs. measurement (TB)            | 40          | 94 (overview)   |
| Null methodology (Exp D)               | 49, 50      | 103--105        |
| Failure pivots                         | 51          | (4 audit slides) |
| Three-layer / ablation                 | 52, 53      | 78--82          |
| Scope / future                         | 56, 57      | --              |
| Anchor map (AM)                        | 60          | 106             |
| Provenance (AC)                        | 61          | --              |

**Page numbers ±1**: the appendix range is approximate; the section
divider for "Appendix --- Defense Audit Diagnostics" shifts the raw
diagnostics by ±1 page depending on metropolis revision. Confirm
exact pages on the live PDF the week of defense.
