# Speaker Notes --- Defense Audit Slides

Speaker notes for the **8 inserted audit slides only**. The existing 45
dissertation slides retain whatever notes the author has elsewhere; this
file is scoped strictly to the new material added in PR #25.

Each slide entry has the same structure:

- **Open** --- the one or two sentences spoken while clicking onto the
  slide. ~15 s, verbatim if helpful.
- **Argument** --- the substantive content. ~45--60 s, bulleted.
- **Land** --- if a committee question is foreseeable here, the
  appendix slide(s) to navigate to.
- **Anticipate** --- the most likely committee question and a one-line
  response.
- **Cut line** --- the single sentence to say if running short on
  time, when the rest of the slide must be skipped.

All times are matched to `timing_map.md`.

---

## #1 --- Defense audit: autonomous ρ(W) vs. driven λ₁

**PDF page 18 · 90 s · Reservoir / LIF**

**Open.** "Before we move on, an honest audit. The Echo-State
Property literature commonly cites the autonomous spectral radius
ρ(W) as the stability criterion. I want to show what that number is
on this reservoir, and what the actual contraction looks like when
the reservoir is doing its job."

**Argument.**

- "The left panel shows ρ(W) = 0.265 --- well under the canonical
  ρ < 1 threshold. That is an undriven property of the matrix W."
- "The right panel shows the Lyapunov exponent λ₁ measured on
  real ERP trajectories via Benettin's algorithm. The driven
  distribution sits firmly in the stable region; 100% of trials
  have λ₁ < 0."
- "Two populations are involved: the original chDynamics Lyapunov
  pool, N = 4,220, and the strict clean-ERP subset used for the
  defense audit, N = 3,165. Both are 100% stable. The subset is
  not a cherry-pick; it is a stricter trial-cleaning gate."
- "The methodological claim is: stability is a measured property
  of the reservoir in operation, not a cited property of W in
  isolation."

**Land.** App "Exp A raw: eigenspectrum of W" (PDF ≈ 95) for the
full spectrum of W; App "Exp A raw: Benettin sample trajectories"
(PDF ≈ 96) for individual trajectories; App "Exp A raw: per-trial
λ₁ scatter" (PDF ≈ 97) for the full per-trial cloud.

**Anticipate.** "Why didn't you tune ρ(W) to the canonical edge of
chaos near 1?" → "ρ(W) bounds the autonomous map; under real ERP
drive the operational stability test is the driven Lyapunov
exponent, which on this data is uniformly negative. Pushing ρ(W)
up would not change the result of that operational test."

**Cut line.** "Autonomous ρ(W) is 0.265; driven λ₁ is uniformly
negative across N = 3,165 trials. Stability is measured, not
asserted."

---

## #3 --- Defense audit: memory-capacity regime for leak β

**PDF page 19 · 90 s · Reservoir / LIF**

**Open.** "The leak factor β = 0.05 in the reservoir is a design
choice; the committee will reasonably ask whether it is the right
choice. Here is the operating characteristic."

**Argument.**

- "The y-axis is total memory capacity, summed over delays.
  Measured directly on the reservoir using the standard MC
  benchmark."
- "Peak MC is at β\* ≈ 0.012 with MC ≈ 0.84. The chosen β = 0.05
  sits at 91% of peak --- inside the plateau, not at the peak."
- "The plateau β ∈ [0.010, 0.118] holds MC ≥ 0.75. β = 0.05 was
  not selected by sweeping MC; it was selected by the time-constant
  argument vs. the 256-step ERP window. The MC sweep, post-hoc,
  shows the time-constant choice lands inside this measured
  plateau."
- "MC is a reservoir-side property of the LIF dynamics. It has no
  bearing on graph propagation; that's why this slide sits here,
  next to the contraction audit, and not later in the graph block."

**Land.** App "Exp C raw: input drive" (PDF ≈ 100); App "Exp C raw:
state response per β" (PDF ≈ 101); App "Exp C raw: memory capacity
vs. τ" (PDF ≈ 102) for the full sweep.

**Anticipate.** "Why not β\* exactly?" → "Because at β\* the
effective time constant is too long relative to the 256-step ERP
window; the reservoir over-integrates. β = 0.05 is the rounder
value that satisfies both the time-constant matching and the MC
plateau constraint."

**Cut line.** "β = 0.05 sits at 91% of measured MC peak, inside the
plateau; it was chosen by time-constant matching and confirmed
post-hoc by the MC sweep."

---

## #2 --- Defense audit: FNN-measured embedding dimension

**PDF page 23 · 75 s · Spike-to-embedding**

**Open.** "The spike-to-embedding pipeline outputs a 64-dimensional
PCA-projected vector. Is 64 enough? Takens motivates the question;
Kennel--Brown FNN answers it."

**Argument.**

- "The Kennel--Brown false-nearest-neighbour test sweeps embedding
  dimension and reports the smallest m for which neighbour
  relationships stabilise."
- "Measured m\* is well under 64 in the PCA-projected space, and
  well under 256 in the raw BSC₆ space. The 64-dim PCA projection
  is **above** the embedding requirement, not at it."
- "This is not a claim of Takens-reconstruction guarantees; it is
  the weaker, useful claim that the embedding capacity comfortably
  exceeds the measured intrinsic dimension."

**Land.** App "Exp B raw: delay embedding" (PDF ≈ 98); App "Exp B
raw: per-trial FNN distribution" (PDF ≈ 99).

**Anticipate.** "Have you proven Takens-equivalence?" → "No, and I
explicitly do not claim it. The claim is operational: the embedding
capacity exceeds the measured embedding dimension on this data."

**Cut line.** "FNN says m\* ≪ 64. The 64-dim embedding is over the
requirement; no Takens guarantee is claimed."

---

## #4 --- Defense audit: theoretical bounds vs. measured operating points

**PDF page 40 · 60 s · Graph propagation**

**Open.** "Three of the experiments I just walked through have
theoretical bounds in the literature. This slide places the
measured operating point against the bound for each."

**Argument.**

- "Three panels: Lyapunov stability bound, Takens embedding
  bound, and memory-capacity bound."
- "In each panel, the theoretical bound is drawn, and the
  measured operating point on this reservoir is marked."
- "The point is not that the bounds are tight --- they are not;
  some are pessimistic --- but that each measured operating
  point is consistent with the bound that applies."

**Land.** App "Defense audit: raw diagnostics overview" (PDF ≈ 94)
for the source index of which raw figures back each panel.

**Anticipate.** "Are the bounds sharp on this data?" → "The
Lyapunov bound is loose --- driven λ₁ sits well inside the
predicted-stable region. The Takens bound is loose by roughly
half --- m\* is well below the embedding ceiling. The MC bound is
tighter --- β = 0.05 is at 91% of peak. Each measurement is
consistent with the bound that applies; I do not claim the bounds
are tight."

**Cut line.** "Three bounds, three measured operating points;
each measurement is consistent with the bound that applies."

---

## #5 --- Defense audit: per-trial channel-permutation null

**PDF page 50 · 75 s · Null / methodology pivot**

**Open.** "The channel-averaged null on the previous slide is the
weak form. The strict spatial null is per-trial channel permutation
--- the harder test."

**Argument.**

- "Each trial gets an independent channel permutation π, applied
  before the classifier sees the trial. A flat classifier cannot
  compensate for trial-by-trial randomised channel order."
- "500 permutations × 5 CV folds, all at the subject level."
- "Stimulus-class permutation, not a global τ --- the test
  measures whether channel identity carries the signal, not
  whether the dataset is shuffled."
- "The observed-vs-null gap is the punchline. CLI flags allow
  swapping to disorder-label permutations when the CSV provides
  the disorder column."

**Land.** App "Exp D raw: feature layout" (PDF ≈ 103); App "Exp D
raw: class-label distribution" (PDF ≈ 104); App "Exp D raw:
observed vs. null example" (PDF ≈ 105).

**Anticipate.** "Did you do per-class or per-subject permutation?"
→ "Per-trial within stimulus class; subject-level CV. The strictest
spatial null that preserves the class balance."

**Cut line.** "Per-trial channel permutation, 500 × 5-fold; the
signal survives the strictest spatial null we can apply."

---

## #6 --- Defense audit: failure gallery --- four pivots

**PDF page 51 · 90 s · Failure-driven rigor**

**Open.** "Four times during this work, I tried something, it
failed, and the failure changed the methodology. I want to show
those four pivots together, not buried in chapters."

**Argument.**

- "Pivot 1 (Exp A): autonomous ρ(W) → driven λ₁. The default
  ESN stability citation bounds the autonomous map; for our
  driven-ERP regime the operationally relevant test is the
  driven Lyapunov measurement, which is what we now report."
- "Pivot 2 (Exp C): MC-peak optimisation → time-constant matching.
  The MC-peak β corresponds to an effective time constant that
  over-integrates relative to the 256-step ERP window, so we
  selected β by time-constant matching and confirmed post-hoc
  that the choice lands inside the MC plateau."
- "Pivot 3 (Exp D, weak null): channel-averaged → per-trial
  permutation. The channel-averaged null was uninformative."
- "Pivot 4 (Exp D, applied): channel-averaged null → per-channel
  reanalysis revealing the layer-specific finding. The null
  forced the methodology that became Contribution 4."

**Land.** Each pivot has its own audit slide above (insertions
#1, #3, #5, and the existing 'null forced methodology' frame
which is PDF 49).

**Anticipate.** "Are there other failures you're not showing?" →
"Yes --- the path-to-silicon SNN training attempt (mentioned on
PDF 24); it didn't change the methodology so it isn't on the
pivot gallery. Happy to walk through it."

**Cut line.** "Four pivots: ρ(W) → λ₁, MC peak → time-constant,
weak null → strict null, channel average → per-channel reanalysis.
Each failure forced a methodology change."

---

## #7 --- Defense claims are anchored to dissertation chapters

**PDF page 60 · 30 s · Contributions / closure**

**Open.** "Before I close with the provenance disclosure, one
slide to anchor each defense claim back to a dissertation chapter."

**Argument.**

- "Each row of this table is a defense claim. Each column maps it
  to the dissertation chapter where the experimental detail
  lives. There is no orphan claim."
- "I do not read the table. Anyone wanting depth on a claim can
  find it via the column."

**Land.** App "AM: figure-to-source backup index" (PDF ≈ 106) for
the figure-level provenance.

**Anticipate.** "Is claim X actually in chapter Y?" → "Yes, the
table is built from the chapter ToCs. The appendix figure-to-source
slide drills further if needed."

**Cut line.** "Every defense claim maps to a chapter. The table is
exhaustive."

---

## #8 --- Research provenance and assistance

**PDF page 61 · 45 s · Final disclosure**

**Open.** "One last slide before I take questions. Research
provenance and assistance --- what was author-owned, what was
AI-assisted, and how that assistance was verified."

**Argument.** Read the three column headers and the bullets under
each as printed on the slide. Do not paraphrase or expand beyond
what the slide states; the AC slide is the canonical disclosure
and should be read from, not summarised from notes.

**Land.** None --- this is a self-contained statement.

**Anticipate.** "What AI tools or assistance specifically?" →
Refer the questioner to the bullets as printed on the AC slide.
Decline to elaborate beyond what the slide itself states.

**Cut line.** Do not cut this slide. If absolutely behind: read
the three column headers, omit the bullets. ~20 s minimum.
