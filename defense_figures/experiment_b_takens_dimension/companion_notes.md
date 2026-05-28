# Experiment B — Companion Notes (Author's Voice)

> Skeleton for the author-written prose that accompanies Experiment B's
> figures. Do not delegate the writing of the Motivation and Derivation
> sections — they are the author's intellectual signature on the
> theorem-grounding claim of this dissertation.

---

## 1. Pre-registration

**Expected outcome:**
- Per-trial m* in the range [3, 12].
- The reservoir's post-PCA effective dimension (64) exceeds m* by a factor
  of 5–20×. The raw reservoir state dimension (256) exceeds it by 20–80×.
- The conclusion will be: Takens-sufficiency is satisfied for this dataset.

**What I will say if m* lands outside [3, 12]:**
> *AUTHOR WRITES HERE — one paragraph stating, before measurement, what
> claim the author will make if m* lands < 3 (suggests aggressive
> low-dimensional structure, possibly under-sampled), 12 < m* < 64 (still
> Takens-sufficient w.r.t. the post-PCA dim), 64 < m* < 256 (Takens-
> sufficient only w.r.t. the raw reservoir state, not the post-PCA
> projection — would require reframing the slide title), or m* > 256
> (the reservoir is structurally insufficient — would require a
> theoretical re-examination of the architectural choice).*

---

## 2. Motivation — *"Why I asked Takens, and not just cited him"*

> *AUTHOR WRITES HERE — 2–3 paragraphs, first-person, narrating:*
>
> *— Many reservoir-computing papers cite Maass-Markram (the separation
>     property) or Takens (the embedding theorem) and proceed to use a
>     reservoir without measuring whether the cited theorem's premise
>     holds for their specific dataset. Why did the author refuse this?*
>
> *— The mentor's note: a citation to Takens is decoration; a measured
>     embedding dimension is doctoral. The author resolved to measure m*
>     on the actual SHAPE ERPs and show that the reservoir's state space
>     exceeds it — making Takens-sufficiency a measured fact about this
>     work, not a borrowed promise about reservoirs in general.*
>
> *— The choice of Kennel-Brown over Cao (or other variants) and why.*

This is the talking point used while introducing the slide. The committee
sees the FNN curve; the author tells them why measuring m* was a
theoretical commitment, not a methodological convenience.

---

## 3. First-principles statement — *"Why m* is the right number"*

### 3.1 Takens' theorem (informal)

Let `Φ: M → M` be a smooth flow on a compact attractor `A ⊂ M` of box
dimension `d`. Let `h: M → ℝ` be a generic observation function. Then for
any embedding dimension `m > 2d`, the delay map

```
Ψ(x) = (h(x), h(Φ^τ(x)), h(Φ^{2τ}(x)), ..., h(Φ^{(m-1)τ}(x)))
```

is, for *generic* choices of `h` and `τ`, an embedding of `A` into `ℝᵐ`.

**Interpretation:** if the latent dynamics live on an attractor of box
dimension `d`, then an `m`-dimensional delay reconstruction with `m > 2d`
preserves the attractor's topology. The dynamical state of the system can
be **recovered** from a single scalar observation channel.

### 3.2 Operational form — Kennel-Brown FNN

The Kennel et al. (1992) FNN algorithm operationalizes the "minimum m" of
Takens' theorem by counting **false neighbors**: pairs of points that are
nearby in `m`-dim reconstruction but are pushed apart when the `(m+1)`th
delay coordinate is added. As `m` increases, the FNN fraction decays; m*
is the smallest dimension where the fraction reaches the noise floor.

**Two tests (both required to be passed for a neighbor to be considered
"true"):**

- **R-criterion** (relative): the added (m+1)th coordinate must not
  increase the distance by more than a factor of R_tol = 15.
- **A-criterion** (absolute): the new total distance must not exceed
  A_tol·σ(x) with A_tol = 2.

### 3.3 Why this is "Takens-sufficient" and not "Takens-equivalent"

The dissertation does not claim that the reservoir literally *implements*
the delay reconstruction. The claim is **architectural sufficiency:** the
reservoir's state space has more than enough dimensions to accommodate the
latent attractor that the ERPs are sampling. By Takens, *some* nonlinear
function of the reservoir's trajectory exists that recovers the latent
state; the downstream readout learns that function from data.

> *AUTHOR ADDS: a one-paragraph statement on the limitations — Takens is
> an existence theorem, not a learnability theorem. The dissertation pairs
> Takens-sufficiency (architectural) with the driven Lyapunov measurement
> from Experiment A (the reservoir actually contracts onto an
> input-driven manifold) to make the operating claim concrete.*

---

## 4. Q&A backup — anticipated committee questions

> *AUTHOR WRITES HERE — for each question, one paragraph of the author's
> own answer.*

- *Why τ = 5? Did the choice rig the answer?*
  (See rawB_2c — m* is similar at τ = 3, 5, 10; the result is robust.)

- *Why Kennel-Brown rather than Cao's E1 statistic?*
  (Kennel's two-test FNN gives a clean threshold-crossing m*; Cao's
  E1(m) requires inspecting a plateau and is more subjective for
  defense-quality reporting.)

- *Did you compute m* per-channel or pooled?*
  (Per (trial, channel). See rawB_2d for the per-trial m* histogram. The
  population median is what the analysis slide reports.)

- *The ERPs are 256 samples long. Isn't that too short for a reliable
  FNN at higher m?*
  (For m = 1..8 and τ = 5, the embedding uses ~216–256 points, plenty
  for nearest-neighbor estimation. The result plateaus well below m = 8.)

- *What about the reservoir's effective dimension — is N_PCA = 64 a fair
  comparison?*
  (The post-PCA dim is the dimensionality of the feature vector consumed
  by the readout. Comparison against the raw N = 256 reservoir state
  gives an even larger sufficiency margin.)

---

## 5. Production log

This section is auto-populated by `make_experiment_b_figures.py` after a
successful run.

> *(Automatic — see script stdout for timestamps and the measured
> m* values.)*
