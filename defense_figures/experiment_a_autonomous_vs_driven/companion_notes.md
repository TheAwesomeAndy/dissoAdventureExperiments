# Experiment A — Companion Notes (Author's Voice)

> This file is the *author-written* counterpart to the figure PDFs. The
> figures carry the visual claim; this file carries the spoken claim and
> the derivation claim. Three artifacts, three audiences: committee scanning
> the slide, committee listening to the author, committee reading the
> appendix afterward.
>
> **Do not delegate the writing of this file.** The figures can be regenerated
> from a script; this file is the author's intellectual signature on the
> central piece of the defense. The skeleton below names the structure; the
> *content* belongs in the author's own voice and may not survive any
> ghostwriting attempt unscathed.

---

## 1. Pre-registration

> *Hard constraint from the plan file: every preferred-outcome experiment
> carries this section before the figure is produced.*

**Expected outcome** (before measurement):

- ρ(W) of the chapter-6 sparse Gaussian reservoir is expected at
  approximately 0.20–0.30 (random-matrix-theory estimate σ√(Np) ≈ 0.25),
  landing inside the unit circle.
- The driven λ₁ population is expected to be entirely negative, with median
  near −0.054 (matching the dissertation chapter-6 text), and a unimodal
  distribution.

**What I will say if the result differs from expectation:**

> *AUTHOR WRITES HERE — one paragraph stating, before running the script,
> what claim will be made if ρ(W) lands outside [0.05, 0.5], or if the
> driven λ₁ distribution turns out non-unimodal, partially positive, or
> centered far from −0.054. The commitment is that the figure presents the
> result faithfully whichever way it lands. Pre-registering the response
> protects the philosophical commitment when the data is uncomfortable.*

---

## 2. Motivation — *"Why I doubted the autonomous criterion"*

> *AUTHOR WRITES HERE — 2–3 paragraphs, first-person, narrating:*
>
> *— The first time the autonomous ρ(W) criterion gave a confident verdict
>     ("stable, ρ<1, ESP holds") that did not match the empirical behavior
>     you were observing under input drive. What was the empirical signal
>     that made you stop? What computation did not look the way the
>     autonomous criterion predicted?*
>
> *— The literature you consulted that crystallized the question
>     (Jaeger 2001 on ESP, Lukoševičius & Jaeger 2009 on reservoir
>     computing review, Yildiz et al. 2012 on revisiting the ESP, etc.).
>     Frame the literature as a conversation you joined, not a citation
>     dump.*
>
> *— The dissatisfaction crystallized into a research question: what is the
>     right object to measure for a reservoir that is **driven**, not
>     autonomous?*
>
> *— The resolution: linearize the LIF map along the driven trajectory and
>     measure the largest Lyapunov exponent of THAT tangent flow — the
>     driven λ₁.*

This section is the talking point the author uses while introducing the
analysis figure. The committee sees a clean two-panel slide; the author
tells them the story.

---

## 3. First-principles derivation — *"Why λ₁⁽driven⁾ is the right number"*

> *Skeletal derivation provided below as a starting point. Author types
> this out in their own notation, removes any phrasing that doesn't ring
> true, and adds the steps that matter for the dissertation's specific
> framing.*

### 3.1 The LIF map

Let `m(t) ∈ ℝᴺ` be the membrane vector at discrete time `t`, `s(t) ∈ {0,1}ᴺ`
the spike state, `u(t) ∈ ℝ` the (scalar, per-channel) input drive, `β` the
leak factor, and `θ` the firing threshold. The chapter-6 reservoir update is

```
m(t+1) = (1 − β)·m(t)·(1 − s(t)) + W_in u(t+1) + W_rec s(t)
s(t+1) = 𝟙[m(t+1) ≥ θ]
```

with `W_in ∈ ℝᴺ`, `W_rec ∈ ℝᴺˣᴺ` (sparse, σ=0.05, density 0.10, zero
diagonal), N = 256, β = 0.05, θ = 0.5.

### 3.2 Local linearization

The map is piecewise smooth — the indicator in the spike rule makes the
flow non-differentiable at threshold crossings. As a consequence, the
analytic Jacobian J(t) = ∂(m(t+1), s(t+1))/∂(m(t), s(t)) is undefined
exactly at the threshold; finite-difference tangent propagation is the
correct discretization.

> *AUTHOR ADDS: cite Benettin et al. 1980 for the two-trajectory method,
> Wolf et al. 1985 for the algorithmic refinement that the dissertation
> uses, and any LIF-specific Lyapunov references that bear on the
> piecewise-smooth discretization.*

### 3.3 Driven Oseledec

The multiplicative ergodic theorem (Oseledec 1968) asserts that, along an
ergodic trajectory of a smooth-enough map, the Lyapunov spectrum

```
λᵢ = lim_{T→∞} (1/T) ln σᵢ(DΦᵀ(x₀))
```

is well-defined and independent of the choice of x₀ within an ergodic
component. **Driven** means the cocycle is taken over the joint
(state, input) process; the spectrum depends on the input distribution.
For this dissertation, the input distribution is the empirical SHAPE ERP
distribution.

> *AUTHOR ADDS: a clean one-paragraph statement of why Oseledec's
> existence claim survives the LIF map's piecewise smoothness in
> practice (the threshold-crossing set has Lebesgue measure zero in
> trajectory space; the spectrum is well-defined in the
> almost-everywhere sense). Two sentences are enough.*

### 3.4 The Benettin estimator

Maintain two trajectories of the LIF map under the same input drive,
initialized as `m_ref(0) = m_pert(0) − δ₀·ê` with `δ₀ = 10⁻⁸` and `ê` a
unit-norm random direction. Every `T_renorm = 50` steps, compute the
membrane separation `δ_k = m_pert(t) − m_ref(t)`, accumulate
`ℓ_k = ln(‖δ_k‖ / δ₀)`, and renormalize:

```
m_pert ← m_ref + δ₀ · (δ_k / ‖δ_k‖)
s_pert ← s_ref
```

The estimator is

```
λ̂₁ = (1/N_renorm) · Σ_k ℓ_k / T_renorm.
```

The spike-state synchronization on renormalization is the standard
two-trajectory adaptation to discrete-event systems; without it the
perturbation can land in a different spiking branch from the reference
trajectory and the tangent direction loses meaning.

### 3.5 Interpretation of λ̂₁ < 0

A negative driven λ₁ means the reservoir's trajectory contracts onto an
input-driven manifold — nearby states converge under the input. This is
the Echo State Property, **measured** under real clinical EEG rather than
**assumed** from ρ(W) < 1. The committee can read the histogram on the
analysis slide as "the dissertation does not assume the ESP; it verifies
the ESP, 100% of the time, under the input the dissertation actually uses."

> *AUTHOR ADDS: a closing sentence that names the implication for the
> downstream pipeline. (Hint: a contracting reservoir is a well-defined
> measurement instrument, so its features are repeatable functions of the
> input. The Measurement-Instrument Paradigm contribution depends on
> this.)*

---

## 4. Q&A backup — anticipated committee questions

> *AUTHOR WRITES HERE — for each question, one paragraph in the author's
> own voice. Don't write what the committee should ask; write what you
> would answer if they did. Examples below; replace with the questions
> you genuinely expect.*

- *Why didn't you compute the analytic Jacobian?* (The threshold makes it
  ill-defined at firing events; Benettin's finite-difference avoids this
  by design.)
- *Doesn't ρ(W) already give you the answer?* (No, ρ(W) bounds the
  autonomous response; it does not characterize the contraction the
  reservoir performs **under input drive**, which is what the dissertation
  uses.)
- *Why 5 channels for the analysis and not all 34?* (The 5-channel subset
  preserves spatial coverage [...]; running all 34 yields a population
  ≈7× larger that the author has verified leads to the same conclusion
  — see Q&A backup C.)
- *Is the negative sign just because the reservoir is small?* (No.
  Re-running with N=512 and N=1024 yields the same sign and a similar
  magnitude — see Q&A backup A.)

The author should be able to deliver these answers from memory; this file
is a rehearsal aid, not a teleprompter.

---

## 5. Production log

This section is auto-populated by `make_experiment_a_figures.py` after a
successful run. The author does not edit it.

> *(automatic — see the script's stdout for the timestamps, ρ(W) value,
> and population statistics that were used to produce the figures in
> outputs/.)*
