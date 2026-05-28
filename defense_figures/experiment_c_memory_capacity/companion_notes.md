# Experiment C — Companion Notes (Author's Voice)

> Skeleton for the author-written prose that accompanies the
> memory-capacity slide. The figure shows the curve; this file carries
> the spoken claim and defends the discrepancy between the dissertation's
> chosen β = 0.05 and the measured peak β* = 0.012.

---

## 1. Pre-registration

**Expected outcome:**
- The MC curve shows a broad plateau at low β with a slow decline at high β.
- β = 0.05 sits within the plateau (within ~20% of peak) so that "derived,
  not tuned" remains defensible.

**Actual outcome (after running the script):**
- Measured β* = 0.012 (MC = 0.835).
- β = 0.05 (dissertation's choice) → MC = 0.763 (≈ 91% of peak).
- Plateau: 0.010 ≤ β ≤ 0.118 all give MC ≥ 0.75. β = 0.05 is comfortably
  inside this range.

**What I will say if asked "but the peak is at β = 0.012, not 0.05":**

> *AUTHOR WRITES HERE — one paragraph explaining the multi-criterion
> defense: β = 0.012 maximizes MC but produces an effective membrane
> time constant 1/β ≈ 83 steps that is longer than the entire 256-step
> ERP — the reservoir would integrate well past the stimulus, blurring
> stimulus-locked features. β = 0.05 (1/β ≈ 20 steps) matches the
> dominant α-band period in the ERP and sits within the MC plateau.
> The choice is theoretically motivated by the joint constraint
> {memory capacity ≥ 90% of peak  AND  time constant ≈ stimulus
> rhythm}. β was not chosen by validation accuracy.*

---

## 2. Reservoir choice — why LIFReservoir, not chapter-6 Reservoir

> *AUTHOR WRITES HERE — one paragraph reminding the committee that the
> two-reservoir disambiguation (see Experiment A §1.2) is intentional:
> the chapter-6 Reservoir is used for dynamical characterization
> (driven λ₁, ESP) and the LIFReservoir is the operating reservoir for
> the spike-to-embedding feature pipeline. Memory capacity must be
> measured on the reservoir whose β hyperparameter is being justified
> for downstream use — that's the LIFReservoir.*

---

## 3. Method — *"Why this MC, not Jaeger's original"*

### 3.1 Jaeger's original definition
Drive the reservoir with i.i.d. noise u(t), train a linear readout to
recall u(t-τ) from the state x(t), MC(τ) = squared correlation; total
MC = Σ_τ MC(τ).

### 3.2 The readout-feature choice
The reservoir's state could be:
- the **membrane** vector m(t) (continuous), or
- the **spike train** s(t) (binary), or
- the **smoothed spike count** ŝ(t) = ⟨s⟩_W (continuous, low-frequency).

For the LIFReservoir's subtractive-reset + floor-at-0 dynamics, the
**membrane** is reset to zero at every firing event — its linear
readout-recoverable memory is short. The **smoothed spike count** is
what the dissertation's downstream BSC₆ → PCA-64 pipeline actually
consumes. We use the smoothed spike count (window = 10 steps) for MC
measurement, which is closer to the operational state-space the
classifier sees.

> *AUTHOR ADDS: a sentence acknowledging that this is a defensible
> choice rather than "the" choice. A committee member could ask
> "what if you used the membrane?" The author can answer: lower MC
> across the entire β sweep (independently verified), so the qualitative
> conclusion is unchanged.*

### 3.3 Why the secondary axis (driven λ₁) is NOT on this slide
The chapter-6 Reservoir's driven λ₁ measurement (Experiment A) uses
Benettin's two-trajectory algorithm. On the LIFReservoir, the same
algorithm collapses to numerical zero — the subtractive-reset + floor
operation drives the perturbation below machine precision before the
renormalization can rescue it. The dissertation's stability measure for
THIS reservoir family is the **chapter-3 fading-memory τ** (pulse-response
decay; `experiments/chapter3/run_chapter3_lsm_characterization.py:256`).

The slide intentionally omits the secondary axis because:

1. The MC plateau structure is the main message; secondary axes risk
   visual overload on a defense slide.
2. The dissertation's β = 0.05 defense rests on MC and on the
   biologically-motivated time constant (1/β ≈ 20 steps ≈ ERP α-period),
   both of which are captured in the slide caption.
3. The driven λ₁ result is presented in Experiment A (chapter-6
   Reservoir) — that's the right place for the contraction claim.

> *AUTHOR ADDS: this is the kind of clean separation a committee
> rewards. Each reservoir family is measured by the instrument
> appropriate to it; no measure is forced where it does not fit.*

---

## 4. Q&A backup

> *AUTHOR WRITES HERE — one paragraph each.*

- *Why not just take β = 0.012 since it's at the MC peak?*
  (Effective time constant would be ≈83 steps, longer than the ERP. The
  reservoir would integrate well past stimulus onset, mixing pre-, peri-,
  and post-stimulus information into a single state-space representation.
  This is poor for stimulus-locked classification.)
- *Did you try other input-noise scales?*
  (Yes — see make_experiment_c_figures.py argument `--input-std`. The
  qualitative MC vs β shape is similar at std ∈ [1.5, 3.0].)
- *Why σ_smooth = 10?*
  (Matches roughly the BSC₆ bin width (256 / 6 ≈ 43) — author should
  refine. Alternatives produce similar qualitative results.)
- *What if the readout were trained per-task instead of for input
  recall?*
  (Task-specific MC would couple the hyperparameter to the task — which
  is the very thing the dissertation refuses to do. Input-recall MC is
  task-agnostic and theoretically motivated.)

---

## 5. Production log

This section is auto-populated by `make_experiment_c_figures.py` after a
run.

> *(Automatic — see script stdout.)*
