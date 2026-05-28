# Handoff to the Next AI Agent — Defense-Deck Audit & Self-Reflection

> **Read this first.** This document is a self-reflective handoff written
> by the AI agent that built rounds 1 and 2 of the defense deck
> (Experiments A/B/C/D + Figures K/F/J + Round-2 figures TB/MR/IP/OQ + RC
> contraction animation). It is addressed to the **next AI agent** picking
> up this work, with the author looking over its shoulder.
>
> **Purpose.** Surface what is built, what is missing, and what I should
> have done differently — in enough detail that the next agent can act
> without re-deriving the context.

## 0. The PhD argument (use this framing, not the one I wrote first)

The dissertation's argument is **not** "I pursued truth." It is, in the
author's own words after auditing this work:

> *"The dissertation repeatedly replaced default assumptions with
> measured operators: autonomous stability with driven Lyapunov
> contraction (Exp A), architectural fashion with FNN-measured
> embedding-dimension sufficiency relative to reservoir capacity
> (Exp B), nominal parameter choice with memory-capacity regime
> analysis (Exp C), and performance claims with spatial null testing
> (Exp D)."*

That sentence is the deck. Every figure must serve it. Slides that read
as therapy, manifesto, or AI-generated self-justification do not serve
the argument and should be deleted.

The committee standard for "complete" is **not** "I have a PDF for each
idea." It is a chain:

1. a question,
2. a mathematical observable,
3. a null or baseline,
4. a measured result,
5. a failure or constraint,
6. a revised experimental decision,
7. a bounded contribution.

**Artifact coverage is not argument closure.** The previous agent
(me) blurred those two and called the deck "8 of 9 pillars covered."
That phrasing should be retired.

## 0.1 Branch state — operational defect (priority 0)

All defense-figure artifacts live on `claude/magical-volta-82g4t`. The
`main` branch does not carry them. A committee-facing workflow cannot
depend on hidden / non-default commit state.

**Action required (user-authorized):** merge `claude/magical-volta-82g4t`
into `main` (or clearly publish which ref the artifacts live on, and
update all dissertation documentation to reference that ref).

This is destructive enough (changes the default branch's contents) that
the previous agent did not perform it without explicit user
authorization. The next agent should request that authorization before
merging, but should treat the unmerged state as a defect, not a
neutrality.

---

## 1. What was built

### 1.1 Experiments (data + figures + companion-notes skeletons)

Claims here are stated **precisely** (after audit correction — earlier
versions overclaimed about Takens and clinical labels).

| Exp | Defensible claim | Script | Outputs |
|---|---|---|---|
| **A** | Autonomous spectral-radius criteria are insufficient unless paired with a measurement under input drive. ρ(W) = 0.2647549015; driven λ₁ measured on 3,165 trajectories. | `defense_figures/experiment_a_autonomous_vs_driven/make_experiment_a_figures.py` | `outputs/rawA_1{a..d}_*.pdf`, `analysisA_1e_autonomous_vs_driven.pdf`, `experiment_a_data.csv`, `_benettin_cache.npz` |
| **B** | FNN provides an empirical bound on embedding dimension needed to reconstruct measured ERP trajectories via delay coordinates; m* ≪ 64 (post-PCA) ≪ 256 (raw). **Takens-motivated question, FNN-measured answer.** Does NOT claim Takens "guarantees" attractor reconstruction. | `defense_figures/experiment_b_takens_dimension/make_experiment_b_figures.py` | `outputs/rawB_2{a..d}_*.pdf`, `analysisB_2e_takens_dimension.pdf`, `experiment_b_data.csv`, `_fnn_cache.npz` |
| **C** | β = 0.05 (MC ≈ 0.763) sits inside the measured MC plateau but is **not** at the measured peak (β* ≈ 0.012, MC ≈ 0.835). Operating-regime mismatch is real and defended by biological time-constant argument. | `defense_figures/experiment_c_memory_capacity/make_experiment_c_figures.py` | `outputs/rawC_3{a..c}_*.pdf`, `analysisC_3d_memory_capacity_peak.pdf`, `experiment_c_data.csv`, `_mc_cache.npz` |
| **D** | **Stimulus-class** classification survives per-trial channel-permutation null (500 perms × 5 CV folds). **Not** a clinical-disorder validation. CLI flag swaps to disorder labels when CSV is provided. | `defense_figures/experiment_d_channel_permutation/make_experiment_d_figures.py` | `outputs/rawD_4{a..c}_*.pdf`, `analysisD_4d_permutation_nulls.pdf`, `experiment_d_data.csv`, `_perm_cache.npz` |

Each experiment ships with a `companion_notes.md` that contains the
pre-registration block, motivation, derivation, Q&A backup, and
production log — **as a skeleton with `AUTHOR WRITES HERE` markers, not
as finished author-voiced prose.**

### 1.2 Scaffold figures (round 1)

| Figure | Purpose | Script | Output |
|---|---|---|---|
| **K** | Opening questions / philosophical frame | `defense_figures/figure_K_opening_questions/make_figure_K.py` | `outputs/figK_opening_questions.pdf` |
| **F** | Theorem scaffold | `defense_figures/figure_F_theorem_scaffold/make_figure_F.py` | `outputs/figF_theorem_scaffold.pdf` |
| **J** | Five named contributions (POC, MIP, Spike-to-Embedding Pipeline, Layer Ablation Methodology, Centered Baseline Comparison) | `defense_figures/figure_J_contributions/make_figure_J.py` | `outputs/figJ_contributions.pdf` |
| **G** | Failure gallery | **NOT BUILT** | **NOT BUILT** |

### 1.3 Round-2 figures (philosophical depth)

| Figure | Purpose | Script | Output |
|---|---|---|---|
| **TB** | Theoretical bounds vs measurement for A/B/C | `defense_figures/figure_TB_theoretical_bounds/make_figure_TB.py` | `outputs/analysisTB_5a_bounds_vs_measurement.pdf`, `experiment_tb_data.csv` |
| **MR** | Methodological refusals (rigor commitments) | `defense_figures/figure_MR_methodological_refusals/make_figure_MR.py` | `outputs/figMR_methodological_refusals.pdf` |
| **IP** | Intellectual provenance (scholarly lineage) | `defense_figures/figure_IP_intellectual_provenance/make_figure_IP.py` | `outputs/figIP_intellectual_provenance.pdf` |
| **OQ** | Open questions (closing) | `defense_figures/figure_OQ_open_questions/make_figure_OQ.py` | `outputs/figOQ_open_questions.pdf` |
| **RC** | Contraction animation (ESP made visceral) | `defense_figures/experiment_a_autonomous_vs_driven/make_animation.py` | `outputs/rawA_1f_contraction_animation.{mp4,gif,pdf}` |

### 1.4 Source data and code (read-only references)

- `chapter6Experiments/results/ch6_exp1_full.pkl` — the 4,220 λ₁ values + X_ds tensor (the data backbone).
- `chapter6Experiments/run_chapter6_exp1_esp.py` — the chapter-6 `Reservoir` class definition.
- `experiments/chapter3/run_chapter3_lsm_characterization.py` — the LIFReservoir.
- `experiments/chapter3/animate_lsm_dynamics.py` — animation infrastructure that the RC animation extends.

---

## 2. Coverage map: the author's stated pillars

The author stated nine pillars for what the committee must walk out
understanding. The coverage as of this handoff:

| # | Pillar | Carried by | Status |
|---|---|---|---|
| 1 | Rigorous research / dissertation | Exp A/B/C/D + MR + F + TB | **Covered.** Numbers, methods, refusals, theorems all visualized. |
| 2 | "AI was not used" | — | **INTEGRITY TENSION** (see §4). |
| 3 | Deep, influential, research-motivated questions | K (4 questions) + IP (6 traditions) | **Covered**, contingent on author refining wording. |
| 4 | Not just innovation for its own sake | MR (5 refusals) + J (contributions as gifts, not features) | **Covered.** |
| 5 | Not random / superficial; scientifically motivated | IP (each question traced to a scholarly tradition) + F (theorem-grounded) | **Covered.** |
| 6 | PhD: pursued philosophy, theory, novel work, truth | K + MR + IP + OQ (philosophy); F + TB (theory); MR (truth); J (novelty) | **Covered.** |
| 7 | Failed results embraced as engine of rigor | Figure G (NOT BUILT) — only Exp C companion-notes addresses one specific case (β = 0.05 vs β* = 0.012) | **MISSING.** |
| 8 | New truth, new innovation, new theory | J (5 named contributions) + TB (each measurement reads against a theorem) | **Covered.** |
| 9 | Deserving of Doctorate of Philosophy | Cumulative; OQ closes with "asking better questions than I can answer" | **Covered**, contingent on G existing. |

**Net.** 7 of 9 strongly covered. 1 (Failure Gallery) missing. 1 (AI
usage) is a defense-integrity question I never flagged.

---

## 3. Critical gaps — concrete tasks for the next agent

### 3.0 Merge to main (priority 0 — operational defect)

See §0.1. All defense-figure work is on `claude/magical-volta-82g4t`;
`main` does not carry it. Request user authorization, then either merge
or open a PR. Until this is done, anyone inspecting `main` will see
only Experiment A as planned and B/C/D/G/J as TBD — which is the
opposite of the true state.

### 3.1 Build Figure G — Failure Gallery (HIGHEST PRIORITY among slides)

**Why it matters.** The author's stated goal explicitly says: *"I deep
dove down to failed results that motivated me to change my experimental
rigor throughout. I embraced the failed results as motivations as to
what to try next and to learn from them."* The deck currently has no
visualization of this. Without G, the deck has questions, theory,
measurements, lineage, refusals, contributions, open questions — but
**no failure narrative.**

**What G should contain** (3–5 case panels, each with: *what I tried* →
*what failed and how I saw it* → *what I changed and why*). Candidate
cases drawn from the dissertation that the next agent can draft from:

1. **β-selection pivot (Exp C / Chapter 6).** Initial expectation:
   the β = 0.05 choice was theoretically derived. Actual: the
   *measured* MC peak is at β* = 0.012, not 0.05. Pivot: rather than
   silently re-tuning, the author kept β = 0.05 anchored in a
   biological time-constant argument and built the Propagation
   Operating Characteristic to defend the choice. Failure → diagnosis
   → reframing.

2. **Channel-permutation null (Exp D / Chapter 5).** Initial null
   (parametric or per-experiment permutation) was too lenient and
   wouldn't have caught spatial-feature leakage. The author switched
   to per-trial channel permutation — strictest spatial null — and
   the classification claim survived. Pivot toward stricter rigor.

3. **Reservoir-architecture pivot (Chapter 3 vs Chapter 5).** If the
   dissertation supersedes an earlier reservoir choice (LIFReservoir
   architecture, BSC₆ encoding parameters, PCA dimension) in favor
   of the chapter-6 reservoir — that's a documented pivot. Show it.

4. **Disorder-label provenance (Exp D caveat).** The session's pickle
   did not contain per-disorder labels. The author held the
   methodology demonstration on stimulus class and refused to fake
   or approximate disorder labels. A small failure (data
   incompleteness) with a rigor-preserving response.

5. **Subject-level vs trial-level CV.** Author refused trial-level
   CV (inflated by 10–15 pp) for subject-level StratifiedGroupKFold.
   This is partly in MR, but if there was a moment in the
   dissertation where trial-level numbers were tried and rejected,
   that *moment* belongs in G.

**Recommended implementation.** Treat this like the MR/IP/OQ figures:
fully draft from the dissertation context, mark each row with `[author
confirm]` where the agent is uncertain about specifics, and let the
author refine. Same `_style.py` template, two- or three-column layout.
Target file: `defense_figures/figure_G_failure_gallery/make_figure_G.py`
and `outputs/figG_failure_gallery.pdf`. Add it to the slide order
between Dive 3 and J.

### 3.2 Build an Acknowledgments / Assistance disclosure slide

**Why it matters.** The author's goal includes "AI was not used." This
is true of the **dissertation research, chapters, and analyses** —
those are the author's. It is **not** true of the **defense slide
typesetting**, which was AI-assisted on numbers and content the author
produced. Without disclosure, the claim is risky. A one-slide
acknowledgments page that says, in the author's own voice:

> *"The research, derivations, analyses, and chapter prose of this
> dissertation are mine. The defense-figure typesetting was assisted by
> [tool name] working from numbers, claims, and concepts I produced and
> verified. Every figure was reviewed and approved by me before this
> defense."*

…protects the author, satisfies committee due diligence, and removes
the integrity tension. Target file:
`defense_figures/figure_AC_acknowledgments/make_figure_AC.py`. The
author should write the exact wording.

### 3.3 Help the author finish the companion notes

Each experiment's `companion_notes.md` has `AUTHOR WRITES HERE`
markers. The next agent can:

- **Draft candidate prose** based on the dissertation chapters
  (chapter 3 for B's reservoir grounding, chapter 6 for A and C, chapter
  5 for D), marked clearly as "DRAFT — author rewrites in own voice."
- **NOT** publish drafted prose as finished — leave the rewrite to
  the author.
- Verify the pre-registration sections, derivation steps, and Q&A
  backups are consistent with what the figures actually show.

### 3.4 Build a dissertation anchor map

A single page mapping each defense figure to its chapter, section, and
page range in the dissertation. The committee will ask. Target:
`defense_figures/figure_AM_anchor_map/make_figure_AM.py`. Requires the
author to provide page numbers; the agent assembles.

### 3.5 Assemble the deck and verify the narrative

The slide order is planned but never assembled and read through.

```
defense_figures/build_deck.py   # NEW — proposed
```

This script would:
1. Concatenate the PDFs in the proposed order (using `pypdf` or
   `pdfunite`).
2. Produce a single `defense_deck_draft.pdf`.
3. Print a contents page with figure → slide-number mapping.
4. Optionally, generate a "rehearsal mode" PDF with companion-notes
   text on facing pages.

Then read the assembled deck end-to-end. Check: does the narrative
flow? Does K motivate F? Does F motivate the experiments? Does TB
close the theoretical loop? Does G land before J? Does OQ close.

### 3.6 Verify the "AI was not used" claim is precisely stated everywhere

Search the dissertation prose, the README, and any author-facing
content for the phrase "AI was not used" or similar. If it appears,
**replace it with a precise version**:

> "AI was not used to produce the dissertation's research, derivations,
> analyses, or written chapters. Defense-figure typesetting was AI-
> assisted on numbers and concepts the author produced and verified."

This protects the author. The next agent should treat any blanket
"no AI" claim as a defect to surface, not a fact to preserve.

---

## 4. Self-reflection — where the prior agent (me) fell short

### 4.0 I confused artifact coverage with argument closure

I declared "8 of 9 pillars covered" based on the existence of PDFs. A
PhD defense is not a folder of PDFs. The argument chain (question →
observable → null → result → failure → revised decision → bounded
contribution) is the standard. The artifacts have pieces of that chain
but it is not assembled into a tested defense argument. The phrasing
"strong coverage on N of M pillars" should not be used again.

### 4.1 I overclaimed in Exp B and Exp D and had to roll it back

- **Exp B docstring originally said** "Takens' theorem mathematically
  guarantees that a sufficiently rich driven dynamical system can
  reconstruct the latent attractor." This is false — Takens provides
  conditions under which a generic delay-embedding observation is
  topologically equivalent to the underlying attractor, but does not
  guarantee that any specific reservoir reconstructs the affective
  attractor. Corrected to "Takens-motivated question, FNN-measured
  answer."
- **Exp B figure title originally said** "The reservoir's capacity
  exceeds the latent attractor's Takens dimension." Wrong — the FNN
  estimate is an empirical bound, not the Takens dimension itself.
  Corrected to "FNN-estimated embedding dimension is small relative
  to the reservoir's state space."
- **Exp D docstring originally said** "My clinical claims are not
  data-mined accidents." The analysis is on stimulus class, not
  clinical disorder. Corrected to make the stimulus-class scope
  explicit.

**Lesson for the next agent.** When writing figure titles and
docstrings, be paranoid about the difference between (a) the theorem
that motivated the question and (b) the measurement that was actually
performed. Use the most technically restrained wording the data
supports. The committee will notice the difference.

### 4.2 I treated "needs author input" as a stop sign instead of a draft request

The original plan named G as required and said it needed author input
because "only the author knows which attempts truly failed." I took
that literally and never drafted G. But the dissertation contains
**named pivots that are public record in the chapters and commits**:
the β-selection mismatch is in chapter 6's text, the channel-
permutation null choice is in chapter 5, the disorder-label caveat is
in the Exp D companion notes. I could have drafted G from these and
let the author refine — exactly what I did, with consent, for MR/IP/OQ
in round 2.

**Lesson for the next agent.** "Needs author input" ≠ "do not start."
If the dissertation already documents a pivot, draft it; mark the
draft `[author confirm]`; commit. The author refines what's there
faster than they write from scratch.

### 4.3 I never named the AI-usage tension

The author's goal includes "AI was not used." The deck was built with
AI. I should have flagged this in plan mode and offered options
(no-AI rebuild; AI-acknowledged disclosure; partial AI scope).
Instead I silently helped build an AI-assisted deck for a claim of "no
AI" — that's a defense-integrity defect.

**Lesson for the next agent.** When a stated goal is in tension with
the work being requested, surface the tension in `AskUserQuestion`
before doing the work, not after.

### 4.4 I produced figures without producing a defense rehearsal harness

A defense deck is not the same as a folder of PDFs. The deck is the
PDFs **in order, timed, with the speaker watching for narrative
gaps**. I produced the PDFs. I never produced the assembly script,
the timing, or the read-through. Half-built.

**Lesson for the next agent.** Build `defense_figures/build_deck.py`
(§3.5) early. Re-run it after every figure addition. Read the
assembled draft aloud at least once.

### 4.5 I produced no dissertation anchor map

The committee asks "where in the dissertation is this?" within the
first three slides. A single page with figure → chapter/section/page
addresses this. I never proposed it.

**Lesson for the next agent.** Build the anchor map before adding
more figures. It will surface inconsistencies in the figure-to-chapter
mapping that other figures may need to absorb.

### 4.6 I left the companion notes as skeletons

The figures alone are insufficient. The spoken claim is in the
companion notes. Those are templates. Without author-voice content
the defense is figures-only. I should have:

- Drafted candidate prose from the dissertation chapters as a
  starting point (marked as DRAFT).
- Required the author to fill in at least the "Pre-registration:
  what I will say if the result differs" sections **before** the
  figure was claimed complete, because that section is what
  carries the philosophical commitment.

### 4.7 I did not verify slide ordering by assembly and read-through

I planned the slide order in the addendum. I never assembled the
PDFs in that order and read them. The narrative may flow; it may
not. I produced the parts; I did not test the whole.

**Lesson for the next agent.** Trust nothing about narrative flow
until you have run the assembled draft past the author and ideally
past a second reader.

---

## 5. Acceptance checklist for the next agent

Before declaring the defense deck "complete," the next agent should
verify each of these:

- [ ] Figure G (Failure Gallery) exists with 3–5 case panels, drafted
      from the dissertation's documented pivots, marked for author
      refinement.
- [ ] Acknowledgments / disclosure slide exists, in the author's
      voice, precisely stating what AI did and did not do.
- [ ] All four `companion_notes.md` files have author-voice content
      replacing the `AUTHOR WRITES HERE` markers.
- [ ] A dissertation anchor map exists (figure → chapter/section/page).
- [ ] `defense_figures/build_deck.py` exists and assembles the deck in
      the proposed slide order.
- [ ] The assembled draft has been read end-to-end by the author at
      least once.
- [ ] All "AI was not used" claims in the repo and dissertation have
      been replaced with a precise version (§3.6).
- [ ] The deck-level slide order in the plan file
      (`/root/.claude/plans/experiment-a-autonomous-foamy-bear.md`,
      "Updated deck-level slide order" section) is updated to include
      G and the Acknowledgments slide.

---

## 6. Slide order with G and AC added (proposed)

1. Title
2. **AC — Acknowledgments / disclosure** (set the integrity baseline)
3. **K — Opening Questions**
4. **MR — Methodological Refusals**
5. **IP — Intellectual Provenance**
6. **F — Theorem Scaffold**
7. **Dive 1 — Neuromorphic Operator**
   - Exp B (Takens)
   - RC contraction animation (visceral preview)
   - Exp A (Lyapunov)
   - Exp C (Memory capacity)
   - TB (Theoretical bounds vs measurements)
8. **Dive 2 — Embedding** (existing chapter-4 figures)
9. **Dive 3 — Clinical**
   - Existing classification figures
   - Exp D (Channel-permutation null)
10. **G — Failure Gallery** (the journey, with humility)
11. **J — Contributions** (the gift to the field)
12. **OQ — Open Questions** (closing)
13. Q&A backup: derivations, scope/humility, anchor map

---

## 7. Final note from the previous agent

The deck as-of-this-handoff has artifact coverage for the
measurement-driven experiments (A, B, C, D), the philosophical
scaffold (K, F, J), the round-2 figures (TB, MR, IP, OQ), and the RC
animation. It has **not** assembled those artifacts into a tested
defense argument.

The single highest-leverage next move is operational, not creative:
**merge this work to `main` so the repo's default state matches reality**
(§3.0). Until that is done, every other claim in this handoff is
contingent on inspecting a non-default branch.

After that, the highest-leverage **creative** move is for the author to
write 3–5 honest paragraphs about specific pivots in the dissertation
(what was tried, what failed, what was learned, what changed as a
result) and hand them to the next agent as the seed for Figure G. The
candidates listed in §3.1 are reasonable starting points but only the
author knows which ones are real.

Slides whose only function is to perform humility or commitment without
carrying a measurement should be cut. The deck is at its strongest when
it stays close to the measured operators (A's driven Lyapunov, B's
FNN-measured embedding-dim sufficiency, C's MC regime, D's spatial
null) and weakest when it ventures into manifesto. The committee will
read it the same way.
