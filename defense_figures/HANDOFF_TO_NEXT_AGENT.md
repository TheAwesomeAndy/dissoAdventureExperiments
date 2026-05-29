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

## 0.1 Branch state

**Done.** Defense-figure artifacts (Experiments A–D, scaffold K/F/J,
round-2 TB/MR/IP/OQ/RC, Figure G, AM, AC, the Q&A closing slide, and
the assembled deck via `build_deck.py`) are on `main`. The committee
can inspect the default branch directly.

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
| **G** | Failure gallery | `defense_figures/figure_G_failure_gallery/make_figure_G.py` | `outputs/figG_failure_gallery.pdf` |
| **AM** | Dissertation anchor map (+ backup source index) | `defense_figures/figure_AM_anchor_map/make_figure_AM.py` | `outputs/figAM_anchor_map.pdf`, `outputs/figAM_source_index.pdf` |
| **AC** | Research provenance and assistance | `defense_figures/figure_AC_research_provenance/make_figure_AC.py` | `outputs/figAC_research_provenance.pdf` |
| **QA** | Closing slide ("Questions") | `defense_figures/figure_QA/make_figure_QA.py` | `outputs/figQA_questions.pdf` |

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

## 2. Argument-closure status (not coverage)

The committee standard (§0) is the chain *question → observable →
null/baseline → measured result → failure or constraint → revised
experimental decision → bounded contribution*. **Artifact existence**
means a PDF was generated. **Argument closure** means the chain is
complete for the claim that artifact carries.

The two columns are not the same.

| Item | Artifact exists? | Argument closure status | Blocker to closure |
|---|---|---|---|
| **Exp A** — autonomous ρ(W) → driven λ₁ | yes | per-experiment closed (post-rollback) | companion notes still skeleton |
| **Exp B** — FNN-measured m\* relative to reservoir capacity | yes | per-experiment closed (post-rollback) | companion notes still skeleton |
| **Exp C** — measured MC regime vs a-priori β | yes | per-experiment closed | companion notes still skeleton |
| **Exp D** — per-trial channel-permutation null (stimulus class) | yes | per-experiment closed (post-rollback); clinical-label version pending CSV | companion notes still skeleton; not a clinical-disorder result |
| **Failure narrative across the deck** (Figure G) | yes | closed (four documented pivots) | — |
| **AI-assistance disclosure** (Figure AC) | yes | closed (one global provenance slide, author-authorized wording) | — |
| **Dissertation anchor map** (Figure AM + backup index) | yes | closed (chapter-level granularity per author direction) | — |
| **Assembled deck PDF in slide order** | yes | closed (15-slide main + 16-page appendix; `build_deck.py`) | — |
| **Companion-notes spoken claim (A/B/C/D)** | skeletons exist | **not closed** | author-voice content missing — see §3.3 |
| **K** opening questions / **IP** lineage / **MR** refusals / **OQ** open questions / **J** contributions | yes | author-confirmation pending | drafted, not author-voiced |
| **F** theorem scaffold / **TB** theoretical bounds | yes | closed (theorems are documentary; TB reads bounds against measurements) | — |
| **RC** ESP-as-motion animation | yes (MP4 + GIF + PDF poster) | closed (visualization is concrete) | — |

The framing "strong coverage on N of M pillars" — used by the previous
agent earlier in this session — is retired (see §4.0). It conflated
artifact existence with argument closure and produced an
over-confident summary. The table above replaces it.

**Operational summary.** The measurement-driven core (A, B, C, D) is
per-experiment closed after the audit rollback. The deck is assembled
(`ARSPI_Net_Defense_Main.pdf`, 15 slides; `ARSPI_Net_Defense_Appendix.pdf`,
16 pages of backup material). The remaining argument-closure gap is
the author-voiced companion notes — currently skeletons. See §3.

---

## 3. Critical gaps — concrete tasks for the next agent

### 3.0 Merge to main

**Done.** See §0.1.

### 3.1 Figure G — Failure Gallery

**Done.** Four panels drawn from documented pivots: Exp A
(autonomous ρ(W) → driven Lyapunov), Exp C (a-priori β = 0.05 →
measured MC regime), Exp D (single global channel permutation →
per-trial channel permutation), Exp D (unavailable clinical labels
→ stimulus-class methodological demonstration). Wording was
author-confirmed and polished before merge. See
`defense_figures/figure_G_failure_gallery/`.

### 3.2 AI-assistance disclosure — Figure AC

**Done.** One global slide titled "Research Provenance and
Assistance" (NOT "AI Disclosure"). Three blocks (Author-owned /
AI-assisted / Verification) above an author-authorized
provenance-statement paragraph. No per-figure AI labels elsewhere
in the deck. Placed at deck position 14 (after OQ, before Q&A).
See `defense_figures/figure_AC_research_provenance/`.

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

### 3.4 Dissertation anchor map — Figure AM

**Done.** Main-deck slide is the 7-row chapter-level table; backup
appendix is the 14-row Figure-to-Source Index. Section and page
columns intentionally omitted (LaTeX-edit brittleness). See
`defense_figures/figure_AM_anchor_map/`.

### 3.5 Assemble the deck — `build_deck.py`

**Done.** `defense_figures/build_deck.py` concatenates the 15
main-deck PDFs into `outputs/ARSPI_Net_Defense_Main.pdf` and the
16 backup PDFs into `outputs/ARSPI_Net_Defense_Appendix.pdf`. The
script aborts loudly on any missing main-deck PDF and writes
`outputs/deck_manifest.json` recording the exact slide order and
source paths. End-to-end read-through is still pending — the
committee-facing rehearsal pass is an author task.

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
