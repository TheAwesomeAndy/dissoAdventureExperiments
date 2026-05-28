# Timing Map

Per-frame speaking-time allocation for the canonical defense
`ARSPI-Net_Defense.tex` (53 main frames + 42 appendix frames = 95 content
frames; 105 PDF pages including title and section dividers).

**Budget**: 45 minutes presenting + 15--20 minutes Q&A.

**Strategy**: the original 45-frame deck was authored to ~40 minutes
(roughly 53 s/frame average with key-result frames taking 90--120 s and
restated-claim frames taking 25--35 s). The eight inserted audit slides
add ~9 minutes of rigor injection. Net presentation runs to ~49 minutes
**at the unforgiving end**; trim by ~4 minutes via the **cut-line lane**
below to land at 45 even.

---

## Section-level budget

| Part | Theme                                       | Frames | Budget    |
|------|---------------------------------------------|--------|-----------|
| 1    | The Engineering Problem and the Inquiry     | 5      | 3 min 30 s |
| 2    | Architecture and Contributions              | 3      | 2 min 30 s |
| 3    | The Neuromorphic Operator                   | 12     | 12 min     |
| 4    | The Three Operationally Distinct Layers     | 7      | 6 min      |
| 5    | The Graph Propagation Regime                | 9      | 8 min      |
| 6    | Intrinsic Interpretability                  | 10     | 8 min      |
| 7    | Scope, Future, and Close                    | 7      | 5 min      |
|      | **Total**                                   | **53** | **45 min** |

Part-divider pages (`Part 1 ---`, `Part 2 ---`, …) are **0 s** speaking
time; advance through them without comment.

---

## Per-frame allocation

Times are wall-clock seconds; cumulative time shown in **m:ss**. PDF page
numbers refer to `ARSPI-Net_Defense.pdf` (105 pages). The 8 inserted
audit slides are marked **AUDIT**.

### Part 1 --- The Engineering Problem and the Inquiry

| PDF | Frame                                | Sec | Cum.  | Note |
|-----|--------------------------------------|-----|-------|------|
|   3 | The epistemological problem          |  60 | 1:00  | Open strong; one slow read. |
|   4 | The unobserved cortical operator F   |  45 | 1:45  | Equation only — name F. |
|   5 | The inverse problem                  |  45 | 2:30  | Statement of φ. |
|   6 | Why conventional pipelines miss      |  35 | 3:05  | Cut to 25 s if behind. |
|   7 | The governing question               |  25 | 3:30  | Single sentence. |

### Part 2 --- Architecture and Contributions

| PDF | Frame                                | Sec | Cum.  | Note |
|-----|--------------------------------------|-----|-------|------|
|   8 | (Part 2 plain title page)            |   0 | 3:30  | Click through. |
|   9 | Positioning                          |  45 | 4:15  | One sentence per row. |
|  10 | Five engineering contributions       |  60 | 5:15  | Read them out. |
|  11 | The unifying empirical finding       |  45 | 6:00  | Single claim. |

### Part 3 --- The Neuromorphic Operator

| PDF | Frame                                                          | Sec | Cum.  | Note |
|-----|----------------------------------------------------------------|-----|-------|------|
|  13 | The neuromorphic operator                                      |  45 | 6:45  | Reframe LIF as F. |
|  14 | The leaky integrate-and-fire neuron                            |  50 | 7:35  | β introduced here. |
|  15 | Driven contraction                                             |  45 | 8:20  | Theory only. |
|  17 | The contraction measurement, visualized                        |  50 | 9:10  | Caption: N=4,220. |
|  18 | **AUDIT** Defense audit: autonomous ρ(W) vs. driven λ₁         |  90 | 10:40 | Read the N=3,165/N=4,220 reconciliation bullet. |
|  19 | **AUDIT** Defense audit: memory-capacity regime for leak β     |  90 | 12:10 | Defend β = 0.05 as MC plateau. |
|  20 | Temporal coding by design: BSC₆                                |  60 | 13:10 | Cut to 45 s if behind. |
|  21 | The spike-to-embedding pipeline (Contribution 2)               |  50 | 14:00 | Pipeline diagram. |
|  22 | The measurement-instrument paradigm                            |  35 | 14:35 | Three principles. |
|  23 | **AUDIT** Defense audit: FNN-measured embedding dimension      |  75 | 15:50 | Kennel--Brown FNN, not Takens. |
|  24 | The path to silicon                                            |  45 | 16:35 | Cut if behind. |
|  25 | The failure of spatial deep learning                           |  55 | 17:30 | Sets up Part 4. |

### Part 4 --- The Three Operationally Distinct Layers

| PDF | Frame                                                          | Sec | Cum.  | Note |
|-----|----------------------------------------------------------------|-----|-------|------|
|  27 | The three-layer structure                                      |  50 | 18:20 | Layer A/B/C named. |
|  28 | Layer A --- the subject-covariance hinge                       |  55 | 19:15 | Centering preview. |
|  29 | Layer A --- centering is the dominant intervention (Contrib 5) |  60 | 20:15 | Key result; do not cut. |
|  30 | Layer B --- seven named observables                            |  45 | 21:00 | Read 3 of 7. |
|  31 | Layer B --- descriptors are condition-sensitive                |  50 | 21:50 | One example. |
|  32 | Layer B --- the temporal family dominates                      |  50 | 22:40 | One contrast. |
|  33 | Layer B --- the excitability--persistence axis                 |  60 | 23:40 | PC1 result. |

### Part 5 --- The Graph Propagation Regime

| PDF | Frame                                                          | Sec | Cum.  | Note |
|-----|----------------------------------------------------------------|-----|-------|------|
|  35 | Layer C --- the graph question                                 |  45 | 24:25 | Three EEG conditions. |
|  36 | The propagation operator                                       |  45 | 25:10 | Operator definition. |
|  37 | The Propagation Operating Characteristic (Contribution 1)      |  90 | 26:40 | **Key result**; 90 s minimum. |
|  38 | The result is robust across graph construction                 |  45 | 27:25 | Robustness sweep. |
|  39 | The diffusion mechanism, measured on real reservoir features   |  60 | 28:25 | Dirichlet drop. |
|  40 | **AUDIT** Defense audit: theoretical bounds vs. measured       |  60 | 29:25 | Closes the graph block. |
|  41 | Reframing the graph layer                                      |  35 | 30:00 | Single line. |
|  42 | Structure--function coupling, κ                                |  60 | 31:00 | Definition + figure. |
|  43 | Coupling is real above the permutation null                    |  50 | 31:50 | One stat result. |

### Part 6 --- Intrinsic Interpretability

| PDF | Frame                                                          | Sec | Cum.  | Note |
|-----|----------------------------------------------------------------|-----|-------|------|
|  45 | Post-hoc vs. intrinsic interpretability                        |  50 | 32:40 | One contrast. |
|  46 | The four-level taxonomy --- measured                           |  60 | 33:40 | Levels 1--4 read out. |
|  47 | What interpretability enables                                  |  45 | 34:25 | Cut if behind. |
|  48 | A theoretically predicted null                                 |  45 | 35:10 | Setup for next slide. |
|  49 | A null result that forced the methodology                      |  60 | 36:10 | The pivot. |
|  50 | **AUDIT** Defense audit: per-trial channel-permutation null    |  75 | 37:25 | Strictest spatial null. |
|  51 | **AUDIT** Defense audit: failure gallery --- four pivots       |  90 | 38:55 | Read the four pivots. |
|  52 | Different disorders, different layers                          |  60 | 39:55 | Table read out. |
|  53 | Layer ablation methodology (Contribution 4)                    |  45 | 40:40 | One sentence on ablation. |
|  54 | The accuracy--interpretability trade                           |  50 | 41:30 | Single point made. |

### Part 7 --- Scope, Future, and Close

| PDF | Frame                                                          | Sec | Cum.  | Note |
|-----|----------------------------------------------------------------|-----|-------|------|
|  56 | Scope of validation                                            |  45 | 42:15 | Honest scope. |
|  57 | Future directions                                              |  35 | 42:50 | Two bullets max. |
|  58 | Contributions restated                                         |  35 | 43:25 | Read the five. |
|  59 | The governing question, answered                               |  35 | 44:00 | One sentence. |
|  60 | **AUDIT** Defense claims are anchored to dissertation chapters |  30 | 44:30 | Point at the AM table; do not read it. |
|  61 | **AUDIT** Research provenance and assistance                   |  45 | 45:15 | Verbatim disclosure read. |
|  62 | (Standout) Thank you / Questions                               |  10 | 45:25 | Step off. |

---

## Audit-frame budget (subtotal)

| PDF | Frame                                                  | Sec |
|-----|--------------------------------------------------------|----:|
|  18 | Defense audit: autonomous ρ(W) vs. driven λ₁           |  90 |
|  19 | Defense audit: memory-capacity regime for leak β       |  90 |
|  23 | Defense audit: FNN-measured embedding dimension        |  75 |
|  40 | Defense audit: theoretical bounds vs. measured points  |  60 |
|  50 | Defense audit: per-trial channel-permutation null      |  75 |
|  51 | Defense audit: failure gallery --- four pivots         |  90 |
|  60 | Defense claims are anchored to dissertation chapters   |  30 |
|  61 | Research provenance and assistance                     |  45 |
|     | **Audit total**                                        | **555** |

555 s = **9 min 15 s**. Matches the manifest's "~9 min of rigor
injection" estimate.

---

## Cut-line lane (if running behind)

In priority order — drop from the bottom of this list first.

1. PDF 41 "Reframing the graph layer" — kill it; the next slide makes
   the same point. **−35 s**.
2. PDF 47 "What interpretability enables" — kill it. **−45 s**.
3. PDF 24 "The path to silicon" — kill it; mention in one sentence on
   slide 23 instead. **−45 s**.
4. PDF 6 "Why conventional pipelines miss" — trim from 35 s to 20 s.
   **−15 s**.
5. PDF 20 BSC₆ — trim from 60 s to 45 s. **−15 s**.

**Cut-lane recoverable**: ~2 min 35 s. Combined with natural pacing
contraction over a 45-minute talk (~1--2 min), the deck lands at 45
even.

Do **not** cut from:

- Audit slides 18, 19, 23, 40, 50, 51 — these are the new defensive
  payload.
- AC slide (61) — verbatim provenance read is a defensive obligation.
- Layer A centering (29), POC (37), Failure gallery (51) — the three
  key-result anchors.

---

## Q&A pacing

After the 45-minute talk: 15--20 min Q&A. Use the **qa_index.md** to
land directly on the relevant main or appendix slide for each question.
Plan one minute per question minimum; do not rush short answers.
