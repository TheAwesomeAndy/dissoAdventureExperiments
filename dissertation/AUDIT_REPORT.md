# Dissertation Language Audit and Revision

**Document audited:** *Affective Reservoir-Spike Processing and Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic Framework for Clinical EEG Analysis* (Andrew Lane, ProQuest Theoretical Revision).

**Scope of edit:** removal of em dashes, banned AI/inflated vocabulary, defensive or pre-emptive phrasing about work not done, and mechanical/AI cadence. No claims, data, citations, or meaning were added or removed. Structural LaTeX (labels, refs, equations, citations) was left untouched.

**Source files audited (all `\include`d chapters plus front matter and appendix):** `main.tex`, `ch1.tex`, `ch2.tex`, `chLSM.tex`, `chLSMEmbeddings.tex`, `chgraph.tex`, `chDynamics.tex`, `chSynthesis.tex`, `chConclusion.tex`, `Appendixa.tex`.

---

## Section 1: Revised text

The complete clean revised version is the set of `.tex` source files in this `dissertation/` directory. Every change is listed exactly in Section 2 below. Files with no listed change (`main.tex`, `chLSMEmbeddings.tex`, `Appendixa.tex`, `Disso.bib`) were audited in full and required no edit; they are included so the source compiles as a coherent whole.

**Overall finding.** The dissertation is human-authored, careful, and unusually disciplined about scope. It states limitations and negative results as plain facts (for example, "X was not performed," "these are exploratory," "raw ERP summaries remain stronger"), which is exactly the register the rules require. There were no instances of apologetic or anticipatory constructions such as "Although we did not," "Unfortunately," "A major weakness," or "One might argue that... but." The edits below are therefore targeted: two em dashes, a set of banned/inflated vocabulary substitutions, one concessive clause, and one anticipatory disclaimer.

---

## Section 2: Detailed audit report

Each entry gives the file and original passage, the rule violated, and the exact revision.

### Rule 1: Em dashes

**1. `ch2.tex` (Validation Design section)**
- Original: "Every data-adaptive step`---`standardization, feature selection, PCA, graph construction that uses cohort statistics, and hyperparameter selection`---`must be fitted using training data only..."
- Violation: Rule 1 (two em dashes, written as LaTeX `---`).
- Revision: "Every data-adaptive step **(**standardization, feature selection, PCA, graph construction that uses cohort statistics, and hyperparameter selection**)** must be fitted using training data only..."
- Note: This was the only em-dash occurrence in the entire dissertation. En-dashes (`--`) used for numeric/chapter ranges and compound terms (for example `structure--function`, `excitability--persistence`, `Chapters~5--8`, `Johnson--Lindenstrauss`) are correct usage and were retained; they are not em dashes.

### Rule 2: Banned / inflated vocabulary

**2. `ch1.tex`** — "together with **comprehensive** clinical metadata" → "together with **detailed** clinical metadata". (Rule 2: *comprehensive*.)

**3. `ch1.tex`** — "$\eta$ **encompasses** measurement noise and physiological artifacts" → "$\eta$ **captures** measurement noise and physiological artifacts". (Rule 2: *encompass*.)

**4. `ch1.tex`** — "motivate the investigation of alternative **paradigms**" → "motivate the investigation of alternative **approaches**". (Rule 2: *paradigm*.)

**5. `ch1.tex`** — "a fundamentally different computational **paradigm**. The present dissertation..." → "a fundamentally different computational **approach**. The present dissertation...". (Rule 2: *paradigm*.)

**6. `ch1.tex`** (subsection title) — "Spiking Neural Networks: A Different Computational **Paradigm**" → "Spiking Neural Networks: A Different **Model of Computation**"; and body "offer a fundamentally different computational **paradigm**" → "offer a fundamentally different **model of computation**". (Rule 2: *paradigm*.)

**7. `ch1.tex`** — "have become a prominent **paradigm** for multichannel EEG analysis" → "have become a prominent **approach** for multichannel EEG analysis". (Rule 2: *paradigm*.)

**8. `ch2.tex`** (section title + comment) — "The Evolution of Neural Computation **Paradigms**" → "The Evolution of Neural Computation". (Rule 2: *paradigm*.)

**9. `ch2.tex`** — "established the **crucial** principle of learning a vector of synaptic weights" → "established the **central** principle of learning a vector of synaptic weights". (Rule 2: *crucial*.)

**10. `ch2.tex`** — "which **encompasses** the vast majority of modern deep learning" → "which **includes** the vast majority of modern deep learning". (Rule 2: *encompass*.)

**11. `ch2.tex`** — "has **revolutionized** fields from computer vision to natural language processing" → "has **reshaped** fields from computer vision to natural language processing". (Rule 2: *revolutionize*.)

**12. `ch2.tex`** — "In this **paradigm**, the analog output of a unit..." → "In this **generation**, the analog output of a unit..." (Rule 2: *paradigm*; "generation" is exact in context, since the passage discusses the second generation.)

**13. `ch2.tex`** (subsection title) — "The Third Generation: The **Paradigm Shift** to Spiking Neural Networks" → "The Third Generation: The **Shift** to Spiking Neural Networks". (Rule 2: *paradigm*.)

**14. `ch2.tex`** — "it is a fundamentally different computational **paradigm**. Maass proved... by **leveraging** the temporal domain" → "it is a fundamentally different computational **model**. Maass proved... by **exploiting** the temporal domain". (Rule 2: *paradigm*, *leverage*.)

**15. `ch2.tex`** — "underpins the dissertation's measurement-instrument **paradigm**" → "underpins the dissertation's measurement-instrument **framing**". (Rule 2: *paradigm*. "Framing" matches the author's own wording later in the same passage: "This measurement-instrument framing treats the reservoir as a standardized nonlinear transform.")

**16. `ch2.tex`** (section title) — "Reservoir Computing: A Framework for **Harnessing** Complex Dynamics" → "Reservoir Computing: A Framework for **Computing with** Complex Dynamics". (Rule 2: *harness*.)

**17. `ch2.tex`** — "Tanaka et al. provided a **comprehensive** review of recent advances" → "Tanaka et al. provided an **extensive** review of recent advances". (Rule 2: *comprehensive*.)

**18. `ch2.tex`** — "is a **burgeoning** field" → "is an **active and growing** field". (Rule 2: inflated term.)

**19. `ch2.tex`** — "Recent work has begun to explore this **synergy**" → "Recent work has begun to explore this **combination**". (Rule 2: *synergy*.)

**20. `ch2.tex`** — "a clear and **compelling** research gap" → "a clear research gap". (Rule 2/5: removed the inflated modifier; the claim is unchanged.)

**21. `chLSM.tex`** — "the choice between them has **profound** consequences" → "the choice between them has **substantial** consequences". (Rule 2: inflated term.)

**22. `chLSM.tex`** — "based on a more **comprehensive** analysis using temporally resolved coding schemes" → "based on a more **detailed** analysis using temporally resolved coding schemes". (Rule 2: *comprehensive*.)

**23. `chLSM.tex`** — "The reservoir computing **paradigm** posits that..." → "The reservoir computing **framework** posits that...". (Rule 2: *paradigm*.)

**24. `chDynamics.tex`** — "a measurement-instrument **paradigm** that clarifies..." → "a measurement-instrument **framing** that clarifies..." (Rule 2: *paradigm*; consistent with the author's own "framing" usage.)

**25. `chDynamics.tex`** (subsection title) — "Measurement-Instrument **Paradigm**" → "Measurement-Instrument **Framing**". (Rule 2: *paradigm*.)

**26. `chDynamics.tex`** — "Whether this axis reflects an **enduring** individual trait..." → "Whether this axis reflects a **stable** individual trait..." (Rule 2: *enduring*; "stable trait" is the standard psychometric term.)

**27. `chSynthesis.tex`** — "reveals a more **nuanced** pattern than the uniformly negative four-class result" → "reveals a more **mixed** pattern than the uniformly negative four-class result". (Rule 2: *nuanced*; "mixed" is exact, since the matrix has 6 negative and 8 positive entries.)

**28. `chConclusion.tex`** — "the **measurement-instrument paradigm** (Chapter 6)" → "the **measurement-instrument framing** (Chapter 6)". (Rule 2: *paradigm*.)

**29. `chgraph.tex`** — "All 211 subjects have **comprehensive** diagnostic metadata" → "All 211 subjects have **complete** diagnostic metadata". (Rule 2: *comprehensive*; "complete" is exact, since the sentence asserts full records for every subject.)

**30. `chgraph.tex`** — "presents a **comprehensive** 10-panel spectral characterization" → "presents a **detailed** 10-panel spectral characterization". (Rule 2: *comprehensive*.)

### Rule 3: Defensive / pre-emptive phrasing

**31. `ch1.tex`** — Original: "This dissertation positions its framework to be compatible with that trajectory, **although the energy advantage is motivational and is not experimentally measured** in the present software implementation."
- Violation: Rule 3 (concessive "although..." softening a statement of work not done).
- Revision: "This dissertation positions its framework to be compatible with that trajectory. **The energy advantage is motivational; it is not experimentally measured** in the present software implementation." The fact is now stated directly, with no concessive.

**32. `ch2.tex`** — Original: "**None of these comparisons should be read as a claim** that the alternatives are fundamentally incapable. The argument is **instead** that for the specific combination of requirements..."
- Violation: Rule 3 (pre-emptive framing that anticipates a reader objection, "should be read as a claim").
- Revision: "**These comparisons do not claim** that the alternatives are fundamentally incapable. The argument is **narrower**: for the specific combination of requirements..." The scope is stated directly rather than as an anticipated defense.

### Rule 4 (sentence rhythm) and Rule 5 (tone)

No standalone edits were required. The two rephrasings under Rule 3 also improve rhythm by replacing concessive clauses with short declarative sentences. The document already mixes short, medium, and long sentences, varies its openings, and states evidence proportionately.

---

## Items observed but not changed (outside the stated rules)

The instructions restrict edits to the prohibited patterns and forbid altering meaning, so the following genuine copyediting issues were left in place and are noted only for the author's attention:

- `chLSM.tex`, final paragraph of the chapter synthesis: an unbalanced parenthesis. A `(` opens at "...quantitative precision **(**establishing the foundation..." and the matching close appears as a double `))` near "...spectral radius**))** a design choice...". This is a punctuation defect, not an AI-language pattern.
- `chLSM.tex`, low-threshold results: "preserved to document the **defended** analysis" reads as a possible slip for "the original analysis." Meaning-level; left as written.

These are the only structural defects found. Correcting them would change punctuation/wording beyond the audit's mandate and is left to the author.
