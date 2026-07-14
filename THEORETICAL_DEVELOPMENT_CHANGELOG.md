# Theoretical Development Audit

This revision expands the dissertation from 330 to 337 compiled pages without removing the original empirical chapters, figures, experiments, or appendices.

## Formal additions

- Chapter 1: observational-equivalence definition, non-identifiability proposition, proof, and identifiable-target corollary.
- Chapter 2: bounded-input bounded-state theorem for the implemented discrete LIF update; proof that representation geometry can improve without information creation.
- Chapter 3: exact subthreshold memory-horizon proposition; spike-itinerary stability lemma; corrected scope of the Lindeberg and Dambre results.
- Chapter 4: non-expansiveness proof for binned spike counts; linear-projection stability; Johnson--Lindenstrauss context for sparse random projection.
- Chapter 5: spectral theorem and corollary for component-wise graph smoothing, with an explicit separation between homogenization and classification effects.
- Chapter 6: normalized permutation-entropy bounds; conditional contraction theorem for fixed spike itineraries; direct removal of references to an external "uploaded note."
- Chapter 7: proof that normalized coupling lies in `[0,1]`; exact second moment of the electrode-permutation null, giving the finite-array RMS baseline `1/sqrt(C-1)`.
- Chapter 8: unified theoretical statement and proof of local robustness for the complete temporal representation.

## Verification

- Built successfully with `latexmk -pdf`.
- Final PDF length: 337 pages.
- No unresolved LaTeX citations or cross-references.
- New theorem pages and page continuations were rendered and visually inspected for clipping, overflow, and equation layout.
- No repository links or platform-specific language remain in the dissertation text or bibliography.
