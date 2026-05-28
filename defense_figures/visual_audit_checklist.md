# Visual Audit Checklist

Defense-day visual checks for the canonical deck `ARSPI-Net_Defense.tex`,
compiled output `ARSPI-Net_Defense.pdf` (105 PDF pages, 95 content frames).

This file covers the **8 inserted audit slides** at the level of per-slide
visual integrity — does text overflow, do fonts render, do figures read at
projector resolution. It does **not** revisit the existing 45 dissertation
slides; those were author-finalised before this PR's scope opened.

Three categories of check per slide:

1. **Layout** — text/figure does not collide with the page footer, slide
   number, or the metropolis progress bar. No bullet wraps awkwardly.
2. **Figure** — the embedded PDF figure renders without rasterisation
   artefacts, has legible axis labels at projector resolution
   (≥ 60 px/cm), and reads in the room from the back row.
3. **Colour / contrast** — colours from the figure files retain enough
   contrast against the white background under projector tint; `\alert{}`
   highlights (`sbaccent` orange) are still distinguishable from `sbnavy`.

---

## Insertion #1 — Defense audit: autonomous ρ(W) vs. driven λ₁

| | |
|---|---|
| PDF page | 18 |
| Frame counter | 14 / 93 |
| Figure source | `pictures/defense_audit/analysisA_1e_autonomous_vs_driven.pdf` |
| Region | Reservoir / LIF |

- [ ] **Layout** — final bullet ("Audit cohort (N=3,165) is the strict
      clean-ERP subset of the N=4,220 Lyapunov pool from chDynamics;
      both populations 100% stable") sits close to the page footer.
      Verify on the actual projector that the last word "stable" does
      not overlap the `14/94` counter. If it does, drop the leading
      `\vspace{0.4em}` to gain a line of clearance, or split the bullet
      onto two short lines.
- [ ] **Figure** — left panel (autonomous criterion) and right panel
      (driven criterion) both readable. The annotated values ρ(W) = 0.265
      (left) and the driven-λ₁ distribution (right) are the two numbers
      the committee will read off the slide; both must be legible.
- [ ] **Colour** — red "Unstable (μ > 1)" region versus green "Stable
      (μ < 1)" region must remain distinguishable; some projectors tint
      reds toward orange. Acceptable fallback: read the annotated values
      out loud.

## Insertion #3 — Defense audit: memory-capacity regime for leak β

| | |
|---|---|
| PDF page | 19 |
| Frame counter | 15 / 93 |
| Figure source | `pictures/defense_audit/analysisC_3d_memory_capacity_peak.pdf` |
| Region | Reservoir / LIF (moved here from graph block per PR #25 audit) |

- [ ] **Layout** — three right-column bullets fit cleanly. The
      `β = 0.05 → MC ≈ 0.763 (91% of peak)` bullet uses `→` (U+2192);
      verify it renders rather than falling back to `\textrightarrow`.
- [ ] **Figure** — peak MC at β\* ≈ 0.012, plateau in [0.010, 0.118]
      with MC ≥ 0.75, and the vertical marker at β = 0.05 should all
      be visually obvious; if any axis label is cropped, regenerate the
      figure at higher DPI.
- [ ] **Colour** — the vertical β = 0.05 marker is the rhetorical anchor
      of the slide; if the marker is grey on the projector, replace it
      in the source figure with `sbaccent` orange (matches the deck's
      `\alert{}` colour).

## Insertion #2 — Defense audit: FNN-measured embedding dimension

| | |
|---|---|
| PDF page | 23 |
| Frame counter | 19 / 93 |
| Figure source | `pictures/defense_audit/analysisB_2e_takens_dimension.pdf` |
| Region | Spike-to-embedding |

- [ ] **Layout** — right-column bullets fit; the `m\* ≪ 64 (post-PCA),
      ≪ 256 (raw)` line uses the `\ll` operator and must render as the
      double-less-than glyph.
- [ ] **Figure** — the FNN curve crossing the false-nearest-neighbour
      threshold must be visually distinct from the threshold line
      itself; if both are blue, regenerate with the threshold in grey
      or dashed.
- [ ] **Colour** — neutral. Embedding-dimension figure is typically
      monochrome; OK.

## Insertion #4 — Defense audit: theoretical bounds vs. measured operating points

| | |
|---|---|
| PDF page | 40 |
| Frame counter | 34 / 93 |
| Figure source | `pictures/defense_audit/analysisTB_5a_bounds_vs_measurement.pdf` |
| Region | Graph propagation |

- [ ] **Layout** — full-bleed centred figure; no right-column text.
      Check that the figure's three panels (Lyapunov, Takens, Memory
      Capacity) are equally tall and that the in-panel annotation
      boxes are inside the panel borders, not floating over them.
- [ ] **Figure** — the per-panel theoretical-bound line and the
      measured-operating-point marker must both be visible at 33% PDF
      zoom (committee read distance).
- [ ] **Colour** — each panel uses a distinct accent colour; verify on
      projector that those three accents are still distinguishable from
      the slide background.

## Insertion #5 — Defense audit: per-trial channel-permutation null

| | |
|---|---|
| PDF page | 50 |
| Frame counter | 43 / 93 |
| Figure source | `pictures/defense_audit/analysisD_4d_observed_vs_null_panel.pdf` |
| Region | Null / methodology pivot |

- [ ] **Layout** — left figure with four subplots, right-column bullets.
      Verify the four-subplot panel does not get scaled into illegibility;
      if so, reduce to two panels (observed vs. one null example) and
      defer the remaining two to appendix `Exp D raw: observed vs. null
      example` (PDF page 103).
- [ ] **Figure** — the gap between observed-statistic distribution and
      permutation-null distribution must be the visual punchline; if
      the gap is < ~10% of axis range, increase axis zoom in the source.
- [ ] **Colour** — observed (typically blue) vs. null (typically grey)
      must be distinguishable.

## Insertion #6 — Defense audit: failure gallery — four pivots

| | |
|---|---|
| PDF page | 51 |
| Frame counter | 44 / 93 |
| Figure source | `pictures/defense_audit/figureG_failure_gallery.pdf` |
| Region | Failure-driven rigor |

- [ ] **Layout** — the four-pivot grid (2×2). Each pivot has a "what we
      tried / what failed / what changed" structure. Verify the smallest
      text block in any of the four panels reads from the back of the
      room; if not, drop one pivot to appendix and present three.
- [ ] **Figure** — this is the most text-dense audit slide; defend
      readability over completeness.
- [ ] **Colour** — neutral by design.

## Insertion #7 — Defense claims are anchored to dissertation chapters

| | |
|---|---|
| PDF page | 60 |
| Frame counter | 52 / 93 |
| Figure source | `pictures/defense_audit/figureAM_anchor_map.pdf` |
| Region | Contributions / closure |

- [ ] **Layout** — table-style figure. Verify all rows of the
      claim-to-chapter map are visible; if the figure was generated at a
      shorter height, the bottom rows may be cropped by the frame border.
- [ ] **Figure** — chapter cell labels (Ch.\ 4, Ch.\ 5, Ch.\ 6, …) must
      read clearly; if they fall back to a sans-serif system font that
      differs from the deck font (Fira Sans via metropolis), regenerate
      the figure with `matplotlib.rcParams['font.family'] = 'sans-serif'`
      pinned to Fira/DejaVu.
- [ ] **Colour** — header band uses sbnavy; verify it matches the
      frame's top bar so the figure reads as a deck artefact, not a
      foreign import.

## Insertion #8 — Research provenance and assistance

| | |
|---|---|
| PDF page | 61 |
| Frame counter | 53 / 93 |
| Figure source | `pictures/defense_audit/figureAC_provenance.pdf` |
| Region | Final disclosure (before standout close) |

- [ ] **Layout** — three-column "author-owned / AI-assisted /
      verification" panel. Verify the three columns are visually
      balanced; if one column is much taller, shorten its top entry to
      restore balance.
- [ ] **Figure** — the disclosure-paragraph text below the three
      columns is the legally / ethically significant content; verify it
      reads at presentation distance and is not abbreviated.
- [ ] **Colour** — neutral. This slide is intentionally low-energy.

---

## Cross-slide session-day checks

- [ ] Projector test: at least one full pass of the deck on the actual
      defense projector at least 24 hours before; specifically inspect
      audit pages 18, 19, 23, 40, 50, 51, 60, 61 (the eight above).
      Total deck is 104 PDF pages, 93 content slides.
- [ ] Font fallback: search for any `\textsubset` / `\ll` / `\to` /
      `\rightarrow` / `\approx` glyph rendered as a missing-box `□`.
- [ ] Page numbers: verify the bottom-right counter advances monotonically
      and the `\section` divider pages are not double-counted.
- [ ] No "Overfull \hbox" warnings on audit slides — these are the only
      slides where new text was inserted; if the log shows any, address
      them before defense.
- [ ] PDF size: the committed PDF is ~5 MB; if it grew above ~15 MB,
      regenerate figures at lower DPI before sending to the committee.
