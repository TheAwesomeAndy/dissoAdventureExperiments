# LSM-in-action animations for the ARSPI-Net defense

Three animations that make the *dynamical* claims of the LSM and graph layer
visible — claims the defense deck currently shows only as static PDF figures.
Each animation reuses the **exact, already-validated operators** from the repo
(`run_chapter3_lsm_characterization.py` and `graph_diffusion_oversmoothing.py`),
so what you see on screen is the same system the dissertation characterizes.

Rendered by `animate_lsm_dynamics.py`. Output lives in `pictures/animations/`.

## Files

| Animation | MP4 (defense) | GIF (preview / GitHub) | Poster (any viewer) |
|---|---|---|---|
| 1 — Graph diffusion | `lsm_graph_diffusion.mp4` | `lsm_graph_diffusion.gif` | `lsm_graph_diffusion_poster.pdf` |
| 2 — Membrane + raster | `lsm_membrane_raster.mp4` | `lsm_membrane_raster.gif` | `lsm_membrane_raster_poster.pdf` |
| 3 — BSC6 trajectory | `lsm_state_trajectory.mp4` | `lsm_state_trajectory.gif` | `lsm_state_trajectory_poster.pdf` |

All paths below are relative to the repo root; adjust if your `.tex` lives
elsewhere. The defense `.tex` already uses a `pictures/...` path convention,
so these slot straight in.

## What each animation shows / which slide it belongs on

**Animation 1 — Graph diffusion / over-smoothing.**
Slide: *"Why the boundary exists: graph diffusion."*
Runs the `graph_diffusion_oversmoothing.py` pipeline on the **real SHAPE data**
(`shape_features_211.pkl`, `X_ds` = 633x256x34). Left: a 34-node functional
electrode graph whose node colours homogenize as `H <- (normalised adjacency) H`
is applied for `K = 0..8`. Right: Dirichlet energy collapses and mean pairwise
cosine rises — the over-smoothing onset at `K=1..2` coincides with where exp03
accuracy falls fastest.

**Animation 2 — Membrane potential + spike raster.**
Slides: *"The neuron: leaky integrate-and-fire"* and *"Temporal coding..."*
Early-burst vs late-burst inputs (the synthetic temporal task, identical to
Ch3/Ch4) driven through one fixed reservoir. Shows input ERP, an example
neuron's membrane integrating/firing/resetting against threshold, the
256-neuron membrane heatmap, the spike raster with the six BSC6 bins filling
in real time, and the population firing rate.

**Animation 3 — BSC6 feature trajectory + linear readout.**
Slide: *"Why a linear readout suffices: the Koopman lens."*
The BSC6 feature is assembled bin-by-bin; the two class-mean trajectories
start identical (empty feature) and trace apart into linearly-separable
clusters. The logistic readout's decision direction is one axis, so the
boundary is a clean vertical line; faint DMD arrows show the one-step linear
(Koopman) flow.

## Embedding in Beamer

GIFs do **not** animate inside a PDF — use them for GitHub / preview only.
For the deck use the MP4 (plays in Adobe Acrobat Reader) with the poster PDF
as the always-visible fallback.

### Option A — embedded video with `media9` (recommended)

```latex
\usepackage{media9}   % in the preamble

% --- on the slide ---
\includemedia[
  activate=onclick, transparent=false,
  width=\linewidth,
  addresource=pictures/animations/lsm_graph_diffusion.mp4,
  flashvars={source=pictures/animations/lsm_graph_diffusion.mp4}
]{\includegraphics[width=\linewidth]{pictures/animations/lsm_graph_diffusion_poster.pdf}}%
 {pictures/animations/lsm_graph_diffusion.mp4}
```

Swap the three filenames for `lsm_membrane_raster` or `lsm_state_trajectory`
on the other two slides. The poster frame is what shows until clicked, and what
prints. Embedded playback needs Adobe Acrobat Reader; other viewers show the
poster.

### Option B — poster image + click-to-open in an external player

Works in every PDF viewer; the video opens in the system player.

```latex
\href{run:pictures/animations/lsm_graph_diffusion.mp4}{%
  \includegraphics[width=\linewidth]{pictures/animations/lsm_graph_diffusion_poster.pdf}}
```

### Option C — static poster only (zero dependencies)

```latex
\includegraphics[width=\linewidth]{pictures/animations/lsm_graph_diffusion_poster.pdf}
```

## Suggested captions

- **Anim 1:** "Repeated message passing is diffusion on the graph Laplacian:
  node features lose contrast (Dirichlet energy down ~87%) by K=2 — the depth
  at which exp03 accuracy falls fastest."
- **Anim 2:** "A fixed LIF reservoir converts the *timing* of an ERP into a
  sparse spike code; the six BSC6 bins capture early- vs late-burst structure."
- **Anim 3:** "The reservoir + BSC6 lift the input so the two classes become
  linearly separable — a single linear readout suffices."

## Regenerating

```bash
pip install numpy scikit-learn matplotlib pillow imageio-ffmpeg
python experiments/chapter3/animate_lsm_dynamics.py --only all \
    --pkl /path/to/shape_features_211.pkl --format both --dpi 100 --fps 18
```

- `--pkl` accepts the real `shape_features_211.pkl` **or** an `.npz` extract
  with an `X_ds` key. Omit it and animation 1 falls back to a synthetic
  spatially-correlated 34-channel example (clearly labelled on the figure).
- Animations 2 and 3 need no external data.
- `--only {diffusion,raster,trajectory,all}`, `--format {gif,mp4,both}`,
  `--fps`, `--dpi`, `--n` (observations used for animation 1's per-channel PCA).
- ffmpeg is provided by the `imageio-ffmpeg` pip package — no system install
  needed.

Note: the membrane/raster GIF is large (full-colour heatmap x ~190 frames);
it is rendered at a lower dpi than its MP4 for that reason. The MP4 is the
defense-quality asset.

---

# Round 2 — Chapter 4 dynamics animations

Three more animations that visualize the **core dynamical claims of Chapter
4**: the LIF reservoir as a driven point process under real SHAPE ERP, the
Echo State Property via the driven Lyapunov exponent, and the BSC6 temporal
code. Renderer: `chapter4Experiments/animate_ch4_dynamics.py`. Operators
copied verbatim from `chapter4Experiments/run_chapter4_observations.py`
(dense Xavier-uniform LIF reservoir — animations 4 & 6) and
`chapter6Experiments/run_chapter6_exp1_esp.py` (10% sparse Gaussian
reservoir + Benettin Lyapunov — animation 5, the architecture under which
the dissertation measures λ₁ = -0.054).

## Files

| Animation | MP4 (defense) | GIF (preview / GitHub) | Poster (any viewer) |
|---|---|---|---|
| 4 — Driven LIF on real ERP | `lsm_driven_erp.mp4` | `lsm_driven_erp.gif` | `lsm_driven_erp_poster.pdf` |
| 5 — Lyapunov convergence + ESP | `lsm_lyapunov_convergence.mp4` | `lsm_lyapunov_convergence.gif` | `lsm_lyapunov_convergence_poster.pdf` |
| 6 — BSC6 bin accumulation | `lsm_bsc6_accumulation.mp4` | `lsm_bsc6_accumulation.gif` | `lsm_bsc6_accumulation_poster.pdf` |

Embed exactly the same way as the Round 1 three — see *Embedding in Beamer*
above for the `\includemedia` (media9), `\href{run:…}`, and poster-only
variants. The path convention is unchanged (`pictures/animations/…`).

## What each animation shows / which slide it belongs on

**Animation 4 — Driven LIF on real SHAPE ERP.**
Slide: *"The reservoir as a driven dynamical system"* / LIF equation slide.
One IAPSNeg and one IAPSPos trial from `shape_features_211.pkl` (the
deterministic median-energy pick of each condition). Top panel: all 34 ERP
channels, the reservoir-input electrode (Pz / ch 8 default) highlighted in
the condition colour. Middle: 8 representative neurons' membrane traces with
threshold guides — the integrate-fire-reset is legible per trace. Bottom:
the 256-neuron spike raster filling in real time. Shared row at the bottom
overlays both conditions' instantaneous population firing rates so the dual
trace makes the "same total energy, different timing" observation
explicit.

**Animation 5 — Driven Lyapunov convergence (ESP gate).**
Slide: *"Driven Lyapunov λ₁ = -0.054 / Echo State Property holds."*
Top-left: Benettin stretching factor `log(|δ_k| / δ_0)` per renormalization
step on one demo trial — all values below zero, the dotted reference line
at 0 makes the contraction visible. Top-right: running cumulative estimate
λ₁(k) approaching the trial's final value (dotted horizontal line); the
"fading memory ⇒ Echo State Property" annotation fades in once the
estimate has converged. Bottom: histogram of final λ₁ across a stratified
representative sample of ~500 trials (167 per IAPSNeg/Neu/Pos, distributed
across the same 5 channels the dissertation analyses). The smoke run shows
mean λ₁ = -0.056, 100% negative — within ±0.005 of the dissertation's
-0.05359.

**Animation 6 — BSC6 bin accumulation on real SHAPE.**
Slide: *"Temporal coding carries the discriminative information"* / BSC6
slide.
Same Neg/Pos pair as animation 4. Top of each column: the spike raster
with six bin guides; the active bin glows in the condition colour as the
time cursor enters it. Middle: six horizontal bars fill proportionally to
the running per-bin spike count (summed across all 256 neurons). Below
that: the live 1536-d BSC6 vector as a 6×256 colourmap, filling row-by-row
as each bin closes. Bottom (across both columns): the same total spike
count as a single rate-code bar — visibly identical between Neg and Pos,
while the per-bin codes differ. The message: timing carries the
information; rate alone does not.

## Suggested captions

- **Anim 4:** "Same fixed reservoir, two real SHAPE trials: the LIF
  membrane integrates, fires, and resets in response to the ERP, producing
  a sparse spike code."
- **Anim 5:** "Driven Lyapunov stays uniformly negative across the cohort
  (mean -0.054, 100% negative). The reservoir is contracting under real
  EEG drive — the Echo State Property holds."
- **Anim 6:** "Six temporal bins capture *when* the reservoir fires, not
  just *how much*. The total spike count (rate code) is the same; the
  per-bin distribution is what differs."

## Regenerating

The chapter-4 animations all need `shape_features_211.pkl`. They reuse the
Round-1 dependencies (`numpy scikit-learn matplotlib pillow imageio-ffmpeg`).

```bash
python chapter4Experiments/animate_ch4_dynamics.py --only all \
    --pkl /path/to/shape_features_211.pkl --format both --dpi 100 --fps 18
```

Flags worth knowing:

- `--only {driven_erp,lyapunov,bsc6,all}` — pick a subset.
- `--erp-channel N` — which electrode drives the LIF reservoir for animations
  4 & 6 (default 8, matching the dissertation's `analysis_channels`).
- `--n-cohort N` — stratified Lyapunov subset size (default 501 ≈ 167 per
  valence × 3 valences). Set higher for a tighter histogram (linear cost).
- `--cache PATH` — pre-computed Lyapunov bundle. Reused unless
  `--rebuild-cache` is passed. Default: `chapter4Experiments/cache/
  ch4_lyapunov_cache.npz`.
- `--lyap-channels N N ...` — channels sampled into the Lyapunov cohort;
  defaults to `0 8 16 24 33`, matching `run_chapter6_exp1_esp.py`.

Runtime budget on a CPU container: Lyapunov cohort ~3 min (500 trials), three
animations ~3×90 s. Cached cohort makes re-renders instant.

The chapter-4 reservoir for animations 4 & 6 is the dense Xavier-uniform
construction used in `run_chapter4_observations.py` (the static figures of
the chapter); the chapter-6 reservoir for animation 5 is the 10% sparse
Gaussian construction used in `run_chapter6_exp1_esp.py` (the architecture
under which the dissertation reports λ₁ = -0.054). Both definitions are
copied verbatim into the renderer so the animations match the published
operators.
