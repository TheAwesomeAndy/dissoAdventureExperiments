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
