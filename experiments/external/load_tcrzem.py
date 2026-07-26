#!/usr/bin/env python3
"""
Loader for the external cohort: DataVerseNL doi:10.34894/TCRZEM
("EEG data of 239 participants during passive viewing of IAPS images").

The preprocessed data are BrainVision segmented files, one per
subject x valence x arousal:  S<id>_Mastoid_<Neg|Neu|Pos>_<High|Low>.{vhdr,dat}.
Each .dat is INT_16, MULTIPLEXED, NumberOfChannels channels, concatenated
1792-sample segments (512 Hz, -1000..+2500 ms epochs). This loader reads the
segments, groups them by VALENCE (Neg/Neu/Pos) to match the SHAPE three-class
design, and returns single-trial epochs for external_replication.prepare_epochs.

Channel count (41 here, incl. reference/EOG) differs from SHAPE's 34; the
replication pipeline is channel-count agnostic, so no montage matching is done.
Per-channel micro-volt scaling is irrelevant because prepare_epochs z-scores
each channel.
"""

import glob
import os
import re

import numpy as np

SEGMENT_POINTS = 1792  # per-epoch samples declared in the BrainVision headers
VALENCE = {"Neg": 0, "Neu": 1, "Pos": 2}
_FNAME = re.compile(r"(S\d+)_Mastoid_(Neg|Neu|Pos)_(High|Low)\.dat$", re.I)


def _n_channels(vhdr_path):
    with open(vhdr_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.strip().lower().startswith("numberofchannels="):
                return int(line.split("=", 1)[1])
    raise ValueError(f"NumberOfChannels not found in {vhdr_path}")


def _read_dat(dat_path, n_ch):
    """Return (n_segments, SEGMENT_POINTS, n_ch) trials from a segmented .dat."""
    raw = np.fromfile(dat_path, dtype="<i2").astype(np.float64)  # INT_16, LE
    if raw.size % n_ch != 0:
        raise ValueError(f"{dat_path}: size {raw.size} not divisible by {n_ch}")
    total_pts = raw.size // n_ch
    data = raw.reshape(total_pts, n_ch)               # MULTIPLEXED -> (pts, ch)
    if total_pts % SEGMENT_POINTS != 0:
        # trim any trailing partial segment defensively
        n_seg = total_pts // SEGMENT_POINTS
        data = data[: n_seg * SEGMENT_POINTS]
    n_seg = data.shape[0] // SEGMENT_POINTS
    return data.reshape(n_seg, SEGMENT_POINTS, n_ch)


def load_trials(data_dir, subjects=None, min_trials_per_valence=4):
    """Load single-trial epochs grouped by valence across the cohort.

    Returns (epochs, trial_conditions, trial_subjects):
      epochs           : (n_trials, 1792, n_ch)
      trial_conditions : (n_trials,) 'Neg'/'Neu'/'Pos'
      trial_subjects   : (n_trials,) subject id string
    High and Low arousal are pooled into their valence, matching the SHAPE
    three-class (Negative/Neutral/Pleasant) design. Subjects lacking any valence
    (or with too few trials in one) are dropped so every retained participant has
    a complete three-condition panel.
    """
    dats = sorted(glob.glob(os.path.join(data_dir, "*_Mastoid_*_*.dat")))
    per_subject = {}
    for dat in dats:
        m = _FNAME.search(os.path.basename(dat))
        if not m:
            continue
        subj, valence = m.group(1), m.group(2).capitalize()
        if subjects is not None and subj not in subjects:
            continue
        vhdr = dat[:-4] + ".vhdr"
        n_ch = _n_channels(vhdr) if os.path.exists(vhdr) else None
        if n_ch is None:
            continue
        trials = _read_dat(dat, n_ch)
        per_subject.setdefault(subj, {}).setdefault(valence, []).append(trials)

    epochs, conds, subs = [], [], []
    for subj in sorted(per_subject):
        vmap = per_subject[subj]
        if not all(v in vmap for v in ("Neg", "Neu", "Pos")):
            continue
        ch_counts = {t.shape[2] for parts in vmap.values() for t in parts}
        if len(ch_counts) != 1:
            continue  # inconsistent montage within subject: skip
        ok = True
        stacked = {}
        for v in ("Neg", "Neu", "Pos"):
            arr = np.concatenate(vmap[v], axis=0)
            if arr.shape[0] < min_trials_per_valence:
                ok = False
                break
            stacked[v] = arr
        if not ok:
            continue
        for v in ("Neg", "Neu", "Pos"):
            for tr in stacked[v]:
                epochs.append(tr)
                conds.append(v)
                subs.append(subj)
    if not epochs:
        raise RuntimeError(f"no complete-panel subjects found in {data_dir}")
    return np.asarray(epochs), np.asarray(conds), np.asarray(subs)
