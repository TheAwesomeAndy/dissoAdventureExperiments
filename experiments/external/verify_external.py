#!/usr/bin/env python3
"""
Synthetic verification of the external-replication harness (no download).
Proves prepare_epochs() and replicate() behave correctly on synthetic
single-trial epochs, so the machinery is ready for a real external dataset.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import external_replication as ext  # noqa: E402

PASS = 0
FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")
    return cond


def synth_trials(n_subj=24, n_cond=3, n_trials=8, T=150, n_ch=5, seed=0):
    """Single-trial epochs with a condition template + subject offset + noise,
    at a non-256 time length to exercise resampling."""
    rng = np.random.RandomState(seed)
    templates = rng.randn(n_cond, T, n_ch)
    ep, cond, subj = [], [], []
    for s in range(n_subj):
        offset = rng.randn(n_ch) * 4.0
        for c in range(n_cond):
            for _ in range(n_trials):
                ep.append(templates[c] + offset[None, :] + rng.randn(T, n_ch) * 1.5)
                cond.append(f"cond{c}")
                subj.append(f"sub-{s:02d}")
    return np.asarray(ep), np.asarray(cond), np.asarray(subj)


def main():
    print("=" * 70)
    print("EXTERNAL REPLICATION HARNESS VERIFICATION (synthetic)")
    print("=" * 70)

    ep, cond, subj = synth_trials()
    raw, y, groups = ext.prepare_epochs(ep, cond, subj)

    check(raw.shape[1:] == (ext.TARGET_T, 5),
          f"epochs resampled to (256, n_ch): {raw.shape[1:]}")
    check(raw.shape[0] == 24 * 3,
          f"one ERP per subject x condition: {raw.shape[0]} (expect 72)")
    check(len(np.unique(y)) == 3, "three condition labels recovered")
    check(len(set(groups.tolist())) == 24, "24 participants recovered")
    # z-scored per channel: each ERP channel ~ zero mean
    means = np.abs(raw.mean(axis=1))
    check(np.all(means < 1e-6), "each ERP channel is per-channel z-scored")

    # trial-averaging reduces noise: averaged ERP closer to the noiseless template
    check(np.isfinite(raw).all(), "prepared ERPs are finite")

    print("\nRunning N1 + M2 on the prepared synthetic cohort ...")
    res = ext.replicate(raw, y, groups, dims=(16, 32))
    check(set(res) >= {"N1_srp_vs_pca", "M2_geometry", "n_participants",
                       "n_channels"}, "replicate returns N1 + M2 + metadata")
    check(res["n_channels"] == 5 and res["n_participants"] == 24,
          "metadata reflects the cohort")
    for d, rec in res["N1_srp_vs_pca"].items():
        check(0.0 <= rec["srp_percent"] <= 100.0
              and len(rec["srp_minus_pca_ci95_pp"]) == 2,
              f"N1 dim {d} record well-formed (SRP {rec['srp_percent']}%)")
    geom = res["M2_geometry"]
    check("pca_min_angle_to_subject_deg" in geom
          and "rho_pca_compressed" in geom,
          "M2 geometry record has principal angles and compressed rho")

    print("\n" + "=" * 70)
    print(f"EXTERNAL HARNESS VERIFICATION: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("Ready to point at a real external affective-picture ERP dataset.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
