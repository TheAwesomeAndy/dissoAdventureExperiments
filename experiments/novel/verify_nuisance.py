#!/usr/bin/env python3
"""
Synthetic verification of the nuisance-alignment study estimators (no data).
Establishes the core M1 phenomenon and the M2/M3 machinery deterministically.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "confirmatory")))

import confirmatory_pipeline as cp   # noqa: E402
import nuisance_study as ns          # noqa: E402

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


def main():
    print("=" * 70)
    print("NUISANCE-ALIGNMENT STUDY VERIFICATION (synthetic)")
    print("=" * 70)

    # ---- generic evaluator sanity -------------------------------------
    X, y, g = ns.make_controlled(40, 3, 60, sigma_subj=0.5, sigma_noise=0.5, seed=0)
    srp = ns.evaluate_compressor(X, y, g, ns._srp_fitter(32, cp.SRP_SEED))["balanced_accuracy"]
    check(srp > 1.0 / 3 + 0.05, f"SRP recovers planted condition signal ({srp:.3f})")

    # ---- M1a robust mechanism: PCA subspace rotates toward subject -----
    print("\n[M1a] geometric mechanism (principal-angle rotation)")
    sweep = ns.angle_rotation_sweep([0.0, 4.0], n_components=16, dim=200,
                                    n_participants=60, subj_rank=20)
    low, high = sweep["curve"][0], sweep["curve"][1]
    check(high["pca_angle_to_subject_deg"] < low["pca_angle_to_subject_deg"],
          f"PCA subspace rotates TOWARD subject as nuisance grows "
          f"({low['pca_angle_to_subject_deg']:.1f} -> {high['pca_angle_to_subject_deg']:.1f} deg)")
    check(high["pca_angle_to_condition_deg"] > low["pca_angle_to_condition_deg"],
          f"PCA subspace rotates AWAY from condition as nuisance grows "
          f"({low['pca_angle_to_condition_deg']:.1f} -> {high['pca_angle_to_condition_deg']:.1f} deg)")
    check(high["pca_angle_to_subject_deg"] < high["pca_angle_to_condition_deg"],
          f"at high nuisance PCA is closer to subject than condition "
          f"({high['pca_angle_to_subject_deg']:.1f} < {high['pca_angle_to_condition_deg']:.1f} deg)")

    # ---- M2 geometry: SRP does not concentrate nuisance like PCA -------
    print("\n[M2] geometry")
    Xh, yh, gh = ns.make_controlled(60, 3, 200, sigma_subj=4.0, sigma_noise=1.0, seed=2)
    geom = ns.compression_geometry(Xh, yh, gh, n_components=16, subj_rank=20, cond_rank=2)
    check(geom["pca_min_angle_to_subject_deg"] < geom["pca_min_angle_to_condition_deg"],
          f"PCA subspace aligns closer to subject than condition "
          f"({geom['pca_min_angle_to_subject_deg']} < {geom['pca_min_angle_to_condition_deg']} deg)")

    # ---- M3 comparison returns all compressors ------------------------
    print("\n[M3] compressor comparison")
    for name, fit in {"srp": ns._srp_fitter(16, 1), "pca": ns._pca_fitter(16),
                      "lda": ns._lda_fitter(),
                      "subsample": ns._subsample_fitter(16, 3)}.items():
        ba = ns.evaluate_compressor(Xh, yh, gh, fit)["balanced_accuracy"]
        check(0.0 <= ba <= 1.0, f"{name} returns valid balanced accuracy ({ba:.3f})")

    # ---- M5 disentanglement returns three ratios ----------------------
    print("\n[M5] disentanglement")
    dis = ns.disentanglement(Xh, Xh, Xh, yh, gh)
    check(set(dis) == {"raw_erp16_rho", "bsc6_rho", "bsc6_srp64_rho"},
          "disentanglement returns all three representation rhos")

    print("\n" + "=" * 70)
    print(f"NUISANCE STUDY VERIFICATION: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
