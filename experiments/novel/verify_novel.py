#!/usr/bin/env python3
"""
Synthetic verification of the novel-experiment estimators (no SHAPE data).

Establishes code behavior and the statistical properties each estimator relies
on, so the machinery runs in CI without the restricted dataset. It does not
establish the empirical SHAPE findings.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "confirmatory")))

import confirmatory_pipeline as cp
import novel_pipeline as nov

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


def synth(n_participants=24, n_cond=3, T=64, n_ch=4, signal=1.0, seed=0):
    rng = np.random.RandomState(seed)
    templates = rng.randn(n_cond, T, n_ch)
    raw, y, groups = [], [], []
    for p in range(n_participants):
        offset = rng.randn(n_ch) * 5.0
        for c in range(n_cond):
            raw.append(signal * templates[c] + offset[None, :] + rng.randn(T, n_ch) * 0.5)
            y.append(c)
            groups.append(p)
    return np.array(raw), np.array(y), np.array(groups)


def main():
    print("=" * 70)
    print("NOVEL EXPERIMENTS VERIFICATION (synthetic data)")
    print("=" * 70)

    raw, y, groups = synth()
    bsc = cp.build_bsc_features(raw, reservoir_seed=42)
    srp = cp.make_srp(bsc.shape[2], n_components=cp.N_SRP)
    d = bsc.shape[2]

    # ---- N1 ------------------------------------------------------------
    print("\n[N1] leak-free compression")
    srp_eval = nov.evaluate_srp(bsc, y, groups, 32, cp.SRP_SEED)
    pca_eval = nov.evaluate_foldlocal_pca(bsc, y, groups, 32)
    check(0.0 <= srp_eval["balanced_accuracy"] <= 1.0
          and 0.0 <= pca_eval["balanced_accuracy"] <= 1.0,
          "SRP and fold-local PCA both return valid balanced accuracy")
    # fold-local PCA must not leak: shuffled-within-participant labels -> ~chance
    rng = np.random.RandomState(1)
    yp = y.copy()
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        yp[gi] = rng.permutation(y[gi])
    leak = nov.evaluate_foldlocal_pca(bsc, yp, groups, 32)["balanced_accuracy"]
    check(leak < pca_eval["balanced_accuracy"],
          f"fold-local PCA does not leak (permuted {leak:.3f} < signal {pca_eval['balanced_accuracy']:.3f})")
    stab = nov.srp_seed_stability(bsc, y, groups, list(range(5)), 32)
    check(stab["n_seeds"] == 5 and stab["max"] >= stab["min"],
          "SRP seed-stability summary is well formed")

    # ---- N2 ------------------------------------------------------------
    print("\n[N2] reservoir instrument")
    # ICC: identical raters -> ICC ~ 1; independent noise -> low ICC
    ident = np.tile(np.random.RandomState(2).randn(30, 1), (1, 3))
    check(nov.icc_2_1(ident) > 0.95, f"ICC of identical raters ~ 1 ({nov.icc_2_1(ident):.3f})")
    noise = np.random.RandomState(3).randn(30, 3)
    check(nov.icc_2_1(noise) < 0.5, f"ICC of independent noise is low ({nov.icc_2_1(noise):.3f})")
    rel = nov.cross_seed_feature_reliability([bsc, cp.build_bsc_features(raw, 7)], srp)
    check(-1.0 <= rel["median_icc"] <= 1.0, f"cross-seed reliability in range ({rel['median_icc']})")
    tf = nov.transfer_function(raw[0, :, 0], np.linspace(0.5, 4, 6))
    check(tf["monotonicity_spearman"] > 0.5,
          f"transfer function is monotonic increasing (rho={tf['monotonicity_spearman']})")
    # robustness: dropping channels should not raise and returns a curve
    curve = nov.robustness_curve(raw, y, groups, nov.perturb_channel_dropout,
                                 [0.0, 0.5], lambda p: cp.project_bsc(cp.build_bsc_features(p, 42), srp))
    check(len(curve) == 2 and curve[0]["level"] == 0.0, "robustness curve returns levels")

    # ---- N3 ------------------------------------------------------------
    print("\n[N3] necessity map + variance decomposition")
    X = cp.project_bsc(bsc, srp)
    vr = nov.subject_condition_variance_ratio(X, y, groups)
    check(vr["subject_fraction"] > vr["condition_fraction"],
          f"planted subject nuisance dominates (subject {vr['subject_fraction']} > "
          f"condition {vr['condition_fraction']})")
    check(vr["rho_subject_to_condition"] > 1.0,
          f"rho > 1 under subject-covariance entanglement ({vr['rho_subject_to_condition']})")
    nmap = nov.necessity_map(bsc, y, groups, srp)
    check(set(nmap) == {"intact", "temporal_bin_shuffle", "temporal_collapse",
                        "electrode_shuffle"}, "necessity map has all four rows")
    check(nmap["intact"]["condition_accuracy_drop_pp"] == 0.0,
          "intact row has zero drop by definition")
    check(nmap["electrode_shuffle"]["condition_balanced_accuracy_percent"]
          <= nmap["intact"]["condition_balanced_accuracy_percent"] + 5.0,
          "electrode shuffle does not increase condition accuracy materially")

    print("\n" + "=" * 70)
    print(f"NOVEL VERIFICATION: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("Verifies estimator behavior on synthetic data, not the SHAPE findings.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
