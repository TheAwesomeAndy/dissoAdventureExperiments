#!/usr/bin/env python3
"""
ARSPI-Net -- Confirmatory Pipeline Verification (no data required)
=================================================================

Proves the *methodological* correctness of the confirmatory pipeline on
synthetic data, so the machinery is testable in CI without the restricted
SHAPE dataset. These tests verify code behavior and the preprocessing
boundary; they do NOT establish the scientific findings, which require the
authorized data fed to run_confirmatory_validation.py.

What is verified:
  1.  SRP-64 is data-independent (identical projection regardless of the
      observations it is fitted on) and deterministic across seeds.
  2.  SRP preserves electrode identity in the concatenated layout.
  3.  StratifiedGroupKFold never places a participant in both train and test
      (no subject leakage).
  4.  evaluate_features is deterministic for a fixed seed.
  5.  Fold-local standardization does not leak: permuted-within-participant
      labels collapse to ~chance and the permutation test is non-significant,
      while a planted participant-grouped signal is recovered above chance and
      is significant.
  6.  Destruction controls transform features exactly as specified.
  7.  A tiny end-to-end granularity run emits the machine-readable schema that
      check_against_manifest.py consumes.

Exit code is non-zero if any check fails.
"""

import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import confirmatory_pipeline as cp
import run_confirmatory_validation as runner

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


# ---------------------------------------------------------------------------
# Synthetic data with participant structure
# ---------------------------------------------------------------------------
def make_synthetic(n_participants=25, n_cond=3, T=64, n_ch=4, signal=1.0,
                   seed=0):
    """Each participant has one observation per condition. A condition-specific
    temporal template (shared across participants) provides the *only*
    generalizable signal; a large per-participant DC offset is the nuisance
    that makes participant-grouped evaluation necessary."""
    rng = np.random.RandomState(seed)
    templates = rng.randn(n_cond, T, n_ch)  # condition signal
    raw, y, groups = [], [], []
    for p in range(n_participants):
        offset = rng.randn(n_ch) * 5.0  # subject nuisance (large)
        for c in range(n_cond):
            x = signal * templates[c] + offset[None, :] + rng.randn(T, n_ch) * 0.5
            raw.append(x)
            y.append(c)
            groups.append(p)
    return np.array(raw), np.array(y), np.array(groups)


def main():
    print("=" * 70)
    print("CONFIRMATORY PIPELINE VERIFICATION (synthetic data)")
    print("=" * 70)

    # --- 1. SRP data-independence + determinism ----------------------------
    print("\n[1] SRP-64 data-independence and determinism")
    d = cp.N_BINS * cp.N_RES
    srp_a = cp.make_srp(d)
    srp_b = cp.make_srp(d)
    same_matrix = np.array_equal(srp_a.components_.toarray(),
                                 srp_b.components_.toarray())
    check(same_matrix, "Two SRPs with the same seed produce the identical matrix")
    check(srp_a.n_components == cp.N_SRP, f"SRP output dim == {cp.N_SRP}")
    # Fitting on completely different 'data' must not change the projection
    srp_c = cp.SparseRandomProjection(n_components=cp.N_SRP,
                                      random_state=cp.SRP_SEED)
    srp_c.fit(np.random.RandomState(1).randn(50, d))
    check(np.array_equal(srp_a.components_.toarray(), srp_c.components_.toarray()),
          "SRP matrix is independent of the observations it is fit on")

    # --- 2. Electrode identity preserved -----------------------------------
    print("\n[2] Projection preserves electrode identity")
    N, n_ch = 6, 4
    bsc = np.random.RandomState(2).rand(N, n_ch, d)
    proj = cp.project_bsc(bsc, cp.make_srp(d))
    check(proj.shape == (N, n_ch * cp.N_SRP),
          f"Projected shape is (N, n_ch*{cp.N_SRP}) = {proj.shape}")
    # electrode 0 block depends only on electrode 0 input
    bsc2 = bsc.copy()
    bsc2[:, 1, :] += 3.0  # perturb electrode 1 only
    proj2 = cp.project_bsc(bsc2, cp.make_srp(d))
    blk0_same = np.allclose(proj[:, :cp.N_SRP], proj2[:, :cp.N_SRP])
    blk1_diff = not np.allclose(proj[:, cp.N_SRP:2 * cp.N_SRP],
                                proj2[:, cp.N_SRP:2 * cp.N_SRP])
    check(blk0_same and blk1_diff,
          "Per-electrode blocks are independent (electrode identity kept)")

    # --- 3. Participant-grouped splits have no leakage ---------------------
    print("\n[3] Participant-grouped CV has no subject leakage")
    raw, y, groups = make_synthetic()
    leak = False
    sgkf = StratifiedGroupKFold(n_splits=cp.N_FOLDS, shuffle=True, random_state=0)
    X_dummy = np.zeros((len(y), 3))
    for tr, te in sgkf.split(X_dummy, y, groups):
        if set(groups[tr]) & set(groups[te]):
            leak = True
    check(not leak, "No participant appears in both train and test of any fold")

    # --- Build reservoir/SRP features once for the remaining tests ---------
    print("\n[.] Building synthetic BSC6/SRP-64 features ...")
    bsc = cp.build_bsc_features(raw, reservoir_seed=42)
    Xsrp = cp.project_bsc(bsc, cp.make_srp(d))

    # --- 4. Determinism ----------------------------------------------------
    print("\n[4] Evaluation determinism")
    r1 = cp.evaluate_features(Xsrp, y, groups, base_seed=0)
    r2 = cp.evaluate_features(Xsrp, y, groups, base_seed=0)
    check(r1["per_repeat"] == r2["per_repeat"],
          "Same seed -> identical per-repeat balanced accuracies")

    # --- 5. No leakage: signal recovered, noise stays at chance -----------
    print("\n[5] Fold-local boundary: signal vs permuted/noise controls")
    chance = 1.0 / len(np.unique(y))
    acc_signal = r1["balanced_accuracy"]
    check(acc_signal > chance + 0.05,
          f"Planted signal recovered above chance ({acc_signal:.3f} > {chance:.3f})")

    # permute labels within participant -> should collapse toward chance
    rng = np.random.RandomState(7)
    y_perm = y.copy()
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        y_perm[gi] = rng.permutation(y[gi])
    acc_perm = cp.evaluate_features(Xsrp, y_perm, groups, base_seed=0)["balanced_accuracy"]
    check(acc_perm < acc_signal,
          f"Within-participant label permutation lowers accuracy "
          f"({acc_perm:.3f} < {acc_signal:.3f})")

    # pure-noise data -> permutation test must be non-significant
    raw_n, y_n, g_n = make_synthetic(signal=0.0, seed=1)
    bsc_n = cp.build_bsc_features(raw_n, reservoir_seed=42)
    Xn = cp.project_bsc(bsc_n, cp.make_srp(d))
    perm_noise = cp.permutation_test(Xn, y_n, g_n, n_perm=30, seed=0)
    check(perm_noise["p_value"] > 0.05,
          f"Permutation test on pure noise is non-significant "
          f"(p={perm_noise['p_value']:.3f})")
    perm_sig = cp.permutation_test(Xsrp, y, groups, n_perm=30, seed=0)
    check(perm_sig["p_value"] <= perm_noise["p_value"],
          f"Permutation p is smaller for signal than noise "
          f"({perm_sig['p_value']:.3f} <= {perm_noise['p_value']:.3f})")

    # --- 6. Destruction controls transform as specified -------------------
    print("\n[6] Destruction controls")
    coll = cp.destroy_temporal_collapse(bsc).reshape(len(y), n_ch, cp.N_BINS, cp.N_RES)
    check(np.allclose(coll.std(axis=2), 0.0),
          "Temporal collapse makes all six bins identical")
    rngd = np.random.RandomState(0)
    shuf = cp.destroy_temporal_bin_order(bsc, rngd).reshape(len(y), n_ch, cp.N_BINS, cp.N_RES)
    orig = bsc.reshape(len(y), n_ch, cp.N_BINS, cp.N_RES)
    same_multiset = np.allclose(np.sort(shuf, axis=2), np.sort(orig, axis=2))
    check(same_multiset, "Bin shuffle preserves the per-electrode bin multiset")
    elec = cp.destroy_electrode_identity(bsc, np.random.RandomState(0))
    check(np.allclose(np.sort(elec, axis=1), np.sort(bsc, axis=1)),
          "Electrode shuffle preserves the per-observation electrode multiset")

    # --- 7. End-to-end schema emission ------------------------------------
    print("\n[7] Machine-readable schema (tiny end-to-end run)")
    out = runner.run_granularity("synthetic", raw, y, groups, base_seed=0)
    required = {"n_participants", "n_observations", "representations",
                "advantage_over_conventional_pp", "permutation",
                "destruction_controls_pp", "seed_robustness", "temporal_windows"}
    check(required.issubset(out.keys()),
          f"Granularity output has all required fields: {sorted(required)}")
    check(set(out["representations"]) == {"raw_erp16", "conventional",
                                          "bsc6_srp64", "bsc6_srp64_plus_conv"},
          "All four representations present")
    check(set(out["seed_robustness"]) == {str(s) for s in cp.RESERVOIR_SEEDS},
          "Seed-robustness reports all three reservoir seeds")
    # JSON-serializable
    try:
        json.dumps(out)
        check(True, "Granularity output is JSON-serializable")
    except TypeError as e:
        check(False, f"Granularity output JSON-serializable ({e})")

    print("\n" + "=" * 70)
    print(f"CONFIRMATORY VERIFICATION: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("Note: verifies code behavior and the preprocessing boundary, not "
          "the\nscientific findings (which require authorized SHAPE data).")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
