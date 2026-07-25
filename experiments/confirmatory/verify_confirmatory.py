#!/usr/bin/env python3
"""Data-independent verification of the ARSPI-Net confirmatory pipeline.

The checks establish implementation behavior on synthetic data. They do not
establish the empirical SHAPE results, which require authorized data.
"""

import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.random_projection import SparseRandomProjection

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import confirmatory_pipeline as cp
import run_confirmatory_validation as runner

PASS = 0
FAIL = 0


def check(condition, label):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")
    return condition


def make_synthetic(
    n_participants=25,
    n_conditions=3,
    time_steps=64,
    n_channels=4,
    signal=1.0,
    seed=0,
):
    rng = np.random.RandomState(seed)
    templates = rng.randn(n_conditions, time_steps, n_channels)
    raw, labels, groups = [], [], []
    for participant in range(n_participants):
        nuisance = rng.randn(n_channels) * 5.0
        for condition in range(n_conditions):
            epoch = (
                signal * templates[condition]
                + nuisance[None, :]
                + rng.randn(time_steps, n_channels) * 0.5
            )
            raw.append(epoch)
            labels.append(condition)
            groups.append(participant)
    return np.asarray(raw), np.asarray(labels), np.asarray(groups)


def main():
    print("=" * 70)
    print("CONFIRMATORY PIPELINE VERIFICATION")
    print("=" * 70)

    feature_dim = cp.N_BINS * cp.N_RES
    srp_a = cp.make_srp(feature_dim)
    srp_b = cp.make_srp(feature_dim)
    check(
        np.array_equal(srp_a.components_.toarray(), srp_b.components_.toarray()),
        "fixed SRP is deterministic",
    )

    srp_c = SparseRandomProjection(
        n_components=cp.N_SRP,
        random_state=cp.SRP_SEED,
    )
    srp_c.fit(np.random.RandomState(1).randn(50, feature_dim))
    check(
        np.array_equal(srp_a.components_.toarray(), srp_c.components_.toarray()),
        "SRP matrix is independent of observed values",
    )
    check(srp_a.n_components == cp.N_SRP, "SRP output dimension is 64")

    rng = np.random.RandomState(2)
    bsc_probe = rng.rand(6, 4, feature_dim)
    projected = cp.project_bsc(bsc_probe, srp_a)
    check(
        projected.shape == (6, 4 * cp.N_SRP),
        "projection preserves four electrode blocks",
    )
    altered = bsc_probe.copy()
    altered[:, 1, :] += 3.0
    projected_altered = cp.project_bsc(altered, cp.make_srp(feature_dim))
    block0_same = np.allclose(
        projected[:, : cp.N_SRP], projected_altered[:, : cp.N_SRP]
    )
    block1_changed = not np.allclose(
        projected[:, cp.N_SRP : 2 * cp.N_SRP],
        projected_altered[:, cp.N_SRP : 2 * cp.N_SRP],
    )
    check(block0_same and block1_changed, "electrode projection blocks are independent")

    raw, y, groups = make_synthetic()
    splitter = StratifiedGroupKFold(
        n_splits=cp.N_FOLDS,
        shuffle=True,
        random_state=0,
    )
    leakage = any(
        set(groups[train_idx]) & set(groups[test_idx])
        for train_idx, test_idx in splitter.split(np.zeros((len(y), 1)), y, groups)
    )
    check(not leakage, "participant-grouped folds contain no subject leakage")

    print("\nBuilding synthetic reservoir representation ...")
    bsc = cp.build_bsc_features(raw, reservoir_seed=42)
    X = cp.project_bsc(bsc, cp.make_srp(feature_dim))
    evaluation_a = cp.evaluate_features(X, y, groups, base_seed=0)
    evaluation_b = cp.evaluate_features(X, y, groups, base_seed=0)

    expected_rows = cp.N_REPEATS * len(y)
    check(
        len(evaluation_a["oof_pred"]) == expected_rows,
        "OOF predictions pool every observation from every repeat",
    )
    check(
        np.array_equal(evaluation_a["oof_true"], np.tile(y, cp.N_REPEATS))
        and np.array_equal(
            evaluation_a["oof_group"], np.tile(groups, cp.N_REPEATS)
        ),
        "pooled labels and participant identifiers align across repeats",
    )
    check(
        evaluation_a["per_repeat"] == evaluation_b["per_repeat"]
        and np.array_equal(evaluation_a["oof_pred"], evaluation_b["oof_pred"]),
        "evaluation is deterministic for fixed seeds",
    )

    chance = 1.0 / len(np.unique(y))
    check(
        evaluation_a["balanced_accuracy"] > chance + 0.05,
        "planted condition signal is recovered above chance",
    )

    y_permuted = y.copy()
    permutation_rng = np.random.RandomState(7)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        y_permuted[idx] = permutation_rng.permutation(y[idx])
    permuted_accuracy = cp.evaluate_features(
        X, y_permuted, groups, base_seed=0
    )["balanced_accuracy"]
    check(
        permuted_accuracy < evaluation_a["balanced_accuracy"],
        "within-participant label permutation lowers balanced accuracy",
    )

    raw_noise, y_noise, groups_noise = make_synthetic(signal=0.0, seed=1)
    bsc_noise = cp.build_bsc_features(raw_noise, reservoir_seed=42)
    X_noise = cp.project_bsc(bsc_noise, cp.make_srp(feature_dim))
    noise_test = cp.permutation_test(
        X_noise, y_noise, groups_noise, n_perm=30, seed=0
    )
    signal_test = cp.permutation_test(X, y, groups, n_perm=30, seed=0)
    check(noise_test["p_value"] > 0.05, "pure-noise permutation test is non-significant")
    check(
        signal_test["p_value"] <= noise_test["p_value"],
        "signal permutation p-value is no larger than noise p-value",
    )

    paired_ci = cp.participant_bootstrap_difference_ci(
        evaluation_a["oof_true"],
        evaluation_a["oof_pred"],
        evaluation_a["oof_pred"],
        evaluation_a["oof_group"],
        n_boot=100,
        seed=5,
    )
    check(
        np.all(np.isfinite(paired_ci)) and np.allclose(paired_ci, [0.0, 0.0]),
        "paired bootstrap returns a zero interval for identical predictions",
    )

    collapsed = cp.destroy_temporal_collapse(bsc).reshape(
        len(y), 4, cp.N_BINS, cp.N_RES
    )
    check(
        np.allclose(collapsed.std(axis=2), 0.0),
        "temporal collapse makes all six bins identical",
    )

    shuffled = cp.destroy_temporal_bin_order(
        bsc, np.random.RandomState(0)
    ).reshape(len(y), 4, cp.N_BINS, cp.N_RES)
    original = bsc.reshape(len(y), 4, cp.N_BINS, cp.N_RES)
    check(
        np.allclose(np.sort(shuffled, axis=2), np.sort(original, axis=2)),
        "temporal-bin shuffle preserves each bin multiset",
    )

    electrode_shuffled = cp.destroy_electrode_identity(
        bsc, np.random.RandomState(0)
    )
    check(
        np.allclose(np.sort(electrode_shuffled, axis=1), np.sort(bsc, axis=1)),
        "electrode shuffle preserves each electrode multiset",
    )

    time = np.arange(1229, dtype=float)
    epoch_a = np.column_stack(
        [np.sin(2 * np.pi * (channel + 1) * time / 1229) for channel in range(34)]
    )
    epoch_b = epoch_a.copy()
    epoch_b[: runner.PRESTIMULUS_SAMPLES] += 1_000.0
    processed_a = runner.preprocess_epoch(epoch_a)
    processed_b = runner.preprocess_epoch(epoch_b)
    preprocessing_ok = (
        processed_a.shape == (runner.TARGET_TIMESTEPS, runner.EXPECTED_CHANNELS)
        and np.allclose(processed_a, processed_b)
        and np.allclose(processed_a.mean(axis=0), 0.0, atol=1e-10)
        and np.allclose(processed_a.std(axis=0), 1.0, atol=1e-10)
    )
    check(
        preprocessing_ok,
        "preprocessing removes prestimulus samples, decimates, and z-scores",
    )

    print("\n" + "=" * 70)
    print(f"CONFIRMATORY VERIFICATION: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print(
        "These checks verify code behavior and preprocessing boundaries, not "
        "the empirical SHAPE findings."
    )
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
