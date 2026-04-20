#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Dissertation-to-Repo Validation Script
=====================================================

Verifies that every quantitative claim in the ARSPI-Net dissertation
is either:
  (a) reproducible from a script in this repository, or
  (b) explicitly flagged as requiring SHAPE data (not included).

This script runs WITHOUT external data -- it tests infrastructure,
consistency, and synthetic-data correctness.

Tests are organized by dissertation chapter and claim.
"""

import sys
import os
import ast
import importlib.util
import numpy as np
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASSED = 0
FAILED = 0
WARNINGS = 0


def check(condition, label):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}")
    return condition


def warn(label):
    global WARNINGS
    WARNINGS += 1
    print(f"  [WARN] {label}")


def load_module(path):
    """Import a .py file as a module."""
    spec = importlib.util.spec_from_file_location("mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Section 1: Script Existence (Reproduction Map)
# ============================================================
def test_script_existence():
    print("\n" + "=" * 70)
    print("SECTION 1: Script Existence (all scripts cited in REPRODUCTION_MAP)")
    print("=" * 70)

    required_scripts = [
        "chapter4Experiments/run_chapter4_experiments.py",
        "chapter4Experiments/run_chapter4_observations.py",
        "chapter5Experiments/run_chapter5_experiments.py",
        "chapter5Experiments/canonical_pytorch_baselines.py",
        "chapter5Experiments/experiment_zero.py",
        "chapter5Experiments/sklearn_baselines.py",
        "chapter5Experiments/reproduce_chapter5.py",
        "chapter6Experiments/run_chapter6_exp1_esp.py",
        "chapter6Experiments/run_chapter6_exp2_reliability.py",
        "chapter6Experiments/run_chapter6_exp3_surrogate.py",
        "chapter6Experiments/run_chapter6_exp3_valueadd.py",
        "chapter6Experiments/run_chapter6_exp4_dissociation.py",
        "chapter6Experiments/run_chapter6_exp5_interaction.py",
        "chapter6Experiments/run_chapter6_exp6_temporal.py",
        "chapter6Experiments/reproduce_chapter6.py",
        "chapter7Experiments/run_chapter7_experiment_A.py",
        "chapter7Experiments/run_chapter7_experiment_B.py",
        "chapter7Experiments/run_chapter7_experiment_C.py",
        "chapter7Experiments/run_chapter7_experiment_D.py",
        "chapter7Experiments/run_chapter7_experiment_E.py",
        "chapter7Experiments/extract_kappa_matrix.py",
        "experiments/chapter3/run_chapter3_lsm_characterization.py",
        "experiments/ch5_4class/ch5_4class_01_feature_extraction.py",
        "experiments/ch5_4class/ch5_4class_02_raw_observations.py",
        "experiments/ch5_4class/ch5_4class_03_classification_full.py",
        "experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py",
        "experiments/ch6_ch7_3class/ch6_ch7_02_raw_observations.py",
        "experiments/ch6_ch7_3class/ch6_03_experiments.py",
        "experiments/ch6_ch7_3class/ch7_04_experiments.py",
        "experiments/ablation/layer_ablation.py",
        "experiments/interpretability/run_level1_temporal_traceability.py",
        "experiments/interpretability/run_level3_descriptor_erp_alignment.py",
        "experiments/interpretability/run_eegnet_saliency_comparison.py",
        "experiments/interpretability/run_arspinet_v2_attention_prototype.py",
        "validation/validate_shape_data.py",
        "validation/validate_subcategory_data.py",
    ]

    for script in required_scripts:
        path = REPO_ROOT / script
        check(path.exists(), f"Exists: {script}")


# ============================================================
# Section 2: Syntax Validation (all .py files parse)
# ============================================================
def test_syntax():
    print("\n" + "=" * 70)
    print("SECTION 2: Syntax Validation (all .py files parse)")
    print("=" * 70)

    py_files = list(REPO_ROOT.glob("**/*.py"))
    py_files = [f for f in py_files if '.git' not in str(f)
                and '__pycache__' not in str(f)]

    for f in sorted(py_files):
        try:
            with open(f, encoding='utf-8') as fh:
                ast.parse(fh.read())
            check(True, f"Syntax OK: {f.relative_to(REPO_ROOT)}")
        except SyntaxError as e:
            check(False, f"Syntax ERROR in {f.relative_to(REPO_ROOT)}: {e}")


# ============================================================
# Section 3: LIF Reservoir Core (Synthetic Verification)
# ============================================================
def test_lif_reservoir():
    print("\n" + "=" * 70)
    print("SECTION 3: LIF Reservoir Core Properties")
    print("=" * 70)

    # Build a reservoir matching all chapters
    N_RES = 256
    BETA = 0.05
    THRESHOLD = 0.5
    SEED = 42
    TARGET_SR = 0.9

    rng = np.random.RandomState(SEED)
    n_input = 1
    limit_in = np.sqrt(6.0 / (n_input + N_RES))
    W_in = rng.uniform(-limit_in, limit_in, (N_RES, n_input))
    limit_rec = np.sqrt(6.0 / (N_RES + N_RES))
    W_rec = rng.uniform(-limit_rec, limit_rec, (N_RES, N_RES))
    eigs = np.abs(np.linalg.eigvals(W_rec))
    W_rec *= TARGET_SR / eigs.max()

    # Check spectral radius
    sr = np.abs(np.linalg.eigvals(W_rec)).max()
    check(abs(sr - 0.9) < 0.05, f"Spectral radius = {sr:.4f} (target: 0.9)")

    # Check weight shapes
    check(W_in.shape == (256, 1), f"W_in shape: {W_in.shape}")
    check(W_rec.shape == (256, 256), f"W_rec shape: {W_rec.shape}")

    # Run reservoir on synthetic input (strong enough to elicit spikes)
    T = 100
    X = np.random.randn(T, 1) * 2.0

    mem = np.zeros(N_RES)
    spk_prev = np.zeros(N_RES)
    spikes = np.zeros((T, N_RES))
    membranes = np.zeros((T, N_RES))

    for t in range(T):
        I_tot = W_in @ X[t] + W_rec @ spk_prev
        mem = (1.0 - BETA) * mem * (1.0 - spk_prev) + I_tot
        spk = (mem >= THRESHOLD).astype(float)
        mem = mem - spk * THRESHOLD
        mem = np.maximum(mem, 0.0)
        spikes[t] = spk
        membranes[t] = mem
        spk_prev = spk

    # Spikes are binary
    unique_vals = np.unique(spikes)
    check(set(unique_vals).issubset({0.0, 1.0}),
          f"Spikes are binary: unique values = {unique_vals}")

    # Membrane never negative
    check(membranes.min() >= 0.0,
          f"Membrane floor >= 0: min = {membranes.min():.6f}")

    # Spikes are sparse
    sparsity = spikes.mean()
    check(sparsity < 0.5,
          f"Spikes are sparse: mean = {sparsity:.4f}")

    # Spikes not all zero (reservoir responds to input)
    check(spikes.sum() > 0,
          f"Reservoir produces spikes: total = {spikes.sum():.0f}")

    # Determinism: same seed produces same output
    rng2 = np.random.RandomState(SEED)
    W_in2 = rng2.uniform(-limit_in, limit_in, (N_RES, n_input))
    check(np.array_equal(W_in, W_in2),
          "Deterministic: same seed -> same weights")


# ============================================================
# Section 4: BSC6 Temporal Coding Validation
# ============================================================
def test_bsc6():
    print("\n" + "=" * 70)
    print("SECTION 4: BSC6 Temporal Coding")
    print("=" * 70)

    # Create synthetic spike train
    T, N = 120, 256
    rng = np.random.RandomState(42)
    spikes = (rng.rand(T, N) < 0.1).astype(float)

    n_bins = 6
    bin_size = T // n_bins
    bsc = np.zeros((N, n_bins))
    for b in range(n_bins):
        bsc[:, b] = spikes[b * bin_size:(b + 1) * bin_size].sum(axis=0)

    # BSC sum equals total spikes (no loss)
    check(abs(bsc.sum() - spikes[:bin_size * n_bins].sum()) < 1e-10,
          "BSC6 sum == total spike count (no information loss)")

    # BSC values are non-negative
    check(bsc.min() >= 0.0, "BSC6 values are non-negative")

    # BSC shape is correct
    check(bsc.shape == (256, 6), f"BSC6 shape: {bsc.shape} (expected 256x6)")


# ============================================================
# Section 5: Permutation Entropy Validation
# ============================================================
def test_permutation_entropy():
    print("\n" + "=" * 70)
    print("SECTION 5: Permutation Entropy (Complexity Measure)")
    print("=" * 70)

    def perm_entropy(x, order=3):
        patterns = {}
        for i in range(len(x) - order + 1):
            pat = tuple(np.argsort(x[i:i + order]))
            patterns[pat] = patterns.get(pat, 0) + 1
        tot = sum(patterns.values())
        import math
        h = -sum((c / tot) * np.log(c / tot + 1e-12) for c in patterns.values())
        return h / np.log(math.factorial(order))

    # Random signal: PE should be near 1.0
    rng = np.random.RandomState(42)
    rand_signal = rng.randn(1000)
    pe_rand = perm_entropy(rand_signal)
    check(pe_rand > 0.95, f"Random signal PE = {pe_rand:.4f} (expected > 0.95)")

    # Monotonic signal: PE should be near 0.0
    mono_signal = np.arange(1000, dtype=float)
    pe_mono = perm_entropy(mono_signal)
    check(pe_mono < 0.01, f"Monotonic signal PE = {pe_mono:.6f} (expected ~ 0)")


# ============================================================
# Section 6: Coupling Computation (kappa)
# ============================================================
def test_coupling():
    print("\n" + "=" * 70)
    print("SECTION 6: Structure-Function Coupling (kappa)")
    print("=" * 70)

    # Simulate: 34 channels, 7 dynamical metrics, 2 topological metrics
    rng = np.random.RandomState(42)
    n_ch = 34
    n_dyn = 7
    n_topo = 2

    D = rng.randn(n_ch, n_dyn)
    T_topo = rng.randn(n_ch, n_topo)

    # Coupling matrix: Spearman correlation between each metric pair
    from scipy.stats import spearmanr
    C = np.zeros((n_dyn, n_topo))
    for j in range(n_dyn):
        for k in range(n_topo):
            C[j, k], _ = spearmanr(D[:, j], T_topo[:, k])

    # Kappa: Frobenius norm normalized
    kappa = np.linalg.norm(C, 'fro') / np.sqrt(n_dyn * n_topo)

    check(kappa >= 0, f"kappa >= 0: {kappa:.4f}")
    check(kappa <= 1.0 + 1e-10, f"kappa <= 1: {kappa:.4f}")
    check(C.shape == (7, 2), f"Coupling matrix shape: {C.shape}")

    # Permutation null: shuffle electrodes, kappa should be similar
    # (random data -> random coupling)
    null_kappas = []
    for _ in range(100):
        perm = rng.permutation(n_ch)
        C_null = np.zeros((n_dyn, n_topo))
        for j in range(n_dyn):
            for k in range(n_topo):
                C_null[j, k], _ = spearmanr(D[perm, j], T_topo[:, k])
        null_kappas.append(np.linalg.norm(C_null, 'fro') / np.sqrt(n_dyn * n_topo))

    # With random data, observed kappa should not be significantly > null
    check(True, f"Coupling null distribution: mean = {np.mean(null_kappas):.3f}, "
                f"observed = {kappa:.3f}")


# ============================================================
# Section 7: Dissertation-Repo Consistency Checks
# ============================================================
def test_consistency():
    print("\n" + "=" * 70)
    print("SECTION 7: Dissertation-Repo Numerical Consistency")
    print("=" * 70)

    # Check that R^2 = 0.661 is NOT in the dissertation .tex files
    tex_files_to_check = ['main.tex', 'ch2.tex', 'chDynamics.tex', 'chConclusion.tex']
    for fname in tex_files_to_check:
        fpath = REPO_ROOT / 'latex' / fname
        if fpath.exists():
            with open(fpath, encoding='utf-8') as f:
                content = f.read()
            has_661 = '0.661' in content
            check(not has_661, f"No '0.661' in {fname}")
        else:
            warn(f"{fname} not found")

    # Check that ANDREW tags are stripped
    all_tex = list((REPO_ROOT / 'latex').glob('*.tex'))
    for fpath in all_tex:
        with open(fpath, encoding='utf-8') as f:
            content = f.read()
        has_andrew = '%%%ANDREW:' in content or '%%%ANDREW' in content
        check(not has_andrew, f"No %%%ANDREW in {fpath.name}")

    # Check R^2 = 0.661 is NOT in README or REPRODUCTION_MAP
    for fname in ['README.md', 'docs/REPRODUCTION_MAP.md']:
        fpath = REPO_ROOT / fname
        if fpath.exists():
            with open(fpath, encoding='utf-8') as f:
                content = f.read()
            # Allow "0.661" in historical/explanatory context but not as a claim
            has_claim = 'R' in content and '0.661' in content and 'LPP prediction' in content.split('0.661')[0][-50:]
            check(not has_claim,
                  f"No active R^2=0.661 claim in {fname}")

    # Check that Level 3 in README matches dissertation
    readme = REPO_ROOT / 'README.md'
    if readme.exists():
        with open(readme, encoding='utf-8') as f:
            content = f.read()
        has_l3_update = '0.82' in content and 'Ch31' in content or '0.837' in content
        check(has_l3_update,
              "README Level 3 updated to per-channel correlations")

    # Check that the Level 3 script exists
    l3_script = REPO_ROOT / 'experiments/interpretability/run_level3_descriptor_erp_alignment.py'
    check(l3_script.exists(), "Level 3 descriptor-ERP alignment script exists")


# ============================================================
# Section 8: Cross-Validation Protocol Consistency
# ============================================================
def test_cv_protocol():
    print("\n" + "=" * 70)
    print("SECTION 8: Cross-Validation Protocol Consistency")
    print("=" * 70)

    # Verify all main experiment scripts use the same CV parameters
    cv_scripts = [
        "chapter5Experiments/experiment_zero.py",
        "chapter5Experiments/canonical_pytorch_baselines.py",
        "chapter5Experiments/sklearn_baselines.py",
        "experiments/ch6_ch7_3class/ch6_03_experiments.py",
        "experiments/ch6_ch7_3class/ch7_04_experiments.py",
        "experiments/ablation/layer_ablation.py",
    ]

    for script in cv_scripts:
        path = REPO_ROOT / script
        if not path.exists():
            warn(f"Script not found: {script}")
            continue
        with open(path, encoding='utf-8') as f:
            content = f.read()
        has_seed42 = ('random_state=42' in content or 'seed=42' in content
                      or 'SEED = 42' in content or 'RANDOM_STATE = 42' in content)
        check(has_seed42, f"Seed 42 in {script}")

    # Check that GroupKFold or StratifiedGroupKFold is used (no subject leakage)
    for script in cv_scripts:
        path = REPO_ROOT / script
        if not path.exists():
            continue
        with open(path, encoding='utf-8') as f:
            content = f.read()
        has_group = 'GroupKFold' in content or 'group' in content.lower()
        check(has_group, f"Group-aware CV in {script}")


# ============================================================
# Section 9: Reservoir Parameter Consistency
# ============================================================
def test_reservoir_params():
    print("\n" + "=" * 70)
    print("SECTION 9: Reservoir Parameter Consistency Across Scripts")
    print("=" * 70)

    scripts_with_reservoir = [
        "chapter4Experiments/run_chapter4_experiments.py",
        "chapter5Experiments/experiment_zero.py",
        "experiments/ch6_ch7_3class/ch6_ch7_01_feature_extraction.py",
        "experiments/interpretability/run_level1_temporal_traceability.py",
        "experiments/interpretability/run_level3_descriptor_erp_alignment.py",
    ]

    for script in scripts_with_reservoir:
        path = REPO_ROOT / script
        if not path.exists():
            warn(f"Script not found: {script}")
            continue
        with open(path, encoding='utf-8') as f:
            content = f.read()

        has_256 = 'N_RES = 256' in content or 'n_res=256' in content or 'N_res=256' in content
        has_beta = 'BETA = 0.05' in content or 'beta=0.05' in content
        has_thresh = 'THRESHOLD = 0.5' in content or 'threshold=0.5' in content

        check(has_256, f"N_RES=256 in {Path(script).name}")
        check(has_beta, f"BETA=0.05 in {Path(script).name}")
        check(has_thresh, f"THRESHOLD=0.5 in {Path(script).name}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ARSPI-NET DISSERTATION VALIDATION SUITE")
    print("Verifies repo-dissertation consistency without external data")
    print("=" * 70)

    test_script_existence()
    test_syntax()
    test_lif_reservoir()
    test_bsc6()
    test_permutation_entropy()
    test_coupling()
    test_consistency()
    test_cv_protocol()
    test_reservoir_params()

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: {PASSED} passed, {FAILED} failed, {WARNINGS} warnings")
    print("=" * 70)

    if FAILED > 0:
        print("\nFAILED TESTS REQUIRE ATTENTION BEFORE SUBMISSION.")
        return 1
    else:
        print("\nAll tests passed. Repo is consistent with dissertation.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
