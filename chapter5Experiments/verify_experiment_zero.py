#!/usr/bin/env python3
"""
Verification script for experiment_zero.py (Baseline Disambiguation).

Tests all core components on synthetic data without requiring the SHAPE dataset.
Validates that the script's infrastructure is correct and produces expected output
structure. Does NOT test classification accuracy on real data (requires SHAPE EEG).

Run:
    python chapter5Experiments/verify_experiment_zero.py
"""

import sys
import os

# Windows cp1252 portability: scripts print Unicode box-drawing chars (─, ═)
# and read source files containing UTF-8 (µ, ≈, ≥). Without this, they crash
# on default Windows consoles. Python 3.7+ has reconfigure; older silently skip.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass
import numpy as np
import tempfile
import pickle

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def main():
    global PASS, FAIL
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("VERIFICATION: experiment_zero.py (Baseline Disambiguation)")
    print("=" * 70)

    # ── 1. Syntax validation ──
    print("\n── Syntax Validation ──")
    exp_zero_path = os.path.join(script_dir, "experiment_zero.py")
    try:
        with open(exp_zero_path, encoding='utf-8') as f:
            compile(f.read(), exp_zero_path, 'exec')
        check("experiment_zero.py parses without syntax errors", True)
    except SyntaxError as e:
        check("experiment_zero.py parses without syntax errors", False, str(e))

    # ── 2. Import validation ──
    print("\n── Import Validation ──")
    sys.path.insert(0, script_dir)
    try:
        from experiment_zero import (LIFReservoir, bsc6_encode, flatten_raw,
                                      subject_center, evaluate)
        check("All core functions importable", True)
    except ImportError as e:
        check("All core functions importable", False, str(e))
        print(f"\nEARLY EXIT: Cannot import core functions.")
        sys.exit(1)

    # ── 3. LIF Reservoir tests ──
    print("\n── LIF Reservoir ──")
    res = LIFReservoir(n_res=64, beta=0.05, threshold=0.5, seed=42)
    check("Reservoir instantiates (n_res=64)", True)
    check("W_in shape is (64, 1)", res.W_in.shape == (64, 1))
    check("W_rec shape is (64, 64)", res.W_rec.shape == (64, 64))

    # Spectral radius check
    eig = np.abs(np.linalg.eigvals(res.W_rec)).max()
    check(f"Spectral radius ~0.9 (actual: {eig:.4f})", abs(eig - 0.9) < 0.05)

    # Forward pass
    x = np.random.randn(100)
    spikes = res.forward(x)
    check("Forward pass returns (100, 64) array", spikes.shape == (100, 64))
    check("Spikes are binary (0 or 1)", set(np.unique(spikes)).issubset({0.0, 1.0}))
    check("Reservoir produces non-silent output", spikes.sum() > 0)

    # ── 4. BSC6 encoding ──
    print("\n── BSC6 Encoding ──")
    bsc = bsc6_encode(spikes, n_bins=6)
    expected_dim = 6 * 64  # n_bins * n_res
    check(f"BSC6 produces {expected_dim}-dim vector", len(bsc) == expected_dim)
    check("BSC6 values are non-negative", np.all(bsc >= 0))
    check("BSC6 has nonzero entries", np.sum(bsc > 0) > 0)
    # BSC6 may exclude trailing timesteps (T % n_bins remainder), so sum <= total spikes
    check("BSC6 sum <= total spikes (remainder excluded)",
          bsc.sum() <= spikes.sum() + 1e-10,
          f"BSC sum={bsc.sum()}, spike sum={spikes.sum()}")

    # ── 5. Flatten raw ──
    print("\n── Flatten Raw ──")
    raw_data = np.random.randn(10, 256, 34)
    flat = flatten_raw(raw_data)
    check("flatten_raw: (10,256,34) -> (10, 8704)", flat.shape == (10, 256 * 34))
    check("Flattening preserves values", np.allclose(flat[0, :34], raw_data[0, 0, :]))

    # ── 6. Subject centering ──
    print("\n── Subject Centering ──")
    features = np.random.randn(12, 50)
    subjects = np.array(["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3 + ["s4"] * 3)

    centered = subject_center(features, subjects)
    check("subject_center returns same shape", centered.shape == features.shape)
    check("Centering changes values", not np.allclose(centered, features))

    # Verify per-subject mean is zero after centering
    all_zero = True
    for s in np.unique(subjects):
        mask = subjects == s
        per_subj_mean = centered[mask].mean(axis=0)
        if not np.allclose(per_subj_mean, 0, atol=1e-10):
            all_zero = False
    check("Per-subject mean is zero after centering", all_zero)

    # Centering does not modify original
    features_copy = features.copy()
    _ = subject_center(features_copy, subjects)
    check("subject_center does not modify input array", np.allclose(features, features_copy))

    # ── 7. Evaluate function (CV pipeline) ──
    print("\n── Evaluate Function (CV Pipeline) ──")
    np.random.seed(42)
    n_samples = 60
    n_features = 20
    n_classes = 3
    features_synth = np.random.randn(n_samples, n_features)
    y_synth = np.repeat([0, 1, 2], n_samples // n_classes)
    subjects_synth = np.array([f"s{i}" for i in range(n_samples // n_classes)
                                for _ in range(n_classes)])

    fold_accs = evaluate(features_synth, y_synth, subjects_synth,
                         "Synthetic test", n_folds=5)
    check("evaluate returns numpy array", isinstance(fold_accs, np.ndarray))
    check("evaluate returns 5 fold accuracies", len(fold_accs) == 5)
    check("All fold accuracies in [0, 1]", np.all((fold_accs >= 0) & (fold_accs <= 1)))
    check("Mean accuracy is finite", np.isfinite(fold_accs.mean()))

    # ── 8. Reservoir determinism ──
    print("\n── Reservoir Determinism ──")
    res1 = LIFReservoir(n_res=64, seed=42)
    res2 = LIFReservoir(n_res=64, seed=42)
    x_test = np.random.randn(50)
    s1 = res1.forward(x_test)
    s2 = res2.forward(x_test)
    check("Same seed produces identical spikes", np.array_equal(s1, s2))

    res3 = LIFReservoir(n_res=64, seed=99)
    s3 = res3.forward(x_test)
    check("Different seed produces different spikes", not np.array_equal(s1, s3))

    # ── 9. Channel-specific seeding ──
    print("\n── Channel-Specific Seeding ──")
    reservoirs = {ch: LIFReservoir(n_res=64, seed=42 + ch * 17) for ch in range(5)}
    spikes_ch = {ch: reservoirs[ch].forward(x_test) for ch in range(5)}
    all_different = all(not np.array_equal(spikes_ch[0], spikes_ch[ch])
                        for ch in range(1, 5))
    check("Different channels produce different reservoir responses", all_different)

    # ── 10. End-to-end mini pipeline ──
    print("\n── End-to-End Mini Pipeline ──")
    np.random.seed(42)
    n_subj, n_conds, T, n_ch = 6, 3, 100, 2
    raw = np.random.randn(n_subj * n_conds, T, n_ch)
    y = np.tile(np.arange(n_conds), n_subj)
    subjs = np.repeat([f"s{i}" for i in range(n_subj)], n_conds)

    # Raw path
    flat = flatten_raw(raw)
    flat_c = subject_center(flat, subjs)
    check("Raw flat shape correct", flat.shape == (n_subj * n_conds, T * n_ch))
    check("Centered flat shape correct", flat_c.shape == flat.shape)

    # Reservoir path (small for speed)
    embeddings = []
    for i in range(len(raw)):
        ch_feats = []
        for ch in range(n_ch):
            r = LIFReservoir(n_res=32, seed=42 + ch * 17)
            sp = r.forward(raw[i, :, ch])
            ch_feats.append(bsc6_encode(sp, n_bins=6))
        embeddings.append(np.concatenate(ch_feats))
    embedding = np.array(embeddings)
    check("Embedding shape is (N, n_ch * 6 * 32)",
          embedding.shape == (n_subj * n_conds, n_ch * 6 * 32))

    embedding_c = subject_center(embedding, subjs)
    check("Centered embedding shape matches", embedding_c.shape == embedding.shape)

    # ── 11. Argparse validation ──
    print("\n── Argparse / Script Structure ──")
    from experiment_zero import main as exp_main
    import inspect
    sig = inspect.signature(exp_main)
    check("main() takes no arguments (uses argparse)", len(sig.parameters) == 0)

    # Check the script has all 4 experimental conditions defined
    with open(exp_zero_path, encoding='utf-8') as f:
        source = f.read()
    check("Script references 'raw_uncentered'", "raw_uncentered" in source)
    check("Script references 'raw_centered'", "raw_centered" in source)
    check("Script references 'res_uncentered'", "res_uncentered" in source)
    check("Script references 'res_centered'", "res_centered" in source)
    check("Script includes SVM cross-check", "SVM" in source or "SVC" in source)
    check("Script saves pickle output", "pickle.dump" in source)

    # ── Summary ──
    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"RESULT: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("All tests passed.")
    print("=" * 70)
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
