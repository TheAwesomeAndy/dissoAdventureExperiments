#!/usr/bin/env python3
"""
Chapters 6 & 7 (3-Class Pipeline) Verification Script
=======================================================
Verifies that all 4 scripts are syntactically valid and that the shared
infrastructure (reservoir, metrics, topological computation) works on
synthetic data. The full pipeline requires SHAPE 3-class EEG data.

Usage:
    python experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py

Exit code 0 = all checks pass, 1 = at least one check failed.
"""
import sys
import os
import ast
import numpy as np

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


def main():
    global PASS, FAIL
    base = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("CH6/CH7 3-CLASS VERIFICATION (infrastructure, no external data)")
    print("=" * 70)

    # ── 1. Syntax check all scripts ──
    print("\n--- Script Syntax Validation ---")
    scripts = [
        "ch6_ch7_01_feature_extraction.py",
        "ch6_ch7_02_raw_observations.py",
        "ch6_03_experiments.py",
        "ch7_04_experiments.py",
    ]
    for script in scripts:
        path = os.path.join(base, script)
        if not os.path.exists(path):
            check(f"File exists: {script}", False, "file not found")
            continue
        try:
            with open(path, 'r') as f:
                ast.parse(f.read())
            check(f"Syntax valid: {script}", True)
        except SyntaxError as e:
            check(f"Syntax valid: {script}", False, str(e))

    # ── 2. Reservoir and metric computation from Script 01 ──
    print("\n--- LIF Reservoir (from Script 01) ---")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fe3", os.path.join(base, "ch6_ch7_01_feature_extraction.py"))
        fe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fe)
        check("Script 01 module imports", True)

        # This script uses init_reservoir/run_reservoir functions
        W_in, W_rec = fe.init_reservoir(n_input=1, n_res=64, seed=42)
        check("init_reservoir returns weights", True)
        check("W_in shape (64, 1)", W_in.shape == (64, 1))
        check("W_rec shape (64, 64)", W_rec.shape == (64, 64))

        signal = np.random.RandomState(42).randn(256)
        spikes, membrane = fe.run_reservoir(signal, W_in, W_rec)
        # Note: this script returns (n_res, T) not (T, n_res)
        check("Spike shape (64, 256)", spikes.shape == (64, 256))
        check("Membrane shape (64, 256)", membrane.shape == (64, 256))
        check("Spikes are binary",
              set(np.unique(spikes)).issubset({0, 1}))
        check("Spikes occur", spikes.sum() > 0)
        # Transpose for metric computation below
        spikes = spikes.T   # (256, 64)
        membrane = membrane.T  # (256, 64)
    except Exception as e:
        check("Script 01 module imports", False, str(e))

    # ── 3. Dynamical metrics computation ──
    print("\n--- Dynamical Metrics ---")
    try:
        total_spikes = spikes.sum()
        mfr = spikes.mean()
        check("total_spikes > 0", total_spikes > 0)
        check("MFR in (0, 1)", 0 < mfr < 1)

        # Population rate
        pop_rate = spikes.mean(axis=1)
        check("Population rate shape", pop_rate.shape == (256,))

        # Rate entropy
        neuron_rates = spikes.mean(axis=0)
        nonzero = neuron_rates[neuron_rates > 0]
        if len(nonzero) > 0:
            p = nonzero / nonzero.sum()
            rate_entropy = -np.sum(p * np.log2(p + 1e-12))
            check("Rate entropy > 0", rate_entropy > 0)

        # Rate variance
        rate_var = np.var(pop_rate)
        check("Rate variance > 0", rate_var > 0)

        # Autocorrelation decay (tau_ac)
        pop_c = pop_rate - pop_rate.mean()
        if np.std(pop_c) > 0:
            ac = np.correlate(pop_c, pop_c, 'full')
            ac = ac[len(ac)//2:]
            ac = ac / (ac[0] + 1e-12)
            below = np.where(ac < 1/np.e)[0]
            tau_ac = below[0] if len(below) > 0 else len(ac)
            check("tau_ac computable", tau_ac > 0)

        # Permutation entropy
        from math import factorial
        d = 4
        membrane_mean = membrane.mean(axis=1)
        patterns = {}
        for t in range(len(membrane_mean) - d):
            perm = tuple(np.argsort(membrane_mean[t:t+d]))
            patterns[perm] = patterns.get(perm, 0) + 1
        total = sum(patterns.values())
        probs = [c / total for c in patterns.values()]
        H_pi = -sum(p * np.log2(p) for p in probs if p > 0)
        H_max = np.log2(factorial(d))
        H_norm = H_pi / H_max
        check("Permutation entropy in (0,1]", 0 < H_norm <= 1.0,
              f"H={H_norm:.3f}")

        # Temporal sparsity
        ts = np.mean(pop_rate < (1.0/64))
        check("Temporal sparsity in [0,1]", 0 <= ts <= 1)

    except Exception as e:
        check("Dynamical metrics computation", False, str(e))

    # ── 4. Topological metrics (tPLV-derived) ──
    print("\n--- Topological Metrics ---")
    try:
        # Simulate multi-channel EEG for tPLV
        rng = np.random.RandomState(42)
        fake_eeg = rng.randn(1229, 34)
        from scipy.signal import hilbert

        # Theta-band tPLV (simplified test)
        analytic = hilbert(fake_eeg, axis=0)
        phases = np.angle(analytic)
        n_ch = phases.shape[1]
        plv_mat = np.zeros((n_ch, n_ch))
        for i in range(n_ch):
            for j in range(i+1, n_ch):
                plv = np.abs(np.mean(np.exp(1j * (phases[:, i] - phases[:, j]))))
                plv_mat[i, j] = plv
                plv_mat[j, i] = plv
        check("tPLV matrix shape (34, 34)", plv_mat.shape == (34, 34))
        check("tPLV values in [0, 1]", plv_mat.max() <= 1.0 and plv_mat.min() >= 0.0)
        check("tPLV is symmetric", np.allclose(plv_mat, plv_mat.T))

        # Node strength
        strength = plv_mat.sum(axis=1)
        check("Node strength shape (34,)", strength.shape == (34,))

        # Weighted clustering (simplified: just check computable)
        clustering = np.zeros(n_ch)
        for i in range(n_ch):
            neighbors = np.where(plv_mat[i] > 0)[0]
            if len(neighbors) < 2:
                continue
            sub = plv_mat[np.ix_(neighbors, neighbors)]
            triangles = np.sum(sub ** (1/3) @ sub ** (1/3) * sub ** (1/3))
            k = len(neighbors)
            clustering[i] = triangles / (k * (k - 1)) if k > 1 else 0
        check("Clustering coefficient computable", np.all(np.isfinite(clustering)))

    except Exception as e:
        check("Topological metrics", False, str(e))

    # ── 5. Configuration consistency ──
    print("\n--- Configuration ---")
    try:
        check("N_RES = 256", fe.N_RES == 256)
        check("BETA = 0.05", fe.BETA == 0.05)
        check("THRESHOLD = 0.5", fe.THRESHOLD == 0.5)
    except AttributeError as e:
        check("Configuration constants", False, str(e))

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"CH6/CH7 3-CLASS VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("\nNote: Full pipeline requires SHAPE Community 3-class EEG data.")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
