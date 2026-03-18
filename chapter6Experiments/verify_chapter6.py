#!/usr/bin/env python3
"""
Chapter 6 Verification Script
==============================
Verifies that all Chapter 6 experiment scripts are syntactically valid and
that the core LIF reservoir + dynamical metric computation pipeline produces
correct outputs on synthetic data.

The full pipeline requires the SHAPE Community EEG category data which is
not included in the repository. This script tests the shared infrastructure
(reservoir, metric computation, surrogate generation) on synthetic inputs.

Usage:
    python chapter6Experiments/verify_chapter6.py

Exit code 0 = all checks pass, 1 = at least one check failed.
"""
import sys
import os
import importlib.util
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
    print("CHAPTER 6 VERIFICATION (infrastructure tests, no external data)")
    print("=" * 70)

    # ── 1. Syntax check all scripts ──
    print("\n--- Script Syntax Validation ---")
    scripts = [
        "run_chapter6_exp1_esp (1).py",
        "run_chapter6_exp2_reliability (1).py",
        "run_chapter6_exp3_surrogate (1).py",
        "run_chapter6_exp3_valueadd (1).py",
        "run_chapter6_exp4_dissociation.py",
        "run_chapter6_exp5_interaction.py",
        "run_chapter6_exp6_temporal.py",
        "reproduce_chapter6.py",
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

    # ── 2. Reservoir implementation test ──
    print("\n--- LIF Reservoir (from Exp 6.1) ---")
    # Load exp1 to get the Reservoir class
    exp1_path = os.path.join(base, "run_chapter6_exp1_esp (1).py")
    try:
        spec = importlib.util.spec_from_file_location("exp1", exp1_path)
        exp1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp1)
        check("Experiment 6.1 module imports", True)

        res = exp1.Reservoir(seed=42)
        check("Reservoir instantiates", True)
        check("Input weights shape (256,)", res.Win.shape == (256,))
        check("Recurrent weights shape (256, 256)",
              res.Wrec.shape == (256, 256))

        # Test with synthetic EEG-like signal
        rng = np.random.RandomState(42)
        signal = rng.randn(1229) * 10  # 1229 steps like real SHAPE data
        M, S = res.run(signal)
        check("Reservoir output shapes correct",
              M.shape == (1229, 256) and S.shape == (1229, 256))
        check("Spikes are binary",
              set(np.unique(S)).issubset({0, 1}))
        # Note: Chapter 6 reservoir does NOT enforce a membrane floor
        # (unlike Chapter 5 which clamps to 0). Negative values are
        # biologically unusual but computationally valid for this analysis.
        check("Membrane potentials are finite",
              np.all(np.isfinite(M)))
        check("Spikes occur (not all silent)", S.sum() > 0,
              f"total spikes={S.sum()}")
        check("Reservoir is sparse (< 30% active)",
              S.mean() < 0.3,
              f"mean activity={S.mean():.3f}")

    except Exception as e:
        check("Experiment 6.1 module imports", False, str(e))

    # ── 3. Dynamical metric computation ──
    print("\n--- Dynamical Metrics ---")
    try:
        # Compute metrics that all scripts share
        total_spikes = S.sum()
        mean_firing_rate = S.mean()
        check("total_spikes > 0", total_spikes > 0)
        check("mean_firing_rate in (0, 1)", 0 < mean_firing_rate < 1)

        # Population rate
        pop_rate = S.mean(axis=1)
        check("Population rate shape (1229,)", pop_rate.shape == (1229,))

        # Rate entropy (Shannon entropy of binned firing rates)
        neuron_rates = S.mean(axis=0)
        nonzero = neuron_rates[neuron_rates > 0]
        if len(nonzero) > 0:
            p = nonzero / nonzero.sum()
            rate_entropy = -np.sum(p * np.log2(p + 1e-12))
            check("Rate entropy > 0", rate_entropy > 0)
        else:
            check("Rate entropy computable", False, "no active neurons")

        # Rate variance
        rate_variance = np.var(neuron_rates)
        check("Rate variance > 0", rate_variance > 0)

        # Autocorrelation decay (tau_ac)
        pop_centered = pop_rate - pop_rate.mean()
        if np.std(pop_centered) > 0:
            autocorr = np.correlate(pop_centered, pop_centered, 'full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            below = np.where(autocorr < 1/np.e)[0]
            tau_ac = below[0] if len(below) > 0 else len(autocorr)
            check("tau_ac computable", tau_ac > 0,
                  f"tau_ac={tau_ac} steps")
        else:
            check("tau_ac computable", False, "zero-variance pop rate")

        # Permutation entropy (order d=4)
        from math import factorial
        d = 4
        membrane_mean = M.mean(axis=1)
        patterns = {}
        for t in range(len(membrane_mean) - d):
            perm = tuple(np.argsort(membrane_mean[t:t+d]))
            patterns[perm] = patterns.get(perm, 0) + 1
        total = sum(patterns.values())
        probs = [c / total for c in patterns.values()]
        H_pi = -sum(p * np.log2(p) for p in probs if p > 0)
        H_max = np.log2(factorial(d))
        H_norm = H_pi / H_max
        check("Permutation entropy in (0, 1]",
              0 < H_norm <= 1.0,
              f"H_pi_norm={H_norm:.3f}")

    except Exception as e:
        check("Dynamical metrics computation", False, str(e))

    # ── 4. ESP convergence test ──
    print("\n--- Echo State Property (ESP) Convergence ---")
    try:
        signal = rng.randn(500) * 10
        M1, S1 = res.run(signal, M0=np.zeros(256))
        M2, S2 = res.run(signal, M0=rng.randn(256) * 0.1)
        # After washout, trajectories should converge
        dist_early = np.linalg.norm(M1[10] - M2[10])
        dist_late = np.linalg.norm(M1[400] - M2[400])
        check("ESP: late distance < early distance (trajectories converge)",
              dist_late < dist_early,
              f"early={dist_early:.4f}, late={dist_late:.4f}")
    except Exception as e:
        check("ESP convergence test", False, str(e))

    # ── 5. Surrogate generation ──
    print("\n--- Surrogate Generation ---")
    try:
        # Phase randomization
        signal_test = rng.randn(1024)
        fft = np.fft.rfft(signal_test)
        phases = np.angle(fft)
        random_phases = rng.uniform(-np.pi, np.pi, len(phases))
        random_phases[0] = 0  # preserve DC
        if len(signal_test) % 2 == 0:
            random_phases[-1] = 0  # preserve Nyquist
        surrogate = np.fft.irfft(np.abs(fft) * np.exp(1j * random_phases),
                                  n=len(signal_test))
        check("Phase-randomized surrogate same length",
              len(surrogate) == len(signal_test))
        check("Phase-randomized preserves power spectrum",
              np.allclose(np.abs(np.fft.rfft(surrogate)),
                         np.abs(fft), atol=1e-10))
        check("Phase-randomized changes signal",
              not np.allclose(surrogate, signal_test))

        # Time shuffle
        shuffled = signal_test.copy()
        rng.shuffle(shuffled)
        check("Time-shuffled preserves amplitude distribution",
              np.allclose(sorted(shuffled), sorted(signal_test)))
        check("Time-shuffled changes temporal order",
              not np.allclose(shuffled, signal_test))

    except Exception as e:
        check("Surrogate generation", False, str(e))

    # ── 6. Verification report exists ──
    print("\n--- Documentation ---")
    report = os.path.join(base, "CHAPTER6_VERIFICATION_REPORT.md")
    check("Verification report exists",
          os.path.exists(report))

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"CHAPTER 6 VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("\nNote: Full end-to-end verification requires the SHAPE Community")
    print("EEG category data (categoriesbatch1-4 directories).")
    print("Experiments 6.1-6.6 produce figures and results only with real data.")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
