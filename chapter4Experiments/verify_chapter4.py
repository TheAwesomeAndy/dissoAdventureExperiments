#!/usr/bin/env python3
"""
Chapter 4 Verification Script
==============================
Runs both Chapter 4 scripts and checks that all claimed results hold.
No external data required — uses synthetic data only.

Usage:
    python chapter4Experiments/verify_chapter4.py

Exit code 0 = all checks pass, 1 = at least one check failed.
"""
import subprocess
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
import re
from pathlib import Path

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
    os.chdir(Path(__file__).resolve().parent.parent)

    print("=" * 70)
    print("CHAPTER 4 VERIFICATION")
    print("=" * 70)

    # ── Run experiments script ──
    print("\n[1/2] Running run_chapter4_experiments.py...")
    r1 = subprocess.run(
        [sys.executable, "chapter4Experiments/run_chapter4_experiments.py"],
        capture_output=True, text=True, timeout=600,
        encoding='utf-8', errors='replace',
        env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
    )
    out1 = r1.stdout + r1.stderr
    check("Experiments script exits cleanly", r1.returncode == 0,
          f"exit code {r1.returncode}")

    # ── Run observations script ──
    print("\n[2/2] Running run_chapter4_observations.py...")
    r2 = subprocess.run(
        [sys.executable, "chapter4Experiments/run_chapter4_observations.py"],
        capture_output=True, text=True, timeout=600,
        encoding='utf-8', errors='replace',
        env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
    )
    out2 = r2.stdout + r2.stderr
    check("Observations script exits cleanly", r2.returncode == 0,
          f"exit code {r2.returncode}")

    # ── Check generated figures ──
    print("\n--- Figure Generation ---")
    fig_dir = Path("pictures/chLSMEmbeddings")
    expected_exp_figs = [
        "ablation_reservoir_size.pdf",
        "fdr_three_way_comparison.pdf",
        "coding_scheme_accuracy_comparison.pdf",
        "pca_explained_variance.pdf",
        "pca_component_visualization.pdf",
        "cross_initialization_robustness.pdf",
        "parameter_sensitivity_heatmap.pdf",
    ]
    expected_obs_figs = [
        "obs01_raw_input_signals.pdf",
        "obs02_raw_spike_rasters.pdf",
        "obs03_raw_bsc6_features.pdf",
        "obs04_raw_embedding_space.pdf",
        "obs05_population_dynamics.pdf",
        "obs06_membrane_dynamics.pdf",
    ]
    for f in expected_exp_figs + expected_obs_figs:
        path = fig_dir / f
        check(f"Figure exists: {f}", path.exists() and path.stat().st_size > 0)

    # ── Parse and verify experiment results ──
    print("\n--- Experiment 1: Reservoir Size Ablation ---")
    for size in [64, 128, 256, 512]:
        m = re.search(rf"N_res = {size}.*?Acc = ([\d.]+)%", out1)
        if m:
            acc = float(m.group(1))
            check(f"N_res={size} accuracy > 95%", acc > 95.0, f"got {acc}%")
        else:
            check(f"N_res={size} result found in output", False)

    print("\n--- Experiment 2: FDR Three-Way Comparison ---")
    m_raw = re.search(r"FDR \(Raw Input\):\s+([\d.]+)", out1)
    m_lsm = re.search(r"FDR \(LSM Reservoir\):\s+([\d.]+)", out1)
    if m_raw and m_lsm:
        fdr_raw = float(m_raw.group(1))
        fdr_lsm = float(m_lsm.group(1))
        ratio = fdr_lsm / fdr_raw if fdr_raw > 0 else 0
        check("FDR: LSM >> Raw (ratio > 6x)", ratio > 6.0,
              f"ratio={ratio:.1f}x")
        check("FDR: LSM > 100", fdr_lsm > 100,
              f"FDR_LSM={fdr_lsm:.1f}")
    else:
        check("FDR results found in output", False)

    print("\n--- Experiment 3: Coding Scheme Comparison ---")
    # BSC6 should be > 90%
    m_bsc6 = re.search(r"bsc6\s+\+\s+logreg\s*:\s+([\d.]+)%", out1)
    if m_bsc6:
        bsc6_acc = float(m_bsc6.group(1))
        check("BSC6 + LogReg > 90%", bsc6_acc > 90.0, f"got {bsc6_acc}%")
    else:
        check("BSC6 result found in output", False)

    # MFR should be ~50% (chance)
    m_mfr = re.search(r"mfr\s+\+\s+logreg\s*:\s+([\d.]+)%", out1)
    if m_mfr:
        mfr_acc = float(m_mfr.group(1))
        check("MFR + LogReg ~ chance (40-60%)", 40.0 <= mfr_acc <= 60.0,
              f"got {mfr_acc}%")
    else:
        check("MFR result found in output", False)

    # BSC6 >> MFR
    if m_bsc6 and m_mfr:
        check("BSC6 > MFR by > 30pp",
              float(m_bsc6.group(1)) - float(m_mfr.group(1)) > 30.0)

    print("\n--- Experiment 4: PCA Dimensionality Reduction ---")
    m_pca64 = re.search(r"PCA-\s*64:.*?Acc = ([\d.]+)%", out1)
    m_full = re.search(r"Full BSC6.*?Acc = ([\d.]+)%", out1)
    if m_pca64:
        check("PCA-64 accuracy > 95%", float(m_pca64.group(1)) > 95.0,
              f"got {m_pca64.group(1)}%")
    if m_pca64 and m_full:
        drop = float(m_full.group(1)) - float(m_pca64.group(1))
        check("PCA-64 drop from full < 5pp", drop < 5.0,
              f"drop={drop:.1f}pp")

    print("\n--- Experiment 5: Cross-Initialization Robustness ---")
    m_rob_bsc6 = re.search(
        r"BSC6\+PCA64:\s+([\d.]+)%\s*±\s*([\d.]+)%", out1)
    m_rob_mfr = re.search(
        r"MFR:\s+([\d.]+)%\s*±\s*([\d.]+)%", out1)
    if m_rob_bsc6:
        mean_acc = float(m_rob_bsc6.group(1))
        std_acc = float(m_rob_bsc6.group(2))
        check("BSC6 robustness mean > 95%", mean_acc > 95.0,
              f"got {mean_acc}%")
        check("BSC6 robustness std < 3%", std_acc < 3.0,
              f"got {std_acc}%")
    if m_rob_mfr:
        mfr_mean = float(m_rob_mfr.group(1))
        check("MFR robustness mean ~ chance (45-55%)",
              45.0 <= mfr_mean <= 55.0, f"got {mfr_mean}%")

    print("\n--- Experiment 6: Parameter Sensitivity ---")
    # Check that all parameter combos > 95%
    sensitivity_accs = re.findall(
        r"β=[\d.]+, M_th=[\d.]+\.\.\.\s*([\d.]+)%", out1)
    if sensitivity_accs:
        accs = [float(a) for a in sensitivity_accs]
        check(f"All {len(accs)} param combos > 95%",
              all(a > 95.0 for a in accs),
              f"min={min(accs):.1f}%")
        check("Parameter landscape range < 5pp",
              max(accs) - min(accs) < 5.0,
              f"range={max(accs)-min(accs):.1f}pp")

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"CHAPTER 4 VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
