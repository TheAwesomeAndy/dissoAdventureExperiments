#!/usr/bin/env python3
"""
Chapter 4 Verification Script
==============================
Runs both Chapter 4 scripts and checks that all claimed results hold.
No external data required. Uses synthetic data only.

Exit code 0 = all checks pass, 1 = at least one check failed.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

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

    print("\n[1/2] Running run_chapter4_experiments.py...")
    r1 = subprocess.run(
        [sys.executable, "chapter4Experiments/run_chapter4_experiments.py"],
        capture_output=True,
        text=True,
        timeout=600,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
    )
    out1 = r1.stdout + r1.stderr
    if r1.returncode != 0:
        print("\n--- run_chapter4_experiments.py output ---")
        print(out1)
        print("--- end subprocess output ---\n")
    check(
        "Experiments script exits cleanly",
        r1.returncode == 0,
        f"exit code {r1.returncode}",
    )

    print("\n[2/2] Running run_chapter4_observations.py...")
    r2 = subprocess.run(
        [sys.executable, "chapter4Experiments/run_chapter4_observations.py"],
        capture_output=True,
        text=True,
        timeout=600,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
    )
    out2 = r2.stdout + r2.stderr
    if r2.returncode != 0:
        print("\n--- run_chapter4_observations.py output ---")
        print(out2)
        print("--- end subprocess output ---\n")
    check(
        "Observations script exits cleanly",
        r2.returncode == 0,
        f"exit code {r2.returncode}",
    )

    print("\n--- Figure Generation ---")
    figure_dir = Path("pictures/chLSMEmbeddings")
    expected_experiment_figures = [
        "ablation_reservoir_size.pdf",
        "fdr_three_way_comparison.pdf",
        "coding_scheme_accuracy_comparison.pdf",
        "pca_explained_variance.pdf",
        "pca_component_visualization.pdf",
        "cross_initialization_robustness.pdf",
        "parameter_sensitivity_heatmap.pdf",
    ]
    expected_observation_figures = [
        "obs01_raw_input_signals.pdf",
        "obs02_raw_spike_rasters.pdf",
        "obs03_raw_bsc6_features.pdf",
        "obs04_raw_embedding_space.pdf",
        "obs05_population_dynamics.pdf",
        "obs06_membrane_dynamics.pdf",
    ]
    for filename in expected_experiment_figures + expected_observation_figures:
        path = figure_dir / filename
        check(
            f"Figure exists: {filename}",
            path.exists() and path.stat().st_size > 0,
        )

    print("\n--- Experiment 1: Reservoir Size Ablation ---")
    for size in [64, 128, 256, 512]:
        match = re.search(rf"N_res = {size}.*?Acc = ([\d.]+)%", out1)
        if match:
            accuracy = float(match.group(1))
            check(
                f"N_res={size} accuracy > 95%",
                accuracy > 95.0,
                f"got {accuracy}%",
            )
        else:
            check(f"N_res={size} result found in output", False)

    print("\n--- Experiment 2: FDR Three-Way Comparison ---")
    raw_match = re.search(r"FDR \(Raw Input\):\s+([\d.]+)", out1)
    lsm_match = re.search(r"FDR \(LSM Reservoir\):\s+([\d.]+)", out1)
    if raw_match and lsm_match:
        raw_fdr = float(raw_match.group(1))
        lsm_fdr = float(lsm_match.group(1))
        ratio = lsm_fdr / raw_fdr if raw_fdr > 0 else 0
        check("FDR: LSM >> Raw (ratio > 6x)", ratio > 6.0, f"ratio={ratio:.1f}x")
        check("FDR: LSM > 100", lsm_fdr > 100, f"FDR_LSM={lsm_fdr:.1f}")
    else:
        check("FDR results found in output", False)

    print("\n--- Experiment 3: Coding Scheme Comparison ---")
    bsc6_match = re.search(r"bsc6\s+\+\s+logreg\s*:\s+([\d.]+)%", out1)
    if bsc6_match:
        bsc6_accuracy = float(bsc6_match.group(1))
        check(
            "BSC6 + LogReg > 90%",
            bsc6_accuracy > 90.0,
            f"got {bsc6_accuracy}%",
        )
    else:
        check("BSC6 result found in output", False)

    mfr_match = re.search(r"mfr\s+\+\s+logreg\s*:\s+([\d.]+)%", out1)
    if mfr_match:
        mfr_accuracy = float(mfr_match.group(1))
        check(
            "MFR + LogReg ~ chance (40-60%)",
            40.0 <= mfr_accuracy <= 60.0,
            f"got {mfr_accuracy}%",
        )
    else:
        check("MFR result found in output", False)

    if bsc6_match and mfr_match:
        check(
            "BSC6 > MFR by > 30pp",
            float(bsc6_match.group(1)) - float(mfr_match.group(1)) > 30.0,
        )

    print("\n--- Experiment 4: PCA Dimensionality Reduction ---")
    pca64_match = re.search(r"PCA-\s*64:.*?Acc = ([\d.]+)%", out1)
    full_match = re.search(r"Full BSC6.*?Acc = ([\d.]+)%", out1)
    if pca64_match:
        check(
            "PCA-64 accuracy > 95%",
            float(pca64_match.group(1)) > 95.0,
            f"got {pca64_match.group(1)}%",
        )
    if pca64_match and full_match:
        drop = float(full_match.group(1)) - float(pca64_match.group(1))
        check("PCA-64 drop from full < 5pp", drop < 5.0, f"drop={drop:.1f}pp")

    print("\n--- Experiment 5: Cross-Initialization Robustness ---")
    robustness_bsc6 = re.search(
        r"BSC6\+PCA64:\s+([\d.]+)%\s*±\s*([\d.]+)%", out1
    )
    robustness_mfr = re.search(r"MFR:\s+([\d.]+)%\s*±\s*([\d.]+)%", out1)
    if robustness_bsc6:
        mean_accuracy = float(robustness_bsc6.group(1))
        std_accuracy = float(robustness_bsc6.group(2))
        check("BSC6 robustness mean > 95%", mean_accuracy > 95.0, f"got {mean_accuracy}%")
        check("BSC6 robustness std < 3%", std_accuracy < 3.0, f"got {std_accuracy}%")
    if robustness_mfr:
        mean_mfr = float(robustness_mfr.group(1))
        check(
            "MFR robustness mean ~ chance (45-55%)",
            45.0 <= mean_mfr <= 55.0,
            f"got {mean_mfr}%",
        )

    print("\n--- Experiment 6: Parameter Sensitivity ---")
    sensitivity_accuracies = re.findall(
        r"β=[\d.]+, M_th=[\d.]+\.\.\.\s*([\d.]+)%", out1
    )
    if sensitivity_accuracies:
        values = [float(value) for value in sensitivity_accuracies]
        check(
            f"All {len(values)} param combos > 95%",
            all(value > 95.0 for value in values),
            f"min={min(values):.1f}%",
        )
        check(
            "Parameter landscape range < 5pp",
            max(values) - min(values) < 5.0,
            f"range={max(values) - min(values):.1f}pp",
        )
    else:
        check("Parameter-sensitivity results found in output", False)

    print("\n" + "=" * 70)
    print(f"CHAPTER 4 VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
