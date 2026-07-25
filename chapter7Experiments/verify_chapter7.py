#!/usr/bin/env python3
"""Verify Chapter 7 scripts and repository-backed outputs.

Experiments B and C execute against the committed CSV outputs from Experiment A.
Experiments A, D, and E require restricted or external inputs and therefore
receive syntax and inventory checks only.
"""

import ast
import os
import re
import subprocess
import sys

import numpy as np

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
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(base)
    os.chdir(repo_root)

    print("=" * 70)
    print("CHAPTER 7 VERIFICATION")
    print("=" * 70)

    print("\n--- Script Syntax Validation ---")
    scripts = [
        "run_chapter7_experiment_A.py",
        "run_chapter7_experiment_B.py",
        "run_chapter7_experiment_C.py",
        "run_chapter7_experiment_D.py",
        "run_chapter7_experiment_E.py",
        "extract_kappa_matrix.py",
        "extract_C_matrices.py",
    ]
    for script in scripts:
        path = os.path.join(base, script)
        if not os.path.exists(path):
            check(f"File exists: {script}", False, "file not found")
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                ast.parse(handle.read())
            check(f"Syntax valid: {script}", True)
        except SyntaxError as error:
            check(f"Syntax valid: {script}", False, str(error))

    print("\n--- Data File Inventory ---")
    data_dir = os.path.join(base, "chapter7_results")
    data_files = {
        "ch7_full_results.pkl": 20_000_000,
        "ch7_expA_analysis.pkl": 100_000,
        "kappa_matrix.csv": 5_000,
        "C_matrices.csv": 50_000,
        "subject_features.csv": 500_000,
        "observation_features.csv": 2_000_000,
    }
    for filename, minimum_size in data_files.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            check(
                f"Data: {filename} ({size:,} bytes)",
                size >= minimum_size,
                f"expected >={minimum_size:,}, got {size:,}",
            )
        else:
            check(f"Data: {filename}", False, "file not found")

    print("\n--- Kappa Matrix Validation ---")
    try:
        import pandas as pd

        kappa = pd.read_csv(os.path.join(data_dir, "kappa_matrix.csv"))
        check("Kappa matrix has 211 rows", len(kappa) == 211, f"got {len(kappa)}")
        expected_columns = {"subject", "Threat", "Mutilation", "Cute", "Erotic"}
        check(
            "Kappa matrix has correct columns",
            set(kappa.columns) == expected_columns,
            f"got {list(kappa.columns)}",
        )
        values = kappa[["Threat", "Mutilation", "Cute", "Erotic"]]
        check(
            "Kappa values in [0, 1]",
            values.min().min() >= 0 and values.max().max() <= 1,
        )
        median_kappa = values.median().median()
        check(
            "Median kappa ~ 0.27 (within 0.05)",
            abs(median_kappa - 0.27) < 0.05,
            f"got {median_kappa:.4f}",
        )
    except Exception as error:
        check("Kappa matrix validation", False, str(error))

    print("\n--- C Matrices Validation ---")
    try:
        c_matrices = pd.read_csv(os.path.join(data_dir, "C_matrices.csv"))
        check(
            "C matrices has 844 rows (211 x 4)",
            len(c_matrices) == 844,
            f"got {len(c_matrices)}",
        )
        correlation_columns = [
            column for column in c_matrices.columns if "_x_" in column
        ]
        check(
            "C matrices has 14 correlation columns",
            len(correlation_columns) == 14,
            f"got {len(correlation_columns)}",
        )
        check(
            "All correlations in [-1, 1]",
            c_matrices[correlation_columns].min().min() >= -1
            and c_matrices[correlation_columns].max().max() <= 1,
        )
    except Exception as error:
        check("C matrices validation", False, str(error))

    print("\n--- Experiment B: Variance Decomposition ---")
    result_b = subprocess.run(
        [sys.executable, os.path.join(base, "run_chapter7_experiment_B.py")],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
    )
    output_b = result_b.stdout + result_b.stderr
    if result_b.returncode != 0:
        print(output_b)
    check(
        "Experiment B exits cleanly",
        result_b.returncode == 0,
        f"exit code {result_b.returncode}",
    )

    match = re.search(r"V_subj:\s+([\d.]+)%", output_b)
    if match:
        value = float(match.group(1))
        check("V_subj ~ 29% (within 5pp)", abs(value - 29.2) < 5.0, f"got {value}%")

    match = re.search(r"V_resid:\s+([\d.]+)%", output_b)
    if match:
        value = float(match.group(1))
        check("V_resid ~ 70% (within 5pp)", abs(value - 70.1) < 5.0, f"got {value}%")

    match = re.search(r"ICC\(3,1\).*?:\s+([\d.]+)", output_b)
    if match:
        value = float(match.group(1))
        check("ICC(3,1) ~ 0.059 (within 0.02)", abs(value - 0.059) < 0.02, f"got {value}")

    match = re.search(r"Cute - Erotic:.*?p = ([\d.e+-]+)", output_b)
    if match:
        value = float(match.group(1))
        check("Cute-Erotic p < 0.05", value < 0.05, f"got p={value}")

    match = re.search(r"Threat - Mutilation:.*?p = ([\d.e+-]+)", output_b)
    if match:
        value = float(match.group(1))
        check("Threat-Mutilation p > 0.05 (null)", value > 0.05, f"got p={value}")

    figure_dir = os.path.join(repo_root, "pictures", "chSynthesis")
    for filename in [
        "fig7_B1_raw_observation.pdf",
        "fig7_B2_raw_paired_differences.pdf",
        "fig7_B3_variance_decomposition.pdf",
    ]:
        path = os.path.join(figure_dir, filename)
        check(
            f"Figure: {filename}",
            os.path.exists(path) and os.path.getsize(path) > 0,
        )

    print("\n--- Experiment C: Category-Conditioned Coupling ---")
    result_c = subprocess.run(
        [sys.executable, os.path.join(base, "run_chapter7_experiment_C.py")],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
    )
    output_c = result_c.stdout + result_c.stderr
    if result_c.returncode != 0:
        print(output_c)
    check(
        "Experiment C exits cleanly",
        result_c.returncode == 0,
        f"exit code {result_c.returncode}",
    )

    match_cute = re.search(
        r"Cute.*?Significant cells.*?(\d+)/14", output_c
    )
    if match_cute:
        check(
            "Cute-Erotic: 0/14 Bonferroni-significant",
            int(match_cute.group(1)) == 0,
            f"got {match_cute.group(1)}/14",
        )

    match_threat = re.search(
        r"Threat.*?Significant cells.*?(\d+)/14", output_c
    )
    if match_threat:
        check(
            "Threat-Mutilation: 0/14 Bonferroni-significant",
            int(match_threat.group(1)) == 0,
            f"got {match_threat.group(1)}/14",
        )

    tau_effects = re.findall(r"tau_ac\s+\w+\s+[\d.-]+\s+([\d.-]+)", output_c)
    other_effects = re.findall(
        r"(total_spikes|mean_firing_rate|rate_entropy|rate_variance)"
        r"\s+\w+\s+[\d.-]+\s+([\d.-]+)",
        output_c,
    )
    if tau_effects and other_effects:
        maximum_tau = max(abs(float(value)) for value in tau_effects[:4])
        maximum_other = max(abs(float(value)) for _, value in other_effects[:8])
        check(
            "tau_ac has larger |d_z| than amplitude metrics",
            maximum_tau > maximum_other,
            f"tau_ac={maximum_tau:.3f}, other={maximum_other:.3f}",
        )

    for filename in [
        "fig7_C1_category_C_matrices.pdf",
        "fig7_C2_raw_difference_matrices.pdf",
        "fig7_C3_difference_significance.pdf",
        "fig7_C4_CE_top_cells.pdf",
    ]:
        path = os.path.join(figure_dir, filename)
        check(
            f"Figure: {filename}",
            os.path.exists(path) and os.path.getsize(path) > 0,
        )

    print("\n--- Subject Features Validation ---")
    try:
        features = pd.read_csv(os.path.join(data_dir, "subject_features.csv"))
        check("Subject features has 211 rows", len(features) == 211, f"got {len(features)}")
        dynamic_columns = [column for column in features.columns if column.startswith("d_")]
        topology_columns = [column for column in features.columns if column.startswith("t_")]
        check(
            "238 dynamical features (34 x 7)",
            len(dynamic_columns) == 238,
            f"got {len(dynamic_columns)}",
        )
        check(
            "68 topological features (34 x 2)",
            len(topology_columns) == 68,
            f"got {len(topology_columns)}",
        )
    except Exception as error:
        check("Subject features validation", False, str(error))

    print("\n" + "=" * 70)
    print(f"CHAPTER 7 VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("Experiments B and C: executed with committed repository data.")
    print("Experiments A, D, E: syntax/inventory checked; external data required.")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
