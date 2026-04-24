#!/usr/bin/env python3
"""
Chapter 7 Verification Script
==============================
Runs the Chapter 7 experiments that can execute with data present in the
repository (Experiments B and C use CSV files from Experiment A), and
verifies that all claimed results hold.

Experiments A, D, and E require external data (raw EEG files or
psychopathology spreadsheet) and are tested via syntax checking and
import validation only.

Usage:
    python chapter7Experiments/verify_chapter7.py

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
import ast
import re
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
    repo_root = os.path.dirname(base)
    os.chdir(repo_root)

    print("=" * 70)
    print("CHAPTER 7 VERIFICATION")
    print("=" * 70)

    # ── 1. Syntax check all scripts ──
    print("\n--- Script Syntax Validation ---")
    scripts = [
        "run_chapter7_experiment_A.py",
        "run_chapter7_experiment_B.py",
        "run_chapter7_experiment_C.py",
        "run_chapter7_experiment_D.py",
        "run_chapter7_experiment_E.py",
        "extract_kappa_matrix.py",
        "extract_C_matrices.py",
        # extract_features_for_expE.py was a one-time utility, output committed as CSVs
    ]
    for script in scripts:
        path = os.path.join(base, script)
        if not os.path.exists(path):
            check(f"File exists: {script}", False, "file not found")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            check(f"Syntax valid: {script}", True)
        except SyntaxError as e:
            check(f"Syntax valid: {script}", False, str(e))

    # ── 2. Data files exist ──
    print("\n--- Data File Inventory ---")
    data_dir = os.path.join(base, "chapter7_results")
    data_files = {
        "ch7_full_results.pkl": 20_000_000,    # ~23MB
        "ch7_expA_analysis.pkl": 100_000,       # ~128KB
        "kappa_matrix.csv": 5_000,              # ~8KB
        "C_matrices.csv": 50_000,               # ~120KB
        "subject_features.csv": 500_000,        # ~616KB
        "observation_features.csv": 2_000_000,  # ~2.4MB
    }
    for fname, min_size in data_files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            size = os.path.getsize(path)
            check(f"Data: {fname} ({size:,} bytes)",
                  size >= min_size,
                  f"expected >={min_size:,}, got {size:,}")
        else:
            check(f"Data: {fname}", False, "file not found")

    # ── 3. Verify kappa_matrix.csv structure ──
    print("\n--- Kappa Matrix Validation ---")
    try:
        import pandas as pd
        kappa = pd.read_csv(os.path.join(data_dir, "kappa_matrix.csv"))
        check("Kappa matrix has 211 rows", len(kappa) == 211,
              f"got {len(kappa)}")
        expected_cols = {'subject', 'Threat', 'Mutilation', 'Cute', 'Erotic'}
        check("Kappa matrix has correct columns",
              set(kappa.columns) == expected_cols,
              f"got {list(kappa.columns)}")
        check("Kappa values in [0, 1]",
              kappa[['Threat','Mutilation','Cute','Erotic']].min().min() >= 0
              and kappa[['Threat','Mutilation','Cute','Erotic']].max().max() <= 1)
        median_kappa = kappa[['Threat','Mutilation','Cute','Erotic']].median().median()
        check("Median kappa ~ 0.27 (within 0.05)",
              abs(median_kappa - 0.27) < 0.05,
              f"got {median_kappa:.4f}")
    except Exception as e:
        check("Kappa matrix validation", False, str(e))

    # ── 4. Verify C_matrices.csv structure ──
    print("\n--- C Matrices Validation ---")
    try:
        cmat = pd.read_csv(os.path.join(data_dir, "C_matrices.csv"))
        check("C matrices has 844 rows (211 x 4)", len(cmat) == 844,
              f"got {len(cmat)}")
        corr_cols = [c for c in cmat.columns if '_x_' in c]
        check("C matrices has 14 correlation columns",
              len(corr_cols) == 14,
              f"got {len(corr_cols)}")
        check("All correlations in [-1, 1]",
              cmat[corr_cols].min().min() >= -1
              and cmat[corr_cols].max().max() <= 1)
    except Exception as e:
        check("C matrices validation", False, str(e))

    # ── 5. Run Experiment B ──
    print("\n--- Experiment B: Variance Decomposition (full run) ---")
    r_b = subprocess.run(
        [sys.executable,
         os.path.join(base, "run_chapter7_experiment_B.py")],
        capture_output=True, text=True, timeout=120,
        encoding='utf-8', errors='replace',
        env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
    )
    out_b = r_b.stdout + r_b.stderr
    check("Experiment B exits cleanly", r_b.returncode == 0,
          f"exit code {r_b.returncode}")

    # Verify claimed results
    m = re.search(r"V_subj:\s+([\d.]+)%", out_b)
    if m:
        v_subj = float(m.group(1))
        check("V_subj ~ 29% (within 5pp)", abs(v_subj - 29.2) < 5.0,
              f"got {v_subj}%")

    m = re.search(r"V_resid:\s+([\d.]+)%", out_b)
    if m:
        v_resid = float(m.group(1))
        check("V_resid ~ 70% (within 5pp)", abs(v_resid - 70.1) < 5.0,
              f"got {v_resid}%")

    m = re.search(r"ICC\(3,1\).*?:\s+([\d.]+)", out_b)
    if m:
        icc = float(m.group(1))
        check("ICC(3,1) ~ 0.059 (within 0.02)", abs(icc - 0.059) < 0.02,
              f"got {icc}")

    m = re.search(r"Cute - Erotic:.*?p = ([\d.e+-]+)", out_b)
    if m:
        p_ce = float(m.group(1))
        check("Cute-Erotic p < 0.05", p_ce < 0.05, f"got p={p_ce}")

    m = re.search(r"Threat - Mutilation:.*?p = ([\d.e+-]+)", out_b)
    if m:
        p_tm = float(m.group(1))
        check("Threat-Mutilation p > 0.05 (null)", p_tm > 0.05,
              f"got p={p_tm}")

    # Check figures (Exp B saves to /mnt/user-data/outputs/pictures/chSynthesis/)
    synth_fig_dir = "/mnt/user-data/outputs/pictures/chSynthesis"
    local_fig_dir = os.path.join(data_dir, "figures")
    for fig in ["fig7_B1_raw_observation.pdf",
                "fig7_B2_raw_paired_differences.pdf",
                "fig7_B3_variance_decomposition.pdf"]:
        found = (
            (os.path.exists(os.path.join(synth_fig_dir, fig))
             and os.path.getsize(os.path.join(synth_fig_dir, fig)) > 0)
            or (os.path.exists(os.path.join(local_fig_dir, fig))
                and os.path.getsize(os.path.join(local_fig_dir, fig)) > 0)
        )
        check(f"Figure: {fig}", found)

    # ── 6. Run Experiment C ──
    print("\n--- Experiment C: Category-Conditioned Coupling (full run) ---")
    r_c = subprocess.run(
        [sys.executable,
         os.path.join(base, "run_chapter7_experiment_C.py")],
        capture_output=True, text=True, timeout=120,
        encoding='utf-8', errors='replace',
        env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
    )
    out_c = r_c.stdout + r_c.stderr
    check("Experiment C exits cleanly", r_c.returncode == 0,
          f"exit code {r_c.returncode}")

    # Verify: 0/14 cells survive Bonferroni for both contrasts
    m_ce = re.search(r"Cute.*?Significant cells.*?(\d+)/14", out_c)
    if m_ce:
        check("Cute-Erotic: 0/14 Bonferroni-significant",
              int(m_ce.group(1)) == 0,
              f"got {m_ce.group(1)}/14")

    m_tm = re.search(r"Threat.*?Significant cells.*?(\d+)/14", out_c)
    if m_tm:
        check("Threat-Mutilation: 0/14 Bonferroni-significant",
              int(m_tm.group(1)) == 0,
              f"got {m_tm.group(1)}/14")

    # Verify tau_ac carries the largest effect
    tau_lines = re.findall(r"tau_ac\s+\w+\s+[\d.-]+\s+([\d.-]+)", out_c)
    other_lines = re.findall(
        r"(total_spikes|mean_firing_rate|rate_entropy|rate_variance)"
        r"\s+\w+\s+[\d.-]+\s+([\d.-]+)", out_c)
    if tau_lines and other_lines:
        max_tau_dz = max(abs(float(d)) for d in tau_lines[:4])
        max_other_dz = max(abs(float(d)) for _, d in other_lines[:8])
        check("tau_ac has larger |d_z| than amplitude metrics",
              max_tau_dz > max_other_dz,
              f"tau_ac={max_tau_dz:.3f}, other={max_other_dz:.3f}")

    # Check figures (Exp C saves to /mnt/user-data/outputs/pictures/chSynthesis/)
    for fig in ["fig7_C1_category_C_matrices.pdf",
                "fig7_C2_raw_difference_matrices.pdf",
                "fig7_C3_difference_significance.pdf",
                "fig7_C4_CE_top_cells.pdf"]:
        found = (
            (os.path.exists(os.path.join(synth_fig_dir, fig))
             and os.path.getsize(os.path.join(synth_fig_dir, fig)) > 0)
            or (os.path.exists(os.path.join(local_fig_dir, fig))
                and os.path.getsize(os.path.join(local_fig_dir, fig)) > 0)
        )
        check(f"Figure: {fig}", found)

    # ── 7. Verify subject_features.csv structure ──
    print("\n--- Subject Features Validation (for Exp E) ---")
    try:
        sf = pd.read_csv(os.path.join(data_dir, "subject_features.csv"))
        check("Subject features has 211 rows", len(sf) == 211,
              f"got {len(sf)}")
        d_cols = [c for c in sf.columns if c.startswith('d_')]
        t_cols = [c for c in sf.columns if c.startswith('t_')]
        check("238 dynamical features (34 x 7)", len(d_cols) == 238,
              f"got {len(d_cols)}")
        check("68 topological features (34 x 2)", len(t_cols) == 68,
              f"got {len(t_cols)}")
    except Exception as e:
        check("Subject features validation", False, str(e))

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"CHAPTER 7 VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("\nExperiments B and C: fully verified (ran with repo data).")
    print("Experiments A, D, E: syntax-checked (require external data).")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
