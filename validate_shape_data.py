#!/usr/bin/env python3
"""
============================================================================
ARSPI-Net Data Validation and Quality Control
============================================================================
Implements Brady's recommended data checks plus comprehensive EEG QC for
the SHAPE Community emotional interrupt task dataset.

Brady's specification (from original email):
  - Each participant: 3 .txt files (IAPSNeg, IAPSNeu, IAPSPos)
  - Sampling rate: 1024 Hz
  - Segment: 1200 ms (200 ms baseline + 1000 ms post-stimulus)
  - Expected dimensions: 1229 rows × 34 columns
  - First 205 rows = baseline (200 ms × 1.024 Hz ≈ 205 samples)
  - Headers removed; raw microvolt values
  - "I would again recommend doing an automated datacheck to make sure
     every file is the same 1229 × 34 dimensions."

This script performs:
  CHECK 1: File inventory and naming convention validation
  CHECK 2: Dimensional consistency (Brady's explicit check)
  CHECK 3: Subject completeness (all 3 conditions present)
  CHECK 4: Numerical integrity (NaN, Inf, non-numeric values)
  CHECK 5: Amplitude range sanity (microvolt plausibility)
  CHECK 6: Flat/dead channel detection (zero or near-zero variance)
  CHECK 7: Extreme outlier detection (> 5 SD from channel mean)
  CHECK 8: Baseline period verification (first 205 rows)
  CHECK 9: Cross-batch duplicate detection
  CHECK 10: Clinical database cross-reference

Usage:
  python validate_shape_data.py \
    --batch1 /path/to/batch1.zip \
    --batch2 /path/to/batch2.zip \
    --batch3 /path/to/batch3.zip \
    --participant_info /path/to/ParticipantInfo.csv \
    --psychopathology /path/to/Psychopathology.xlsx \
    --output_dir /path/to/qc_output/

Output:
  - Console report with PASS/WARN/FAIL for each check
  - qc_report.txt: Full text report
  - qc_summary.csv: Per-file QC metrics
  - qc_flagged_subjects.csv: Subjects requiring manual review
  - qc_amplitude_distributions.pdf: Channel amplitude distributions
  - qc_subject_heatmap.pdf: Per-subject data quality heatmap

Author: Andrew Lane
Date: March 2026
============================================================================
"""

import numpy as np
import pandas as pd
import zipfile
import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — Brady's Specification
# ══════════════════════════════════════════════════════════════════
EXPECTED_ROWS = 1229
EXPECTED_COLS = 34
SAMPLING_RATE = 1024  # Hz
BASELINE_SAMPLES = 205  # ~200 ms at 1024 Hz
POSTSTIM_SAMPLES = EXPECTED_ROWS - BASELINE_SAMPLES  # 1024 samples = 1000 ms
CONDITIONS = ['IAPSNeg', 'IAPSNeu', 'IAPSPos']
CONDITION_LABELS = {
    'IAPSNeg': 'Negative (Unpleasant)',
    'IAPSNeu': 'Neutral',
    'IAPSPos': 'Pleasant'
}

# QC Thresholds
AMPLITUDE_WARN_UV = 200    # µV — warn if any sample exceeds this
AMPLITUDE_FAIL_UV = 500    # µV — flag as likely artifact
FLAT_CHANNEL_STD = 0.01    # µV — channel is "dead" if std below this
OUTLIER_SD_THRESHOLD = 5.0 # flag samples > 5 SD from channel mean
BASELINE_MEAN_WARN = 5.0   # µV — baseline should be near zero (BC'd)


class QCReport:
    """Accumulates QC results and generates the final report."""
    
    def __init__(self):
        self.checks = []
        self.file_metrics = []
        self.flagged_subjects = []
        self.warnings = []
        self.errors = []
        self.pass_count = 0
        self.warn_count = 0
        self.fail_count = 0
    
    def add_check(self, name, status, detail=""):
        """Record a check result. Status: PASS, WARN, FAIL."""
        self.checks.append({'name': name, 'status': status, 'detail': detail})
        icon = {'PASS': '✓', 'WARN': '⚠', 'FAIL': '✗'}[status]
        color_code = {'PASS': '', 'WARN': '', 'FAIL': ''}[status]
        print(f"  [{icon}] {status}: {name}")
        if detail:
            for line in detail.split('\n'):
                if line.strip():
                    print(f"      {line.strip()}")
        if status == 'PASS': self.pass_count += 1
        elif status == 'WARN': self.warn_count += 1
        else: self.fail_count += 1
    
    def summary(self):
        total = self.pass_count + self.warn_count + self.fail_count
        print(f"\n{'='*70}")
        print(f"QC SUMMARY: {self.pass_count} PASS, {self.warn_count} WARN, "
              f"{self.fail_count} FAIL out of {total} checks")
        print(f"{'='*70}")
        if self.fail_count == 0 and self.warn_count == 0:
            print("ALL CHECKS PASSED. Data is ready for processing.")
        elif self.fail_count == 0:
            print("No critical failures. Review warnings before processing.")
        else:
            print("CRITICAL FAILURES DETECTED. Address before processing.")
    
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Text report
        with open(os.path.join(output_dir, 'qc_report.txt'), 'w') as f:
            f.write(f"ARSPI-Net Data Quality Control Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            for check in self.checks:
                f.write(f"[{check['status']}] {check['name']}\n")
                if check['detail']:
                    for line in check['detail'].split('\n'):
                        f.write(f"    {line}\n")
                f.write("\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"SUMMARY: {self.pass_count} PASS, {self.warn_count} WARN, "
                    f"{self.fail_count} FAIL\n")
        
        # File metrics CSV
        if self.file_metrics:
            df = pd.DataFrame(self.file_metrics)
            df.to_csv(os.path.join(output_dir, 'qc_file_metrics.csv'), index=False)
        
        # Flagged subjects
        if self.flagged_subjects:
            df = pd.DataFrame(self.flagged_subjects)
            df.to_csv(os.path.join(output_dir, 'qc_flagged_subjects.csv'), index=False)
        
        print(f"\nQC reports saved to {output_dir}/")


def parse_filename(filename):
    """Extract subject ID and condition from SHAPE filename.
    
    Expected: SHAPE_Community_{SUBJ}_{CONDITION}_BC.txt
    Returns: (subject_id, condition) or (None, None) on failure.
    """
    basename = os.path.basename(filename)
    if not basename.endswith('.txt'):
        return None, None
    
    basename = basename.replace('.txt', '').replace('_BC', '')
    
    # Try standard format
    if 'SHAPE_Community_' in basename:
        rest = basename.replace('SHAPE_Community_', '')
        # Split and find subject ID (numeric) and condition
        for cond in CONDITIONS:
            if cond in rest:
                subj = rest.replace(f'_{cond}', '').strip('_')
                if subj.isdigit():
                    return subj, cond
    
    return None, None


def load_file_from_zip(zf, filepath):
    """Load a single data file from a zip archive.
    
    Returns: numpy array or None on failure, plus error string.
    """
    try:
        with zf.open(filepath) as f:
            content = f.read().decode('utf-8')
        
        # Check for empty file
        if not content.strip():
            return None, "Empty file"
        
        # Try to parse as numeric array
        lines = content.strip().split('\n')
        data = []
        for li, line in enumerate(lines):
            vals = line.split()
            try:
                row = [float(v) for v in vals]
                data.append(row)
            except ValueError as e:
                return None, f"Non-numeric value at line {li+1}: {e}"
        
        return np.array(data), None
    
    except Exception as e:
        return None, str(e)


def run_qc(batch_paths, participant_info_path=None, 
           psychopathology_path=None, output_dir='qc_output'):
    """Run the complete QC pipeline."""
    
    report = QCReport()
    
    print("="*70)
    print("ARSPI-Net Data Quality Control")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 1: File Inventory and Naming Convention
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("CHECK 1: File Inventory and Naming Convention")
    print(f"{'─'*70}")
    
    all_files = {}  # {filepath: (batch_name, subject_id, condition)}
    bad_names = []
    
    for batch_path in batch_paths:
        batch_name = os.path.basename(batch_path).replace('.zip', '')
        zf = zipfile.ZipFile(batch_path)
        
        data_files = [f for f in zf.namelist() 
                      if f.endswith('.txt') and '__MACOSX' not in f and not f.endswith('/')]
        
        print(f"  {batch_name}: {len(data_files)} files")
        
        for filepath in data_files:
            subj, cond = parse_filename(filepath)
            if subj is None or cond is None:
                bad_names.append(filepath)
            else:
                all_files[filepath] = (batch_name, subj, cond)
    
    total_files = len(all_files)
    if bad_names:
        report.add_check("Naming convention",  "WARN",
            f"{len(bad_names)} files with unparseable names:\n" + 
            "\n".join(f"  {f}" for f in bad_names[:10]))
    else:
        report.add_check("Naming convention", "PASS",
            f"All {total_files} files match SHAPE_Community_{{SUBJ}}_{{COND}}_BC.txt")
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 2: Dimensional Consistency (Brady's Explicit Check)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"CHECK 2: Dimensional Consistency (Brady's check: {EXPECTED_ROWS} × {EXPECTED_COLS})")
    print(f"{'─'*70}")
    
    dim_pass = 0
    dim_fail = []
    file_data_cache = {}  # cache loaded data for subsequent checks
    
    for filepath, (batch_name, subj, cond) in all_files.items():
        zf = zipfile.ZipFile([p for p in batch_paths if batch_name in p][0])
        data, error = load_file_from_zip(zf, filepath)
        
        if error:
            dim_fail.append((filepath, f"Load error: {error}"))
            continue
        
        file_data_cache[filepath] = data
        
        if data.shape == (EXPECTED_ROWS, EXPECTED_COLS):
            dim_pass += 1
        else:
            dim_fail.append((filepath, f"Shape {data.shape}, expected ({EXPECTED_ROWS}, {EXPECTED_COLS})"))
    
    if dim_fail:
        detail = f"{dim_pass}/{total_files} correct\nFailed files:\n"
        for fp, reason in dim_fail[:20]:
            detail += f"  {os.path.basename(fp)}: {reason}\n"
        report.add_check("Dimensional consistency", "FAIL", detail)
    else:
        report.add_check("Dimensional consistency", "PASS",
            f"All {dim_pass} files are {EXPECTED_ROWS} × {EXPECTED_COLS}")
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 3: Subject Completeness
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("CHECK 3: Subject Completeness (3 conditions per subject)")
    print(f"{'─'*70}")
    
    subjects = defaultdict(lambda: {'conditions': set(), 'batch': set(), 'files': []})
    for filepath, (batch_name, subj, cond) in all_files.items():
        subjects[subj]['conditions'].add(cond)
        subjects[subj]['batch'].add(batch_name)
        subjects[subj]['files'].append(filepath)
    
    complete = [s for s, v in subjects.items() if len(v['conditions']) == 3]
    incomplete = [(s, v) for s, v in subjects.items() if len(v['conditions']) < 3]
    
    if incomplete:
        detail = f"{len(complete)} complete, {len(incomplete)} incomplete:\n"
        for subj, v in sorted(incomplete, key=lambda x: x[0]):
            missing = set(CONDITIONS) - v['conditions']
            detail += f"  Subject {subj}: has {sorted(v['conditions'])}, missing {sorted(missing)}\n"
            report.flagged_subjects.append({
                'subject': subj, 'reason': 'incomplete_conditions',
                'detail': f"Missing: {sorted(missing)}"
            })
        report.add_check("Subject completeness", "WARN", detail)
    else:
        report.add_check("Subject completeness", "PASS",
            f"All {len(complete)} subjects have Neg/Neu/Pos")
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 4: Numerical Integrity
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("CHECK 4: Numerical Integrity (NaN, Inf, non-numeric)")
    print(f"{'─'*70}")
    
    nan_files = []
    inf_files = []
    
    for filepath, data in file_data_cache.items():
        n_nan = np.isnan(data).sum()
        n_inf = np.isinf(data).sum()
        if n_nan > 0:
            nan_files.append((filepath, n_nan))
        if n_inf > 0:
            inf_files.append((filepath, n_inf))
    
    if nan_files or inf_files:
        detail = ""
        if nan_files:
            detail += f"{len(nan_files)} files with NaN values:\n"
            for fp, n in nan_files[:10]:
                detail += f"  {os.path.basename(fp)}: {n} NaN values\n"
        if inf_files:
            detail += f"{len(inf_files)} files with Inf values:\n"
            for fp, n in inf_files[:10]:
                detail += f"  {os.path.basename(fp)}: {n} Inf values\n"
        report.add_check("Numerical integrity", "FAIL", detail)
    else:
        report.add_check("Numerical integrity", "PASS",
            f"All {len(file_data_cache)} files: zero NaN, zero Inf")
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 5: Amplitude Range Sanity
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"CHECK 5: Amplitude Range (warn > {AMPLITUDE_WARN_UV} µV, fail > {AMPLITUDE_FAIL_UV} µV)")
    print(f"{'─'*70}")
    
    amp_stats = {'global_min': np.inf, 'global_max': -np.inf, 
                 'warn_files': [], 'fail_files': []}
    
    for filepath, data in file_data_cache.items():
        fmin, fmax = data.min(), data.max()
        amp_stats['global_min'] = min(amp_stats['global_min'], fmin)
        amp_stats['global_max'] = max(amp_stats['global_max'], fmax)
        
        max_abs = max(abs(fmin), abs(fmax))
        _, subj, cond = all_files[filepath]
        
        metrics = {
            'file': os.path.basename(filepath),
            'subject': subj, 'condition': cond,
            'min_uV': round(fmin, 2), 'max_uV': round(fmax, 2),
            'mean_uV': round(data.mean(), 4), 'std_uV': round(data.std(), 2),
            'max_abs_uV': round(max_abs, 2)
        }
        report.file_metrics.append(metrics)
        
        if max_abs > AMPLITUDE_FAIL_UV:
            amp_stats['fail_files'].append((filepath, max_abs))
        elif max_abs > AMPLITUDE_WARN_UV:
            amp_stats['warn_files'].append((filepath, max_abs))
    
    detail = (f"Global range: [{amp_stats['global_min']:.1f}, {amp_stats['global_max']:.1f}] µV\n"
              f"Files > {AMPLITUDE_WARN_UV} µV: {len(amp_stats['warn_files'])}\n"
              f"Files > {AMPLITUDE_FAIL_UV} µV: {len(amp_stats['fail_files'])}")
    
    if amp_stats['fail_files']:
        detail += "\nExtreme amplitude files:\n"
        for fp, amp in sorted(amp_stats['fail_files'], key=lambda x: -x[1])[:10]:
            detail += f"  {os.path.basename(fp)}: max |amplitude| = {amp:.1f} µV\n"
            _, subj, _ = all_files[fp]
            report.flagged_subjects.append({
                'subject': subj, 'reason': 'extreme_amplitude',
                'detail': f'max |amp| = {amp:.1f} µV in {os.path.basename(fp)}'
            })
        report.add_check("Amplitude range", "WARN", detail)
    elif amp_stats['warn_files']:
        report.add_check("Amplitude range", "WARN", detail)
    else:
        report.add_check("Amplitude range", "PASS", detail)
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 6: Flat/Dead Channel Detection
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"CHECK 6: Flat/Dead Channels (std < {FLAT_CHANNEL_STD} µV)")
    print(f"{'─'*70}")
    
    flat_channels = []
    
    for filepath, data in file_data_cache.items():
        ch_stds = data.std(axis=0)  # std per channel
        flat = np.where(ch_stds < FLAT_CHANNEL_STD)[0]
        if len(flat) > 0:
            _, subj, cond = all_files[filepath]
            flat_channels.append((subj, cond, flat.tolist(), ch_stds[flat].tolist()))
    
    if flat_channels:
        detail = f"{len(flat_channels)} files with flat channels:\n"
        for subj, cond, chs, stds in flat_channels[:15]:
            detail += f"  Subject {subj} {cond}: channels {chs} (std = {[f'{s:.4f}' for s in stds]})\n"
            report.flagged_subjects.append({
                'subject': subj, 'reason': 'flat_channel',
                'detail': f'{cond}: channels {chs}'
            })
        report.add_check("Flat channel detection", "WARN", detail)
    else:
        report.add_check("Flat channel detection", "PASS",
            "No channels with std < {:.2f} µV in any file".format(FLAT_CHANNEL_STD))
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 7: Extreme Outlier Detection
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"CHECK 7: Extreme Outliers (samples > {OUTLIER_SD_THRESHOLD} SD from channel mean)")
    print(f"{'─'*70}")
    
    outlier_files = []
    total_outliers = 0
    
    for filepath, data in file_data_cache.items():
        ch_means = data.mean(axis=0)
        ch_stds = data.std(axis=0)
        ch_stds[ch_stds < 1e-10] = 1e-10  # avoid division by zero
        
        z_scores = np.abs((data - ch_means) / ch_stds)
        n_outliers = (z_scores > OUTLIER_SD_THRESHOLD).sum()
        total_outliers += n_outliers
        
        if n_outliers > 0:
            pct = n_outliers / data.size * 100
            _, subj, cond = all_files[filepath]
            outlier_files.append((subj, cond, n_outliers, pct))
            
            # Add per-channel outlier counts to file metrics
            for m in report.file_metrics:
                if m['subject'] == subj and m['condition'] == cond:
                    m['n_outliers'] = n_outliers
                    m['pct_outliers'] = round(pct, 4)
    
    n_total_samples = sum(d.size for d in file_data_cache.values())
    outlier_pct = total_outliers / n_total_samples * 100 if n_total_samples > 0 else 0
    
    detail = (f"Total outlier samples: {total_outliers:,} / {n_total_samples:,} "
              f"({outlier_pct:.3f}%)\n"
              f"Files with outliers: {len(outlier_files)} / {len(file_data_cache)}")
    
    if outlier_pct > 1.0:
        report.add_check("Outlier detection", "WARN", detail)
    else:
        report.add_check("Outlier detection", "PASS", detail)
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 8: Baseline Period Verification
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"CHECK 8: Baseline Period (first {BASELINE_SAMPLES} rows, "
          f"mean should be ≈ 0 if BC'd)")
    print(f"{'─'*70}")
    
    baseline_issues = []
    baseline_means = []
    
    for filepath, data in file_data_cache.items():
        baseline = data[:BASELINE_SAMPLES, :]
        bl_mean = np.abs(baseline.mean(axis=0))  # per-channel baseline mean
        max_bl_mean = bl_mean.max()
        avg_bl_mean = bl_mean.mean()
        baseline_means.append(avg_bl_mean)
        
        if max_bl_mean > BASELINE_MEAN_WARN:
            _, subj, cond = all_files[filepath]
            baseline_issues.append((subj, cond, max_bl_mean, avg_bl_mean))
    
    avg_all = np.mean(baseline_means) if baseline_means else 0
    detail = (f"Average baseline mean (absolute): {avg_all:.3f} µV\n"
              f"Files with baseline |mean| > {BASELINE_MEAN_WARN} µV: "
              f"{len(baseline_issues)}")
    
    if baseline_issues:
        detail += "\nFiles with large baseline offset:\n"
        for subj, cond, mx, avg in sorted(baseline_issues, key=lambda x: -x[2])[:10]:
            detail += f"  Subject {subj} {cond}: max channel |mean| = {mx:.2f} µV\n"
        report.add_check("Baseline verification", "WARN", detail)
    else:
        report.add_check("Baseline verification", "PASS", detail)
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 9: Cross-Batch Duplicate Detection
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("CHECK 9: Cross-Batch Duplicate Detection")
    print(f"{'─'*70}")
    
    multi_batch = [(s, v) for s, v in subjects.items() if len(v['batch']) > 1]
    
    if multi_batch:
        # Check if duplicates have identical data
        identical_dupes = []
        different_dupes = []
        for subj, v in multi_batch:
            # Compare files across batches
            files_by_cond = defaultdict(list)
            for fp in v['files']:
                _, _, cond = all_files[fp]
                if fp in file_data_cache:
                    files_by_cond[cond].append((fp, file_data_cache[fp]))
            
            for cond, file_list in files_by_cond.items():
                if len(file_list) > 1:
                    d1, d2 = file_list[0][1], file_list[1][1]
                    if np.allclose(d1, d2, atol=1e-10):
                        identical_dupes.append((subj, cond))
                    else:
                        different_dupes.append((subj, cond, 
                            np.abs(d1 - d2).max()))
        
        detail = f"{len(multi_batch)} subjects appear in multiple batches\n"
        if identical_dupes:
            detail += f"  Identical duplicates: {len(identical_dupes)} (safe to deduplicate)\n"
        if different_dupes:
            detail += f"  DIFFERENT data across batches: {len(different_dupes)}\n"
            for subj, cond, maxdiff in different_dupes[:5]:
                detail += f"    Subject {subj} {cond}: max difference = {maxdiff:.6f}\n"
        
        status = "FAIL" if different_dupes else "WARN"
        report.add_check("Cross-batch duplicates", status, detail)
    else:
        report.add_check("Cross-batch duplicates", "PASS",
            "No subjects appear in multiple batches")
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 10: Clinical Database Cross-Reference
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("CHECK 10: Clinical Database Cross-Reference")
    print(f"{'─'*70}")
    
    eeg_subject_ids = set(int(s) for s in subjects.keys())
    
    if participant_info_path and os.path.exists(participant_info_path):
        df_part = pd.read_csv(participant_info_path)
        part_ids = set(df_part['ID'].values)
        eeg_no_part = eeg_subject_ids - part_ids
        part_no_eeg = part_ids - eeg_subject_ids
        
        detail = (f"EEG subjects: {len(eeg_subject_ids)}\n"
                  f"ParticipantInfo IDs: {len(part_ids)}\n"
                  f"EEG with ParticipantInfo: {len(eeg_subject_ids & part_ids)}\n"
                  f"EEG without ParticipantInfo: {len(eeg_no_part)}")
        if eeg_no_part:
            detail += f"\n  Missing IDs: {sorted(eeg_no_part)}"
            for sid in eeg_no_part:
                report.flagged_subjects.append({
                    'subject': str(sid), 'reason': 'no_participant_info',
                    'detail': 'No entry in ParticipantInfo.csv'
                })
        
        status = "WARN" if eeg_no_part else "PASS"
        report.add_check("ParticipantInfo match", status, detail)
    else:
        report.add_check("ParticipantInfo match", "WARN",
            "ParticipantInfo.csv not provided — skipped")
    
    if psychopathology_path and os.path.exists(psychopathology_path):
        df_psych = pd.read_excel(psychopathology_path)
        psych_ids = set(df_psych['ID'].values)
        eeg_no_psych = eeg_subject_ids - psych_ids
        
        detail = (f"EEG subjects: {len(eeg_subject_ids)}\n"
                  f"Psychopathology IDs: {len(psych_ids)}\n"
                  f"EEG with Psychopathology: {len(eeg_subject_ids & psych_ids)}\n"
                  f"EEG without Psychopathology: {len(eeg_no_psych)}")
        if eeg_no_psych:
            detail += f"\n  Missing IDs: {sorted(eeg_no_psych)}"
        
        status = "WARN" if eeg_no_psych else "PASS"
        report.add_check("Psychopathology match", status, detail)
    else:
        report.add_check("Psychopathology match", "WARN",
            "Psychopathology file not provided — skipped")
    
    # ══════════════════════════════════════════════════════════════
    # GENERATE QC FIGURES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("Generating QC Figures")
    print(f"{'─'*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Amplitude distribution per channel (aggregated)
    all_ch_data = {ch: [] for ch in range(EXPECTED_COLS)}
    for data in file_data_cache.values():
        if data.shape[1] == EXPECTED_COLS:
            for ch in range(EXPECTED_COLS):
                all_ch_data[ch].extend(data[:, ch].tolist())
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Box plot of channel amplitudes
    ax = axes[0]
    bp_data = [np.array(all_ch_data[ch]) for ch in range(EXPECTED_COLS)]
    bp = ax.boxplot(bp_data, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Channel Amplitude Distributions (all files aggregated, outliers hidden)')
    ax.set_xticks(range(1, EXPECTED_COLS+1, 2))
    ax.set_xticklabels(range(0, EXPECTED_COLS, 2))
    ax.grid(axis='y', alpha=0.3)
    
    # Per-channel std
    ax = axes[1]
    ch_stds_all = []
    for filepath, data in file_data_cache.items():
        if data.shape[1] == EXPECTED_COLS:
            ch_stds_all.append(data.std(axis=0))
    if ch_stds_all:
        ch_stds_arr = np.array(ch_stds_all)  # (n_files, 34)
        ax.boxplot(ch_stds_arr, showfliers=True, patch_artist=True,
                   boxprops=dict(facecolor='#e74c3c', alpha=0.6))
        ax.axhline(FLAT_CHANNEL_STD, color='red', linestyle='--', 
                    label=f'Flat threshold ({FLAT_CHANNEL_STD} µV)')
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Per-File Channel Std (µV)')
        ax.set_title('Per-Channel Variability Across Files')
        ax.set_xticks(range(1, EXPECTED_COLS+1, 2))
        ax.set_xticklabels(range(0, EXPECTED_COLS, 2))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'qc_amplitude_distributions.pdf')
    plt.savefig(fig_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {fig_path}")
    
    # Figure 2: Per-subject QC heatmap
    if report.file_metrics:
        df_metrics = pd.DataFrame(report.file_metrics)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(complete) * 0.12)))
        
        for col_idx, (metric, title) in enumerate([
            ('max_abs_uV', 'Max |Amplitude| (µV)'),
            ('std_uV', 'Global Std (µV)'),
            ('mean_uV', 'Global Mean (µV)')
        ]):
            ax = axes[col_idx]
            pivot = df_metrics.pivot_table(index='subject', columns='condition', 
                                            values=metric, aggfunc='first')
            if not pivot.empty:
                cmap = 'YlOrRd' if metric == 'max_abs_uV' else 'viridis'
                im = ax.imshow(pivot.values, aspect='auto', cmap=cmap)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=4)
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45)
                ax.set_title(title, fontsize=10)
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle('Per-Subject Data Quality Heatmap', fontsize=12, fontweight='bold')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'qc_subject_heatmap.pdf')
        plt.savefig(fig_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved {fig_path}")
    
    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    report.summary()
    
    # Print dataset summary for downstream use
    n_complete = len(complete)
    print(f"\n{'─'*70}")
    print(f"DATASET READY FOR PROCESSING:")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Complete subjects (3 conditions): {n_complete}")
    print(f"  Total observations: {n_complete * 3}")
    print(f"  Dimensions per file: {EXPECTED_ROWS} × {EXPECTED_COLS}")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  Baseline: {BASELINE_SAMPLES} samples ({BASELINE_SAMPLES/SAMPLING_RATE*1000:.0f} ms)")
    print(f"  Post-stimulus: {POSTSTIM_SAMPLES} samples ({POSTSTIM_SAMPLES/SAMPLING_RATE*1000:.0f} ms)")
    print(f"{'─'*70}")
    
    report.save(output_dir)
    return report


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ARSPI-Net SHAPE Data Quality Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic check (batches only):
  python validate_shape_data.py --batch1 batch1.zip --batch2 batch2.zip --batch3 batch3.zip

  # Full check with clinical databases:
  python validate_shape_data.py \\
    --batch1 batch1attempt2.zip \\
    --batch2 batch2attempt2.zip \\
    --batch3 batch3attempt2.zip \\
    --participant_info ParticipantInfo.csv \\
    --psychopathology SHAPE_Community_Andrew_Psychopathology.xlsx \\
    --output_dir qc_output/
        """)
    
    parser.add_argument('--batch1', required=True, help='Path to batch 1 zip')
    parser.add_argument('--batch2', required=True, help='Path to batch 2 zip')
    parser.add_argument('--batch3', required=True, help='Path to batch 3 zip')
    parser.add_argument('--participant_info', default=None,
                        help='Path to ParticipantInfo.csv')
    parser.add_argument('--psychopathology', default=None,
                        help='Path to Psychopathology .xlsx')
    parser.add_argument('--output_dir', default='qc_output',
                        help='Output directory for QC reports')
    
    args = parser.parse_args()
    
    batch_paths = [args.batch1, args.batch2, args.batch3]
    for p in batch_paths:
        if not os.path.exists(p):
            print(f"ERROR: File not found: {p}")
            sys.exit(1)
    
    run_qc(batch_paths, args.participant_info, args.psychopathology, args.output_dir)
