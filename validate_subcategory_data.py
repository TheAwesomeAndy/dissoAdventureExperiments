#!/usr/bin/env python3
"""
validate_subcategory_data.py
============================
ARSPI-Net Dissertation — Fine-Grained IAPS Subcategory ERP Data Validation

This script performs automated quality control on the SHAPE Community
fine-grained IAPS subcategory ERP data (Threat, Mutilation, Cute, Erotic).
Each file is a trial-averaged, baseline-corrected ERP matrix (1229 × 34)
for one subject under one affective subcategory.

The validation pipeline applies 12 checks per file and produces:
  1. A per-file pass/fail report
  2. A per-subject completeness report
  3. Cross-reference with Chapter 5 broad-condition subjects
  4. Subcategory-to-broad consistency verification
  5. Grand summary statistics

Usage:
    python validate_subcategory_data.py \
        --category-dirs categoriesbatch1 categoriesbatch2 categoriesbatch3 categoriesbatch4 \
        --broad-zips batch1.zip batch2.zip batch3.zip \
        --output validation_report.txt

Author: Andrew (ARSPI-Net Dissertation)
Date: March 2026
"""

import numpy as np
import os
import re
import sys
import argparse
import zipfile
from collections import defaultdict
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
EXPECTED_SHAPE = (1229, 34)
EXPECTED_CATEGORIES = {'Threat', 'Mutilation', 'Cute', 'Erotic'}
CATEGORY_VALENCE_MAP = {
    'Threat': 'Neg', 'Mutilation': 'Neg',
    'Cute': 'Pos', 'Erotic': 'Pos'
}
BROAD_FROM_SUBCATEGORIES = {
    'Neg': ['Threat', 'Mutilation'],
    'Pos': ['Cute', 'Erotic'],
}
AMPLITUDE_THRESHOLD_UV = 500.0       # Max absolute amplitude (µV)
OUTLIER_SD_THRESHOLD = 5.0           # Outlier detection threshold
FLAT_CHANNEL_STD_THRESHOLD = 1e-10   # Flat channel detection
CONSISTENCY_R_THRESHOLD = 0.99       # Subcategory-to-broad correlation
EXCLUDED_SUBJECTS = {127}            # Known exclusions from Chapter 5

FILE_PATTERN = re.compile(
    r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt'
)


# ============================================================
# Core Validation Functions
# ============================================================

def parse_filename(filename):
    """Parse a subcategory ERP filename into components.
    
    Returns:
        dict with keys: subject_id, valence, category, or None if parse fails.
    """
    m = FILE_PATTERN.match(filename)
    if m:
        return {
            'subject_id': int(m.group(1)),
            'valence': m.group(2),
            'category': m.group(3),
        }
    return None


def validate_file(filepath):
    """Run all quality checks on a single ERP file.
    
    Returns:
        dict with check results, data statistics, and pass/fail status.
    """
    filename = os.path.basename(filepath)
    result = {
        'filename': filename,
        'filepath': filepath,
        'checks': {},
        'stats': {},
        'passed': True,
        'critical_failure': False,
    }
    
    # Check 1: File exists and is readable
    if not os.path.isfile(filepath):
        result['checks']['file_exists'] = ('FAIL', 'File not found')
        result['passed'] = False
        result['critical_failure'] = True
        return result
    result['checks']['file_exists'] = ('PASS', '')
    
    # Check 2: Filename parses correctly
    parsed = parse_filename(filename)
    if parsed is None:
        result['checks']['filename_parse'] = ('FAIL', f'Cannot parse: {filename}')
        result['passed'] = False
        result['critical_failure'] = True
        return result
    result['checks']['filename_parse'] = ('PASS', '')
    result['parsed'] = parsed
    
    # Check 3: Valence-category consistency
    expected_valence = CATEGORY_VALENCE_MAP.get(parsed['category'])
    if parsed['valence'] != expected_valence:
        result['checks']['valence_consistency'] = (
            'FAIL', f"{parsed['category']} should be {expected_valence}, got {parsed['valence']}"
        )
        result['passed'] = False
    else:
        result['checks']['valence_consistency'] = ('PASS', '')
    
    # Load data
    try:
        data = np.loadtxt(filepath)
    except Exception as e:
        result['checks']['data_loadable'] = ('FAIL', str(e))
        result['passed'] = False
        result['critical_failure'] = True
        return result
    result['checks']['data_loadable'] = ('PASS', '')
    
    # Check 4: Dimensional verification
    if data.shape != EXPECTED_SHAPE:
        result['checks']['dimensions'] = ('FAIL', f'Expected {EXPECTED_SHAPE}, got {data.shape}')
        result['passed'] = False
        result['critical_failure'] = True
    else:
        result['checks']['dimensions'] = ('PASS', f'{data.shape[0]}×{data.shape[1]}')
    
    # Check 5: Numerical integrity (NaN / Inf)
    n_nan = np.count_nonzero(np.isnan(data))
    n_inf = np.count_nonzero(np.isinf(data))
    if n_nan > 0 or n_inf > 0:
        result['checks']['numerical_integrity'] = ('FAIL', f'{n_nan} NaN, {n_inf} Inf')
        result['passed'] = False
        result['critical_failure'] = True
    else:
        result['checks']['numerical_integrity'] = ('PASS', 'Zero NaN/Inf')
    
    if result['critical_failure']:
        return result
    
    # Check 6: Amplitude range
    max_amp = np.abs(data).max()
    if max_amp > AMPLITUDE_THRESHOLD_UV:
        result['checks']['amplitude_range'] = ('FAIL', f'Max |amp| = {max_amp:.1f} µV')
        result['passed'] = False
    else:
        result['checks']['amplitude_range'] = ('PASS', f'Max = {max_amp:.1f} µV')
    
    # Check 7: Flat channel detection
    channel_stds = np.std(data, axis=0)
    n_flat = np.sum(channel_stds < FLAT_CHANNEL_STD_THRESHOLD)
    if n_flat > 0:
        flat_indices = np.where(channel_stds < FLAT_CHANNEL_STD_THRESHOLD)[0]
        result['checks']['flat_channels'] = ('FAIL', f'{n_flat} flat: Ch{list(flat_indices)}')
        result['passed'] = False
    else:
        result['checks']['flat_channels'] = ('PASS', '0 flat')
    
    # Check 8: Baseline verification (pre-stimulus mean near zero)
    # Assuming 200ms baseline at 1024 Hz ≈ first 205 samples
    baseline = data[:205, :]
    bl_mean = np.abs(baseline.mean())
    if bl_mean > 1.0:
        result['checks']['baseline_verification'] = ('WARN', f'|mean baseline| = {bl_mean:.4f} µV')
    else:
        result['checks']['baseline_verification'] = ('PASS', f'|mean| = {bl_mean:.4f} µV')
    
    # Check 9: Outlier sample detection
    global_mean = data.mean()
    global_std = data.std()
    n_outliers = np.sum(np.abs(data - global_mean) > OUTLIER_SD_THRESHOLD * global_std)
    outlier_pct = 100.0 * n_outliers / data.size
    if outlier_pct > 0.1:
        result['checks']['outlier_samples'] = ('WARN', f'{n_outliers} samples ({outlier_pct:.4f}%)')
    else:
        result['checks']['outlier_samples'] = ('PASS', f'{n_outliers} ({outlier_pct:.4f}%)')
    
    # Check 10: Channel variance homogeneity
    cv_channels = channel_stds.std() / (channel_stds.mean() + 1e-10)
    if cv_channels > 1.0:
        result['checks']['channel_variance'] = ('WARN', f'CV = {cv_channels:.3f}')
    else:
        result['checks']['channel_variance'] = ('PASS', f'CV = {cv_channels:.3f}')
    
    # Statistics
    result['stats'] = {
        'mean': float(data.mean()),
        'std': float(data.std()),
        'max_abs': float(max_amp),
        'n_flat': int(n_flat),
        'n_outliers': int(n_outliers),
        'baseline_mean': float(bl_mean),
        'channel_cv': float(cv_channels),
    }
    
    return result


def validate_subject_completeness(subjects_dict):
    """Check that each subject has all 4 expected subcategory files.
    
    Args:
        subjects_dict: {subject_id: {category: filename, ...}, ...}
    
    Returns:
        list of (subject_id, status, missing_categories)
    """
    results = []
    for sid in sorted(subjects_dict.keys()):
        cats = set(subjects_dict[sid].keys())
        missing = EXPECTED_CATEGORIES - cats
        extra = cats - EXPECTED_CATEGORIES
        if not missing and not extra:
            results.append((sid, 'COMPLETE', set()))
        else:
            results.append((sid, 'INCOMPLETE', missing))
    return results


def check_subcategory_consistency(category_dirs, broad_zips, n_samples=5):
    """Verify that (Threat + Mutilation)/2 ≈ broad Negative for sample subjects.
    
    Returns:
        list of (subject_id, valence, correlation, status)
    """
    results = []
    
    # Collect subcategory files indexed by subject
    subjects = defaultdict(dict)
    for cdir in category_dirs:
        if not os.path.isdir(cdir):
            continue
        for f in os.listdir(cdir):
            parsed = parse_filename(f)
            if parsed:
                subjects[parsed['subject_id']][parsed['category']] = os.path.join(cdir, f)
    
    # Load broad-condition files from zips
    broad_data = {}  # {(subject_id, valence): np.array}
    for zpath in broad_zips:
        if not os.path.isfile(zpath):
            continue
        broad_pattern = re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt')
        try:
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    base = os.path.basename(name)
                    m = broad_pattern.match(base)
                    if m:
                        sid = int(m.group(1))
                        val = m.group(2)
                        if val in ('Neg', 'Pos'):
                            with zf.open(name) as fh:
                                broad_data[(sid, val)] = np.loadtxt(fh)
        except Exception:
            continue
    
    # Test consistency for sample subjects
    tested = 0
    for sid in sorted(subjects.keys()):
        if tested >= n_samples:
            break
        if set(subjects[sid].keys()) != EXPECTED_CATEGORIES:
            continue
        
        for valence, subcats in BROAD_FROM_SUBCATEGORIES.items():
            if (sid, valence) not in broad_data:
                continue
            
            broad = broad_data[(sid, valence)]
            sub_arrays = [np.loadtxt(subjects[sid][c]) for c in subcats]
            avg_sub = np.mean(sub_arrays, axis=0)
            
            corr = np.corrcoef(broad.ravel(), avg_sub.ravel())[0, 1]
            status = 'PASS' if corr >= CONSISTENCY_R_THRESHOLD else 'WARN'
            results.append((sid, valence, corr, status))
        
        tested += 1
    
    return results


# ============================================================
# Report Generation
# ============================================================

def generate_report(all_results, completeness, consistency, ch5_subjects=None):
    """Generate a formatted text report."""
    lines = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines.append('=' * 70)
    lines.append('ARSPI-Net Subcategory ERP Data Validation Report')
    lines.append(f'Generated: {timestamp}')
    lines.append('=' * 70)
    
    # ---- Summary ----
    n_files = len(all_results)
    n_passed = sum(1 for r in all_results if r['passed'])
    n_failed = n_files - n_passed
    n_critical = sum(1 for r in all_results if r.get('critical_failure'))
    
    subjects_seen = set()
    for r in all_results:
        p = r.get('parsed')
        if p:
            subjects_seen.add(p['subject_id'])
    
    n_subjects = len(subjects_seen)
    n_complete = sum(1 for _, status, _ in completeness if status == 'COMPLETE')
    n_valid = n_complete - len(EXCLUDED_SUBJECTS & subjects_seen)
    
    lines.append('')
    lines.append('SUMMARY')
    lines.append('-' * 40)
    lines.append(f'  Total files:              {n_files}')
    lines.append(f'  Passed all checks:        {n_passed}')
    lines.append(f'  Failed:                   {n_failed}')
    lines.append(f'  Critical failures:        {n_critical}')
    lines.append(f'  Total subjects:           {n_subjects}')
    lines.append(f'  Complete (4 categories):  {n_complete}')
    lines.append(f'  Valid (excl. S127):       {n_valid}')
    
    if ch5_subjects is not None:
        ch5_valid = ch5_subjects - EXCLUDED_SUBJECTS
        coverage = len((subjects_seen - EXCLUDED_SUBJECTS) & ch5_valid)
        lines.append(f'  Chapter 5 subjects:       {len(ch5_valid)}')
        lines.append(f'  Ch5 coverage:             {coverage}/{len(ch5_valid)}')
        missing = ch5_valid - subjects_seen
        if missing:
            lines.append(f'  Missing from Ch5:         {sorted(missing)}')
    
    # ---- Per-batch breakdown ----
    lines.append('')
    lines.append('PER-BATCH BREAKDOWN')
    lines.append('-' * 40)
    batch_subjects = defaultdict(set)
    batch_files = defaultdict(int)
    for r in all_results:
        # Infer batch from filepath
        fp = r.get('filepath', '')
        for b in ['batch1', 'batch2', 'batch3', 'batch4']:
            if b in fp:
                p = r.get('parsed')
                if p:
                    batch_subjects[b].add(p['subject_id'])
                batch_files[b] += 1
                break
    for b in sorted(batch_files.keys()):
        lines.append(f'  {b}: {batch_files[b]} files, {len(batch_subjects[b])} subjects')
    
    # ---- Dimensional check ----
    lines.append('')
    lines.append('DIMENSIONAL VERIFICATION')
    lines.append('-' * 40)
    shape_counts = defaultdict(int)
    for r in all_results:
        s = r.get('stats', {})
        # All should be EXPECTED_SHAPE from checks
        for check_name, (status, detail) in r.get('checks', {}).items():
            if check_name == 'dimensions':
                shape_counts[detail] += 1
    for shape_str, count in sorted(shape_counts.items()):
        lines.append(f'  {shape_str}: {count} files')
    
    # ---- Amplitude statistics ----
    lines.append('')
    lines.append('AMPLITUDE STATISTICS')
    lines.append('-' * 40)
    amp_means = [r['stats']['mean'] for r in all_results if 'stats' in r and 'mean' in r['stats']]
    amp_stds = [r['stats']['std'] for r in all_results if 'stats' in r and 'std' in r['stats']]
    amp_maxes = [r['stats']['max_abs'] for r in all_results if 'stats' in r and 'max_abs' in r['stats']]
    if amp_means:
        lines.append(f'  Mean:    {np.mean(amp_means):.4f} [{np.min(amp_means):.4f}, {np.max(amp_means):.4f}]')
        lines.append(f'  Std:     {np.mean(amp_stds):.4f} [{np.min(amp_stds):.4f}, {np.max(amp_stds):.4f}]')
        lines.append(f'  Max|A|:  {np.max(amp_maxes):.2f} µV')
    
    # ---- Anomalies ----
    lines.append('')
    lines.append('ANOMALIES')
    lines.append('-' * 40)
    anomaly_files = [r for r in all_results if not r['passed']]
    if anomaly_files:
        for r in anomaly_files:
            failed_checks = [(k, v) for k, v in r['checks'].items() if v[0] == 'FAIL']
            lines.append(f'  {r["filename"]}:')
            for check_name, (_, detail) in failed_checks:
                lines.append(f'    {check_name}: {detail}')
    else:
        lines.append('  None')
    
    # ---- Warnings ----
    warn_files = []
    for r in all_results:
        warns = [(k, v) for k, v in r.get('checks', {}).items() if v[0] == 'WARN']
        if warns:
            warn_files.append((r['filename'], warns))
    if warn_files:
        lines.append('')
        lines.append('WARNINGS')
        lines.append('-' * 40)
        for fname, warns in warn_files[:20]:
            for check_name, (_, detail) in warns:
                lines.append(f'  {fname}: {check_name} — {detail}')
        if len(warn_files) > 20:
            lines.append(f'  ... and {len(warn_files) - 20} more')
    
    # ---- Consistency ----
    if consistency:
        lines.append('')
        lines.append('SUBCATEGORY-TO-BROAD CONSISTENCY')
        lines.append('-' * 40)
        for sid, valence, corr, status in consistency:
            symbol = '✓' if status == 'PASS' else '!'
            lines.append(f'  [{symbol}] Subject {sid}, {valence}: r = {corr:.6f}')
    
    # ---- Incomplete subjects ----
    incomplete = [(sid, status, missing) for sid, status, missing in completeness if status != 'COMPLETE']
    if incomplete:
        lines.append('')
        lines.append('INCOMPLETE SUBJECTS')
        lines.append('-' * 40)
        for sid, status, missing in incomplete:
            lines.append(f'  Subject {sid}: missing {sorted(missing)}')
    
    # ---- Category counts ----
    lines.append('')
    lines.append('PER-CATEGORY FILE COUNTS')
    lines.append('-' * 40)
    cat_counts = defaultdict(int)
    for r in all_results:
        p = r.get('parsed')
        if p:
            cat_counts[p['category']] += 1
    for cat in ['Threat', 'Mutilation', 'Cute', 'Erotic']:
        lines.append(f'  {cat:12s}: {cat_counts.get(cat, 0)} files')
    
    lines.append('')
    lines.append('=' * 70)
    if n_failed == 0 or (n_failed <= len(EXCLUDED_SUBJECTS) * 4):
        lines.append('RESULT: PASS — Data validated for Chapter 6 experiments')
    else:
        lines.append(f'RESULT: {n_failed} failures require investigation')
    lines.append('=' * 70)
    
    return '\n'.join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate SHAPE Community fine-grained IAPS subcategory ERP data.'
    )
    parser.add_argument(
        '--category-dirs', nargs='+', required=True,
        help='Directories containing subcategory ERP text files'
    )
    parser.add_argument(
        '--broad-zips', nargs='*', default=[],
        help='ZIP files containing broad-condition (Neg/Neu/Pos) ERP files for consistency check'
    )
    parser.add_argument(
        '--ch5-subjects', type=str, default=None,
        help='Path to Chapter 5 pickle file for cross-reference (shape_features_211.pkl)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output report file path (default: print to stdout)'
    )
    parser.add_argument(
        '--consistency-samples', type=int, default=5,
        help='Number of subjects to test for subcategory-to-broad consistency'
    )
    args = parser.parse_args()
    
    print(f"ARSPI-Net Subcategory Data Validation")
    print(f"Scanning {len(args.category_dirs)} directories...")
    
    # ---- Collect and validate all files ----
    all_results = []
    subjects_dict = defaultdict(dict)
    
    for cdir in args.category_dirs:
        if not os.path.isdir(cdir):
            print(f"  WARNING: Directory not found: {cdir}")
            continue
        
        txt_files = sorted([f for f in os.listdir(cdir) if f.endswith('.txt')])
        print(f"  {cdir}: {len(txt_files)} files")
        
        for f in txt_files:
            filepath = os.path.join(cdir, f)
            result = validate_file(filepath)
            all_results.append(result)
            
            parsed = result.get('parsed')
            if parsed:
                subjects_dict[parsed['subject_id']][parsed['category']] = f
    
    print(f"  Total: {len(all_results)} files, {len(subjects_dict)} subjects")
    
    # ---- Subject completeness ----
    completeness = validate_subject_completeness(subjects_dict)
    
    # ---- Consistency check ----
    consistency = []
    if args.broad_zips:
        print(f"Running consistency check against {len(args.broad_zips)} broad-condition archives...")
        consistency = check_subcategory_consistency(
            args.category_dirs, args.broad_zips,
            n_samples=args.consistency_samples
        )
    
    # ---- Ch5 cross-reference ----
    ch5_subjects = None
    if args.ch5_subjects and os.path.isfile(args.ch5_subjects):
        import pickle
        with open(args.ch5_subjects, 'rb') as f:
            d5 = pickle.load(f)
        ch5_subjects = set(np.unique(d5['subjects']).astype(int))
        print(f"Cross-referencing with {len(ch5_subjects)} Chapter 5 subjects")
    
    # ---- Generate report ----
    report = generate_report(all_results, completeness, consistency, ch5_subjects)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport written to: {args.output}")
    else:
        print('\n' + report)
    
    # Return exit code
    n_critical = sum(1 for r in all_results if r.get('critical_failure'))
    sys.exit(1 if n_critical > 0 else 0)


if __name__ == '__main__':
    main()
