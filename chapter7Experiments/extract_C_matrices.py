#!/usr/bin/env python3
"""
Extract full C matrices from ch7_full_results.pkl as a compact CSV.
Usage:
  python3 extract_C_matrices.py
Output:
  /mnt/user-data/outputs/C_matrices.csv
  844 rows × 16 columns (subject, category, then 14 correlation values)
  Each row is one (subject, category) observation.
  The 14 correlation columns are named {dyn_metric}_x_{topo_metric}.
"""
import pickle
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_FILE = os.path.join(SCRIPT_DIR, 'chapter7_results', 'ch7_full_results.pkl')
OUT_FILE = os.path.join(SCRIPT_DIR, 'chapter7_results', 'C_matrices.csv')

r = pickle.load(open(PKL_FILE, 'rb'))
subjects = sorted(r['completed_subjects'])
cats = ['Threat', 'Mutilation', 'Cute', 'Erotic']
dyn_names = r['dyn_names']   # 7 metrics
topo_names = r['topo_names'] # 2 metrics

# Build column names for the 14 flattened C entries
col_names = []
for d in dyn_names:
    for t in topo_names:
        col_names.append(f'{d}_x_{t}')

# Header
header = 'subject,category,' + ','.join(col_names)
print(header)
lines = [header]

for sid in subjects:
    for cat in cats:
        key = (sid, cat)
        if key not in r['coupling_C']:
            continue
        C = r['coupling_C'][key]  # (7, 2)
        # Flatten row-major: [d0_t0, d0_t1, d1_t0, d1_t1, ...]
        vals = ','.join(f'{C[j, k]:.6f}' for j in range(len(dyn_names))
                        for k in range(len(topo_names)))
        line = f'{sid},{cat},{vals}'
        lines.append(line)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f'\nWritten {len(lines)-1} rows to {OUT_FILE}')
print(f'Columns: subject, category, + {len(col_names)} correlation values')
print(f'Column names: {col_names}')
