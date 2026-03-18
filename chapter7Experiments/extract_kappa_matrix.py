#!/usr/bin/env python3
"""
Extract kappa matrix from ch7_full_results.pkl as a compact CSV.
Run this where the pickle file lives, then paste the output back.
Usage:
  python3 extract_kappa_matrix.py
"""
import pickle, numpy as np
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_FILE = os.path.join(SCRIPT_DIR, 'chapter7_results', 'ch7_full_results.pkl')
r = pickle.load(open(PKL_FILE, 'rb'))
subjects = sorted(r['completed_subjects'])
cats = ['Threat', 'Mutilation', 'Cute', 'Erotic']
print(f"subject,Threat,Mutilation,Cute,Erotic")
for sid in subjects:
    vals = [f"{r['coupling_kappa'][(sid,c)]:.6f}" for c in cats]
    print(f"{sid},{','.join(vals)}")
