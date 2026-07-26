#!/usr/bin/env python3
"""
Run the external-cohort replication on DataVerseNL doi:10.34894/TCRZEM.

    python experiments/external/run_tcrzem_replication.py \
        --data ./external_data/tcrzem --out external_results.json

Loads the BrainVision segmented EEG (grouped to Negative/Neutral/Positive
valence), trial-averages per subject x valence, and runs N1 (SRP-64 vs
fold-local PCA-64) and M2 (principal-angle / rho geometry) with the committed
operators. The EEG is restricted-download and gitignored; only the summary
statistics written to --out are retained.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import load_tcrzem as lt
import external_replication as ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./external_data/tcrzem")
    ap.add_argument("--out", default="external_results.json")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="cap subjects (for a quick pass)")
    ap.add_argument("--dims", default="16,32,64")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

    subjects = None
    if args.max_subjects:
        import glob, re
        found = sorted({re.match(r"(S\d+)_", os.path.basename(p)).group(1)
                        for p in glob.glob(os.path.join(args.data, "*_Mastoid_*.dat"))})
        subjects = set(found[: args.max_subjects])

    epochs, cond, subj = lt.load_trials(args.data, subjects=subjects)
    log(f"loaded {epochs.shape[0]} trials, {epochs.shape[2]} channels")
    raw, y, groups = ext.prepare_epochs(epochs, cond, subj)
    log(f"prepared {raw.shape[0]} ERPs / {len(set(groups.tolist()))} participants")

    dims = tuple(int(d) for d in args.dims.split(","))
    res = ext.replicate(raw, y, groups, dims=dims)
    for d, rec in res["N1_srp_vs_pca"].items():
        log(f"N1 dim={d}: SRP {rec['srp_percent']:.1f}% vs PCA "
            f"{rec['foldlocal_pca_percent']:.1f}%  diff {rec['srp_minus_pca_pp']:+.1f} "
            f"CI {rec['srp_minus_pca_ci95_pp']}")
    g = res["M2_geometry"]
    log(f"M2: PCA angle-to-subject {g['pca_min_angle_to_subject_deg']} deg, "
        f"to-condition {g['pca_min_angle_to_condition_deg']} deg; "
        f"rho pca={g['rho_pca_compressed']} srp={g['rho_srp_compressed']} amb={g['rho_ambient']}")

    out = {
        "dataset": "DataVerseNL doi:10.34894/TCRZEM (IAPS passive viewing, Neg/Neu/Pos)",
        "paradigm": "affective-picture ERP, three valence conditions",
        "cohort": "external replication of SHAPE three-condition claims",
        "results": res,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
