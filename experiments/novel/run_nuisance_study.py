#!/usr/bin/env python3
"""
Run the nuisance-alignment compression study (M1, M2, M3, M5).

  M1  controlled nuisance-ratio sweep  (synthetic, no data)
  M3  compressor comparison            (synthetic, no data)
  M2  geometric mechanism              (real data, if --data3 given)
  M5  reservoir disentanglement        (real data, if --data3 given)

    python experiments/novel/run_nuisance_study.py --data3 ./batch_data \
        --out nuisance_results.json

The synthetic parts run with no dataset and are the generalizable core claim;
the real-data parts anchor the mechanism on SHAPE.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "experiments", "confirmatory"))
sys.path.insert(0, _HERE)

import confirmatory_pipeline as cp
import run_confirmatory_validation as conf
import nuisance_study as ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data3", default=None, help="three-condition batch_data dir")
    ap.add_argument("--out", default="nuisance_results.json")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

    out = {"study": "nuisance-alignment compression", "components": {}}

    # ---- M1a: geometric mechanism (generalizable) ---------------------
    ratios = [0.0, 0.5, 1.0, 2.0, 4.0] if args.fast else \
             [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    m1a = ns.angle_rotation_sweep(ratios, n_components=16, dim=200,
                                  n_participants=60, subj_rank=20)
    out["components"]["M1a_angle_rotation"] = m1a
    for row in m1a["curve"]:
        log(f"M1a ratio={row['nuisance_ratio']:>4}: PCA angle-to-subject "
            f"{row['pca_angle_to_subject_deg']:>5.1f} deg, to-condition "
            f"{row['pca_angle_to_condition_deg']:>5.1f} deg")

    # ---- M1b: accuracy consequence (regime-dependent) -----------------
    m1b = ns.accuracy_sweep(ratios, n_components=16, dim=200, n_participants=60)
    out["components"]["M1b_accuracy_sweep"] = m1b
    log(f"M1b SRP>PCA accuracy crossover ratio = {m1b['crossover_ratio']}")

    # ---- M3: compressor comparison at a high nuisance ratio -----------
    Xc, yc, gc = ns.make_controlled(60, 3, 200, sigma_subj=3.0, sigma_noise=1.0, seed=1)
    comparators = {
        "srp64": ns._srp_fitter(64, cp.SRP_SEED),
        "foldlocal_pca64": ns._pca_fitter(64),
        "supervised_lda": ns._lda_fitter(),
        "random_subsample64": ns._subsample_fitter(64, 7),
    }
    m3 = {name: ns.pct(ns.evaluate_compressor(Xc, yc, gc, fit)["balanced_accuracy"])
          for name, fit in comparators.items()}
    out["components"]["M3_compressor_comparison"] = {
        "nuisance_ratio": 3.0, "dim": 200, "balanced_accuracy_percent": m3}
    log(f"M3 @ratio=3.0: {m3}")

    # ---- Real-data mechanism (M2, M5) ---------------------------------
    if args.data3 and os.path.isdir(args.data3):
        raw, y, groups = conf.load_3class(args.data3)
        log(f"real three_condition: {raw.shape[0]} obs / {len(np.unique(groups))} participants")
        bsc = cp.build_bsc_features(raw, reservoir_seed=42)
        bsc_flat = bsc.reshape(bsc.shape[0], -1)
        log("BSC6 built")

        geom = ns.compression_geometry(bsc_flat, y, groups, n_components=64)
        out["components"]["M2_geometry"] = geom
        log(f"M2 rho ambient={geom['rho_ambient']} pca={geom['rho_pca_compressed']} "
            f"srp={geom['rho_srp_compressed']}; PCA min-angle subj="
            f"{geom['pca_min_angle_to_subject_deg']} cond={geom['pca_min_angle_to_condition_deg']}")

        srp64 = cp.make_srp(bsc.shape[2], n_components=cp.N_SRP)
        srp_feat = cp.project_bsc(bsc, srp64)
        raw16 = conf.raw_erp_bins(raw, n_bins=16)
        dis = ns.disentanglement(raw16, bsc_flat, srp_feat, y, groups)
        out["components"]["M5_disentanglement"] = dis
        log(f"M5 rho: rawERP16={dis['raw_erp16_rho']} bsc6={dis['bsc6_rho']} "
            f"bsc6/srp64={dis['bsc6_srp64_rho']}")
    else:
        log("no --data3: skipping real-data M2/M5 (synthetic M1/M3 completed)")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
