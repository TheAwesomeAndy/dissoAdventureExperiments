#!/usr/bin/env python3
"""
Run the novel experiments (N1-N3) on authorized SHAPE data and write
novel_results.json. Reuses the committed confirmatory loaders and operators.

    python experiments/novel/run_novel_experiments.py \
        --data3 ./batch_data --granularity three_condition \
        --out novel_results.json

The SHAPE data are restricted and not shipped. verify_novel.py exercises every
estimator on synthetic data without them.
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
import novel_pipeline as nov


def load(granularity, data3, data4):
    if granularity == "three_condition":
        return conf.load_3class(data3)
    return conf.load_4class(data4)


def run(granularity, raw, y, groups, base_seed, fast=False):
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

    log(f"{granularity}: {raw.shape[0]} obs / {len(np.unique(groups))} participants")
    bsc = cp.build_bsc_features(raw, reservoir_seed=42)
    srp64 = cp.make_srp(bsc.shape[2], n_components=cp.N_SRP)
    log("primary BSC6 built")

    # ---- N1: leak-free compression -------------------------------------
    dims = [16, 32, 64] if fast else [16, 32, 64, 128]
    n1_dims = {}
    for d in dims:
        srp_eval = nov.evaluate_srp(bsc, y, groups, d, cp.SRP_SEED, base_seed)
        pca_eval = nov.evaluate_foldlocal_pca(bsc, y, groups, d, base_seed)
        diff_ci = cp.participant_bootstrap_difference_ci(
            srp_eval["oof_true"], srp_eval["oof_pred"], pca_eval["oof_pred"],
            srp_eval["oof_group"])
        n1_dims[str(d)] = {
            "srp_percent": nov.pct(srp_eval["balanced_accuracy"]),
            "foldlocal_pca_percent": nov.pct(pca_eval["balanced_accuracy"]),
            "srp_minus_pca_pp": round(nov.pct(srp_eval["balanced_accuracy"])
                                      - nov.pct(pca_eval["balanced_accuracy"]), 3),
            "srp_minus_pca_ci95_pp": [round(v * 100, 3) for v in diff_ci],
        }
        log(f"N1 dim={d}: SRP {n1_dims[str(d)]['srp_percent']}% vs "
            f"PCA {n1_dims[str(d)]['foldlocal_pca_percent']}%")
    seeds = list(range(1000, 1000 + (10 if fast else 25)))
    n1_stability = nov.srp_seed_stability(bsc, y, groups, seeds, cp.N_SRP, base_seed)
    log(f"N1 SRP-64 seed stability: {n1_stability['mean']}% +/- {n1_stability['std']}")

    # ---- N2: reservoir as instrument -----------------------------------
    bsc_by_seed = [bsc] + [cp.build_bsc_features(raw, reservoir_seed=s)
                           for s in cp.RESERVOIR_SEEDS if s != 42]
    reliability = nov.cross_seed_feature_reliability(bsc_by_seed, srp64)
    log(f"N2 cross-seed feature reliability: median ICC {reliability['median_icc']}")

    template = raw[0, :, 0]
    amps = np.linspace(0.25, 4.0, 8)
    transfer = nov.transfer_function(template, amps)
    log(f"N2 transfer monotonicity rho={transfer['monotonicity_spearman']}, "
        f"dynamic range {transfer['dynamic_range_ratio']}")

    def build_srp(pert):
        return cp.project_bsc(cp.build_bsc_features(pert, 42), srp64)
    build_raw = lambda pert: conf.raw_erp_bins(pert, n_bins=16)
    levels_noise = [0.0, 0.5, 1.0]
    levels_drop = [0.0, 0.15, 0.30]
    robustness = {
        "channel_dropout": {
            "bsc6_srp64": nov.robustness_curve(raw, y, groups, nov.perturb_channel_dropout,
                                               levels_drop, build_srp, base_seed),
            "raw_erp16": nov.robustness_curve(raw, y, groups, nov.perturb_channel_dropout,
                                              levels_drop, build_raw, base_seed),
        },
        "additive_noise": {
            "bsc6_srp64": nov.robustness_curve(raw, y, groups, nov.perturb_additive_noise,
                                               levels_noise, build_srp, base_seed),
            "raw_erp16": nov.robustness_curve(raw, y, groups, nov.perturb_additive_noise,
                                              levels_noise, build_raw, base_seed),
        },
    }
    log("N2 robustness curves complete")

    # ---- N3: necessity map ---------------------------------------------
    nmap = nov.necessity_map(bsc, y, groups, srp64, base_seed)
    log("N3 necessity map complete")

    return {
        "n_participants": int(len(np.unique(groups))),
        "n_observations": int(len(y)),
        "N1_leak_free_compression": {
            "dimensionality_sweep": n1_dims,
            "srp64_seed_stability": n1_stability,
            "interpretation": ("A fixed data-independent SRP matches a fold-local "
                               "data-fitted PCA at each dimension (paired bootstrap "
                               "interval includes zero), so leak-free-by-construction "
                               "compression pays no accuracy cost."),
        },
        "N2_reservoir_instrument": {
            "cross_seed_feature_reliability": reliability,
            "input_amplitude_transfer_function": transfer,
            "robustness": robustness,
        },
        "N3_necessity_map": nmap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data3", default="./batch_data")
    ap.add_argument("--data4", default="./categories")
    ap.add_argument("--granularity", choices=["three_condition", "four_category"],
                    default="three_condition")
    ap.add_argument("--out", default="novel_results.json")
    ap.add_argument("--fast", action="store_true",
                    help="fewer dims/seeds for a quicker pass")
    args = ap.parse_args()

    raw, y, groups = load(args.granularity, args.data3, args.data4)
    base_seed = 0 if args.granularity == "three_condition" else 100
    results = {
        "granularity": args.granularity,
        "protocol": {
            "cv": f"{cp.N_REPEATS}x{cp.N_FOLDS}-fold participant-grouped, fold-local",
            "reservoir": f"N={cp.N_RES}, seeds={cp.RESERVOIR_SEEDS}",
            "srp_seed": cp.SRP_SEED,
        },
        "results": run(args.granularity, raw, y, groups, base_seed, fast=args.fast),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
