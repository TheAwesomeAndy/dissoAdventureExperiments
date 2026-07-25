#!/usr/bin/env python3
"""
ARSPI-Net -- Confirmatory Validation Runner (real data)
=======================================================

Regenerates the corrected confirmatory classification results reported in the
dissertation and writes them as machine-readable JSON. Run this against the
authorized SHAPE data to reproduce every headline number and control:

    python run_confirmatory_validation.py \
        --data3 ./batch_data --data4 ./categories \
        --out confirmatory_results.json

The pipeline itself lives in confirmatory_pipeline.py and is verified without
data by verify_confirmatory.py. The output JSON can be checked automatically
against ../../results_manifest.json with check_against_manifest.py.

The SHAPE dataset is restricted (https://lab-can.com/shape/) and is NOT part of
this repository. Without it, use verify_confirmatory.py to confirm the
machinery; this runner needs the data to produce the numbers.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy.signal import decimate, butter, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "chapter5Experiments"))

import confirmatory_pipeline as cp


# ============================================================================
# Data loading
# ============================================================================
def load_3class(data_dir):
    """(N, 256, 34) z-scored EEG, y in {0,1,2}, participant ids -- reuses the
    validated experiment_zero loader."""
    from experiment_zero import load_shape_3class
    raw, y, subjects = load_shape_3class(data_dir)
    groups = np.array([int(s) for s in subjects])
    return raw, np.asarray(y), groups


_CAT4 = {"Threat": 0, "Mutilation": 1, "Cute": 2, "Erotic": 3}


def load_4class(data_dir, target_T=256, exclude=(127,)):
    """Self-contained 4-class loader matching the documented format:
    categoriesbatch{1-4}/SHAPE_Community_<id>_IAPS<val>_<Category>_BC.txt,
    preprocessed identically to the 3-class path (decimate to 256, z-score)."""
    pattern = re.compile(
        r"SHAPE_Community_(\d+)_IAPS[A-Za-z]+_(Threat|Mutilation|Cute|Erotic)_BC\.txt$")
    found = {}
    for root, _dirs, fnames in os.walk(data_dir):
        for fn in fnames:
            m = pattern.search(fn)
            if not m:
                continue
            sid, cat = int(m.group(1)), m.group(2)
            if sid in exclude:
                continue
            found[(sid, cat)] = os.path.join(root, fn)

    raw_list, y_list, grp_list = [], [], []
    for (sid, cat), path in sorted(found.items()):
        X = np.loadtxt(path)
        T0, n_ch = X.shape
        if T0 > 300:
            factor = max(1, T0 // target_T)
            Xds = np.zeros((T0 // factor, n_ch))
            for ch in range(n_ch):
                d = decimate(X[:, ch], factor)
                Xds[:, ch] = d[:Xds.shape[0]]
            Xds = Xds[:target_T]
        else:
            Xds = X[:target_T]
        for ch in range(n_ch):
            mu, sd = Xds[:, ch].mean(), Xds[:, ch].std()
            Xds[:, ch] = (Xds[:, ch] - mu) / sd if sd > 0 else 0.0
        raw_list.append(Xds)
        y_list.append(_CAT4[cat])
        grp_list.append(sid)
    return np.array(raw_list), np.array(y_list), np.array(grp_list)


# ============================================================================
# Representation builders
# ============================================================================
def raw_erp_bins(raw_data, n_bins=16):
    """Raw ERP reference: mean amplitude in n_bins temporal bins per channel."""
    N, T, n_ch = raw_data.shape
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    feats = np.zeros((N, n_ch * n_bins))
    for i in range(N):
        col = 0
        for ch in range(n_ch):
            for b in range(n_bins):
                feats[i, col] = raw_data[i, edges[b]:edges[b + 1], ch].mean()
                col += 1
    return feats


def conventional_features(raw_data, fs=256.0):
    """Conventional band-power + Hjorth features per channel (the matched
    non-neuromorphic baseline family)."""
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    N, T, n_ch = raw_data.shape
    nyq = fs / 2.0
    filters = [butter(3, [lo / nyq, hi / nyq], btype="band") for lo, hi in bands]
    feats = np.zeros((N, n_ch * (len(bands) + 3)))
    for i in range(N):
        col = 0
        for ch in range(n_ch):
            x = raw_data[i, :, ch]
            for (b, a) in filters:
                feats[i, col] = np.mean(filtfilt(b, a, x) ** 2)
                col += 1
            # Hjorth activity, mobility, complexity
            dx = np.diff(x)
            ddx = np.diff(dx)
            var0 = np.var(x) + 1e-12
            var1 = np.var(dx) + 1e-12
            var2 = np.var(ddx) + 1e-12
            activity = var0
            mobility = np.sqrt(var1 / var0)
            complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
            feats[i, col:col + 3] = [activity, mobility, complexity]
            col += 3
    return feats


def pp(x):
    """percentage-point helper (round)."""
    return round(float(x) * 100, 4)


# ============================================================================
# Orchestration for one granularity
# ============================================================================
def run_granularity(name, raw_data, y, groups, base_seed):
    print(f"\n=== {name}: N={len(y)} obs, {len(np.unique(groups))} participants ===")
    srp = cp.make_srp(cp.N_BINS * cp.N_RES)

    # BSC6 at the primary reservoir seed (42) -> SRP-64 features
    bsc = cp.build_bsc_features(raw_data, reservoir_seed=42)
    Xsrp = cp.project_bsc(bsc, srp)

    Xraw = raw_erp_bins(raw_data, n_bins=16)
    Xconv = conventional_features(raw_data)
    Xfuse = np.concatenate([Xsrp, Xconv], axis=1)

    reps = {}
    for rep_name, X in [("raw_erp16", Xraw), ("conventional", Xconv),
                        ("bsc6_srp64", Xsrp), ("bsc6_srp64_plus_conv", Xfuse)]:
        res = cp.evaluate_features(X, y, groups, base_seed=base_seed)
        ci = cp.participant_bootstrap_ci(res["oof_true"], res["oof_pred"],
                                         res["oof_group"])
        reps[rep_name] = {"balanced_accuracy": pp(res["balanced_accuracy"]),
                          "ci95": [pp(ci[0] / 100), pp(ci[1] / 100)]}
        print(f"  {rep_name:24s} {reps[rep_name]['balanced_accuracy']:.1f}% "
              f"{reps[rep_name]['ci95']}")

    advantage = reps["bsc6_srp64"]["balanced_accuracy"] - \
        reps["conventional"]["balanced_accuracy"]

    perm = cp.permutation_test(Xsrp, y, groups, seed=base_seed)

    # Destruction controls (drop vs intact BSC6/SRP-64)
    rng = np.random.RandomState(base_seed + 101)
    intact = reps["bsc6_srp64"]["balanced_accuracy"]
    controls = {}
    for cname, dbsc in [
        ("temporal_bin_shuffle", cp.destroy_temporal_bin_order(bsc, rng)),
        ("temporal_collapse", cp.destroy_temporal_collapse(bsc)),
        ("electrode_shuffle", cp.destroy_electrode_identity(bsc, rng)),
    ]:
        Xd = cp.project_bsc(dbsc, srp)
        rd = cp.evaluate_features(Xd, y, groups, base_seed=base_seed)
        controls[cname] = round(pp(rd["balanced_accuracy"]) - intact, 4)
        print(f"  control {cname:22s} {controls[cname]:+.1f} pp")

    # Seed robustness
    seed_acc = {}
    for rs in cp.RESERVOIR_SEEDS:
        b = cp.build_bsc_features(raw_data, reservoir_seed=rs)
        rr = cp.evaluate_features(cp.project_bsc(b, srp), y, groups,
                                  base_seed=base_seed)
        seed_acc[str(rs)] = pp(rr["balanced_accuracy"])

    # Temporal-window ablation
    windows = {}
    for wname, (lo, hi) in cp.TEMPORAL_WINDOWS.items():
        rw = raw_data if (lo, hi) == (0.0, 1.0) else cp.window_slice(raw_data, lo, hi)
        bw = cp.build_bsc_features(rw, reservoir_seed=42)
        rr = cp.evaluate_features(cp.project_bsc(bw, srp), y, groups,
                                  base_seed=base_seed)
        windows[wname] = pp(rr["balanced_accuracy"])

    return {
        "n_participants": int(len(np.unique(groups))),
        "n_observations": int(len(y)),
        "representations": reps,
        "advantage_over_conventional_pp": round(advantage, 4),
        "permutation": {"p_value": perm["p_value"],
                        "n_permutations": perm["n_permutations"]},
        "destruction_controls_pp": controls,
        "seed_robustness": seed_acc,
        "temporal_windows": windows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data3", default="./batch_data",
                    help="3-class SHAPE directory (batch_data/)")
    ap.add_argument("--data4", default="./categories",
                    help="4-class SHAPE directory (categories/)")
    ap.add_argument("--out", default="confirmatory_results.json")
    ap.add_argument("--skip4", action="store_true",
                    help="run only the three-condition granularity")
    args = ap.parse_args()

    results = {"protocol": {
        "cv": "5 repeats x 5-fold participant-grouped (StratifiedGroupKFold)",
        "preprocessing": "fold-local StandardScaler on training participants only",
        "compression": f"fixed SparseRandomProjection {cp.N_BINS*cp.N_RES}->{cp.N_SRP}, "
                       f"seed={cp.SRP_SEED}, shared across electrodes (data-independent)",
        "readout": "LogisticRegression (accuracy); RidgeClassifier (permutation)",
        "reservoir": f"N={cp.N_RES}, beta=0.05, theta=0.5, seeds={cp.RESERVOIR_SEEDS}",
        "intervals": "participant-bootstrap 95%",
    }, "granularities": {}}

    if not os.path.isdir(args.data3):
        print(f"[error] 3-class data dir not found: {args.data3}\n"
              f"        SHAPE data is restricted and not shipped. See README.",
              file=sys.stderr)
        sys.exit(2)

    raw3, y3, g3 = load_3class(args.data3)
    results["granularities"]["three_condition"] = run_granularity(
        "three_condition", raw3, y3, g3, base_seed=0)

    if not args.skip4 and os.path.isdir(args.data4):
        raw4, y4, g4 = load_4class(args.data4)
        results["granularities"]["four_category"] = run_granularity(
            "four_category", raw4, y4, g4, base_seed=100)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
