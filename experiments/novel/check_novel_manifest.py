#!/usr/bin/env python3
"""
Consistency checker: committed novel/nuisance/external result JSONs vs the
exploratory entries in results_manifest.json.

The confirmatory checker covers only the confirmatory rows; this closes the gap
for the exploratory rows -- a green CI run also establishes that the committed
exploratory result files and their manifest entries agree.

Hardening (audit response):
  * Every committed result file is MANDATORY. A missing file is a provenance
    failure, not a silently skipped comparison.
  * Every mapped field is MANDATORY on both sides. A missing key on either the
    JSON or the manifest is a failure.
  * The tolerance is strict (TOL). These are serialized copies of the SAME run
    recorded to three decimals, not independent floating-point reruns, so they
    must be identical up to last-digit rounding.
  * Zero comparisons is a FAILURE, never a pass.

CI-safe: needs only the committed JSONs, no restricted data.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MANIFEST = os.path.join(_REPO, "results_manifest.json")
_EXT = os.path.join(_REPO, "experiments", "external")

# Serialized copies recorded to 3 decimals -> identical up to last-digit rounding.
TOL = 0.001

FAIL = 0
CHECKED = 0


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def entry(manifest, mid):
    for r in manifest["results"]:
        if r.get("id") == mid:
            return r["value"]
    return None


def dig(obj, path):
    """Follow a dotted/indexed path; return (found, value)."""
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return False, None
    return True, cur


def cmp(label, got_ok, got, want_ok, want, tol=TOL):
    global FAIL, CHECKED
    if not got_ok or not want_ok:
        FAIL += 1
        side = []
        if not got_ok:
            side.append("json-missing")
        if not want_ok:
            side.append("manifest-missing")
        print(f"  [MISSING] {label}: {','.join(side)}")
        return
    try:
        ok = abs(float(got) - float(want)) <= tol
    except (TypeError, ValueError):
        ok = got == want
    if ok:
        CHECKED += 1
        print(f"  [ok] {label}: {got} ~ {want}")
    else:
        FAIL += 1
        print(f"  [MISMATCH] {label}: json={got} manifest={want}")


def require_file(path):
    global FAIL
    if not os.path.exists(path):
        FAIL += 1
        print(f"  [MISSING FILE] {os.path.relpath(path, _REPO)} "
              f"(committed result file is mandatory)")
        return None
    return load(path)


def check_pairs(label_prefix, jobj, mval, pairs):
    """pairs: list of (label, json_path, manifest_path)."""
    for lbl, jpath, mpath in pairs:
        jok, jv = dig(jobj, jpath)
        mok, mv = dig(mval, mpath) if mval is not None else (False, None)
        cmp(f"{label_prefix} {lbl}", jok, jv, mok, mv)


def main():
    m = load(_MANIFEST)

    # ---- N1 three-condition ------------------------------------------
    d = require_file(os.path.join(_HERE, "novel_results_3class.json"))
    v = entry(m, "novel_n1_srp_vs_pca")
    if d is not None:
        n1 = d["results"]["N1_leak_free_compression"]
        check_pairs("N1/3class", n1, v, [
            ("d64 srp", ["dimensionality_sweep", "64", "srp_percent"], ["srp_percent"]),
            ("d64 pca", ["dimensionality_sweep", "64", "foldlocal_pca_percent"], ["foldlocal_pca_percent"]),
            ("d64 diff", ["dimensionality_sweep", "64", "srp_minus_pca_pp"], ["srp_minus_pca_pp"]),
            ("d64 ci-lo", ["dimensionality_sweep", "64", "srp_minus_pca_ci95_pp", 0], ["ci95_pp", 0]),
            ("d64 ci-hi", ["dimensionality_sweep", "64", "srp_minus_pca_ci95_pp", 1], ["ci95_pp", 1]),
            ("seed mean", ["srp64_seed_stability", "mean"], ["srp64_seed_stability_mean"]),
            ("seed std", ["srp64_seed_stability", "std"], ["srp64_seed_stability_std"]),
        ])

    # ---- N1 four-category --------------------------------------------
    d = require_file(os.path.join(_HERE, "novel_results_4class.json"))
    v = entry(m, "novel_n1_srp_vs_pca_4class")
    if d is not None:
        n1 = d["results"]["N1_leak_free_compression"]
        check_pairs("N1/4class", n1, v, [
            ("d64 srp", ["dimensionality_sweep", "64", "srp_percent"], ["srp_percent"]),
            ("d64 pca", ["dimensionality_sweep", "64", "foldlocal_pca_percent"], ["foldlocal_pca_percent"]),
            ("d64 diff", ["dimensionality_sweep", "64", "srp_minus_pca_pp"], ["srp_minus_pca_pp"]),
            ("d64 ci-lo", ["dimensionality_sweep", "64", "srp_minus_pca_ci95_pp", 0], ["ci95_pp", 0]),
            ("d64 ci-hi", ["dimensionality_sweep", "64", "srp_minus_pca_ci95_pp", 1], ["ci95_pp", 1]),
            ("d16 diff", ["dimensionality_sweep", "16", "srp_minus_pca_pp"], ["dim16_diff_pp"]),
            ("d32 diff", ["dimensionality_sweep", "32", "srp_minus_pca_pp"], ["dim32_diff_pp"]),
            ("seed mean", ["srp64_seed_stability", "mean"], ["srp64_seed_stability_mean"]),
            ("seed std", ["srp64_seed_stability", "std"], ["srp64_seed_stability_std"]),
        ])

    # ---- M2 three-condition (global + blockwise) ---------------------
    d = require_file(os.path.join(_HERE, "nuisance_results.json"))
    v = entry(m, "novel_m2_pca_nuisance_alignment")
    if d is not None:
        comp = d["components"]
        check_pairs("M2/3class global", comp.get("M2_geometry", {}), v, [
            ("angle-subject", ["pca_min_angle_to_subject_deg"], ["pca_min_angle_to_subject_deg"]),
            ("angle-condition", ["pca_min_angle_to_condition_deg"], ["pca_min_angle_to_condition_deg"]),
            ("rho_pca", ["rho_pca_compressed"], ["rho_pca_compressed"]),
            ("rho_srp", ["rho_srp_compressed"], ["rho_srp_compressed"]),
            ("rho_ambient", ["rho_ambient"], ["rho_ambient"]),
        ])
        vb = (v or {}).get("blockwise_n1_operator")
        check_pairs("M2/3class blockwise", comp.get("M2_geometry_blockwise", {}),
                    vb if vb is not None else None, [
            ("angle-subject", ["pca_min_angle_to_subject_deg"], ["pca_min_angle_to_subject_deg"]),
            ("angle-condition", ["pca_min_angle_to_condition_deg"], ["pca_min_angle_to_condition_deg"]),
            ("rho_pca", ["rho_pca_compressed"], ["rho_pca_compressed"]),
            ("total-dims", ["total_compressed_dims"], ["total_compressed_dims"]),
        ])

    # ---- M2 four-category (global + blockwise) -----------------------
    d = require_file(os.path.join(_HERE, "nuisance_results_4class.json"))
    v = entry(m, "novel_m2_pca_nuisance_alignment_4class")
    if d is not None:
        comp = d["components"]
        check_pairs("M2/4class global", comp.get("M2_geometry", {}), v, [
            ("angle-subject", ["pca_min_angle_to_subject_deg"], ["pca_min_angle_to_subject_deg"]),
            ("rho_pca", ["rho_pca_compressed"], ["rho_pca_compressed"]),
        ])
        vb = (v or {}).get("blockwise_n1_operator")
        check_pairs("M2/4class blockwise", comp.get("M2_geometry_blockwise", {}),
                    vb if vb is not None else None, [
            ("angle-subject", ["pca_min_angle_to_subject_deg"], ["pca_min_angle_to_subject_deg"]),
            ("rho_pca", ["rho_pca_compressed"], ["rho_pca_compressed"]),
        ])

    # ---- External TCRZEM (genuine independent cohort) ----------------
    d = require_file(os.path.join(_EXT, "external_results.json"))
    vn = entry(m, "external_n1_srp_vs_pca_tcrzem")
    vg = entry(m, "external_m2_pca_nuisance_alignment_tcrzem")
    if d is not None:
        r = d["results"]
        for dim in ("16", "32", "64"):
            check_pairs(f"EXT/N1 d{dim}", r["N1_srp_vs_pca"].get(dim, {}), vn, [
                ("srp", ["srp_percent"], [f"dim{dim}", "srp"]),
                ("pca", ["foldlocal_pca_percent"], [f"dim{dim}", "pca"]),
                ("diff", ["srp_minus_pca_pp"], [f"dim{dim}", "diff_pp"]),
            ])
        check_pairs("EXT/M2", r.get("M2_geometry", {}), vg, [
            ("angle-subject", ["pca_min_angle_to_subject_deg"], ["pca_min_angle_to_subject_deg"]),
            ("angle-condition", ["pca_min_angle_to_condition_deg"], ["pca_min_angle_to_condition_deg"]),
            ("rho_pca", ["rho_pca_compressed"], ["rho_pca_compressed"]),
            ("rho_srp", ["rho_srp_compressed"], ["rho_srp_compressed"]),
        ])

    print(f"\n{CHECKED} checked, {FAIL} failure(s)")
    if CHECKED == 0:
        print("  [FAIL] zero comparisons performed -- not a pass")
        return 1
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
