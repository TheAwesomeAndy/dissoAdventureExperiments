#!/usr/bin/env python3
"""
Consistency checker: committed novel/nuisance/external result JSONs vs the
exploratory entries in results_manifest.json.

The confirmatory checker only covers confirmatory rows; this closes the gap the
audit identified -- a green CI run now also establishes that the committed
exploratory result files and their manifest entries agree. It compares the
headline numbers that appear in both places (accuracies, differences, principal
angles, rho), so a stale JSON or a hand-edited manifest number is caught.

Run with no argument for the default repo paths (CI-safe: needs only the
committed JSONs, no restricted data).
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MANIFEST = os.path.join(_REPO, "results_manifest.json")

TOL = 0.05  # percentage-point / degree tolerance for identical committed values
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


def cmp(label, got, want, tol=TOL):
    global FAIL, CHECKED
    if got is None or want is None:
        FAIL += 1
        print(f"  [MISSING] {label}: got={got} want={want}")
        return
    if abs(float(got) - float(want)) > tol:
        FAIL += 1
        print(f"  [MISMATCH] {label}: json={got} manifest={want}")
    else:
        CHECKED += 1
        print(f"  [ok] {label}: {got} ~ {want}")


def opt(path):
    return path if os.path.exists(path) else None


def main():
    m = load(_MANIFEST)
    # --- N1 three-condition (novel_results_3class.json) ---
    p = opt(os.path.join(_HERE, "novel_results_3class.json"))
    if p:
        d = load(p)["results"]["N1_leak_free_compression"]["dimensionality_sweep"]["64"]
        v = entry(m, "novel_n1_srp_vs_pca")
        cmp("N1/3class srp", d["srp_percent"], v["srp_percent"])
        cmp("N1/3class pca", d["foldlocal_pca_percent"], v["foldlocal_pca_percent"])
        cmp("N1/3class diff", d["srp_minus_pca_pp"], v["srp_minus_pca_pp"])

    # --- N1 four-category (novel_results_4class.json) ---
    p = opt(os.path.join(_HERE, "novel_results_4class.json"))
    if p:
        d = load(p)["results"]["N1_leak_free_compression"]["dimensionality_sweep"]["64"]
        v = entry(m, "novel_n1_srp_vs_pca_4class")
        cmp("N1/4class srp", d["srp_percent"], v["srp_percent"])
        cmp("N1/4class pca", d["foldlocal_pca_percent"], v["foldlocal_pca_percent"])

    # --- M2 three-condition (nuisance_results.json) ---
    p = opt(os.path.join(_HERE, "nuisance_results.json"))
    if p:
        g = load(p)["components"]["M2_geometry"]
        v = entry(m, "novel_m2_pca_nuisance_alignment")
        cmp("M2/3class angle-subject", g["pca_min_angle_to_subject_deg"],
            v["pca_min_angle_to_subject_deg"])
        cmp("M2/3class rho_pca", g["rho_pca_compressed"], v["rho_pca_compressed"])

    # --- M2 four-category (nuisance_results_4class.json) ---
    p = opt(os.path.join(_HERE, "nuisance_results_4class.json"))
    if p:
        g = load(p)["components"]["M2_geometry"]
        v = entry(m, "novel_m2_pca_nuisance_alignment_4class")
        if v:
            cmp("M2/4class angle-subject", g["pca_min_angle_to_subject_deg"],
                v["pca_min_angle_to_subject_deg"])

    # --- External (experiments/external/external_results.json) ---
    p = opt(os.path.join(_REPO, "experiments", "external", "external_results.json"))
    if p:
        r = load(p)["results"]
        v = entry(m, "external_n1_srp_vs_pca_tcrzem")
        if v:
            cmp("EXT/N1 srp d64", r["N1_srp_vs_pca"]["64"]["srp_percent"],
                v["dim64"]["srp"])
        vg = entry(m, "external_m2_pca_nuisance_alignment_tcrzem")
        if vg:
            cmp("EXT/M2 angle-subject", r["M2_geometry"]["pca_min_angle_to_subject_deg"],
                vg["pca_min_angle_to_subject_deg"])

    print(f"\n{CHECKED} checked, {FAIL} mismatch(es)")
    if CHECKED == 0 and FAIL == 0:
        print("  (no committed result JSONs found -- nothing to compare)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
