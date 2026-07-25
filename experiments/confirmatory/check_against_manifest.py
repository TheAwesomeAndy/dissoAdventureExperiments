#!/usr/bin/env python3
"""
Compare a produced confirmatory_results.json against results_manifest.json.
=========================================================================

After running run_confirmatory_validation.py on the authorized SHAPE data,
this checker confirms the regenerated numbers agree with the values recorded
in the repository manifest, within tolerance. It is the automated link between
"documented number" and "number a committed script actually produced".

    python check_against_manifest.py confirmatory_results.json \
        --manifest ../../results_manifest.json --tol 1.5

Without a results file it runs in --schema-only mode, validating that the
manifest's confirmatory entries carry full provenance and now point at the
committed runner (no residual "script": null among confirmatory results).
Schema-only mode requires no SHAPE data and runs in CI.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MANIFEST = os.path.abspath(os.path.join(_HERE, "..", "..",
                                                 "results_manifest.json"))

# manifest id -> (granularity, representation) for the headline accuracies
_ACC_MAP = {
    "conf_3class_srp64": ("three_condition", "bsc6_srp64"),
    "conf_4class_srp64": ("four_category", "bsc6_srp64"),
    "conf_3class_rawerp16": ("three_condition", "raw_erp16"),
    "conf_4class_rawerp16": ("four_category", "raw_erp16"),
    "conf_3class_conventional": ("three_condition", "conventional"),
    "conf_4class_conventional": ("four_category", "conventional"),
    "conf_3class_fusion": ("three_condition", "bsc6_srp64_plus_conv"),
    "conf_4class_fusion": ("four_category", "bsc6_srp64_plus_conv"),
}

FAIL = 0


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [MISMATCH] {msg}")


def load(path):
    with open(path) as f:
        return json.load(f)


def schema_only(manifest):
    print("[schema-only] validating manifest confirmatory provenance")
    required = {"data_source", "sample_count", "split_strategy",
                "preprocessing_scope", "estimator", "seed_set",
                "output_table", "script", "tier"}
    conf = [r for r in manifest["results"] if r.get("tier") == "confirmatory"]
    print(f"  {len(conf)} confirmatory entries")
    for r in conf:
        missing = required - set(r.keys())
        if missing:
            fail(f"{r['id']} missing fields: {sorted(missing)}")
        if r.get("script") in (None, "", "null"):
            fail(f"{r['id']} still has no generating script (script={r.get('script')!r})")
    return FAIL == 0


def compare(results, manifest, tol):
    print(f"[compare] tolerance = {tol} percentage points")
    by_id = {r["id"]: r for r in manifest["results"]}
    grans = results.get("granularities", {})
    for mid, (gran, rep) in _ACC_MAP.items():
        if mid not in by_id:
            continue
        want = by_id[mid]["value"].get("balanced_accuracy")
        if want is None:
            continue
        want_pct = want * 100 if want <= 1.0 else want
        got = grans.get(gran, {}).get("representations", {}).get(rep, {}).get(
            "balanced_accuracy")
        if got is None:
            fail(f"{mid}: no produced value for {gran}/{rep}")
            continue
        if abs(got - want_pct) > tol:
            fail(f"{mid}: produced {got:.1f}% vs manifest {want_pct:.1f}% "
                 f"(> {tol} pp)")
        else:
            print(f"  [ok] {mid}: {got:.1f}% ~ {want_pct:.1f}%")
    return FAIL == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", help="confirmatory_results.json")
    ap.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    ap.add_argument("--tol", type=float, default=1.5,
                    help="allowed |diff| in percentage points")
    args = ap.parse_args()

    manifest = load(args.manifest)
    if not args.results:
        ok = schema_only(manifest)
    else:
        ok = compare(load(args.results), manifest, args.tol)

    print("OK" if ok else f"FAILED ({FAIL} mismatch(es))")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
