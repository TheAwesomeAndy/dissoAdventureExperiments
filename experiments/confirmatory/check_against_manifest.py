#!/usr/bin/env python3
"""Validate confirmatory provenance and compare generated outputs to the manifest.

Without a generated result file, the script performs a data-independent schema
check suitable for CI. With confirmatory_results.json, it compares every
confirmatory manifest estimand represented in results_manifest.json: cohort
sizes, representation accuracies and intervals, the reservoir-minus-
conventional contrast and interval, permutation results, and destruction
controls.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MANIFEST = os.path.abspath(
    os.path.join(_HERE, "..", "..", "results_manifest.json")
)

_ACCURACY_MAP = {
    "conf_3class_srp64": ("three_condition", "bsc6_srp64"),
    "conf_4class_srp64": ("four_category", "bsc6_srp64"),
    "conf_3class_rawerp16": ("three_condition", "raw_erp16"),
    "conf_4class_rawerp16": ("four_category", "raw_erp16"),
    "conf_3class_conventional": ("three_condition", "conventional"),
    "conf_4class_conventional": ("four_category", "conventional"),
    "conf_3class_fusion": ("three_condition", "bsc6_srp64_plus_conv"),
    "conf_4class_fusion": ("four_category", "bsc6_srp64_plus_conv"),
}
_SUPPORTED_CONFIRMATORY_IDS = set(_ACCURACY_MAP) | {
    "conf_advantage_over_conventional",
    "conf_permutation_test",
    "conf_destruction_controls",
}

FAIL = 0
CHECKED = 0


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fail(message):
    global FAIL
    FAIL += 1
    print(f"  [MISMATCH] {message}")


def ok(message):
    global CHECKED
    CHECKED += 1
    print(f"  [ok] {message}")


def as_percent(value):
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def compare_scalar(label, produced, expected, tolerance):
    if produced is None:
        fail(f"{label}: generated value is absent")
        return
    produced = float(produced)
    expected = float(expected)
    if abs(produced - expected) > tolerance:
        fail(
            f"{label}: generated {produced:.4f}, expected {expected:.4f}, "
            f"tolerance {tolerance:.4f}"
        )
    else:
        ok(f"{label}: {produced:.4f} ~ {expected:.4f}")


def compare_interval(label, produced, expected, tolerance):
    if produced is None or len(produced) != 2:
        fail(f"{label}: generated interval is absent or malformed")
        return
    if expected is None or len(expected) != 2:
        fail(f"{label}: manifest interval is absent or malformed")
        return
    for bound, got, want in zip(("lower", "upper"), produced, expected):
        compare_scalar(f"{label} {bound}", got, want, tolerance)


def schema_only(manifest):
    print("[schema-only] validating confirmatory provenance")
    required = {
        "id",
        "tier",
        "data_source",
        "sample_count",
        "split_strategy",
        "preprocessing_scope",
        "estimator",
        "seed_set",
        "output_table",
        "script",
        "value",
    }
    confirmatory = [
        result for result in manifest.get("results", [])
        if result.get("tier") == "confirmatory"
    ]
    print(f"  {len(confirmatory)} confirmatory entries")
    ids = {result.get("id") for result in confirmatory}
    unsupported = ids - _SUPPORTED_CONFIRMATORY_IDS
    missing = _SUPPORTED_CONFIRMATORY_IDS - ids
    if unsupported:
        fail(f"unrecognized confirmatory result ids: {sorted(unsupported)}")
    if missing:
        fail(f"expected confirmatory result ids absent: {sorted(missing)}")

    for result in confirmatory:
        absent = required - set(result)
        if absent:
            fail(f"{result.get('id')} missing fields: {sorted(absent)}")
        script = result.get("script")
        if not script:
            fail(f"{result.get('id')} has no generating script")
        elif script != "experiments/confirmatory/run_confirmatory_validation.py":
            fail(f"{result.get('id')} points to unexpected script: {script}")
        else:
            ok(f"{result.get('id')} has canonical generating script")
    return FAIL == 0


def compare_results(generated, manifest, tolerance_pp):
    by_id = {result["id"]: result for result in manifest["results"]}
    granularities = generated.get("granularities", {})

    dataset = manifest["dataset"]["cohorts"]
    for granularity in ("three_condition", "four_category"):
        produced = granularities.get(granularity, {})
        expected = dataset[granularity]
        compare_scalar(
            f"{granularity} participant count",
            produced.get("n_participants"),
            expected["participants"],
            0.0,
        )
        compare_scalar(
            f"{granularity} observation count",
            produced.get("n_observations"),
            expected["observations"],
            0.0,
        )

    for manifest_id, (granularity, representation) in _ACCURACY_MAP.items():
        manifest_result = by_id[manifest_id]
        expected_value = manifest_result["value"]
        produced_record = (
            granularities.get(granularity, {})
            .get("representations", {})
            .get(representation, {})
        )
        compare_scalar(
            f"{manifest_id} balanced accuracy (%)",
            produced_record.get("balanced_accuracy_percent"),
            as_percent(expected_value["balanced_accuracy"]),
            tolerance_pp,
        )
        if "ci95" in expected_value:
            compare_interval(
                f"{manifest_id} 95% interval (%)",
                produced_record.get("ci95_percent"),
                [as_percent(value) for value in expected_value["ci95"]],
                tolerance_pp,
            )

    advantage = by_id["conf_advantage_over_conventional"]["value"]
    for granularity, prefix in (
        ("three_condition", "three_condition"),
        ("four_category", "four_category"),
    ):
        produced = (
            granularities[granularity]["contrasts"]["bsc6_minus_conventional"]
        )
        compare_scalar(
            f"{granularity} BSC6 minus conventional (pp)",
            produced.get("difference_pp"),
            advantage[f"{prefix}_pp"],
            tolerance_pp,
        )
        compare_interval(
            f"{granularity} BSC6 minus conventional interval (pp)",
            produced.get("ci95_pp"),
            advantage[f"{prefix}_ci95"],
            tolerance_pp,
        )

    permutation = by_id["conf_permutation_test"]["value"]
    for granularity in ("three_condition", "four_category"):
        produced = granularities[granularity].get("permutation", {})
        compare_scalar(
            f"{granularity} permutation count",
            produced.get("n_permutations"),
            permutation["n_permutations"],
            0.0,
        )
        compare_scalar(
            f"{granularity} permutation p-value",
            produced.get("p_value"),
            permutation["p_value"],
            1e-4,
        )

    controls = by_id["conf_destruction_controls"]["value"]
    for control_name, manifest_control in controls.items():
        for granularity, prefix in (
            ("three_condition", "three_condition"),
            ("four_category", "four_category"),
        ):
            produced = (
                granularities[granularity]
                .get("destruction_controls", {})
                .get(control_name, {})
            )
            compare_scalar(
                f"{granularity} {control_name} drop (pp)",
                produced.get("difference_pp"),
                manifest_control[f"{prefix}_pp"],
                tolerance_pp,
            )
            interval_key = f"{prefix}_ci95"
            if interval_key in manifest_control:
                compare_interval(
                    f"{granularity} {control_name} interval (pp)",
                    produced.get("ci95_pp"),
                    manifest_control[interval_key],
                    tolerance_pp,
                )

    return FAIL == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help="confirmatory_results.json")
    parser.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    parser.add_argument(
        "--tol",
        type=float,
        default=1.5,
        help="absolute tolerance in percentage points",
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    if args.results:
        print(f"[compare] tolerance = {args.tol} percentage points")
        success = compare_results(load_json(args.results), manifest, args.tol)
    else:
        success = schema_only(manifest)

    print(f"Checked {CHECKED} quantities")
    print("OK" if success else f"FAILED ({FAIL} mismatch(es))")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
