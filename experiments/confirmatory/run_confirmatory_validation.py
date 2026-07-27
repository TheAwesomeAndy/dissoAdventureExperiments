#!/usr/bin/env python3
"""
ARSPI-Net confirmatory validation runner.

The runner reconstructs the corrected three-condition and four-category
confirmatory streams from authorized SHAPE text files. It implements the
preprocessing stated in the corrected dissertation: remove the first 205
prestimulus samples, anti-alias decimate the remaining 1024 samples by four,
retain 256 samples, and z-score each epoch and channel. Subject 127 is excluded
from both granularities; Subject 195 is additionally excluded from the
four-category stream.

The restricted SHAPE data are not distributed with this repository. Synthetic
verification of the machinery is provided by verify_confirmatory.py.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.signal import butter, decimate, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import confirmatory_pipeline as cp

PRESTIMULUS_SAMPLES = 205
DOWNSAMPLE_FACTOR = 4
TARGET_TIMESTEPS = 256
EXPECTED_CHANNELS = 34

_COND3 = ("Neg", "Neu", "Pos")
_COND3_MAP = {"Neg": 0, "Neu": 1, "Pos": 2}
_CAT4 = {"Threat": 0, "Mutilation": 1, "Cute": 2, "Erotic": 3}


def preprocess_epoch(
    raw,
    prestimulus_samples=PRESTIMULUS_SAMPLES,
    downsample_factor=DOWNSAMPLE_FACTOR,
    target_timesteps=TARGET_TIMESTEPS,
):
    """Apply the audited SHAPE epoch preprocessing exactly."""
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("raw epoch must be a two-dimensional array")
    if x.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"expected {EXPECTED_CHANNELS} channels, received {x.shape[1]}"
        )
    required = prestimulus_samples + downsample_factor * target_timesteps
    if x.shape[0] < required:
        raise ValueError(
            f"epoch has {x.shape[0]} samples; at least {required} are required"
        )

    poststimulus = x[prestimulus_samples:required, :]
    downsampled = np.column_stack(
        [
            decimate(
                poststimulus[:, ch],
                downsample_factor,
                zero_phase=True,
            )
            for ch in range(x.shape[1])
        ]
    )
    if downsampled.shape != (target_timesteps, EXPECTED_CHANNELS):
        raise RuntimeError(
            "audited preprocessing produced unexpected shape "
            f"{downsampled.shape}"
        )

    mean = downsampled.mean(axis=0, keepdims=True)
    std = downsampled.std(axis=0, keepdims=True)
    if np.any(std <= 0):
        bad = np.flatnonzero(std.ravel() <= 0).tolist()
        raise ValueError(f"constant channel(s) after preprocessing: {bad}")
    normalized = (downsampled - mean) / std
    if not np.all(np.isfinite(normalized)):
        raise ValueError("preprocessed epoch contains non-finite values")
    return normalized


def _complete_panels(found, labels, excluded, expected_participants, name):
    subjects = sorted({sid for sid, label in found if sid not in excluded})
    incomplete = {
        sid: [label for label in labels if (sid, label) not in found]
        for sid in subjects
    }
    incomplete = {sid: missing for sid, missing in incomplete.items() if missing}
    if incomplete:
        preview = list(incomplete.items())[:10]
        raise ValueError(f"{name} contains incomplete participant panels: {preview}")
    if expected_participants is not None and len(subjects) != expected_participants:
        raise ValueError(
            f"{name} expected {expected_participants} participants after exclusions, "
            f"found {len(subjects)}"
        )
    return subjects


def load_3class(data_dir, validate_cohort=True):
    """Load the corrected three-condition cohort from SHAPE text files."""
    pattern = re.compile(
        r"SHAPE_Community_(\d+)_IAPS(Neg|Neu|Pos)_BC\.txt$"
    )
    found = {}
    for root, _dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            match = pattern.search(filename)
            if not match:
                continue
            sid, condition = int(match.group(1)), match.group(2)
            key = (sid, condition)
            if key in found:
                raise ValueError(f"duplicate three-condition file for {key}")
            found[key] = os.path.join(root, filename)

    excluded = {127}
    subjects = _complete_panels(
        found,
        _COND3,
        excluded,
        211 if validate_cohort else None,
        "three-condition cohort",
    )
    raw_list, labels, groups = [], [], []
    for sid in subjects:
        for condition in _COND3:
            raw_list.append(preprocess_epoch(np.loadtxt(found[(sid, condition)])))
            labels.append(_COND3_MAP[condition])
            groups.append(sid)

    raw = np.asarray(raw_list)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)
    if validate_cohort and (raw.shape[0] != 633 or len(np.unique(groups)) != 211):
        raise RuntimeError("three-condition cohort does not equal 211/633")
    return raw, y, groups


def load_4class(data_dir, validate_cohort=True):
    """Load the corrected four-category cohort from SHAPE text files."""
    pattern = re.compile(
        r"SHAPE_Community_(\d+)_IAPS(?:Neg|Pos)_"
        r"(Threat|Mutilation|Cute|Erotic)_BC\.txt$"
    )
    found = {}
    for root, _dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            match = pattern.search(filename)
            if not match:
                continue
            sid, category = int(match.group(1)), match.group(2)
            key = (sid, category)
            if key in found:
                raise ValueError(f"duplicate four-category file for {key}")
            found[key] = os.path.join(root, filename)

    excluded = {127, 195}
    categories = tuple(_CAT4.keys())
    subjects = _complete_panels(
        found,
        categories,
        excluded,
        210 if validate_cohort else None,
        "four-category cohort",
    )
    raw_list, labels, groups = [], [], []
    for sid in subjects:
        for category in categories:
            raw_list.append(preprocess_epoch(np.loadtxt(found[(sid, category)])))
            labels.append(_CAT4[category])
            groups.append(sid)

    raw = np.asarray(raw_list)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)
    if validate_cohort and (raw.shape[0] != 840 or len(np.unique(groups)) != 210):
        raise RuntimeError("four-category cohort does not equal 210/840")
    return raw, y, groups


def raw_erp_bins(raw_data, n_bins=16):
    """Mean amplitude in temporal bins for every channel."""
    raw = np.asarray(raw_data)
    n_obs, time_steps, n_ch = raw.shape
    edges = np.linspace(0, time_steps, n_bins + 1, dtype=int)
    features = np.empty((n_obs, n_ch * n_bins), dtype=np.float64)
    column = 0
    for ch in range(n_ch):
        for bin_index in range(n_bins):
            features[:, column] = raw[
                :, edges[bin_index] : edges[bin_index + 1], ch
            ].mean(axis=1)
            column += 1
    return features


def conventional_features(raw_data, fs=256.0):
    """Five-band power plus Hjorth activity, mobility, and complexity."""
    raw = np.asarray(raw_data)
    bands = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
    nyquist = fs / 2.0
    filters = [butter(3, [lo / nyquist, hi / nyquist], btype="band") for lo, hi in bands]
    n_obs, _, n_ch = raw.shape
    features = np.empty((n_obs, n_ch * (len(bands) + 3)), dtype=np.float64)
    column = 0
    for ch in range(n_ch):
        channel = raw[:, :, ch]
        for b, a in filters:
            filtered = filtfilt(b, a, channel, axis=1)
            features[:, column] = np.mean(filtered**2, axis=1)
            column += 1
        first = np.diff(channel, axis=1)
        second = np.diff(first, axis=1)
        var0 = np.var(channel, axis=1) + 1e-12
        var1 = np.var(first, axis=1) + 1e-12
        var2 = np.var(second, axis=1) + 1e-12
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
        features[:, column] = var0
        features[:, column + 1] = mobility
        features[:, column + 2] = complexity
        column += 3
    return features


def percent(value):
    return round(float(value) * 100.0, 4)


def _representation_record(result):
    ci = cp.participant_bootstrap_ci(
        result["oof_true"],
        result["oof_pred"],
        result["oof_group"],
    )
    return {
        "balanced_accuracy_percent": percent(result["balanced_accuracy"]),
        "ci95_percent": [percent(ci[0]), percent(ci[1])],
        "per_repeat_percent": [percent(value) for value in result["per_repeat"]],
    }


def _paired_difference_record(result_a, result_b, seed):
    ci = cp.participant_bootstrap_difference_ci(
        result_a["oof_true"],
        result_a["oof_pred"],
        result_b["oof_pred"],
        result_a["oof_group"],
        seed=seed,
    )
    point = result_a["balanced_accuracy"] - result_b["balanced_accuracy"]
    return {
        "difference_pp": percent(point),
        "ci95_pp": [percent(ci[0]), percent(ci[1])],
    }


def run_granularity(name, raw_data, y, groups, base_seed):
    """Run all confirmatory representations and controls for one granularity."""
    print(
        f"\n=== {name}: {len(y)} observations, "
        f"{len(np.unique(groups))} participants ==="
    )
    srp = cp.make_srp(cp.N_BINS * cp.N_RES)
    bsc = cp.build_bsc_features(raw_data, reservoir_seed=42)
    projected = cp.project_bsc(bsc, srp)
    raw_reference = raw_erp_bins(raw_data, n_bins=16)
    conventional = conventional_features(raw_data)
    fusion = np.concatenate([projected, conventional], axis=1)

    matrices = {
        "raw_erp16": raw_reference,
        "conventional": conventional,
        "bsc6_srp64": projected,
        "bsc6_srp64_plus_conv": fusion,
    }
    evaluations = {
        key: cp.evaluate_features(X, y, groups, base_seed=base_seed)
        for key, X in matrices.items()
    }
    representations = {
        key: _representation_record(result)
        for key, result in evaluations.items()
    }
    for key, record in representations.items():
        print(
            f"  {key:24s} {record['balanced_accuracy_percent']:.1f}% "
            f"{record['ci95_percent']}"
        )

    contrasts = {
        "bsc6_minus_conventional": _paired_difference_record(
            evaluations["bsc6_srp64"], evaluations["conventional"], base_seed + 201
        ),
        "fusion_minus_bsc6": _paired_difference_record(
            evaluations["bsc6_srp64_plus_conv"],
            evaluations["bsc6_srp64"],
            base_seed + 202,
        ),
        "raw_erp16_minus_bsc6": _paired_difference_record(
            evaluations["raw_erp16"], evaluations["bsc6_srp64"], base_seed + 203
        ),
    }

    permutation = cp.permutation_test(projected, y, groups, seed=base_seed)

    rng = np.random.RandomState(base_seed + 101)
    control_features = {
        "temporal_bin_shuffle": cp.destroy_temporal_bin_order(bsc, rng),
        "temporal_collapse": cp.destroy_temporal_collapse(bsc),
        "electrode_shuffle": cp.destroy_electrode_identity(bsc, rng),
    }
    controls = {}
    intact = evaluations["bsc6_srp64"]
    for control_name, destroyed_bsc in control_features.items():
        destroyed = cp.evaluate_features(
            cp.project_bsc(destroyed_bsc, srp),
            y,
            groups,
            base_seed=base_seed,
        )
        controls[control_name] = _paired_difference_record(
            destroyed,
            intact,
            base_seed + 300 + len(controls),
        )
        print(
            f"  control {control_name:22s} "
            f"{controls[control_name]['difference_pp']:+.1f} pp"
        )

    seed_accuracy = {}
    for reservoir_seed in cp.RESERVOIR_SEEDS:
        seeded_bsc = cp.build_bsc_features(raw_data, reservoir_seed=reservoir_seed)
        result = cp.evaluate_features(
            cp.project_bsc(seeded_bsc, srp),
            y,
            groups,
            base_seed=base_seed,
        )
        seed_accuracy[str(reservoir_seed)] = percent(result["balanced_accuracy"])

    temporal_windows = {}
    for window_name, (lo, hi) in cp.TEMPORAL_WINDOWS.items():
        windowed = (
            raw_data
            if (lo, hi) == (0.0, 1.0)
            else cp.window_slice(raw_data, lo, hi)
        )
        window_bsc = cp.build_bsc_features(windowed, reservoir_seed=42)
        result = cp.evaluate_features(
            cp.project_bsc(window_bsc, srp),
            y,
            groups,
            base_seed=base_seed,
        )
        temporal_windows[window_name] = percent(result["balanced_accuracy"])

    return {
        "n_participants": int(len(np.unique(groups))),
        "n_observations": int(len(y)),
        "representations": representations,
        "contrasts": contrasts,
        "permutation": {
            "observed_balanced_accuracy_percent": percent(
                permutation["observed_balanced_accuracy"]
            ),
            "p_value": permutation["p_value"],
            "n_permutations": permutation["n_permutations"],
        },
        "destruction_controls": controls,
        "seed_robustness_percent": seed_accuracy,
        "temporal_windows_percent": temporal_windows,
    }


def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unavailable"


def _git(args):
    try:
        return subprocess.run(
            ["git", "-C", _REPO, *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _pkg_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - version probing must never abort a run
        return "unknown"


def _count_files(path, suffix=".txt"):
    if not path or not os.path.isdir(path):
        return None
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += sum(1 for f in files if f.endswith(suffix))
    return n


def build_provenance(args, results, cohorts, started_iso):
    """Automatic execution-provenance block: distinguishes a reproducible run
    from an unexplained output file. Emitted by the runner, not hand-entered.

    `results_sha256` hashes the scientific payload (protocol + granularities)
    with sorted keys, so it excludes the provenance block itself and is stable
    across reruns of identical code and data.
    """
    dirty = _git(["status", "--porcelain"])
    payload = json.dumps(
        {"protocol": results["protocol"], "granularities": results["granularities"]},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return {
        "generated_by": "experiments/confirmatory/run_confirmatory_validation.py",
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_working_tree": "dirty" if dirty and dirty != "unknown" else (
            "unknown" if dirty == "unknown" else "clean"),
        "command_line": "run_confirmatory_validation.py "
                        f"--data3 {args.data3} --data4 {args.data4}"
                        + (" --skip4" if args.skip4 else ""),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": {
            "numpy": _pkg_version("numpy"),
            "scipy": _pkg_version("scipy"),
            "sklearn": _pkg_version("sklearn"),
        },
        "code_sha256": {
            "run_confirmatory_validation.py":
                _sha256_file(os.path.join(_HERE, "run_confirmatory_validation.py")),
            "confirmatory_pipeline.py":
                _sha256_file(os.path.join(_HERE, "confirmatory_pipeline.py")),
        },
        "input_file_counts": {
            "three_condition": _count_files(args.data3),
            "four_category": None if args.skip4 else _count_files(args.data4),
        },
        "cohorts": cohorts,
        "seeds": {
            "srp_seed": cp.SRP_SEED,
            "reservoir_seeds": list(cp.RESERVOIR_SEEDS),
            "base_seed_three_condition": 0,
            "base_seed_four_category": 100,
        },
        "started_utc": started_iso,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_sha256": hashlib.sha256(payload).hexdigest(),
    }


def main():
    started_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data3", default="./batch_data")
    parser.add_argument("--data4", default="./categories")
    parser.add_argument("--out", default="confirmatory_results.json")
    parser.add_argument("--skip4", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.data3):
        parser.error(
            f"three-condition data directory not found: {args.data3}; "
            "the restricted SHAPE data are not shipped with the repository"
        )
    if not args.skip4 and not os.path.isdir(args.data4):
        parser.error(
            f"four-category data directory not found: {args.data4}; "
            "use --skip4 to run the three-condition stream alone"
        )

    results = {
        "schema_version": "2.0.0",
        "protocol": {
            "cv": "5 repeats x 5-fold participant-grouped StratifiedGroupKFold",
            "preprocessing": (
                "remove samples 0-204; anti-alias decimate samples 205-1228 "
                "by 4 to 256 samples; per-epoch/channel z-score; fold-local "
                "StandardScaler on training participants"
            ),
            "compression": (
                f"fixed data-independent SparseRandomProjection "
                f"{cp.N_BINS * cp.N_RES}->{cp.N_SRP}, seed={cp.SRP_SEED}, "
                "shared across electrodes"
            ),
            "readout": "LogisticRegression; RidgeClassifier for permutation",
            "reservoir": (
                f"N={cp.N_RES}, beta=0.05, threshold=0.5, "
                f"seeds={cp.RESERVOIR_SEEDS}"
            ),
            "intervals": (
                "participant-cluster bootstrap over predictions pooled from all "
                "five CV repeats; paired bootstrap for contrasts"
            ),
            "exclusions": {
                "three_condition": [127],
                "four_category": [127, 195],
            },
        },
        "granularities": {},
    }

    cohorts = {}
    raw3, y3, groups3 = load_3class(args.data3)
    cohorts["three_condition"] = {
        "participants": int(len(np.unique(groups3))),
        "observations": int(len(y3)),
    }
    results["granularities"]["three_condition"] = run_granularity(
        "three_condition", raw3, y3, groups3, base_seed=0
    )

    if not args.skip4:
        raw4, y4, groups4 = load_4class(args.data4)
        cohorts["four_category"] = {
            "participants": int(len(np.unique(groups4))),
            "observations": int(len(y4)),
        }
        results["granularities"]["four_category"] = run_granularity(
            "four_category", raw4, y4, groups4, base_seed=100
        )

    results["provenance"] = build_provenance(args, results, cohorts, started_iso)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(f"\nWrote {args.out}")
    print(f"results_sha256={results['provenance']['results_sha256']}")


if __name__ == "__main__":
    main()
