#!/usr/bin/env python3
"""Corrected, leakage-safe SHAPE novelty experiments.

This script intentionally does not use ``shape_features_211.pkl`` for headline
results.  The supplied pickle is audited separately because its ``X_ds`` array
is a stride-4 slice of rows 0:1024, whereas the documented specification is to
remove the first 205 baseline rows and antialias-decimate the 1,024-row
poststimulus interval by four.

The primary representation is a fixed 256-unit LIF reservoir followed by six
10-sample early-window spike-count bins.  A data-independent sparse random
projection supplies a common 64-dimensional neuron basis for every electrode;
unlike PCA, it cannot learn from held-out participants.  All classification
scaling and readout fitting occur inside participant-grouped folds.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate, welch
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection


THREE_CLASS = {
    "IAPSNeg": 0,
    "IAPSNeu": 1,
    "IAPSPos": 2,
}
FOUR_CLASS = {
    "IAPSNeg_Mutilation": 0,
    "IAPSNeg_Threat": 1,
    "IAPSPos_Cute": 2,
    "IAPSPos_Erotic": 3,
}
CLASS_NAMES = {
    "three": ["Negative", "Neutral", "Pleasant"],
    "four": ["Mutilation", "Threat", "Cute", "Erotic"],
}


class LIFReservoir:
    """Audited nonnegative reset-by-subtraction reservoir."""

    def __init__(self, n_res: int = 256, beta: float = 0.05,
                 threshold: float = 0.5, seed: int = 42):
        rng = np.random.RandomState(seed)
        lim_in = np.sqrt(6.0 / (1 + n_res))
        self.w_in = rng.uniform(-lim_in, lim_in, n_res)
        lim_rec = np.sqrt(6.0 / (2 * n_res))
        self.w_rec = rng.uniform(-lim_rec, lim_rec, (n_res, n_res))
        eig = np.abs(np.linalg.eigvals(self.w_rec)).max()
        self.w_rec *= 0.9 / eig
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def bsc6(self, x: np.ndarray, start: int = 10,
             stop: int = 70) -> np.ndarray:
        """Six 10-sample spike-count bins from [start, stop)."""
        if stop - start != 60:
            raise ValueError("BSC6 requires a 60-sample window")
        mem = np.zeros(self.n_res, dtype=np.float64)
        prev = np.zeros(self.n_res, dtype=np.float64)
        counts = np.zeros((6, self.n_res), dtype=np.uint8)
        for t in range(stop):
            current = self.w_in * float(x[t]) + self.w_rec @ prev
            mem = (1.0 - self.beta) * mem * (1.0 - prev) + current
            spk = (mem >= self.threshold).astype(np.float64)
            mem = np.maximum(mem - spk * self.threshold, 0.0)
            if t >= start:
                counts[(t - start) // 10] += spk.astype(np.uint8)
            prev = spk
        return counts.reshape(-1)


def parse_files(root: Path, mapping: dict[str, int]):
    out = defaultdict(dict)
    pat = re.compile(r"SHAPE_Community_(\d+)_(.+)_BC\.txt$")
    for path in root.rglob("*.txt"):
        match = pat.match(path.name)
        if not match:
            continue
        subject, condition = int(match.group(1)), match.group(2)
        if condition in mapping:
            out[subject][mapping[condition]] = path
    return out


def corrected_preprocess(path: Path) -> np.ndarray:
    raw = np.loadtxt(path)
    if raw.shape != (1229, 34):
        raise ValueError(f"{path.name}: expected (1229, 34), got {raw.shape}")
    post = raw[205:1229]
    if not np.isfinite(post).all():
        raise ValueError(f"{path.name}: nonfinite values")
    down = np.stack([decimate(post[:, c], 4) for c in range(34)], axis=1)
    down = down[:256]
    sd = down.std(axis=0)
    if np.any(sd < 1e-10):
        raise ValueError(f"{path.name}: constant channel")
    return ((down - down.mean(axis=0)) / sd).astype(np.float32)


def conventional_features(x: np.ndarray, fs: int = 256) -> np.ndarray:
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
    feats = np.zeros((34, 8), dtype=np.float32)
    for c in range(34):
        sig = x[:, c].astype(np.float64)
        f, p = welch(sig, fs=fs, nperseg=128)
        for b, (lo, hi) in enumerate(bands):
            keep = (f >= lo) & (f <= hi)
            feats[c, b] = np.trapezoid(p[keep], f[keep])
        d1, d2 = np.diff(sig), np.diff(sig, n=2)
        activity = np.var(sig)
        mobility = np.sqrt(np.var(d1) / (activity + 1e-12))
        complexity = (np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) /
                      (mobility + 1e-12))
        feats[c, 5:] = activity, mobility, complexity
    return feats.reshape(-1)


def temporal_bin_features(x: np.ndarray, n_bins: int = 16) -> np.ndarray:
    return np.stack(np.split(x, n_bins, axis=0)).mean(axis=1).T.reshape(-1)


def early_six_bin_features(x: np.ndarray) -> np.ndarray:
    return np.stack(np.split(x[10:70], 6, axis=0)).mean(axis=1).T.reshape(-1)


def build_cache(root: Path, mapping: dict[str, int], cache: Path,
                reservoir: LIFReservoir):
    grouped = parse_files(root, mapping)
    required = set(mapping.values())
    audit = {"subjects_discovered": len(grouped), "excluded": []}
    arrays = defaultdict(list)
    labels, subjects = [], []
    start = time.time()
    for ordinal, subject in enumerate(sorted(grouped), start=1):
        if set(grouped[subject]) != required:
            audit["excluded"].append({"subject": subject,
                                      "reason": "incomplete condition panel"})
            continue
        prepared = {}
        try:
            for label in sorted(required):
                prepared[label] = corrected_preprocess(grouped[subject][label])
        except ValueError as exc:
            audit["excluded"].append({"subject": subject, "reason": str(exc)})
            continue
        for label in sorted(required):
            x = prepared[label]
            arrays["x"].append(x)
            arrays["conventional"].append(conventional_features(x))
            arrays["erp16"].append(temporal_bin_features(x, 16))
            arrays["early6"].append(early_six_bin_features(x))
            bsc = np.empty((34, 6 * reservoir.n_res), dtype=np.uint8)
            for channel in range(34):
                bsc[channel] = reservoir.bsc6(x[:, channel])
            arrays["bsc"].append(bsc)
            labels.append(label)
            subjects.append(subject)
        if ordinal % 20 == 0:
            print(f"  {root.name}: examined {ordinal}/{len(grouped)} subjects",
                  flush=True)
    payload = {key: np.stack(value) for key, value in arrays.items()}
    payload["y"] = np.asarray(labels, dtype=np.int8)
    payload["subjects"] = np.asarray(subjects, dtype=np.int16)
    audit["subjects_included"] = int(np.unique(payload["subjects"]).size)
    audit["observations"] = int(payload["y"].size)
    audit["elapsed_seconds"] = time.time() - start
    payload["audit_json"] = np.asarray(json.dumps(audit))
    np.savez_compressed(cache, **payload)
    print(f"  saved {cache} ({audit['subjects_included']} subjects)")
    return payload, audit


def load_or_build(root: Path, mapping: dict[str, int], cache: Path,
                  reservoir: LIFReservoir, rebuild: bool):
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        payload = {key: z[key] for key in z.files}
        return payload, json.loads(str(payload["audit_json"]))
    return build_cache(root, mapping, cache, reservoir)


def fixed_projection_features(bsc: np.ndarray, seed: int = 42):
    n, channels, width = bsc.shape
    pooled = bsc.reshape(n * channels, width).astype(np.float32)
    projector = SparseRandomProjection(
        n_components=64, density=1 / np.sqrt(width), random_state=seed)
    projected = projector.fit_transform(pooled)
    if hasattr(projected, "toarray"):
        projected = projected.toarray()
    ordered = np.asarray(projected, dtype=np.float32).reshape(n, channels, 64)

    collapsed = bsc.reshape(n, channels, 6, 256).sum(axis=2).astype(np.float32)
    collapsed_projector = SparseRandomProjection(
        n_components=64, density=1 / np.sqrt(256), random_state=seed)
    collapsed = collapsed_projector.fit_transform(collapsed.reshape(-1, 256))
    if hasattr(collapsed, "toarray"):
        collapsed = collapsed.toarray()
    collapsed = np.asarray(collapsed, dtype=np.float32).reshape(n, channels, 64)

    rng = np.random.RandomState(seed + 900)
    shuffled_bsc = bsc.reshape(n, channels, 6, 256).copy()
    for i in range(n):
        for c in range(channels):
            shuffled_bsc[i, c] = shuffled_bsc[i, c, rng.permutation(6)]
    shuffled = projector.transform(shuffled_bsc.reshape(-1, width))
    if hasattr(shuffled, "toarray"):
        shuffled = shuffled.toarray()
    shuffled = np.asarray(shuffled, dtype=np.float32).reshape(n, channels, 64)

    channel_shuffled = ordered.copy()
    for i in range(n):
        channel_shuffled[i] = channel_shuffled[i, rng.permutation(channels)]
    return {
        "bsc_srp64": ordered.reshape(n, -1),
        "bsc_time_collapsed_srp64": collapsed.reshape(n, -1),
        "bsc_bin_shuffled_srp64": shuffled.reshape(n, -1),
        "bsc_channel_shuffled_srp64": channel_shuffled.reshape(n, -1),
        "bsc_total_by_bin": bsc.reshape(n, channels, 6, 256).sum(axis=3).reshape(n, -1),
    }


def grouped_splits(groups: np.ndarray, n_splits: int, seed: int):
    unique = np.unique(groups).copy()
    np.random.RandomState(seed).shuffle(unique)
    for held_out in np.array_split(unique, n_splits):
        test = np.isin(groups, held_out)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             repeats: int = 5, folds: int = 5, alpha: float = 1e-3):
    predictions = np.empty((repeats, y.size), dtype=np.int8)
    fold_rows = []
    for repeat in range(repeats):
        for fold, (train, test) in enumerate(
                grouped_splits(groups, folds, 7132026 + repeat)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(X[train])
            test_x = scaler.transform(X[test])
            # SGD optimizes the same multinomial logistic loss as the linear
            # readout used elsewhere in the dissertation, but makes repeated
            # grouped validation and permutation testing computationally
            # tractable.  Alpha is fixed before evaluation, not selected on
            # held-out folds.
            model = SGDClassifier(
                loss="log_loss", alpha=alpha, max_iter=5000, tol=1e-3,
                random_state=42)
            model.fit(train_x, y[train])
            pred = model.predict(test_x)
            predictions[repeat, test] = pred
            fold_rows.append({
                "repeat": repeat,
                "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                "macro_f1": f1_score(y[test], pred, average="macro"),
                "n_test": int(test.size),
            })
    correctness = (predictions == y[None, :]).mean(axis=0)
    subject_scores = np.asarray([
        correctness[groups == subject].mean() for subject in np.unique(groups)
    ])
    rng = np.random.RandomState(20260714)
    draws = rng.choice(subject_scores, (5000, subject_scores.size), replace=True).mean(1)
    return {
        "balanced_accuracy": float(np.mean([
            balanced_accuracy_score(y, row) for row in predictions])),
        "macro_f1": float(np.mean([f1_score(y, row, average="macro")
                                    for row in predictions])),
        "subject_bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "predictions": predictions,
        "fold_rows": fold_rows,
        "subject_scores": subject_scores,
    }


def paired_subject_bootstrap(a, b, groups, y, n_boot: int = 10000):
    unique = np.unique(groups)
    ca = (a["predictions"] == y[None, :]).mean(axis=0)
    cb = (b["predictions"] == y[None, :]).mean(axis=0)
    diff = np.asarray([(ca[groups == s] - cb[groups == s]).mean() for s in unique])
    rng = np.random.RandomState(7132026)
    draws = rng.choice(diff, (n_boot, diff.size), replace=True).mean(1)
    return {
        "mean_accuracy_difference": float(diff.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(2 * min((draws <= 0).mean(),
                                                (draws >= 0).mean())),
    }


def within_subject_permute(y: np.ndarray, groups: np.ndarray, rng):
    out = y.copy()
    for subject in np.unique(groups):
        idx = np.flatnonzero(groups == subject)
        out[idx] = rng.permutation(out[idx])
    return out


def ridge_grouped_score(X, y, groups, folds: int = 5):
    pred = np.empty_like(y)
    for train, test in grouped_splits(groups, folds, 7132026):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(X[train])
        test_x = scaler.transform(X[test])
        model = RidgeClassifier(alpha=10.0)
        model.fit(train_x, y[train])
        pred[test] = model.predict(test_x)
    return float(balanced_accuracy_score(y, pred))


def permutation_test(X, y, groups, observed, n_perm: int = 100):
    # A ridge linear readout is used for both the observed and permuted scores.
    # Its deterministic closed-form optimization avoids convergence differences
    # between real and randomly permuted multiclass labels.
    observed_ridge = ridge_grouped_score(X, y, groups, folds=5)
    rng = np.random.RandomState(14072026)
    null = []
    for permutation in range(n_perm):
        yp = within_subject_permute(y, groups, rng)
        null.append(ridge_grouped_score(X, yp, groups, folds=5))
        if (permutation + 1) % 20 == 0:
            print(f"    permutation {permutation + 1}/{n_perm}", flush=True)
    null = np.asarray(null)
    return {
        "n_permutations": n_perm,
        "primary_sgd_observed": float(observed),
        "ridge_observed": observed_ridge,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_value": float((1 + np.sum(null >= observed_ridge)) / (n_perm + 1)),
        "null_quantiles": np.quantile(null, [0.025, 0.5, 0.975]).tolist(),
    }


def pickle_audit(pickle_path: Path, three_payload):
    with pickle_path.open("rb") as handle:
        legacy = pickle.load(handle)
    subjects = legacy["subjects"]
    y = legacy["y"]
    raw_root_x = three_payload["x"]
    raw_subjects = three_payload["subjects"]
    raw_y = three_payload["y"]
    comparisons = []
    source_root = Path(args.three_root)
    grouped = parse_files(source_root, THREE_CLASS)
    for subject, label in [(6, 0), (6, 1), (6, 2)]:
        path = grouped[subject][label]
        raw = np.loadtxt(path)
        stride = raw[:1024:4]
        stride = (stride - stride.mean(0)) / stride.std(0)
        legacy_idx = np.flatnonzero((subjects == subject) & (y == label))[0]
        corrected_idx = np.flatnonzero(
            (raw_subjects == subject) & (raw_y == label))[0]
        comparisons.append({
            "subject": subject,
            "label": label,
            "legacy_equals_stride_max_abs": float(
                np.max(np.abs(legacy["X_ds"][legacy_idx] - stride))),
            "legacy_vs_corrected_correlation": float(np.corrcoef(
                legacy["X_ds"][legacy_idx].ravel(),
                raw_root_x[corrected_idx].ravel())[0, 1]),
        })
    return {
        "keys": list(legacy),
        "shapes": {key: list(np.shape(value)) for key, value in legacy.items()},
        "subjects": int(np.unique(subjects).size),
        "comparisons": comparisons,
    }


def dataset_experiments(name: str, payload: dict, repeats: int, folds: int,
                        permutations: int):
    y = payload["y"].astype(int)
    groups = payload["subjects"].astype(int)
    projected = fixed_projection_features(payload["bsc"])
    features = {
        "erp_16bin": payload["erp16"],
        "raw_early_6bin": payload["early6"],
        "conventional": payload["conventional"],
        **projected,
    }
    features["bsc_plus_conventional"] = np.concatenate(
        [features["bsc_srp64"], features["conventional"]], axis=1)
    results = {}
    for feature_name, matrix in features.items():
        print(f"  {name}: evaluating {feature_name} {matrix.shape}", flush=True)
        results[feature_name] = evaluate(
            matrix, y, groups, repeats=repeats, folds=folds)
    comparisons = {
        "bsc_vs_conventional": paired_subject_bootstrap(
            results["bsc_srp64"], results["conventional"], groups, y),
        "fusion_vs_bsc": paired_subject_bootstrap(
            results["bsc_plus_conventional"], results["bsc_srp64"], groups, y),
        "ordered_vs_bin_shuffled": paired_subject_bootstrap(
            results["bsc_srp64"], results["bsc_bin_shuffled_srp64"], groups, y),
        "ordered_vs_time_collapsed": paired_subject_bootstrap(
            results["bsc_srp64"], results["bsc_time_collapsed_srp64"], groups, y),
        "ordered_vs_channel_shuffled": paired_subject_bootstrap(
            results["bsc_srp64"], results["bsc_channel_shuffled_srp64"], groups, y),
    }
    primary = results["bsc_srp64"]["balanced_accuracy"]
    perm = permutation_test(features["bsc_srp64"], y, groups, primary,
                            n_perm=permutations)

    # Remove large arrays from JSON while retaining fold-level evidence.
    serializable = {}
    for feature_name, result in results.items():
        serializable[feature_name] = {
            key: value for key, value in result.items()
            if key not in ("predictions", "subject_scores")
        }
    return {
        "n_observations": int(y.size),
        "n_subjects": int(np.unique(groups).size),
        "class_names": CLASS_NAMES[name],
        "features": serializable,
        "comparisons": comparisons,
        "bsc_permutation_test": perm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-root", required=True)
    parser.add_argument("--four-root", required=True)
    parser.add_argument("--pickle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reservoir = LIFReservoir()
    three, three_audit = load_or_build(
        Path(args.three_root), THREE_CLASS, output / "three_corrected_cache.npz",
        reservoir, args.rebuild)
    four, four_audit = load_or_build(
        Path(args.four_root), FOUR_CLASS, output / "four_corrected_cache.npz",
        reservoir, args.rebuild)

    report = {
        "method": {
            "preprocessing": "rows 205:1229, scipy.signal.decimate(q=4), per-epoch/channel z-score",
            "reservoir": "fixed shared 256-unit LIF; beta=.05; threshold=.5; spectral radius=.9",
            "coding": "six 10-sample bins over corrected samples [10,70)",
            "projection": "data-independent SparseRandomProjection 1536->64, shared across channels",
            "validation": f"{args.repeats}x repeated {args.folds}-fold participant-grouped CV",
        },
        "three_audit": three_audit,
        "four_audit": four_audit,
        "legacy_pickle_audit": pickle_audit(Path(args.pickle), three),
        "three": dataset_experiments(
            "three", three, args.repeats, args.folds, args.permutations),
        "four": dataset_experiments(
            "four", four, args.repeats, args.folds, args.permutations),
    }
    with (output / "corrected_novelty_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "three": {k: v["balanced_accuracy"] for k, v in
                  report["three"]["features"].items()},
        "four": {k: v["balanced_accuracy"] for k, v in
                 report["four"]["features"].items()},
    }, indent=2))
