#!/usr/bin/env python3
"""Reservoir-seed and temporal-window robustness on corrected SHAPE data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_corrected_shape_novelty import (  # noqa: E402
    LIFReservoir, evaluate, fixed_projection_features)


WINDOWS = {
    "early_39_273ms": (10, 70),
    "middle_273_508ms": (70, 130),
    "late_508_742ms": (130, 190),
    "latest_742_977ms": (190, 250),
    "full_39_977ms": (10, 250),
}


def multiwindow_bsc(reservoir: LIFReservoir, x: np.ndarray):
    """Run one trajectory and accumulate six equal bins for each window."""
    mem = np.zeros(reservoir.n_res, dtype=np.float64)
    prev = np.zeros(reservoir.n_res, dtype=np.float64)
    counts = {name: np.zeros((6, reservoir.n_res), dtype=np.uint8)
              for name in WINDOWS}
    for t in range(250):
        current = reservoir.w_in * float(x[t]) + reservoir.w_rec @ prev
        mem = (1.0 - reservoir.beta) * mem * (1.0 - prev) + current
        spk = (mem >= reservoir.threshold).astype(np.float64)
        mem = np.maximum(mem - spk * reservoir.threshold, 0.0)
        for name, (start, stop) in WINDOWS.items():
            if start <= t < stop:
                width = (stop - start) // 6
                b = min(5, (t - start) // width)
                counts[name][b] += spk.astype(np.uint8)
        prev = spk
    return {name: value.reshape(-1) for name, value in counts.items()}


def extract_seed_early(x, seed):
    reservoir = LIFReservoir(seed=seed)
    out = np.empty((x.shape[0], 34, 1536), dtype=np.uint8)
    for i in range(x.shape[0]):
        for channel in range(34):
            out[i, channel] = reservoir.bsc6(x[i, :, channel])
        if (i + 1) % 60 == 0:
            print(f"  seed {seed}: {i + 1}/{x.shape[0]}", flush=True)
    return out


def extract_temporal(x, seed=42):
    reservoir = LIFReservoir(seed=seed)
    out = {name: np.empty((x.shape[0], 34, 1536), dtype=np.uint8)
           for name in WINDOWS}
    for i in range(x.shape[0]):
        for channel in range(34):
            values = multiwindow_bsc(reservoir, x[i, :, channel])
            for name in WINDOWS:
                out[name][i, channel] = values[name]
        if (i + 1) % 30 == 0:
            print(f"  temporal: {i + 1}/{x.shape[0]}", flush=True)
    return out


def score_bsc(bsc, y, groups, projection_seed=42):
    x = fixed_projection_features(bsc, seed=projection_seed)["bsc_srp64"]
    result = evaluate(x, y, groups, repeats=3, folds=5)
    return {key: value for key, value in result.items()
            if key not in ("predictions", "subject_scores")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    source = np.load(args.cache, allow_pickle=False)
    x = source["x"]
    y = source["y"].astype(int)
    groups = source["subjects"].astype(int)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed_robustness": {}, "temporal_windows": {}}

    for seed in (7, 42, 99):
        path = output / f"three_bsc_seed{seed}.npz"
        if seed == 42:
            bsc = source["bsc"]
        elif path.exists() and not args.rebuild:
            bsc = np.load(path, allow_pickle=False)["bsc"]
        else:
            start = time.time()
            bsc = extract_seed_early(x, seed)
            np.savez_compressed(path, bsc=bsc)
            print(f"  seed {seed} extraction: {time.time() - start:.1f}s")
        report["seed_robustness"][str(seed)] = score_bsc(
            bsc, y, groups, projection_seed=42)

    temporal_path = output / "three_bsc_temporal_seed42.npz"
    if temporal_path.exists() and not args.rebuild:
        z = np.load(temporal_path, allow_pickle=False)
        temporal = {name: z[name] for name in WINDOWS}
    else:
        start = time.time()
        temporal = extract_temporal(x, seed=42)
        np.savez_compressed(temporal_path, **temporal)
        print(f"  temporal extraction: {time.time() - start:.1f}s")
    for name, bsc in temporal.items():
        print(f"  evaluating {name}", flush=True)
        report["temporal_windows"][name] = score_bsc(
            bsc, y, groups, projection_seed=42)

    with (output / "reservoir_robustness.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "seeds": {key: value["balanced_accuracy"]
                  for key, value in report["seed_robustness"].items()},
        "windows": {key: value["balanced_accuracy"]
                    for key, value in report["temporal_windows"].items()},
    }, indent=2))
