#!/usr/bin/env python3
"""Create the corrected SHAPE validation figure used in Chapter 5."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--robustness", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = json.load(open(args.results))
    robustness = json.load(open(args.robustness))

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 180,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    colors = {"three": "#254F73", "four": "#C05A47"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), constrained_layout=True)

    # A: primary baselines.
    labels = ["ERP\n16-bin", "Conventional", "BSC$_6$\nSRP-64", "BSC +\nconventional"]
    keys = ["erp_16bin", "conventional", "bsc_srp64", "bsc_plus_conventional"]
    x = np.arange(len(keys)); width = 0.36
    for offset, dataset in [(-width / 2, "three"), (width / 2, "four")]:
        vals, lo, hi = [], [], []
        for key in keys:
            row = results[dataset]["features"][key]
            vals.append(100 * row["balanced_accuracy"])
            lo.append(100 * (row["balanced_accuracy"] - row["subject_bootstrap_ci95"][0]))
            hi.append(100 * (row["subject_bootstrap_ci95"][1] - row["balanced_accuracy"]))
        axes[0].bar(x + offset, vals, width, color=colors[dataset],
                    label="Three conditions" if dataset == "three" else "Four categories")
        axes[0].errorbar(x + offset, vals, yerr=[lo, hi], fmt="none",
                         ecolor="black", elinewidth=.8, capsize=2)
    axes[0].axhline(33.33, color=colors["three"], linestyle=":", linewidth=1)
    axes[0].axhline(25, color=colors["four"], linestyle=":", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(20, 75)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_title("A  Corrected subject-held-out performance", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")

    # B: destruction controls.
    control_labels = ["Ordered", "Time\ncollapsed", "Bin\nshuffled", "Channel\nshuffled", "Population\ncounts"]
    control_keys = ["bsc_srp64", "bsc_time_collapsed_srp64",
                    "bsc_bin_shuffled_srp64", "bsc_channel_shuffled_srp64",
                    "bsc_total_by_bin"]
    for dataset, marker in [("three", "o"), ("four", "s")]:
        vals = [100 * results[dataset]["features"][key]["balanced_accuracy"]
                for key in control_keys]
        axes[1].plot(np.arange(len(vals)), vals, marker=marker, linewidth=1.8,
                     color=colors[dataset], label="Three conditions" if dataset == "three" else "Four categories")
    axes[1].axhline(33.33, color=colors["three"], linestyle=":", linewidth=1)
    axes[1].axhline(25, color=colors["four"], linestyle=":", linewidth=1)
    axes[1].set_xticks(np.arange(5), control_labels)
    axes[1].set_ylim(20, 67)
    axes[1].set_ylabel("Balanced accuracy (%)")
    axes[1].set_title("B  Temporal and electrode-order controls", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")

    # C: temporal windows; seed range shown as a band on the early value.
    window_order = ["early_39_273ms", "middle_273_508ms", "late_508_742ms",
                    "latest_742_977ms", "full_39_977ms"]
    window_labels = ["39–273", "273–508", "508–742", "742–977", "39–977"]
    values = [100 * robustness["temporal_windows"][key]["balanced_accuracy"]
              for key in window_order]
    axes[2].plot(np.arange(5), values, color="#467A52", marker="o", linewidth=2)
    seeds = [100 * row["balanced_accuracy"]
             for row in robustness["seed_robustness"].values()]
    axes[2].vlines(0, min(seeds), max(seeds), color="#1B1B1B", linewidth=4,
                   alpha=.55, label="Seeds 7, 42, 99")
    axes[2].axhline(33.33, color="#777777", linestyle=":", linewidth=1)
    axes[2].set_xticks(np.arange(5), window_labels, rotation=24, ha="right")
    axes[2].set_ylim(30, 68)
    axes[2].set_xlabel("Poststimulus window (ms)")
    axes[2].set_ylabel("Balanced accuracy (%)")
    axes[2].set_title("C  Temporal-window and seed robustness", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, loc="lower left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=.6, alpha=.8)
        ax.set_axisbelow(True)
    fig.suptitle("Corrected leakage-safe validation of the fixed-reservoir representation",
                 fontsize=12, fontweight="bold")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
