#!/usr/bin/env python3
"""Chapter 7 Experiment C: category-conditioned coupling matrices.

The script expands the scalar coupling statistic into its 7 x 2 Spearman
correlation matrix and evaluates two prespecified within-valence contrasts.
Cellwise Wilcoxon tests use Bonferroni correction; a subject-level sign-flip
permutation evaluates the Frobenius norm of the complete mean difference.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "chapter7_results")
C_MATRIX_FILE = os.path.join(RESULTS_DIR, "C_matrices.csv")
FIGURE_DIR = os.environ.get(
    "ARSPI_FIGURE_DIR",
    os.path.join(REPO_ROOT, "pictures", "chSynthesis"),
)
CATEGORIES = ["Threat", "Mutilation", "Cute", "Erotic"]
ALPHA = 0.05
N_CELLS = 14
N_PATTERN_PERMUTATIONS = 5000
RANDOM_SEED = 42


def load_coupling_tensor():
    frame = pd.read_csv(C_MATRIX_FILE)
    correlation_columns = [column for column in frame.columns if "_x_" in column]
    dynamical = []
    topological = []
    for column in correlation_columns:
        dynamic_name, topology_name = column.rsplit("_x_", 1)
        if dynamic_name not in dynamical:
            dynamical.append(dynamic_name)
        if topology_name not in topological:
            topological.append(topology_name)
    subjects = sorted(frame["subject"].unique())
    tensor = np.empty((len(subjects), len(CATEGORIES), len(dynamical), len(topological)))
    for subject_index, subject in enumerate(subjects):
        for category_index, category in enumerate(CATEGORIES):
            rows = frame[
                (frame["subject"] == subject)
                & (frame["category"] == category)
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"expected one C matrix for subject={subject}, "
                    f"category={category}; found {len(rows)}"
                )
            row = rows.iloc[0]
            for dynamic_index, dynamic_name in enumerate(dynamical):
                for topology_index, topology_name in enumerate(topological):
                    tensor[
                        subject_index,
                        category_index,
                        dynamic_index,
                        topology_index,
                    ] = row[f"{dynamic_name}_x_{topology_name}"]
    if tensor.shape != (211, 4, 7, 2):
        raise ValueError(f"expected C tensor (211, 4, 7, 2), received {tensor.shape}")
    return tensor, dynamical, topological


def cell_statistics(delta, dynamical, topological, alpha_corrected):
    records = []
    for dynamic_index, dynamic_name in enumerate(dynamical):
        for topology_index, topology_name in enumerate(topological):
            values = delta[:, dynamic_index, topology_index]
            statistic, p_value = wilcoxon(values)
            effect = values.mean() / (values.std(ddof=0) + 1e-10)
            records.append({
                "dynamic": dynamic_name,
                "topology": topology_name,
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "effect": float(effect),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha_corrected),
            })
    return records


def pattern_test(delta, seed):
    observed = float(np.linalg.norm(delta.mean(axis=0), ord="fro"))
    rng = np.random.RandomState(seed)
    null = np.empty(N_PATTERN_PERMUTATIONS)
    for permutation in range(N_PATTERN_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=len(delta))
        null[permutation] = np.linalg.norm(
            (delta * signs[:, None, None]).mean(axis=0), ord="fro"
        )
    p_value = float(
        (1 + np.sum(null >= observed)) / (1 + N_PATTERN_PERMUTATIONS)
    )
    return observed, null, p_value


def save_category_means(category_means, dynamical, topological):
    figure, axes = plt.subplots(1, 4, figsize=(18, 6))
    maximum = max(float(np.abs(category_means).max()), 1e-8)
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    for category_index, (category, axis) in enumerate(zip(CATEGORIES, axes)):
        image = axis.imshow(
            category_means[category_index],
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
        )
        axis.set_xticks(range(len(topological)))
        axis.set_xticklabels(topological)
        axis.set_yticks(range(len(dynamical)))
        axis.set_yticklabels(dynamical if category_index == 0 else [])
        axis.set_title(category)
        for dynamic_index in range(len(dynamical)):
            for topology_index in range(len(topological)):
                value = category_means[
                    category_index, dynamic_index, topology_index
                ]
                axis.text(
                    topology_index,
                    dynamic_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.6 * maximum else "black",
                )
        if category_index == 3:
            plt.colorbar(image, ax=axis, label="Mean Spearman rho", shrink=0.7)
    figure.tight_layout()
    figure.savefig(
        os.path.join(FIGURE_DIR, "fig7_C1_category_C_matrices.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_raw_differences(contrasts, dynamical, topological):
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, contrast in zip(axes, contrasts):
        mean_difference = contrast["delta"].mean(axis=0)
        maximum = max(float(np.abs(mean_difference).max()), 0.01)
        image = axis.imshow(
            mean_difference,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
            aspect="auto",
        )
        axis.set_xticks(range(len(topological)))
        axis.set_xticklabels(topological)
        axis.set_yticks(range(len(dynamical)))
        axis.set_yticklabels(dynamical)
        axis.set_title(f"Mean Delta C: {contrast['label']}")
        for dynamic_index in range(len(dynamical)):
            for topology_index in range(len(topological)):
                axis.text(
                    topology_index,
                    dynamic_index,
                    f"{mean_difference[dynamic_index, topology_index]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        plt.colorbar(image, ax=axis, label="Mean Delta rho", shrink=0.7)
    figure.tight_layout()
    figure.savefig(
        os.path.join(FIGURE_DIR, "fig7_C2_raw_difference_matrices.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_significance_matrices(contrasts, dynamical, topological, alpha_corrected):
    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, contrast in zip(axes, contrasts):
        delta = contrast["delta"]
        mean_difference = delta.mean(axis=0)
        p_values = np.empty((len(dynamical), len(topological)))
        for dynamic_index in range(len(dynamical)):
            for topology_index in range(len(topological)):
                _, p_values[dynamic_index, topology_index] = wilcoxon(
                    delta[:, dynamic_index, topology_index]
                )
        maximum = max(float(np.abs(mean_difference).max()), 0.005)
        image = axis.imshow(
            mean_difference,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
            aspect="auto",
        )
        axis.set_xticks(range(len(topological)))
        axis.set_xticklabels(topological)
        axis.set_yticks(range(len(dynamical)))
        axis.set_yticklabels(dynamical)
        axis.set_title(f"Delta C: {contrast['label']}")
        for dynamic_index in range(len(dynamical)):
            for topology_index in range(len(topological)):
                significant = p_values[dynamic_index, topology_index] < alpha_corrected
                axis.text(
                    topology_index,
                    dynamic_index,
                    f"{mean_difference[dynamic_index, topology_index]:.4f}"
                    + ("*" if significant else ""),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold" if significant else "normal",
                )
        plt.colorbar(image, ax=axis, label="Mean Delta rho", shrink=0.7)
    figure.tight_layout()
    figure.savefig(
        os.path.join(FIGURE_DIR, "fig7_C3_difference_significance.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_top_cells(delta, records, dynamical, topological):
    ranked = sorted(records, key=lambda record: abs(record["effect"]), reverse=True)
    dynamic_index = {name: index for index, name in enumerate(dynamical)}
    topology_index = {name: index for index, name in enumerate(topological)}
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, record in zip(axes.flat, ranked[:6]):
        values = delta[
            :,
            dynamic_index[record["dynamic"]],
            topology_index[record["topology"]],
        ]
        axis.hist(values, bins=30, alpha=0.7, edgecolor="black", density=True)
        axis.axvline(0, color="black", lw=1)
        axis.axvline(values.mean(), linestyle="--", lw=2)
        axis.set_title(
            f"{record['dynamic']} x {record['topology']}\n"
            f"d_z={record['effect']:.3f}, p={record['p_value']:.4f}"
        )
        axis.set_xlabel("Delta rho (Cute - Erotic)")
    figure.tight_layout()
    figure.savefig(
        os.path.join(FIGURE_DIR, "fig7_C4_CE_top_cells.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    start = time.time()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    tensor, dynamical, topological = load_coupling_tensor()
    alpha_corrected = ALPHA / N_CELLS
    category_means = tensor.mean(axis=0)
    contrasts = [
        {
            "label": "Cute - Erotic",
            "delta": tensor[:, 2] - tensor[:, 3],
            "seed": RANDOM_SEED,
        },
        {
            "label": "Threat - Mutilation",
            "delta": tensor[:, 0] - tensor[:, 1],
            "seed": RANDOM_SEED + 1,
        },
    ]

    print("=" * 70)
    print("EXPERIMENT 7.3: Category-Conditioned Coupling Structure")
    print("=" * 70)
    print(
        f"Data: {tensor.shape[0]} subjects, {len(CATEGORIES)} categories, "
        f"{len(dynamical)} x {len(topological)} coupling cells"
    )
    for category_index, category in enumerate(CATEGORIES):
        print(f"\n{category} group-mean C matrix")
        for dynamic_index, dynamic_name in enumerate(dynamical):
            print(
                f"  {dynamic_name:<24} "
                f"{category_means[category_index, dynamic_index, 0]: .4f} "
                f"{category_means[category_index, dynamic_index, 1]: .4f}"
            )

    save_category_means(category_means, dynamical, topological)
    save_raw_differences(contrasts, dynamical, topological)

    all_records = {}
    pattern_results = {}
    for contrast in contrasts:
        records = cell_statistics(
            contrast["delta"], dynamical, topological, alpha_corrected
        )
        all_records[contrast["label"]] = records
        significant = sum(record["significant"] for record in records)
        print(
            f"\n{contrast['label']} Significant cells (Bonferroni): "
            f"{significant}/{N_CELLS}"
        )
        print(f"  {'Metric':<24} {'Topo':<12} {'Mean':>9} {'d_z':>9} {'p':>12}")
        for record in records:
            print(
                f"  {record['dynamic']:<24} {record['topology']:<12} "
                f"{record['mean']:>9.4f} {record['effect']:>9.4f} "
                f"{record['p_value']:>12.4e}"
            )
        observed, null, p_value = pattern_test(
            contrast["delta"], contrast["seed"]
        )
        pattern_results[contrast["label"]] = (observed, null, p_value)
        print(
            f"  Pattern Frobenius norm = {observed:.4f}, "
            f"null median = {np.median(null):.4f}, p = {p_value:.4f}"
        )

    cute_erotic = contrasts[0]["delta"]
    amplitude_indices = [0, 1, 2, 3]
    temporal_indices = [5, 6]
    for family_name, indices in [
        ("Amplitude-tracking", amplitude_indices),
        ("Temporal-structure", temporal_indices),
    ]:
        effects = []
        p_values = []
        for dynamic_index in indices:
            for topology_index in range(len(topological)):
                values = cute_erotic[:, dynamic_index, topology_index]
                _, p_value = wilcoxon(values)
                effects.append(abs(values.mean() / (values.std(ddof=0) + 1e-10)))
                p_values.append(p_value)
        print(
            f"{family_name}: mean|d_z|={np.mean(effects):.4f}, "
            f"min p={min(p_values):.4e}"
        )

    save_significance_matrices(
        contrasts, dynamical, topological, alpha_corrected
    )
    save_top_cells(
        cute_erotic,
        all_records["Cute - Erotic"],
        dynamical,
        topological,
    )

    for filename in [
        "fig7_C1_category_C_matrices.pdf",
        "fig7_C2_raw_difference_matrices.pdf",
        "fig7_C3_difference_significance.pdf",
        "fig7_C4_CE_top_cells.pdf",
    ]:
        print(f"Saved: {filename}")
    print(f"Completed in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
