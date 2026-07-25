#!/usr/bin/env python3
"""Chapter 7 Experiment B: variance decomposition of coupling strength.

The observable is the four-category coupling matrix K with one row per subject.
The script partitions total sum of squares into subject, category, and residual
components, evaluates the category term with within-subject permutation, and
reports ICC(3,1) and the two prespecified within-valence contrasts.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "chapter7_results")
KAPPA_FILE = os.path.join(RESULTS_DIR, "kappa_matrix.csv")
FIGURE_DIR = os.environ.get(
    "ARSPI_FIGURE_DIR",
    os.path.join(REPO_ROOT, "pictures", "chSynthesis"),
)
RESULTS_FILE = os.path.join(RESULTS_DIR, "ch7_expB_results.npz")
CATEGORIES = ["Threat", "Mutilation", "Cute", "Erotic"]
N_PERMUTATIONS = 5000
RANDOM_SEED = 42


def variance_partition(K):
    """Return sums of squares and normalized variance components."""
    n_subjects, n_categories = K.shape
    grand_mean = K.mean()
    subject_means = K.mean(axis=1)
    category_means = K.mean(axis=0)
    ss_total = np.sum((K - grand_mean) ** 2)
    ss_subject = n_categories * np.sum((subject_means - grand_mean) ** 2)
    ss_category = n_subjects * np.sum((category_means - grand_mean) ** 2)
    ss_residual = ss_total - ss_subject - ss_category
    return {
        "grand_mean": grand_mean,
        "subject_means": subject_means,
        "category_means": category_means,
        "ss_total": ss_total,
        "ss_subject": ss_subject,
        "ss_category": ss_category,
        "ss_residual": ss_residual,
        "v_subject": ss_subject / ss_total,
        "v_category": ss_category / ss_total,
        "v_residual": ss_residual / ss_total,
    }


def category_permutation_test(K, partition):
    """Permute category labels independently within each subject."""
    n_subjects, n_categories = K.shape
    df_category = n_categories - 1
    df_residual = (n_subjects - 1) * (n_categories - 1)
    observed_f = (
        partition["ss_category"] / df_category
    ) / (partition["ss_residual"] / df_residual)
    rng = np.random.RandomState(RANDOM_SEED)
    null_f = np.empty(N_PERMUTATIONS)
    for permutation in range(N_PERMUTATIONS):
        permuted = K.copy()
        for subject in range(n_subjects):
            permuted[subject] = permuted[
                subject, rng.permutation(n_categories)
            ]
        category_means = permuted.mean(axis=0)
        ss_category = n_subjects * np.sum(
            (category_means - partition["grand_mean"]) ** 2
        )
        ss_residual = (
            partition["ss_total"]
            - partition["ss_subject"]
            - ss_category
        )
        null_f[permutation] = (
            ss_category / df_category
        ) / (ss_residual / df_residual)
    p_value = (1 + np.sum(null_f >= observed_f)) / (1 + N_PERMUTATIONS)
    return float(observed_f), null_f, float(p_value)


def icc_3_1(partition, n_subjects, n_categories):
    mean_subject = partition["ss_subject"] / (n_subjects - 1)
    mean_residual = partition["ss_residual"] / (
        (n_subjects - 1) * (n_categories - 1)
    )
    return float(
        (mean_subject - mean_residual)
        / (mean_subject + (n_categories - 1) * mean_residual)
    )


def save_raw_figures(K, partition):
    n_subjects = len(K)
    subject_means = partition["subject_means"]
    category_means = partition["category_means"]
    subject_stds = K.std(axis=1, ddof=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    image = axes[0].imshow(
        K[:50], aspect="auto", cmap="viridis", vmin=0, vmax=0.75
    )
    axes[0].set_xlabel("Category")
    axes[0].set_ylabel("Subject (first 50)")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(CATEGORIES, rotation=45)
    axes[0].set_title("Raw coupling matrix")
    plt.colorbar(image, ax=axes[0], label="kappa", shrink=0.8)

    axes[1].scatter(subject_means, subject_stds, alpha=0.4, s=20)
    axes[1].set_xlabel("Subject-mean kappa")
    axes[1].set_ylabel("Within-subject std(kappa)")
    axes[1].set_title("Subject mean and variability")

    for subject in range(0, n_subjects, 3):
        axes[2].plot(range(4), K[subject], color="gray", alpha=0.12, lw=0.6)
    axes[2].plot(
        range(4), category_means, "o-", lw=2.5, markersize=8,
        label="Category mean"
    )
    axes[2].errorbar(
        range(4),
        category_means,
        yerr=K.std(axis=0, ddof=1) / np.sqrt(n_subjects),
        capsize=5,
        fmt="none",
    )
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(CATEGORIES)
    axes[2].set_ylabel("kappa")
    axes[2].set_title("Within-subject category trajectories")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURE_DIR, "fig7_B1_raw_observation.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    contrasts = [
        ("Threat", "Mutilation", 0, 1),
        ("Cute", "Erotic", 2, 3),
    ]
    for ax, (first, second, first_index, second_index) in zip(axes, contrasts):
        difference = K[:, first_index] - K[:, second_index]
        ax.hist(difference, bins=35, alpha=0.7, edgecolor="black", density=True)
        ax.axvline(0, color="black", lw=1.5)
        ax.axvline(
            difference.mean(), linestyle="--", lw=2,
            label=f"Mean = {difference.mean():.4f}"
        )
        ax.axvline(
            np.median(difference), linestyle=":", lw=2,
            label=f"Median = {np.median(difference):.4f}"
        )
        ax.set_xlabel(f"Delta kappa ({first} - {second})")
        ax.set_ylabel("Density")
        ax.set_title(f"{first} - {second}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURE_DIR, "fig7_B2_raw_paired_differences.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_analysis_figure(partition, observed_f, null_f, p_value):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    components = ["Subject\nidentity", "Affective\ncategory", "Residual"]
    values = [
        partition["v_subject"],
        partition["v_category"],
        partition["v_residual"],
    ]
    bars = axes[0].bar(
        components,
        [100 * value for value in values],
        edgecolor="black",
        width=0.6,
    )
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("Variance decomposition of kappa")
    axes[0].set_ylim(0, 85)

    axes[1].hist(null_f, bins=50, alpha=0.7, density=True, label="Null F")
    axes[1].axvline(
        observed_f, linestyle="--", lw=2.5,
        label=f"Observed F = {observed_f:.2f}"
    )
    axes[1].axvline(
        np.percentile(null_f, 95), linestyle=":", lw=1.5,
        label="95th percentile"
    )
    axes[1].set_xlabel("F statistic")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Within-subject permutation, p = {p_value:.3f}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGURE_DIR, "fig7_B3_variance_decomposition.pdf"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    start = time.time()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frame = pd.read_csv(KAPPA_FILE)
    K = frame[CATEGORIES].to_numpy(dtype=float)
    n_subjects, n_categories = K.shape
    if (n_subjects, n_categories) != (211, 4):
        raise ValueError(f"expected coupling matrix (211, 4), received {K.shape}")

    partition = variance_partition(K)
    subject_stds = K.std(axis=1, ddof=1)
    print("=" * 70)
    print("EXPERIMENT 7.2: Variance Decomposition of Coupling Strength")
    print("=" * 70)
    print(f"Data: {n_subjects} subjects x {n_categories} categories")
    print(f"Grand mean kappa: {partition['grand_mean']:.4f}")
    for index, category in enumerate(CATEGORIES):
        values = K[:, index]
        print(
            f"  {category:<14} mean={values.mean():.4f} "
            f"median={np.median(values):.4f} std={values.std(ddof=1):.4f}"
        )
    print(
        "Within-subject std(kappa): "
        f"mean={subject_stds.mean():.4f}, median={np.median(subject_stds):.4f}"
    )

    save_raw_figures(K, partition)
    observed_f, null_f, category_p = category_permutation_test(K, partition)
    icc = icc_3_1(partition, n_subjects, n_categories)
    friedman_stat, friedman_p = friedmanchisquare(
        K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    )

    print("\nVariance decomposition of kappa:")
    print(f"  V_subj:  {100 * partition['v_subject']:.1f}%")
    print(f"  V_cat:   {100 * partition['v_category']:.1f}%")
    print(f"  V_resid: {100 * partition['v_residual']:.1f}%")
    print(f"  F_cat = {observed_f:.4f}")
    print(f"  permutation p = {category_p:.4f}")
    print(f"  ICC(3,1) of kappa across categories: {icc:.4f}")
    print(f"  Friedman chi-square = {friedman_stat:.4f}, p = {friedman_p:.4e}")

    contrast_records = []
    for first, second, first_index, second_index in [
        ("Threat", "Mutilation", 0, 1),
        ("Cute", "Erotic", 2, 3),
    ]:
        difference = K[:, first_index] - K[:, second_index]
        statistic, p_value = wilcoxon(difference)
        effect = difference.mean() / (difference.std(ddof=1) + 1e-10)
        print(
            f"  {first} - {second}: median Delta kappa = "
            f"{np.median(difference):.4f}, d_z = {effect:.4f}, "
            f"p = {p_value:.4e}"
        )
        contrast_records.append(
            [first_index, second_index, np.median(difference), effect, p_value]
        )

    save_analysis_figure(partition, observed_f, null_f, category_p)
    np.savez(
        RESULTS_FILE,
        v_subject=partition["v_subject"],
        v_category=partition["v_category"],
        v_residual=partition["v_residual"],
        observed_f=observed_f,
        null_f=null_f,
        category_p=category_p,
        icc=icc,
        friedman_stat=friedman_stat,
        friedman_p=friedman_p,
        contrasts=np.asarray(contrast_records),
    )
    for filename in [
        "fig7_B1_raw_observation.pdf",
        "fig7_B2_raw_paired_differences.pdf",
        "fig7_B3_variance_decomposition.pdf",
    ]:
        print(f"Saved: {filename}")
    print(f"Completed in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
