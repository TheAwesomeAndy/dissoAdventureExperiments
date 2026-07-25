#!/usr/bin/env python3
"""Chapter 4 synthetic spike-to-embedding characterization.

The controlled task separates two equal-energy inputs that differ in temporal
ordering. The script evaluates reservoir size, representation separability,
spike coding, PCA compression, initialization sensitivity, and the local
parameter neighborhood used by the dissertation. All random states are fixed.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUTDIR = Path("pictures/chLSMEmbeddings")
OUTDIR.mkdir(parents=True, exist_ok=True)


class LIFReservoir:
    """Fixed recurrent LIF reservoir used by the Chapter 4 controls."""

    def __init__(self, n_input, n_res, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (n_input + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, n_input))
        limit_rec = np.sqrt(6.0 / (2 * n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        radius = np.abs(np.linalg.eigvals(self.W_rec)).max()
        if radius > 0:
            self.W_rec *= 0.9 / radius
        self.beta = beta
        self.threshold = threshold
        self.n_res = n_res

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        membrane = np.zeros(self.n_res)
        previous_spikes = np.zeros(self.n_res)
        spikes = np.zeros((len(X), self.n_res))
        membrane_trace = np.zeros_like(spikes)
        for t, sample in enumerate(X):
            current = self.W_in @ sample + self.W_rec @ previous_spikes
            membrane = (
                (1.0 - self.beta)
                * membrane
                * (1.0 - previous_spikes)
                + current
            )
            emitted = (membrane >= self.threshold).astype(float)
            membrane = np.maximum(membrane - emitted * self.threshold, 0.0)
            spikes[t] = emitted
            membrane_trace[t] = membrane
            previous_spikes = emitted
        return spikes, membrane_trace


def generate_temporal_task(
    n_trials_per_class=200,
    n_input=33,
    T=150,
    noise_std=0.15,
    amp_jitter=0.2,
    timing_jitter=3,
    seed=42,
):
    """Generate equal-energy early- and late-emphasis temporal patterns."""
    rng = np.random.RandomState(seed)
    X_all, y_all = [], []
    n_active = 5
    base_amp = 1.5
    for label in range(2):
        amplitude_profile = [1.0, 0.6, 0.3] if label == 0 else [0.3, 0.6, 1.0]
        for _ in range(n_trials_per_class):
            x = np.zeros((T, n_input))
            for start, scale in zip([20, 35, 50], amplitude_profile):
                jitter = rng.randint(-timing_jitter, timing_jitter + 1)
                actual_start = max(0, min(T - 8, start + jitter))
                amplitude = base_amp * scale * (
                    1.0 + rng.uniform(-amp_jitter, amp_jitter)
                )
                x[actual_start : actual_start + 8, :n_active] = amplitude
            x[15:65] += rng.randn(50, n_input) * noise_std
            X_all.append(x)
            y_all.append(label)
    return X_all, np.asarray(y_all)


def extract_mfr(spikes, t_start=10, t_end=70):
    return spikes[t_start:t_end].mean(axis=0)


def extract_lfs(spikes, t_start=10, t_end=70):
    window = spikes[t_start:t_end]
    first = np.full(window.shape[1], len(window), dtype=float)
    for neuron in range(window.shape[1]):
        times = np.flatnonzero(window[:, neuron] > 0)
        if len(times):
            first[neuron] = times[0]
    return first / len(window)


def extract_bsc(spikes, n_bins, t_start=10, t_end=70):
    window = spikes[t_start:t_end]
    bin_size = len(window) // n_bins
    return np.concatenate([
        window[index * bin_size : (index + 1) * bin_size].sum(axis=0)
        for index in range(n_bins)
    ])


def extract_features(spikes, method="bsc6", t_start=10, t_end=70):
    if method == "mfr":
        return extract_mfr(spikes, t_start, t_end)
    if method == "lfs":
        return extract_lfs(spikes, t_start, t_end)
    if method == "bsc3":
        return extract_bsc(spikes, 3, t_start, t_end)
    if method == "bsc6":
        return extract_bsc(spikes, 6, t_start, t_end)
    raise ValueError(f"unknown coding method: {method}")


def run_classification(features, labels, classifier="logreg", n_folds=5):
    features = np.asarray(features)
    labels = np.asarray(labels)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, test_idx in splitter.split(features, labels):
        scaler = StandardScaler().fit(features[train_idx])
        train = scaler.transform(features[train_idx])
        test = scaler.transform(features[test_idx])
        if classifier == "logreg":
            model = LogisticRegression(
                C=0.1,
                solver="liblinear",
                random_state=42,
                max_iter=1000,
            )
        elif classifier == "svm_rbf":
            model = SVC(
                kernel="rbf",
                C=1.0,
                random_state=42,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"unknown classifier: {classifier}")
        model.fit(train, labels[train_idx])
        accuracies.append(model.score(test, labels[test_idx]))
    return float(np.mean(accuracies)), float(np.std(accuracies))


def compute_fdr(features, labels, eps_factor=1e-4):
    features = np.asarray(features)
    labels = np.asarray(labels)
    active = features.var(axis=0) > 1e-12
    if active.sum() < 2:
        return 0.0
    features = features[:, active]
    classes = np.unique(labels)
    class_means = [features[labels == label].mean(axis=0) for label in classes]
    difference = (class_means[0] - class_means[1]).reshape(-1, 1)
    between = difference @ difference.T
    within = np.zeros((features.shape[1], features.shape[1]))
    for label, mean in zip(classes, class_means):
        centered = features[labels == label] - mean
        within += centered.T @ centered
    trace = np.trace(within)
    if trace < 1e-12:
        return 0.0
    regularized = within + eps_factor * trace / features.shape[1] * np.eye(
        features.shape[1]
    )
    try:
        value = float(np.trace(np.linalg.solve(regularized, between)))
    except np.linalg.LinAlgError:
        return 0.0
    return value if np.isfinite(value) and value >= 0 else 0.0


def run_reservoir(X_list, n_res=256, beta=0.05, threshold=0.5, seed=42):
    reservoir = LIFReservoir(
        n_input=33,
        n_res=n_res,
        beta=beta,
        threshold=threshold,
        seed=seed,
    )
    return [reservoir.forward(x)[0] for x in X_list]


def experiment_and_plot_size_ablation(X_list, y):
    print("=" * 60)
    print("EXPERIMENT 1: Reservoir Size Ablation")
    print("=" * 60)
    results = {}
    for n_res in [64, 128, 256, 512]:
        print(f"  N_res = {n_res}...", end=" ", flush=True)
        spikes = run_reservoir(X_list, n_res=n_res)
        features = np.asarray([extract_bsc(s, 6) for s in spikes])
        projection = PCA(
            n_components=min(64, features.shape[1], features.shape[0] - 1)
        ).fit_transform(features)
        mean, std = run_classification(projection, y)
        results[n_res] = (mean, std)
        print(f"Acc = {mean * 100:.1f}% ± {std * 100:.1f}%")

    sizes = sorted(results)
    means = [results[size][0] * 100 for size in sizes]
    stds = [results[size][1] * 100 for size in sizes]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(
        [str(size) for size in sizes],
        means,
        yerr=stds,
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
        width=0.5,
    )
    ax.set_xlabel("Reservoir Size ($N_{res}$)")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Classification Accuracy vs. Reservoir Size")
    ax.set_ylim(max(40, min(means) - 10), min(102, max(means) + 8))
    selected = sizes.index(256)
    ax.annotate(
        "Selected",
        xy=(selected, means[selected] + stds[selected] + 0.5),
        ha="center",
        fontsize=8,
        fontweight="bold",
    )
    fig.savefig(OUTDIR / "ablation_reservoir_size.pdf")
    plt.close(fig)
    print("  -> Saved ablation_reservoir_size.pdf")
    return results


def experiment_and_plot_fdr(X_list, y, spikes_list):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: FDR Three-Way Comparison")
    print("=" * 60)
    raw_features = np.asarray([
        np.concatenate([
            x[10 + index * 10 : 10 + (index + 1) * 10].sum(axis=0)
            for index in range(6)
        ])
        for x in X_list
    ])
    kernel = np.ones(5) / 5.0
    filtered_features = []
    for x in X_list:
        filtered = np.zeros_like(x)
        for channel in range(x.shape[1]):
            filtered[:, channel] = np.convolve(
                x[:, channel], kernel, mode="same"
            )
        filtered_features.append(np.concatenate([
            filtered[10 + index * 10 : 10 + (index + 1) * 10].sum(axis=0)
            for index in range(6)
        ]))
    filtered_features = np.asarray(filtered_features)
    reservoir_features = np.asarray([extract_bsc(s, 6) for s in spikes_list])

    values = {
        "raw": compute_fdr(raw_features, y),
        "filter": compute_fdr(filtered_features, y),
        "lsm": compute_fdr(reservoir_features, y),
    }
    print(f"  FDR (Raw Input):     {values['raw']:.3f}")
    print(f"  FDR (Linear Filter): {values['filter']:.3f}")
    print(f"  FDR (LSM Reservoir): {values['lsm']:.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["Raw Input\n(Binned)", "Linear Filter\n(5-pt MA)", "LSM Reservoir\n(BSC$_6$)"]
    heights = [values["raw"], values["filter"], values["lsm"]]
    bars = ax.bar(labels, heights, edgecolor="black", linewidth=0.5, width=0.55)
    for bar, value in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(heights) * 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Separability Enhancement by the LSM Transformation")
    ax.set_ylim(0, max(heights) * 1.25 if max(heights) > 0 else 1.0)
    fig.savefig(OUTDIR / "fdr_three_way_comparison.pdf")
    plt.close(fig)
    print("  -> Saved fdr_three_way_comparison.pdf")
    return values


def experiment_and_plot_coding(spikes_list, y):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Neural Coding Scheme Comparison")
    print("=" * 60)
    methods = ["mfr", "lfs", "bsc3", "bsc6"]
    classifiers = ["logreg", "svm_rbf"]
    results = {}
    for method in methods:
        features = np.asarray([extract_features(s, method) for s in spikes_list])
        results[method] = {}
        for classifier in classifiers:
            mean, std = run_classification(features, y, classifier)
            results[method][classifier] = (mean, std)
            print(
                f"  {method:6s} + {classifier:8s}: "
                f"{mean * 100:.1f}% ± {std * 100:.1f}%"
            )

    x = np.arange(len(methods))
    width = 0.32
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    lr_mean = [results[m]["logreg"][0] * 100 for m in methods]
    lr_std = [results[m]["logreg"][1] * 100 for m in methods]
    svm_mean = [results[m]["svm_rbf"][0] * 100 for m in methods]
    svm_std = [results[m]["svm_rbf"][1] * 100 for m in methods]
    ax.bar(x - width / 2, lr_mean, width, yerr=lr_std, capsize=3, label="Logistic Regression")
    ax.bar(x + width / 2, svm_mean, width, yerr=svm_std, capsize=3, label="SVM (RBF)")
    ax.set_xlabel("Coding Scheme")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Classification Accuracy by Coding Scheme and Classifier")
    ax.set_xticks(x)
    ax.set_xticklabels(["MFR", "LFS", "BSC$_3$", "BSC$_6$"])
    all_values = lr_mean + svm_mean
    ax.set_ylim(max(40, min(all_values) - 8), min(102, max(all_values) + 6))
    ax.legend(loc="lower right")
    fig.savefig(OUTDIR / "coding_scheme_accuracy_comparison.pdf")
    plt.close(fig)
    print("  -> Saved coding_scheme_accuracy_comparison.pdf")
    return results


def experiment_and_plot_pca(spikes_list, y):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: PCA Dimensionality Reduction")
    print("=" * 60)
    features = np.asarray([extract_bsc(s, 6) for s in spikes_list])
    pca_full = PCA().fit(features)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    pca_accuracies = {}
    for n_components in [5, 10, 20, 32, 64, 128, 256]:
        if n_components > min(features.shape):
            continue
        projected = PCA(n_components=n_components).fit_transform(features)
        mean, std = run_classification(projected, y)
        pca_accuracies[n_components] = (mean, std)
        variance = cumulative[min(n_components - 1, len(cumulative) - 1)] * 100
        print(
            f"  PCA-{n_components:3d}: Acc = {mean * 100:.1f}% ± "
            f"{std * 100:.1f}%, Var = {variance:.1f}%"
        )
    full_mean, full_std = run_classification(features, y)
    print(
        f"  Full BSC6 (d={features.shape[1]}): Acc = {full_mean * 100:.1f}% "
        f"± {full_std * 100:.1f}%"
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))
    n_show = min(200, len(cumulative))
    ax.plot(range(1, n_show + 1), cumulative[:n_show] * 100, linewidth=1.5)
    ax.axvline(64, linestyle="--", linewidth=1, label="$d = 64$ (selected)")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("PCA Explained Variance Curve (BSC$_6$ Features)")
    ax.set_xlim(0, n_show)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    fig.savefig(OUTDIR / "pca_explained_variance.pdf")
    plt.close(fig)
    print("  -> Saved pca_explained_variance.pdf")

    pca64 = PCA(n_components=min(64, features.shape[1], features.shape[0] - 1)).fit(features)
    n_neurons = spikes_list[0].shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    for index, ax in enumerate(axes.flat):
        component = pca64.components_[index].reshape(6, n_neurons)
        temporal = component.mean(axis=1)
        ax.bar(range(6), temporal, edgecolor="black", linewidth=0.5, width=0.6)
        ax.axhline(0, linewidth=0.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(["10-20", "20-30", "30-40", "40-50", "50-60", "60-70"], rotation=30, fontsize=7)
        ax.set_title(f"PC{index + 1}", fontsize=10, fontweight="bold")
    fig.suptitle("Temporal Structure of Top 4 Principal Components", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "pca_component_visualization.pdf")
    plt.close(fig)
    print("  -> Saved pca_component_visualization.pdf")
    return {"cumvar": cumulative, "pca_accs": pca_accuracies}


def experiment_and_plot_robustness(X_list, y, n_seeds=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Cross-Initialization Robustness")
    print("=" * 60)
    bsc6, bsc3, mfr = [], [], []
    for seed in range(n_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        spikes = run_reservoir(X_list, n_res=256, seed=seed * 7 + 13)
        bsc6_features = np.asarray([extract_bsc(s, 6) for s in spikes])
        projected = PCA(
            n_components=min(64, bsc6_features.shape[1], bsc6_features.shape[0] - 1)
        ).fit_transform(bsc6_features)
        bsc6_accuracy, _ = run_classification(projected, y)
        bsc3_accuracy, _ = run_classification(
            np.asarray([extract_bsc(s, 3) for s in spikes]), y
        )
        mfr_accuracy, _ = run_classification(
            np.asarray([extract_mfr(s) for s in spikes]), y
        )
        bsc6.append(bsc6_accuracy)
        bsc3.append(bsc3_accuracy)
        mfr.append(mfr_accuracy)
        print(
            f"BSC6={bsc6_accuracy * 100:.1f}%, "
            f"BSC3={bsc3_accuracy * 100:.1f}%, "
            f"MFR={mfr_accuracy * 100:.1f}%"
        )
    results = {
        "bsc6": np.asarray(bsc6),
        "bsc3": np.asarray(bsc3),
        "mfr": np.asarray(mfr),
    }
    print(
        f"\n  BSC6+PCA64: {results['bsc6'].mean() * 100:.1f}% ± "
        f"{results['bsc6'].std() * 100:.1f}%"
    )
    print(
        f"  BSC3:       {results['bsc3'].mean() * 100:.1f}% ± "
        f"{results['bsc3'].std() * 100:.1f}%"
    )
    print(
        f"  MFR:        {results['mfr'].mean() * 100:.1f}% ± "
        f"{results['mfr'].std() * 100:.1f}%"
    )

    data = [results["mfr"] * 100, results["bsc3"] * 100, results["bsc6"] * 100]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.boxplot(data, tick_labels=["MFR", "BSC$_3$", "BSC$_6$+PCA-64"], patch_artist=True, widths=0.4)
    for index, values in enumerate(data):
        jitter = np.random.RandomState(0).uniform(-0.08, 0.08, len(values))
        ax.scatter(np.ones(len(values)) * (index + 1) + jitter, values, s=20, alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness Across 10 Random Initializations")
    ax.set_ylim(max(35, min(values.min() for values in data) - 5), 105)
    ax.axhline(50, linestyle=":", alpha=0.3, label="Chance")
    ax.legend(loc="lower right")
    fig.savefig(OUTDIR / "cross_initialization_robustness.pdf")
    plt.close(fig)
    print("  -> Saved cross_initialization_robustness.pdf")
    return results


def experiment_and_plot_sensitivity(X_list, y):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Parameter Sensitivity")
    print("=" * 60)
    betas = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    grid = np.zeros((len(betas), len(thresholds)))
    for i, beta in enumerate(betas):
        for j, threshold in enumerate(thresholds):
            print(f"  β={beta:.2f}, M_th={threshold:.1f}...", end=" ", flush=True)
            spikes = run_reservoir(X_list, beta=beta, threshold=threshold)
            features = np.asarray([extract_bsc(s, 6) for s in spikes])
            projected = PCA(
                n_components=min(64, features.shape[1], features.shape[0] - 1)
            ).fit_transform(features)
            mean, _ = run_classification(projected, y)
            grid[i, j] = mean * 100
            print(f"{mean * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    lower = max(50, grid.min() - 5)
    image = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=lower, vmax=100)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{value:.1f}" for value in thresholds])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"{value:.2f}" for value in betas])
    ax.set_xlabel("Firing Threshold ($M_{th}$)")
    ax.set_ylabel("Membrane Decay ($\\beta$)")
    ax.set_title("Accuracy (%) vs. LSM Parameters")
    for i in range(len(betas)):
        for j in range(len(thresholds)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8, fontweight="bold")
    selected = plt.Rectangle(
        (thresholds.index(0.5) - 0.5, betas.index(0.05) - 0.5),
        1,
        1,
        fill=False,
        edgecolor="blue",
        linewidth=2.5,
    )
    ax.add_patch(selected)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("Accuracy (%)")
    fig.savefig(OUTDIR / "parameter_sensitivity_heatmap.pdf")
    plt.close(fig)
    print("  -> Saved parameter_sensitivity_heatmap.pdf")
    return {"betas": betas, "thresholds": thresholds, "grid": grid}


def main():
    print("ARSPI-Net Chapter 4: Redesigned Experimental Pipeline")
    print("Task: Temporal Pattern Discrimination (Early vs Late Burst)")
    print("=" * 60)
    X_list, y = generate_temporal_task(
        noise_std=0.5,
        amp_jitter=0.3,
        timing_jitter=5,
        seed=42,
    )
    print(
        f"Generated {len(X_list)} trials "
        f"({int(y.sum())} class 1, {int(len(y) - y.sum())} class 0)"
    )
    print("\nRunning baseline reservoir (N=256, β=0.05, M_th=0.5)...")
    baseline_spikes = run_reservoir(X_list, n_res=256, seed=42)
    total_spikes = sum(spikes.sum() for spikes in baseline_spikes)
    print(
        f"  Total spikes: {total_spikes:.0f}, "
        f"Mean/trial: {total_spikes / len(baseline_spikes):.1f}"
    )

    experiment_and_plot_size_ablation(X_list, y)
    fdr_results = experiment_and_plot_fdr(X_list, y, baseline_spikes)
    coding_results = experiment_and_plot_coding(baseline_spikes, y)
    experiment_and_plot_pca(baseline_spikes, y)
    robustness = experiment_and_plot_robustness(X_list, y, n_seeds=10)
    experiment_and_plot_sensitivity(X_list, y)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE - SUMMARY")
    print("=" * 60)
    print(
        f"\nFDR: Raw={fdr_results['raw']:.2f}, "
        f"Filter={fdr_results['filter']:.2f}, LSM={fdr_results['lsm']:.2f}"
    )
    print("\nCoding (LogReg):")
    for method in ["mfr", "lfs", "bsc3", "bsc6"]:
        mean, std = coding_results[method]["logreg"]
        print(f"  {method:6s}: {mean * 100:.1f}% ± {std * 100:.1f}%")
    print("\nRobustness:")
    print(
        f"  BSC6+PCA64: {np.mean(robustness['bsc6']) * 100:.1f}% ± "
        f"{np.std(robustness['bsc6']) * 100:.1f}%"
    )
    print(
        f"  MFR:        {np.mean(robustness['mfr']) * 100:.1f}% ± "
        f"{np.std(robustness['mfr']) * 100:.1f}%"
    )
    print(f"\nFigures saved to {OUTDIR}/:")
    for figure in sorted(OUTDIR.glob("*.pdf")):
        print(f"  {figure.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
