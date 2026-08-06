"""
Reproduce paper Figure 4 layout: ARI and NMI vs. embedding dimension h,
comparing the author's Python SpectralBridges against our C++ implementation.

Protocol (matching paper section 4.6):
  - 10 runs per (h, method) pair
  - Each run: fresh 20,000-sample subset of dataset, fresh PCA
  - Paired seeds: both methods see identical (subset, seed) per run
  - Mean +/- std reported; error bars in plot are 1 std

Outputs:
  - results_h_sweep_<dataset>_<mode>.csv : raw per-run records (written incrementally)
  - figure4_reproduction_<dataset>_<mode>.png : grouped bar plot
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sbcluster import SpectralBridges
sys.path.append(os.path.abspath("build/"))
import specbridge

# ---- Configuration ------------------------------------------------------
H_DIMS = [8, 16, 32, 64, 784]
N_SAMPLES = 20000
N_CLUSTERS = 10
M_VORONOI = 250
N_RUNS = 5
BASE_SEED = 22188
PERPLEXITY = 2.0
N_ITER = 20
NUM_THREADS = 36


def run_one(method, X_pca, y_true, seed):
    """Run a single fit; return (ari, nmi, elapsed_seconds)."""
    if method == "author":
        model = SpectralBridges(
            n_clusters=N_CLUSTERS,
            n_nodes=M_VORONOI,
            perplexity=PERPLEXITY,
            n_iter=N_ITER,
            random_state=seed,
        )
        t0 = time.time()
        model.fit(X_pca)
        elapsed = time.time() - t0
        labels = model.labels_

    elif method in ("cpu", "gpu"):
        use_gpu = (method == "gpu")
        model = specbridge.SpectralClustering(
            n_clusters=N_CLUSTERS,
            num_voronoi=M_VORONOI,
            n_iter=N_ITER,
            target_perplexity=PERPLEXITY,
            random_state=seed,
            use_gpu=use_gpu,
        )
        t0 = time.time()
        result = model.fit(X_pca)
        elapsed = time.time() - t0
        labels = np.array(result.labels)

    else:
        raise ValueError(method)

    return (
        adjusted_rand_score(y_true, labels),
        normalized_mutual_info_score(y_true, labels),
        elapsed,
    )


def make_plot(records, dataset_name, mode_name, methods_to_run, plot_path):
    """Grouped bar plot: ARI and NMI vs h."""
    h_labels = [f"h={h}" if h != 784 else "h=784 (full)" for h in H_DIMS]

    def stats(metric_idx, method):
        means, stds = [], []
        for h in H_DIMS:
            arr = np.array([r[metric_idx] for r in records[h][method]])
            means.append(arr.mean())
            stds.append(arr.std(ddof=1))
        return np.array(means), np.array(stds)

    x = np.arange(len(H_DIMS))
    n_bars = len(methods_to_run)
    width = 0.8 / n_bars

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    method_info = {
        "author": {"label": "Author (Python)", "color": "#9b8bd0"},
        "cpu": {"label": "Ours (cpu)", "color": "#5a3a8a"},
        "gpu": {"label": "Ours (cuda)", "color": "#2c1e4a"},
    }

    # ARI
    ax = axes[0]
    max_ari = 0
    for i, method in enumerate(methods_to_run):
        m_mean, m_std = stats(0, method)
        if len(m_mean) > 0:
            max_ari = max(max_ari, m_mean.max())
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, m_mean, width, yerr=m_std,
               label=method_info[method]["label"], color=method_info[method]["color"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, rotation=20)
    ax.set_ylabel("ARI Score")
    if max_ari > 0:
        ax.set_ylim(0, max_ari * 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # NMI
    ax = axes[1]
    max_nmi = 0
    for i, method in enumerate(methods_to_run):
        m_mean, m_std = stats(1, method)
        if len(m_mean) > 0:
            max_nmi = max(max_nmi, m_mean.max())
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(x + offset, m_mean, width, yerr=m_std,
               label=method_info[method]["label"], color=method_info[method]["color"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, rotation=20)
    ax.set_ylabel("NMI Score")
    if max_nmi > 0:
        ax.set_ylim(0, max_nmi * 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    dataset_title = "MNIST" if dataset_name == "mnist" else "Fashion MNIST"
    if mode_name == "both":
        mode_title = "cpu & cuda"
    elif mode_name == "gpu":
        mode_title = "cuda"
    else:
        mode_title = "cpu"

    fig.suptitle(
        f"Spectral Bridges on {dataset_title}: Author Python vs. Ours ({mode_title}) "
        f"({N_RUNS} runs, n={N_SAMPLES}, m={M_VORONOI})"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


def print_summary_table(records, methods_to_run):
    print("\n" + "=" * 110)
    print(f"Summary across h values ({N_RUNS} paired runs each)")
    print("=" * 110)
    
    headers = [f"{'h':<6}"]
    for m in methods_to_run:
        headers.append(f"ARI {m:<10}")
    for m in methods_to_run:
        headers.append(f"NMI {m:<10}")
        
    print(" | ".join(headers))
    print("-" * 110)
    
    for h in H_DIMS:
        row = [f"{h:<6}"]
        # ARI
        for m in methods_to_run:
            a = np.array(records[h][m])
            row.append(f"{a[:,0].mean():.4f} +/- {a[:,0].std(ddof=1):.4f}")
        # NMI
        for m in methods_to_run:
            a = np.array(records[h][m])
            row.append(f"{a[:,1].mean():.4f} +/- {a[:,1].std(ddof=1):.4f}")
            
        print(" | ".join(row))
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Spectral Bridges MNIST/Fashion-MNIST tester")
    parser.add_argument("--mode", type=str, choices=["cpu", "gpu", "both"], default="cpu",
                        help="Execution mode (cpu, gpu, or both)")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashion"], default="mnist",
                        help="Dataset to use (mnist or fashion)")
    args = parser.parse_args()

    dataset_name = args.dataset
    mode_name = args.mode

    if mode_name == "both":
        methods_to_run = ["author", "cpu", "gpu"]
    elif mode_name == "gpu":
        methods_to_run = ["author", "gpu"]
    else:
        methods_to_run = ["author", "cpu"]

    csv_path = f"results_h_sweep_{dataset_name}_{mode_name}.csv"
    plot_path = f"figure4_reproduction_{dataset_name}_{mode_name}.png"

    if dataset_name == "mnist":
        print("Loading MNIST...")
        X_full, y_full = fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
    else:
        print("Loading Fashion MNIST...")
        X_full, y_full = fetch_openml(
            "Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto"
        )
        
    X_full = X_full.astype(np.float32) / 255.0
    y_full = y_full.astype(int)

    specbridge.set_num_threads(NUM_THREADS)

    records = {h: {m: [] for m in methods_to_run} for h in H_DIMS}

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["h_dim", "run", "seed", "method", "ari", "nmi", "time_s"])
    csv_file.flush()

    try:
        for h in H_DIMS:
            print(f"\n=== h = {h} ===")
            for run_idx in range(N_RUNS):
                seed = BASE_SEED + run_idx
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(X_full), N_SAMPLES, replace=False)
                X_sub = X_full[idx]
                y_true = y_full[idx]

                pca = PCA(n_components=h, random_state=seed)
                X_pca = np.ascontiguousarray(
                    pca.fit_transform(X_sub), dtype=np.float32
                )

                for method in methods_to_run:
                    ari, nmi, t = run_one(method, X_pca, y_true, seed)
                    records[h][method].append((ari, nmi, t))
                    writer.writerow([h, run_idx, seed, method, ari, nmi, t])
                    csv_file.flush()

                print_str = f"  run {run_idx + 1:2d}/{N_RUNS} seed={seed} | "
                for m in methods_to_run:
                    a_a, a_n, a_t = records[h][m][-1]
                    print_str += f"{m}: ARI={a_a:.3f} NMI={a_n:.3f} t={a_t:.2f}s | "
                print(print_str.strip(" | "))
    finally:
        csv_file.close()

    print_summary_table(records, methods_to_run)
    make_plot(records, dataset_name, mode_name, methods_to_run, plot_path)


if __name__ == "__main__":
    main()
