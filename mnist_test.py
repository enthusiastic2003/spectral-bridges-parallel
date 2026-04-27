"""
Reproduce paper Figure 4 layout: ARI and NMI vs. embedding dimension h,
comparing the author's Python SpectralBridges against our CUDA implementation.

Protocol (matching paper section 4.6):
  - 10 runs per (h, method) pair
  - Each run: fresh 20,000-sample subset of MNIST, fresh PCA
  - Paired seeds: both methods see identical (subset, seed) per run
  - Mean +/- std reported; error bars in plot are 1 std

Outputs:
  - results_h_sweep.csv : raw per-run records (written incrementally)
  - figure4_reproduction.png : grouped bar plot
"""

import os
import sys
import time
import csv
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
N_RUNS = 10
BASE_SEED = 22188
PERPLEXITY = 2.0
N_ITER = 20
NUM_THREADS = 12

CSV_PATH = "results_h_sweep.csv"
PLOT_PATH = "figure4_reproduction.png"


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
    elif method == "cuda":
        model = specbridge.SpectralClustering(
            n_clusters=N_CLUSTERS,
            num_voronoi=M_VORONOI,
            n_iter=N_ITER,
            target_perplexity=PERPLEXITY,
            random_state=seed,
            use_gpu=True,
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


def make_plot(records):
    """Grouped bar plot: ARI and NMI vs h, author vs cuda."""
    h_labels = [f"h={h}" if h != 784 else "h=784 (full)" for h in H_DIMS]

    def stats(metric_idx, method):
        means, stds = [], []
        for h in H_DIMS:
            arr = np.array([r[metric_idx] for r in records[h][method]])
            means.append(arr.mean())
            stds.append(arr.std(ddof=1))
        return np.array(means), np.array(stds)

    ari_auth_m, ari_auth_s = stats(0, "author")
    ari_cuda_m, ari_cuda_s = stats(0, "cuda")
    nmi_auth_m, nmi_auth_s = stats(1, "author")
    nmi_cuda_m, nmi_cuda_s = stats(1, "cuda")

    x = np.arange(len(H_DIMS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ARI
    ax = axes[0]
    ax.bar(x - width / 2, ari_auth_m, width, yerr=ari_auth_s,
           label="Author (Python)", color="#9b8bd0", capsize=3)
    ax.bar(x + width / 2, ari_cuda_m, width, yerr=ari_cuda_s,
           label="Ours (CUDA)", color="#5a3a8a", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, rotation=20)
    ax.set_ylabel("ARI Score")
    ax.set_ylim(0, max(ari_auth_m.max(), ari_cuda_m.max()) * 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # NMI
    ax = axes[1]
    ax.bar(x - width / 2, nmi_auth_m, width, yerr=nmi_auth_s,
           label="Author (Python)", color="#9b8bd0", capsize=3)
    ax.bar(x + width / 2, nmi_cuda_m, width, yerr=nmi_cuda_s,
           label="Ours (CUDA)", color="#5a3a8a", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, rotation=20)
    ax.set_ylabel("NMI Score")
    ax.set_ylim(0, max(nmi_auth_m.max(), nmi_cuda_m.max()) * 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Spectral Bridges on MNIST: Author Python vs. Our CUDA "
        f"({N_RUNS} runs, n={N_SAMPLES}, m={M_VORONOI})"
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"\nSaved plot to {PLOT_PATH}")


def print_summary_table(records):
    print("\n" + "=" * 84)
    print(f"Summary across h values ({N_RUNS} paired runs each)")
    print("=" * 84)
    print(f"{'h':<6} | {'ARI auth':<16} | {'ARI cuda':<16} | "
          f"{'NMI auth':<16} | {'NMI cuda':<16}")
    print("-" * 84)
    for h in H_DIMS:
        a = np.array(records[h]["author"])
        c = np.array(records[h]["cuda"])
        print(
            f"{h:<6} | "
            f"{a[:,0].mean():.4f} +/- {a[:,0].std(ddof=1):.4f}  | "
            f"{c[:,0].mean():.4f} +/- {c[:,0].std(ddof=1):.4f}  | "
            f"{a[:,1].mean():.4f} +/- {a[:,1].std(ddof=1):.4f}  | "
            f"{c[:,1].mean():.4f} +/- {c[:,1].std(ddof=1):.4f}"
        )
    print("=" * 84)


def main():
    print("Loading MNIST...")
    X_full, y_full = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    X_full = X_full.astype(np.float32) / 255.0
    y_full = y_full.astype(int)

    specbridge.set_num_threads(NUM_THREADS)

    # records[h][method] = list of (ari, nmi, time) tuples
    records = {h: {"author": [], "cuda": []} for h in H_DIMS}

    # Open CSV up front so partial results survive a crash.
    csv_file = open(CSV_PATH, "w", newline="")
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

                for method in ("author", "cuda"):
                    ari, nmi, t = run_one(method, X_pca, y_true, seed)
                    records[h][method].append((ari, nmi, t))
                    writer.writerow([h, run_idx, seed, method, ari, nmi, t])
                    csv_file.flush()

                a_a, a_n, a_t = records[h]["author"][-1]
                c_a, c_n, c_t = records[h]["cuda"][-1]
                print(
                    f"  run {run_idx + 1:2d}/{N_RUNS} seed={seed} | "
                    f"ARI auth={a_a:.3f} cuda={c_a:.3f} | "
                    f"NMI auth={a_n:.3f} cuda={c_n:.3f} | "
                    f"t auth={a_t:.2f}s cuda={c_t:.2f}s"
                )
    finally:
        csv_file.close()

    print_summary_table(records)
    make_plot(records)


if __name__ == "__main__":
    main()