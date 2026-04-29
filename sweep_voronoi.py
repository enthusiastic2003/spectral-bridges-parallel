"""
Sweep num_voronoi for specbridge.SpectralClustering and compare to sklearn's SpectralClustering.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sbcluster import SpectralBridges

# Assuming sbcluster handles spectral bridges internally, but we use specbridge ourself
sys.path.append(os.path.abspath("build/"))
import specbridge

# ---- Configuration ------------------------------------------------------
PCA_DIMS = 49
N_SAMPLES = 20000
N_CLUSTERS = 10
M_VORONOI_RANGE = list(range(11, 120))
N_RUNS = 5
BASE_SEED = 22188
PERPLEXITY = 2.0
N_ITER = 20
NUM_THREADS = 36

# Sklearn best hyperparameters
SKLEARN_PARAMS = {
    "n_clusters": N_CLUSTERS,
    "affinity": "nearest_neighbors",
    "n_neighbors": 6,
    "assign_labels": "kmeans",
    "eigen_solver": "lobpcg",
    "n_jobs": -1
}

def main():
    parser = argparse.ArgumentParser(description="Sweep num_voronoi vs Sklearn")
    parser.add_argument("--mode", type=str, choices=["cpu", "gpu", "both"], default="cpu",
                        help="Execution mode (cpu, gpu, or both)")
    args = parser.parse_args()

    mode_name = args.mode

    if mode_name == "both":
        methods_to_run = ["cpu", "gpu", "author"]
    else:
        methods_to_run = [mode_name, "author"]

    print("Loading MNIST...")
    X_full, y_full = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    X_full = X_full.astype(np.float32) / 255.0
    y_full = y_full.astype(int)

    specbridge.set_num_threads(NUM_THREADS)

    # Cache datasets for runs to save massive amounts of time
    datasets = []
    print("Preparing data and PCA for each run...")
    for run_idx in range(N_RUNS):
        seed = BASE_SEED + run_idx
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_full), N_SAMPLES, replace=False)
        X_sub = X_full[idx]
        y_true = y_full[idx]

        pca = PCA(n_components=PCA_DIMS, random_state=seed)
        X_pca = np.ascontiguousarray(pca.fit_transform(X_sub), dtype=np.float32)
        
        datasets.append((seed, X_pca, y_true))

    records_ours = {m: {v: [] for v in M_VORONOI_RANGE} for m in methods_to_run}
    records_sklearn = []

    try:
        print("\n=== Running Sklearn (Best Params) ===")
        for run_idx, (seed, X_pca, y_true) in enumerate(datasets):
            t0 = time.time()
            sk_model = SpectralClustering(**{**SKLEARN_PARAMS, "random_state": seed})
            labels = sk_model.fit_predict(X_pca)
            elapsed = time.time() - t0
            
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
            records_sklearn.append((ari, nmi, elapsed))
            print(f"  run {run_idx + 1:2d}/{N_RUNS} | Sklearn ARI={ari:.3f} NMI={nmi:.3f} t={elapsed:.2f}s")
            
        for m in methods_to_run:
            print(f"\n=== Running {m} ===")
            for v in M_VORONOI_RANGE:
                aris, nmis, times = [], [], []
                for run_idx, (seed, X_pca, y_true) in enumerate(datasets):
                    if m == "author":
                        model = SpectralBridges(
                            n_clusters=N_CLUSTERS,
                            n_nodes=v,
                            perplexity=PERPLEXITY,
                            n_iter=N_ITER,
                            random_state=seed,
                        )
                        t0 = time.time()
                        model.fit(X_pca)
                        elapsed = time.time() - t0
                        labels = model.labels_
                    else:
                        use_gpu = (m == "gpu")
                        model = specbridge.SpectralClustering(
                            n_clusters=N_CLUSTERS,
                            num_voronoi=v,
                            n_iter=N_ITER,
                            target_perplexity=PERPLEXITY,
                            random_state=seed,
                            use_gpu=use_gpu,
                        )
                        t0 = time.time()
                        result = model.fit(X_pca)
                        elapsed = time.time() - t0
                        labels = np.array(result.labels)
                    
                    ari = adjusted_rand_score(y_true, labels)
                    nmi = normalized_mutual_info_score(y_true, labels)
                    records_ours[m][v].append((ari, nmi, elapsed))
                    aris.append(ari)
                    nmis.append(nmi)
                    times.append(elapsed)
                print(f"  v={v:2d} | avg ARI={np.mean(aris):.3f} NMI={np.mean(nmis):.3f} t={np.mean(times):.2f}s")

    except Exception as e:
        print(f"Exception: {e}")

    # Plotting
    sk_ari_mean = np.mean([r[0] for r in records_sklearn])
    sk_ari_std = np.std([r[0] for r in records_sklearn], ddof=1) if N_RUNS > 1 else 0
    sk_nmi_mean = np.mean([r[1] for r in records_sklearn])
    sk_nmi_std = np.std([r[1] for r in records_sklearn], ddof=1) if N_RUNS > 1 else 0
    sk_time_mean = np.mean([r[2] for r in records_sklearn])
    sk_time_std = np.std([r[2] for r in records_sklearn], ddof=1) if N_RUNS > 1 else 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color palette
    colors = {"cpu": "#5a3a8a", "gpu": "#1b9e77", "author": "#d95f02"}

    # ARI Plot
    ax = axes[0]
    ax.axhline(sk_ari_mean, color="#e7298a", linestyle="--", linewidth=2, label=f"Sklearn Best (ARI={sk_ari_mean:.3f})")
    if sk_ari_std > 0:
        ax.fill_between(M_VORONOI_RANGE, sk_ari_mean - sk_ari_std, sk_ari_mean + sk_ari_std, color="#e7298a", alpha=0.1)

    for m in methods_to_run:
        m_label = f"Ours ({m})" if m != "author" else "Author (Python)"
        ari_means = [np.mean([r[0] for r in records_ours[m][v]]) for v in M_VORONOI_RANGE]
        ari_stds = [np.std([r[0] for r in records_ours[m][v]], ddof=1) if N_RUNS > 1 else 0 for v in M_VORONOI_RANGE]
        
        ax.plot(M_VORONOI_RANGE, ari_means, marker="o", markersize=4, label=m_label, color=colors.get(m, "blue"))
        # if np.any(np.array(ari_stds) > 0):
        #     ax.fill_between(M_VORONOI_RANGE, np.array(ari_means) - np.array(ari_stds), np.array(ari_means) + np.array(ari_stds), alpha=0.2, color=colors.get(m, "blue"))

    ax.set_xlabel("num_voronoi")
    ax.set_ylabel("ARI Score")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("ARI vs num_voronoi")

    # NMI Plot
    ax = axes[1]
    ax.axhline(sk_nmi_mean, color="#e7298a", linestyle="--", linewidth=2, label=f"Sklearn Best (NMI={sk_nmi_mean:.3f})")
    if sk_nmi_std > 0:
        ax.fill_between(M_VORONOI_RANGE, sk_nmi_mean - sk_nmi_std, sk_nmi_mean + sk_nmi_std, color="#e7298a", alpha=0.1)

    for m in methods_to_run:
        m_label = f"Ours ({m})" if m != "author" else "Author (Python)"
        nmi_means = [np.mean([r[1] for r in records_ours[m][v]]) for v in M_VORONOI_RANGE]
        nmi_stds = [np.std([r[1] for r in records_ours[m][v]], ddof=1) if N_RUNS > 1 else 0 for v in M_VORONOI_RANGE]
        
        ax.plot(M_VORONOI_RANGE, nmi_means, marker="o", markersize=4, label=m_label, color=colors.get(m, "blue"))
        # if np.any(np.array(nmi_stds) > 0):
        #     ax.fill_between(M_VORONOI_RANGE, np.array(nmi_means) - np.array(nmi_stds), np.array(nmi_means) + np.array(nmi_stds), alpha=0.2, color=colors.get(m, "blue"))

    ax.set_xlabel("num_voronoi")
    ax.set_ylabel("NMI Score")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("NMI vs num_voronoi")

    # Time Plot
    ax = axes[2]
    ax.axhline(sk_time_mean, color="#e7298a", linestyle="--", linewidth=2, label=f"Sklearn Best (t={sk_time_mean:.2f}s)")
    if sk_time_std > 0:
        ax.fill_between(M_VORONOI_RANGE, sk_time_mean - sk_time_std, sk_time_mean + sk_time_std, color="#e7298a", alpha=0.1)

    for m in methods_to_run:
        m_label = f"Ours ({m})" if m != "author" else "Author (Python)"
        time_means = [np.mean([r[2] for r in records_ours[m][v]]) for v in M_VORONOI_RANGE]
        time_stds = [np.std([r[2] for r in records_ours[m][v]], ddof=1) if N_RUNS > 1 else 0 for v in M_VORONOI_RANGE]
        
        ax.plot(M_VORONOI_RANGE, time_means, marker="o", markersize=4, label=m_label, color=colors.get(m, "blue"))
        if np.any(np.array(time_stds) > 0):
            ax.fill_between(M_VORONOI_RANGE, np.array(time_means) - np.array(time_stds), np.array(time_means) + np.array(time_stds), alpha=0.2, color=colors.get(m, "blue"))

    ax.set_xlabel("num_voronoi")
    ax.set_ylabel("Time (seconds)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Execution Time vs num_voronoi")

    fig.suptitle(f"Sweeping num_voronoi (1-50) vs Sklearn Best\nPCA=49, perplexity=2.0 ({N_RUNS} runs, n={N_SAMPLES})", fontsize=14)
    fig.tight_layout()
    
    plot_path = f"sweep_voronoi_{mode_name}.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")

if __name__ == "__main__":
    main()
