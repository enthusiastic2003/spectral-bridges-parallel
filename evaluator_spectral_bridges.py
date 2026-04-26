import sys
import os
import time
import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 1. Add the build folder to Python's search path
sys.path.append(os.path.abspath("build/"))

import specbridge as sb
from sbcluster import SpectralBridges


def generate_synthetic_data(points_per_cluster=5_000_000, cluster_std=0.8, random_state=42):
    rng = np.random.default_rng(random_state)
    centers = np.array([
        [-6.0, -6.0],
        [ 6.0, -6.0],
        [-6.0,  6.0],
        [ 6.0,  6.0],
    ])

    X_parts = []
    y_parts = []
    for i, c in enumerate(centers):
        X_parts.append(rng.normal(loc=c, scale=cluster_std, size=(points_per_cluster, 2)))
        y_parts.append(np.full(points_per_cluster, i, dtype=int))

    X = np.vstack(X_parts).astype(np.float32)
    y_true = np.concatenate(y_parts)
    return X, y_true


def run_benchmarks(X, n_repeats=3, n_clusters=4, skip_baseline=False):
    print("-" * 50)

    y_baseline = None

    if not skip_baseline:
        print("=== Baseline: Author's Implementation (actual_sb) ===")

        actual_sb = SpectralBridges(
            n_clusters=n_clusters,
            n_nodes=200,
            random_state=42,
            n_iter=20
        )

        start_time = time.time()
        y_baseline = actual_sb.fit_predict(X)
        baseline_time = time.time() - start_time

        print(f"Baseline completed in {baseline_time:.4f} seconds.")
        print("-" * 50)

    experiments = [
        {"name": "CPU", "use_gpu": False},
        {"name": "GPU", "use_gpu": True}
    ]

    for exp in experiments:
        print(f"=== Experiment: specbridge [{exp['name']}] ===")
        print(f"Running with {sb.get_max_threads()} threads; {sb.get_num_procs()} processors")
        print("GPU enabled:", exp["use_gpu"])

        bridge = sb.SpectralClustering(
            n_clusters=n_clusters,
            num_voronoi=200,
            n_iter=20,
            target_perplexity=2.0,
            random_state=42,
            use_gpu=exp["use_gpu"]
        )

        times, aris, nmis = [], [], []

        for i in range(n_repeats):
            print(f"  Run {i+1}/{n_repeats}...", end=" ", flush=True)

            start_time = time.time()
            result = bridge.fit(X)
            run_time = time.time() - start_time

            times.append(run_time)

            if not skip_baseline:
                y_pred = result.labels
                ari = adjusted_rand_score(y_baseline, y_pred)
                nmi = normalized_mutual_info_score(y_baseline, y_pred)

                aris.append(ari)
                nmis.append(nmi)

                print(f"Time: {run_time:.4f}s | ARI: {ari:.4f} | NMI: {nmi:.4f}")
            else:
                print(f"Time: {run_time:.4f}s")

        print(f"\n  -> {exp['name']} Summary ({n_repeats} runs):")
        print(f"     Average Time: {np.mean(times):.4f}s ± {np.std(times):.4f}s")

        if not skip_baseline:
            print(f"     Average ARI:  {np.mean(aris):.4f}")
            print(f"     Average NMI:  {np.mean(nmis):.4f}")

        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Disable author's implementation and metric computation")
    parser.add_argument("--points-per-cluster", type=int, default=5_000_000)
    parser.add_argument("--repeats", type=int, default=5)

    args = parser.parse_args()

    print("Generating synthetic dataset...")
    X, y_true = generate_synthetic_data(points_per_cluster=args.points_per_cluster)
    print(f"Dataset Shape: {X.shape} | Dtype: {X.dtype}")

    run_benchmarks(
        X,
        n_repeats=args.repeats,
        skip_baseline=args.skip_baseline
    )