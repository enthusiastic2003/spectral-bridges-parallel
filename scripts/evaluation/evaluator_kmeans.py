import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# Add common output folders for the compiled Python extension.
sys.path.append(os.path.abspath("build"))
sys.path.append(os.path.abspath("bin"))

import specbridge as sb


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_data(points_per_cluster: int, cluster_std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-6.0, -6.0], [6.0, -6.0], [-6.0, 6.0], [6.0, 6.0]],
        dtype=np.float32,
    )
    chunks = [
        rng.normal(loc=c, scale=cluster_std, size=(points_per_cluster, 2)).astype(np.float32)
        for c in centers
    ]
    return np.vstack(chunks)


# ---------------------------------------------------------------------------
# Runners — each returns (result, times) where result has .labels and
# optionally .centers
# ---------------------------------------------------------------------------

def run_cpu(X, n_clusters, n_iter, random_state, threads, runs):
    sb.set_num_threads(threads)
    model = sb.KMeans(n_clusters=n_clusters, n_iter=n_iter, random_state=random_state)
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = model.fit(X)
        times.append(time.perf_counter() - t0)
    return result, times


def run_gpu(X, n_clusters, n_iter, random_state, runs):
    times, result = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = sb.fit_kmeans_cuda(
            X, n_clusters=n_clusters, n_iter=n_iter, random_state=random_state,
        )
        times.append(time.perf_counter() - t0)
    return result, times


def run_sklearn(X, n_clusters, n_iter, random_state, runs):
    model = SKKMeans(
        n_clusters=n_clusters,
        n_init=1,
        max_iter=n_iter,
        algorithm="lloyd",
        init="k-means++",
        tol=0.0,
        random_state=random_state,
    )
    times, fitted = [], None
    for _ in range(runs):
        t0 = time.perf_counter()
        fitted = model.fit(X)
        times.append(time.perf_counter() - t0)
    return (
        SimpleNamespace(labels=fitted.labels_, centers=fitted.cluster_centers_),
        times,
    )


# ---------------------------------------------------------------------------
# Correctness metrics
# ---------------------------------------------------------------------------

def compute_inertia(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> float:
    """Within-cluster sum of squared distances. Recomputed independently so we
    don't have to trust each implementation to report it."""
    labels = np.asarray(labels)
    inertia = 0.0
    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            continue
        pts = X[mask]
        center = pts.mean(axis=0)
        diff = pts - center
        inertia += float(np.einsum("ij,ij->", diff, diff))
    return inertia


def compute_centers(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            centers[k] = X[mask].mean(axis=0)
    return centers


def matched_centroid_distance(c_a: np.ndarray, c_b: np.ndarray) -> float:
    """Max Euclidean distance between centroids after optimal Hungarian matching."""
    # Cost matrix of pairwise distances, then optimal assignment.
    diff = c_a[:, None, :] - c_b[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    row_ind, col_ind = linear_sum_assignment(dist)
    return float(dist[row_ind, col_ind].max())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_time(t: float) -> str:
    return f"{t*1000:.1f} ms" if t < 1 else f"{t:.3f} s"


def summarize_times(name: str, times: list[float]) -> dict:
    arr = np.asarray(times)
    return {
        "name": name,
        "min": arr.min(),
        "mean": arr.mean(),
        "std": arr.std(),
        "all": arr.tolist(),
    }


def print_timing_table(rows: list[dict], baseline_min: float):
    print()
    print(f"{'Implementation':<12} {'Min':>10} {'Mean':>10} {'Std':>10} {'Speedup vs sklearn':>22}")
    print("-" * 68)
    for r in rows:
        speedup = baseline_min / r["min"]
        print(
            f"{r['name']:<12} {fmt_time(r['min']):>10} {fmt_time(r['mean']):>10} "
            f"{fmt_time(r['std']):>10} {speedup:>20.2f}x"
        )


def print_correctness_table(rows: list[dict]):
    print()
    print(
        f"{'Implementation':<12} {'Inertia':>16} {'Δ Inertia %':>14} "
        f"{'ARI vs sklearn':>16} {'Max centroid Δ':>18}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r['name']:<12} {r['inertia']:>16.4e} {r['delta_inertia_pct']:>13.4f}% "
            f"{r['ari']:>16.6f} {r['centroid_dist']:>18.4e}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU and GPU k-means against the sklearn baseline."
    )
    parser.add_argument(
        "--impls", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"],
        help="Which custom implementations to compare (sklearn baseline always runs).",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8, help="Only used for cpu impl")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--points-per-cluster", type=int, default=2_000_000)
    parser.add_argument("--cluster-std", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X = make_data(args.points_per_cluster, args.cluster_std, args.seed)
    print(f"Dataset: X.shape={X.shape}, dtype={X.dtype}")
    print(f"Config:  n_clusters={args.n_clusters}, n_iter={args.n_iter}, runs={args.runs}, seed={args.seed}")

    # --- Baseline always runs first ---------------------------------------
    print("\n[1/?] sklearn baseline...")
    sk_result, sk_times = run_sklearn(
        X, args.n_clusters, args.n_iter, args.seed, args.runs,
    )
    sk_labels = np.asarray(sk_result.labels)
    sk_centers = compute_centers(X, sk_labels, args.n_clusters)
    sk_inertia = compute_inertia(X, sk_labels, args.n_clusters)

    timing_rows = [summarize_times("sklearn", sk_times)]
    correctness_rows = [{
        "name": "sklearn",
        "inertia": sk_inertia,
        "delta_inertia_pct": 0.0,
        "ari": 1.0,
        "centroid_dist": 0.0,
    }]

    # --- Custom implementations -------------------------------------------
    runners = {
        "cpu": lambda: run_cpu(X, args.n_clusters, args.n_iter, args.seed, args.threads, args.runs),
        "gpu": lambda: run_gpu(X, args.n_clusters, args.n_iter, args.seed, args.runs),
    }

    for i, impl in enumerate(args.impls, start=2):
        print(f"\n[{i}/{len(args.impls)+1}] {impl} implementation...")
        if impl == "cpu":
            print(f"   threads={args.threads} | omp_max={sb.get_max_threads()} | procs={sb.get_num_procs()}")
        result, times = runners[impl]()
        labels = np.asarray(result.labels)
        centers = compute_centers(X, labels, args.n_clusters)
        inertia = compute_inertia(X, labels, args.n_clusters)

        timing_rows.append(summarize_times(impl, times))
        correctness_rows.append({
            "name": impl,
            "inertia": inertia,
            "delta_inertia_pct": 100.0 * (inertia - sk_inertia) / sk_inertia,
            "ari": adjusted_rand_score(sk_labels, labels),
            "centroid_dist": matched_centroid_distance(sk_centers, centers),
        })

    # --- Report -----------------------------------------------------------
    print("\n=== Timing ===")
    print_timing_table(timing_rows, baseline_min=timing_rows[0]["min"])

    print("\n=== Correctness (vs sklearn) ===")
    print_correctness_table(correctness_rows)
    print(
        "\nNotes:\n"
        "  • ARI = Adjusted Rand Index (1.0 = identical partitions, permutation-invariant).\n"
        "  • Δ Inertia % = relative gap to sklearn's objective. Negative means tighter clusters.\n"
        "  • Max centroid Δ = worst centroid distance after Hungarian matching.\n"
        "  • Speedup uses min times (less noisy than mean for short runs)."
    )


if __name__ == "__main__":
    main()