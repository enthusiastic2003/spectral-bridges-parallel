"""Three-way k-means comparison: CPU, GPU, and sklearn.

Runs the four test configurations from tests/test_kmeans_cuda.cpp and reports
raw correctness and timing metrics for each — no pass/fail.

Correctness metrics (sklearn is the reference):
  - Inertia (within-cluster sum of squares), and Δ% vs sklearn.
  - ARI (Adjusted Rand Index) — permutation-invariant agreement on partitions.

Inertia is computed using each implementation's own returned centroids in
float64, matching the C++ test's convention.
"""

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import adjusted_rand_score

sys.path.append(os.path.abspath("build"))
sys.path.append(os.path.abspath("bin"))

import specbridge as sb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_data(n: int, d: int, n_true_clusters: int, seed: int) -> np.ndarray:
    """Match makeSyntheticData(): centers ~ U(-5, 5)^d, noise ~ N(0, 0.3),
    each of n points assigned to a uniformly random center."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5.0, 5.0, size=(n_true_clusters, d)).astype(np.float32)
    assignments = rng.integers(0, n_true_clusters, size=n)
    noise = rng.normal(0.0, 0.3, size=(n, d)).astype(np.float32)
    return centers[assignments] + noise


def compute_inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """Σ ||x_i - centroids[labels[i]]||² with float64 accumulation."""
    X64 = X.astype(np.float64, copy=False)
    C64 = centroids.astype(np.float64, copy=False)
    diff = X64 - C64[labels]
    return float(np.einsum("ij,ij->", diff, diff))


# ---------------------------------------------------------------------------
# Test cases — copied from the C++ file
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    n: int
    d: int
    n_clusters: int
    n_iter: int
    seed: int


CASES = [
    TestCase("small",    1_000,   8,   5, 20, 42),
    TestCase("medium",  10_000,  16,  10, 20, 42),
    TestCase("large",  10_000_000,  32,  50, 20, 42),
    TestCase("tall-d",  10_000,  64,  10, 20, 42),
]


# ---------------------------------------------------------------------------
# Runners — each returns (labels, centroids, time_ms)
# ---------------------------------------------------------------------------

def run_cpu(X, tc):
    t0 = time.perf_counter()
    model = sb.KMeans(n_clusters=tc.n_clusters, n_iter=tc.n_iter, random_state=tc.seed)
    r = model.fit(X)
    t_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(r.labels), np.asarray(r.centroids, dtype=np.float32), t_ms


def run_gpu(X, tc):
    t0 = time.perf_counter()
    r = sb.fit_kmeans_cuda(X, n_clusters=tc.n_clusters, n_iter=tc.n_iter, random_state=tc.seed)
    t_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(r.labels), np.asarray(r.centroids, dtype=np.float32), t_ms


def run_sklearn(X, tc):
    t0 = time.perf_counter()
    model = SKKMeans(
        n_clusters=tc.n_clusters,
        n_init=1,
        max_iter=tc.n_iter,
        algorithm="lloyd",
        init="k-means++",
        tol=0.0,
        random_state=tc.seed,
    )
    model.fit(X)
    t_ms = (time.perf_counter() - t0) * 1000.0
    return model.labels_, model.cluster_centers_.astype(np.float32), t_ms


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_ms(t: float) -> str:
    return f"{t:.1f} ms" if t < 1000 else f"{t/1000:.3f} s"


def report_case(tc: TestCase) -> None:
    print(f"--- {tc.name} ---")
    print(f"    n={tc.n} d={tc.d} clusters={tc.n_clusters} "
          f"iter={tc.n_iter} seed={tc.seed}")

    X = make_synthetic_data(tc.n, tc.d, tc.n_clusters, tc.seed)

    sk_labels, sk_centroids, sk_ms = run_sklearn(X, tc)
    cpu_labels, cpu_centroids, cpu_ms = run_cpu(X, tc)
    gpu_labels, gpu_centroids, gpu_ms = run_gpu(X, tc)

    sk_inertia = compute_inertia(X, sk_centroids, sk_labels)
    cpu_inertia = compute_inertia(X, cpu_centroids, cpu_labels)
    gpu_inertia = compute_inertia(X, gpu_centroids, gpu_labels)

    rows = [
        ("sklearn", sk_ms,  sk_inertia,  0.0,
                                          1.0),
        ("cpu",     cpu_ms, cpu_inertia, 100.0 * (cpu_inertia - sk_inertia) / sk_inertia,
                                          adjusted_rand_score(sk_labels, cpu_labels)),
        ("gpu",     gpu_ms, gpu_inertia, 100.0 * (gpu_inertia - sk_inertia) / sk_inertia,
                                          adjusted_rand_score(sk_labels, gpu_labels)),
    ]

    print(f"    {'Impl':<8} {'Time':>10} {'Speedup':>9} "
          f"{'Inertia':>14} {'ΔInertia%':>11} {'ARI vs sk':>11}")
    for name, t_ms, inertia, delta_pct, ari in rows:
        speedup = sk_ms / max(t_ms, 1e-6)
        print(f"    {name:<8} {fmt_ms(t_ms):>10} {speedup:>8.2f}x "
              f"{inertia:>14.4e} {delta_pct:>10.4f}% {ari:>11.6f}")
    print()


def main():
    print("Three-way k-means comparison (CPU vs GPU vs sklearn)")
    print(f"omp_max={sb.get_max_threads()} | procs={sb.get_num_procs()}")
    print()
    for tc in CASES:
        report_case(tc)


if __name__ == "__main__":
    main()