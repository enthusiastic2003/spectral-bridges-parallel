"""Diagnostic for the GPU k-means convergence issue.

Hypotheses to discriminate:
  H1: GPU isn't running n_iter iterations (off-by-one, early stop).
  H2: GPU has a correctness bug in assign or accumulate at k>4.
  H3: Random init from raw points is just catastrophic at higher d/k.

Run with the 'medium' config and sweep n_iter. Also run sklearn with the
same n_iter and *random* init (not k-means++) for an apples-to-apples
upper bound on what random-init Lloyd's should achieve.
"""

import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans as SKKMeans

sys.path.append(os.path.abspath("build"))
sys.path.append(os.path.abspath("bin"))
import specbridge as sb


def make_data(n, d, k, seed):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-5.0, 5.0, size=(k, d)).astype(np.float32)
    assignments = rng.integers(0, k, size=n)
    noise = rng.normal(0.0, 0.3, size=(n, d)).astype(np.float32)
    return centers[assignments] + noise


def inertia(X, centroids, labels):
    diff = X.astype(np.float64) - centroids.astype(np.float64)[labels]
    return float(np.einsum("ij,ij->", diff, diff))


def main():
    # medium config from C++ test
    n, d, k, seed = 10_000, 16, 10, 42
    X = make_data(n, d, k, seed)

    print(f"Data: n={n} d={d} k={k} seed={seed}")
    print()

    # --- Reference: sklearn with k-means++ init (best case) ---
    sk_kpp = SKKMeans(n_clusters=k, n_init=1, max_iter=300, init="k-means++",
                      algorithm="lloyd", tol=0.0, random_state=seed).fit(X)
    inertia_kpp = inertia(X, sk_kpp.cluster_centers_, sk_kpp.labels_)
    print(f"sklearn (k-means++, 300 iter): inertia = {inertia_kpp:.4e}")

    # --- Reference: sklearn with RANDOM init, same n_iter sweep ---
    # This is the apples-to-apples baseline for what GPU's random-init
    # Lloyd's should achieve, isolating init-quality from algorithm correctness.
    print()
    print("sklearn with RANDOM init (Lloyd's only), varying max_iter:")
    print(f"  {'iter':>6}  {'inertia':>14}")
    for it in [1, 2, 5, 10, 20, 50, 100]:
        sk = SKKMeans(n_clusters=k, n_init=1, max_iter=it, init="random",
                      algorithm="lloyd", tol=0.0, random_state=seed).fit(X)
        ine = inertia(X, sk.cluster_centers_, sk.labels_)
        print(f"  {it:>6}  {ine:>14.4e}")

    # --- GPU: same sweep ---
    print()
    print("GPU (specbridge), varying n_iter:")
    print(f"  {'iter':>6}  {'inertia':>14}  {'time (ms)':>10}")
    for it in [1, 2, 5, 10, 20, 50, 100]:
        t0 = time.perf_counter()
        r = sb.fit_kmeans_cuda(X, n_clusters=k, n_iter=it, random_state=seed)
        t_ms = (time.perf_counter() - t0) * 1000
        labels = np.asarray(r.labels)
        centroids = np.asarray(r.centroids, dtype=np.float32)
        ine = inertia(X, centroids, labels)
        # also check: are all clusters non-empty?
        sizes = np.bincount(labels, minlength=k)
        empty = int((sizes == 0).sum())
        # and: are any centroids exactly zero (suggesting they were never updated)?
        zero_rows = int(np.all(centroids == 0, axis=1).sum())
        print(f"  {it:>6}  {ine:>14.4e}  {t_ms:>10.1f}   "
              f"empty_clusters={empty}  zero_centroids={zero_rows}  "
              f"sizes={sizes.tolist()}")

    # --- GPU: check determinism ---
    print()
    print("GPU determinism check (same seed, 3 runs at n_iter=20):")
    for trial in range(3):
        r = sb.fit_kmeans_cuda(X, n_clusters=k, n_iter=20, random_state=seed)
        ine = inertia(X, np.asarray(r.centroids, dtype=np.float32), np.asarray(r.labels))
        print(f"  trial {trial}: inertia = {ine:.4e}")


if __name__ == "__main__":
    main()