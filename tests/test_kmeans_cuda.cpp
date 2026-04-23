// tests/test_kmeans_cuda.cpp
// Validates fitKMeansCuda against the CPU KMeans implementation.
//
// We compare *inertia* (within-cluster sum of squares), not labels or
// centroids directly, because:
//   - Cluster IDs are permutation-invariant between runs.
//   - Float reduction order differs between CPU OpenMP and GPU atomics,
//     so centroids drift slightly.
// Inertia is the quantity k-means minimizes, so similar inertia means
// similar-quality clusterings.

#include "kmeans.hpp"
#include "kmeans_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---------- helpers ----------

static Matrix makeSyntheticData(int n, int d, int n_true_clusters,
                                uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    std::uniform_real_distribution<float> center_dist(-5.0f, 5.0f);

    Matrix centers(n_true_clusters * d);
    for (int c = 0; c < n_true_clusters; c++)
        for (int k = 0; k < d; k++)
            centers[c * d + k] = center_dist(rng);

    Matrix X(n * d);
    std::uniform_int_distribution<int> assign(0, n_true_clusters - 1);
    for (int i = 0; i < n; i++) {
        int c = assign(rng);
        for (int k = 0; k < d; k++)
            X[i * d + k] = centers[c * d + k] + noise(rng);
    }
    return X;
}

static double computeInertia(const Matrix& X, const Matrix& centroids,
                             const std::vector<int>& labels,
                             int n, int d)
{
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        int c = labels[i];
        double sq = 0.0;
        for (int k = 0; k < d; k++) {
            double diff = (double)X[i * d + k] - (double)centroids[c * d + k];
            sq += diff * diff;
        }
        total += sq;
    }
    return total;
}

static std::vector<int> sortedClusterSizes(const std::vector<int>& labels,
                                           int n_clusters)
{
    std::vector<int> sizes(n_clusters, 0);
    for (int l : labels) sizes[l]++;
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

// ---------- test cases ----------

struct TestCase {
    const char* name;
    int n, d, n_clusters, n_iter;
    uint64_t seed;
    double inertia_rel_tol;
};

static int runTest(const TestCase& tc) {
    printf("--- %s ---\n", tc.name);
    printf("    n=%d d=%d clusters=%d iter=%d seed=%lu\n",
           tc.n, tc.d, tc.n_clusters, tc.n_iter,
           (unsigned long)tc.seed);

    Matrix X = makeSyntheticData(tc.n, tc.d, tc.n_clusters, tc.seed);

    // --- CPU run ---
    auto t0 = std::chrono::steady_clock::now();
    KMeans km_cpu(tc.n_clusters, tc.n_iter, -1, tc.seed);
    KMeansResult r_cpu = km_cpu.fit(X, tc.n, tc.d);
    auto t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double inertia_cpu = computeInertia(X, r_cpu.centroids, r_cpu.labels,
                                        tc.n, tc.d);

    // --- GPU run ---
    auto t2 = std::chrono::steady_clock::now();
    KMeansResult r_gpu = fitKMeansCuda(X, tc.n, tc.d,
                                       tc.n_clusters, tc.n_iter, tc.seed);
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double inertia_gpu = computeInertia(X, r_gpu.centroids, r_gpu.labels,
                                        tc.n, tc.d);

    // --- compare ---
    double rel_diff = std::abs(inertia_cpu - inertia_gpu) /
                      std::max(inertia_cpu, 1e-12);

    printf("    CPU: inertia=%.4f  time=%.2f ms\n", inertia_cpu, cpu_ms);
    printf("    GPU: inertia=%.4f  time=%.2f ms\n", inertia_gpu, gpu_ms);
    printf("    relative inertia diff: %.4e (tol %.1e)\n",
           rel_diff, tc.inertia_rel_tol);
    printf("    speedup: %.2fx\n", cpu_ms / std::max(gpu_ms, 1e-6));

    auto cpu_sizes = sortedClusterSizes(r_cpu.labels, tc.n_clusters);
    auto gpu_sizes = sortedClusterSizes(r_gpu.labels, tc.n_clusters);
    if (cpu_sizes != gpu_sizes) {
        printf("    NOTE: sorted cluster sizes differ\n");
        printf("      CPU: ");
        for (int s : cpu_sizes) printf("%d ", s);
        printf("\n      GPU: ");
        for (int s : gpu_sizes) printf("%d ", s);
        printf("\n");
    }

    bool pass = (rel_diff < tc.inertia_rel_tol);
    printf("    %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    int failures = 0;

    // Tolerance is generous because CPU uses k-means++ init and the current
    // GPU version uses uniform random init — different initializations can
    // converge to different local minima. Tighten once GPU k-means++ is in.
    TestCase cases[] = {
        {"small",     1000,  8,   5, 20, 42, 0.10},
        {"medium",   10000, 16,  10, 20, 42, 0.10},
        {"large",   100000, 32,  50, 20, 42, 0.15},
        {"tall-d",   10000, 64,  10, 20, 42, 0.10},
    };

    for (const auto& tc : cases) failures += runTest(tc);

    printf("========================================\n");
    printf("Total failures: %d / %d\n",
           failures, (int)(sizeof(cases) / sizeof(cases[0])));
    return failures == 0 ? 0 : 1;
}