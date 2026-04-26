#include "spectral.hpp"
#include "kmeans_cuda.hpp"
#include "kmeans.hpp"
#include "affinity_gpu.hpp"
#include "affinity.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <iostream>
#include <iomanip>

/*
X=Matrix
m = num_vornoi
n = num datapoints
d = dimensionality of datapoints
k = num clusters
*/

SpectralClustering::SpectralClustering(
    int n_clusters, int num_vornoi, int n_iter, float target_perplexity, uint64_t random_state, bool use_gpu)
    : n_clusters(n_clusters), n_iter(n_iter), num_vornoi(num_vornoi), random_state(random_state), target_perplexity(target_perplexity), use_gpu(use_gpu){}

SBResult SpectralClustering::fit(
    const Matrix& X,
    int n,
    int d)
{
    // Delegate to the free function; core spectral logic lives there.
    return spectralBridges(X, n, d, n_clusters, num_vornoi, target_perplexity, n_iter, random_state, use_gpu);
}


// Helper lambda to make printing cleaner
auto print_duration = [](const std::string& name, std::chrono::duration<double> duration) {
    std::cout << std::fixed << std::setprecision(4) 
              << "  [Profile] " << name << ": " << duration.count() << " s\n";
};

SpectralResult spectralClustering(
    const MatrixD& affinity,
    int m, int k,
    uint64_t random_state,
    bool use_gpu)
{
    auto start_all = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------
    // Phase 3.1: Laplacian Construction
    // ---------------------------------------------------------
    auto start_laplacian = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd A(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            A(i, j) = affinity[i * m + j];

    Eigen::VectorXd d_vec(m);
    for (int i = 0; i < m; i++) {
        double row_mean = A.row(i).mean();
        if (row_mean <= 0.0) {
            d_vec(i) = 0.0;
        } else {
            d_vec(i) = std::pow(row_mean, -0.5);
        }
    }

    Eigen::MatrixXd L = -(d_vec.asDiagonal() * A * d_vec.asDiagonal());
    double tol = 1e-8;
    for (int i = 0; i < m; i++) {
        L(i, i) = static_cast<double>(m) + tol;
    }
    auto end_laplacian = std::chrono::high_resolution_clock::now();
    print_duration("    -> Laplacian Setup", end_laplacian - start_laplacian);

    // ---------------------------------------------------------
    // Phase 3.2: Eigen Decomposition
    // ---------------------------------------------------------
    auto start_eigen = std::chrono::high_resolution_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed");

    Eigen::VectorXd eigvals = solver.eigenvalues();
    Eigen::MatrixXd eigvecs = solver.eigenvectors();
    auto end_eigen = std::chrono::high_resolution_clock::now();
    print_duration("    -> Eigen Decomposition", end_eigen - start_eigen);

    // ---------------------------------------------------------
    // Phase 3.3: Eigenvector Extraction & Normalization
    // ---------------------------------------------------------
    Eigen::MatrixXd U = eigvecs.leftCols(k); 
    for (int i = 0; i < m; i++) {
        double norm = U.row(i).norm();
        if (norm > 1e-10)
            U.row(i) /= norm;
    }

    float ngap = 0.0f;
    if (k < m) {
        double lk   = eigvals(k);
        double lkm1 = eigvals(k - 1);
        ngap = (std::abs(lkm1) > 1e-10) ? static_cast<float>((lk - lkm1) / lkm1) : 0.0f;
    }

    std::vector<float> eigvals_vec(m);
    for (int i = 0; i < m; i++)
        eigvals_vec[i] = static_cast<float>(eigvals(i));

    // ---------------------------------------------------------
    // Phase 3.4: Downstream K-Means
    // ---------------------------------------------------------
    auto start_km2 = std::chrono::high_resolution_clock::now();
    Matrix U_flat(m * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U_flat[i * k + j] = static_cast<float>(U(i, j));

    KMeans km(k, 20, -1, random_state);
    auto kmResult = km.fit(U_flat, m, k);
    auto end_km2 = std::chrono::high_resolution_clock::now();
    print_duration("    -> Spectral K-Means", end_km2 - start_km2);

    auto end_all = std::chrono::high_resolution_clock::now();
    print_duration("  Total SpectralClustering Phase", end_all - start_all);

    return {kmResult.labels, eigvals_vec, ngap};
}

SBResult spectralBridges(
    const Matrix& X,
    int n, int d,
    int k, int m,
    float target_perplexity,
    int n_iter,
    uint64_t random_state,
    bool use_gpu)
{
    std::cout << "\n=== Starting SpectralBridges Pipeline ===\n";

    // ---------------------------------------------------------
    // Step 1: Initial K-Means (Vector Quantization)
    // ---------------------------------------------------------
    auto start_km = std::chrono::high_resolution_clock::now();
    // KMeans km(m, n_iter, -1, random_state);
    KMeansResult kmResult;
    if(use_gpu) {
        kmResult = fitKMeansCuda(X, n, d, m, n_iter, random_state);
    }
    else{
        KMeans km(m, n_iter, -1, random_state);
        kmResult = km.fit(X, n, d);
    }

    auto end_km = std::chrono::high_resolution_clock::now();
    print_duration("Step 1: Initial K-Means (VQ)", end_km - start_km);

    // ---------------------------------------------------------
    // Step 2: Affinity Computation
    // ---------------------------------------------------------
    auto start_aff = std::chrono::high_resolution_clock::now();
    MatrixD aff;
    if(use_gpu) {
        aff = computeAffinityGPU(X, kmResult, n, m, d, target_perplexity);
    }
    else{
        aff = computeAffinity(X, kmResult, n, m, d, target_perplexity);
    }
    auto end_aff = std::chrono::high_resolution_clock::now();
    print_duration(use_gpu ? "Step 2: Affinity (GPU)" : "Step 2: Affinity (CPU)", end_aff - start_aff);

    // ---------------------------------------------------------
    // Step 3: Spectral Clustering Core
    // ---------------------------------------------------------
    std::cout << "  [Profile] Entering Step 3: Spectral Core...\n";
    auto start_sc = std::chrono::high_resolution_clock::now();
    SpectralResult sc = spectralClustering(aff, m, k, random_state, use_gpu);
    auto end_sc = std::chrono::high_resolution_clock::now();
    print_duration("Step 3: Total Spectral Core", end_sc - start_sc);

    // ---------------------------------------------------------
    // Step 4: Label Propagation
    // ---------------------------------------------------------
    auto start_prop = std::chrono::high_resolution_clock::now();
    std::vector<int> pointLabels(n);
    for (int i = 0; i < n; i++)
        pointLabels[i] = sc.labels[kmResult.labels[i]];

    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; i++)
        clusters[pointLabels[i]].push_back(i);
    auto end_prop = std::chrono::high_resolution_clock::now();
    print_duration("Step 4: Label Propagation", end_prop - start_prop);

    std::cout << "===========================================\n\n";

    return {clusters, pointLabels, sc.eigvals, sc.ngap};
}