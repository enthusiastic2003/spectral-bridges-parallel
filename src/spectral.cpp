#include "spectral.hpp"
// #include "kmeans_cuda.hpp"
#include "affinity_gpu.hpp"
#include "affinity.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

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

SpectralResult spectralClustering(
    const MatrixD& affinity,
    int m, int k,
    uint64_t random_state,
    bool use_gpu)
{
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

    // Step 2 — eigen decomposition (L is symmetric, use SelfAdjointEigenSolver)
    // Returns eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed");

    Eigen::VectorXd eigvals = solver.eigenvalues();
    Eigen::MatrixXd eigvecs = solver.eigenvectors(); // columns are eigenvectors

    // Step 3 — take first k eigenvectors, row-normalize
    Eigen::MatrixXd U = eigvecs.leftCols(k); // [m × k]
    for (int i = 0; i < m; i++) {
        double norm = U.row(i).norm();
        if (norm > 1e-10)
            U.row(i) /= norm;
    }

    // Step 4 — normalized eigengap
    // Python uses: (λ[k] - λ[k-1]) / λ[k-1]
    float ngap = 0.0f;
    if (k < m) {
        double lk   = eigvals(k);
        double lkm1 = eigvals(k - 1);
        ngap = (std::abs(lkm1) > 1e-10) ? static_cast<float>((lk - lkm1) / lkm1) : 0.0f;
    }

    // Collect eigenvalues before downstream k-means to avoid any accidental
    // mutation if later stages are unstable.
    std::vector<float> eigvals_vec(m);
    for (int i = 0; i < m; i++)
        eigvals_vec[i] = static_cast<float>(eigvals(i));

    // Step 5 — k-means on rows of U (use project KMeans implementation).
    Matrix U_flat(m * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U_flat[i * k + j] = static_cast<float>(U(i, j));

    KMeans km(k, 20, -1, random_state);
    auto kmResult = km.fit(U_flat, m, k);

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
    // Step 1 — vector quantization
    KMeans km(m, n_iter, -1, random_state);
    auto kmResult = km.fit(X, n, d);

    // std::cout << "K-means completed. Voronoi centers computed: " << m << std::endl;

    MatrixD aff;
    if(use_gpu) {
        std::cout << "Using GPU for affinity computation." << std::endl;
        // Step 2 — affinity matrix
        aff = computeAffinityGPU(X, kmResult, n, m, d, target_perplexity);
    }
    else{
        std::cout << "Using CPU for affinity computation." << std::endl;
        // Step 2 — affinity matrix
        aff = computeAffinity(X, kmResult, n, m, d, target_perplexity);
    }

    // std::cout << "Affinity matrix computed." << std::endl;

    // Step 3 — spectral clustering on affinity
    SpectralResult sc = spectralClustering(aff, m, k, random_state, use_gpu);

    // std::cout << "Spectral clustering completed. Eigenvalues and labels computed." << std::endl;

    // Step 4 — propagate region labels back to points
    std::vector<int> pointLabels(n);
    for (int i = 0; i < n; i++)
        pointLabels[i] = sc.labels[kmResult.labels[i]];

    // Group point indices by cluster
    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; i++)
        clusters[pointLabels[i]].push_back(i);

    // std::cout << "Point labels propagated back to clusters." << std::endl;
    
    return {clusters, pointLabels, sc.eigvals, sc.ngap};
}