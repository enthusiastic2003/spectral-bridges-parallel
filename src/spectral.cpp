#include "spectral.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

SpectralClustering::SpectralClustering(int n_clusters, int n_iter, uint64_t random_state)
    : n_clusters(n_clusters), n_iter(n_iter), random_state(random_state) {}

SpectralResult SpectralClustering::fit(
    const Matrix& affinity,
    int m) const
{
    // Delegate to the free function; core spectral logic lives there.
    return spectralClustering(affinity, m, n_clusters, random_state);
}

SpectralResult spectralClustering(
    const Matrix& affinity,
    int m, int k,
    uint64_t random_state)
{
    // Step 1 — build normalized Laplacian L = D^(-1/2) (D - A) D^(-1/2)
    // where D is the degree matrix (row sums of affinity)
    Eigen::MatrixXf A(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            A(i, j) = affinity[i * m + j];

    // Degree vector
    Eigen::VectorXf deg = A.rowwise().sum();

    // D^(-1/2) — avoid division by zero
    Eigen::VectorXf deg_inv_sqrt(m);
    for (int i = 0; i < m; i++)
        deg_inv_sqrt(i) = (deg(i) > 1e-10f) ? 1.0f / std::sqrt(deg(i)) : 0.0f;

    // L = I - D^(-1/2) A D^(-1/2)
    Eigen::MatrixXf L(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            L(i, j) = (i == j ? 1.0f : 0.0f)
                      - deg_inv_sqrt(i) * A(i, j) * deg_inv_sqrt(j);

    // Step 2 — eigen decomposition (L is symmetric, use SelfAdjointEigenSolver)
    // Returns eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(L);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("Eigen decomposition failed");

    Eigen::VectorXf eigvals = solver.eigenvalues();
    Eigen::MatrixXf eigvecs = solver.eigenvectors(); // columns are eigenvectors

    // Step 3 — take first k eigenvectors, row-normalize
    Eigen::MatrixXf U = eigvecs.leftCols(k); // [m × k]
    for (int i = 0; i < m; i++) {
        float norm = U.row(i).norm();
        if (norm > 1e-10f)
            U.row(i) /= norm;
    }

    // Step 4 — normalized eigengap
    // ngap = (λ[k] - λ[k-1]) / λ[k]
    float ngap = 0.0f;
    if (k < m) {
        float lk   = eigvals(k);
        float lkm1 = eigvals(k - 1);
        ngap = (lk > 1e-10f) ? (lk - lkm1) / lk : 0.0f;
    }

    // Step 5 — k-means on rows of U
    Matrix U_flat(m * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U_flat[i * k + j] = U(i, j);

    KMeans km(k, 20, -1, random_state);
    auto kmResult = km.fit(U_flat, m, k);

    // Collect eigenvalues
    std::vector<float> eigvals_vec(m);
    for (int i = 0; i < m; i++)
        eigvals_vec[i] = eigvals(i);

    return {kmResult.labels, eigvals_vec, ngap};
}

SBResult spectralBridges(
    const Matrix& X,
    int n, int d,
    int k, int m,
    float p, float M,
    int n_iter,
    uint64_t random_state)
{
    // Step 1 — vector quantization
    KMeans km(m, n_iter, -1, random_state);
    auto kmResult = km.fit(X, n, d);

    // Step 2 — affinity matrix
    Matrix aff = computeAffinity(X, kmResult, n, m, d, p, M);

    // Step 3 — spectral clustering on affinity
    SpectralResult sc = spectralClustering(aff, m, k, random_state);

    // Step 4 — propagate region labels back to points
    std::vector<int> pointLabels(n);
    for (int i = 0; i < n; i++)
        pointLabels[i] = sc.labels[kmResult.labels[i]];

    // Group point indices by cluster
    std::vector<std::vector<int>> clusters(k);
    for (int i = 0; i < n; i++)
        clusters[pointLabels[i]].push_back(i);

    return {clusters, pointLabels, sc.eigvals, sc.ngap};
}