#include "spectral.hpp"
#include "kmeans_cuda.hpp"
#include "kmeans.hpp"
#include "affinity_gpu.hpp"
#include "spectral_cuda.hpp"
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
// Add at the top of the file (with the other includes):
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
//   #include <Spectra/MatOp/DenseSymShiftSolve.h>
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

// Replace these two includes at the top of src/spectral.cpp
// (remove the SymEigsShiftSolver / DenseSymShiftSolve ones if you had them):
//
//   #include <Spectra/SymEigsSolver.h>
//   #include <Spectra/MatOp/DenseSymMatProd.h>


SpectralResult spectralClustering(
    const Matrix& affinity,
    int m, int k,
    int n_iter,
    uint64_t random_state
    )
{
    auto start_all = std::chrono::high_resolution_clock::now();
    std::cout << "Running spectral clustering on CPU with m=" << m << ", k=" << k << "\n";

    // ---------------------------------------------------------
    // Phase 3.1: Laplacian Construction  (unchanged)
    // ---------------------------------------------------------
    auto start_laplacian = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXf A(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            A(i, j) = affinity[i * m + j];

    Eigen::VectorXf d_vec(m);
    for (int i = 0; i < m; i++) {
        float row_mean = A.row(i).mean();
        if (row_mean <= 0.0f) {
            d_vec(i) = 0.0f;
        } else {
            d_vec(i) = std::pow(row_mean, -0.5f);
        }
    }

    Eigen::MatrixXf L = -(d_vec.asDiagonal() * A * d_vec.asDiagonal());
    float tol = 1e-8f;
    for (int i = 0; i < m; i++) {
        L(i, i) = static_cast<float>(m) + tol;
    }
    auto end_laplacian = std::chrono::high_resolution_clock::now();
    print_duration("    -> Laplacian Setup", end_laplacian - start_laplacian);

    // ---------------------------------------------------------
    // Phase 3.2: Eigen Decomposition (Spectra, regularized matvec mode)
    //
    // STRATEGY: Lanczos naturally amplifies large eigenvalues, so finding
    // smallest eigenvalues directly via SmallestAlge converges poorly.
    // Shift-invert (SymEigsShiftSolver) helps but requires an LU/LDLT
    // factorization that we observed does not parallelize in our build.
    //
    // Instead, build M = c*I - L, where c is an upper bound on the largest
    // eigenvalue of L. Then:
    //   - eigenvectors of M and L are identical
    //   - eigenvalue mapping is lambda_M = c - lambda_L
    //   - LARGEST eigenvalues of M correspond to SMALLEST of L
    //
    // We can now use the standard SymEigsSolver with LargestAlge, which is
    // Lanczos's natural fast-converging mode. The inner kernel is plain
    // dense-symmetric-matvec (SGEMV via OpenBLAS), which DOES parallelize.
    //
    // Bound on lambda_max(L): the diagonal of L is exactly (m + tol) by
    // construction; off-diagonals are bounded in magnitude by 1 (from the
    // affinity normalization). Gershgorin gives lambda_max <= (m + tol) + m,
    // so c = 2*(m + tol) is a safe upper bound.
    // ---------------------------------------------------------
    auto start_eigen = std::chrono::high_resolution_clock::now();

    int n_requested = std::min(k + 1, m);
    Eigen::VectorXf eigvals;
    Eigen::MatrixXf eigvecs;

    bool used_spectra = false;
    try {
        // Build M = c*I - L. We allocate a fresh matrix; this is m^2 floats
        // (~256 MB at m=8000 -- one-time cost, fully parallelizable).
        const float c = 2.0f * (static_cast<float>(m) + tol);
        Eigen::MatrixXf M = -L;
        M.diagonal().array() += c;

        // ncv: Krylov subspace size. Larger = faster convergence but more
        // memory and per-iter cost. 4*n_requested is a robust choice for
        // LargestAlge mode at small n_requested relative to m.
        int ncv = std::min(std::max(4 * n_requested, 20), m);

        Spectra::DenseSymMatProd<float> op(M);
        Spectra::SymEigsSolver<Spectra::DenseSymMatProd<float>>
            solver(op, n_requested, ncv);

        solver.init();
        int nconv = solver.compute(
            Spectra::SortRule::LargestAlge,   // natural Lanczos mode
            /*maxit=*/ 1000,
            /*tol=*/   1e-6f,
            Spectra::SortRule::LargestAlge    // sort returned descending in M
        );

        if (solver.info() == Spectra::CompInfo::Successful && nconv >= n_requested) {
            // Map eigenvalues of M back to eigenvalues of L: lambda_L = c - lambda_M.
            // Spectra returned them sorted DESCENDING in M, which means
            // ASCENDING in L -- exactly what downstream code expects.
            Eigen::VectorXf raw = solver.eigenvalues();
            eigvals.resize(n_requested);
            for (int i = 0; i < n_requested; i++) {
                eigvals(i) = c - raw(i);
            }
            // Eigenvectors are unchanged (M and L share eigenvectors).
            eigvecs = solver.eigenvectors();
            used_spectra = true;
        } else {
            std::cout << "    -> Spectra did not converge (nconv=" << nconv
                      << "), falling back to dense solver\n";
        }
    } catch (const std::exception& e) {
        std::cout << "    -> Spectra threw (" << e.what()
                  << "), falling back to dense solver\n";
    }

    if (!used_spectra) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(L);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed");
        eigvals = solver.eigenvalues().head(n_requested);
        eigvecs = solver.eigenvectors().leftCols(n_requested);
    }

    auto end_eigen = std::chrono::high_resolution_clock::now();
    print_duration(used_spectra
                       ? "    -> Eigen Decomposition (Spectra-matvec)"
                       : "    -> Eigen Decomposition (dense)",
                   end_eigen - start_eigen);

    // ---------------------------------------------------------
    // Phase 3.3: Eigenvector Extraction & Normalization  (unchanged)
    // ---------------------------------------------------------
    Eigen::MatrixXf U = eigvecs.leftCols(k);
    for (int i = 0; i < m; i++) {
        float norm = U.row(i).norm();
        if (norm > 1e-10f)
            U.row(i) /= norm;
    }

    float ngap = 0.0f;
    if (k < n_requested && k >= 1) {
        float lk   = eigvals(k);
        float lkm1 = eigvals(k - 1);
        ngap = (std::abs(lkm1) > 1e-10f) ? ((lk - lkm1) / lkm1) : 0.0f;
    }

    std::vector<float> eigvals_vec(m, 0.0f);
    for (int i = 0; i < n_requested; i++)
        eigvals_vec[i] = eigvals(i);

    // ---------------------------------------------------------
    // Phase 3.4: Downstream K-Means  (unchanged)
    // ---------------------------------------------------------
    auto start_km2 = std::chrono::high_resolution_clock::now();
    Matrix U_flat(m * k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U_flat[i * k + j] = U(i, j);

    KMeans km(k, n_iter, -1, random_state);
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
    Matrix aff;
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
    SpectralResult sc;
    
    if((use_gpu)) { // GPU eigendecomposition can be slower for large m due to cuSOLVER overheads
        sc = spectralClusteringCuda(aff, m, k, n_iter, random_state);
    }
    else{
        auto eigen_threads = Eigen::nbThreads();
        std::cout << "    -> Using Eigen with " << eigen_threads << " threads for eigendecomposition\n"; 
        sc = spectralClustering(aff, m, k, n_iter, random_state);
    }

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