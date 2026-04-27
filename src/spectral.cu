// spectral_cuda.cu
//
// GPU implementation of spectralClustering matching the CPU/Eigen version.
//
// Pipeline:
//   3.1 Laplacian:  L = -D^{-1/2} A D^{-1/2}, then add (m + tol) on the diagonal,
//                   where D_ii = (mean of row i of A)^{-1/2}.
//   3.2 Eigen:      symmetric eigendecomposition via cusolverDnDsyevd
//                   (ascending eigenvalues, eigenvectors as columns).
//   3.3 U + norm:   take first k columns of eigenvectors, L2-normalize each row.
//                   Compute ngap = (lambda_k - lambda_{k-1}) / lambda_{k-1}.
//   3.4 K-Means:    reuses fitKMeansCuda on the float-cast U.
//
// Notes on layout: cuSOLVER and Eigen both use column-major storage for the
// eigenvector matrix, so "first k columns" is just the first m*k doubles of
// the eigenvector buffer.

#include "spectral_cuda.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// ---------- error-checking helpers ----------------------------------------

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(_e) + " at " +        \
                                     __FILE__ + ":" +                          \
                                     std::to_string(__LINE__));               \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(expr)                                                   \
    do {                                                                       \
        cusolverStatus_t _s = (expr);                                          \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(                                          \
                std::string("cuSOLVER error ") + std::to_string(_s) +          \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                      \
    } while (0)

// ---------- kernels --------------------------------------------------------

// Compute row means of an m x m row-major matrix A_in (the affinity).
// Also writes A_out as a column-major copy of A_in so that downstream BLAS-
// style operations match Eigen/cuSOLVER expectations.
//
// One block per row; threads in the block do a shared-memory reduction.
__global__ void rowMeanAndTransposeKernel(const double* __restrict__ A_in_row,
                                          double* __restrict__ A_out_col,
                                          double* __restrict__ row_mean,
                                          int m) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    double sum = 0.0;
    for (int j = tid; j < m; j += blockDim.x) {
        double v = A_in_row[row * m + j];
        sum += v;
        // Row-major (row, j)  ->  Column-major index = j * m + row.
        A_out_col[j * m + row] = v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) sdata[tid] += sdata[tid + off];
        __syncthreads();
    }
    if (tid == 0) {
        row_mean[row] = sdata[0] / static_cast<double>(m);
    }
}

// d_vec[i] = (row_mean[i] > 0) ? row_mean[i]^{-1/2} : 0.
__global__ void buildDvecKernel(const double* __restrict__ row_mean,
                                double* __restrict__ d_vec,
                                int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    double rm = row_mean[i];
    d_vec[i] = (rm > 0.0) ? rsqrt(rm) : 0.0;
}

// In-place: A <- -D * A * D, where D = diag(d_vec).
// A is column-major, m x m. Element (i, j) lives at A[j*m + i].
// Scaling by D on both sides means A(i,j) *= d_vec[i] * d_vec[j].
// Then add (m + tol) to the diagonal.
__global__ void scaleAndShiftDiagKernel(double* A,
                                        const double* __restrict__ d_vec,
                                        int m,
                                        double diag_shift) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= m) return;

    int idx = j * m + i;  // column-major
    double v = A[idx];
    double scaled = -v * d_vec[i] * d_vec[j];
    if (i == j) scaled += diag_shift;
    A[idx] = scaled;
}

// Take the first k columns of a column-major (m x m) eigenvector matrix and
// produce a row-major (m x k) float matrix with each row L2-normalized.
//
// One block per row; threads in the block cooperate on the row's norm.
__global__ void extractAndRowNormalizeKernel(const double* __restrict__ V_col,
                                             float* __restrict__ U_row_f,
                                             int m,
                                             int k) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Sum of squares for this row across the first k columns.
    double sumsq = 0.0;
    for (int j = tid; j < k; j += blockDim.x) {
        double v = V_col[j * m + row];   // column-major (row, j)
        sumsq += v * v;
    }
    sdata[tid] = sumsq;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) sdata[tid] += sdata[tid + off];
        __syncthreads();
    }

    double norm = sqrt(sdata[0]);
    double inv = (norm > 1e-10) ? (1.0 / norm) : 0.0;

    for (int j = tid; j < k; j += blockDim.x) {
        double v = V_col[j * m + row];
        // store as row-major float: row*k + j
        U_row_f[row * k + j] = static_cast<float>(v * inv);
    }
}

// Helper lambda to make printing cleaner
auto print_duration_cuda = [](const std::string& name, std::chrono::duration<double> duration) {
    std::cout << std::fixed << std::setprecision(4) 
              << "  [Profile] " << name << ": " << duration.count() << " s\n";
};

// ---------- main entry point ----------------------------------------------

SpectralResult spectralClusteringCuda(const MatrixD& affinity,
                                      int m,
                                      int k,
                                      int n_iter,
                                      uint64_t random_state) {
    auto start_all = std::chrono::high_resolution_clock::now();
    std::cout << "Running spectral clustering on GPU with m=" << m << ", k=" << k << "\n";

    // -----------------------------------------------------------------
    // Phase 3.1: Laplacian construction
    // -----------------------------------------------------------------
    auto start_laplacian = std::chrono::high_resolution_clock::now();

    // Upload affinity (assumed row-major double, size m*m).
    double* d_A_row = nullptr;   // row-major copy (input)
    double* d_L     = nullptr;   // column-major working buffer (becomes L)
    double* d_rmean = nullptr;
    double* d_dvec  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_row, sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_L,     sizeof(double) * m * m));
    CUDA_CHECK(cudaMalloc(&d_rmean, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_dvec,  sizeof(double) * m));

    // affinity is std::vector<double> (or similar) of length m*m, row-major.
    CUDA_CHECK(cudaMemcpy(d_A_row, affinity.data(),
                          sizeof(double) * m * m,
                          cudaMemcpyHostToDevice));

    // Row means + transpose to column-major into d_L.
    {
        int threads = 256;
        size_t shmem = sizeof(double) * threads;
        rowMeanAndTransposeKernel<<<m, threads, shmem>>>(
            d_A_row, d_L, d_rmean, m);
        CUDA_CHECK(cudaGetLastError());
    }

    // d_vec = rowmean^{-1/2}
    {
        int threads = 256;
        int blocks  = (m + threads - 1) / threads;
        buildDvecKernel<<<blocks, threads>>>(d_rmean, d_dvec, m);
        CUDA_CHECK(cudaGetLastError());
    }

    // L = -D L D + (m + tol) * I    (in place on d_L)
    {
        const double tol = 1e-8;
        const double diag_shift = static_cast<double>(m) + tol;
        dim3 block(16, 16);
        dim3 grid((m + 15) / 16, (m + 15) / 16);
        scaleAndShiftDiagKernel<<<grid, block>>>(d_L, d_dvec, m, diag_shift);
        CUDA_CHECK(cudaGetLastError());
    }

    // d_A_row and d_rmean are no longer needed.
    CUDA_CHECK(cudaFree(d_A_row));
    CUDA_CHECK(cudaFree(d_rmean));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_laplacian = std::chrono::high_resolution_clock::now();
    print_duration_cuda(" -> Laplacian Setup (CUDA)", end_laplacian - start_laplacian);

    // -----------------------------------------------------------------
    // Phase 3.2: Eigendecomposition (cusolverDnDsyevd)
    //   - Computes all eigenvalues (ascending) + eigenvectors of L.
    //   - Eigenvectors overwrite d_L on output (column-major).
    // -----------------------------------------------------------------
    auto start_eigen = std::chrono::high_resolution_clock::now();

    cusolverDnHandle_t cusolverH = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    double* d_eigvals = nullptr;
    int*    d_info    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_eigvals, sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(&d_info,    sizeof(int)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR,   // eigenvalues + eigenvectors
        CUBLAS_FILL_MODE_LOWER,     // L is symmetric; use lower triangle
        m, d_L, m, d_eigvals, &lwork));

    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(double) * lwork));

    CUSOLVER_CHECK(cusolverDnDsyevd(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        m, d_L, m, d_eigvals,
        d_work, lwork, d_info));

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        // Cleanup before throwing
        cudaFree(d_work);
        cudaFree(d_info);
        cudaFree(d_eigvals);
        cudaFree(d_dvec);
        cudaFree(d_L);
        cusolverDnDestroy(cusolverH);
        throw std::runtime_error("cusolverDnDsyevd failed, info=" +
                                 std::to_string(h_info));
    }

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_dvec));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end_eigen = std::chrono::high_resolution_clock::now();
    print_duration_cuda(" -> Eigen Decomposition (CUDA)", end_eigen - start_eigen);

    // -----------------------------------------------------------------
    // Phase 3.3: Eigenvector extraction & row normalization
    // -----------------------------------------------------------------
    // Copy eigenvalues to host so we can compute ngap and the float vector.
    std::vector<double> eigvals_host(m);
    CUDA_CHECK(cudaMemcpy(eigvals_host.data(), d_eigvals,
                          sizeof(double) * m, cudaMemcpyDeviceToHost));

    float ngap = 0.0f;
    if (k < m && k >= 1) {
        double lk   = eigvals_host[k];
        double lkm1 = eigvals_host[k - 1];
        ngap = (std::abs(lkm1) > 1e-10)
                   ? static_cast<float>((lk - lkm1) / lkm1)
                   : 0.0f;
    }

    std::vector<float> eigvals_vec(m);
    for (int i = 0; i < m; ++i) {
        eigvals_vec[i] = static_cast<float>(eigvals_host[i]);
    }

    // Build row-normalized U (m x k), float, row-major, on device.
    float* d_U_f = nullptr;
    CUDA_CHECK(cudaMalloc(&d_U_f, sizeof(float) * m * k));
    {
        int threads = 128;
        // pick at least k threads-worth of work per block, capped at 512
        if (k > threads) threads = std::min(512, ((k + 31) / 32) * 32);
        size_t shmem = sizeof(double) * threads;
        extractAndRowNormalizeKernel<<<m, threads, shmem>>>(
            d_L, d_U_f, m, k);
        CUDA_CHECK(cudaGetLastError());
    }

    // Pull U back to host (Matrix is the float host buffer expected by
    // fitKMeansCuda). If your fitKMeansCuda has a device-pointer overload,
    // skip this copy and pass d_U_f directly.
    Matrix U_flat(m * k);
    CUDA_CHECK(cudaMemcpy(U_flat.data(), d_U_f,
                          sizeof(float) * m * k, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_U_f));
    CUDA_CHECK(cudaFree(d_eigvals));
    CUDA_CHECK(cudaFree(d_L));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    // -----------------------------------------------------------------
    // Phase 3.4: Downstream k-means (existing CUDA path)
    // -----------------------------------------------------------------
    auto start_km2 = std::chrono::high_resolution_clock::now();
    KMeansResult kmResult =
        fitKMeansCuda(U_flat, m, k, k, n_iter, random_state);
    auto end_km2 = std::chrono::high_resolution_clock::now();
    print_duration_cuda(" -> Spectral K-Means (CUDA)", end_km2 - start_km2);

    auto end_all = std::chrono::high_resolution_clock::now();
    print_duration_cuda(" Total SpectralClustering Phase (CUDA)",
                   end_all - start_all);

    return {kmResult.labels, eigvals_vec, ngap};
}