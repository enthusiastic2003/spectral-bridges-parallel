// affinity_gpu.cu
//
// GPU-accelerated affinity matrix computation for Spectral Bridges.
//
// Pipeline (everything stays resident on device until the final D2H copy):
//
//   1. Sort points by Voronoi label and write contiguous, padded per-region
//      blocks of (x - mu_i) into device memory.
//                  X_centered : (m, n_max, d) row-major within each slab
//                  counts     : (m,)  -- actual row count per region
//
//   2. Per-region GEMM via cuBLAS strided-batched SGEMM:
//          projs[i] = X_centered[i] @ segments[i]^T          shape (n_max, m)
//      where segments[i, j, :] = mu_j - mu_i, dists[i, j] = ||segments[i,j]||^2
//      with dists[i,i] := 1 to avoid division by zero.
//
//   3. Per-region elementwise: projs[i] /= dists[i]  (broadcast over rows).
//
//   4. Column-wise log-sum-exp reduction over the n_max rows of each
//      projs[i], skipping the (n_max - counts[i]) padded rows. Produces:
//          log_affinity[i, j] = p * max + LSE(p * (log_proj - max))
//      stored as (m, m) row-major.
//
//   5. Symmetrize + count-normalize + exponentiate:
//          A[i,j] = exp( (logaddexp(L[i,j], L[j,i]) - log(c_i + c_j)) / p )
//      Upper-triangle pass, both halves written.
//
// The result is the affinity matrix A (m × m). Perplexity calibration and
// downstream Laplacian / spectral steps are handled separately.
//
// Notes on numerical conventions (matched to the Python author):
//   - All floating-point work is in single precision.
//   - log() of clipped projections uses tiny = std::numeric_limits<float>::min().
//   - Padded rows in each region's slab are tagged with PADDED_SENTINEL so the
//     logsumexp kernel can mask them out.

#include "affinity_gpu.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t _e = (stmt);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error at ") +           \
                                     __FILE__ + ":" +                          \
                                     std::to_string(__LINE__) + " : " +        \
                                     cudaGetErrorString(_e));                  \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(stmt)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (stmt);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            throw std::runtime_error(std::string("cuBLAS error at ") +         \
                                     __FILE__ + ":" +                          \
                                     std::to_string(__LINE__));                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel 1: build per-region padded centered slabs
// ---------------------------------------------------------------------------
// Inputs:
//   X         : (n, d) row-major float
//   point_perm: (n,)   int -- sorted permutation, points grouped by region
//   region_offset, counts, centroids as in struct comments
//
// Output:
//   X_centered : (m, n_max, d) float, row-major within each (n_max, d) slab.
//                Real rows 0..counts[i]-1 contain (x - mu_i); padded rows
//                counts[i]..n_max-1 are zeroed.
//
// Each thread block handles one region; threads cooperate over rows and dims.
//
__global__ void build_centered_slabs_kernel(
    const float*  __restrict__ X,
    const int*    __restrict__ point_perm,
    const int*    __restrict__ region_offset,
    const int*    __restrict__ counts,
    const float*  __restrict__ centroids,
    int n, int m, int d, int n_max,
    float* __restrict__ X_centered)
{
    int region = blockIdx.x;
    if (region >= m) return;

    int n_i      = counts[region];
    int off      = region_offset[region];
    float* slab  = X_centered + (size_t)region * n_max * d;

    // Cache this region's centroid in shared memory.
    extern __shared__ float s_mu[];
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        s_mu[k] = centroids[region * d + k];
    }
    __syncthreads();

    // Real rows: write (x - mu) for each point in this region.
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n_i;
         row += gridDim.y * blockDim.y) {
        int orig_idx = point_perm[off + row];
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            slab[row * d + k] = X[orig_idx * d + k] - s_mu[k];
        }
    }

    // Padded rows: zero them out so the GEMM produces zeros, which the
    // logsumexp kernel will mask off via the per-region count.
    for (int row = n_i + blockIdx.y * blockDim.y + threadIdx.y; row < n_max;
         row += gridDim.y * blockDim.y) {
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            slab[row * d + k] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: build segments and dists for each region, in one pass
// ---------------------------------------------------------------------------
// segments[i, j, :] = centroids[j] - centroids[i]            (m, m, d)
// dists[i, j]       = ||segments[i, j]||^2, with dists[i,i]=1  (m, m)
//
// Each block handles one (i, j) pair; threads parallelize across d.
//
__global__ void build_segments_and_dists_kernel(
    const float* __restrict__ centroids,
    int m, int d,
    float* __restrict__ segments,    // (m, m, d) row-major in (j, k) per i
    float* __restrict__ dists)       // (m, m)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= m || j >= m) return;

    // Shared partial sums for ||.||^2
    extern __shared__ float s_partial[];
    float local = 0.0f;

    float* seg_ij = segments + ((size_t)i * m + j) * d;

    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float v = centroids[j * d + k] - centroids[i * d + k];
        seg_ij[k] = v;
        local += v * v;
    }
    s_partial[threadIdx.x] = local;
    __syncthreads();

    // Block-wide reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sq = s_partial[0];
        if (i == j) sq = 1.0f;         // avoid divide-by-zero on diagonal
        dists[i * m + j] = sq;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: column-wise log-sum-exp over each region's projs[i] slab
// ---------------------------------------------------------------------------
// For a given (i, j):
//   for each row r in region i (r < counts[i]):
//     v       = projs[i, r, j] / dists[i, j]
//     v_clip  = max(v, tiny)                           // clip <= 0 to tiny
//     lp      = log(v_clip)
//   col_max[j]    = max_r lp
//   log_aff[i, j] = p * col_max[j] + log( sum_r exp( p * (lp - col_max[j]) ) )
//
// Each block handles one (i, j) pair: blockIdx.x = i, blockIdx.y = j.
// Threads stride over rows, then do shared-memory reductions for max and sum.
//
__global__ void column_logsumexp_kernel(
    const float* __restrict__ projs_all,     // (m, n_max, m)
    const float* __restrict__ dists,         // (m, m)
    const int*   __restrict__ counts,        // (m,)
    int m, int n_max,
    float p,
    float* __restrict__ log_affinity)        // (m, m)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= m || j >= m) return;

    const float tiny    = std::numeric_limits<float>::min();
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    int n_i = counts[i];
    if (n_i == 0) {
        if (threadIdx.x == 0) log_affinity[i * m + j] = NEG_INF;
        return;
    }

    const float* projs_i = projs_all + (size_t)i * n_max * m;
    const float  inv_d   = 1.0f / dists[i * m + j];

    extern __shared__ float s_buf_f[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

    // Pass 1: column max of log(clip(v, tiny)) over the n_i real rows.
    float local_max = NEG_INF;
    for (int r = tid; r < n_i; r += bs) {
        float v  = projs_i[r * m + j] * inv_d;
        float vc = v > tiny ? v : tiny;
        float lp = logf(vc);
        if (lp > local_max) local_max = lp;
    }
    s_buf_f[tid] = local_max;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = s_buf_f[tid];
            float b = s_buf_f[tid + s];
            s_buf_f[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    float col_max = s_buf_f[0];
    __syncthreads();

    if (col_max == NEG_INF) {
        if (tid == 0) log_affinity[i * m + j] = NEG_INF;
        return;
    }

    // Pass 2: sum_r exp(p * (lp - col_max)).
    float local_sum = 0.0f;
    for (int r = tid; r < n_i; r += bs) {
        float v  = projs_i[r * m + j] * inv_d;
        float vc = v > tiny ? v : tiny;
        float lp = logf(vc);
        local_sum += expf(p * (lp - col_max));
    }
    s_buf_f[tid] = local_sum;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf_f[tid] += s_buf_f[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        log_affinity[i * m + j] = p * col_max + logf(s_buf_f[0]);
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: symmetrize, count-normalize, exponentiate
// ---------------------------------------------------------------------------
// A[i, j] = exp( ( logaddexp(L[i,j], L[j,i]) - log(c_i + c_j) ) / p )
//
// One thread per upper-triangle entry; both halves written.
//
__device__ __forceinline__ float logaddexp_dev(float a, float b) {
    float NEG_INF = std::numeric_limits<float>::infinity();
    if (a == NEG_INF) return b;
    if (b == NEG_INF) return a;
    float mx = a > b ? a : b;
    float mn = a > b ? b : a;
    return mx + log1pf(expf(mn - mx));
}

__global__ void symmetrize_normalize_exp_kernel(
    const float* __restrict__ log_affinity,
    const int*   __restrict__ counts,
    int m, float p,
    float* __restrict__ A)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= m || j < i) return;

    float Lij = log_affinity[i * m + j];
    float Lji = log_affinity[j * m + i];

    float log_count = logf((float)(counts[i] + counts[j]));
    float log_sym   = logaddexp_dev(Lij, Lji) - log_count;
    float sym       = expf(log_sym / p);

    A[i * m + j] = sym;
    A[j * m + i] = sym;
}

__global__ void subtract_max_kernel(float* __restrict__ A,
                                    size_t N,
                                    float maxVal)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
        A[i] -= maxVal;
    }
}

__global__ void compute_row_entropies_kernel(
    const float* __restrict__ A,
    int m,
    float gamma,
    float* __restrict__ entropies)
{
    int i = blockIdx.x;
    if (i >= m) return;

    extern __shared__ float s_buf_f[];
    int tid = threadIdx.x;
    int bs = blockDim.x;

    const float NEG_INF = -std::numeric_limits<float>::infinity();
    float local_max = NEG_INF;
    const float* row = A + (size_t)i * m;
    for (int j = tid; j < m; j += bs) {
        if (j == i) continue;
        float val = gamma * row[j];
        if (val > local_max) local_max = val;
    }
    s_buf_f[tid] = local_max;
    __syncthreads();

    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = s_buf_f[tid];
            float b = s_buf_f[tid + s];
            s_buf_f[tid] = a > b ? a : b;
        }
        __syncthreads();
    }

    float row_max = s_buf_f[0];
    if (row_max == NEG_INF) {
        if (tid == 0) entropies[i] = 0.0f;
        return;
    }

    float local_sum = 0.0f;
    for (int j = tid; j < m; j += bs) {
        if (j == i) continue;
        local_sum += expf(gamma * row[j] - row_max);
    }
    s_buf_f[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf_f[tid] += s_buf_f[tid + s];
        __syncthreads();
    }

    float log_sum = row_max + logf(s_buf_f[0]);
    float local_entropy = 0.0f;
    for (int j = tid; j < m; j += bs) {
        if (j == i) continue;
        float log_P = gamma * row[j] - log_sum;
        float P = expf(log_P);
        local_entropy -= P * log_P;
    }
    s_buf_f[tid] = local_entropy;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf_f[tid] += s_buf_f[tid + s];
        __syncthreads();
    }

    if (tid == 0) entropies[i] = s_buf_f[0];
}

__global__ void exp_scale_kernel(float* __restrict__ A,
                                 size_t N,
                                 float gamma)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
        A[i] = expf(gamma * A[i]);
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

/*
    Compute the affinity matrix for the given data and k-means result.
    X = Matrix of shape [n × d]
    km = KMeansResult containing centroids and labels
    n = number of data points
    m = number of Voronoi regions (k-means clusters)
    d = dimensionality of data
    target_perplexity = perplexity value for scaling affinities
*/

Matrix computeAffinityGPU(
    const Matrix& X,
    const KMeansResult& km,
    int n, int m, int d,
    float target_perplexity)
{

    std::cout << "Computing affinity matrix on GPU..."<< " "<< target_perplexity << std::endl;
    float p = target_perplexity;

    // -------- Host-side: sort points by Voronoi label --------
    std::vector<int> counts_h(m, 0);
    for (int i = 0; i < n; i++) counts_h[km.labels[i]]++;

    int n_max = 0;
    for (int c : counts_h) n_max = std::max(n_max, c);

    std::vector<int> region_offset_h(m + 1, 0);
    for (int i = 0; i < m; i++) region_offset_h[i + 1] = region_offset_h[i] + counts_h[i];

    // Build permutation: point_perm[region_offset[r] + k] = k-th point in region r
    std::vector<int> point_perm_h(n);
    {
        std::vector<int> cursor(m, 0);
        for (int i = 0; i < n; i++) {
            int r = km.labels[i];
            point_perm_h[region_offset_h[r] + cursor[r]++] = i;
        }
    }

    // -------- Device allocations --------
    float  *d_X = nullptr, *d_centroids = nullptr;
    int    *d_point_perm = nullptr, *d_region_offset = nullptr, *d_counts = nullptr;
    float  *d_X_centered = nullptr;
    float  *d_segments = nullptr, *d_dists = nullptr;
    float  *d_projs = nullptr;
    float  *d_log_affinity = nullptr;
    float  *d_A = nullptr;

    const size_t bytes_X          = (size_t)n * d * sizeof(float);
    const size_t bytes_centroids  = (size_t)m * d * sizeof(float);
    const size_t bytes_perm       = (size_t)n     * sizeof(int);
    const size_t bytes_offset     = (size_t)(m+1) * sizeof(int);
    const size_t bytes_counts     = (size_t)m     * sizeof(int);
    const size_t bytes_Xc         = (size_t)m * n_max * d * sizeof(float);
    const size_t bytes_segments   = (size_t)m * m * d     * sizeof(float);
    const size_t bytes_dists      = (size_t)m * m         * sizeof(float);
    // const size_t bytes_projs      = (size_t)m * n_max * m * sizeof(float);
    const int    CHUNK            = std::min(10, m);  // tune for your GPU
    const size_t bytes_projs      = (size_t)CHUNK * n_max * m * sizeof(float);
    const size_t bytes_log_aff    = (size_t)m * m         * sizeof(float);
    const size_t bytes_A          = bytes_log_aff;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X),             bytes_X));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids),     bytes_centroids));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_point_perm),    bytes_perm));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_region_offset), bytes_offset));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts),        bytes_counts));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X_centered),    bytes_Xc));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segments),      bytes_segments));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dists),         bytes_dists));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_projs),         bytes_projs));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_log_affinity),  bytes_log_aff));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A),             bytes_A));
    float *d_entropies = nullptr;
    const size_t bytes_entropies = (size_t)m * sizeof(float);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_entropies), bytes_entropies));

    // -------- H2D copies --------
    CUDA_CHECK(cudaMemcpy(d_X,             X.data(),                bytes_X,          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids,     km.centroids.data(),     bytes_centroids,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_point_perm,    point_perm_h.data(),     bytes_perm,       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_region_offset, region_offset_h.data(),  bytes_offset,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counts,        counts_h.data(),         bytes_counts,     cudaMemcpyHostToDevice));

    // -------- Kernel 1: build padded centered slabs --------
    {
        dim3 block(32, 8);
        // Cap y-grid so we don't launch absurd numbers of blocks for huge n_max.
        int gy = std::min(64, (n_max + (int)block.y - 1) / (int)block.y);
        dim3 grid(m, std::max(1, gy));
        size_t shm = d * sizeof(float);
        build_centered_slabs_kernel<<<grid, block, shm>>>(
            d_X, d_point_perm, d_region_offset, d_counts, d_centroids,
            n, m, d, n_max, d_X_centered);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------- Kernel 2: build segments and dists --------
    {
        int threads = 64;
        while (threads > d && threads > 32) threads /= 2;
        if (threads < 32) threads = 32;
        dim3 grid(m, m);
        size_t shm = threads * sizeof(float);
        build_segments_and_dists_kernel<<<grid, threads, shm>>>(
            d_centroids, m, d, d_segments, d_dists);
        CUDA_CHECK(cudaGetLastError());
    }


    // ###################################################
    // -------- Tiled GEMM + column logsumexp over regions --------
    //
    // We process CHUNK source-regions at a time. For each chunk:
    //   1. cuBLAS strided-batched SGEMM fills d_projs with CHUNK slabs of
    //      shape (n_max, m), one per region in the chunk.
    //   2. column_logsumexp_kernel reduces each slab into one row of
    //      d_log_affinity, written at offset chunk_start.
    //
    // Pointer offsets per chunk:
    //   d_X_centered + chunk_start * n_max * d   (input slabs)
    //   d_segments   + chunk_start * m     * d   (segment matrices)
    //   d_dists      + chunk_start * m            (per-region dist rows)
    //   d_counts     + chunk_start                (per-region counts)
    //   d_log_affinity + chunk_start * m          (output rows)
    //
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    {
        const float alpha = 1.0f, beta = 0.0f;

        // Pick logsumexp launch params once; they only depend on n_max.
        int lse_threads = 256;
        while (lse_threads > 32 && lse_threads > n_max) lse_threads /= 2;
        if (lse_threads < 32) lse_threads = 32;
        size_t lse_shm = lse_threads * sizeof(float);

        for (int chunk_start = 0; chunk_start < m; chunk_start += CHUNK) {
            int chunk = std::min(CHUNK, m - chunk_start);

            const float* seg_chunk =
                d_segments + (size_t)chunk_start * m * d;
            const float* xc_chunk =
                d_X_centered + (size_t)chunk_start * n_max * d;

            // GEMM: CHUNK batched (m x n_max) outputs.
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                /*m_*/ m, /*n_*/ n_max, /*k_*/ d,
                &alpha,
                seg_chunk,    /*lda*/ d, /*strideA*/ (long long)m * d,
                xc_chunk,     /*ldb*/ d, /*strideB*/ (long long)n_max * d,
                &beta,
                d_projs,      /*ldc*/ m, /*strideC*/ (long long)n_max * m,
                chunk));

            // Column logsumexp: writes `chunk` rows of d_log_affinity
            // starting at row `chunk_start`.
            dim3 lse_grid(chunk, m);
            column_logsumexp_kernel<<<lse_grid, lse_threads, lse_shm>>>(
                d_projs,
                d_dists  + (size_t)chunk_start * m,
                d_counts + chunk_start,
                /*m=*/ m,        // number of target regions (cols of projs)
                n_max, p,
                d_log_affinity + (size_t)chunk_start * m);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    // ###################################################


    

    // -------- Kernel 4: symmetrize / normalize / exp --------
    {
        dim3 block(16, 16);
        dim3 grid((m + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        symmetrize_normalize_exp_kernel<<<grid, block>>>(
            d_log_affinity, d_counts, m, p, d_A);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------- D2H final result --------
    size_t total = static_cast<size_t>(m) * static_cast<size_t>(m);
    thrust::device_ptr<float> d_A_ptr(d_A);
    float maxVal = *thrust::max_element(d_A_ptr, d_A_ptr + total);

    {
        const int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        blocks = std::min(blocks, 1024);
        subtract_max_kernel<<<blocks, threads>>>(d_A, total, maxVal);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------- Perplexity calibration via binary search --------
    float low = 0.0f;
    float high = 1000.0f;
    float gamma = 0.5f * (low + high);
    int max_iter = 16;
    float tol = 1e-2f;
    thrust::device_ptr<float> d_entropies_ptr(d_entropies);

    for (int iter = 0; iter < max_iter; ++iter) {
        int threads = 256;
        int blocks = m;
        size_t shm = threads * sizeof(float);
        compute_row_entropies_kernel<<<blocks, threads, shm>>>(
            d_A, m, gamma, d_entropies);
        CUDA_CHECK(cudaGetLastError());

        float sum_entropy = thrust::reduce(d_entropies_ptr, d_entropies_ptr + m, 0.0f);
        float mean_entropy = sum_entropy / static_cast<float>(m);
        float current_perp = expf(mean_entropy);

        if (current_perp > target_perplexity) {
            low = gamma;
        } else {
            high = gamma;
        }
        gamma = 0.5f * (low + high);

        if (fabsf(current_perp - target_perplexity) / target_perplexity < tol) {
            break;
        }
    }

    // -------- Final exponential transform in device memory --------
    {
        const int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        blocks = std::min(blocks, 1024);
        exp_scale_kernel<<<blocks, threads>>>(d_A, total, gamma);
        CUDA_CHECK(cudaGetLastError());
    }

    Matrix A_host(m * m);
    CUDA_CHECK(cudaMemcpy(A_host.data(), d_A, bytes_A, cudaMemcpyDeviceToHost));

    // -------- Cleanup --------
    cublasDestroy(handle);
    cudaFree(d_X);
    cudaFree(d_centroids);
    cudaFree(d_point_perm);
    cudaFree(d_region_offset);
    cudaFree(d_counts);
    cudaFree(d_X_centered);
    cudaFree(d_segments);
    cudaFree(d_dists);
    cudaFree(d_projs);
    cudaFree(d_log_affinity);
    cudaFree(d_A);
    cudaFree(d_entropies);

    return A_host;
}