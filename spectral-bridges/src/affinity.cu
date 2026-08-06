// affinity_gpu.cu
//
// GPU-accelerated affinity matrix computation for Spectral Bridges.
//
// Pipeline (everything stays resident on device until the final D2H copy):
//
//   1. Sort points by Voronoi label and write contiguous, padded per-region
//      blocks of (x - mu_i) into device memory.
//                  X_centered : (CHUNK, n_max, d) row-major within each slab
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
// Memory strategy — all three large intermediate buffers are tiled by CHUNK:
//   d_X_centered : (CHUNK, n_max, d)   -- rebuilt each chunk
//   d_segments   : (CHUNK, m, d)       -- rebuilt each chunk
//   d_dists      : (CHUNK, m)          -- rebuilt each chunk
//   d_projs      : (CHUNK, n_max, m)   -- GEMM output, reused each chunk
//
// CHUNK is chosen automatically at runtime from cudaMemGetInfo so that
// the four tiled buffers plus the two permanent m×m buffers fit in
// MEM_FRACTION (85%) of free device memory.  It is clamped to [1, m].

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
// Kernel 1: build per-region padded centered slabs FOR ONE CHUNK
// ---------------------------------------------------------------------------
// Writes (chunk_size, n_max, d) into X_centered[0..chunk_size-1].
// region index within the chunk = blockIdx.x; global region = chunk_start + blockIdx.x.
//
__global__ void build_centered_slabs_kernel(
    const float*  __restrict__ X,
    const int*    __restrict__ point_perm,
    const int*    __restrict__ region_offset,
    const int*    __restrict__ counts,
    const float*  __restrict__ centroids,
    int chunk_start, int n, int m, int d, int n_max,
    float* __restrict__ X_centered)   // (chunk_size, n_max, d)
{
    int local_region = blockIdx.x;
    int region       = chunk_start + local_region;
    if (region >= m) return;

    int n_i      = counts[region];
    int off      = region_offset[region];
    float* slab  = X_centered + (size_t)local_region * n_max * d;

    extern __shared__ float s_mu[];
    for (int k = threadIdx.x; k < d; k += blockDim.x)
        s_mu[k] = centroids[region * d + k];
    __syncthreads();

    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n_i;
         row += gridDim.y * blockDim.y) {
        int orig_idx = point_perm[off + row];
        for (int k = threadIdx.x; k < d; k += blockDim.x)
            slab[row * d + k] = X[orig_idx * d + k] - s_mu[k];
    }

    for (int row = n_i + blockIdx.y * blockDim.y + threadIdx.y; row < n_max;
         row += gridDim.y * blockDim.y) {
        for (int k = threadIdx.x; k < d; k += blockDim.x)
            slab[row * d + k] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2b: build segments and dists for ONE CHUNK of source regions
// ---------------------------------------------------------------------------
// segments : (chunk_size, m, d)
// dists    : (chunk_size, m)
// blockIdx.x = local source region (0..chunk_size-1)
// blockIdx.y = target region j (0..m-1)
//
__global__ void build_segments_and_dists_chunk_kernel(
    const float* __restrict__ centroids,
    int chunk_start, int chunk_size, int m, int d,
    float* __restrict__ segments,   // (chunk_size, m, d)
    float* __restrict__ dists)      // (chunk_size, m)
{
    int local_i = blockIdx.x;
    int i       = chunk_start + local_i;
    int j       = blockIdx.y;
    if (i >= m || j >= m || local_i >= chunk_size) return;

    extern __shared__ float s_partial[];
    float local = 0.0f;
    float* seg_ij = segments + ((size_t)local_i * m + j) * d;

    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float v = centroids[j * d + k] - centroids[i * d + k];
        seg_ij[k] = v;
        local += v * v;
    }
    s_partial[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_partial[threadIdx.x] += s_partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sq = s_partial[0];
        if (i == j) sq = 1.0f;
        dists[local_i * m + j] = sq;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: column-wise log-sum-exp (unchanged except pointer/index offsets)
// ---------------------------------------------------------------------------
__global__ void column_logsumexp_kernel(
    const float* __restrict__ projs_all,     // (chunk_size, n_max, m)
    const float* __restrict__ dists,         // (chunk_size, m)
    const int*   __restrict__ counts,        // counts[chunk_start .. chunk_start+chunk_size-1]
    int m, int n_max,
    float p,
    float* __restrict__ log_affinity)        // row chunk_start of (m, m)
{
    int local_i = blockIdx.x;   // within chunk
    int j       = blockIdx.y;
    if (j >= m) return;

    const float tiny    = std::numeric_limits<float>::min();
    const float NEG_INF = -std::numeric_limits<float>::infinity();

    int n_i = counts[local_i];
    if (n_i == 0) {
        if (threadIdx.x == 0) log_affinity[local_i * m + j] = NEG_INF;
        return;
    }

    const float* projs_i = projs_all + (size_t)local_i * n_max * m;
    const float  inv_d   = 1.0f / dists[local_i * m + j];

    extern __shared__ float s_buf_f[];
    int tid = threadIdx.x;
    int bs  = blockDim.x;

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
            float a = s_buf_f[tid], b = s_buf_f[tid + s];
            s_buf_f[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float col_max = s_buf_f[0];
    __syncthreads();

    if (col_max == NEG_INF) {
        if (tid == 0) log_affinity[local_i * m + j] = NEG_INF;
        return;
    }

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

    if (tid == 0)
        log_affinity[local_i * m + j] = p * col_max + logf(s_buf_f[0]);
}

// ---------------------------------------------------------------------------
// Kernel 4: symmetrize, count-normalize, exponentiate (unchanged)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float logaddexp_dev(float a, float b) {
    float INF = std::numeric_limits<float>::infinity();
    if (a == -INF) return b;
    if (b == -INF) return a;
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

__global__ void subtract_max_kernel(float* __restrict__ A, size_t N, float maxVal)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride)
        A[i] -= maxVal;
}

__global__ void compute_row_entropies_kernel(
    const float* __restrict__ A,
    int m, float gamma,
    float* __restrict__ entropies)
{
    int i = blockIdx.x;
    if (i >= m) return;

    extern __shared__ float s_buf_f[];
    int tid = threadIdx.x, bs = blockDim.x;
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    const float* row = A + (size_t)i * m;

    float local_max = NEG_INF;
    for (int j = tid; j < m; j += bs) {
        if (j == i) continue;
        float val = gamma * row[j];
        if (val > local_max) local_max = val;
    }
    s_buf_f[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) { float a = s_buf_f[tid], b = s_buf_f[tid+s]; s_buf_f[tid] = a>b?a:b; }
        __syncthreads();
    }
    float row_max = s_buf_f[0];
    if (row_max == NEG_INF) { if (tid == 0) entropies[i] = 0.0f; return; }

    float local_sum = 0.0f;
    for (int j = tid; j < m; j += bs) {
        if (j == i) continue;
        local_sum += expf(gamma * row[j] - row_max);
    }
    s_buf_f[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) s_buf_f[tid] += s_buf_f[tid+s];
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
        if (tid < s) s_buf_f[tid] += s_buf_f[tid+s];
        __syncthreads();
    }
    if (tid == 0) entropies[i] = s_buf_f[0];
}

__global__ void exp_scale_kernel(float* __restrict__ A, size_t N, float gamma)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride)
        A[i] = expf(gamma * A[i]);
}

// ---------------------------------------------------------------------------
// Helper: compute the largest CHUNK that fits in available device memory
// ---------------------------------------------------------------------------
// Permanent buffers (always live):
//   d_X         : n * d
//   d_centroids : m * d
//   d_perm/off/counts : n + (m+1) + m  ints  ≈ negligible
//   d_log_affinity : m * m
//   d_A            : m * m
//   d_entropies    : m
//
// Per-chunk buffers (scaled by CHUNK):
//   d_X_centered : CHUNK * n_max * d
//   d_segments   : CHUNK * m    * d
//   d_dists      : CHUNK * m
//   d_projs      : CHUNK * n_max * m
//
// We solve for the largest CHUNK such that:
//   permanent + chunk_per_unit * CHUNK  <=  MEM_FRACTION * free_mem
//
static int computeChunk(int m, int n, int n_max, int d)
{
    constexpr double MEM_FRACTION = 0.82;   // leave 18% headroom
    constexpr size_t BYTES = 4;             // float32

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    size_t budget = static_cast<size_t>(MEM_FRACTION * static_cast<double>(free_mem));

    // Permanent allocations (bytes)
    size_t perm = (size_t)n * d * BYTES          // d_X
                + (size_t)m * d * BYTES          // d_centroids
                + (size_t)n     * sizeof(int)    // d_point_perm
                + (size_t)(m+1) * sizeof(int)    // d_region_offset
                + (size_t)m     * sizeof(int)    // d_counts
                + (size_t)m * m * BYTES          // d_log_affinity
                + (size_t)m * m * BYTES          // d_A
                + (size_t)m     * BYTES;         // d_entropies

    if (perm >= budget) {
        // Permanent buffers alone exceed budget — still try CHUNK=1 and
        // let cudaMalloc fail with a clear error rather than crashing here.
        std::cerr << "[affinity_gpu] WARNING: permanent buffers ("
                  << perm / (1<<20) << " MiB) exceed "
                  << static_cast<int>(MEM_FRACTION * 100) << "% of free GPU memory ("
                  << free_mem / (1<<20) << " MiB). Attempting CHUNK=1.\n";
        return 1;
    }

    size_t remaining = budget - perm;

    // Per-CHUNK cost (bytes per unit of CHUNK)
    size_t per_chunk = (size_t)n_max * d * BYTES   // d_X_centered slice
                     + (size_t)m     * d * BYTES   // d_segments slice
                     + (size_t)m         * BYTES   // d_dists slice
                     + (size_t)n_max * m * BYTES;  // d_projs slice

    if (per_chunk == 0) return m;   // degenerate case

    int chunk = static_cast<int>(remaining / per_chunk);
    chunk = std::max(1, std::min(chunk, m));

    std::cout << "[affinity_gpu] free GPU mem: " << free_mem / (1<<20) << " MiB"
              << "  |  permanent: " << perm / (1<<20) << " MiB"
              << "  |  per-chunk: " << per_chunk / (1<<20) << " MiB"
              << "  |  CHUNK=" << chunk << "\n";
    return chunk;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

Matrix computeAffinityGPU(
    const Matrix& X,
    const KMeansResult& km,
    int n, int m, int d,
    float target_perplexity)
{
    std::cout << "Computing affinity matrix on GPU... perplexity=" << target_perplexity << "\n";
    float p = target_perplexity;

    // -------- Host-side: sort points by Voronoi label --------
    std::vector<int> counts_h(m, 0);
    for (int i = 0; i < n; i++) counts_h[km.labels[i]]++;

    int n_max = 0;
    for (int c : counts_h) n_max = std::max(n_max, c);

    std::vector<int> region_offset_h(m + 1, 0);
    for (int i = 0; i < m; i++)
        region_offset_h[i + 1] = region_offset_h[i] + counts_h[i];

    std::vector<int> point_perm_h(n);
    {
        std::vector<int> cursor(m, 0);
        for (int i = 0; i < n; i++) {
            int r = km.labels[i];
            point_perm_h[region_offset_h[r] + cursor[r]++] = i;
        }
    }

    // -------- Choose CHUNK dynamically --------
    const int CHUNK = computeChunk(m, n, n_max, d);

    // -------- Permanent device allocations --------
    float *d_X = nullptr, *d_centroids = nullptr;
    int   *d_point_perm = nullptr, *d_region_offset = nullptr, *d_counts = nullptr;
    float *d_log_affinity = nullptr, *d_A = nullptr, *d_entropies = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X),
                          (size_t)n * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids),
                          (size_t)m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_point_perm),
                          (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_region_offset),
                          (size_t)(m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts),
                          (size_t)m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_log_affinity),
                          (size_t)m * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A),
                          (size_t)m * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_entropies),
                          (size_t)m * sizeof(float)));

    // -------- Per-chunk device allocations (sized for at most CHUNK regions) --------
    float *d_X_centered = nullptr, *d_segments = nullptr;
    float *d_dists = nullptr, *d_projs = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X_centered),
                          (size_t)CHUNK * n_max * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segments),
                          (size_t)CHUNK * m * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dists),
                          (size_t)CHUNK * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_projs),
                          (size_t)CHUNK * n_max * m * sizeof(float)));

    // -------- H2D copies --------
    CUDA_CHECK(cudaMemcpy(d_X,             X.data(),               (size_t)n * d * sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids,     km.centroids.data(),    (size_t)m * d * sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_point_perm,    point_perm_h.data(),    (size_t)n     * sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_region_offset, region_offset_h.data(), (size_t)(m+1) * sizeof(int),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counts,        counts_h.data(),        (size_t)m     * sizeof(int),      cudaMemcpyHostToDevice));

    // -------- cuBLAS handle --------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // -------- Kernel launch params (computed once) --------
    // Slab-building kernel
    dim3 slab_block(32, 8);
    int slab_gy = std::max(1, std::min(64, (n_max + (int)slab_block.y - 1) / (int)slab_block.y));

    // Segments kernel: thread count covers d, shared mem for partial sums
    int seg_threads = 64;
    while (seg_threads > d && seg_threads > 32) seg_threads /= 2;
    if (seg_threads < 32) seg_threads = 32;
    size_t seg_shm = seg_threads * sizeof(float);

    // Logsumexp kernel
    int lse_threads = 256;
    while (lse_threads > 32 && lse_threads > n_max) lse_threads /= 2;
    if (lse_threads < 32) lse_threads = 32;
    size_t lse_shm = lse_threads * sizeof(float);

    const float alpha = 1.0f, beta = 0.0f;

    // -------- Main chunk loop --------
    for (int chunk_start = 0; chunk_start < m; chunk_start += CHUNK) {
        int chunk = std::min(CHUNK, m - chunk_start);

        // -- Kernel 1: build padded centered slabs for this chunk --
        {
            dim3 grid(chunk, slab_gy);
            size_t shm = d * sizeof(float);
            build_centered_slabs_kernel<<<grid, slab_block, shm>>>(
                d_X, d_point_perm, d_region_offset, d_counts, d_centroids,
                chunk_start, n, m, d, n_max,
                d_X_centered);
            CUDA_CHECK(cudaGetLastError());
        }

        // -- Kernel 2: build segments and dists for this chunk --
        {
            dim3 grid(chunk, m);
            build_segments_and_dists_chunk_kernel<<<grid, seg_threads, seg_shm>>>(
                d_centroids, chunk_start, chunk, m, d,
                d_segments, d_dists);
            CUDA_CHECK(cudaGetLastError());
        }

        // -- GEMM: d_projs[local_i] = d_X_centered[local_i] @ d_segments[local_i]^T --
        // Each batch element:  (n_max x d) @ (m x d)^T  ->  (n_max x m)
        // cuBLAS column-major:  C = A^T * B  with m=m, n=n_max, k=d
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            /*m_*/ m, /*n_*/ n_max, /*k_*/ d,
            &alpha,
            d_segments,    /*lda*/ d, /*strideA*/ (long long)m * d,
            d_X_centered,  /*ldb*/ d, /*strideB*/ (long long)n_max * d,
            &beta,
            d_projs,       /*ldc*/ m, /*strideC*/ (long long)n_max * m,
            chunk));

        // -- Kernel 3: column logsumexp -> d_log_affinity rows [chunk_start .. chunk_start+chunk) --
        {
            dim3 lse_grid(chunk, m);
            column_logsumexp_kernel<<<lse_grid, lse_threads, lse_shm>>>(
                d_projs,
                d_dists,
                d_counts + chunk_start,
                m, n_max, p,
                d_log_affinity + (size_t)chunk_start * m);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // -------- Free per-chunk buffers now — no longer needed --------
    cudaFree(d_X_centered); d_X_centered = nullptr;
    cudaFree(d_segments);   d_segments   = nullptr;
    cudaFree(d_dists);      d_dists      = nullptr;
    cudaFree(d_projs);      d_projs      = nullptr;

    // -------- Kernel 4: symmetrize / normalize / exp --------
    {
        dim3 block(16, 16);
        dim3 grid((m + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        symmetrize_normalize_exp_kernel<<<grid, block>>>(
            d_log_affinity, d_counts, m, p, d_A);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------- Subtract max for numerical stability --------
    {
        size_t total = (size_t)m * m;
        thrust::device_ptr<float> d_A_ptr(d_A);
        float maxVal = *thrust::max_element(d_A_ptr, d_A_ptr + total);

        const int threads = 256;
        int blocks = std::min(1024, (int)((total + threads - 1) / threads));
        subtract_max_kernel<<<blocks, threads>>>(d_A, total, maxVal);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------- Perplexity calibration via binary search --------
    {
        float low = 0.0f, high = 1000.0f;
        float gamma = 0.5f * (low + high);
        const int max_iter = 16;
        const float tol    = 1e-2f;

        thrust::device_ptr<float> d_ent_ptr(d_entropies);

        for (int iter = 0; iter < max_iter; ++iter) {
            compute_row_entropies_kernel<<<m, 256, 256 * sizeof(float)>>>(
                d_A, m, gamma, d_entropies);
            CUDA_CHECK(cudaGetLastError());

            float mean_entropy = thrust::reduce(d_ent_ptr, d_ent_ptr + m, 0.0f) / (float)m;
            float current_perp = expf(mean_entropy);

            if (current_perp > target_perplexity) low  = gamma;
            else                                   high = gamma;
            gamma = 0.5f * (low + high);

            if (fabsf(current_perp - target_perplexity) / target_perplexity < tol)
                break;
        }

        // Final exp transform
        size_t total = (size_t)m * m;
        int threads = 256;
        int blocks = std::min(1024, (int)((total + threads - 1) / threads));
        exp_scale_kernel<<<blocks, threads>>>(d_A, total, gamma);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------- D2H final result --------
    Matrix A_host(m * m);
    CUDA_CHECK(cudaMemcpy(A_host.data(), d_A,
                          (size_t)m * m * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // -------- Cleanup --------
    cublasDestroy(handle);
    cudaFree(d_X);
    cudaFree(d_centroids);
    cudaFree(d_point_perm);
    cudaFree(d_region_offset);
    cudaFree(d_counts);
    cudaFree(d_log_affinity);
    cudaFree(d_A);
    cudaFree(d_entropies);

    return A_host;
}
