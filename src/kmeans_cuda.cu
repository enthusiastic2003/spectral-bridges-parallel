// src/kmeans_cuda.cu
//
// CUDA k-means — single unified path.
//
// Per iteration: 3 kernel launches
//   1. assignKernel   : tiled centroids in shared memory; falls back to
//                       a register-only loop when even one centroid row
//                       does not fit in shared mem.
//   2. accumKernel    : global atomics for centroid sums, shared-mem
//                       atomics for the (always-tiny) counts.
//   3. divideKernel   : newCentroids /= counts, with empty-cluster
//                       fallback to the previous centroid.
//
// Memory: only 4 device buffers are kept live across iterations
//   d_X       n·d floats      (input, read-only)
//   d_C       k·d floats      (current centroids)
//   d_newC    k·d floats      (next centroids; ping-pong with d_C)
//   d_labels  n   ints
//   d_counts  k   ints
//
#include "kmeans.hpp"
#include "kmeans_cuda.hpp"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        throw std::runtime_error("CUDA error");                            \
    }                                                                      \
} while (0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

constexpr int BLOCK = 256;

// ───────────────────────────────────────────────────────────────
//  ASSIGN
//
//  Streams centroids through shared memory in tiles of `tile_k`
//  rows. If even one centroid row will not fit in shared mem
//  (tile_k == 0), reads centroids straight from global memory.
// ───────────────────────────────────────────────────────────────
__global__ void assignKernel(
    const float* __restrict__ X,
    const float* __restrict__ C,
    int*         __restrict__ labels,
    int n, int d, int k, int tile_k)
{
    extern __shared__ float s_C[];                  // [tile_k * d], may be unused

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float best_dist = FLT_MAX;
    int   best_c    = 0;

    if (tile_k > 0) {
        // Tiled path: cooperatively load centroid tiles into shared mem.
        for (int c_start = 0; c_start < k; c_start += tile_k) {
            const int cur_tile = min(tile_k, k - c_start);
            const int total    = cur_tile * d;

            for (int i = tid; i < total; i += blockDim.x)
                s_C[i] = C[(size_t)c_start * d + i];
            __syncthreads();

            if (gid < n) {
                const float* xi = X + (size_t)gid * d;
                for (int c = 0; c < cur_tile; ++c) {
                    float dist = 0.0f;
                    const float* cc = s_C + c * d;
                    for (int dim = 0; dim < d; ++dim) {
                        float diff = xi[dim] - cc[dim];
                        dist += diff * diff;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_c    = c_start + c;
                    }
                }
            }
            __syncthreads();
        }
    } else {
        // Fallback: d is so large that not even one centroid row fits
        // in shared memory. Read centroids from global memory directly.
        if (gid < n) {
            const float* xi = X + (size_t)gid * d;
            for (int c = 0; c < k; ++c) {
                float dist = 0.0f;
                const float* cc = C + (size_t)c * d;
                for (int dim = 0; dim < d; ++dim) {
                    float diff = xi[dim] - cc[dim];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best_c = c; }
            }
        }
    }

    if (gid < n) labels[gid] = best_c;
}

// ───────────────────────────────────────────────────────────────
//  ACCUMULATE
//
//  Centroid sums: global atomics (k·d is bounded by output size,
//  contention is manageable for typical k).
//  Counts: shared-mem atomics, then one global atomic per cluster
//  per block at the end.
// ───────────────────────────────────────────────────────────────
__global__ void accumKernel(
    const float* __restrict__ X,
    const int*   __restrict__ labels,
    float*       __restrict__ newCentroids,
    int*         __restrict__ counts,
    int n, int d, int k)
{
    extern __shared__ int s_counts[];               // [k]

    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) s_counts[i] = 0;
    __syncthreads();

    const int gid = blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        const int c = labels[gid];
        atomicAdd(&s_counts[c], 1);
        const float* xi = X + (size_t)gid * d;
        float* cc = newCentroids + (size_t)c * d;
        for (int dim = 0; dim < d; ++dim)
            atomicAdd(&cc[dim], xi[dim]);
    }
    __syncthreads();

    for (int i = tid; i < k; i += blockDim.x)
        atomicAdd(&counts[i], s_counts[i]);
}

// ───────────────────────────────────────────────────────────────
//  DIVIDE  —  newCentroids /= counts, fall back to old on empty
// ───────────────────────────────────────────────────────────────
__global__ void divideKernel(
    float*       __restrict__ newCentroids,
    const int*   __restrict__ counts,
    const float* __restrict__ oldCentroids,
    int k, int d)
{
    const int j   = blockIdx.x;
    const int tid = threadIdx.x;
    if (j >= k) return;

    const int c = counts[j];
    float* nc       = newCentroids + (size_t)j * d;
    const float* oc = oldCentroids + (size_t)j * d;

    if (c > 0) {
        const float inv = 1.0f / (float)c;
        for (int dim = tid; dim < d; dim += blockDim.x)
            nc[dim] *= inv;
    } else {
        for (int dim = tid; dim < d; dim += blockDim.x)
            nc[dim] = oc[dim];
    }
}

// ───────────────────────────────────────────────────────────────
//  HOST ENTRY POINT
// ───────────────────────────────────────────────────────────────
KMeansResult fitKMeansCuda(
    const Matrix& X, int n, int d,
    int n_clusters, int n_iter, uint64_t random_state)
{
    const int k = n_clusters;

    // Query device shared-memory limit (used to size assign tile).
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int smem_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
    const int smem_floats = smem_bytes / (int)sizeof(float);

    // Tile size for the assign kernel: as many full centroid rows as fit.
    // 0 means "even one row doesn't fit" → kernel takes the global path.
    const int tile_k_max = (d > 0) ? (smem_floats / d) : k;
    const int tile_k     = std::min(std::max(tile_k_max, 0), k);
    const size_t assign_smem = (size_t)tile_k * d * sizeof(float);
    const size_t accum_smem  = (size_t)k * sizeof(int);

    float *d_X = nullptr, *d_C = nullptr, *d_newC = nullptr;
    int   *d_labels = nullptr, *d_counts = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_X,      (size_t)n * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,      (size_t)k * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_newC,   (size_t)k * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counts, (size_t)k * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_X, X.data(),
                              (size_t)n * d * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Init centroids via the host seeder.
        std::mt19937_64 rng(random_state);
        KMeans seeder(k, /*n_iter=*/n_iter, /*n_local_trials=*/-1, random_state);
        KMeansResult seed = seeder.initCentroids(X, n, d, rng);
        CUDA_CHECK(cudaMemcpy(d_C, seed.centroids.data(),
                              (size_t)k * d * sizeof(float),
                              cudaMemcpyHostToDevice));

        const int nblocks = (n + BLOCK - 1) / BLOCK;

        for (int iter = 0; iter < n_iter; ++iter) {
            CUDA_CHECK(cudaMemsetAsync(d_newC,   0, (size_t)k * d * sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)k * sizeof(int)));

            assignKernel<<<nblocks, BLOCK, assign_smem>>>(
                d_X, d_C, d_labels, n, d, k, tile_k);
            CUDA_CHECK_KERNEL();

            accumKernel<<<nblocks, BLOCK, accum_smem>>>(
                d_X, d_labels, d_newC, d_counts, n, d, k);
            CUDA_CHECK_KERNEL();

            divideKernel<<<k, BLOCK>>>(
                d_newC, d_counts, d_C, k, d);
            CUDA_CHECK_KERNEL();

            std::swap(d_C, d_newC);
        }

        Matrix centroids(k * d);
        std::vector<int> labels(n);
        CUDA_CHECK(cudaMemcpy(centroids.data(), d_C,
                              (size_t)k * d * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(labels.data(), d_labels,
                              (size_t)n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        cudaFree(d_X);     cudaFree(d_C);       cudaFree(d_newC);
        cudaFree(d_labels); cudaFree(d_counts);

        return {centroids, labels, k, d};
    }
    catch (...) {
        cudaFree(d_X);     cudaFree(d_C);       cudaFree(d_newC);
        cudaFree(d_labels); cudaFree(d_counts);
        throw;
    }
}