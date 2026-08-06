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
#include "kmeans_cuda.hpp"
#include <iostream>
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
//  K-MEANS++ INIT KERNELS
// ───────────────────────────────────────────────────────────────

// Update minDists against the most recently chosen centroid.
// One thread per point. Fused min-update (no separate label pass —
// labels get rewritten by assignKernel on iter 0 anyway).
__global__ void initUpdateMinDistsKernel(
    const float* __restrict__ X,
    const float* __restrict__ last,    // d floats: the new centroid
    float*       __restrict__ minDists,
    int n, int d)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    const float* xi = X + (size_t)gid * d;
    float dist = 0.0f;
    for (int dim = 0; dim < d; ++dim) {
        float diff = xi[dim] - last[dim];
        dist += diff * diff;
    }
    float cur = minDists[gid];
    if (dist < cur) minDists[gid] = dist;
}

// Evaluate inertia for ALL trial candidates in one launch.
// Grid: (blocksPerCand, trials). Each block accumulates a partial
// sum for one candidate and atomicAdds it into inertia[t].
__global__ void initCandidateInertiaKernel(
    const float* __restrict__ X,
    const float* __restrict__ minDists,
    const int*   __restrict__ candIdx,    // [trials]
    float*       __restrict__ inertia,    // [trials], pre-zeroed
    int n, int d, int trials)
{
    const int t = blockIdx.y;
    if (t >= trials) return;

    const float* candPtr = X + (size_t)candIdx[t] * d;

    const int stride = blockDim.x * gridDim.x;
    float local = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride)
    {
        const float* xi = X + (size_t)idx * d;
        float dist = 0.0f;
        for (int dim = 0; dim < d; ++dim) {
            float diff = xi[dim] - candPtr[dim];
            dist += diff * diff;
        }
        const float m = minDists[idx];
        local += (dist < m) ? dist : m;
    }

    // Block reduction in shared memory (no CUB dependency).
    __shared__ float s_red[BLOCK];
    s_red[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s_red[threadIdx.x] += s_red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&inertia[t], s_red[0]);
}


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
//  K-MEANS++ SEEDER (device-resident)
//
//  Writes k centroids into d_C using d_X already on the device.
//  Weighted sampling stays on the host: cheap, runs k times, and
//  matches std::discrete_distribution semantics exactly.
// ───────────────────────────────────────────────────────────────
static void initCentroidsCuda(
    const float* d_X, float* d_C,
    int n, int d, int k,
    int n_local_trials, std::mt19937_64& rng)
{
    std::cout << "  Initializing centroids with k-means++(CUDA) seeding ("
              << ((n_local_trials < 0) ? "auto" : std::to_string(n_local_trials))
              << " local trials)" << std::endl;
    const int trials = (n_local_trials < 0)
                     ? (2 + static_cast<int>(std::log((double)k)))
                     : n_local_trials;

    float* d_minDists = nullptr;
    int*   d_candIdx  = nullptr;
    float* d_inertia  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_minDists, (size_t)n      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_candIdx,  (size_t)trials * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inertia,  (size_t)trials * sizeof(float)));

    // minDists ← +inf
    {
        std::vector<float> hMin(n, FLT_MAX);
        CUDA_CHECK(cudaMemcpy(d_minDists, hMin.data(),
                              (size_t)n * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // First centroid: uniform random index → d_C[0]
    std::uniform_int_distribution<int> uni(0, n - 1);
    int first = uni(rng);
    CUDA_CHECK(cudaMemcpy(d_C, d_X + (size_t)first * d,
                          (size_t)d * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    const int nblocks = (n + BLOCK - 1) / BLOCK;

    std::vector<float> hMinDists(n);
    std::vector<int>   hCandIdx(trials);
    std::vector<float> hInertia(trials);

    for (int c = 1; c <= k; ++c) {
        // 1) Update minDists against centroid (c-1)
        const float* d_last = d_C + (size_t)(c - 1) * d;
        initUpdateMinDistsKernel<<<nblocks, BLOCK>>>(
            d_X, d_last, d_minDists, n, d);
        CUDA_CHECK_KERNEL();

        if (c == k) break;

        // 2) Sample `trials` candidates on host, weighted by minDists
        CUDA_CHECK(cudaMemcpy(hMinDists.data(), d_minDists,
                              (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::discrete_distribution<int> weighted(
            hMinDists.begin(), hMinDists.end());
        for (int t = 0; t < trials; ++t) hCandIdx[t] = weighted(rng);
        CUDA_CHECK(cudaMemcpy(d_candIdx, hCandIdx.data(),
                              (size_t)trials * sizeof(int),
                              cudaMemcpyHostToDevice));

        // 3) Score all candidates in one launch
        CUDA_CHECK(cudaMemsetAsync(d_inertia, 0,
                                   (size_t)trials * sizeof(float)));
        const int blocksPerCand = std::min(nblocks, 512);
        dim3 grid(blocksPerCand, trials);
        initCandidateInertiaKernel<<<grid, BLOCK>>>(
            d_X, d_minDists, d_candIdx, d_inertia, n, d, trials);
        CUDA_CHECK_KERNEL();

        // 4) Argmin over `trials` floats on host
        CUDA_CHECK(cudaMemcpy(hInertia.data(), d_inertia,
                              (size_t)trials * sizeof(float),
                              cudaMemcpyDeviceToHost));
        int bestT = 0;
        for (int t = 1; t < trials; ++t)
            if (hInertia[t] < hInertia[bestT]) bestT = t;

        // 5) Write chosen centroid into slot c of d_C
        CUDA_CHECK(cudaMemcpy(d_C + (size_t)c * d,
                              d_X + (size_t)hCandIdx[bestT] * d,
                              (size_t)d * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_minDists);
    cudaFree(d_candIdx);
    cudaFree(d_inertia);
}

// ───────────────────────────────────────────────────────────────
//  HOST ENTRY POINT
// ───────────────────────────────────────────────────────────────
KMeansResult fitKMeansCuda(
    const Matrix& X, int n, int d,
    int n_clusters, int n_iter, uint64_t random_state)
{
    std::cout << "  Running CUDA k-means with n=" << n
              << "  , d=" << d
              << "  , k=" << n_clusters
              << "  , n_iter=" << n_iter
              << std::endl;

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

        // Init centroids on device — d_X is already populated.
        std::mt19937_64 rng(random_state);
        initCentroidsCuda(d_X, d_C, n, d, k, /*n_local_trials=*/-1, rng);

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