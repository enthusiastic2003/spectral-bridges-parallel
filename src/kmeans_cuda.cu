    // // src/kmeans_cuda.cu
    // //
    // // Robust CUDA k-means – works for any n, d, k.  No cuBLAS dependency.
    // //
    // // Three runtime paths (selected automatically based on k × d vs shared-mem budget):
    // //
    // //   Path A  "fused"         – small k·d : assign + accumulate in ONE kernel,
    // //                             centroids AND accumulators live in shared memory.
    // //
    // //   Path B  "tiled+shared"  – medium k·d : tiled assign kernel (centroids
    // //                             streamed through shared mem in tiles) + separate
    // //                             accumulate kernel with shared-mem accumulators.
    // //
    // //   Path C  "tiled+global"  – large k·d : same tiled assign, accumulate
    // //                             via global atomics (contention is naturally low
    // //                             when k·d is large).
    // //
    // // All three paths replace the old cuBLAS SGEMM + six-kernel pipeline with
    // // at most 3 kernel launches per iteration, and reduce global-atomic
    // // pressure by orders of magnitude for small-k workloads.
    // //
    // #include "kmeans.hpp"
    // #include "kmeans_cuda.hpp"

    // #include <cuda_runtime.h>
    // #include <cfloat>
    // #include <cstdio>
    // #include <algorithm>
    // #include <random>
    // #include <stdexcept>
    // #include <vector>

    // // ─────────────────────── error checking ───────────────────────
    // #define CUDA_CHECK(call) do {                                              \
    //     cudaError_t err = (call);                                              \
    //     if (err != cudaSuccess) {                                              \
    //         fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
    //                 __FILE__, __LINE__, cudaGetErrorString(err));              \
    //         throw std::runtime_error("CUDA error");                            \
    //     }                                                                      \
    // } while (0)

    // #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

    // // ─────────────────────── constants ────────────────────────────
    // constexpr int BLOCK = 256;

    // // ═══════════════════════════════════════════════════════════════
    // //  PATH A : fused assign + block-level accumulate
    // //
    // //  Shared-memory layout:
    // //    s_C      [k*d]   centroids  (read-only after load)
    // //    s_accum  [k*d]   per-block centroid sums
    // //    s_counts [k]     per-block counts  (int)
    // //
    // //  Requirement: (2·k·d)·4 + k·4  ≤  device shared-mem limit.
    // // ═══════════════════════════════════════════════════════════════
    // __global__ void fusedAssignAccumKernel(
    //     const float* __restrict__ X,
    //     const float* __restrict__ C,
    //     int*         __restrict__ labels,
    //     float*       __restrict__ newCentroids,
    //     int*         __restrict__ counts,
    //     int n, int d, int k)
    // {
    //     extern __shared__ float smem[];
    //     float* s_C      = smem;
    //     float* s_accum  = s_C + k * d;
    //     int*   s_counts = (int*)(s_accum + k * d);

    //     const int tid = threadIdx.x;

    //     for (int i = tid; i < k * d; i += blockDim.x) s_C[i]      = C[i];
    //     for (int i = tid; i < k * d; i += blockDim.x) s_accum[i]   = 0.0f;
    //     for (int i = tid; i < k;     i += blockDim.x) s_counts[i]  = 0;
    //     __syncthreads();

    //     const int gid = blockIdx.x * blockDim.x + tid;
    //     if (gid < n) {
    //         const float* xi = X + (size_t)gid * d;

    //         float best_dist = FLT_MAX;
    //         int   best_c    = 0;
    //         for (int c = 0; c < k; ++c) {
    //             float dist = 0.0f;
    //             const float* cc = s_C + c * d;
    //             for (int dim = 0; dim < d; ++dim) {
    //                 float diff = xi[dim] - cc[dim];
    //                 dist += diff * diff;
    //             }
    //             if (dist < best_dist) { best_dist = dist; best_c = c; }
    //         }

    //         labels[gid] = best_c;

    //         atomicAdd(&s_counts[best_c], 1);
    //         for (int dim = 0; dim < d; ++dim)
    //             atomicAdd(&s_accum[best_c * d + dim], xi[dim]);
    //     }
    //     __syncthreads();

    //     for (int i = tid; i < k * d; i += blockDim.x)
    //         atomicAdd(&newCentroids[i], s_accum[i]);
    //     for (int i = tid; i < k; i += blockDim.x)
    //         atomicAdd(&counts[i], s_counts[i]);
    // }

    // // ═══════════════════════════════════════════════════════════════
    // //  TILED ASSIGN KERNEL  –  used by paths B and C
    // //
    // //  Streams centroids through shared memory in tiles of `tile_k`
    // //  rows at a time.  Each thread tracks its running best across
    // //  all tiles.  Works for any k and d (as long as one centroid
    // //  row fits in shared mem; see globalAssignKernel below).
    // //
    // //  Shared-memory: tile_k · d  floats.
    // // ═══════════════════════════════════════════════════════════════
    // __global__ void tiledAssignKernel(
    //     const float* __restrict__ X,
    //     const float* __restrict__ C,
    //     int*         __restrict__ labels,
    //     int n, int d, int k, int tile_k)
    // {
    //     extern __shared__ float s_C[];            // [tile_k * d]

    //     const int tid = threadIdx.x;
    //     const int gid = blockIdx.x * blockDim.x + tid;

    //     float best_dist = FLT_MAX;
    //     int   best_c    = 0;

    //     for (int c_start = 0; c_start < k; c_start += tile_k) {
    //         const int c_end    = min(c_start + tile_k, k);
    //         const int cur_tile = c_end - c_start;
    //         const int total    = cur_tile * d;

    //         // cooperatively load centroid tile
    //         for (int i = tid; i < total; i += blockDim.x)
    //             s_C[i] = C[c_start * d + i];
    //         __syncthreads();

    //         if (gid < n) {
    //             const float* xi = X + (size_t)gid * d;
    //             for (int c = 0; c < cur_tile; ++c) {
    //                 float dist = 0.0f;
    //                 const float* cc = s_C + c * d;
    //                 for (int dim = 0; dim < d; ++dim) {
    //                     float diff = xi[dim] - cc[dim];
    //                     dist += diff * diff;
    //                 }
    //                 if (dist < best_dist) {
    //                     best_dist = dist;
    //                     best_c    = c_start + c;
    //                 }
    //             }
    //         }
    //         __syncthreads();          // before next tile load
    //     }

    //     if (gid < n) labels[gid] = best_c;
    // }

    // // Fallback assign: even a single centroid row is wider than shared mem
    // __global__ void globalAssignKernel(
    //     const float* __restrict__ X,
    //     const float* __restrict__ C,
    //     int*         __restrict__ labels,
    //     int n, int d, int k)
    // {
    //     const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (gid >= n) return;

    //     const float* xi = X + (size_t)gid * d;
    //     float best_dist = FLT_MAX;
    //     int   best_c    = 0;

    //     for (int c = 0; c < k; ++c) {
    //         float dist = 0.0f;
    //         const float* cc = C + (size_t)c * d;
    //         for (int dim = 0; dim < d; ++dim) {
    //             float diff = xi[dim] - cc[dim];
    //             dist += diff * diff;
    //         }
    //         if (dist < best_dist) { best_dist = dist; best_c = c; }
    //     }
    //     labels[gid] = best_c;
    // }

    // // ═══════════════════════════════════════════════════════════════
    // //  PATH B ACCUMULATE : shared-memory accumulators
    // //
    // //  Shared-memory: k·d floats  +  k ints.
    // //  Requirement: k·d·4 + k·4  ≤  device shared-mem limit.
    // // ═══════════════════════════════════════════════════════════════
    // __global__ void sharedAccumKernel(
    //     const float* __restrict__ X,
    //     const int*   __restrict__ labels,
    //     float*       __restrict__ newCentroids,
    //     int*         __restrict__ counts,
    //     int n, int d, int k)
    // {
    //     extern __shared__ float smem[];
    //     float* s_accum  = smem;                   // [k*d]
    //     int*   s_counts = (int*)(smem + k * d);   // [k]

    //     const int tid = threadIdx.x;

    //     for (int i = tid; i < k * d; i += blockDim.x) s_accum[i]  = 0.0f;
    //     for (int i = tid; i < k;     i += blockDim.x) s_counts[i] = 0;
    //     __syncthreads();

    //     const int gid = blockIdx.x * blockDim.x + tid;
    //     if (gid < n) {
    //         const int c = labels[gid];
    //         atomicAdd(&s_counts[c], 1);
    //         const float* xi = X + (size_t)gid * d;
    //         for (int dim = 0; dim < d; ++dim)
    //             atomicAdd(&s_accum[c * d + dim], xi[dim]);
    //     }
    //     __syncthreads();

    //     for (int i = tid; i < k * d; i += blockDim.x)
    //         atomicAdd(&newCentroids[i], s_accum[i]);
    //     for (int i = tid; i < k; i += blockDim.x)
    //         atomicAdd(&counts[i], s_counts[i]);
    // }

    // // ═══════════════════════════════════════════════════════════════
    // //  PATH C ACCUMULATE : global atomics only
    // //
    // //  Counts still go through shared mem (always tiny).
    // //  Centroid sums use global atomics – contention is low because
    // //  k·d is large (that's why we're on this path).
    // // ═══════════════════════════════════════════════════════════════
    // __global__ void globalAccumKernel(
    //     const float* __restrict__ X,
    //     const int*   __restrict__ labels,
    //     float*       __restrict__ newCentroids,
    //     int*         __restrict__ counts,
    //     int n, int d, int k)
    // {
    //     extern __shared__ int s_counts[];         // [k]

    //     const int tid = threadIdx.x;
    //     for (int i = tid; i < k; i += blockDim.x) s_counts[i] = 0;
    //     __syncthreads();

    //     const int gid = blockIdx.x * blockDim.x + tid;
    //     if (gid < n) {
    //         const int c = labels[gid];
    //         atomicAdd(&s_counts[c], 1);
    //         const float* xi = X + (size_t)gid * d;
    //         for (int dim = 0; dim < d; ++dim)
    //             atomicAdd(&newCentroids[c * d + dim], xi[dim]);
    //     }
    //     __syncthreads();

    //     for (int i = tid; i < k; i += blockDim.x)
    //         atomicAdd(&counts[i], s_counts[i]);
    // }

    // // ═══════════════════════════════════════════════════════════════
    // //  DIVIDE KERNEL  –  newCentroids /= counts  (shared across paths)
    // // ═══════════════════════════════════════════════════════════════
    // __global__ void divideCentroidsKernel(
    //     float*       __restrict__ newCentroids,
    //     const int*   __restrict__ counts,
    //     const float* __restrict__ oldCentroids,
    //     int k, int d)
    // {
    //     const int j   = blockIdx.x;
    //     const int tid = threadIdx.x;
    //     if (j >= k) return;

    //     const int c = counts[j];
    //     if (c > 0) {
    //         const float inv = 1.0f / (float)c;
    //         for (int dim = tid; dim < d; dim += blockDim.x)
    //             newCentroids[j * d + dim] *= inv;
    //     } else {
    //         for (int dim = tid; dim < d; dim += blockDim.x)
    //             newCentroids[j * d + dim] = oldCentroids[j * d + dim];
    //     }
    // }

    // // ═══════════════════════════════════════════════════════════════
    // //  HOST ENTRY POINT
    // // ═══════════════════════════════════════════════════════════════
    // KMeansResult fitKMeansCuda(
    //     const Matrix& X, int n, int d,
    //     int n_clusters, int n_iter, uint64_t random_state)
    // {
    //     const int k = n_clusters;

    //     // ── query device shared-memory limit ──
    //     int device = 0;
    //     CUDA_CHECK(cudaGetDevice(&device));
    //     int smem_bytes = 0;
    //     CUDA_CHECK(cudaDeviceGetAttribute(
    //         &smem_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
    //     const int smem_floats = smem_bytes / (int)sizeof(float);

    //     // ── decide execution path ──
    //     //   fused  needs  2·k·d floats + k ints   in shared mem
    //     //   shared accum  needs  k·d floats + k ints
    //     //   global accum  needs  k ints  (always fits)
    //     const size_t fused_smem_bytes = (size_t)(2 * k * d) * sizeof(float)
    //                                 + (size_t)k * sizeof(int);
    //     const size_t accum_smem_bytes = (size_t)(k * d) * sizeof(float)
    //                                 + (size_t)k * sizeof(int);

    //     enum class Path { FUSED, TILED_SHARED, TILED_GLOBAL };
    //     Path path;
    //     if (fused_smem_bytes <= (size_t)smem_bytes)
    //         path = Path::FUSED;
    //     else if (accum_smem_bytes <= (size_t)smem_bytes)
    //         path = Path::TILED_SHARED;
    //     else
    //         path = Path::TILED_GLOBAL;

    //     // Tile size for tiled assign kernels (paths B & C)
    //     const int tile_k    = (d > 0) ? std::max(1, smem_floats / d) : k;
    //     const int eff_tile  = std::min(tile_k, k);
    //     const size_t tile_assign_smem = (size_t)eff_tile * d * sizeof(float);
    //     const bool tile_fits = (d <= smem_floats);  // at least 1 row fits

    //     // ── allocate device memory ──
    //     float *d_X = nullptr, *d_C = nullptr, *d_newC = nullptr;
    //     int   *d_labels = nullptr, *d_counts = nullptr;

    //     try {
    //         CUDA_CHECK(cudaMalloc(&d_X,      (size_t)n * d * sizeof(float)));
    //         CUDA_CHECK(cudaMalloc(&d_C,      (size_t)k * d * sizeof(float)));
    //         CUDA_CHECK(cudaMalloc(&d_newC,   (size_t)k * d * sizeof(float)));
    //         CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));
    //         CUDA_CHECK(cudaMalloc(&d_counts, (size_t)k * sizeof(int)));

    //         CUDA_CHECK(cudaMemcpy(d_X, X.data(),
    //                             (size_t)n * d * sizeof(float),
    //                             cudaMemcpyHostToDevice));

    //         // ── init centroids: random sample ──
    //         std::mt19937_64 rng(random_state);
    //         KMeans seeder(k, /*n_iter=*/0, /*n_local_trials=*/-1, random_state);
    //         KMeansResult seed = seeder.initCentroids(X, n, d, rng);
    //         CUDA_CHECK(cudaMemcpy(d_C, seed.centroids.data(),
    //                             (size_t)k * d * sizeof(float),
    //                             cudaMemcpyHostToDevice));

    //         const int nblocks = (n + BLOCK - 1) / BLOCK;

    //         // ── iterate ──
    //         for (int iter = 0; iter < n_iter; ++iter) {

    //             CUDA_CHECK(cudaMemsetAsync(d_newC,   0, (size_t)k * d * sizeof(float)));
    //             CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)k * sizeof(int)));

    //             switch (path) {

    //             case Path::FUSED:
    //                 fusedAssignAccumKernel<<<nblocks, BLOCK, fused_smem_bytes>>>(
    //                     d_X, d_C, d_labels, d_newC, d_counts, n, d, k);
    //                 CUDA_CHECK_KERNEL();
    //                 break;

    //             case Path::TILED_SHARED:
    //                 if (tile_fits) {
    //                     tiledAssignKernel<<<nblocks, BLOCK, tile_assign_smem>>>(
    //                         d_X, d_C, d_labels, n, d, k, eff_tile);
    //                 } else {
    //                     globalAssignKernel<<<nblocks, BLOCK>>>(
    //                         d_X, d_C, d_labels, n, d, k);
    //                 }
    //                 CUDA_CHECK_KERNEL();

    //                 sharedAccumKernel<<<nblocks, BLOCK, accum_smem_bytes>>>(
    //                     d_X, d_labels, d_newC, d_counts, n, d, k);
    //                 CUDA_CHECK_KERNEL();
    //                 break;

    //             case Path::TILED_GLOBAL: {
    //                 if (tile_fits) {
    //                     tiledAssignKernel<<<nblocks, BLOCK, tile_assign_smem>>>(
    //                         d_X, d_C, d_labels, n, d, k, eff_tile);
    //                 } else {
    //                     globalAssignKernel<<<nblocks, BLOCK>>>(
    //                         d_X, d_C, d_labels, n, d, k);
    //                 }
    //                 CUDA_CHECK_KERNEL();

    //                 size_t ga_smem = (size_t)k * sizeof(int);
    //                 globalAccumKernel<<<nblocks, BLOCK, ga_smem>>>(
    //                     d_X, d_labels, d_newC, d_counts, n, d, k);
    //                 CUDA_CHECK_KERNEL();
    //                 break;
    //             }
    //             }

    //             divideCentroidsKernel<<<k, BLOCK>>>(
    //                 d_newC, d_counts, d_C, k, d);
    //             CUDA_CHECK_KERNEL();

    //             std::swap(d_C, d_newC);
    //         }

    //         // ── download results ──
    //         Matrix centroids(k * d);
    //         std::vector<int> labels(n);
    //         CUDA_CHECK(cudaMemcpy(centroids.data(), d_C,
    //                             (size_t)k * d * sizeof(float),
    //                             cudaMemcpyDeviceToHost));
    //         CUDA_CHECK(cudaMemcpy(labels.data(), d_labels,
    //                             (size_t)n * sizeof(int),
    //                             cudaMemcpyDeviceToHost));

    //         cudaFree(d_X);     cudaFree(d_C);       cudaFree(d_newC);
    //         cudaFree(d_labels); cudaFree(d_counts);

    //         return {centroids, labels, k, d};
    //     }
    //     catch (...) {
    //         cudaFree(d_X);     cudaFree(d_C);       cudaFree(d_newC);
    //         cudaFree(d_labels); cudaFree(d_counts);
    //         throw;
    //     }
    // }


    // src/kmeans_cuda.cu
#include "kmeans.hpp"
#include "kmeans_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <algorithm>
#include <random>
#include <stdexcept>

// ---------------- Error checking ----------------
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        throw std::runtime_error("CUDA error");                         \
    }                                                                   \
} while(0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

#define CUBLAS_CHECK_OR_THROW(call) do {                                \
    cublasStatus_t s = (call);                                          \
    if (s != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                     \
                __FILE__, __LINE__, (int)s);                            \
        throw std::runtime_error("cuBLAS error");                       \
    }                                                                   \
} while(0)

// ---------------- Kernels ----------------

__global__ void transposeAoS2SoA(const float* __restrict__ in, float* __restrict__ out, int n, int d) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (size_t)n * d) {
        int r = idx / d; 
        int c = idx % d; 
        out[c * (size_t)n + r] = in[idx];
    }
}

__global__ void addSqNormsKernel(
    float* D, const float* x_sq, const float* c_sq, int n, int m)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m) return;
    int i = idx / m;
    int j = idx % m;
    D[idx] += x_sq[i] + c_sq[j];
}

__global__ void computeSqNormsKernel(
    const float* X, float* sqnorms, int n, int d)
{
    int i = blockIdx.x;
    if (i >= n) return;
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int k = tid; k < d; k += blockDim.x) {
        float v = X[i * d + k];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) sqnorms[i] = sdata[0];
}

__global__ void argminRowKernel(
    const float* D, int* labels, int n, int m)
{
    int i = blockIdx.x;
    if (i >= n) return;
    __shared__ float s_min[256];
    __shared__ int   s_idx[256];
    int tid = threadIdx.x;
    float local_min = FLT_MAX;
    int   local_idx = 0;
    for (int j = tid; j < m; j += blockDim.x) {
        float v = D[i * m + j];
        if (v < local_min) { local_min = v; local_idx = j; }
    }
    s_min[tid] = local_min;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && s_min[tid + s] < s_min[tid]) {
            s_min[tid] = s_min[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) labels[i] = s_idx[0];
}

// ===== The New Warp-Optimized Accumulation Kernel =====
__global__ void accumulateCentroidsSharedKernel(
    const float* __restrict__ X_SoA, const int* __restrict__ labels,
    float* __restrict__ newCentroids, int* __restrict__ counts,
    int n, int d, int k)
{
    extern __shared__ float smem[];
    float* s_accum = smem;
    int* s_counts = (int*)(s_accum + k * d);

    int tid = threadIdx.x;
    int lane = tid % 32;

    for (int i = tid; i < k * d; i += blockDim.x) s_accum[i] = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) s_counts[i] = 0;
    __syncthreads();

    size_t gid = (size_t)blockIdx.x * blockDim.x + tid;
    if (gid < n) {
        int best_c = labels[gid];
        
        // Fast warp match for counts (Safely masked to prevent deadlocks)
        unsigned active_lanes = __activemask();
        uint32_t match_mask = __match_any_sync(active_lanes, best_c);
        int leader = __ffs(match_mask) - 1;
        if (lane == leader) {
            int pop_count = __popc(match_mask);
            atomicAdd(&s_counts[best_c], pop_count);
        }

        // Staggered shared memory atomics for fast SoA reads
        #pragma unroll
        for (int i = 0; i < d; ++i) {
            int dim = (lane + i) % d;
            atomicAdd(&s_accum[best_c * d + dim], X_SoA[dim * (size_t)n + gid]);
        }
    }
    __syncthreads();

    // Push block totals to global memory
    for (int i = tid; i < k * d; i += blockDim.x)
        atomicAdd(&newCentroids[i], s_accum[i]);
    for (int i = tid; i < k; i += blockDim.x)
        atomicAdd(&counts[i], s_counts[i]);
}

__global__ void divideCentroidsKernel(
    float* newCentroids, const int* counts,
    const float* oldCentroids,
    int k, int d)
{
    int j = blockIdx.x;
    int tid = threadIdx.x;
    if (j >= k) return;
    int c = counts[j];
    if (c > 0) {
        float inv = 1.0f / (float)c;
        for (int dim = tid; dim < d; dim += blockDim.x)
            newCentroids[j * d + dim] *= inv;
    } else {
        for (int dim = tid; dim < d; dim += blockDim.x)
            newCentroids[j * d + dim] = oldCentroids[j * d + dim];
    }
}

// ---------------- Helpers ----------------

static size_t accumShmemBytes(int k, int d) {
    return (size_t)(k * d) * sizeof(float) + (size_t)k * sizeof(int);
}

// ---------------- Public entry point ----------------

KMeansResult fitKMeansCuda(
    const Matrix& X, int n, int d,
    int n_clusters, int n_iter, uint64_t random_state)
{
    const int k = n_clusters;

    float *d_X = nullptr, *d_X_SoA = nullptr, *d_C = nullptr, *d_newC = nullptr;
    float *d_D = nullptr, *d_X_sqnorm = nullptr, *d_C_sqnorm = nullptr;
    int   *d_labels = nullptr, *d_counts = nullptr;
    cublasHandle_t handle = nullptr;

    try {
        size_t accum_shmem = accumShmemBytes(k, d);
        cudaFuncSetAttribute((const void*)accumulateCentroidsSharedKernel, 
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 
                             (int)accum_shmem);

        // Allocations
        CUDA_CHECK(cudaMalloc(&d_X,      (size_t)n * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_X_SoA,  (size_t)n * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,      (size_t)k * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_newC,   (size_t)k * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counts, (size_t)k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_D,        (size_t)n * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_X_sqnorm, (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_sqnorm, (size_t)k * sizeof(float)));
        
        CUBLAS_CHECK_OR_THROW(cublasCreate(&handle));

        // Copy input and generate SoA format
        CUDA_CHECK(cudaMemcpy(d_X, X.data(), (size_t)n * d * sizeof(float), cudaMemcpyHostToDevice));
        
        int tp_block = 256;
        int tp_grid = ((size_t)n * d + tp_block - 1) / tp_block;
        transposeAoS2SoA<<<tp_grid, tp_block>>>(d_X, d_X_SoA, n, d);
        CUDA_CHECK_KERNEL();

        // CPU Seed Parity
        std::mt19937_64 rng(random_state);
        KMeans seeder(k, 0, -1, random_state);
        KMeansResult seed = seeder.initCentroids(X, n, d, rng);
        CUDA_CHECK(cudaMemcpy(d_C, seed.centroids.data(), (size_t)k * d * sizeof(float), cudaMemcpyHostToDevice));

        // Precompute X norms (Never changes)
        computeSqNormsKernel<<<n, 256>>>(d_X, d_X_sqnorm, n, d);
        CUDA_CHECK_KERNEL();

        const int block = 256;
        const int grid  = (n + block - 1) / block;

        for (int iter = 0; iter < n_iter; iter++) {
            CUDA_CHECK(cudaMemsetAsync(d_newC,   0, (size_t)k * d * sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(d_counts, 0, (size_t)k * sizeof(int)));

            // 1. cuBLAS Tensor Core Matrix Multiplication for Distances
            computeSqNormsKernel<<<k, 256>>>(d_C, d_C_sqnorm, k, d);
            CUDA_CHECK_KERNEL();

            const float alpha = -2.0f, beta = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        k, n, d, &alpha,
                        d_C, d, d_X, d,
                        &beta, d_D, k);
            
            int total_dist = n * k;
            addSqNormsKernel<<<(total_dist + 255)/256, 256>>>(d_D, d_X_sqnorm, d_C_sqnorm, n, k);
            CUDA_CHECK_KERNEL();

            // 2. Assign Labels
            argminRowKernel<<<n, 256>>>(d_D, d_labels, n, k);
            CUDA_CHECK_KERNEL();

            // 3. Ultra-Fast Shared Memory Accumulation
            accumulateCentroidsSharedKernel<<<grid, block, accum_shmem>>>(
                d_X_SoA, d_labels, d_newC, d_counts, n, d, k);
            CUDA_CHECK_KERNEL();

            // 4. Divide and Swap
            divideCentroidsKernel<<<k, 256>>>(d_newC, d_counts, d_C, k, d);
            CUDA_CHECK_KERNEL();

            std::swap(d_C, d_newC);
        }

        Matrix centroids((size_t)k * d);
        std::vector<int> labels(n);
        CUDA_CHECK(cudaMemcpy(centroids.data(), d_C, (size_t)k * d * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(labels.data(), d_labels, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

        cublasDestroy(handle);
        cudaFree(d_X); cudaFree(d_X_SoA); cudaFree(d_C); cudaFree(d_newC);
        cudaFree(d_labels); cudaFree(d_counts); cudaFree(d_D);
        cudaFree(d_X_sqnorm); cudaFree(d_C_sqnorm);

        return {centroids, labels, k, d};
    }
    catch (...) {
        if (handle) cublasDestroy(handle);
        if (d_X) cudaFree(d_X); if (d_X_SoA) cudaFree(d_X_SoA); 
        if (d_C) cudaFree(d_C); if (d_newC) cudaFree(d_newC);
        if (d_labels) cudaFree(d_labels); if (d_counts) cudaFree(d_counts);
        if (d_D) cudaFree(d_D); if (d_X_sqnorm) cudaFree(d_X_sqnorm); 
        if (d_C_sqnorm) cudaFree(d_C_sqnorm);
        throw;
    }
}