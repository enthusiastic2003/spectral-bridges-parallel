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

// ---------- Error checking macros ----------
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        throw std::runtime_error("CUDA error");                         \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t s = (call);                                          \
    if (s != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                     \
                __FILE__, __LINE__, (int)s);                            \
        throw std::runtime_error("cuBLAS error");                       \
    }                                                                   \
} while(0)

#define CUDA_CHECK_KERNEL() do {                                        \
    CUDA_CHECK(cudaGetLastError());                                     \
} while(0)

// ---------- Kernel forward declarations ----------
__global__ void addSqNormsKernel(float*, const float*, const float*, int, int);
__global__ void computeSqNormsKernel(const float*, float*, int, int);
__global__ void argminRowKernel(const float*, int*, int, int);
__global__ void accumulateCentroidsKernel(const float*, const int*, float*, int*, int, int, int);
__global__ void divideCentroidsKernel(float*, const int*, const float*, int, int);

// ---------- Pairwise squared distances via cuBLAS ----------
static void pairwiseSqDistCuda(
    cublasHandle_t handle,
    const float* d_X, const float* d_C,
    const float* d_X_sqnorm, const float* d_C_sqnorm,
    float* d_D, int n, int m, int d)
{
    const float alpha = -2.0f, beta = 0.0f;
    // Row-major [n x d] and [m x d]; output row-major [n x m].
    // Interpret as column-major [d x n] and [d x m] -> output [m x n] col-major == [n x m] row-major
    CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, d,
                &alpha,
                d_C, d,
                d_X, d,
                &beta,
                d_D, m));

    int total = n * m;
    addSqNormsKernel<<<(total + 255)/256, 256>>>(d_D, d_X_sqnorm, d_C_sqnorm, n, m);
    CUDA_CHECK_KERNEL();
}

// ---------- Kernel definitions ----------
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

__global__ void accumulateCentroidsKernel(
    const float* X, const int* labels,
    float* newCentroids, int* counts,
    int n, int d, int n_clusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = labels[i];
    atomicAdd(&counts[c], 1);
    for (int k = 0; k < d; k++) {
        atomicAdd(&newCentroids[c * d + k], X[i * d + k]);
    }
}

__global__ void divideCentroidsKernel(
    float* newCentroids, const int* counts,
    const float* oldCentroids,
    int n_clusters, int d)
{
    int j = blockIdx.x;
    int tid = threadIdx.x;
    if (j >= n_clusters) return;

    int c = counts[j];
    if (c > 0) {
        float inv = 1.0f / (float)c;
        for (int k = tid; k < d; k += blockDim.x)
            newCentroids[j * d + k] *= inv;
    } else {
        for (int k = tid; k < d; k += blockDim.x)
            newCentroids[j * d + k] = oldCentroids[j * d + k];
    }
}

// ---------- Public entry point ----------
KMeansResult fitKMeansCuda(
    const Matrix& X, int n, int d,
    int n_clusters, int n_iter, uint64_t random_state)
{
    float *d_X = nullptr, *d_C = nullptr, *d_newC = nullptr;
    float *d_D = nullptr, *d_X_sqnorm = nullptr, *d_C_sqnorm = nullptr;
    int *d_labels = nullptr, *d_counts = nullptr;

    cublasHandle_t handle = nullptr;

    const int k = n_clusters;

    try {
        CUDA_CHECK(cudaMalloc(&d_X,        n * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,        n_clusters * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_newC,     n_clusters * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_D,        (size_t)n * n_clusters * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_X_sqnorm, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_sqnorm, n_clusters * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels,   n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_counts,   n_clusters * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_X, X.data(), n * d * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Precompute ||X[i]||^2
        computeSqNormsKernel<<<n, 256>>>(d_X, d_X_sqnorm, n, d);
        CUDA_CHECK_KERNEL();

        // Initialize centroids: random sample of points (simple; replace with k-means++ later)
        std::mt19937_64 rng(random_state);
            KMeans seeder(k, /*n_iter=*/0, /*n_local_trials=*/-1, random_state);
            KMeansResult seed = seeder.initCentroids(X, n, d, rng);
            CUDA_CHECK(cudaMemcpy(d_C, seed.centroids.data(),
                                (size_t)k * d * sizeof(float),
                                cudaMemcpyHostToDevice));

        CUBLAS_CHECK(cublasCreate(&handle));

        for (int iter = 0; iter < n_iter; iter++) {
            // Centroid norms change each iteration
            computeSqNormsKernel<<<n_clusters, 256>>>(d_C, d_C_sqnorm, n_clusters, d);
            CUDA_CHECK_KERNEL();

            // Pairwise distances + argmin
            pairwiseSqDistCuda(handle, d_X, d_C, d_X_sqnorm, d_C_sqnorm,
                               d_D, n, n_clusters, d);
            argminRowKernel<<<n, 256>>>(d_D, d_labels, n, n_clusters);
            CUDA_CHECK_KERNEL();

            // Update
            CUDA_CHECK(cudaMemsetAsync(d_newC,   0, n_clusters * d * sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(d_counts, 0, n_clusters * sizeof(int)));

            accumulateCentroidsKernel<<<(n + 255)/256, 256>>>(
                d_X, d_labels, d_newC, d_counts, n, d, n_clusters);
            CUDA_CHECK_KERNEL();

            divideCentroidsKernel<<<n_clusters, 256>>>(
                d_newC, d_counts, d_C, n_clusters, d);
            CUDA_CHECK_KERNEL();

            std::swap(d_C, d_newC);
        }

        Matrix centroids(n_clusters * d);
        std::vector<int> labels(n);
        CUDA_CHECK(cudaMemcpy(centroids.data(), d_C,
                              n_clusters * d * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(labels.data(), d_labels,
                              n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        cublasDestroy(handle);
        cudaFree(d_X);  cudaFree(d_C);     cudaFree(d_newC);
        cudaFree(d_D);  cudaFree(d_X_sqnorm); cudaFree(d_C_sqnorm);
        cudaFree(d_labels); cudaFree(d_counts);

        return {centroids, labels, n_clusters, d};
    }
    catch (...) {
        if (handle) cublasDestroy(handle);
        cudaFree(d_X);  cudaFree(d_C);     cudaFree(d_newC);
        cudaFree(d_D);  cudaFree(d_X_sqnorm); cudaFree(d_C_sqnorm);
        cudaFree(d_labels); cudaFree(d_counts);
        throw;
    }
}