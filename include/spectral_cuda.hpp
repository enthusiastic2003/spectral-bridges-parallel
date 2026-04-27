#pragma once
#include <cstdint>
#include <vector>

// Forward-declare your existing types if they live in another header.
// Adjust includes to match your project.
#include "kmeans_cuda.hpp"   // for KMeansResult, fitKMeansCuda, Matrix
#include "spectral.hpp"      // for SpectralResult, n_iter, print_duration

// CUDA path for spectral clustering. Performs Laplacian construction,
// symmetric eigendecomposition (cuSOLVER), row normalization, and
// downstream k-means entirely on the GPU (with a single host hop for
// the eigengap value and final labels).
SpectralResult spectralClusteringCuda(
    const Matrix& affinity,
    int m,
    int k,
    int n_iter,
    uint64_t random_state);