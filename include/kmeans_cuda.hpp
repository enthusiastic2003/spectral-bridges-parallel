// include/kmeans_cuda.hpp
#pragma once
#include "kmeans.hpp"
#include <cstdint>

KMeansResult fitKMeansCuda(
    const Matrix& X, int n, int d,
    int n_clusters, int n_iter, uint64_t random_state);