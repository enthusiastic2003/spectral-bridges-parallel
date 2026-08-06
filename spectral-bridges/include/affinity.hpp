#pragma once
#include "kmeans.hpp"

// Returns flat [n_nodes × n_nodes] affinity matrix
Matrix computeAffinity(
    const Matrix& X,          // full dataset [n × d]
    const KMeansResult& km,   // centroids [m × d] + labels [n]
    int n, int m, int d,
    float M                   // scaling factor (default 1e4)
);