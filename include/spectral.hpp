#pragma once
#include "kmeans.hpp"
#include "affinity.hpp"

struct SpectralResult {
    std::vector<int> labels;    // per-region cluster labels [m]
    std::vector<float> eigvals; // eigenvalues of Laplacian [m]
    float ngap;                 // normalized eigengap
};

class SpectralClustering {
public:
    int n_clusters;
    int n_iter;
    uint64_t random_state;

    SpectralClustering(int n_clusters,
                       int n_iter = 20,
                       uint64_t random_state = 42);

    SpectralResult fit(
        const Matrix& affinity,  // [m × m]
        int m                    // number of regions/nodes
    ) const;
};

// Backward-compatible free function.
SpectralResult spectralClustering(
    const Matrix& affinity,  // [m × m]
    int m,                   // number of voronoi regions
    int k,                   // number of clusters
    uint64_t random_state
);

// Top level
struct SBResult {
    std::vector<std::vector<int>> clusterPointIndices; // which points in each cluster
    std::vector<int> labels;                           // per-point label [n]
    std::vector<float> eigvals;
    float ngap;
};

SBResult spectralBridges(
    const Matrix& X,
    int n, int d,
    int k,           // number of clusters
    int m,           // number of voronoi regions
    float p,
    float M,
    int n_iter,
    uint64_t random_state
);