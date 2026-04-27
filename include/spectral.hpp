#pragma once
#include "kmeans.hpp"
#include "affinity.hpp"

struct SpectralResult {
    std::vector<int> labels;    // per-region cluster labels [m]
    std::vector<float> eigvals; // eigenvalues of Laplacian [m]
    float ngap;                 // normalized eigengap
};

// Top level
struct SBResult {
    std::vector<std::vector<int>> clusterPointIndices; // which points in each cluster
    std::vector<int> labels;                           // per-point label [n]
    std::vector<float> eigvals;
    float ngap;
};

class SpectralClustering {
public:
    int n_clusters;
    int n_iter;
	int num_vornoi;
    uint64_t random_state;
    float target_perplexity;
    bool use_gpu;


    SpectralClustering(int n_clusters,
                        int num_vornoi,
                       int n_iter = 20,
		                float target_perplexity = 2.0f,
                       uint64_t random_state = 42,
                       bool use_gpu = false);


    SBResult fit(
        const Matrix& X,  // [m × m]
        int n,                    // number of datapoints
		int d					// dimensionality of datapoints
    );
};

// Backward-compatible free function.
SpectralResult spectralClustering(
    const MatrixD& affinity, // [m × m]
    int m,                   // number of voronoi regions
    int k,                   // number of clusters
    int n_iter,
    uint64_t random_state
);

SBResult spectralBridges(
    const Matrix& X,
    int n, int d,
    int k,           // number of clusters
    int m,           // number of voronoi regions
    float target_perplexity,
    int n_iter,
    uint64_t random_state,
    bool use_gpu
);