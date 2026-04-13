#pragma once
#include <vector>
#include <cstdint>
#include <random>

// Row-major matrix: data[i * cols + j] = element at row i, col j
using Matrix = std::vector<float>;
using MatrixD = std::vector<double>;

struct KMeansResult {
    Matrix centroids;    // shape: [n_clusters × dim]
    std::vector<int> labels; // shape: [n_points]
    int centroid_rows = 0;
    int centroid_cols = 0;
};

class KMeans {
public:
    int n_clusters;
    int n_iter;
    int n_local_trials;
    uint64_t random_state;

    KMeans(int n_clusters, int n_iter = 20,
           int n_local_trials = -1,   // -1 = auto: 2 + log(k)
           uint64_t random_state = 42);

    KMeansResult fit(const Matrix& X, int n, int d);
    KMeansResult initCentroids(const Matrix& X, int n, int d, std::mt19937_64& rng);



private:
    std::vector<float> pairwiseDists(const Matrix& X, const Matrix& C,
                                      int n, int m, int d);
};