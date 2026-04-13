#include "kmeans.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <omp.h>
#include <iostream>

KMeans::KMeans(int n_clusters, int n_iter, int n_local_trials, uint64_t random_state)
    : n_clusters(n_clusters), n_iter(n_iter),
      n_local_trials(n_local_trials), random_state(random_state) {}

// Euclidean squared distances between every row of X (n×d) and every row of C (m×d)
// Returns flat [n × m] matrix
std::vector<float> KMeans::pairwiseDists(const Matrix& X, const Matrix& C,
                                          int n, int m, int d) {
    std::vector<float> D(n * m, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float dist = 0.0f;
            for (int k = 0; k < d; k++) {
                float diff = X[i*d + k] - C[j*d + k];
                dist += diff * diff;
            }
            D[i*m + j] = dist;
        }
    }
    return D;
}

KMeansResult KMeans::initCentroids(const Matrix& X, int n, int d, std::mt19937_64& rng) {
    int trials = (n_local_trials < 0)
        ? (2 + static_cast<int>(std::log(n_clusters)))
        : n_local_trials;

    Matrix centroids(n_clusters * d);

    std::uniform_int_distribution<int> uni(0, n - 1);
    int first = uni(rng);
    std::copy_n(X.begin() + first * d, d, centroids.begin());

    std::vector<float> minDists(n, std::numeric_limits<float>::max());

    std::vector<int> labels(n, -1);

    for (int c = 1; c < n_clusters + 1; c++) {

        const float* last = centroids.data() + (c - 1) * d;
        // Embarrassingly parallel — each i writes only to minDists[i]
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            float dist = 0.0f;
            for (int k = 0; k < d; k++) {
                float diff = X[i*d + k] - last[k];
                dist += diff * diff;
            }
            // minDists[i] = std::min(minDists[i], dist);
            if (dist < minDists[i]) {
                minDists[i] = dist;
                labels[i] = c - 1;
            }

        }

        if (c == n_clusters) break;

        std::discrete_distribution<int> weighted(minDists.begin(), minDists.end());
        int bestCandidate = -1;
        float bestInertia = std::numeric_limits<float>::max();

        for (int t = 0; t < trials; t++) {
            int cand = weighted(rng);
            const float* candPtr = X.data() + cand * d;

            // Embarrassingly parallel — reduction over independent per-point contributions
            float inertia = 0.0f;
            #pragma omp parallel for reduction(+:inertia) schedule(static)
            for (int i = 0; i < n; i++) {
                float dist = 0.0f;
                for (int k = 0; k < d; k++) {
                    float diff = X[i*d + k] - candPtr[k];
                    dist += diff * diff;
                }
                inertia += std::min(minDists[i], dist);
            }

            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestCandidate = cand;
            }
        }

        std::copy_n(X.begin() + bestCandidate * d, d,
                    centroids.begin() + c * d);
    }

    return {centroids, labels, n_clusters, d};
}

KMeansResult KMeans::fit(const Matrix& X, int n, int d) {
    std::cout << "Total processors: " << omp_get_num_procs() << ", threads: " << omp_get_max_threads() << std::endl;
    std::mt19937_64 rng(random_state);
    KMeansResult result = initCentroids(X, n, d, rng);
    Matrix centroids = result.centroids;
    std::cout << "Centroids initialized using k-means++." << std::endl;
    std::vector<int> labels(n);

    for (int iter = 0; iter < n_iter; iter++) {
        // Assignment step: find nearest centroid for each point
        #pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < n; i++) {
            float best = std::numeric_limits<float>::max();
            int bestIdx = 0;
            for (int j = 0; j < n_clusters; j++) {
                float dist = 0.0f;
                for (int k = 0; k < d; k++) {
                    float diff = X[i*d + k] - centroids[j*d + k];
                    dist += diff * diff;
                }
                if (dist < best) { best = dist; bestIdx = j; }
            }
            labels[i] = bestIdx;
        }

        // Update step: recompute centroids as cluster means
        Matrix newCentroids(n_clusters * d, 0.0f);
        std::vector<int> counts(n_clusters, 0);
        #pragma omp parallel
        {
            Matrix localCentroids(n_clusters * d, 0.0f);
            std::vector<int> localCounts(n_clusters, 0);

            #pragma omp for schedule(dynamic, 128)
            for (int i = 0; i < n; i++) {
                int c = labels[i];
                localCounts[c]++;
                for (int k = 0; k < d; k++) {
                    localCentroids[c*d + k] += X[i*d + k];
                }
            }

            #pragma omp critical
            {
                for (int j = 0; j < n_clusters; j++) {
                    counts[j] += localCounts[j];
                    for (int k = 0; k < d; k++) {
                        newCentroids[j*d + k] += localCentroids[j*d + k];
                    }
                }
            }
        }
        for (int j = 0; j < n_clusters; j++) {
            if (counts[j] > 0)
                for (int k = 0; k < d; k++)
                    newCentroids[j*d + k] /= counts[j];
            else
                // Empty cluster: keep old centroid
                std::copy_n(centroids.begin() + j*d, d,
                            newCentroids.begin() + j*d);
        }
        centroids = std::move(newCentroids);
    }

    return {centroids, labels, n_clusters, d};
}