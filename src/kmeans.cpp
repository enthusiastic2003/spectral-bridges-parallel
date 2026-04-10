#include "kmeans.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>


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

Matrix KMeans::initCentroids(const Matrix& X, int n, int d, std::mt19937_64& rng) {
    int trials = (n_local_trials < 0)
                 ? (2 + static_cast<int>(std::log(n_clusters)))
                 : n_local_trials;

    Matrix centroids(n_clusters * d);

    // Pick first centroid uniformly at random
    std::uniform_int_distribution<int> uni(0, n - 1);
    int first = uni(rng);
    std::copy_n(X.begin() + first * d, d, centroids.begin());

    // Min distances to chosen centroids so far — init to infinity
    std::vector<float> minDists(n, std::numeric_limits<float>::max());

    for (int c = 1; c < n_clusters; c++) {
        // Update minDists using the last chosen centroid only
        const float* last = centroids.data() + (c - 1) * d;
        for (int i = 0; i < n; i++) {
            float dist = 0.0f;
            for (int k = 0; k < d; k++) {
                float diff = X[i*d + k] - last[k];
                dist += diff * diff;
            }
            minDists[i] = std::min(minDists[i], dist);
        }

        float totalDist = std::accumulate(minDists.begin(), minDists.end(), 0.0f);

        // Sample 'trials' candidates proportional to minDists
        std::discrete_distribution<int> weighted(minDists.begin(), minDists.end());
        int bestCandidate = -1;
        float bestInertia = std::numeric_limits<float>::max();

        for (int t = 0; t < trials; t++) {
            int cand = weighted(rng);
            // Compute inertia if we add this candidate
            float inertia = 0.0f;
            const float* candPtr = X.data() + cand * d;
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
    return centroids;
}

KMeansResult KMeans::fit(const Matrix& X, int n, int d) {
    std::mt19937_64 rng(random_state);
    Matrix centroids = initCentroids(X, n, d, rng);
    std::vector<int> labels(n);

    for (int iter = 0; iter < n_iter; iter++) {
        // Assignment step: find nearest centroid for each point
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
        for (int i = 0; i < n; i++) {
            int c = labels[i];
            counts[c]++;
            for (int k = 0; k < d; k++)
                newCentroids[c*d + k] += X[i*d + k];
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