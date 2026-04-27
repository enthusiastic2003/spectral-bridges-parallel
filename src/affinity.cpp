#include "affinity.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <limits>
#include <chrono>
#include <omp.h>
#include <iostream>

/*
    Compute the affinity matrix for the given data and k-means result.
    X = Matrix of shape [n × d]
    km = KMeansResult containing centroids and labels
    n = number of data points
    m = number of Voronoi regions (k-means clusters)
    d = dimensionality of data
    target_perplexity = perplexity value for scaling affinities
*/

Matrix computeAffinity(
    const Matrix& X,
    const KMeansResult& km,
    int n, int m, int d,
    float target_perplexity)
{
	float p = target_perplexity;
    // Step 1 — center each Voronoi region around its centroid
    // X_centered[i] is a flat [nᵢ × d] matrix of (x - µᵢ) for points in region i
    std::vector<Matrix> X_centered(m);
    std::vector<int> counts(m, 0);

    for (int i = 0; i < n; i++)
        counts[km.labels[i]]++;
    
    for (int i = 0; i < m; i++)
        X_centered[i].resize(counts[i] * d);

    // Fill X_centered — track insertion position per region
    std::vector<int> insertPos(m, 0);
    for (int i = 0; i < n; i++) {
        int c = km.labels[i];
        int pos = insertPos[c];
        for (int k = 0; k < d; k++)
            X_centered[c][pos * d + k] = X[i * d + k] - km.centroids[c * d + k];
        insertPos[c]++;
    }

    auto logaddexp = [](float a, float b) {
        if (a == -std::numeric_limits<float>::infinity()) return b;
        if (b == -std::numeric_limits<float>::infinity()) return a;
        float mx = std::max(a, b);
        float mn = std::min(a, b);
        return mx + std::log1p(std::exp(mn - mx));
    };

    // Step 2 — compute row-wise log affinity matrix [m × m]
    Matrix log_affinity(m * m, -std::numeric_limits<float>::infinity());
    const float tiny = std::numeric_limits<float>::min();

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        int ni = counts[i];
        if (ni == 0) {
            continue;
        }

        Eigen::MatrixXf segments(m, d);
        for (int j = 0; j < m; j++) {
            for (int kk = 0; kk < d; kk++) {
                segments(j, kk) = static_cast<float>(km.centroids[j * d + kk])
                                  - static_cast<float>(km.centroids[i * d + kk]);
            }
        }

        Eigen::VectorXf dists = segments.rowwise().squaredNorm();
        dists(i) = 1.0f;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            Xc_float(X_centered[i].data(), ni, d);
        Eigen::MatrixXf Xc = Xc_float;

        Eigen::MatrixXf projs = Xc * segments.transpose();
        projs.array().rowwise() /= dists.transpose().array();

        Eigen::ArrayXXf log_proj = projs.array().max(tiny).log();
        Eigen::ArrayXf col_max = log_proj.colwise().maxCoeff();
        Eigen::ArrayXXf shifted = p * (log_proj.rowwise() - col_max.transpose());

        for (int j = 0; j < m; j++) {
            float mx = shifted.col(j).maxCoeff();
            float lse = mx + std::log((shifted.col(j) - mx).exp().sum());
            log_affinity[i * m + j] = p * col_max(j) + lse;
        }
    }

    // Step 3 — symmetrize and normalize by counts in log-space,
    // then exponentiate exactly as Python does.
    Matrix affinity(m * m, 0.0);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            float log_sym = logaddexp(log_affinity[i * m + j], log_affinity[j * m + i])
                             - std::log(static_cast<float>(counts[i] + counts[j]));
            float sym = std::exp(log_sym / p);
            affinity[i * m + j] = sym;
            affinity[j * m + i] = sym;
        }
    }

    // Step 4 — subtract max before perplexity calibration
    float maxVal = *std::max_element(affinity.begin(), affinity.end());
    Matrix aff_float(m * m);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m * m; i++) {
        aff_float[i] = affinity[i] - maxVal;
    }

    // Step 5 — Scale affinity matrix using Perplexity Calibration (Binary Search)
    float low = 0.0f;
    float high = 1000.0f;
    float gamma = (low + high) / 2.0f;
    int max_iter = 16;
    float tol = 1e-2f;

    for (int iter = 0; iter < max_iter; ++iter) {
        float mean_entropy = 0.0f;

        #pragma omp parallel for reduction(+:mean_entropy)
        for (int i = 0; i < m; ++i) {
            float max_log_A = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < m; ++j) {
                if (i == j) continue;
                float log_A_val = gamma * aff_float[i * m + j];
                if (log_A_val > max_log_A) max_log_A = log_A_val;
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < m; ++j) {
                if (i == j) continue;
                sum_exp += std::exp(gamma * aff_float[i * m + j] - max_log_A);
            }
            float log_sum_A = max_log_A + std::log(sum_exp);

            float row_entropy = 0.0f;
            for (int j = 0; j < m; ++j) {
                if (i == j) continue;
                float log_P = gamma * aff_float[i * m + j] - log_sum_A;
                float P = std::exp(log_P);
                if (P > 0.0f) {
                    row_entropy -= P * log_P;
                }
            }
            mean_entropy += row_entropy;
        }

        mean_entropy /= static_cast<float>(m);
        float current_perp = std::exp(mean_entropy);

        if (current_perp > target_perplexity) {
            low = gamma;
        } else {
            high = gamma;
        }
        gamma = (low + high) / 2.0f;

        if (std::abs(current_perp - target_perplexity) / target_perplexity < tol) {
            break;
        }
    }

    // Step 6 — Final exponential transform (do not zero diagonal;
    // Python keeps diagonal values from exp(gamma * A_shifted)).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m * m; i++) {
        aff_float[i] = std::exp(gamma * aff_float[i]);
    }

    return aff_float;
}