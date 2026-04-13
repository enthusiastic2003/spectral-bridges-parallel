#include "affinity.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <omp.h>

Matrix computeAffinity(
    const Matrix& X,
    const KMeansResult& km,
    int n, int m, int d,
    float M)
{
	float p = DEFAULT_P;
    // Step 1 — center each Voronoi region around its centroid
    // X_centered[i] is a flat [nᵢ × d] matrix of (x - µᵢ) for points in region i
    std::vector<Matrix> X_centered(m);
    std::vector<int> counts(m, 0);

    #pragma omp parallel for reduction(+:counts[:m])
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

    // Step 2 — compute raw affinity matrix [m × m]
    Matrix affinity(m * m, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        int ni = counts[i];

        // segments[j] = µⱼ - µᵢ, shape [m × d]
        // dists[j]    = ‖µⱼ - µᵢ‖², shape [m]
        std::vector<float> dists(m, 0.0f);
        Matrix segments(m * d);

        for (int j = 0; j < m; j++) {
            for (int k = 0; k < d; k++) {
                float s = km.centroids[j * d + k] - km.centroids[i * d + k];
                segments[j * d + k] = s;
                dists[j] += s * s;
            }
        }
        dists[i] = 1.0f; // avoid division by zero on diagonal

        // For each point in region i, compute projection onto every segment
        // projs[pt, j] = dot(X_centered[i][pt], segments[j]) / dists[j]
        // clipped to [0, ∞), then raised to power p, then summed over pt
        std::vector<float> aff_row(m, 0.0f);

        for (int pt = 0; pt < ni; pt++) {
            for (int j = 0; j < m; j++) {
                // dot product of point pt with segment j
                float dot = 0.0f;
                for (int k = 0; k < d; k++)
                    dot += X_centered[i][pt * d + k] * segments[j * d + k];

                // normalize and clip
                float t = dot / dists[j];
                if (t < 0.0f) t = 0.0f;

                // raise to power p and accumulate
                aff_row[j] += std::pow(t, p);
            }
        }
        for (int j = 0; j < m; j++)
            affinity[i * m + j] = aff_row[j];
    }

    // Step 3 — symmetrize and normalize by counts
    // affinity = ((A + Aᵀ) / counts) ^ (1/p)
    // where counts[i,j] = nᵢ + nⱼ
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = i; j < m; j++) {
            float sym = (affinity[i * m + j] + affinity[j * m + i])
                        / (float)(counts[i] + counts[j]);
            sym = std::pow(sym, 1.0f / p);
            affinity[i * m + j] = sym;
            affinity[j * m + i] = sym;
        }
    }

    // Step 4 — subtract 0.5 * max for numerical stability
    float maxVal = *std::max_element(affinity.begin(), affinity.end());
    for (float& v : affinity)
        v -= 0.5f * maxVal;

    // Step 5 — exponential transform: ã = exp(γ * a)
    // γ = log(M) / (q90 - q10)
    std::vector<float> sorted_aff(affinity.begin(), affinity.end());
    std::sort(sorted_aff.begin(), sorted_aff.end());
    int total = m * m;
    float q10 = sorted_aff[static_cast<int>(0.1f * total)];
    float q90 = sorted_aff[static_cast<int>(0.9f * total)];

    float gamma = std::log(M) / (q90 - q10);
    #pragma omp parallel for
    for (int i = 0; i < (int)affinity.size(); i++)
        affinity[i] = std::exp(gamma * affinity[i]);

    return affinity;
}