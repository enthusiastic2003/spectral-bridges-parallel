#include <iostream>
#include "kmeans.hpp"
#include "affinity.hpp"
#include "spectral.hpp"

int main() {
    // Small synthetic test: 6 points in 2D, expect 2 clusters
    Matrix X = {
        // region around (0, 0)
        0.1f,  0.1f,
        -0.1f, 0.1f,
        0.0f, -0.1f,
        // region around (1, 0) -- close to region 0, same cluster
        0.9f,  0.1f,
        1.1f, -0.1f,
        1.0f,  0.0f,
        // region around (10, 0) -- far away, different cluster
        9.9f,  0.1f,
        10.1f,-0.1f,
        10.0f, 0.0f,
    };

    // KMeans km(2, 20, -1, 42);
    // auto result = km.fit(X, 6, 2);

    // std::cout << "Labels: ";
    // for (int l : result.labels)
    //     std::cout << l << " ";
    // std::cout << "\n";

    // std::cout << "Centroids:\n";
    // for (int c = 0; c < 2; c++) {
    //     std::cout << "  [" << result.centroids[c*2] << ", "
    //                        << result.centroids[c*2+1] << "]\n";
    // }

    // int n = 6, d = 2, m = 2;
    // KMeans km(m, 20, -1, 42);
    // auto result = km.fit(X, n, d);

    // Matrix aff = computeAffinity(X, result, n, m, d, 2.0f, 1e4f);

    // std::cout << "Affinity matrix [" << m << "x" << m << "]:\n";
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < m; j++)
    //         std::cout << aff[i * m + j] << " ";
    //     std::cout << "\n";
    // }

    int n = 9, d = 2;
    int k = 2; // clusters
    int m = 3; // voronoi regions

    auto result = spectralBridges(X, n, d, k, m,
                                  2.0f, 1e4f, 20, 42);

    std::cout << "Point labels: ";
    for (int l : result.labels)
        std::cout << l << " ";
    std::cout << "\n";

    std::cout << "Ngap: " << result.ngap << "\n";

    // Expected: points 0-5 same label, points 6-8 different label
    bool correct = (result.labels[0] == result.labels[1] &&
                    result.labels[1] == result.labels[2] &&
                    result.labels[2] == result.labels[3] &&
                    result.labels[3] == result.labels[4] &&
                    result.labels[4] == result.labels[5] &&
                    result.labels[0] != result.labels[6]);

    std::cout << (correct ? "PASSED" : "FAILED") << "\n";
    return 0;
}