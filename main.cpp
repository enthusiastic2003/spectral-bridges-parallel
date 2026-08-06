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

    int n = 9, d = 2;
    int k = 2; // clusters
    int m = 3; // voronoi regions

    auto result = spectralBridges(X, n, d, k, m,
                                  1e4f, 20, 42, false);

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