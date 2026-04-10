#include <iostream>
#include <cassert>
#include <cmath>
#include "kmeans.hpp"

// Check that two clearly separated clusters get correctly assigned
void test_two_clusters() {
    Matrix X = {
        0.1f, 0.2f,
        0.2f, 0.1f,
        0.15f, 0.15f,
        5.0f, 5.1f,
        5.1f, 4.9f,
        4.9f, 5.0f
    };

    KMeans km(2, 20, -1, 42);
    auto result = km.fit(X, 6, 2);

    // First 3 points should share a label, last 3 should share a label
    assert(result.labels[0] == result.labels[1]);
    assert(result.labels[1] == result.labels[2]);
    assert(result.labels[3] == result.labels[4]);
    assert(result.labels[4] == result.labels[5]);
    assert(result.labels[0] != result.labels[3]);

    std::cout << "test_two_clusters passed\n";
}

int main() {
    test_two_clusters();
    std::cout << "All tests passed\n";
    return 0;
}