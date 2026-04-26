#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include "spectral.hpp"
#include <algorithm>

// Load a CSV into a flat float matrix, returns nrows
int loadCSV(const std::string& path, Matrix& data, int& cols) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::vector<float> vals;
    std::string line;
    int rows = 0;
    cols = 0;

    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string cell;
        int c = 0;
        while (std::getline(ss, cell, ',')) {
            vals.push_back(std::stof(cell));
            c++;
        }
        if (cols == 0) cols = c;
        rows++;
    }
    data = vals;
    return rows;
}

std::vector<int> loadLabels(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    std::vector<int> labels;
    std::string line;
    while (std::getline(f, line))
        if (!line.empty())
            labels.push_back(std::stoi(line));
    return labels;
}

void loadMetrics(const std::string& path, float& ari, float& nmi) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    f >> ari >> nmi;
}

// ARI computation
// Adjusted Rand Index measures similarity between two clusterings
float computeARI(const std::vector<int>& a, const std::vector<int>& b, int n) {
    // Find number of classes in each
    int ka = *std::max_element(a.begin(), a.end()) + 1;
    int kb = *std::max_element(b.begin(), b.end()) + 1;

    // Contingency table
    std::vector<std::vector<int>> cont(ka, std::vector<int>(kb, 0));
    for (int i = 0; i < n; i++)
        cont[a[i]][b[i]]++;

    // Row and column sums
    std::vector<int> rowSum(ka, 0), colSum(kb, 0);
    for (int i = 0; i < ka; i++)
        for (int j = 0; j < kb; j++) {
            rowSum[i] += cont[i][j];
            colSum[j] += cont[i][j];
        }

    // Sum of C(nij, 2)
    float sumComb = 0.0f;
    for (int i = 0; i < ka; i++)
        for (int j = 0; j < kb; j++)
            if (cont[i][j] >= 2)
                sumComb += (float)cont[i][j] * (cont[i][j] - 1) / 2.0f;

    float sumA = 0.0f, sumB = 0.0f;
    for (int i = 0; i < ka; i++)
        if (rowSum[i] >= 2)
            sumA += (float)rowSum[i] * (rowSum[i] - 1) / 2.0f;
    for (int j = 0; j < kb; j++)
        if (colSum[j] >= 2)
            sumB += (float)colSum[j] * (colSum[j] - 1) / 2.0f;

    float total = (float)n * (n - 1) / 2.0f;
    float expected = sumA * sumB / total;
    float maxVal   = (sumA + sumB) / 2.0f;

    if (std::abs(maxVal - expected) < 1e-10f)
        return 1.0f;

    return (sumComb - expected) / (maxVal - expected);
}

// NMI computation
float computeNMI(const std::vector<int>& a, const std::vector<int>& b, int n) {
    int ka = *std::max_element(a.begin(), a.end()) + 1;
    int kb = *std::max_element(b.begin(), b.end()) + 1;

    std::vector<std::vector<int>> cont(ka, std::vector<int>(kb, 0));
    for (int i = 0; i < n; i++)
        cont[a[i]][b[i]]++;

    std::vector<int> rowSum(ka, 0), colSum(kb, 0);
    for (int i = 0; i < ka; i++)
        for (int j = 0; j < kb; j++) {
            rowSum[i] += cont[i][j];
            colSum[j] += cont[i][j];
        }

    // Mutual information
    float mi = 0.0f;
    for (int i = 0; i < ka; i++)
        for (int j = 0; j < kb; j++)
            if (cont[i][j] > 0)
                mi += (float)cont[i][j] / n
                      * std::log((float)cont[i][j] * n
                                 / ((float)rowSum[i] * colSum[j]));

    // Entropies
    float ha = 0.0f, hb = 0.0f;
    for (int i = 0; i < ka; i++)
        if (rowSum[i] > 0)
            ha -= (float)rowSum[i]/n * std::log((float)rowSum[i]/n);
    for (int j = 0; j < kb; j++)
        if (colSum[j] > 0)
            hb -= (float)colSum[j]/n * std::log((float)colSum[j]/n);

    float denom = (ha + hb) / 2.0f;
    return (denom < 1e-10f) ? 1.0f : mi / denom;
}

struct TestCase {
    std::string name;
    int k;       // n_clusters
    int m;       // n_nodes
    float minARI; // minimum acceptable ARI vs ground truth
    float minNMI;
};

void runTest(const TestCase& tc) {
    std::string base = "tests/data/" + tc.name;

    Matrix X;
    int d, n;
    n = loadCSV(base + "_X.csv", X, d);

    std::vector<int> trueLabels  = loadLabels(base + "_true.csv");
    std::vector<int> pyLabels    = loadLabels(base + "_labels.csv");

    float pyARI, pyNMI;
    loadMetrics(base + "_metrics.txt", pyARI, pyNMI);

    // Run C++ implementation
    auto result = spectralBridges(X, n, d, tc.k, tc.m,
                                   2.0, 20, 42, true);

    // Metrics vs ground truth
    float cppARI = computeARI(trueLabels, result.labels, n);
    float cppNMI = computeNMI(trueLabels, result.labels, n);

    // Metrics vs Python labels (clustering agreement)
    float vsARI = computeARI(pyLabels, result.labels, n);
    float vsNMI = computeNMI(pyLabels, result.labels, n);

    bool passQuality = (cppARI >= tc.minARI && cppNMI >= tc.minNMI);

    std::cout << "\n=== " << tc.name << " (n=" << n << ", d=" << d
              << ", k=" << tc.k << ", m=" << tc.m << ") ===\n";
    std::cout << "  vs ground truth  — ARI: " << cppARI
              << "  NMI: " << cppNMI << "\n";
    std::cout << "  vs Python output — ARI: " << vsARI
              << "  NMI: " << vsNMI << "\n";
    std::cout << "  Python reference — ARI: " << pyARI
              << "  NMI: " << pyNMI << "\n";
    std::cout << "  Ngap: " << result.ngap << "\n";
    std::cout << "  Quality check (ARI>=" << tc.minARI
              << ", NMI>=" << tc.minNMI << "): "
              << (passQuality ? "PASSED" : "FAILED") << "\n";
}

int main() {
    std::vector<TestCase> tests = {
        {"blobs",   3, 30,  0.85f, 0.85f},
        {"moons",   2, 20,  0.70f, 0.70f},
        {"circles", 2, 20,  0.70f, 0.70f},
        {"large",   5, 50,  0.80f, 0.80f},
    };

    for (auto& tc : tests)
        runTest(tc);

    std::cout << "\nDone.\n";
    return 0;
}