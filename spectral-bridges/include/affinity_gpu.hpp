#pragma once

#include <cstdint>
#include <vector>
#include "kmeans.hpp"

// Public types match your CPU implementation's conventions.
using Matrix  = std::vector<float>;   // row-major


// GPU-accelerated affinity computation.
//
// Inputs:
//   X                 : flat (n × d) row-major, float
//   km                : labels (length n) and centroids (m × d)
//   n, m, d           : sizes
//   target_perplexity : reused as the exponent `p` for the affinity power
//                       (this matches your CPU code's current convention;
//                        the perplexity calibration step is NOT done here)
//
// Output:
//   (m × m) row-major double affinity matrix, post-symmetrization.
//   This is the matrix that would be fed into perplexity calibration.
Matrix computeAffinityGPU(
    const Matrix& X,
    const KMeansResult& km,
    int n, int m, int d,
    float target_perplexity);