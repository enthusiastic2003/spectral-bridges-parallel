# Spectral Bridges Parity Report

## Goal

Bring the C++/pybind implementation in this repository to behavior parity with the author's Python implementation (`sbcluster`), especially for the case where output appeared degenerate (all labels equal).

## What Was Wrong In The Initial Implementation

### 1) Affinity construction was not numerically equivalent

- Initial C++ affinity logic used a different numeric pathway than the author code.
- Key mismatch: not fully reproducing the author's stable log-domain workflow (`log`/`logsumexp` style accumulation).
- This made the C++ path more fragile for large-scale runs and could flatten graph structure.

### 2) Type/signature inconsistency in spectral API

- `spectralClustering` implementation operated on `MatrixD` (`double`) while header declaration expected `Matrix` (`float`).
- This mismatch was corrected in `include/spectral.hpp`.

### 3) OpenMP loop usage had correctness risk

- A triangular nested loop was used with `collapse(2)`, which is invalid for non-rectangular iteration spaces.
- This can produce incorrect results in practice.

### 4) Python binding exposed incorrect labels in `SBResult`

- Internally, propagation produced balanced cluster occupancy.
- But Python-side `result.labels` appeared constant due to conversion/buffer behavior in bindings.
- This was the primary reason users observed "all labels are 2" despite non-degenerate internal clustering.

### 5) Parameter/API divergence from author naming/intent

- The public parameter previously named `M` (scaling factor style) diverged from the author's `perplexity` semantics.
- The API was aligned to `target_perplexity` in both C++ and bindings.

## Changes Made To Achieve Parity

## A) Affinity parity changes (`src/affinity.cpp`, `include/affinity.hpp`)

- Switched affinity return type to `MatrixD`.
- Rewrote affinity computation to mirror author numerical semantics:
  - float64 pathway,
  - stable log-domain accumulation,
  - log-add-exp style symmetrization/normalization,
  - perplexity-based binary search scaling.
- Fixed OpenMP misuse by removing invalid `collapse(2)` on triangular loops.

## B) Spectral interface and execution alignment (`include/spectral.hpp`, `src/spectral.cpp`)

- Unified `spectralClustering` signature on `MatrixD`.
- Kept Laplacian/eigendecomposition path compatible with author behavior.
- Updated constructor and pipeline to use `target_perplexity` consistently.

## C) Binding fixes (`python/bindings.cpp`)

- Replaced manual numpy buffer-copy conversion for `SBResult.labels` and `SBResult.eigvals` with direct STL casting (`py::cast`).
- This resolved Python-visible label corruption and made `result.labels` consistent with internal C++ output.
- Updated binding argument and property names from `M` to `target_perplexity`.

## D) Supporting type updates (`include/kmeans.hpp`)

- Added shared `MatrixD` alias for consistent double-precision matrix passing where needed.

## Validation Summary

- End-to-end parity check against author `sbcluster` on synthetic 4-cluster data now matches.
- Observed outcome after fixes:
  - C++ output labels: four balanced clusters,
  - author output labels: four balanced clusters,
  - ARI between C++ and author outputs: `1.0` in tested parity script.

## Notes

- This report documents parity-related fixes and the root cause of the observed degeneracy symptom.
- Separate quality/performance behavior on non-convex datasets (for example moons/circles) may still require additional algorithmic tuning and is not the same issue as the Python-visible all-one-label bug.
