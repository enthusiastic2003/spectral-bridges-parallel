# %%
import sys
import os

# 1. Add the build folder to Python's search path
sys.path.append(os.path.abspath("bin/"))

# %%
import specbridge as sb

# %%
import numpy as np
import matplotlib.pyplot as plt
# Reproducible synthetic data from 4 clearly separated 2D clusters
rng = np.random.default_rng(42)
points_per_cluster = 5_000_000
cluster_std = 0.8
centers = np.array([
    [-6.0, -6.0],
    [ 6.0, -6.0],
    [-6.0,  6.0],
    [ 6.0,  6.0],
])

X_parts = []
for c in centers:
    X_parts.append(rng.normal(loc=c, scale=cluster_std, size=(points_per_cluster, 2)))

X = np.vstack(X_parts).astype(np.float32)

# %%
bridge = sb.SpectralClustering(
    n_clusters=4,
    num_voronoi = 1000,
    n_iter = 20,
    M = 10e4,
    random_state=42
)

# %%
result = bridge.fit(X)

