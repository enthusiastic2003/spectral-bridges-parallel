# %%
import sys
import os

# 1. Add the build folder to Python's search path
sys.path.append(os.path.abspath("build/"))

# %%
import specbridge as sb

# %%
import numpy as np
import matplotlib.pyplot as plt
# Reproducible synthetic data from 4 clearly separated 2D clusters
rng = np.random.default_rng(42)
points_per_cluster = 500_000
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
    num_voronoi = 100,
    n_iter = 20,
     target_perplexity=2.0,
    random_state=42,
    use_gpu=True,  # Set to True to test GPU affinity computation
)

scaling_threads = [6]
import time
for threads in scaling_threads:
    sb.set_num_threads(threads)
    print(f"Running with {sb.get_max_threads()} threads; {sb.get_num_procs()} processors")
    start_time = time.time()
    result = bridge.fit(X)
    print("uniques: ", np.unique(result.labels))
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print("*"*40)


from sbcluster import SpectralBridges, ngap_scorer

actual_sb = SpectralBridges(
    n_clusters=4,
    n_nodes=100,
    random_state=42,
    n_iter=20
)

start_time = time.time()
y = actual_sb.fit_predict(X)
end_time = time.time()
print(f"Actual SpectralBridges completed in {end_time - start_time:.2f} seconds")


