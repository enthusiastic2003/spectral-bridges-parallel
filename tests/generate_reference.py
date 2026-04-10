import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from spectralbridges import SpectralBridges

def save_dataset(X, name):
    pd.DataFrame(X).to_csv(f"tests/data/{name}_X.csv", index=False, header=False)

def run_and_save(X, y_true, name, n_clusters, n_nodes, seed=42):
    sb = SpectralBridges(n_clusters=n_clusters, n_nodes=n_nodes, random_state=seed)
    sb.fit(X)
    labels = sb.predict(X)

    # Save data
    pd.DataFrame(X).to_csv(f"tests/data/{name}_X.csv", index=False, header=False)
    pd.DataFrame(labels).to_csv(f"tests/data/{name}_labels.csv", index=False, header=False)
    if y_true is not None:
        pd.DataFrame(y_true).to_csv(f"tests/data/{name}_true.csv", index=False, header=False)

    ari = adjusted_rand_score(y_true, labels) if y_true is not None else None
    nmi = normalized_mutual_info_score(y_true, labels) if y_true is not None else None
    print(f"{name}: ARI={ari:.4f}, NMI={nmi:.4f}")

    # Save metrics for C++ to compare against
    with open(f"tests/data/{name}_metrics.txt", "w") as f:
        f.write(f"{ari}\n{nmi}\n")

    return labels

np.random.seed(42)

import os
os.makedirs("tests/data", exist_ok=True)

# Dataset 1: Moons
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
X_moons = X_moons.astype(np.float32)
run_and_save(X_moons, y_moons, "moons", n_clusters=2, n_nodes=20)

# Dataset 2: Circles
X_circles, y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
X_circles = X_circles.astype(np.float32)
run_and_save(X_circles, y_circles, "circles", n_clusters=2, n_nodes=20)

# Dataset 3: 3 Gaussian blobs (simple, sanity check)
from sklearn.datasets import make_blobs
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
X_blobs = X_blobs.astype(np.float32)
run_and_save(X_blobs, y_blobs, "blobs", n_clusters=3, n_nodes=30)

# Dataset 4: larger blobs for stress test
X_large, y_large = make_blobs(n_samples=1000, centers=5, cluster_std=0.8, random_state=42)
X_large = X_large.astype(np.float32)
run_and_save(X_large, y_large, "large", n_clusters=5, n_nodes=50)

print("Reference data saved to tests/data/")