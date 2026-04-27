import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# --- Author's Provided Implementation ---
# (Assumed to be in the same directory or pasted here as SpectralBridges)
# from your_script import SpectralBridges 
from sbcluster import SpectralBridges

import sys
import os
sys.path.append(os.path.abspath("build/"))
# --- Your Hybrid CUDA/C++ Implementation ---
import specbridge

def run_mnist_comparative_benchmark():
    # 1. Dataset Configuration (Mirroring Paper Settings)
    n_samples = 20000
    n_clusters = 10
    h_dim = 64
    m_voronoi = 250
    
    print(f"Loading MNIST and applying PCA (h={h_dim})...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Preprocessing
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)
    
    np.random.seed(42)
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[indices]
    y_true = y[indices]
    
    pca = PCA(n_components=h_dim, random_state=42)
    X_pca = np.ascontiguousarray(pca.fit_transform(X_sub), dtype=np.float32)

    # 2. Run Author's Python Implementation
    print(f"Running Author's SpectralBridges (Python)...")
    author_model = SpectralBridges(
        n_clusters=n_clusters,
        n_nodes=m_voronoi,
        perplexity=2.0,
        n_iter=20,
        random_state=42
    )
    
    t0_auth = time.time()
    author_model.fit(X_pca)
    t_auth = time.time() - t0_auth
    
    ari_auth = adjusted_rand_score(y_true, author_model.labels_)
    nmi_auth = normalized_mutual_info_score(y_true, author_model.labels_)

    # 3. Run Your C++/CUDA Implementation
    print(f"Running Your SpectralClustering (CUDA)...")
    # specbridge.set_num_threads(1)  # Ensure single-threaded for fair timing
    cuda_model = specbridge.SpectralClustering(
        n_clusters=n_clusters,
        num_voronoi=m_voronoi,
        n_iter=20,
        target_perplexity=2.0,
        random_state=42,
        use_gpu=True
    )
    
    t0_cuda = time.time()
    cuda_result = cuda_model.fit(X_pca)
    t_cuda = time.time() - t0_cuda
    
    # Extract labels from your SBResult object
    cuda_labels = np.array(cuda_result.labels)
    ari_cuda = adjusted_rand_score(y_true, cuda_labels)
    nmi_cuda = normalized_mutual_info_score(y_true, cuda_labels)

    # 4. Results Table
    print("\n" + "="*60)
    print("MNIST Experiment 4: Author Python vs. Your CUDA")
    print("="*60)
    print(f"{'Metric':<10} | {'Author (Python)':<18} | {'Ours (CUDA)':<15}")
    print("-" * 60)
    print(f"{'ARI':<10} | {ari_auth:<18.4f} | {ari_cuda:<15.4f}")
    print(f"{'NMI':<10} | {nmi_auth:<18.4f} | {nmi_cuda:<15.4f}")
    print(f"{'Time (s)':<10} | {t_auth:<18.4f} | {t_cuda:<15.4f}")
    print("-" * 60)
    print(f"Speedup: {t_auth / t_cuda:.2f}x faster than Author's Python code")
    print("="*60)

if __name__ == "__main__":
    run_mnist_comparative_benchmark()