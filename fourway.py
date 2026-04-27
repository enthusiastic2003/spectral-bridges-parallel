import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering as SklearnSpectral
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Author's implementation
from sbcluster import SpectralBridges
# Your C++/CUDA implementation
import sys
import os
sys.path.append(os.path.abspath("build/"))
import specbridge


def parse_args():
    parser = argparse.ArgumentParser(description="Four-way spectral clustering benchmark")
    parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip the standard sklearn spectral clustering baseline",
    )
    parser.add_argument(
        "--no-author",
        action="store_true",
        help="Skip the author's sbcluster implementation",
    )
    parser.add_argument(
        "--no-cpu",
        action="store_true",
        help="Skip our CPU implementation",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip our GPU implementation",
    )
    return parser.parse_args()


def run_four_way_benchmark():
    n_samples = 10_000
    n_clusters = 2
    
    print(f"Generating Moons dataset (N={n_samples})...")
    X, y_true = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    X = np.ascontiguousarray(X, dtype=np.float32)

    # ---------------------------------------------------------
    # 1. Baseline: Standard Full Spectral Clustering
    # ---------------------------------------------------------
    print("\nRunning Standard Spectral Clustering (O(N^3) Baseline)...")
    t0_sc = time.time()
    # standard_sc = SklearnSpectral(
    #     n_clusters=n_clusters, 
    #     affinity='rbf', 
    #     gamma=78.68808875036952,  # Tuned specifically for this scale
    #     random_state=42,
    #     assign_labels='discretize',
    #     eigen_solver='amg',           # Uses pyamg for faster sparse eigendecomposition
    # )
    # # The ultimate, fully-optimized sklearn baseline
    standard_sc = SklearnSpectral(
        n_clusters=n_clusters, 
        affinity='nearest_neighbors', 
        n_neighbors=50,               # Tuned to maintain that 0.99 ARI
        eigen_solver='amg',           # Uses pyamg for faster sparse eigendecomposition
        assign_labels='cluster_qr',   # Bypasses the k-means convergence bottleneck
        n_jobs=-1,                    # Uses all CPU cores for the neighbor search
        random_state=42
    )
    sc_labels = standard_sc.fit_predict(X)
    time_sc = time.time() - t0_sc
    
    ari_sc = adjusted_rand_score(y_true, sc_labels)
    nmi_sc = normalized_mutual_info_score(y_true, sc_labels)
    
    print(f"Standard SC -> ARI: {ari_sc:.4f} | NMI: {nmi_sc:.4f} | Time: {time_sc:.4f} s")

    # ---------------------------------------------------------
    # 2. Setup for Spectral Bridges
    # ---------------------------------------------------------
    m_values = [10, 50, 100, 200, 400, 800, 1200]
    
    metrics = {
        'author': {'ari': [], 'nmi': [], 'speedup': []},
        'ours_cpu': {'ari': [], 'nmi': [], 'speedup': []},
        'ours_gpu': {'ari': [], 'nmi': [], 'speedup': []}
    }
    
    print("\nStarting Spectral Bridges Benchmark...")
    print("-" * 105)
    print(f"{'m':<5} | {'ARI (Auth/CPU/GPU)':<25} | {'NMI (Auth/CPU/GPU)':<25} | {'Speedup over SC (Auth/CPU/GPU)':<30}")
    print("-" * 105)

    for m in m_values:
        # A. Author's Implementation
        author_model = SpectralBridges(
            n_clusters=n_clusters, n_nodes=m, perplexity=2.0, n_iter=20, random_state=42
        )
        t0 = time.time()
        author_model_labels = author_model.fit_predict(X)

        t_auth = time.time() - t0
        ari_auth = adjusted_rand_score(y_true, author_model_labels)
        nmi_auth = normalized_mutual_info_score(y_true, author_model_labels)
        
        # B. Our Implementation (CPU - OpenMP using all available threads)
        cpu_model = specbridge.SpectralClustering(
            n_clusters=n_clusters, num_voronoi=m, n_iter=20, target_perplexity=2.0, random_state=42, use_gpu=False
        )
        t0 = time.time()
        res_cpu = cpu_model.fit(X)
        t_cpu = time.time() - t0
        ari_cpu = adjusted_rand_score(y_true, res_cpu.labels)
        nmi_cpu = normalized_mutual_info_score(y_true, res_cpu.labels)
        
        # C. Our Implementation (GPU - CUDA)
        gpu_model = specbridge.SpectralClustering(
            n_clusters=n_clusters, num_voronoi=m, n_iter=20, target_perplexity=2.0, random_state=42, use_gpu=True
        )
        t0 = time.time()
        res_gpu = gpu_model.fit(X)
        t_gpu = time.time() - t0
        ari_gpu = adjusted_rand_score(y_true, res_gpu.labels)
        nmi_gpu = normalized_mutual_info_score(y_true, res_gpu.labels)
        
        # Store metrics (Speedup = Standard Time / New Time)
        metrics['author']['ari'].append(ari_auth)
        metrics['author']['nmi'].append(nmi_auth)
        metrics['author']['speedup'].append(time_sc / t_auth)

        metrics['ours_cpu']['ari'].append(ari_cpu)
        metrics['ours_cpu']['nmi'].append(nmi_cpu)
        metrics['ours_cpu']['speedup'].append(time_sc / t_cpu)

        metrics['ours_gpu']['ari'].append(ari_gpu)
        metrics['ours_gpu']['nmi'].append(nmi_gpu)
        metrics['ours_gpu']['speedup'].append(time_sc / t_gpu)
        
        # Print row
        ari_str = f"{ari_auth:.3f} / {ari_cpu:.3f} / {ari_gpu:.3f}"
        nmi_str = f"{nmi_auth:.3f} / {nmi_cpu:.3f} / {nmi_gpu:.3f}"
        spd_str = f"{time_sc/t_auth:.1f}x / {time_sc/t_cpu:.1f}x / {time_sc/t_gpu:.1f}x"
        print(f"{m:<5} | {ari_str:<25} | {nmi_str:<25} | {spd_str:<30}")

    # ---------------------------------------------------------
    # 3. Plotting the Results
    # ---------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: ARI
    ax1.axhline(y=ari_sc, color='gray', linestyle=':', label='Standard SC')
    ax1.plot(m_values, metrics['author']['ari'], label="Author's (Python)", marker='o', linestyle='--')
    ax1.plot(m_values, metrics['ours_cpu']['ari'], label='Ours (C++/OpenMP)', marker='s', alpha=0.8)
    ax1.plot(m_values, metrics['ours_gpu']['ari'], label='Ours (C++/CUDA)', marker='^', alpha=0.8)
    ax1.set_title('Clustering Accuracy (ARI)', fontsize=12)
    ax1.set_xlabel('m (Voronoi Regions)')
    ax1.set_ylabel('Adjusted Rand Index')
    ax1.legend()
    
    # Plot 2: NMI
    ax2.axhline(y=nmi_sc, color='gray', linestyle=':', label='Standard SC')
    ax2.plot(m_values, metrics['author']['nmi'], label="Author's (Python)", marker='o', linestyle='--')
    ax2.plot(m_values, metrics['ours_cpu']['nmi'], label='Ours (C++/OpenMP)', marker='s', alpha=0.8)
    ax2.plot(m_values, metrics['ours_gpu']['nmi'], label='Ours (C++/CUDA)', marker='^', alpha=0.8)
    ax2.set_title('Clustering Quality (NMI)', fontsize=12)
    ax2.set_xlabel('m (Voronoi Regions)')
    ax2.set_ylabel('Normalized Mutual Information')
    ax2.legend()
    
    # Plot 3: Speedup
    ax3.axhline(y=1.0, color='gray', linestyle=':', label='Standard SC (1x)')
    ax3.plot(m_values, metrics['author']['speedup'], label="Author's (Python)", marker='o', linestyle='--')
    ax3.plot(m_values, metrics['ours_cpu']['speedup'], label='Ours (C++/OpenMP)', marker='s')
    ax3.plot(m_values, metrics['ours_gpu']['speedup'], label='Ours (C++/CUDA)', marker='^')
    ax3.set_title(f'Speedup Multiplier vs Standard SC (N={n_samples})', fontsize=12)
    ax3.set_xlabel('m (Voronoi Regions)')
    ax3.set_ylabel('Speedup (X times faster)')
    ax3.set_yscale('log') # Log scale is best to see the massive GPU scaling
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("four_way_comparison.png", dpi=300)
    print("\nBenchmark complete! Results saved to 'four_way_comparison.png'.")

if __name__ == "__main__":
    run_four_way_benchmark()