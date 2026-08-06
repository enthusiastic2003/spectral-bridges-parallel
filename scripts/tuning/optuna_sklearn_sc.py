"""
Optuna hyperparameter tuning for sklearn's SpectralClustering on a 20k subset of MNIST.
"""

import optuna
import numpy as np
import time
import warnings
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# Suppress warnings from Arpack/LOBPCG or KMeans
warnings.filterwarnings("ignore")

# --- Configuration ---
N_SAMPLES = 20000
N_CLUSTERS = 10
SEED = 42
N_TRIALS = 50

def load_and_prepare_data():
    print("Loading MNIST dataset...")
    X_full, y_full = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    X_full = X_full.astype(np.float32) / 255.0
    y_full = y_full.astype(int)

    # Subsample
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_full), N_SAMPLES, replace=False)
    X_sub = X_full[idx]
    y_sub = y_full[idx]
    return X_sub, y_sub

print("Initializing data...")
X_sub, y_true = load_and_prepare_data()
print(f"Data loaded: {X_sub.shape} points.")

def objective(trial):
    # 1. Feature Preprocessing (PCA)
    # Applying Spectral Clustering directly on 784 dimensions can be problematic for similarity graphs
    # Search over realistic PCA dimensions matching the spectral bridges setup (8 to ~128)
    n_components = trial.suggest_int("pca_n_components", 8, 128, log=True)
    pca = PCA(n_components=n_components, random_state=SEED)
    X_processed = pca.fit_transform(X_sub)
        
    # 2. SpectralClustering Graph Construction
    # 'nearest_neighbors' builds a sparse graph and is usually tractable for 20k
    # 'rbf' might build a dense graph or become very slow, but we include it for completeness
    affinity = trial.suggest_categorical("affinity", ["nearest_neighbors", "rbf"])
    
    sc_kwargs = {
        "n_clusters": N_CLUSTERS,
        "affinity": affinity,
        "random_state": SEED,
        "n_init": 10,
        "n_jobs": -1 # Use all available cores for the pairwise distances / k-means
    }
    
    if affinity == "nearest_neighbors":
        n_neighbors = trial.suggest_int("n_neighbors", 5, 200)
        sc_kwargs["n_neighbors"] = n_neighbors
    elif affinity == "rbf":
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        sc_kwargs["gamma"] = gamma
        
    # 3. Embedding and Clustering Method
    assign_labels = trial.suggest_categorical("assign_labels", ["kmeans", "discretize", "cluster_qr"])
    sc_kwargs["assign_labels"] = assign_labels
    
    eigen_solver = trial.suggest_categorical("eigen_solver", ["arpack", "lobpcg"])
    sc_kwargs["eigen_solver"] = eigen_solver

    model = SpectralClustering(**sc_kwargs)
    
    try:
        t0 = time.time()
        labels = model.fit_predict(X_processed)
        elapsed = time.time() - t0
        
        ari = adjusted_rand_score(y_true, labels)
        
        # Report time as a user attribute to analyze later
        trial.set_user_attr("elapsed_time_sec", elapsed)
        
        # We maximize ARI
        return ari
        
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        # Prune or return a very poor score if it crashes (e.g. MemoryError, non-convergence)
        raise optuna.exceptions.TrialPruned()

def main():
    print(f"\nStarting Optuna study on {N_SAMPLES} samples of MNIST for sklearn SpectralClustering...")
    print(f"We will run {N_TRIALS} trials optimizing for Adjusted Rand Index (ARI).\n")
    
    # Create the study. We can use a sqlite database to persist the study in case of crashes
    study = optuna.create_study(
        direction="maximize", 
        study_name="sklearn_sc_mnist20k",
        storage="sqlite:///optuna_sklearn_sc.db",
        load_if_exists=True
    )
    
    # We run trials sequentially (n_jobs=1) because SpectralClustering uses parallel threads internally
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    
    print("\n" + "="*50)
    print("Optimization Complete!")
    print("="*50)
    
    if len(study.best_trials) == 0:
        print("No successful trials.")
        return
        
    trial = study.best_trial
    print(f"Best Trial #{trial.number}")
    print(f"  Value (ARI): {trial.value:.4f}")
    if "elapsed_time_sec" in trial.user_attrs:
        print(f"  Time taken:  {trial.user_attrs['elapsed_time_sec']:.2f} s")
        
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # We can also plot optimization history if wanted
    print("\n(To view detailed visualizations, you can run: `optuna-dashboard sqlite:///optuna_sklearn_sc.db`)")

if __name__ == "__main__":
    main()
