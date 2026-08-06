import optuna
import time
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# 1. Generate the same benchmark dataset
N_SAMPLES = 4000
X, y_true = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=42)
X = np.ascontiguousarray(X, dtype=np.float32)

def objective(trial):
    """
    Optuna objective function to find the best hyperparameters 
    for sklearn's Spectral Clustering.
    """
    # Suggest categorical parameter for the affinity type
    affinity = trial.suggest_categorical("affinity", ["nearest_neighbors"])
    
    # Conditional hyperparameter tuning based on the chosen affinity
    if affinity == "rbf":
        # Search gamma on a log scale from 0.1 to 100.0
        gamma = trial.suggest_float("gamma", 0.1, 100.0, log=True)
        n_neighbors = 10 # Ignored by rbf, but required for the class init
    else:
        # Search n_neighbors from 5 to 200
        n_neighbors = trial.suggest_int("n_neighbors", 5, 200)
        gamma = 1.0 # Ignored by nearest_neighbors
        
    # Suggest labeling strategy (cluster_qr is often faster/more stable than kmeans)
    assign_labels = trial.suggest_categorical("assign_labels", ["kmeans", "discretize", "cluster_qr"])
    
    # Initialize and run the model
    model = SpectralClustering(
        n_clusters=2,
        affinity=affinity,
        gamma=gamma,
        n_neighbors=n_neighbors,
        assign_labels=assign_labels,
        random_state=42,
        n_jobs=-1 # Use all CPU cores for the nearest neighbors graph
    )
    
    # Predict and score
    labels = model.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    
    return ari

def run_optuna_tuning():
    print(f"Starting Optuna hyperparameter search for Spectral Clustering (N={N_SAMPLES})...")
    
    # Create a study object and specify the direction is 'maximize' (we want the highest ARI)
    study = optuna.create_study(direction="maximize", study_name="Spectral_Clustering_Tuning")
    
    # Run the optimization for 50 trials
    t0 = time.time()
    study.optimize(objective, n_trials=50)
    total_time = time.time() - t0
    
    print("\n" + "="*50)
    print("Optimization Finished!")
    print(f"Total Tuning Time: {total_time:.2f} seconds")
    print("="*50)
    
    # Fetch the best results
    best_trial = study.best_trial
    print(f"Best ARI Score: {best_trial.value:.4f}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    # Note: You may need to run `pip install optuna` if you haven't already
    run_optuna_tuning()