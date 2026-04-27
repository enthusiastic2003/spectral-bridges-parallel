"""
Scaling benchmarks for the CUDA Spectral Bridges port.

Two experiments, both with paired-seed protocol and 5 runs per point:

  1. n-scaling (fixed m=250):
       Generates synthetic Gaussian mixture data. Sweeps n from 1k to 1M
       in log-spaced steps. Author's Python and our CUDA both run; CUDA path
       additionally exposes per-stage timings.

  2. m-scaling (fixed n=20k MNIST PCA h=64):
       Sweeps m from 50 to 2000.

For each (experiment, point, method, run) we record:
  - total elapsed time
  - ARI / NMI (sanity check; we are NOT averaging accuracy here, just confirming
    no run silently failed)
  - per-stage breakdown for the CUDA path when available

Outputs:
  - results_n_scaling.csv
  - results_m_scaling.csv
  - figure_n_scaling.png
  - figure_m_scaling.png

Re-run plotting only:
  python scaling_benchmarks.py --plot-only

Run a single experiment:
  python scaling_benchmarks.py --experiment n
  python scaling_benchmarks.py --experiment m
"""

import argparse
import csv
import os
import sys
import time
import traceback
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sbcluster import SpectralBridges
sys.path.append(os.path.abspath("build/"))
import specbridge


# ---- Configuration ------------------------------------------------------

# n-scaling: log-spaced from 1k to 1M
N_VALUES = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000,
            250_000, 500_000, 1_000_000]
N_SCALING_M = 250
N_SCALING_D = 64        # synthetic data dimensionality
N_SCALING_K = 10        # number of true clusters in synthetic data

# m-scaling: fixed n, sweep m
M_VALUES = [50, 100, 250, 500, 1000, 1500, 2000]
M_SCALING_N = 20_000
M_SCALING_H = 64
M_SCALING_K = 10

N_RUNS = 5
BASE_SEED = 22188
PERPLEXITY = 2.0
N_ITER = 20
NUM_THREADS = 12

# Bail out on a single run after this long (seconds). Useful so a slow Python
# run at n=1M doesn't hold up the rest of the sweep.
PER_RUN_TIMEOUT_S = 60 * 30   # 30 min

CSV_N = "results_n_scaling.csv"
CSV_M = "results_m_scaling.csv"
PLOT_N = "figure_n_scaling.png"
PLOT_M = "figure_m_scaling.png"


# ---- Data generation ----------------------------------------------------

def make_synthetic(n, d, k, seed):
    """Gaussian mixture with k well-separated centers in d dims."""
    X, y = make_blobs(
        n_samples=n, n_features=d, centers=k,
        cluster_std=1.5, center_box=(-20.0, 20.0),
        random_state=seed,
    )
    return np.ascontiguousarray(X, dtype=np.float32), y.astype(int)


_MNIST_CACHE = {"X": None, "y": None}

def get_mnist():
    if _MNIST_CACHE["X"] is None:
        print("Loading MNIST (one-time)...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="auto")
        _MNIST_CACHE["X"] = X.astype(np.float32) / 255.0
        _MNIST_CACHE["y"] = y.astype(int)
    return _MNIST_CACHE["X"], _MNIST_CACHE["y"]


def make_mnist_pca(n, h, seed):
    X_full, y_full = get_mnist()
    rng = np.random.RandomState(seed)
    if n <= len(X_full):
        idx = rng.choice(len(X_full), n, replace=False)
    else:
        idx = rng.choice(len(X_full), n, replace=True)  # not used in current sweeps
    X_sub = X_full[idx]
    y = y_full[idx]
    pca = PCA(n_components=h, random_state=seed)
    X_pca = np.ascontiguousarray(pca.fit_transform(X_sub), dtype=np.float32)
    return X_pca, y


# ---- Method runners -----------------------------------------------------

def run_author(X, y_true, k, m, seed):
    model = SpectralBridges(
        n_clusters=k, n_nodes=m,
        perplexity=PERPLEXITY, n_iter=N_ITER, random_state=seed,
    )
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    ari = adjusted_rand_score(y_true, model.labels_)
    nmi = normalized_mutual_info_score(y_true, model.labels_)
    return {"total_s": elapsed, "ari": ari, "nmi": nmi}


def run_cuda(X, y_true, k, m, seed):
    """Runs the CUDA pipeline. Per-stage timings come from the C++ profiler
    if exposed via the result object; otherwise only total is recorded."""
    model = specbridge.SpectralClustering(
        n_clusters=k, num_voronoi=m,
        n_iter=N_ITER, target_perplexity=PERPLEXITY,
        random_state=seed, use_gpu=True,
    )
    t0 = time.time()
    result = model.fit(X)
    elapsed = time.time() - t0
    labels = np.array(result.labels)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    out = {"total_s": elapsed, "ari": ari, "nmi": nmi}

    # If your SBResult exposes per-stage timings, capture them.
    # (Adjust attribute names if your bindings use different ones.)
    for attr in ("t_kmeans", "t_affinity", "t_spectral", "t_propagate",
                 "t_laplacian", "t_eigen", "t_spectral_kmeans"):
        if hasattr(result, attr):
            out[attr] = getattr(result, attr)
    return out


# ---- Sweep drivers ------------------------------------------------------

CSV_FIELDS = [
    "experiment", "point", "run", "seed", "method",
    "n", "m", "d_or_h", "k",
    "total_s", "ari", "nmi",
    "t_kmeans", "t_affinity", "t_spectral", "t_propagate",
    "t_laplacian", "t_eigen", "t_spectral_kmeans",
    "status", "error",
]


@contextmanager
def csv_writer(path):
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()
    f.flush()
    try:
        yield w, f
    finally:
        f.close()


def emit_row(writer, file_handle, base, result, status="ok", error=""):
    row = {field: "" for field in CSV_FIELDS}
    row.update(base)
    row.update(result or {})
    row["status"] = status
    row["error"] = error
    writer.writerow(row)
    file_handle.flush()


def sweep_n(writer, file_handle):
    print("\n" + "=" * 60)
    print("Experiment 1: n-scaling (synthetic, fixed m=%d)" % N_SCALING_M)
    print("=" * 60)

    for n in N_VALUES:
        print(f"\n--- n = {n:,} ---")
        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            try:
                X, y = make_synthetic(n, N_SCALING_D, N_SCALING_K, seed)
            except MemoryError as e:
                print(f"  run {run_idx + 1}: data gen OOM, skipping rest of n={n}")
                emit_row(writer, file_handle,
                         {"experiment": "n", "point": n, "run": run_idx,
                          "seed": seed, "method": "data_gen",
                          "n": n, "m": N_SCALING_M, "d_or_h": N_SCALING_D,
                          "k": N_SCALING_K},
                         None, status="oom", error=str(e))
                break

            for method, runner in (("author", run_author), ("cuda", run_cuda)):
                base = {"experiment": "n", "point": n, "run": run_idx,
                        "seed": seed, "method": method,
                        "n": n, "m": N_SCALING_M, "d_or_h": N_SCALING_D,
                        "k": N_SCALING_K}
                try:
                    res = runner(X, y, N_SCALING_K, N_SCALING_M, seed)
                    emit_row(writer, file_handle, base, res, status="ok")
                    print(f"  run {run_idx + 1}/{N_RUNS} {method:>6}: "
                          f"{res['total_s']:7.2f}s  "
                          f"ARI={res['ari']:.3f}  NMI={res['nmi']:.3f}")
                except Exception as e:
                    print(f"  run {run_idx + 1}/{N_RUNS} {method:>6}: FAILED ({type(e).__name__})")
                    emit_row(writer, file_handle, base, None,
                             status="error", error=f"{type(e).__name__}: {e}")
                    traceback.print_exc(limit=2)


def sweep_m(writer, file_handle):
    print("\n" + "=" * 60)
    print(f"Experiment 2: m-scaling (MNIST h={M_SCALING_H}, n={M_SCALING_N:,})")
    print("=" * 60)

    for m in M_VALUES:
        print(f"\n--- m = {m} ---")
        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            X, y = make_mnist_pca(M_SCALING_N, M_SCALING_H, seed)

            for method, runner in (("author", run_author), ("cuda", run_cuda)):
                base = {"experiment": "m", "point": m, "run": run_idx,
                        "seed": seed, "method": method,
                        "n": M_SCALING_N, "m": m, "d_or_h": M_SCALING_H,
                        "k": M_SCALING_K}
                try:
                    res = runner(X, y, M_SCALING_K, m, seed)
                    emit_row(writer, file_handle, base, res, status="ok")
                    print(f"  run {run_idx + 1}/{N_RUNS} {method:>6}: "
                          f"{res['total_s']:7.2f}s  "
                          f"ARI={res['ari']:.3f}  NMI={res['nmi']:.3f}")
                except Exception as e:
                    print(f"  run {run_idx + 1}/{N_RUNS} {method:>6}: FAILED ({type(e).__name__})")
                    emit_row(writer, file_handle, base, None,
                             status="error", error=f"{type(e).__name__}: {e}")
                    traceback.print_exc(limit=2)


# ---- Plotting (works directly from CSV) ---------------------------------

def load_csv(path):
    """Returns dict[(point, method)] -> list of dicts (one per successful run)."""
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            for k in ("point", "n", "m", "d_or_h", "k", "run"):
                if row.get(k):
                    row[k] = int(float(row[k]))
            for k in ("total_s", "ari", "nmi", "t_kmeans", "t_affinity",
                      "t_spectral", "t_propagate", "t_laplacian", "t_eigen",
                      "t_spectral_kmeans"):
                if row.get(k):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def aggregate(rows, point_key="point"):
    """Group by (point, method) and compute mean/std of total_s."""
    agg = {}
    for r in rows:
        key = (r[point_key], r["method"])
        agg.setdefault(key, []).append(r["total_s"])
    summary = {}
    for key, times in agg.items():
        arr = np.array(times)
        summary[key] = (arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0,
                        len(arr))
    return summary


def plot_scaling(rows, point_key, point_label, title, out_path,
                 log_x=True, log_y=True):
    if not rows:
        print(f"No rows for {title}, skipping plot.")
        return
    summary = aggregate(rows, point_key)
    points = sorted({k[0] for k in summary.keys()})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: total time vs point, per method
    ax = axes[0]
    for method, color, marker in (("author", "#9b8bd0", "o"),
                                  ("cuda",   "#5a3a8a", "s")):
        means = [summary.get((p, method), (np.nan,) * 3)[0] for p in points]
        stds  = [summary.get((p, method), (np.nan,) * 3)[1] for p in points]
        ax.errorbar(points, means, yerr=stds, label=method,
                    color=color, marker=marker, capsize=3, lw=2)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel(point_label)
    ax.set_ylabel("Total time per fit (seconds)")
    ax.set_title(f"{title}: total time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Right: speedup vs point
    ax = axes[1]
    speedups, speedups_std = [], []
    for p in points:
        a = summary.get((p, "author"))
        c = summary.get((p, "cuda"))
        if a is None or c is None or c[0] == 0:
            speedups.append(np.nan)
            speedups_std.append(0)
            continue
        s = a[0] / c[0]
        # crude std propagation
        rel_var = (a[1] / a[0]) ** 2 + (c[1] / c[0]) ** 2 if a[0] and c[0] else 0
        speedups.append(s)
        speedups_std.append(s * np.sqrt(rel_var))
    ax.errorbar(points, speedups, yerr=speedups_std,
                color="#5a3a8a", marker="D", capsize=3, lw=2)
    ax.axhline(1.0, color="grey", ls="--", alpha=0.6,
               label="parity (CUDA = Python)")
    if log_x: ax.set_xscale("log")
    ax.set_xlabel(point_label)
    ax.set_ylabel("Speedup (Python time / CUDA time)")
    ax.set_title(f"{title}: speedup")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def make_plots():
    rows_n = load_csv(CSV_N)
    if rows_n:
        plot_scaling(rows_n, "n", "n (number of points, log scale)",
                     "n-scaling (synthetic, m=%d)" % N_SCALING_M, PLOT_N)
    rows_m = load_csv(CSV_M)
    if rows_m:
        plot_scaling(rows_m, "m", "m (Voronoï regions, log scale)",
                     f"m-scaling (MNIST, n={M_SCALING_N:,})", PLOT_M)


# ---- Entry point --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["n", "m", "both"],
                        default="both")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running, just regenerate plots from CSVs")
    args = parser.parse_args()

    if args.plot_only:
        make_plots()
        return

    specbridge.set_num_threads(NUM_THREADS)

    if args.experiment in ("n", "both"):
        with csv_writer(CSV_N) as (w, f):
            sweep_n(w, f)

    if args.experiment in ("m", "both"):
        with csv_writer(CSV_M) as (w, f):
            sweep_m(w, f)

    make_plots()


if __name__ == "__main__":
    main()