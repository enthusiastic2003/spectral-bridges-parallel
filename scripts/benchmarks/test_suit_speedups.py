"""
Scaling benchmarks for the CUDA Spectral Bridges port.
========================================================

This script runs the two scaling experiments that constitute the actual
contribution of this work: how does our CUDA implementation's runtime scale
with respect to (a) dataset size n, and (b) the number of Voronoi regions m,
relative to the author's reference Python implementation.

Why these two experiments and not others?
-----------------------------------------
The Spectral Bridges paper claims overall time complexity O(n*m*d) for the
affinity step and O(m^3) for the eigendecomposition. So the two interesting
axes for "does the GPU port pay off" are exactly n (affinity dominates) and
m (eigendecomp dominates). Sweeping these in isolation -- one fixed while
the other varies -- lets us read off the empirical complexity of each
implementation and identify where the GPU wins.

We also keep accuracy (ARI/NMI) for each run as a side-channel sanity check.

Protocol per (experiment, point):
  - 5 runs with paired seeds (both methods see the same data and the same seed)
  - Each run regenerates data fresh from the chosen seed
  - Failures are logged but do not stop the sweep -- partial CSVs are useful

MNIST loading:
  This script loads MNIST from a local arff.gz file -- no openml, no network.
  Set MNIST_ARFF_PATH below to wherever your file lives. On first load the
  arff is converted to a fast .npz cache; subsequent loads use the cache.

Outputs:
  - results_n_scaling.csv   (one row per (point, method, run))
  - results_m_scaling.csv
  - figure_n_scaling.png
  - figure_m_scaling.png

Re-run plotting only (no recomputation):
  python scaling_benchmarks.py --plot-only

Run a single experiment:
  python scaling_benchmarks.py --experiment n
  python scaling_benchmarks.py --experiment m

Print configuration without running anything (dry run):
  python scaling_benchmarks.py --dry-run
"""

import argparse
import csv
import gzip
import os
import platform
import resource          # for peak RSS reporting (Linux/macOS)
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sbcluster import SpectralBridges
sys.path.append(os.path.abspath("build/"))
import specbridge


# =========================================================================
# Configuration
# =========================================================================

# Path to the MNIST arff file (the version-1 form from openml). The script
# will create a .npz cache next to it on first load for fast subsequent runs.
MNIST_ARFF_PATH = "mnist_784.arff.gz"
MNIST_NPZ_CACHE = "mnist_cache.npz"

# n-scaling: log-spaced from 1k to 1M.
N_VALUES = [
    1_000, 2_500, 5_000, 10_000,
    25_000, 50_000, 100_000,
     500_000, 1_000_000,
     5_000_000, 10_000_000
]
N_SCALING_M = 250
N_SCALING_D = 64
N_SCALING_K = 10

# m-scaling: 50 to 2000 on real MNIST data.
M_VALUES = [50, 100, 250, 500, 1000, 1500, 2000, 4000, 8000]
# M_VALUES = [12000]
M_SCALING_N = 20_000
M_SCALING_H = 64
M_SCALING_K = 10

N_RUNS = 5
BASE_SEED = 22188
PERPLEXITY = 2.0
N_ITER = 20
NUM_THREADS = 36

CSV_N = "results_n_scaling.csv"
CSV_M = "results_m_scaling.csv"
PLOT_N = "figure_n_scaling.png"
PLOT_M = "figure_m_scaling.png"


# =========================================================================
# Logging utilities
# =========================================================================

_T0 = time.time()


def log(msg, level="info"):
    elapsed = time.time() - _T0
    stamp = datetime.now().strftime("%H:%M:%S")
    prefix = {
        "info":  "    ",
        "step":  ">>> ",
        "warn":  "!!! ",
        "error": "XXX ",
        "ok":    "    ",
    }.get(level, "    ")
    print(f"[{stamp} | +{elapsed:7.1f}s] {prefix}{msg}", flush=True)


def banner(title):
    print("\n" + "=" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 78, flush=True)


def section(title):
    print("\n" + "-" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("-" * 78, flush=True)


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def fmt_eta(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "??:??:??"
    return str(timedelta(seconds=int(seconds)))


def print_environment():
    banner("Environment fingerprint")
    log(f"Hostname           : {socket.gethostname()}")
    log(f"Platform           : {platform.platform()}")
    log(f"Python             : {sys.version.split()[0]}")
    log(f"NumPy              : {np.__version__}")
    try:
        import sklearn, scipy
        log(f"scikit-learn       : {sklearn.__version__}")
        log(f"SciPy              : {scipy.__version__}")
    except ImportError:
        pass
    log(f"OMP_NUM_THREADS    : {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    log(f"MKL_NUM_THREADS    : {os.environ.get('MKL_NUM_THREADS', '<unset>')}")
    log(f"specbridge threads : {NUM_THREADS}")
    log(f"MNIST arff path    : {MNIST_ARFF_PATH}  "
        f"(exists: {os.path.exists(MNIST_ARFF_PATH)})")
    log(f"MNIST npz cache    : {MNIST_NPZ_CACHE}  "
        f"(exists: {os.path.exists(MNIST_NPZ_CACHE)})")

    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
        for line in out.splitlines():
            log(f"GPU                : {line}")
    except Exception:
        log("GPU                : (nvidia-smi unavailable)", level="warn")

    log(f"Process peak RSS   : {peak_rss_mb():.1f} MB (so far)")


def print_config():
    banner("Configuration")
    log(f"N_VALUES       : {N_VALUES}")
    log(f"N_SCALING_M    : {N_SCALING_M}")
    log(f"N_SCALING_D    : {N_SCALING_D}")
    log(f"N_SCALING_K    : {N_SCALING_K}")
    log(f"M_VALUES       : {M_VALUES}")
    log(f"M_SCALING_N    : {M_SCALING_N}")
    log(f"M_SCALING_H    : {M_SCALING_H}")
    log(f"M_SCALING_K    : {M_SCALING_K}")
    log(f"N_RUNS         : {N_RUNS}")
    log(f"BASE_SEED      : {BASE_SEED}")
    log(f"PERPLEXITY     : {PERPLEXITY}")
    log(f"N_ITER         : {N_ITER}")
    log(f"NUM_THREADS    : {NUM_THREADS}")


# =========================================================================
# Data loading
# =========================================================================

def make_synthetic(n, d, k, seed):
    """Gaussian mixture with k well-separated centers in d dims."""
    log(f"  generating synthetic data: n={n:,}, d={d}, k={k}, seed={seed}")
    t0 = time.time()
    X, y = make_blobs(
        n_samples=n, n_features=d, centers=k,
        cluster_std=1.5, center_box=(-20.0, 20.0),
        random_state=seed,
    )
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = y.astype(int)
    log(f"  data generated in {time.time() - t0:.2f}s, "
        f"X shape={X.shape}, dtype={X.dtype}, "
        f"size={X.nbytes / 1e6:.1f} MB")
    return X, y


def load_mnist_from_arff(path):
    """Parse an MNIST arff(.gz) file produced by openml.

    Returns:
      X : float32, shape (n, 784), values in [0, 1]
      y : int,     shape (n,),     digit labels 0-9
    """
    log(f"  parsing arff (slow, ~30-60s for full MNIST): {path}")
    t0 = time.time()

    # scipy.io.arff handles both .arff and uncompressed text streams; we open
    # the gzip ourselves so the same code works for .arff and .arff.gz.
    if path.endswith(".gz"):
        f = gzip.open(path, "rt")
    else:
        f = open(path, "r")
    try:
        data, _meta = arff.loadarff(f)
    finally:
        f.close()

    # MNIST openml arff: 784 pixel columns + 1 class column at the end.
    field_names = data.dtype.names
    pixel_fields = field_names[:-1]
    label_field = field_names[-1]
    if len(pixel_fields) != 784:
        log(f"  WARNING: expected 784 pixel columns, got {len(pixel_fields)}",
            level="warn")

    # Stack pixel columns into (n, 784) and rescale to [0, 1].
    X = np.stack([data[f] for f in pixel_fields], axis=1).astype(np.float32) / 255.0

    # Labels are usually byte strings (b'0', b'1', ...) in arff; sometimes
    # numeric. Handle both.
    y_raw = data[label_field]
    if y_raw.dtype.kind in ("S", "O"):
        y = np.array([int(v) for v in y_raw], dtype=int)
    else:
        y = y_raw.astype(int)

    log(f"  arff parsed in {time.time() - t0:.1f}s, "
        f"X shape={X.shape}, label range=[{y.min()}, {y.max()}]")
    return X, y


def save_npz_cache(X, y, path):
    log(f"  caching to {path} for faster subsequent loads")
    t0 = time.time()
    np.savez_compressed(path, X=X, y=y)
    log(f"  cache saved in {time.time() - t0:.1f}s, "
        f"file size={os.path.getsize(path) / 1e6:.1f} MB")


def load_npz_cache(path):
    log(f"  loading from cache: {path}")
    t0 = time.time()
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    log(f"  cache loaded in {time.time() - t0:.2f}s, X shape={X.shape}")
    return X, y


_MNIST_CACHE = {"X": None, "y": None}


def get_mnist():
    """Return (X, y) for full MNIST. Tries .npz cache first, falls back to
    parsing the arff. The first arff parse writes the .npz cache so future
    runs are fast."""
    if _MNIST_CACHE["X"] is not None:
        return _MNIST_CACHE["X"], _MNIST_CACHE["y"]

    # Prefer the .npz cache if present.
    if os.path.exists(MNIST_NPZ_CACHE):
        X, y = load_npz_cache(MNIST_NPZ_CACHE)
    elif os.path.exists(MNIST_ARFF_PATH):
        X, y = load_mnist_from_arff(MNIST_ARFF_PATH)
        try:
            save_npz_cache(X, y, MNIST_NPZ_CACHE)
        except Exception as e:
            log(f"  could not write cache (non-fatal): {e}", level="warn")
    else:
        raise FileNotFoundError(
            f"Neither {MNIST_NPZ_CACHE} nor {MNIST_ARFF_PATH} exists. "
            f"Provide MNIST as either a .npz with 'X' and 'y' arrays, or "
            f"the original openml arff(.gz) file."
        )

    _MNIST_CACHE["X"] = X
    _MNIST_CACHE["y"] = y
    return X, y


def make_mnist_pca(n, h, seed):
    """Sample n points from MNIST and PCA-reduce to h dimensions."""
    X_full, y_full = get_mnist()
    rng = np.random.RandomState(seed)
    if n <= len(X_full):
        idx = rng.choice(len(X_full), n, replace=False)
    else:
        log(f"  WARNING: n={n} > MNIST size ({len(X_full)}), oversampling",
            level="warn")
        idx = rng.choice(len(X_full), n, replace=True)
    X_sub = X_full[idx]
    y = y_full[idx]
    pca = PCA(n_components=h, random_state=seed)
    X_pca = np.ascontiguousarray(pca.fit_transform(X_sub), dtype=np.float32)
    return X_pca, y


# =========================================================================
# Method runners
# =========================================================================

def run_author(X, y_true, k, m, seed):
    log(f"    [author] starting fit (n={len(X):,}, d={X.shape[1]}, m={m})")
    model = SpectralBridges(
        n_clusters=k, n_nodes=m,
        perplexity=PERPLEXITY, n_iter=N_ITER, random_state=seed,
    )
    t0 = time.time()
    model.fit(X)
    elapsed = time.time() - t0
    ari = adjusted_rand_score(y_true, model.labels_)
    nmi = normalized_mutual_info_score(y_true, model.labels_)
    log(f"    [author] done in {elapsed:.2f}s, ARI={ari:.3f}, NMI={nmi:.3f}",
        level="ok")
    return {"total_s": elapsed, "ari": ari, "nmi": nmi}


def run_cuda(X, y_true, k, m, seed):
    log(f"    [cuda]   starting fit (n={len(X):,}, d={X.shape[1]}, m={m})")
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

    extracted = {}
    for attr in ("t_kmeans", "t_affinity", "t_spectral", "t_propagate",
                 "t_laplacian", "t_eigen", "t_spectral_kmeans"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            out[attr] = val
            extracted[attr] = val

    log(f"    [cuda]   done in {elapsed:.2f}s, ARI={ari:.3f}, NMI={nmi:.3f}",
        level="ok")
    if extracted:
        log(f"    [cuda]   stages: " +
            ", ".join(f"{k.replace('t_', '')}={v:.3f}s"
                      for k, v in extracted.items()))
    return out


# =========================================================================
# CSV writer
# =========================================================================

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
    log(f"opening CSV for write: {path}")
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()
    f.flush()
    try:
        yield w, f
    finally:
        f.close()
        log(f"closed CSV: {path}")


def emit_row(writer, file_handle, base, result, status="ok", error=""):
    row = {field: "" for field in CSV_FIELDS}
    row.update(base)
    row.update(result or {})
    row["status"] = status
    row["error"] = error
    writer.writerow(row)
    file_handle.flush()


# =========================================================================
# Sweep drivers
# =========================================================================

def estimate_remaining(history, total_remaining_units):
    if not history:
        return None
    avg = np.mean(history[-5:])
    return avg * total_remaining_units


def sweep_n(writer, file_handle):
    banner(f"Experiment 1: n-scaling  (synthetic, fixed m={N_SCALING_M})")
    log(f"sweeping n over {N_VALUES}")
    log(f"each n: {N_RUNS} paired runs (author + cuda)")

    sweep_start = time.time()
    point_times = []

    for point_idx, n in enumerate(N_VALUES):
        section(f"n = {n:,}  "
                f"({point_idx + 1}/{len(N_VALUES)} | "
                f"ETA {fmt_eta(estimate_remaining(point_times, len(N_VALUES) - point_idx))})")

        point_t0 = time.time()
        ok_count = err_count = 0

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            log(f"run {run_idx + 1}/{N_RUNS}  (seed={seed})", level="step")

            try:
                X, y = make_synthetic(n, N_SCALING_D, N_SCALING_K, seed)
            except MemoryError as e:
                log(f"data gen OOM at n={n:,}, skipping rest of point",
                    level="error")
                emit_row(
                    writer, file_handle,
                    {"experiment": "n", "point": n, "run": run_idx,
                     "seed": seed, "method": "data_gen",
                     "n": n, "m": N_SCALING_M, "d_or_h": N_SCALING_D,
                     "k": N_SCALING_K},
                    None, status="oom", error=str(e),
                )
                break

            for method, runner in (("author", run_author),
                                   ("cuda",   run_cuda)):
                base = {"experiment": "n", "point": n, "run": run_idx,
                        "seed": seed, "method": method,
                        "n": n, "m": N_SCALING_M, "d_or_h": N_SCALING_D,
                        "k": N_SCALING_K}
                try:
                    res = runner(X, y, N_SCALING_K, N_SCALING_M, seed)
                    emit_row(writer, file_handle, base, res, status="ok")
                    ok_count += 1
                except Exception as e:
                    log(f"{method} FAILED: {type(e).__name__}: {e}",
                        level="error")
                    emit_row(writer, file_handle, base, None,
                             status="error",
                             error=f"{type(e).__name__}: {e}")
                    traceback.print_exc(limit=3)
                    err_count += 1

            del X, y

        elapsed = time.time() - point_t0
        point_times.append(elapsed)
        log(f"finished n={n:,} in {elapsed:.1f}s  "
            f"({ok_count} ok, {err_count} errors)  "
            f"peak RSS={peak_rss_mb():.0f} MB", level="ok")

    log(f"n-sweep total wall time: {time.time() - sweep_start:.1f}s")


def sweep_m(writer, file_handle):
    banner(f"Experiment 2: m-scaling  (MNIST h={M_SCALING_H}, n={M_SCALING_N:,})")
    log(f"sweeping m over {M_VALUES}")

    # Prime the MNIST cache once so subsequent runs don't re-parse.
    log("priming MNIST cache before sweep starts...")
    get_mnist()

    sweep_start = time.time()
    point_times = []

    for point_idx, m in enumerate(M_VALUES):
        section(f"m = {m}  "
                f"({point_idx + 1}/{len(M_VALUES)} | "
                f"ETA {fmt_eta(estimate_remaining(point_times, len(M_VALUES) - point_idx))})")

        point_t0 = time.time()
        ok_count = err_count = 0

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            log(f"run {run_idx + 1}/{N_RUNS}  (seed={seed})", level="step")

            X, y = make_mnist_pca(M_SCALING_N, M_SCALING_H, seed)

            for method, runner in (("author", run_author),
                                   ("cuda",   run_cuda)):
                base = {"experiment": "m", "point": m, "run": run_idx,
                        "seed": seed, "method": method,
                        "n": M_SCALING_N, "m": m, "d_or_h": M_SCALING_H,
                        "k": M_SCALING_K}
                try:
                    res = runner(X, y, M_SCALING_K, m, seed)
                    emit_row(writer, file_handle, base, res, status="ok")
                    ok_count += 1
                except Exception as e:
                    log(f"{method} FAILED: {type(e).__name__}: {e}",
                        level="error")
                    emit_row(writer, file_handle, base, None,
                             status="error",
                             error=f"{type(e).__name__}: {e}")
                    traceback.print_exc(limit=3)
                    err_count += 1

        elapsed = time.time() - point_t0
        point_times.append(elapsed)
        log(f"finished m={m} in {elapsed:.1f}s  "
            f"({ok_count} ok, {err_count} errors)  "
            f"peak RSS={peak_rss_mb():.0f} MB", level="ok")

    log(f"m-sweep total wall time: {time.time() - sweep_start:.1f}s")


# =========================================================================
# Plotting (works directly from CSV, so you can replot without rerunning)
# =========================================================================

def load_csv(path):
    if not os.path.exists(path):
        log(f"no CSV at {path}, skipping load", level="warn")
        return None
    log(f"loading CSV: {path}")
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
    log(f"  loaded {len(rows)} successful rows from {path}")
    return rows


def aggregate(rows, point_key="point"):
    agg = {}
    for r in rows:
        key = (r[point_key], r["method"])
        agg.setdefault(key, []).append(r["total_s"])
    summary = {}
    for key, times in agg.items():
        arr = np.array(times)
        summary[key] = (
            arr.mean(),
            arr.std(ddof=1) if len(arr) > 1 else 0.0,
            len(arr),
        )
    return summary


def print_summary_table(rows, point_key, point_label):
    if not rows:
        return
    summary = aggregate(rows, point_key)
    points = sorted({k[0] for k in summary.keys()})

    print()
    log(f"Timing summary across {point_label}:")
    print(f"  {'point':>10}  {'author (s)':>18}  {'cuda (s)':>18}  "
          f"{'speedup':>10}")
    print(f"  {'-' * 10}  {'-' * 18}  {'-' * 18}  {'-' * 10}")
    for p in points:
        a = summary.get((p, "author"))
        c = summary.get((p, "cuda"))
        a_str = f"{a[0]:7.3f} +/- {a[1]:6.3f}" if a else "         n/a"
        c_str = f"{c[0]:7.3f} +/- {c[1]:6.3f}" if c else "         n/a"
        if a and c and c[0] > 0:
            sp_str = f"{a[0] / c[0]:7.2f}x"
        else:
            sp_str = "      n/a"
        print(f"  {p:>10}  {a_str:>18}  {c_str:>18}  {sp_str:>10}")
    print()


def plot_scaling(rows, point_key, point_label, title, out_path,
                 log_x=True, log_y=True):
    if not rows:
        log(f"no rows for {title}, skipping plot", level="warn")
        return

    summary = aggregate(rows, point_key)
    points = sorted({k[0] for k in summary.keys()})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

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
        rel_var = ((a[1] / a[0]) ** 2 + (c[1] / c[0]) ** 2
                   if a[0] and c[0] else 0)
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
    log(f"saved plot: {out_path}", level="ok")


def make_plots():
    banner("Plotting")
    rows_n = load_csv(CSV_N)
    if rows_n:
        print_summary_table(rows_n, "n", "n")
        plot_scaling(rows_n, "n",
                     "n (number of points, log scale)",
                     f"n-scaling (synthetic, m={N_SCALING_M})", PLOT_N, log_x=True, log_y=True)
    rows_m = load_csv(CSV_M)
    if rows_m:
        print_summary_table(rows_m, "m", "m")
        plot_scaling(rows_m, "m",
                     "m (Voronoi regions, log scale)",
                     f"m-scaling (MNIST, n={M_SCALING_N:,})", PLOT_M, log_x=True, log_y=True)


# =========================================================================
# Entry point
# =========================================================================

def main():
    global MNIST_ARFF_PATH, MNIST_NPZ_CACHE

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", choices=["n", "m", "both"],
                        default="both",
                        help="Which scaling sweep to run (default: both)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running, just regenerate plots from CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration without running anything")
    parser.add_argument("--mnist-arff", default=None,
                        help=f"Override MNIST arff path (default: {MNIST_ARFF_PATH})")
    parser.add_argument("--mnist-cache", default=None,
                        help=f"Override MNIST npz cache path (default: {MNIST_NPZ_CACHE})")
    args = parser.parse_args()

    # Allow CLI override of MNIST paths without editing the script.
    if args.mnist_arff:
        MNIST_ARFF_PATH = args.mnist_arff
    if args.mnist_cache:
        MNIST_NPZ_CACHE = args.mnist_cache

    print_environment()
    print_config()

    if args.dry_run:
        banner("Dry run requested -- exiting without running anything")
        return

    if args.plot_only:
        make_plots()
        return

    log(f"setting specbridge num_threads to {NUM_THREADS}")
    specbridge.set_num_threads(NUM_THREADS)

    overall_start = time.time()

    if args.experiment in ("n", "both"):
        with csv_writer(CSV_N) as (w, f):
            sweep_n(w, f)

    if args.experiment in ("m", "both"):
        with csv_writer(CSV_M) as (w, f):
            sweep_m(w, f)

    log(f"all sweeps complete in {time.time() - overall_start:.1f}s",
        level="ok")
    make_plots()
    log("done.", level="ok")


if __name__ == "__main__":
    main()