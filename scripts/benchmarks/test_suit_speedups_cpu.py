"""
CPU-only thread-scaling companion to test_suit_speedups.py.

Sweeps your CUDA package's CPU path (specbridge.SpectralClustering with
use_gpu=False) across multiple thread counts at the same (n, m) points used
by the main n- and m-scaling sweeps.

CSV format is identical to test_suit_speedups.py output, so the files concat
cleanly. The thread count is encoded into the `method` column as
"cpu_{n_threads}" (e.g. cpu_4, cpu_16). When you combine CSVs later, you'll
have method values like {author, cuda, cpu_4, cpu_8, cpu_16, cpu_36}.

This is a CPU-only experiment by design -- author Python and CUDA GPU paths
are NOT run here.

Usage:
  # Run both sweeps with default thread counts [4, 8, 16, 36]
  python cpu_thread_sweep.py

  # Run only the n sweep
  python cpu_thread_sweep.py --experiment n

  # Custom thread list
  python cpu_thread_sweep.py --threads 4 8 16

  # Re-plot from existing CSVs without rerunning
  python cpu_thread_sweep.py --plot-only

Outputs:
  - results_n_scaling_cpu.csv
  - results_m_scaling_cpu.csv
  - figure_n_scaling_cpu.png
  - figure_m_scaling_cpu.png
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

# Reuse everything that already works in the main script.
from test_suit_speedups import (
    # configuration constants
    N_VALUES, N_SCALING_M, N_SCALING_D, N_SCALING_K,
    M_VALUES, M_SCALING_N, M_SCALING_H, M_SCALING_K,
    N_RUNS, BASE_SEED, PERPLEXITY, N_ITER,
    # data loaders
    make_synthetic, make_mnist_pca, get_mnist,
    # logging helpers
    log, banner, section, peak_rss_mb, fmt_eta, estimate_remaining,
    print_environment, print_config,
    # CSV machinery (we use the same fields, no schema change)
    CSV_FIELDS, csv_writer, emit_row,
    # plotting helpers we'll reuse / extend
    load_csv,
)
import test_suit_speedups as base   # for mutating MNIST paths if needed

sys.path.append(os.path.abspath("build/"))
import specbridge

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# =========================================================================
# Configuration
# =========================================================================

DEFAULT_THREAD_COUNTS = [4, 8, 16, 36]

CSV_N_CPU = "results_n_scaling_cpu.csv"
CSV_M_CPU = "results_m_scaling_cpu.csv"
PLOT_N_CPU = "figure_n_scaling_cpu.png"
PLOT_M_CPU = "figure_m_scaling_cpu.png"


# =========================================================================
# CPU runner
# =========================================================================

def run_cpu(X, y_true, k, m, seed, n_threads):
    """Run the CUDA package on the CPU path with `n_threads` OpenMP threads.

    Note: set_num_threads is process-global. We set it before each fit, so
    consecutive calls with the same thread count don't re-set redundantly,
    but that's a no-op cost so we don't bother optimizing it.
    """
    log(f"    [cpu_{n_threads}] starting fit "
        f"(n={len(X):,}, d={X.shape[1]}, m={m})")

    specbridge.set_num_threads(n_threads)

    model = specbridge.SpectralClustering(
        n_clusters=k, num_voronoi=m,
        n_iter=N_ITER, target_perplexity=PERPLEXITY,
        random_state=seed, use_gpu=False,
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

    log(f"    [cpu_{n_threads}] done in {elapsed:.2f}s, "
        f"ARI={ari:.3f}, NMI={nmi:.3f}", level="ok")
    if extracted:
        log(f"    [cpu_{n_threads}] stages: " +
            ", ".join(f"{k.replace('t_', '')}={v:.3f}s"
                      for k, v in extracted.items()))
    return out


# =========================================================================
# Sweep drivers
# =========================================================================

def sweep_n_cpu(writer, file_handle, thread_counts):
    banner(f"CPU n-scaling  (synthetic, fixed m={N_SCALING_M}, "
           f"threads={thread_counts})")
    log(f"sweeping n over {N_VALUES}")
    log(f"each n: {N_RUNS} runs x {len(thread_counts)} thread counts = "
        f"{N_RUNS * len(thread_counts)} fits")

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

            for n_threads in thread_counts:
                method = f"cpu_{n_threads}"
                base_row = {"experiment": "n", "point": n, "run": run_idx,
                            "seed": seed, "method": method,
                            "n": n, "m": N_SCALING_M, "d_or_h": N_SCALING_D,
                            "k": N_SCALING_K}
                try:
                    res = run_cpu(X, y, N_SCALING_K, N_SCALING_M, seed,
                                  n_threads)
                    emit_row(writer, file_handle, base_row, res, status="ok")
                    ok_count += 1
                except Exception as e:
                    log(f"{method} FAILED: {type(e).__name__}: {e}",
                        level="error")
                    emit_row(writer, file_handle, base_row, None,
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

    log(f"CPU n-sweep total wall time: {time.time() - sweep_start:.1f}s")


def sweep_m_cpu(writer, file_handle, thread_counts):
    banner(f"CPU m-scaling  (MNIST h={M_SCALING_H}, n={M_SCALING_N:,}, "
           f"threads={thread_counts})")
    log(f"sweeping m over {M_VALUES}")
    log(f"each m: {N_RUNS} runs x {len(thread_counts)} thread counts = "
        f"{N_RUNS * len(thread_counts)} fits")

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

            for n_threads in thread_counts:
                method = f"cpu_{n_threads}"
                base_row = {"experiment": "m", "point": m, "run": run_idx,
                            "seed": seed, "method": method,
                            "n": M_SCALING_N, "m": m, "d_or_h": M_SCALING_H,
                            "k": M_SCALING_K}
                try:
                    res = run_cpu(X, y, M_SCALING_K, m, seed, n_threads)
                    emit_row(writer, file_handle, base_row, res, status="ok")
                    ok_count += 1
                except Exception as e:
                    log(f"{method} FAILED: {type(e).__name__}: {e}",
                        level="error")
                    emit_row(writer, file_handle, base_row, None,
                             status="error",
                             error=f"{type(e).__name__}: {e}")
                    traceback.print_exc(limit=3)
                    err_count += 1

        elapsed = time.time() - point_t0
        point_times.append(elapsed)
        log(f"finished m={m} in {elapsed:.1f}s  "
            f"({ok_count} ok, {err_count} errors)  "
            f"peak RSS={peak_rss_mb():.0f} MB", level="ok")

    log(f"CPU m-sweep total wall time: {time.time() - sweep_start:.1f}s")


# =========================================================================
# Plotting
# =========================================================================

def aggregate_by_method(rows, point_key="point"):
    """Group by (point, method) -> (mean, std, n_runs) of total_s.
    Method strings are arbitrary -- e.g. cpu_4, cpu_16."""
    bucket = {}
    for r in rows:
        v = r.get("total_s")
        if not isinstance(v, (int, float)):
            continue
        key = (r[point_key], r["method"])
        bucket.setdefault(key, []).append(float(v))
    summary = {}
    for key, vals in bucket.items():
        arr = np.array(vals)
        summary[key] = (
            arr.mean(),
            arr.std(ddof=1) if len(arr) > 1 else 0.0,
            len(arr),
        )
    return summary


def print_cpu_table(rows, point_key, point_label):
    if not rows:
        return
    summary = aggregate_by_method(rows, point_key)
    points = sorted({k[0] for k in summary.keys()})
    methods = sorted({k[1] for k in summary.keys()},
                     key=lambda m: int(m.split("_")[1])
                     if m.startswith("cpu_") else 99999)

    print()
    log(f"CPU timing summary across {point_label}:")
    header = f"  {'point':>10}"
    for m in methods:
        header += f"  {m:>16}"
    print(header)
    print("  " + "-" * 10 + ("  " + "-" * 16) * len(methods))
    for p in points:
        line = f"  {p:>10}"
        for m in methods:
            entry = summary.get((p, m))
            if entry:
                line += f"  {entry[0]:7.3f} +/- {entry[1]:5.3f}"
            else:
                line += f"  {'n/a':>16}"
        print(line)
    print()


def plot_cpu_scaling(rows, point_key, point_label, title, out_path):
    """Two-panel plot: total time per thread count, and parallel speedup
    relative to the smallest thread count tested."""
    if not rows:
        log(f"no rows for {title}, skipping plot", level="warn")
        return

    summary = aggregate_by_method(rows, point_key)
    points = sorted({k[0] for k in summary.keys()})
    methods = sorted({k[1] for k in summary.keys()},
                     key=lambda m: int(m.split("_")[1])
                     if m.startswith("cpu_") else 99999)

    # Visual progression: lighter for fewer threads, darker for more.
    palette = ["#d4a5e8", "#9b8bd0", "#7e57c2", "#5a3a8a",
               "#4a2a7a", "#3a1a6a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: total time per fit, one line per thread count ---
    ax = axes[0]
    for i, method in enumerate(methods):
        means = [summary.get((p, method), (np.nan,) * 3)[0] for p in points]
        stds  = [summary.get((p, method), (np.nan,) * 3)[1] for p in points]
        ax.errorbar(points, means, yerr=stds, label=method,
                    color=palette[i % len(palette)], marker="o",
                    capsize=3, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{point_label} (log scale)")
    ax.set_ylabel("Total time per fit (seconds, log scale)")
    ax.set_title(f"{title}: CPU total time")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    # --- Right panel: parallel speedup vs smallest thread count ---
    # Speedup is computed relative to the smallest thread count we have data
    # for (typically cpu_4). This shows how well your CPU implementation
    # scales with more threads.
    ax = axes[1]
    if methods:
        baseline_method = methods[0]   # smallest thread count
        baseline_threads = int(baseline_method.split("_")[1])

        for i, method in enumerate(methods):
            speedups, sp_std = [], []
            for p in points:
                base_entry = summary.get((p, baseline_method))
                this_entry = summary.get((p, method))
                if (base_entry is None or this_entry is None
                        or this_entry[0] == 0):
                    speedups.append(np.nan)
                    sp_std.append(0)
                    continue
                s = base_entry[0] / this_entry[0]
                rel_var = ((base_entry[1] / base_entry[0]) ** 2 +
                           (this_entry[1] / this_entry[0]) ** 2
                           if base_entry[0] and this_entry[0] else 0)
                speedups.append(s)
                sp_std.append(s * np.sqrt(rel_var))
            ax.errorbar(points, speedups, yerr=sp_std,
                        label=method, color=palette[i % len(palette)],
                        marker="D", capsize=3, lw=2)

        # Ideal-scaling reference: doubling threads -> 2x speedup, etc.
        # Plot as horizontal lines at thread_count / baseline_threads.
        for i, method in enumerate(methods):
            t = int(method.split("_")[1])
            ideal = t / baseline_threads
            ax.axhline(ideal, color=palette[i % len(palette)],
                       ls=":", alpha=0.4, lw=1)

    ax.set_xscale("log")
    ax.set_xlabel(f"{point_label} (log scale)")
    ax.set_ylabel(f"Speedup vs {baseline_method}" if methods else "Speedup")
    ax.set_title(f"{title}: parallel speedup")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log(f"saved plot: {out_path}", level="ok")


def make_cpu_plots():
    banner("Plotting (CPU thread sweeps)")
    rows_n = load_csv(CSV_N_CPU)
    if rows_n:
        print_cpu_table(rows_n, "n", "n")
        plot_cpu_scaling(rows_n, "n",
                         "n (number of points)",
                         f"CPU n-scaling (synthetic, m={N_SCALING_M})",
                         PLOT_N_CPU)
    rows_m = load_csv(CSV_M_CPU)
    if rows_m:
        print_cpu_table(rows_m, "m", "m")
        plot_cpu_scaling(rows_m, "m",
                         "m (Voronoi regions)",
                         f"CPU m-scaling (MNIST, n={M_SCALING_N:,})",
                         PLOT_M_CPU)


# =========================================================================
# Entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", choices=["n", "m", "both"],
                        default="both",
                        help="Which sweep to run (default: both)")
    parser.add_argument("--threads", type=int, nargs="+",
                        default=DEFAULT_THREAD_COUNTS,
                        help=f"Thread counts to sweep "
                             f"(default: {DEFAULT_THREAD_COUNTS})")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running, just regenerate plots from CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configuration without running anything")
    parser.add_argument("--mnist-arff", default=None,
                        help="Override MNIST arff path")
    parser.add_argument("--mnist-cache", default=None,
                        help="Override MNIST npz cache path")
    args = parser.parse_args()

    # MNIST path overrides need to mutate the imported module's globals so
    # base.get_mnist() sees the new paths.
    if args.mnist_arff:
        base.MNIST_ARFF_PATH = args.mnist_arff
    if args.mnist_cache:
        base.MNIST_NPZ_CACHE = args.mnist_cache

    print_environment()
    print_config()
    log(f"CPU thread counts to sweep: {args.threads}")

    if args.dry_run:
        banner("Dry run requested -- exiting without running anything")
        return

    if args.plot_only:
        make_cpu_plots()
        return

    overall_start = time.time()

    if args.experiment in ("n", "both"):
        with csv_writer(CSV_N_CPU) as (w, f):
            sweep_n_cpu(w, f, args.threads)

    if args.experiment in ("m", "both"):
        with csv_writer(CSV_M_CPU) as (w, f):
            sweep_m_cpu(w, f, args.threads)

    log(f"all CPU sweeps complete in {time.time() - overall_start:.1f}s",
        level="ok")
    make_cpu_plots()
    log("done.", level="ok")


if __name__ == "__main__":
    main()