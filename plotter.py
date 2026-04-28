"""
Plot n-scaling results from results_n_scaling.csv.

Produces three views:
  1. Total time vs n (log-log), author vs CUDA with error bars.
  2. Speedup ratio vs n (log-x), with parity reference line.
  3. Per-stage breakdown for the CUDA path (stacked bars or lines), if the
     CSV has any of t_kmeans / t_affinity / t_spectral / t_propagate populated.

Usage:
  python plot_n_scaling.py
  python plot_n_scaling.py --csv my_results.csv --out my_plot.png
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ---- Defaults -----------------------------------------------------------
DEFAULT_CSV = "results_n_scaling.csv"
DEFAULT_OUT = "figure_n_scaling.png"

NUMERIC_FIELDS = (
    "total_s", "ari", "nmi",
    "t_kmeans", "t_affinity", "t_spectral", "t_propagate",
    "t_laplacian", "t_eigen", "t_spectral_kmeans",
)
INT_FIELDS = ("point", "n", "m", "d_or_h", "k", "run")


# ---- CSV loading --------------------------------------------------------

def load_csv(path):
    """Return list of row dicts with numeric fields parsed. Drops failed rows."""
    if not os.path.exists(path):
        sys.exit(f"CSV not found: {path}")

    rows = []
    n_total = n_skipped = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if row.get("status") != "ok":
                n_skipped += 1
                continue
            for k in INT_FIELDS:
                if row.get(k):
                    try:
                        row[k] = int(float(row[k]))
                    except ValueError:
                        pass
            for k in NUMERIC_FIELDS:
                if row.get(k):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)

    print(f"Loaded {len(rows)} ok rows from {path} "
          f"({n_skipped} skipped, {n_total} total)")
    return rows


# ---- Aggregation --------------------------------------------------------

def aggregate(rows, field="total_s"):
    """Group by (n, method) -> (mean, std, count) of `field`."""
    bucket = {}
    for r in rows:
        v = r.get(field)
        if v is None or v == "" or not isinstance(v, (int, float)):
            continue
        key = (r["n"], r["method"])
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


def stage_columns_present(rows):
    """Return the list of stage timing columns that have at least one value
    (in the cuda runs)."""
    candidates = ["t_kmeans", "t_affinity", "t_spectral", "t_propagate",
                  "t_laplacian", "t_eigen", "t_spectral_kmeans"]
    have = []
    for col in candidates:
        for r in rows:
            if r.get("method") == "cuda" and isinstance(r.get(col), (int, float)):
                have.append(col)
                break
    return have


# ---- Plotting -----------------------------------------------------------

def plot_total_time(ax, summary, points):
    for method, color, marker, label in (
        ("author", "#9b8bd0", "o", "Author (Python)"),
        ("cuda",   "#5a3a8a", "s", "Ours (CUDA)"),
    ):
        means = [summary.get((p, method), (np.nan,) * 3)[0] for p in points]
        stds  = [summary.get((p, method), (np.nan,) * 3)[1] for p in points]
        ax.errorbar(points, means, yerr=stds,
                    label=label, color=color, marker=marker,
                    capsize=3, lw=2, markersize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (number of points, log scale)")
    ax.set_ylabel("Total time per fit (seconds, log scale)")
    ax.set_title("Total runtime vs. dataset size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")


def plot_speedup(ax, summary, points):
    speedups, sp_std = [], []
    for p in points:
        a = summary.get((p, "author"))
        c = summary.get((p, "cuda"))
        if a is None or c is None or c[0] == 0:
            speedups.append(np.nan)
            sp_std.append(0.0)
            continue
        s = a[0] / c[0]
        # First-order error propagation: var(a/c) ~ s^2 * ((sa/a)^2 + (sc/c)^2)
        rel_var = ((a[1] / a[0]) ** 2 + (c[1] / c[0]) ** 2
                   if a[0] and c[0] else 0.0)
        speedups.append(s)
        sp_std.append(s * np.sqrt(rel_var))

    ax.errorbar(points, speedups, yerr=sp_std,
                color="#5a3a8a", marker="D", capsize=3, lw=2, markersize=7,
                label="Speedup (Python / CUDA)")
    ax.axhline(1.0, color="grey", ls="--", alpha=0.6, label="parity")
    ax.set_xscale("log")
    ax.set_xlabel("n (number of points, log scale)")
    ax.set_ylabel("Speedup factor")
    ax.set_title("CUDA speedup vs. dataset size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")

    # Annotate each point with its numeric speedup
    for p, s in zip(points, speedups):
        if np.isfinite(s):
            ax.annotate(f"{s:.2f}x", (p, s),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="#5a3a8a")


def plot_stages(ax, rows, points, stage_cols):
    """Stacked bar plot of mean per-stage times for the CUDA path."""
    # Build mean per stage per n
    stage_means = {col: [] for col in stage_cols}
    for p in points:
        per_stage = {col: [] for col in stage_cols}
        for r in rows:
            if r["method"] != "cuda" or r["n"] != p:
                continue
            for col in stage_cols:
                v = r.get(col)
                if isinstance(v, (int, float)):
                    per_stage[col].append(v)
        for col in stage_cols:
            stage_means[col].append(np.mean(per_stage[col])
                                    if per_stage[col] else 0.0)

    # Stacked bars on a log x-axis: matplotlib doesn't stack well on log,
    # so use index positions and label them with the n values.
    x_idx = np.arange(len(points))
    bottoms = np.zeros(len(points))

    # Colors that are visually distinct against the purple theme
    palette = ["#5a3a8a", "#9b8bd0", "#d4a5e8", "#7e57c2",
               "#b39ddb", "#ce93d8", "#ab47bc"]

    for i, col in enumerate(stage_cols):
        vals = np.array(stage_means[col])
        label = col.replace("t_", "")
        ax.bar(x_idx, vals, bottom=bottoms,
               label=label, color=palette[i % len(palette)],
               edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{p:,}" for p in points], rotation=30, ha="right")
    ax.set_xlabel("n (number of points)")
    ax.set_ylabel("Mean time per stage (seconds)")
    ax.set_title("CUDA per-stage breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def print_summary_table(summary, points):
    print()
    print(f"  {'n':>10}  {'author (s)':>20}  {'cuda (s)':>20}  "
          f"{'speedup':>10}  {'#runs (a/c)':>12}")
    print(f"  {'-' * 10}  {'-' * 20}  {'-' * 20}  {'-' * 10}  {'-' * 12}")
    for p in points:
        a = summary.get((p, "author"))
        c = summary.get((p, "cuda"))
        a_str = f"{a[0]:9.3f} +/- {a[1]:7.3f}" if a else "          n/a"
        c_str = f"{c[0]:9.3f} +/- {c[1]:7.3f}" if c else "          n/a"
        sp = f"{(a[0] / c[0]):8.2f}x" if (a and c and c[0]) else "      n/a"
        runs = f"{a[2] if a else 0}/{c[2] if c else 0}"
        print(f"  {p:>10,}  {a_str:>20}  {c_str:>20}  {sp:>10}  {runs:>12}")
    print()


# ---- Entry point --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-stages", action="store_true",
                        help="Skip the per-stage breakdown panel")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        sys.exit("No usable rows; nothing to plot.")

    summary = aggregate(rows, "total_s")
    points = sorted({k[0] for k in summary.keys()})
    if not points:
        sys.exit("No n values found in the data.")

    print_summary_table(summary, points)

    stage_cols = [] if args.no_stages else stage_columns_present(rows)
    have_stages = bool(stage_cols)
    if have_stages:
        print(f"Detected per-stage columns: {stage_cols}")
    else:
        print("No per-stage timing columns populated; skipping breakdown panel.")

    n_panels = 3 if have_stages else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 2:
        ax_total, ax_speedup = axes
    else:
        ax_total, ax_speedup, ax_stages = axes

    plot_total_time(ax_total, summary, points)
    plot_speedup(ax_speedup, summary, points)
    if have_stages:
        plot_stages(ax_stages, rows, points, stage_cols)

    fig.suptitle("Spectral Bridges n-scaling: Author Python vs. CUDA",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()