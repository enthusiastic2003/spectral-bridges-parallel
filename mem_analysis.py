"""
Minimal profiling runner.

Runs ONE CUDA fit at each of 3 representative n values (with fixed m=250)
and 3 representative m values (with fixed n=20000). No internal measurement
of memory or time -- those are handled externally by `nsys` (GPU memory) and
`/usr/bin/time -v` (host RSS).

Reuses data loaders and the CUDA runner from test_suit_speedups.py.

Usage:
  /usr/bin/time -v nsys profile -o profile_n10M --trace=cuda \
      python profile_run.py --point n=10000000

  # Or run all six points one after another (in a single process):
  /usr/bin/time -v nsys profile -o profile_all --trace=cuda \
      python profile_run.py --all

For paper reporting, prefer the per-point runs because /usr/bin/time and nsys
will both report the *maximum across the entire process* -- if you run all six
points sequentially you'll only see the peak of the largest one.
"""

import argparse
import os
import sys

from test_suit_speedups import (
    make_synthetic, make_mnist_pca, get_mnist,
    run_cuda,
    log, banner, print_environment,
    N_SCALING_D, N_SCALING_K,
    M_SCALING_H, M_SCALING_K,
    BASE_SEED, NUM_THREADS,
)

from test_suit_speedups_cpu import run_cpu

sys.path.append(os.path.abspath("build/"))
import specbridge


# ---- Configuration ------------------------------------------------------

# 3 representative n values (fixed m = 250, synthetic data)
N_POINTS = [
    ("n=10000",    {"experiment": "n", "n": 10_000,     "m": 250}),
    ("n=100000",   {"experiment": "n", "n": 100_000,    "m": 250}),
    ("n=10000000", {"experiment": "n", "n": 10_000_000, "m": 250}),
]

# 3 representative m values (fixed n = 20k, MNIST h=64)
M_POINTS = [
    ("m=250",  {"experiment": "m", "n": 20_000, "m": 250}),
    ("m=2000", {"experiment": "m", "n": 20_000, "m": 2000}),
    ("m=8000", {"experiment": "m", "n": 20_000, "m": 8000}),
]

ALL_POINTS = dict(N_POINTS + M_POINTS)


# ---- Single fit ---------------------------------------------------------

def run_point(label, spec, seed, device):
    banner(f"Profiling point: {label}")
    n = spec["n"]
    m = spec["m"]

    if spec["experiment"] == "n":
        d = N_SCALING_D
        k = N_SCALING_K
        log(f"generating synthetic data: n={n:,}, d={d}, k={k}")
        X, y = make_synthetic(n, d, k, seed)
    else:
        h = M_SCALING_H
        k = M_SCALING_K
        log(f"loading MNIST PCA data: n={n:,}, h={h}, k={k}")
        X, y = make_mnist_pca(n, h, seed)

    log(f"starting {device.upper()} fit (n={n:,}, m={m})")

    if device == "cuda":
        specbridge.set_num_threads(NUM_THREADS)  # Ensure CUDA runner uses the same num_threads
        result = run_cuda(X, y, k, m, seed)
    else:
        specbridge.set_num_threads(NUM_THREADS)  # Ensure CPU runner uses the same num_threads
        result = run_cpu(X, y, k, m, seed, NUM_THREADS)


    log(f"done: total_s={result.get('total_s', 0):.2f}, "
        f"ARI={result.get('ari', 0):.3f}, "
        f"NMI={result.get('nmi', 0):.3f}")

    del X, y
    return result


# ---- Entry point --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--point", choices=list(ALL_POINTS.keys()),
                        help="Run a single representative point")
    parser.add_argument("--all", action="store_true",
                        help="Run all six points sequentially in one process")
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help=f"Random seed (default: {BASE_SEED})")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on (default: cuda): cuda or cpu")
    
    args = parser.parse_args()

    if not args.point and not args.all:
        parser.error("specify either --point <name> or --all")

    print_environment()
    log(f"setting specbridge num_threads to {NUM_THREADS}")
    specbridge.set_num_threads(NUM_THREADS)

    # Prime MNIST cache once if we'll need it
    needs_mnist = (
        args.all or
        (args.point and ALL_POINTS[args.point]["experiment"] == "m")
    )
    if needs_mnist:
        log("priming MNIST cache...")
        get_mnist()

    if args.all:
        for label, spec in N_POINTS + M_POINTS:
            run_point(label, spec, args.seed, args.device)
    else:
        run_point(args.point, ALL_POINTS[args.point], args.seed, args.device)

    log("done.")


if __name__ == "__main__":
    main()