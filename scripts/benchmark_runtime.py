"""Runtime benchmark script.

Measures wall-clock time of TVT and baseline transferability metrics on
synthetic feature data matching the benchmark's typical dimensions.

Usage::

    python scripts/benchmark_runtime.py [--n-models 26] [--n-samples 2000]
        [--n-dim 2048] [--n-classes 100] [--repeats 5]

Results are NOT portable across hardware; always report the platform.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

PLATFORM_NOTE = """
NOTE: Wall-clock timing is hardware-dependent.  Report the platform alongside
any timing numbers.  The TVT paper reports 4.85 s on an unspecified server.
"""


def time_tvt(features_by_model, labels, pca_dim=64, p=1, fusion="geo", repeats=3):
    from tvt import TVT

    scorer = TVT(pca_dim=pca_dim, p=p, fusion=fusion)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        scorer.fit_score(features_by_model, labels)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-models", type=int, default=26)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-dim", type=int, default=2048)
    parser.add_argument("--n-classes", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    labels = rng.integers(0, args.n_classes, size=args.n_samples).astype(np.int64)
    features_by_model = {
        f"model_{i}": rng.standard_normal(
            (args.n_samples, args.n_dim)
        ).astype(np.float32)
        for i in range(args.n_models)
    }

    print(f"Benchmarking TVT on {args.n_models} models × {args.n_samples} samples "
          f"× {args.n_dim} dims, {args.n_classes} classes, {args.repeats} repeats")
    print(PLATFORM_NOTE)

    t = time_tvt(features_by_model, labels, repeats=args.repeats)
    print(f"TVT (geo, pca_dim=64, p=1): {t:.2f} s  (median of {args.repeats} runs)")
    print()
    print("Paper reports 4.85 s on benchmark data.  Synthetic data here may differ.")


if __name__ == "__main__":
    main()
