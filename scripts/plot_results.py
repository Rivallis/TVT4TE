"""Plotting script for TVT results.

Regenerates the paper's main figures from stored JSON result files:

* Fusion ablation bar chart (§7.2)
* Hyperparameter sensitivity heat-map (§7.4)
* Per-dataset R² scatter plot (§7.5)

Usage::

    python scripts/plot_results.py \\
        --fusion-results results/fusion_ablation.json \\
        --sensitivity-results results/sensitivity.json \\
        --out figures/

"""

from __future__ import annotations

import argparse
import json
import os


# Reference data from the paper (§7.2, §7.4).
FUSION_ABLATION = {
    "VAR only": 0.4900,
    "Global only": 0.5210,
    "Local only": 0.5340,
    "Local + Global": 0.5691,
    "Local + VAR": 0.5763,
    "Global + VAR": 0.5876,
    "PCA fusion": 0.6346,
    "Rank mean": 0.6511,
    "Arithmetic mean": 0.6633,
    "Geometric mean (TVT)": 0.6661,
}

SENSITIVITY_TABLE = {
    # (p, pca_dim): tau_w
    (1, 16): 0.571,
    (1, 32): 0.638,
    (1, 64): 0.666,
    (1, 128): 0.650,
    (2, 16): 0.584,
    (2, 32): 0.622,
    (2, 64): 0.651,
    (2, 128): 0.645,
}

R2_TABLE = {
    "Caltech101": 0.701,
    "DTD": 0.701,
    "Pets": 0.601,
    "CIFAR-10": 0.636,
    "Flowers": 0.495,
    "VOC2007": 0.444,
    "SUN397": 0.548,
    "CIFAR-100": 0.334,
    "Food": 0.379,
    "Aircraft": 0.139,
    "Cars": 0.121,
}


def plot_fusion_ablation(results: dict, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.  pip install tvt4te[plot]")
        return

    labels = list(results.keys())
    values = list(results.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=["#4C72B0"] * (len(labels) - 1) + ["#DD8452"])
    ax.set_xlabel("Weighted Kendall τ_w")
    ax.set_title("Fusion ablation — overall τ_w")
    ax.set_xlim(0, 0.75)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fusion_ablation.pdf")
    fig.savefig(path)
    print(f"Saved {path}")


def plot_sensitivity(results: dict, out_dir: str):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plot.  pip install tvt4te[plot]")
        return

    p_vals = [1, 2]
    dim_vals = [16, 32, 64, 128]
    data = np.array([[results[(p, d)] for d in dim_vals] for p in p_vals])

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(data, vmin=0.55, vmax=0.68, cmap="Blues")
    ax.set_xticks(range(len(dim_vals)))
    ax.set_xticklabels([str(d) for d in dim_vals])
    ax.set_yticks(range(len(p_vals)))
    ax.set_yticklabels([f"p={p}" for p in p_vals])
    ax.set_xlabel("PCA dimension")
    ax.set_title("Hyperparameter sensitivity (overall τ_w)")
    for i, p in enumerate(p_vals):
        for j, d in enumerate(dim_vals):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sensitivity.pdf")
    fig.savefig(path)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures/")
    parser.add_argument("--fusion-results", default=None,
                        help="JSON file with fusion results (uses paper values if omitted).")
    parser.add_argument("--sensitivity-results", default=None,
                        help="JSON file with sensitivity results (uses paper values if omitted).")
    args = parser.parse_args()

    fusion_data = FUSION_ABLATION
    if args.fusion_results and os.path.exists(args.fusion_results):
        with open(args.fusion_results) as f:
            fusion_data = json.load(f)

    sens_data = SENSITIVITY_TABLE
    if args.sensitivity_results and os.path.exists(args.sensitivity_results):
        with open(args.sensitivity_results) as f:
            raw = json.load(f)
            sens_data = {tuple(k): v for k, v in raw.items()}

    plot_fusion_ablation(fusion_data, args.out)
    plot_sensitivity(sens_data, args.out)
    print("Done.  Figures saved to", args.out)


if __name__ == "__main__":
    main()
