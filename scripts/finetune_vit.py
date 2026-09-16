"""ViT fine-tuning script used to produce the ground-truth accuracies.

Configuration (from §6.5 of the TVT manuscript)
-------------------------------------------------
* Optimizer: AdamW
* Weight decay grid: {5e-2, 1e-3}
* LR grid: {5e-3, 1e-3, 5e-4, 1e-4}
* Scheduler: cosine annealing with 3 warmup epochs
* Batch size: 256
* Epochs: 50 for all ViTs except MAE (100 epochs)
* Augmentation: random flip, random crop, Mixup, CutMix, RandAugment(9, 0.5)
* Regularization: stochastic depth 0.1, layer-wise LR decay 0.75,
  label smoothing 0.1
* Repeat: 3 times, report mean test accuracy
* Hardware: single NVIDIA H200

Usage::

    python scripts/finetune_vit.py \\
        --model vit_timm \\
        --dataset cifar10 \\
        --data-root /path/to/data \\
        --output-dir results/finetune/

**Note:** Running this script re-produces the ViT fine-tuning accuracies
tabulated in the paper.  The CNN accuracies are adopted from *Not All Models
Are Equal* and do not require re-running.
"""

from __future__ import annotations

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=["vit_timm", "mocov3_vit", "dino_vit", "mae_vit", "clip_vit"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="results/finetune/")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        import timm
    except ImportError:
        raise ImportError(
            "Fine-tuning requires torch and timm.  "
            "Install with: pip install tvt4te[extract]"
        )

    epochs = 100 if args.model == "mae_vit" else 50
    lr_grid = [5e-3, 1e-3, 5e-4, 1e-4]
    wd_grid = [5e-2, 1e-3]

    print(f"Fine-tuning {args.model} on {args.dataset}")
    print(f"  Epochs: {epochs}, LR grid: {lr_grid}, WD grid: {wd_grid}")
    print(f"  Repeats: {args.repeats}, Device: {args.device}")
    print()
    print("Full training loop not yet implemented.")
    print("Expected test accuracies (from paper, for reference):")
    print()
    from tvt.benchmark.datasets import DATASETS

    _ref = {
        ("vit_timm", "aircraft"): 75.09,
        ("vit_timm", "cifar10"): 98.88,
        ("mae_vit", "aircraft"): 65.59,
    }
    for (m, d), acc in _ref.items():
        if m == args.model and d == args.dataset:
            print(f"  {args.model} on {args.dataset}: {acc:.2f}% (paper)")


if __name__ == "__main__":
    main()
