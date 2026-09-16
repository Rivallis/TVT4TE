# Reproduction Guide

This document explains how to reproduce the key results from the TVT paper
(§7.1, §7.2, §7.4, §7.5) from a clean checkout.

## Prerequisites

```bash
git clone https://github.com/Rivallis/TVT4TE.git
cd TVT4TE
pip install -e ".[extract,plot]"
```

## Step 1 — Download datasets

Use the URLs in `tvt/benchmark/datasets.py` to download the 11 benchmark
datasets.  Example for CIFAR-10:

```bash
python -c "
from torchvision.datasets import CIFAR10
CIFAR10('/data/cifar10', download=True)
"
```

Repeat for all 11 datasets; store each in `/data/<dataset_name>/`.

## Step 2 — Extract features

For each model × dataset pair:

```bash
tvt extract \
    --model resnet50 \
    --dataset cifar10 \
    --data-root /data \
    --out features/resnet50_cifar10
```

This writes `features/resnet50_cifar10_features.npy` and `..._labels.npy`.

For self-supervised ResNet-50 and ViT models, download weights first (see
`tvt/benchmark/models.py` for URLs) and load them manually.

## Step 3 — Reproduce §7.1 (main comparison)

```bash
tvt benchmark --config configs/default.yaml --out results/
```

This reads pre-extracted `.npy` files, scores all models, and writes per-dataset,
per-regime, and overall weighted Kendall τ_w to `results/tau_w.json`.

Target values (from the paper):

| Regime | TVT τ_w |
|---|---|
| ID avg | 0.731 |
| OOD avg | 0.788 |
| FG avg | 0.480 |
| **Overall** | **0.666** |

## Step 4 — Reproduce §7.2 (fusion ablation)

```bash
tvt bench-fusion --config configs/fusion_ablation.yaml
python scripts/plot_results.py --out figures/
```

## Step 5 — Reproduce §7.4 (hyperparameter sensitivity)

```bash
tvt bench-sensitivity --config configs/sensitivity.yaml
python scripts/plot_results.py --out figures/
```

## Step 6 — Reproduce §7.5 (R² correlation)

The per-dataset R² values are computed from the TVT scores and ground-truth
fine-tuning accuracies in `data/gt_accuracies_vit.csv`.  No additional script
is needed; the evaluation harness (`tvt/benchmark/evaluate.py`) returns them.

## Timing (§7.3)

```bash
python scripts/benchmark_runtime.py --n-models 26 --n-samples 2000 --n-dim 2048
```

**Important:** wall-clock numbers are not portable.  The paper reports 4.85 s on
a single server GPU.  Report your hardware alongside any timing you measure.

## ViT fine-tuning (§6.5)

To re-produce the ViT ground-truth accuracies from scratch (not required to
reproduce §7.1 if you use the pre-tabulated CSV):

```bash
python scripts/finetune_vit.py \
    --model vit_timm \
    --dataset cifar10 \
    --data-root /data \
    --output-dir results/finetune/
```

Reference hardware: single NVIDIA H200.
