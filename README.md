# TVT4TE

> **Status: code release pending** — manuscript under review.

We present **TVT (Three-View Transferability)**, an efficient transferability
estimation metric that unifies three complementary views of latent feature
geometry — local and global class discriminability and feature variability —
evaluated on an expanded pool of 26 PTMs spanning both CNN and ViT architectures
trained with supervised and self-supervised learning.

TVT achieves weighted Kendall's τ_w of **0.666** averaged over the benchmark
versus **0.591** for the previous best baseline (SFDA), at **4.85 s** per
evaluation versus 86.5 s for SFDA.

---

## Installation

```bash
git clone https://github.com/Rivallis/TVT4TE.git
cd TVT4TE
pip install -e .
```

For feature extraction (requires `timm` + `torchvision`):

```bash
pip install -e ".[extract]"
```

For plotting figures:

```bash
pip install -e ".[plot]"
```

---

## Quick start

```python
from tvt import TVT, KernelRegression
import numpy as np

# Pool-level scoring (correct path — normalisation is pool-relative)
scorer = TVT(pca_dim=64, p=1, fusion="geo")
features_by_model = {
    "resnet50": np.random.randn(500, 2048).astype("float32"),
    "vit_b":    np.random.randn(500, 768).astype("float32"),
}
labels = np.random.randint(0, 10, 500)
scores = scorer.fit_score(features_by_model, labels)
# {'resnet50': 0.xxx, 'vit_b': 0.xxx}

# Single-view local metric (paper-verbatim KernelRegression)
kr = KernelRegression(args=None)
l = kr.score(train_features, train_labels, val_features, val_labels)
```

---

## CLI

```
tvt extract   --model <name> --dataset <name> --data-root <path> --out <path>
tvt score     --features <path> --labels <path> --config <yaml>
tvt benchmark --config configs/default.yaml --out results/
tvt bench-fusion      --config configs/fusion_ablation.yaml
tvt bench-sensitivity --config configs/sensitivity.yaml
```

Run `tvt --help` or `tvt <command> --help` for full flag documentation.

---

## Results

### Main comparison — weighted Kendall's τ_w

| Method | ID Avg | OOD Avg | FG Avg | **Overall** |
|---|---|---|---|---|
| SFDA (prev. best) | 0.722 | 0.600 | 0.451 | 0.591 |
| **TVT (ours)** | **0.731** | **0.788** | **0.480** | **0.666** |

Full per-dataset table in [docs/benchmark.md](docs/benchmark.md).

### Fusion ablation

| Scheme | τ_w |
|---|---|
| VAR only | 0.490 |
| Global only | 0.521 |
| Local only | 0.534 |
| PCA fusion | 0.635 |
| Rank mean | 0.651 |
| Arithmetic mean | 0.663 |
| **Geometric mean (default)** | **0.666** |

### Hyperparameter sensitivity

| p \ dim | 16 | 32 | **64** | 128 |
|---|---|---|---|---|
| p=1 | 0.571 | 0.638 | **0.666** | 0.650 |
| p=2 | 0.584 | 0.622 | 0.651 | 0.645 |

Default: PCA dim 64, p = 1.

---

## Known limitations

TVT explains fine-grained datasets (Aircraft, Cars) poorly (R² ≈ 0.12–0.14)
because it relies on static feature-space geometry.  This is acknowledged in the
paper as future work.

---

## Repository layout

```
TVT4TE/
├── tvt/
│   ├── __init__.py          # TVT class + KernelRegression re-export
│   ├── fusion.py            # arith / geo / pca / rank fusion
│   ├── extract.py           # feature extraction via timm
│   ├── cli.py               # tvt CLI entry point
│   ├── metrics/
│   │   ├── local.py         # KernelRegression (verbatim) + LocalDiscriminability
│   │   ├── global_.py       # GlobalDiscriminability (Ledoit-Wolf)
│   │   └── variability.py   # FeatureVariability (Schatten p-norm)
│   └── benchmark/
│       ├── datasets.py      # 11 dataset registry
│       ├── models.py        # 26 PTM registry
│       └── evaluate.py      # weighted Kendall tau harness
├── tests/
│   └── test_metrics.py
├── scripts/
│   ├── finetune_vit.py
│   ├── benchmark_runtime.py
│   └── plot_results.py
├── docs/
│   ├── benchmark.md
│   ├── method.md
│   └── reproduction.md
├── configs/
│   ├── default.yaml
│   ├── fusion_ablation.yaml
│   └── sensitivity.yaml
├── data/
│   └── gt_accuracies_vit.csv
└── pyproject.toml
```

---

## Citation

> Citation will be added upon acceptance.

## License

See [LICENSE](LICENSE).
 
