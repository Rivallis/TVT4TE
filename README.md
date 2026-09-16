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

## Supplementary materials
We provide additional information relevant to this study on the following aspects:

Supplementary content overview
A: Datasets employed for evaluating transferability estimation metrics
B: Architectures of pretrained models
C: Extended pool of pretrained models for transferability estimation benchmarking
D: Fine-tuning procedures on downstream tasks
E: Python implementations of the proposed local discriminativity metric for transferability estimation
F: Correlation analyses between TVT scores and fine-tuning accuracies across tasks
G: Importance analyses of individual and combined metrics for transferability estimation


The details can be referred at the following URL: 
https://drive.google.com/file/d/1UoBPdE2rCalktXbS_CVu7y1jwZTDfzwC/view?usp=drive_link


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
 
