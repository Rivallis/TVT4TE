# Benchmark

## Task regimes

The 11 benchmark datasets are partitioned into three regimes relative to
ImageNet — the pretraining source of every PTM in the pool.  This partition is a
**contribution of the TVT paper** and should not be flattened:

| Regime | Datasets | Rationale |
|---|---|---|
| **ID** (In-Domain) | Caltech-101, CIFAR-10, CIFAR-100, VOC2007 | Distribution close to ImageNet; PTM ranking is largely stable post fine-tuning. |
| **OOD** (Out-of-Domain) | DTD, SUN397 | Texture and scene-context shift; intermediate difficulty. |
| **FG** (Fine-Grained) | Aircraft, Cars, Flowers, Food, Pets | Subtle inter-class distinctions; requires amplification of signals suppressed during coarse ImageNet pretraining. |

The regime structure matters for interpretation: TVT excels on ID and OOD but
explains less variance on FG datasets (Aircraft R² = 0.139, Cars R² = 0.121),
because fine-grained discrimination depends on features that static feature-space
analysis cannot capture well.

## Evaluation protocol

Quality is the **weighted Kendall's τ_w** (Vigna 2015 construction,
`scipy.stats.weightedtau`) between TVT scores and fine-tuning accuracies over
all 26 PTMs, computed:

1. **Per dataset** — 11 values.
2. **Per-regime average** — mean over datasets within each regime.
3. **Overall** — mean of the three regime averages.

This three-level averaging gives equal weight to each regime regardless of how
many datasets it contains.

## Baselines

The comparison baselines (kNN, LP, LogME, NLEEP, PARC, SFDA, ETran, NCTI, ITM)
are **not re-implemented** in this repository.  Their τ_w values are taken
directly from each paper's published tables.  See `data/baseline_scores.csv`
(TODO: add this file).

If you wish to re-run the baselines, refer to the respective codebases:

- [LogME](https://github.com/thuml/LogME)
- [LEEP / NLEEP](https://github.com/thuml/NLEEP)
- [SFDA](https://github.com/TencentARC/SFDA)
- [ETran](https://github.com/huggingface/etran)
- [NCTI](https://github.com/YuanGongND/NCTI)

## SUN397 note

SUN397 does not have an official benchmark test split.  In this protocol,
evaluation uses the training portion only.  The `sun397` entry in
`tvt/benchmark/datasets.py` sets `test_size=None`; attempting to load a test
split raises `ValueError`.

## Data download

Use the per-dataset URLs in `tvt/benchmark/datasets.py` to download data.  No
dataset is bundled or redistributed in this repository.

```bash
# Example: download CIFAR-10 via torchvision
python -c "from torchvision.datasets import CIFAR10; CIFAR10('/path/to/data', download=True)"
```

For datasets without a torchvision loader, see each dataset's official page.
