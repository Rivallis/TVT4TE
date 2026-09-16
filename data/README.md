# Ground-Truth Fine-Tuning Accuracies

This directory contains ground-truth fine-tuning accuracy tables used by the
TVT benchmark to compute weighted Kendall's τ_w.

## Files

| File | Contents |
|---|---|
| `gt_accuracies_vit.csv` | ViT-Base fine-tuning results (5 models × 11 datasets) reported in the TVT paper. |
| `gt_accuracies_cnn_sl.csv` | Supervised CNN results (11 models × 11 datasets) — adopted from *Not All Models Are Equal* (Nguyen et al., 2022, arXiv:2207.03036). **Not yet included; add results from the paper's supplementary.** |
| `gt_accuracies_cnn_ssl.csv` | Self-supervised ResNet-50 results (10 models × 11 datasets) — same source. **Not yet included.** |

## Protocol

**ViTs** were fine-tuned end-to-end with AdamW (see §6.5 of the manuscript).
Each result is the average of 3 independent runs.

**CNNs** adopt the protocol of *Not All Models Are Equal*: grid search over
learning rate ∈ {1e-1, 1e-2, 1e-3, 1e-4} and weight decay ∈ {1e-3, 1e-4,
1e-5, 1e-6, 0}.

## License

Each dataset's fine-tuning accuracy is derived from publicly available
benchmarks.  See each dataset's individual license before use.
