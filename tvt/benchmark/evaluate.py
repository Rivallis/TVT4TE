"""Benchmark evaluation harness for TVT.

Computes weighted Kendall's :math:`\\tau_w` between TVT scores and
ground-truth fine-tuning accuracies, per dataset, per regime, and overall.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import weightedtau

from .datasets import REGIMES

__all__ = ["evaluate", "weighted_tau"]


def weighted_tau(scores: np.ndarray, gt: np.ndarray) -> float:
    """Weighted Kendall's rank correlation (Vigna 2015 construction).

    Parameters
    ----------
    scores:
        TVT scores, one per model.
    gt:
        Ground-truth fine-tuning accuracies, one per model.

    Returns
    -------
    float
        :math:`\\tau_w` in :math:`[-1, 1]`.
    """
    result = weightedtau(scores, gt)
    return float(result.statistic)


def evaluate(
    tvt_scores: Dict[str, Dict[str, float]],
    gt_accuracies: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-dataset, per-regime, and overall :math:`\\tau_w`.

    Parameters
    ----------
    tvt_scores:
        ``{dataset_name: {model_name: score}}``.
    gt_accuracies:
        ``{dataset_name: {model_name: accuracy}}``.
    model_names:
        Ordered list of model names.  If ``None``, the intersection of keys
        in the first dataset entry is used.

    Returns
    -------
    dict
        Keys are dataset names, regime names (``"ID"``, ``"OOD"``, ``"FG"``),
        and ``"overall"``.
    """
    datasets = sorted(set(tvt_scores.keys()) & set(gt_accuracies.keys()))
    if model_names is None:
        first_ds = datasets[0]
        model_names = sorted(
            set(tvt_scores[first_ds].keys()) & set(gt_accuracies[first_ds].keys())
        )

    results: Dict[str, float] = {}

    for ds in datasets:
        s = np.array([tvt_scores[ds][m] for m in model_names])
        g = np.array([gt_accuracies[ds][m] for m in model_names])
        results[ds] = weighted_tau(s, g)

    # Per-regime averages.
    regime_avgs = {}
    for regime, regime_datasets in REGIMES.items():
        vals = [results[d] for d in regime_datasets if d in results]
        if vals:
            regime_avgs[regime] = float(np.mean(vals))
            results[regime] = regime_avgs[regime]

    # Overall (mean of regime averages).
    if regime_avgs:
        results["overall"] = float(np.mean(list(regime_avgs.values())))

    return results
