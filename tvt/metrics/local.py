"""Local discriminability metric for TVT.

This module implements the kernel-density-based local discriminability score,
:math:`\\hat{T}^{\\text{local}}`, as described in the TVT paper.

The :class:`KernelRegression` class is **reproduced verbatim from the paper's
supplementary material** and must not be refactored (it is checked against the
printed listing in the manuscript).
"""

from __future__ import annotations

import numpy as np
import torch


__all__ = ["KernelRegression", "LocalDiscriminability"]


class KernelRegression(object):
    """Kernel regression scorer — verbatim from the paper's supplementary.

    The kernel scale :math:`\\lambda_i` is derived from the within-class
    variance of the *source* (train) distribution; the val split is scored
    against those kernel centres.

    Parameters
    ----------
    args:
        Arbitrary namespace / object whose attributes may configure the scorer.
        The published implementation accepts an ``args`` object for forward
        compatibility; the current version ignores it.
    """

    def __init__(self, args):
        self.args = args

    def compute_variance_per_sample(self, D, target_labels, src_labels):
        """Compute variance at each sample."""
        M = target_labels.reshape(-1, 1) == src_labels.reshape(1, -1)
        D_ = D * M
        return D_.sum(-1, keepdim=True) / M.sum(-1, keepdim=True)

    def score(self, train_features: np.ndarray, train_labels: np.ndarray,
                    val_features: np.ndarray, val_labels: np.ndarray):
        C = np.max(train_labels) + 1
        D = torch.cdist(
            torch.from_numpy(val_features).float(),
            torch.from_numpy(train_features).float()
            ).pow(2)
        src = torch.Tensor(train_labels).to(torch.long)
        target = torch.Tensor(val_labels).to(torch.long)
        sig2 = self.compute_variance_per_sample(D, target, src)

        Y = torch.eye(C)[src]

        pred = D.div(-sig2).softmax(dim=-1).mm(Y)
        l = pred[range(len(target)), target].log().mean().item()  # higher is better

        return l


class LocalDiscriminability:
    """Convenience wrapper around :class:`KernelRegression`.

    Parameters
    ----------
    args:
        Passed through to :class:`KernelRegression`.  May be ``None``.

    Examples
    --------
    >>> ld = LocalDiscriminability()
    >>> score = ld.score(train_X, train_y, val_X, val_y)
    """

    def __init__(self, args=None) -> None:
        self._kr = KernelRegression(args)

    def score(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
    ) -> float:
        """Compute :math:`\\hat{T}^{\\text{local}}`.

        Parameters
        ----------
        train_features:
            Source feature matrix ``(N_train, d)``.
        train_labels:
            Integer labels ``(N_train,)``.
        val_features:
            Val feature matrix ``(N_val, d)``.
        val_labels:
            Integer labels ``(N_val,)``.

        Returns
        -------
        float
            Mean log-posterior (higher is more transferable).
        """
        return self._kr.score(
            train_features.astype(np.float32),
            train_labels.astype(np.int64),
            val_features.astype(np.float32),
            val_labels.astype(np.int64),
        )
