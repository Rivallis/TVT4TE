"""Global discriminability metric for TVT.

This module implements :math:`\\hat{T}^{\\text{global}}`, the Gaussian class
posterior discriminability score.

Design choices (§9.3)
---------------------
Several implementation details are not pinned in the manuscript:

**Covariance estimator** — we use *Ledoit-Wolf* shrinkage
(:class:`sklearn.covariance.LedoitWolf`), which is well-suited to the regime
where the feature dimension *d* may exceed the per-class sample count.  The
paper's supplementary describes this as "a conventional method available in
standard libraries such as NumPy and scikit-learn"; Ledoit-Wolf is the canonical
scikit-learn estimator for high-dimensional data.

**Shared vs. per-class covariance** — we use a *shared* (pooled within-class)
covariance matrix.  This corresponds to Linear Discriminant Analysis, which the
paper frames as "linear separability" modelling.  A per-class covariance would
correspond to Quadratic Discriminant Analysis.

**Regularisation** — Ledoit-Wolf automatically regularises when *d > N*, so no
additional diagonal ridge is required.  For robustness, if ``np.linalg.inv``
fails, a small ridge :math:`10^{-6} \\mathbf{I}` is added.

**Feature normalisation** — no pre-whitening or L2 normalisation is applied;
the Ledoit-Wolf estimator handles ill-conditioning internally.

If you wish to experiment with alternative estimators, subclass
:class:`GlobalDiscriminability` and override ``_estimate_covariance``.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

__all__ = ["GlobalDiscriminability"]


class GlobalDiscriminability:
    """Gaussian class posterior discriminability.

    Each class is modelled as a single Gaussian with a shared, regularised
    covariance matrix (Ledoit-Wolf estimator).  The score is the mean
    class-posterior probability at the correct class:

    .. math::

        \\hat{T}^{\\text{global}} = \\frac{1}{N} \\sum_{i=1}^{N} P(y_i \\mid \\mathbf{x}_i)

    where :math:`P(y = c \\mid \\mathbf{x})` is derived from a uniform-prior
    Gaussian discriminant model.

    Parameters
    ----------
    assume_centered:
        Passed to :class:`sklearn.covariance.LedoitWolf`.  Defaults to False.

    Examples
    --------
    >>> gd = GlobalDiscriminability()
    >>> score = gd.score(X, y)
    """

    def __init__(self, assume_centered: bool = False) -> None:
        self.assume_centered = assume_centered

    def _estimate_covariance(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Pooled within-class covariance via Ledoit-Wolf.

        Parameters
        ----------
        X:
            Feature matrix ``(N, d)``.
        y:
            Integer labels ``(N,)``.

        Returns
        -------
        np.ndarray
            Covariance matrix ``(d, d)``.
        """
        classes = np.unique(y)
        d = X.shape[1]
        pooled = np.zeros((d, d))
        total = 0
        for c in classes:
            Xc = X[y == c]
            nc = len(Xc)
            if nc < 2:
                continue
            lw = LedoitWolf(assume_centered=self.assume_centered)
            lw.fit(Xc)
            pooled += lw.covariance_ * (nc - 1)
            total += nc - 1
        if total == 0:
            return np.eye(d)
        return pooled / total

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute :math:`\\hat{T}^{\\text{global}}`.

        Parameters
        ----------
        X:
            Feature matrix ``(N, d)``.
        y:
            Integer class labels ``(N,)``.

        Returns
        -------
        float
            Mean class-posterior probability at the correct class; in
            :math:`[0, 1]` (higher means more transferable).
        """
        X = X.astype(np.float64)
        y = np.asarray(y, dtype=np.int64)
        classes = np.unique(y)
        N, d = X.shape

        # Class means.
        means = {c: X[y == c].mean(axis=0) for c in classes}

        # Shared covariance.
        Sigma = self._estimate_covariance(X, y)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(d))

        # Mahalanobis-like quadratic forms: (N, C)
        C = len(classes)
        log_unnorm = np.zeros((N, C))
        for j, c in enumerate(classes):
            diff = X - means[c]  # (N, d)
            log_unnorm[:, j] = -0.5 * np.einsum(
                "ni,ij,nj->n", diff, Sigma_inv, diff
            )

        # Softmax to get posteriors.
        log_unnorm -= log_unnorm.max(axis=1, keepdims=True)  # numerical stability
        posteriors = np.exp(log_unnorm)
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        # Map original labels to column indices.
        label_to_idx = {c: j for j, c in enumerate(classes)}
        col_idx = np.array([label_to_idx[yi] for yi in y])

        return float(posteriors[np.arange(N), col_idx].mean())
