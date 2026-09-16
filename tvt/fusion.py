"""Score fusion schemes for TVT.

All schemes operate on a normalised score matrix of shape ``(M, L)`` where *M*
is the number of models and *L* is the number of views (3 for TVT: local,
global, var), and return a fused score vector of shape ``(M,)``.

Available schemes
-----------------
``"arith"``
    Arithmetic mean.
``"geo"``
    Geometric mean (default).  Uses :math:`\\epsilon = 10^{-6}` for numerical
    stability in the log.
``"pca"``
    Weighted sum, where weights are the leading eigenvector of the
    :math:`L \\times L` covariance matrix of the *column* score vectors,
    normalised to sum to 1 and sign-fixed so all weights are non-negative.

    **Open item (§9.5):** the manuscript defers implementation details to
    supplementary material that does not contain them.  The implementation here
    follows the natural reading (leading eigenvector of the score covariance).
``"rank"``
    Mean of per-view ascending ranks (rank 1 = worst).
"""

from __future__ import annotations

import warnings
import numpy as np

__all__ = ["fuse"]

_EPS = 1e-6


def _fuse_arith(normed: np.ndarray) -> np.ndarray:
    return normed.mean(axis=1)


def _fuse_geo(normed: np.ndarray) -> np.ndarray:
    log_sum = np.log(normed + _EPS).mean(axis=1)
    return np.exp(log_sum) - _EPS


def _fuse_pca(normed: np.ndarray) -> np.ndarray:
    """PCA-weighted fusion.

    Weights are the leading eigenvector of the covariance matrix of the *L*
    score columns, normalised to sum to 1.  If the leading eigenvector has any
    negative components they are sign-flipped column-wise (the eigenvector is
    determined only up to sign).

    When *M < L* (fewer models than views), the covariance matrix is
    rank-deficient; we fall back to arithmetic mean and emit a warning.
    """
    M, L = normed.shape
    if M < L:
        warnings.warn(
            f"pca fusion requires at least {L} models; got {M}.  "
            "Falling back to arithmetic mean.",
            UserWarning,
            stacklevel=3,
        )
        return _fuse_arith(normed)

    cov = np.cov(normed.T)  # (L, L)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Leading eigenvector corresponds to the *largest* eigenvalue.
    leading = eigenvectors[:, np.argmax(eigenvalues)]
    # Sign convention: weights must be non-negative.
    if leading.sum() < 0:
        leading = -leading
    if np.any(leading < 0):
        leading = np.abs(leading)
    weights = leading / leading.sum()
    return normed @ weights


def _fuse_rank(normed: np.ndarray) -> np.ndarray:
    M, L = normed.shape
    # Rank each column in ascending order (rank 1 = smallest score = worst).
    ranks = np.zeros_like(normed)
    for col in range(L):
        order = np.argsort(normed[:, col])
        ranks[order, col] = np.arange(1, M + 1)
    return ranks.mean(axis=1)


_SCHEMES = {
    "arith": _fuse_arith,
    "geo": _fuse_geo,
    "pca": _fuse_pca,
    "rank": _fuse_rank,
}


def fuse(normed: np.ndarray, scheme: str = "geo") -> np.ndarray:
    """Fuse normalised per-view scores into a single score per model.

    Parameters
    ----------
    normed:
        Min-max normalised score matrix of shape ``(M, L)`` where *M* is the
        number of models and *L* the number of views.
    scheme:
        One of ``"arith"``, ``"geo"``, ``"pca"``, ``"rank"``.

    Returns
    -------
    np.ndarray
        Fused scores of shape ``(M,)``.
    """
    if scheme not in _SCHEMES:
        raise ValueError(
            f"Unknown fusion scheme {scheme!r}.  "
            f"Choose from {list(_SCHEMES.keys())}."
        )
    return _SCHEMES[scheme](normed)
