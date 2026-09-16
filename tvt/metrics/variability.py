"""Feature variability metric for TVT.

This module implements :math:`\\hat{T}^{\\text{var}}`, an unsupervised measure
of the representational capacity of frozen features via the Schatten *p*-norm of
the feature matrix.

Special cases (verified by tests):

* ``p=1`` — nuclear norm (sum of singular values)
* ``p=2`` — Frobenius norm (:math:`\\sqrt{\\sum \\sigma_k^2}`)
* ``p→∞`` — spectral norm (largest singular value)
"""

from __future__ import annotations

import numpy as np

__all__ = ["FeatureVariability"]


class FeatureVariability:
    """Schatten *p*-norm of the feature matrix.

    .. math::

        \\hat{T}^{\\text{var}} = \\|\\mathbf{X}\\|_p
            = \\left(\\sum_{k=1}^{\\min(d,N)} \\sigma_k^p(\\mathbf{X})\\right)^{1/p}

    The SVD is computed on whichever orientation is smaller to avoid forming
    the large :math:`d \\times d` Gram matrix when :math:`d \\gg N`.

    Parameters
    ----------
    p:
        Schatten norm order.  Default is 1 (nuclear norm).

    Examples
    --------
    >>> fv = FeatureVariability(p=1)
    >>> score = fv.score(X)
    """

    def __init__(self, p: float = 1) -> None:
        if p <= 0:
            raise ValueError(f"p must be positive; got {p}")
        self.p = p

    def score(self, X: np.ndarray) -> float:
        """Compute :math:`\\|\\mathbf{X}\\|_p`.

        Parameters
        ----------
        X:
            Feature matrix ``(N, d)``.

        Returns
        -------
        float
            Schatten *p*-norm (higher means more representational capacity).
        """
        X = np.asarray(X, dtype=np.float64)
        # Use the economy SVD on the smaller dimension.
        singular_values = np.linalg.svd(X, compute_uv=False)
        return float(np.sum(singular_values ** self.p) ** (1.0 / self.p))
