"""TVT4TE — Three-View Transferability for pretrained vision models.

**Status: code release pending** (manuscript under review).

Public API
----------
>>> from tvt import TVT, KernelRegression
>>> scorer = TVT(pca_dim=64, p=1, fusion="geo")
>>> scores = scorer.fit_score(features_by_model, labels)

Design note — pool-relative normalization
-----------------------------------------
All three component metrics are min-max normalised *across the model pool* before
fusion.  Scoring a single model in isolation is therefore ill-defined.  The
:class:`TVT` class exposes two paths:

* ``fit_score(features_by_model, labels)`` — the correct pool-level path;
  normalises over all models supplied in the dict.
* ``score_single(features, labels, ref_range=None)`` — convenience path for a
  single model; if ``ref_range`` is ``None`` the raw (un-normalised) geometric
  mean of the three component scores is returned together with a warning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

from .fusion import fuse
from .metrics.global_ import GlobalDiscriminability
from .metrics.local import KernelRegression, LocalDiscriminability
from .metrics.variability import FeatureVariability

__all__ = ["TVT", "KernelRegression"]

logger = logging.getLogger(__name__)


class TVT:
    """Three-View Transferability scorer.

    Parameters
    ----------
    pca_dim:
        If > 0, reduce features to this many dimensions with PCA before scoring.
        Default is 64 (the paper's best setting, see §7.4).
    p:
        Schatten *p*-norm order for the variability metric.  Default is 1
        (nuclear norm).
    fusion:
        Fusion scheme — one of ``"geo"`` (default), ``"arith"``, ``"pca"``,
        ``"rank"``.
    train_frac:
        Fraction of each model's features used as the *source* split for the
        local metric when no explicit train/val split is supplied.  The remaining
        fraction is the *val* split.

        **Open item (§9.2):** the manuscript does not specify whether the
        reported numbers use the dataset's native train/test split or a random
        split of the training set.  The default here (``train_frac=0.8``,
        ``seed=42``) is the natural reading; pass ``train_frac=None`` to supply
        explicit ``train_features``/``val_features`` arrays instead.
    seed:
        Random seed for the train/val split when ``train_frac`` is not None.
    """

    def __init__(
        self,
        pca_dim: int = 64,
        p: float = 1,
        fusion: str = "geo",
        train_frac: Optional[float] = 0.8,
        seed: int = 42,
    ) -> None:
        self.pca_dim = pca_dim
        self.p = p
        self.fusion = fusion
        self.train_frac = train_frac
        self.seed = seed

        self._local = LocalDiscriminability()
        self._global = GlobalDiscriminability()
        self._var = FeatureVariability(p=p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_pca(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA reduction if ``pca_dim`` > 0.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(N, d)``.

        Returns
        -------
        np.ndarray
            Reduced feature matrix of shape ``(N, min(pca_dim, d))``.
        """
        if self.pca_dim <= 0 or X.shape[1] <= self.pca_dim:
            return X
        # Zero-centre then project onto leading eigenvectors.
        X_c = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        return X_c @ Vt[: self.pca_dim].T

    def _train_val_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split *X*, *y* into train and val portions.

        Parameters
        ----------
        X:
            Feature matrix ``(N, d)``.
        y:
            Integer label vector ``(N,)``.

        Returns
        -------
        train_X, train_y, val_X, val_y
        """
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        split = int(len(X) * self.train_frac)
        return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

    def _score_one(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_X: Optional[np.ndarray] = None,
        train_y: Optional[np.ndarray] = None,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """Compute raw (un-normalised) component scores for one model.

        Returns
        -------
        (local_score, global_score, var_score)
        """
        X = self._maybe_pca(X)

        if train_X is None:
            if self.train_frac is None:
                raise ValueError(
                    "Either supply explicit train/val arrays or set train_frac."
                )
            train_X, train_y, val_X, val_y = self._train_val_split(X, y)
        else:
            train_X = self._maybe_pca(train_X)
            val_X = self._maybe_pca(val_X)

        local = self._local.score(train_X, train_y, val_X, val_y)
        glob = self._global.score(X, y)
        var = self._var.score(X)
        return local, glob, var

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_score(
        self,
        features_by_model: Dict[str, np.ndarray],
        labels: np.ndarray,
        *,
        train_features_by_model: Optional[Dict[str, np.ndarray]] = None,
        train_labels: Optional[np.ndarray] = None,
        val_features_by_model: Optional[Dict[str, np.ndarray]] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Score a pool of pretrained models.

        This is the **correct** API for pool-level fused scores because
        min-max normalisation is computed across all models in *features_by_model*.

        Parameters
        ----------
        features_by_model:
            Mapping ``{model_name: feature_matrix}`` where each feature matrix
            has shape ``(N, d_m)``.  Labels are shared across models and are
            used by the global and variability metrics (which operate on the full
            combined feature set) as well as the random train/val split when no
            explicit split is provided.
        labels:
            Integer class labels ``(N,)`` for the full feature set.
        train_features_by_model:
            Optional per-model train split.  When supplied, *train_labels*,
            *val_features_by_model*, and *val_labels* must also be provided.
        train_labels:
            Labels for the train split ``(N_train,)``.  Required when
            *train_features_by_model* is provided.
        val_features_by_model:
            Optional per-model val split.
        val_labels:
            Labels for the val split ``(N_val,)``.

        Returns
        -------
        dict
            Mapping ``{model_name: fused_score}``.
        """
        model_names = list(features_by_model.keys())
        raw: Dict[str, Tuple[float, float, float]] = {}

        use_explicit_split = train_features_by_model is not None
        if use_explicit_split and train_labels is None:
            raise ValueError(
                "train_labels must be provided when train_features_by_model is given."
            )
        for name in model_names:
            X = features_by_model[name]
            if use_explicit_split:
                raw[name] = self._score_one(
                    X,
                    labels,
                    train_X=train_features_by_model[name],
                    train_y=train_labels,
                    val_X=val_features_by_model[name],
                    val_y=val_labels,
                )
            else:
                raw[name] = self._score_one(X, labels)

        # Stack into (M, 3) matrix for normalisation and fusion.
        M = len(model_names)
        scores_matrix = np.array([raw[n] for n in model_names])  # (M, 3)

        # Min-max normalise each view across the model pool.
        lo = scores_matrix.min(axis=0, keepdims=True)
        hi = scores_matrix.max(axis=0, keepdims=True)
        denom = hi - lo
        denom[denom == 0] = 1.0  # avoid division by zero for constant views
        normed = (scores_matrix - lo) / denom  # (M, 3)

        # Fuse.
        fused_scores = fuse(normed, scheme=self.fusion)  # (M,)

        return {name: float(fused_scores[i]) for i, name in enumerate(model_names)}

    def score_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ref_range: Optional[Dict[str, Tuple[float, float]]] = None,
        train_X: Optional[np.ndarray] = None,
        train_y: Optional[np.ndarray] = None,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> float:
        """Score a single model.

        Because normalization is pool-relative, this function either requires a
        *ref_range* dict (pre-computed min/max from a reference pool) or returns
        an un-normalised raw score with a warning.

        Parameters
        ----------
        X:
            Feature matrix ``(N, d)``.
        y:
            Labels ``(N,)``.
        ref_range:
            Optional mapping ``{"local": (min, max), "global": (min, max),
            "var": (min, max)}`` used to normalise the component scores.
        train_X, train_y, val_X, val_y:
            Explicit train/val split for the local metric.

        Returns
        -------
        float
            Fused score (normalised if *ref_range* is given; raw otherwise).
        """
        local, glob, var = self._score_one(X, y, train_X, train_y, val_X, val_y)
        raw = np.array([[local, glob, var]])

        if ref_range is not None:
            keys = ["local", "global", "var"]
            lo = np.array([ref_range[k][0] for k in keys])
            hi = np.array([ref_range[k][1] for k in keys])
            denom = hi - lo
            denom[denom == 0] = 1.0
            normed = (raw - lo) / denom
        else:
            warnings.warn(
                "score_single called without ref_range: normalisation is "
                "pool-relative and the returned score is therefore not "
                "comparable to pool-level fused scores.",
                UserWarning,
                stacklevel=2,
            )
            normed = raw  # un-normalised

        return float(fuse(normed, scheme=self.fusion)[0])
