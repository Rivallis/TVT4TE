"""Numerical sanity tests for TVT metrics and fusion."""

from __future__ import annotations

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Feature Variability tests                                            #
# ------------------------------------------------------------------ #
from tvt.metrics.variability import FeatureVariability


def _toy_svd_matrix(n=20, d=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_variability_p1_equals_nuclear_norm():
    X = _toy_svd_matrix()
    fv = FeatureVariability(p=1)
    sv = np.linalg.svd(X, compute_uv=False)
    expected = float(np.sum(sv))
    assert abs(fv.score(X) - expected) < 1e-4


def test_variability_p2_equals_frobenius():
    X = _toy_svd_matrix()
    fv = FeatureVariability(p=2)
    expected = float(np.linalg.norm(X, "fro"))
    assert abs(fv.score(X) - expected) < 1e-4


def test_variability_large_p_approaches_spectral():
    X = _toy_svd_matrix()
    sv = np.linalg.svd(X, compute_uv=False)
    spectral = float(sv[0])
    fv = FeatureVariability(p=100)
    assert abs(fv.score(X) - spectral) < 0.5  # large but finite p is close


def test_variability_positive():
    X = _toy_svd_matrix()
    for p in [1, 2, 5]:
        assert FeatureVariability(p=p).score(X) > 0


def test_variability_rank_deficient():
    """Should return a finite value even on a rank-deficient matrix."""
    X = np.zeros((10, 10), dtype=np.float32)
    X[:5, :5] = np.eye(5)
    fv = FeatureVariability(p=1)
    val = fv.score(X)
    assert np.isfinite(val)


# ------------------------------------------------------------------ #
# Global discriminability tests                                        #
# ------------------------------------------------------------------ #
from tvt.metrics.global_ import GlobalDiscriminability


def _toy_gauss_data(seed=1):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((30, 4)) + np.array([2, 0, 0, 0])
    X1 = rng.standard_normal((30, 4)) + np.array([-2, 0, 0, 0])
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30)
    return X, y


def test_global_finite():
    X, y = _toy_gauss_data()
    gd = GlobalDiscriminability()
    val = gd.score(X, y)
    assert np.isfinite(val)


def test_global_in_range():
    X, y = _toy_gauss_data()
    gd = GlobalDiscriminability()
    val = gd.score(X, y)
    assert 0.0 <= val <= 1.0


def test_global_rank_deficient():
    """Rank-deficient covariance (d > N per class) must still return finite."""
    rng = np.random.default_rng(42)
    # 5 samples per class, 50-dim features → rank deficient per class.
    X = rng.standard_normal((10, 50)).astype(np.float32)
    y = np.array([0] * 5 + [1] * 5)
    gd = GlobalDiscriminability()
    val = gd.score(X, y)
    assert np.isfinite(val), f"Expected finite score, got {val}"


def test_global_separable_higher_than_random():
    """Well-separated classes should score higher than random noise."""
    rng = np.random.default_rng(7)
    # Separable
    X_sep = np.vstack([
        rng.standard_normal((20, 4)) + np.array([5, 0, 0, 0]),
        rng.standard_normal((20, 4)) + np.array([-5, 0, 0, 0]),
    ]).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)
    # Random
    X_rand = rng.standard_normal((40, 4)).astype(np.float32)

    gd = GlobalDiscriminability()
    assert gd.score(X_sep, y) > gd.score(X_rand, y)


# ------------------------------------------------------------------ #
# Local discriminability tests                                         #
# ------------------------------------------------------------------ #
from tvt.metrics.local import LocalDiscriminability


def test_local_finite():
    rng = np.random.default_rng(3)
    train_X = rng.standard_normal((40, 8)).astype(np.float32)
    train_y = (np.arange(40) < 20).astype(np.int64)
    val_X = rng.standard_normal((20, 8)).astype(np.float32)
    val_y = (np.arange(20) < 10).astype(np.int64)

    ld = LocalDiscriminability()
    val = ld.score(train_X, train_y, val_X, val_y)
    assert np.isfinite(val)


# ------------------------------------------------------------------ #
# Fusion tests                                                         #
# ------------------------------------------------------------------ #
from tvt.fusion import fuse


def test_fusion_geo_default():
    """Geometric mean should be in a sensible range."""
    normed = np.array([[0.8, 0.9, 0.7], [0.3, 0.4, 0.5]])
    result = fuse(normed, scheme="geo")
    assert result.shape == (2,)
    assert np.all(result >= 0)


def test_fusion_arith():
    normed = np.array([[1.0, 0.0, 0.5]])
    result = fuse(normed, scheme="arith")
    assert abs(result[0] - 0.5) < 1e-9


def test_fusion_rank():
    normed = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.5, 0.5]])
    result = fuse(normed, scheme="rank")
    # Model 0 should rank lowest, model 1 highest.
    assert result[1] > result[2] > result[0]


def test_fusion_unknown_scheme():
    with pytest.raises(ValueError):
        fuse(np.ones((3, 3)), scheme="bad_scheme")


# ------------------------------------------------------------------ #
# TVT pool-level API test                                              #
# ------------------------------------------------------------------ #
from tvt import TVT


def test_tvt_fit_score_pool():
    rng = np.random.default_rng(99)
    N, d = 60, 16
    labels = np.repeat(np.arange(3), N // 3)
    features_by_model = {
        f"model_{i}": rng.standard_normal((N, d)).astype(np.float32)
        for i in range(4)
    }
    scorer = TVT(pca_dim=8, p=1, fusion="geo")
    scores = scorer.fit_score(features_by_model, labels)
    assert set(scores.keys()) == set(features_by_model.keys())
    for v in scores.values():
        assert np.isfinite(v)
