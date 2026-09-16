# Method

## TVT — Three-View Transferability

TVT estimates how well a pretrained model φ_m will transfer to a downstream
dataset T without fine-tuning any model.  It extracts the frozen features
X = {φ_m(I; θ_m⁰)}  ∈ ℝ^(d_m × N) and scores them from three complementary
viewpoints.

## The three views

### Local discriminability

A kernel density estimate with per-sample kernel scales derived from within-class
variance.  Implemented in `tvt/metrics/local.py` as `KernelRegression`, reproduced
verbatim from the paper's supplementary (the class must remain identical to the
printed listing).

The score is the mean log-posterior at the correct class:

    T̂_local = (1/N) Σ_i log p(y_i | x_i)

**Train/val split (open item §9.2):** the benchmark uses a random 80/20 split
of the feature set (seed 42) when no native train/test split is explicitly
supplied.  Pass `train_frac=None` to `TVT` and provide explicit `train_features`
/ `val_features` arrays to use the dataset's own split.

### Global discriminability

A Gaussian class posterior with a shared pooled covariance (Ledoit-Wolf
shrinkage).  Implemented in `tvt/metrics/global_.py`.

    T̂_global = (1/N) Σ_i P(y_i | x_i)

**Design choices (open item §9.3):**
- **Covariance estimator:** Ledoit-Wolf (`sklearn.covariance.LedoitWolf`),
  selected because it handles d > N gracefully and is the standard scikit-learn
  estimator for high-dimensional data.
- **Shared covariance:** pooled within-class (LDA-style); per-class would be QDA.
- **No feature whitening or L2 normalisation** applied before scoring.

### Feature variability

An unsupervised Schatten p-norm of the feature matrix via SVD:

    T̂_var = (Σ_k σ_k^p)^(1/p)

Default p = 1 (nuclear norm).  Implemented in `tvt/metrics/variability.py`.

## Normalization and fusion

Each metric is min-max normalized **across the model pool**, so scoring a single
model in isolation is semantically ill-defined.  Use `TVT.fit_score()` (pool-level)
for comparable fused scores; `TVT.score_single()` with `ref_range` for single-model
inference against a pre-computed reference range.

**PCA fusion (open item §9.5):** the manuscript defers implementation details to
supplementary material that does not actually contain them.  The implementation
uses the leading eigenvector of the 3×3 covariance matrix of the three normalised
score vectors, sign-fixed to be non-negative and normalised to sum to 1.

**Single-model path (open item §9.4):** `TVT.score_single()` without `ref_range`
returns the raw (un-normalised) fused score and issues a `UserWarning`.

## Known limitations

**TVT explains Aircraft and Cars poorly** (R² ≈ 0.12–0.14).  Both datasets
require fine-grained discrimination that depends on features actively suppressed
during coarse ImageNet pretraining.  Static feature-space geometry analysis
cannot capture these dynamics.  This is acknowledged in the paper as future work.
