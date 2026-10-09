"""Small unit tests for the new uncertainty and interpretation utilities."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from scripts import run_ci_interpretable_pipeline as advanced


def test_stratified_bootstrap_ci_returns_finite_auc_bounds():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    prob = np.array([0.1, 0.3, 0.4, 0.2, 0.6, 0.7, 0.8, 0.9])
    observed, ci = advanced._stratified_bootstrap_ci(y, prob, n_bootstrap=100, seed=11)
    assert observed["oof_auc"] == roc_auc_score(y, prob)
    assert 0.0 <= ci["oof_auc_ci_lower"] <= ci["oof_auc_ci_upper"] <= 1.0


def test_delong_auc_test_matches_sklearn_auc():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    a = np.array([0.10, 0.25, 0.30, 0.65, 0.75, 0.95, 0.20, 0.80])
    b = np.array([0.20, 0.35, 0.40, 0.50, 0.60, 0.85, 0.15, 0.70])
    auc_a, auc_b, diff, z, p = advanced._delong_auc_test(y, a, b)
    assert np.isclose(auc_a, roc_auc_score(y, a))
    assert np.isclose(auc_b, roc_auc_score(y, b))
    assert np.isclose(diff, auc_a - auc_b)
    assert np.isfinite(z) and 0.0 <= p <= 1.0


def test_holm_adjustment_is_bounded_and_order_preserving():
    adjusted = advanced._holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert np.all(adjusted >= 0.0) and np.all(adjusted <= 1.0)
    assert adjusted[0] <= adjusted[1]
    assert adjusted[0] <= adjusted[2]


def test_pca_loading_weights_normalize_each_component():
    pca = PCA(n_components=2).fit(np.array([[0, 0, 0], [1, 2, 3], [2, 4, 5], [3, 6, 9]], dtype=float))
    weights = advanced._component_loading_weights(pca)
    assert weights.shape == (2, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)


def test_kernel_shap_wrapper_on_tiny_probability_model():
    if advanced.shap is None:
        import pytest
        pytest.skip("SHAP is optional for --skip-shap runs")
    from sklearn.linear_model import LogisticRegression

    X_train = np.array([[-2.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    y_train = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=0).fit(X_train, y_train)
    explainer = advanced.shap.KernelExplainer(
        lambda values: model.predict_proba(np.asarray(values, dtype=float))[:, 1],
        X_train[:2], link="identity"
    )
    values = advanced._get_binary_shap_matrix(explainer, X_train[2:], nsamples=20)
    assert values.shape == X_train[2:].shape
    assert np.isfinite(values).all()
