"""
Tests for models/ensemble.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble import (
    EnsembleModel,
    train_ensemble,
    ensemble_predict,
    model_agreement,
    HAS_XGBOOST,
    HAS_LIGHTGBM,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate simple binary classification data."""
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({
        "feature_1": rng.normal(0, 1, n),
        "feature_2": rng.normal(0, 1, n),
        "feature_3": rng.normal(0, 1, n),
        "feature_4": rng.normal(0, 1, n),
        "feature_5": rng.normal(0, 1, n),
    })
    # Target correlated with features
    score = X["feature_1"] * 0.5 + X["feature_2"] * 0.3 + rng.normal(0, 0.5, n)
    y = (score > 0).astype(int)
    return X, y


@pytest.fixture
def fitted_ensemble(sample_data):
    """Pre-fitted ensemble model."""
    X, y = sample_data
    return train_ensemble(X, y)


@pytest.fixture
def test_features(sample_data):
    """Test feature matrix."""
    X, _ = sample_data
    return X.iloc[:50]


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------

class TestEnsembleModel:
    def test_default_weights(self):
        model = EnsembleModel()
        assert model.xgb_weight == 0.5
        assert model.lgb_weight == 0.5

    def test_custom_weights(self):
        model = EnsembleModel(xgb_weight=0.7)
        assert model.xgb_weight == 0.7
        assert abs(model.lgb_weight - 0.3) < 1e-10

    def test_fit_sets_is_fitted(self, sample_data):
        X, y = sample_data
        model = EnsembleModel()
        assert not model.is_fitted
        model.fit(X, y)
        assert model.is_fitted

    def test_stores_feature_cols(self, sample_data):
        X, y = sample_data
        model = EnsembleModel()
        model.fit(X, y)
        assert model.feature_cols == list(X.columns)

    @pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
    def test_xgb_model_fitted(self, fitted_ensemble):
        assert fitted_ensemble.xgb_model is not None

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
    def test_lgb_model_fitted(self, fitted_ensemble):
        assert fitted_ensemble.lgb_model is not None

    def test_predict_proba_shape(self, fitted_ensemble, test_features):
        proba = fitted_ensemble.predict_proba(test_features)
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self, fitted_ensemble, test_features):
        proba = fitted_ensemble.predict_proba(test_features)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_in_range(self, fitted_ensemble, test_features):
        proba = fitted_ensemble.predict_proba(test_features)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_not_fitted_raises(self, test_features):
        model = EnsembleModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(test_features)

    def test_custom_xgb_params(self, sample_data):
        X, y = sample_data
        model = EnsembleModel(xgb_params={"n_estimators": 50, "max_depth": 2,
                                           "verbosity": 0, "random_state": 42})
        model.fit(X, y)
        assert model.is_fitted

    def test_scale_pos_weight_applied(self, sample_data):
        X, y = sample_data
        model = EnsembleModel()
        model.fit(X, y, scale_pos_weight=3.0)
        assert model.is_fitted


# ---------------------------------------------------------------------------
# train_ensemble
# ---------------------------------------------------------------------------

class TestTrainEnsemble:
    def test_returns_fitted_model(self, sample_data):
        X, y = sample_data
        model = train_ensemble(X, y)
        assert model.is_fitted

    def test_custom_weight(self, sample_data):
        X, y = sample_data
        model = train_ensemble(X, y, xgb_weight=0.8)
        assert model.xgb_weight == 0.8

    def test_auto_scale_pos_weight(self, sample_data):
        X, y = sample_data
        model = train_ensemble(X, y)
        # Should not crash — scale_pos_weight auto-computed
        assert model.is_fitted


# ---------------------------------------------------------------------------
# ensemble_predict
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    def test_returns_1d_array(self, fitted_ensemble, test_features):
        proba = ensemble_predict(fitted_ensemble, test_features)
        assert proba.ndim == 1
        assert len(proba) == 50

    def test_values_in_range(self, fitted_ensemble, test_features):
        proba = ensemble_predict(fitted_ensemble, test_features)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_not_all_same(self, fitted_ensemble, test_features):
        """Predictions should vary across different inputs."""
        proba = ensemble_predict(fitted_ensemble, test_features)
        assert np.std(proba) > 0.01


# ---------------------------------------------------------------------------
# model_agreement
# ---------------------------------------------------------------------------

class TestModelAgreement:
    @pytest.mark.skipif(
        not (HAS_XGBOOST and HAS_LIGHTGBM),
        reason="both xgboost and lightgbm required",
    )
    def test_agreement_keys(self, fitted_ensemble, test_features):
        result = model_agreement(fitted_ensemble, test_features)
        assert result is not None
        assert "mean_agreement" in result
        assert "disagreement_pct" in result
        assert "correlation" in result
        assert "per_sample_agreement" in result

    @pytest.mark.skipif(
        not (HAS_XGBOOST and HAS_LIGHTGBM),
        reason="both xgboost and lightgbm required",
    )
    def test_agreement_in_range(self, fitted_ensemble, test_features):
        result = model_agreement(fitted_ensemble, test_features)
        assert 0.0 <= result["mean_agreement"] <= 1.0
        assert 0.0 <= result["disagreement_pct"] <= 1.0
        assert -1.0 <= result["correlation"] <= 1.0

    @pytest.mark.skipif(
        not (HAS_XGBOOST and HAS_LIGHTGBM),
        reason="both xgboost and lightgbm required",
    )
    def test_per_sample_agreement_length(self, fitted_ensemble, test_features):
        result = model_agreement(fitted_ensemble, test_features)
        assert len(result["per_sample_agreement"]) == 50

    def test_single_model_returns_none(self, sample_data):
        """If only one model is available, agreement is None."""
        X, y = sample_data
        model = EnsembleModel()
        model.xgb_model = None  # force single model
        model.lgb_model = None
        model.is_fitted = False
        result = model_agreement(model, X.iloc[:10])
        assert result is None


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

class TestFeatureImportances:
    def test_returns_series(self, fitted_ensemble):
        imp = fitted_ensemble.feature_importances()
        assert isinstance(imp, pd.Series)

    def test_length_matches_features(self, fitted_ensemble):
        imp = fitted_ensemble.feature_importances()
        assert len(imp) == 5  # 5 features in fixture

    def test_values_non_negative(self, fitted_ensemble):
        imp = fitted_ensemble.feature_importances()
        assert np.all(imp >= 0)

    def test_sums_to_approximately_one(self, fitted_ensemble):
        imp = fitted_ensemble.feature_importances()
        assert abs(imp.sum() - 1.0) < 0.1  # importance is approximately normalised


# ---------------------------------------------------------------------------
# Ensemble vs single model comparison
# ---------------------------------------------------------------------------

class TestEnsembleVsSingle:
    @pytest.mark.skipif(
        not (HAS_XGBOOST and HAS_LIGHTGBM),
        reason="both xgboost and lightgbm required",
    )
    def test_ensemble_differs_from_pure_xgb(self, sample_data, test_features):
        """Ensemble predictions should differ from pure XGBoost."""
        X, y = sample_data
        # Pure XGBoost
        model_xgb = EnsembleModel(xgb_weight=1.0)
        model_xgb.fit(X, y)
        proba_xgb = ensemble_predict(model_xgb, test_features)

        # 50/50 ensemble
        model_ens = train_ensemble(X, y, xgb_weight=0.5)
        proba_ens = ensemble_predict(model_ens, test_features)

        # Should not be identical
        assert not np.allclose(proba_xgb, proba_ens, atol=1e-6)

    @pytest.mark.skipif(
        not (HAS_XGBOOST and HAS_LIGHTGBM),
        reason="both xgboost and lightgbm required",
    )
    def test_xgb_weight_1_equals_pure_xgb(self, sample_data, test_features):
        """With xgb_weight=1.0, ensemble should match pure XGBoost."""
        X, y = sample_data
        model = EnsembleModel(xgb_weight=1.0)
        model.fit(X, y)
        proba = ensemble_predict(model, test_features)

        # Train standalone XGBoost with same params
        import xgboost as xgb_lib
        standalone = xgb_lib.XGBClassifier(**model.xgb_params)
        standalone.fit(X, y)
        proba_standalone = standalone.predict_proba(test_features)[:, 1]

        np.testing.assert_allclose(proba, proba_standalone, atol=1e-6)
