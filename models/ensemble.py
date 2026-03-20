"""
models/ensemble.py
==================
Ensemble model stacking: combine XGBoost + LightGBM for more robust signals.

Why ensemble?
    - XGBoost and LightGBM learn slightly different decision boundaries
    - Averaging their predictions reduces variance (less overfitting)
    - Disagreement between models is a useful uncertainty signal
    - Historical studies show 2-5% AUC improvement from simple averaging

Architecture:
    1. Train XGBoost (existing model, same hyperparameters)
    2. Train LightGBM (complementary learner, different splits)
    3. Combine via weighted average (default 50/50, tunable)
    4. Track model agreement as a confidence metric

Usage:
    from models.ensemble import (
        EnsembleModel,
        train_ensemble,
        ensemble_predict,
        model_agreement,
    )
"""

import numpy as np
import pandas as pd
import pickle
import os

# LightGBM may not be installed — graceful fallback
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class EnsembleModel:
    """
    Stacked ensemble of XGBoost + LightGBM classifiers.

    Stores both models and produces a weighted average prediction.
    Falls back to single-model prediction if one library is unavailable.

    Parameters
    ----------
    xgb_weight : float
        Weight for XGBoost predictions in [0, 1]. LightGBM gets (1 - xgb_weight).
    xgb_params : dict, optional
        Override XGBoost hyperparameters.
    lgb_params : dict, optional
        Override LightGBM hyperparameters.
    """

    def __init__(self, xgb_weight=0.5, xgb_params=None, lgb_params=None):
        self.xgb_weight = xgb_weight
        self.lgb_weight = 1.0 - xgb_weight
        self.xgb_model = None
        self.lgb_model = None
        self.feature_cols = None
        self.is_fitted = False

        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 30,
            "eval_metric": "auc",
            "random_state": 42,
            "verbosity": 0,
        }

        self.lgb_params = lgb_params or {
            "n_estimators": 300,
            "max_depth": 5,          # slightly deeper than XGBoost
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_samples": 30,
            "random_state": 43,      # different seed for diversity
            "verbosity": -1,
        }

    def fit(self, X, y, scale_pos_weight=None):
        """
        Train both models on the same data.

        Parameters
        ----------
        X : DataFrame or array
            Feature matrix.
        y : array-like
            Binary target.
        scale_pos_weight : float, optional
            Class imbalance weight.  Applied to both models.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_cols = list(X.columns)

        # XGBoost
        if HAS_XGBOOST:
            xgb_p = self.xgb_params.copy()
            if scale_pos_weight is not None:
                xgb_p["scale_pos_weight"] = scale_pos_weight
            self.xgb_model = xgb.XGBClassifier(**xgb_p)
            self.xgb_model.fit(X, y)

        # LightGBM
        if HAS_LIGHTGBM:
            lgb_p = self.lgb_params.copy()
            if scale_pos_weight is not None:
                lgb_p["scale_pos_weight"] = scale_pos_weight
            self.lgb_model = lgb.LGBMClassifier(**lgb_p)
            self.lgb_model.fit(X, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Return weighted average predicted probabilities.

        Parameters
        ----------
        X : DataFrame or array
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2) — columns are [P(class=0), P(class=1)].
        """
        if not self.is_fitted:
            raise RuntimeError("EnsembleModel is not fitted yet.")

        probas = []
        weights = []

        if self.xgb_model is not None:
            probas.append(self.xgb_model.predict_proba(X))
            weights.append(self.xgb_weight)

        if self.lgb_model is not None:
            probas.append(self.lgb_model.predict_proba(X))
            weights.append(self.lgb_weight)

        if not probas:
            raise RuntimeError("No models available. Install xgboost or lightgbm.")

        # Normalise weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        combined = sum(w * p for w, p in zip(weights, probas))
        return combined

    def feature_importances(self):
        """
        Return averaged feature importance from both models.

        Returns
        -------
        pd.Series or None
            Feature importance indexed by feature name.
        """
        importances = []
        weights = []

        if self.xgb_model is not None:
            importances.append(self.xgb_model.feature_importances_)
            weights.append(self.xgb_weight)

        if self.lgb_model is not None:
            importances.append(self.lgb_model.feature_importances_)
            weights.append(self.lgb_weight)

        if not importances:
            return None

        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        combined = sum(w * imp for w, imp in zip(weights, importances))

        if self.feature_cols is not None:
            return pd.Series(combined, index=self.feature_cols)
        return pd.Series(combined)


def train_ensemble(X_train, y_train, xgb_weight=0.5):
    """
    Convenience function to train an ensemble model.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : array-like
        Binary target.
    xgb_weight : float
        XGBoost weight in [0, 1].

    Returns
    -------
    EnsembleModel
        Fitted model.
    """
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = EnsembleModel(xgb_weight=xgb_weight)
    model.fit(X_train, y_train, scale_pos_weight=scale_pos_weight)
    return model


def ensemble_predict(model, X):
    """
    Get class-1 probabilities from an ensemble model.

    Parameters
    ----------
    model : EnsembleModel
        Fitted ensemble.
    X : DataFrame or array
        Features.

    Returns
    -------
    np.ndarray
        1-D array of P(class=1) for each sample.
    """
    return model.predict_proba(X)[:, 1]


def model_agreement(model, X):
    """
    Measure agreement between XGBoost and LightGBM predictions.

    Higher agreement = higher confidence in the ensemble prediction.
    Low agreement = models disagree, prediction is uncertain.

    Parameters
    ----------
    model : EnsembleModel
        Fitted ensemble with both models.
    X : DataFrame or array
        Features.

    Returns
    -------
    dict
        {"mean_agreement": float, "disagreement_pct": float,
         "correlation": float, "per_sample_agreement": np.ndarray}
        Returns None if only one model is available.
    """
    if model.xgb_model is None or model.lgb_model is None:
        return None

    xgb_proba = model.xgb_model.predict_proba(X)[:, 1]
    lgb_proba = model.lgb_model.predict_proba(X)[:, 1]

    # Agreement = 1 - |xgb - lgb| (per sample)
    per_sample = 1.0 - np.abs(xgb_proba - lgb_proba)

    # Percentage of samples where models disagree on class (>0.5 vs <0.5)
    xgb_class = (xgb_proba > 0.5).astype(int)
    lgb_class = (lgb_proba > 0.5).astype(int)
    disagree_pct = float(np.mean(xgb_class != lgb_class))

    # Correlation between probability predictions
    if len(xgb_proba) < 2:
        corr = 1.0
    else:
        corr = float(np.corrcoef(xgb_proba, lgb_proba)[0, 1])

    return {
        "mean_agreement": float(np.mean(per_sample)),
        "disagreement_pct": disagree_pct,
        "correlation": corr,
        "per_sample_agreement": per_sample,
    }
