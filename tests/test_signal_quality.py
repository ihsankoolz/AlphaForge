"""
tests/test_signal_quality.py
=============================
Unit tests for analytics/signal_quality.py — rolling AUC, bucket accuracy,
distribution shift, recalibration triggers, feature importance drift.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from analytics.signal_quality import (
    rolling_auc, _manual_auc, bucket_accuracy,
    signal_distribution_shift, needs_recalibration,
    feature_importance_drift, signal_quality_report,
)


# ---------------------------------------------------------------------------
# Manual AUC
# ---------------------------------------------------------------------------

class TestManualAuc:
    """Tests for the Wilcoxon-Mann-Whitney AUC implementation."""

    def test_perfect_separation(self):
        """Perfect model: all positives scored higher → AUC = 1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert _manual_auc(y_true, y_score) == pytest.approx(1.0)

    def test_random_model(self):
        """Random scores → AUC ≈ 0.5."""
        np.random.seed(42)
        y_true = np.array([1, 0] * 500)
        y_score = np.random.uniform(0, 1, 1000)
        auc = _manual_auc(y_true, y_score)
        assert 0.4 < auc < 0.6

    def test_inverse_model(self):
        """Perfectly wrong model → AUC = 0.0."""
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        assert _manual_auc(y_true, y_score) == pytest.approx(0.0)

    def test_no_positives(self):
        """No positive class → AUC = 0.5 (undefined, return neutral)."""
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.5, 0.6, 0.7])
        assert _manual_auc(y_true, y_score) == 0.5


# ---------------------------------------------------------------------------
# Rolling AUC
# ---------------------------------------------------------------------------

class TestRollingAuc:
    """Tests for rolling AUC computation."""

    def test_basic_rolling(self):
        """Should produce AUC values for sufficient data."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.25, n).astype(float)
        y_score = y_true * 0.3 + np.random.uniform(0, 0.5, n)
        results = rolling_auc(y_true, y_score, window=60, step=5)
        assert len(results) > 0
        assert all("auc" in r for r in results)
        assert all("end_idx" in r for r in results)

    def test_too_short(self):
        """Data shorter than window → empty results."""
        results = rolling_auc([1, 0, 1], [0.8, 0.2, 0.7], window=60)
        assert results == []

    def test_auc_values_bounded(self):
        """All AUC values should be between 0 and 1."""
        np.random.seed(42)
        n = 300
        y_true = np.random.binomial(1, 0.3, n).astype(float)
        y_score = np.random.uniform(0, 1, n)
        results = rolling_auc(y_true, y_score, window=60, step=5)
        for r in results:
            assert 0.0 <= r["auc"] <= 1.0


# ---------------------------------------------------------------------------
# Bucket accuracy
# ---------------------------------------------------------------------------

class TestBucketAccuracy:
    """Tests for prediction bucket calibration."""

    def test_basic_buckets(self):
        """Should produce the requested number of buckets."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 100).astype(float)
        y_score = np.random.uniform(0, 1, 100)
        buckets = bucket_accuracy(y_true, y_score, n_buckets=5)
        assert len(buckets) == 5
        assert all("avg_pred" in b for b in buckets)
        assert all("actual_win_rate" in b for b in buckets)

    def test_empty_input(self):
        """Empty input → empty buckets."""
        assert bucket_accuracy([], [], n_buckets=5) == []

    def test_bucket_avg_pred_increasing(self):
        """Average prediction should increase across buckets."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 200).astype(float)
        y_score = np.random.uniform(0, 1, 200)
        buckets = bucket_accuracy(y_true, y_score, n_buckets=5)
        avg_preds = [b["avg_pred"] for b in buckets]
        # Each bucket should have higher avg prediction than the previous
        for i in range(1, len(avg_preds)):
            assert avg_preds[i] >= avg_preds[i - 1]


# ---------------------------------------------------------------------------
# Distribution shift
# ---------------------------------------------------------------------------

class TestSignalDistributionShift:
    """Tests for KL divergence between score distributions."""

    def test_identical_distributions(self):
        """Same distribution → KL ≈ 0."""
        np.random.seed(42)
        scores = np.random.uniform(0.2, 0.8, 1000)
        kl = signal_distribution_shift(scores, scores)
        assert kl < 0.01

    def test_different_distributions(self):
        """Shifted distribution → KL > 0."""
        np.random.seed(42)
        old = np.random.normal(0.4, 0.1, 1000)
        new = np.random.normal(0.7, 0.1, 1000)
        kl = signal_distribution_shift(old, new)
        assert kl > 0.1

    def test_empty_input(self):
        """Empty input → KL = 0."""
        assert signal_distribution_shift([], [0.5, 0.6]) == 0.0
        assert signal_distribution_shift([0.5], []) == 0.0

    def test_non_negative(self):
        """KL divergence should always be non-negative."""
        np.random.seed(42)
        old = np.random.uniform(0, 1, 100)
        new = np.random.uniform(0, 1, 100)
        kl = signal_distribution_shift(old, new)
        assert kl >= 0.0


# ---------------------------------------------------------------------------
# Recalibration trigger
# ---------------------------------------------------------------------------

class TestNeedsRecalibration:
    """Tests for recalibration trigger logic."""

    def test_good_auc_no_trigger(self):
        """AUC well above threshold → no trigger."""
        result = needs_recalibration([0.70, 0.68, 0.72, 0.69])
        assert result["trigger"] is False
        assert result["reason"] == "ok"

    def test_low_auc_triggers(self):
        """AUC below threshold → trigger."""
        result = needs_recalibration([0.52, 0.50, 0.48], threshold=0.55)
        assert result["trigger"] is True
        assert result["reason"] == "auc_below_threshold"

    def test_insufficient_data(self):
        """Too few AUC values → no trigger, reason = insufficient."""
        result = needs_recalibration([0.50], min_periods=3)
        assert result["trigger"] is False
        assert result["reason"] == "insufficient_data"

    def test_recent_avg_auc_correct(self):
        """Should average the last min_periods values."""
        result = needs_recalibration([0.40, 0.60, 0.80], min_periods=2)
        # Average of last 2: (0.60 + 0.80) / 2 = 0.70
        assert result["recent_avg_auc"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Feature importance drift
# ---------------------------------------------------------------------------

class TestFeatureImportanceDrift:
    """Tests for feature importance cosine similarity."""

    def test_identical_importance(self):
        """Same importance → cosine_similarity = 1.0."""
        imp = {"return_1d": 0.3, "rsi_14": 0.5, "macd_hist": 0.2}
        result = feature_importance_drift(imp, imp)
        assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-10)

    def test_completely_different(self):
        """Orthogonal importance → cosine_similarity ≈ 0."""
        old = {"feature_a": 1.0, "feature_b": 0.0}
        new = {"feature_a": 0.0, "feature_b": 1.0}
        result = feature_importance_drift(old, new)
        assert result["cosine_similarity"] == pytest.approx(0.0, abs=1e-10)

    def test_empty_input(self):
        """Empty importance → similarity = 1.0 (no drift detectable)."""
        result = feature_importance_drift({}, {})
        assert result["cosine_similarity"] == 1.0

    def test_top_changes_reported(self):
        """Should report top changed features."""
        old = {"a": 0.5, "b": 0.3, "c": 0.2}
        new = {"a": 0.1, "b": 0.1, "c": 0.8}
        result = feature_importance_drift(old, new)
        assert len(result["top_changes"]) > 0
        # Feature "c" should have the largest change
        top_feature = result["top_changes"][0]["feature"]
        assert top_feature == "c"

    def test_new_features_handled(self):
        """Features appearing only in new should not crash."""
        old = {"a": 0.5, "b": 0.5}
        new = {"a": 0.3, "b": 0.3, "c": 0.4}
        result = feature_importance_drift(old, new)
        assert 0 <= result["cosine_similarity"] <= 1.0


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------

class TestSignalQualityReport:
    """Tests for the composite signal quality report."""

    def test_basic_report_structure(self):
        """Report should contain all expected keys."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.25, n).astype(float)
        y_score = y_true * 0.3 + np.random.uniform(0.2, 0.5, n)
        report = signal_quality_report(y_true, y_score)
        assert "rolling_auc" in report
        assert "overall_auc" in report
        assert "bucket_accuracy" in report
        assert "recalibration" in report
        assert "distribution_shift_kl" in report
        assert "feature_drift" in report

    def test_with_old_scores(self):
        """When old scores provided, distribution shift should be computed."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.3, n).astype(float)
        y_score = np.random.uniform(0, 1, n)
        old_scores = np.random.uniform(0, 0.5, n)
        report = signal_quality_report(y_true, y_score,
                                       scores_old=old_scores)
        assert report["distribution_shift_kl"] is not None
        assert report["distribution_shift_kl"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
