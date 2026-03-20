"""
tests/test_regime_analysis.py
==============================
Unit tests for analytics/regime_analysis.py — transition matrix,
regime-conditional performance, persistence, accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from analytics.regime_analysis import (
    transition_matrix, regime_conditional_performance,
    regime_persistence, regime_accuracy, regime_report,
)


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    """Tests for empirical regime transition matrix."""

    def test_basic_transitions(self):
        """Known sequence should produce correct transition counts."""
        # 0→0, 0→1, 1→1, 1→2, 2→0
        regimes = [0, 0, 1, 1, 2, 0]
        result = transition_matrix(regimes, labels=[0, 1, 2])
        counts = result["counts"]
        # From 0: 0→0=1, 0→1=1, 0→2=0
        assert counts[0][0] == 1
        assert counts[0][1] == 1
        assert counts[0][2] == 0
        # From 1: 1→0=0, 1→1=1, 1→2=1
        assert counts[1][1] == 1
        assert counts[1][2] == 1
        # From 2: 2→0=1
        assert counts[2][0] == 1

    def test_rows_sum_to_one(self):
        """Each row of the transition matrix should sum to 1.0."""
        regimes = [0, 0, 1, 1, 2, 0, 1, 2, 2, 0]
        result = transition_matrix(regimes, labels=[0, 1, 2])
        matrix = result["matrix"]
        for row in matrix:
            row_sum = sum(row)
            if row_sum > 0:
                assert row_sum == pytest.approx(1.0, abs=1e-10)

    def test_single_regime(self):
        """All same regime → self-transition = 1.0."""
        regimes = [0, 0, 0, 0, 0]
        result = transition_matrix(regimes, labels=[0])
        assert result["matrix"][0][0] == pytest.approx(1.0)

    def test_too_short(self):
        """Sequence of length 1 → empty matrix."""
        result = transition_matrix([0])
        assert result["matrix"] == []

    def test_empty(self):
        """Empty sequence → empty matrix."""
        result = transition_matrix([])
        assert result["matrix"] == []

    def test_string_labels(self):
        """Should work with string regime labels."""
        regimes = ["bull", "bull", "bear", "choppy", "bull"]
        result = transition_matrix(
            regimes, labels=["bull", "choppy", "bear"])
        assert len(result["labels"]) == 3
        assert len(result["matrix"]) == 3


# ---------------------------------------------------------------------------
# Regime-conditional performance
# ---------------------------------------------------------------------------

class TestRegimeConditionalPerformance:
    """Tests for per-regime performance metrics."""

    def test_basic_performance(self):
        """Bull regime with positive returns should have positive Sharpe."""
        returns = np.array([0.01, 0.005, -0.002, -0.01, 0.003, 0.008])
        regimes = np.array([0, 0, 0, 2, 2, 0])
        result = regime_conditional_performance(returns, regimes)
        assert "0" in result  # bull
        assert "2" in result  # bear
        assert result["0"]["count"] == 4
        assert result["2"]["count"] == 2

    def test_bull_better_than_bear(self):
        """Bull mean return should exceed bear mean return."""
        np.random.seed(42)
        bull_returns = np.random.normal(0.005, 0.01, 50)
        bear_returns = np.random.normal(-0.005, 0.015, 50)
        returns = np.concatenate([bull_returns, bear_returns])
        regimes = np.array([0] * 50 + [2] * 50)
        result = regime_conditional_performance(returns, regimes)
        assert result["0"]["mean_return"] > result["2"]["mean_return"]

    def test_empty_input(self):
        """Empty inputs → empty result."""
        assert regime_conditional_performance([], []) == {}

    def test_win_rate_bounded(self):
        """Win rate should be between 0 and 1."""
        returns = [0.01, -0.01, 0.005, -0.002, 0.01]
        regimes = [0, 0, 0, 0, 0]
        result = regime_conditional_performance(returns, regimes)
        assert 0.0 <= result["0"]["win_rate"] <= 1.0

    def test_sharpe_sign_matches_returns(self):
        """Positive mean returns → positive Sharpe."""
        returns = [0.01, 0.02, 0.005, 0.015]
        regimes = [0, 0, 0, 0]
        result = regime_conditional_performance(returns, regimes)
        assert result["0"]["sharpe"] > 0


# ---------------------------------------------------------------------------
# Regime persistence
# ---------------------------------------------------------------------------

class TestRegimePersistence:
    """Tests for average regime duration."""

    def test_known_durations(self):
        """Known pattern should produce correct durations."""
        # Bull for 3 days, then bear for 2, then bull for 4
        regimes = [0, 0, 0, 2, 2, 0, 0, 0, 0]
        result = regime_persistence(regimes)
        assert "0" in result
        assert "2" in result
        # Bull: episodes of 3 and 4 → avg = 3.5
        assert result["0"]["avg_duration"] == pytest.approx(3.5)
        assert result["0"]["n_episodes"] == 2
        assert result["0"]["max_duration"] == 4
        assert result["0"]["min_duration"] == 3
        # Bear: 1 episode of 2
        assert result["2"]["avg_duration"] == 2.0
        assert result["2"]["n_episodes"] == 1

    def test_single_regime(self):
        """All same regime → one episode with full length."""
        regimes = [1, 1, 1, 1, 1]
        result = regime_persistence(regimes)
        assert result["1"]["avg_duration"] == 5.0
        assert result["1"]["n_episodes"] == 1

    def test_alternating(self):
        """Alternating regimes → each episode = 1 day."""
        regimes = [0, 1, 0, 1, 0, 1]
        result = regime_persistence(regimes)
        assert result["0"]["avg_duration"] == 1.0
        assert result["1"]["avg_duration"] == 1.0

    def test_empty(self):
        """Empty input → empty result."""
        assert regime_persistence([]) == {}


# ---------------------------------------------------------------------------
# Regime accuracy
# ---------------------------------------------------------------------------

class TestRegimeAccuracy:
    """Tests for predicted vs realised regime accuracy."""

    def test_perfect_bull_prediction(self):
        """All bull predictions with positive returns → 100% accuracy."""
        predicted = [0, 0, 0, 0, 0]
        returns = [0.01, 0.005, 0.002, 0.008, 0.003]
        result = regime_accuracy(predicted, returns)
        assert result["overall_accuracy"] == 1.0

    def test_wrong_predictions(self):
        """Bull predictions with negative returns → low accuracy."""
        predicted = [0, 0, 0, 0]
        returns = [-0.02, -0.01, -0.03, -0.015]
        result = regime_accuracy(predicted, returns)
        assert result["overall_accuracy"] == 0.0

    def test_string_labels(self):
        """Should handle string labels like 'bull', 'bear'."""
        predicted = ["bull", "bull", "bear", "bear"]
        returns = [0.01, 0.005, -0.01, -0.02]
        result = regime_accuracy(predicted, returns)
        assert result["overall_accuracy"] == 1.0

    def test_empty_input(self):
        """Empty inputs → 0 accuracy."""
        result = regime_accuracy([], [])
        assert result["overall_accuracy"] == 0.0

    def test_regime_distribution(self):
        """Distribution should count predictions per regime."""
        predicted = [0, 0, 0, 1, 1, 2]
        returns = [0.01, 0.01, 0.01, 0.001, -0.001, -0.01]
        result = regime_accuracy(predicted, returns)
        assert result["regime_distribution"]["0"] == 3
        assert result["regime_distribution"]["1"] == 2
        assert result["regime_distribution"]["2"] == 1


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------

class TestRegimeReport:
    """Tests for the composite regime analysis report."""

    def test_report_structure(self):
        """Report should contain all expected sections."""
        regimes = [0, 0, 1, 1, 2, 0, 0, 1, 2, 2]
        returns = [0.01, 0.005, -0.002, 0.001, -0.01,
                   0.008, 0.003, -0.001, -0.005, -0.008]
        report = regime_report(regimes, returns)
        assert "transition_matrix" in report
        assert "conditional_performance" in report
        assert "persistence" in report
        assert "accuracy" in report

    def test_report_with_benchmark(self):
        """Report with benchmark should compute accuracy against it."""
        regimes = [0, 0, 2, 2]
        returns = [0.01, 0.005, -0.01, -0.005]
        benchmark = [0.008, 0.003, -0.012, -0.004]
        report = regime_report(regimes, returns,
                               benchmark_returns=benchmark)
        assert report["accuracy"]["overall_accuracy"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
