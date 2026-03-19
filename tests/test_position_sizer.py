"""
tests/test_position_sizer.py
============================
Unit tests for the Kelly Criterion position sizer.

Tests edge cases that could cause silent bugs in production:
  - Zero/NaN volatility (division by zero)
  - Probability at exact thresholds (boundary conditions)
  - All probabilities below threshold (empty portfolio)
  - Single stock (degenerate case)
  - Volatility = 0 (would cause infinite scaling)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from risk.position_sizer import (
    kelly_weights,
    volatility_adjust,
    normalise_weights,
    compute_positions,
    calibrate_kelly_odds,
)
from config.settings import MIN_PROB, MAX_POSITION, KELLY_FRACTION, KELLY_ODDS


class TestKellyWeights:
    """Tests for the core Kelly sizing formula."""

    def test_basic_sizing(self):
        """High probability should produce positive weight."""
        proba = pd.Series({"AAPL": 0.70, "MSFT": 0.50})
        w = kelly_weights(proba)
        assert w["AAPL"] > 0
        assert w["AAPL"] > w.get("MSFT", 0)  # higher prob → larger weight

    def test_below_threshold_excluded(self):
        """Stocks below MIN_PROB should not appear in output."""
        proba = pd.Series({"AAPL": 0.30, "MSFT": 0.20})
        w = kelly_weights(proba)
        assert w.empty

    def test_at_exact_threshold(self):
        """Stocks at exactly MIN_PROB should be excluded (> not >=)."""
        proba = pd.Series({"AAPL": MIN_PROB})
        w = kelly_weights(proba)
        assert w.empty

    def test_just_above_threshold(self):
        """Stock barely above threshold should have a small positive weight."""
        proba = pd.Series({"AAPL": MIN_PROB + 0.01})
        w = kelly_weights(proba)
        assert len(w) == 1
        assert w["AAPL"] > 0
        assert w["AAPL"] < 0.1  # should be small

    def test_probability_one(self):
        """Probability = 1.0 should produce the maximum Kelly fraction."""
        proba = pd.Series({"AAPL": 1.0})
        w = kelly_weights(proba)
        # Kelly = (1.0 * b - 0) / b = 1.0, then * KELLY_FRACTION
        expected = KELLY_FRACTION
        assert abs(w["AAPL"] - expected) < 0.001

    def test_empty_input(self):
        """Empty input should return empty output."""
        proba = pd.Series(dtype=float)
        w = kelly_weights(proba)
        assert w.empty

    def test_all_weights_non_negative(self):
        """Kelly weights should never be negative (long-only)."""
        proba = pd.Series({f"S{i}": 0.35 + i * 0.05 for i in range(10)})
        w = kelly_weights(proba)
        assert (w >= 0).all()

    def test_half_kelly_applied(self):
        """Output should be exactly half of full Kelly."""
        proba = pd.Series({"AAPL": 0.70})
        b = KELLY_ODDS
        p, q = 0.70, 0.30
        full_kelly = (p * b - q) / b
        expected = full_kelly * KELLY_FRACTION
        w = kelly_weights(proba)
        assert abs(w["AAPL"] - expected) < 0.001


class TestVolatilityAdjust:
    """Tests for volatility targeting on top of Kelly weights."""

    def _make_features(self, symbols, vol_values, date):
        """Helper to create a minimal features DataFrame."""
        rows = []
        for sym, vol in zip(symbols, vol_values):
            rows.append({
                "date": date - pd.Timedelta(days=1),  # strictly before date
                "symbol": sym,
                "volatility_20d": vol,
            })
        return pd.DataFrame(rows)

    def test_high_vol_reduces_weight(self):
        """A stock with 2x target vol should get roughly half the weight."""
        date = pd.Timestamp("2024-01-10")
        kelly_w = pd.Series({"AAPL": 0.10})
        features = self._make_features(["AAPL"], [0.02], date)  # 2% vs 1% target
        adjusted = volatility_adjust(kelly_w, features, date)
        assert adjusted["AAPL"] < kelly_w["AAPL"]
        assert abs(adjusted["AAPL"] - 0.05) < 0.01  # ~half

    def test_low_vol_increases_weight(self):
        """A stock with 0.5x target vol should get roughly double the weight."""
        date = pd.Timestamp("2024-01-10")
        kelly_w = pd.Series({"AAPL": 0.05})
        features = self._make_features(["AAPL"], [0.005], date)  # 0.5% vs 1% target
        adjusted = volatility_adjust(kelly_w, features, date)
        assert adjusted["AAPL"] > kelly_w["AAPL"]

    def test_zero_vol_unchanged(self):
        """Zero volatility should leave weight unchanged (not infinite)."""
        date = pd.Timestamp("2024-01-10")
        kelly_w = pd.Series({"AAPL": 0.10})
        features = self._make_features(["AAPL"], [0.0], date)
        adjusted = volatility_adjust(kelly_w, features, date)
        assert adjusted["AAPL"] == kelly_w["AAPL"]

    def test_nan_vol_unchanged(self):
        """NaN volatility should leave weight unchanged."""
        date = pd.Timestamp("2024-01-10")
        kelly_w = pd.Series({"AAPL": 0.10})
        features = self._make_features(["AAPL"], [np.nan], date)
        adjusted = volatility_adjust(kelly_w, features, date)
        assert adjusted["AAPL"] == kelly_w["AAPL"]

    def test_no_lookahead(self):
        """Volatility should use data strictly BEFORE the trade date."""
        date = pd.Timestamp("2024-01-10")
        kelly_w = pd.Series({"AAPL": 0.10})
        # Only data on the trade date itself — should have no history
        features = pd.DataFrame([{
            "date": date,  # same day, not before
            "symbol": "AAPL",
            "volatility_20d": 0.02,
        }])
        adjusted = volatility_adjust(kelly_w, features, date)
        # No data before date → falls back to unadjusted weight
        assert adjusted["AAPL"] == kelly_w["AAPL"]

    def test_empty_kelly_returns_empty(self):
        """Empty Kelly weights should return empty."""
        date = pd.Timestamp("2024-01-10")
        kelly_w = pd.Series(dtype=float)
        features = pd.DataFrame()
        adjusted = volatility_adjust(kelly_w, features, date)
        assert adjusted.empty


class TestNormaliseWeights:
    """Tests for weight capping and normalisation."""

    def test_cap_applied(self):
        """No weight should exceed MAX_POSITION."""
        weights = pd.Series({"AAPL": 0.50, "MSFT": 0.30})
        normed = normalise_weights(weights)
        assert normed.max() <= MAX_POSITION + 0.001

    def test_max_positions_respected(self):
        """Should keep at most max_positions stocks."""
        weights = pd.Series({f"S{i}": 0.05 for i in range(20)})
        normed = normalise_weights(weights, max_positions=5)
        assert len(normed) <= 5

    def test_sum_at_most_one(self):
        """Weights should sum to at most 1.0."""
        weights = pd.Series({"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.30,
                             "NVDA": 0.30})
        normed = normalise_weights(weights)
        assert normed.sum() <= 1.001

    def test_low_confidence_keeps_cash(self):
        """Small weights that sum to < 1.0 should NOT be inflated to 1.0."""
        weights = pd.Series({"AAPL": 0.05, "MSFT": 0.05})
        normed = normalise_weights(weights)
        assert normed.sum() < 0.5  # should stay small, preserving cash signal

    def test_empty_input(self):
        """Empty input should return empty."""
        normed = normalise_weights(pd.Series(dtype=float))
        assert normed.empty


class TestComputePositions:
    """Integration tests for the full sizing pipeline."""

    def test_full_pipeline_basic(self):
        """Full pipeline should produce sensible weights."""
        date = pd.Timestamp("2024-01-10")
        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "pred_proba": [0.80, 0.60, 0.25],  # GOOGL below threshold
        })
        features = pd.DataFrame([
            {"date": date - pd.Timedelta(days=1), "symbol": "AAPL", "volatility_20d": 0.015},
            {"date": date - pd.Timedelta(days=1), "symbol": "MSFT", "volatility_20d": 0.012},
            {"date": date - pd.Timedelta(days=1), "symbol": "GOOGL", "volatility_20d": 0.010},
        ])

        weights = compute_positions(signals, features, date)
        assert "AAPL" in weights.index
        assert "MSFT" in weights.index
        assert "GOOGL" not in weights.index  # below threshold
        assert weights.sum() <= 1.0

    def test_empty_signals(self):
        """Empty signals should produce empty weights."""
        date = pd.Timestamp("2024-01-10")
        signals = pd.DataFrame(columns=["symbol", "pred_proba"])
        features = pd.DataFrame()
        weights = compute_positions(signals, features, date)
        assert weights.empty

    def test_all_below_threshold(self):
        """All stocks below threshold should produce empty weights."""
        date = pd.Timestamp("2024-01-10")
        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "pred_proba": [0.20, 0.30],
        })
        features = pd.DataFrame()
        weights = compute_positions(signals, features, date)
        assert weights.empty


class TestCalibrateKellyOdds:
    """Tests for empirical odds calibration."""

    def test_returns_float(self):
        """Should return a float odds ratio."""
        signals = pd.DataFrame({
            "pred_proba": [0.50] * 200,
            "forward_return_5d": [0.03, -0.02] * 100,
        })
        odds = calibrate_kelly_odds(signals)
        assert isinstance(odds, float)
        assert odds > 0

    def test_bounded(self):
        """Odds should be clamped between 0.5 and 5.0."""
        # Extreme case: all winners, no losers
        signals = pd.DataFrame({
            "pred_proba": [0.50] * 200,
            "forward_return_5d": [0.05] * 200,
        })
        odds = calibrate_kelly_odds(signals)
        assert 0.5 <= odds <= 5.0

    def test_insufficient_data_fallback(self):
        """Should fall back to default with <100 rows above threshold."""
        signals = pd.DataFrame({
            "pred_proba": [0.50] * 10,
            "forward_return_5d": [0.03] * 10,
        })
        odds = calibrate_kelly_odds(signals)
        assert odds == KELLY_ODDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
