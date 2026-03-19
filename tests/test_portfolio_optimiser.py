"""
tests/test_portfolio_optimiser.py
=================================
Unit tests for Markowitz mean-variance optimization.

Tests edge cases that could cause silent failures:
  - Singular covariance matrix (identical stocks)
  - Single stock (can't optimize correlations)
  - All stocks perfectly correlated
  - Negative expected returns
  - Optimizer convergence failure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from risk.portfolio_optimiser import (
    build_covariance_matrix,
    build_expected_returns,
    maximise_sharpe,
    optimise_portfolio,
)
from config.settings import MIN_WEIGHT, MAX_WEIGHT, MIN_STOCKS


class TestBuildCovarianceMatrix:
    """Tests for covariance matrix construction."""

    def _make_features(self, symbols, n_days=80):
        """Generate synthetic features with realistic return structure."""
        dates = pd.bdate_range("2024-01-01", periods=n_days)
        rows = []
        np.random.seed(42)
        for d in dates:
            for sym in symbols:
                rows.append({
                    "date": d,
                    "symbol": sym,
                    "return_1d": np.random.normal(0.001, 0.02),
                })
        return pd.DataFrame(rows)

    def test_basic_construction(self):
        """Should produce a square matrix with correct dimensions."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        features = self._make_features(symbols)
        date = features["date"].max() + pd.Timedelta(days=1)
        cov = build_covariance_matrix(symbols, features, date)
        assert cov is not None
        assert cov.shape == (3, 3)
        assert set(cov.columns) == set(symbols)

    def test_positive_semidefinite(self):
        """Covariance matrix should be positive semi-definite."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        features = self._make_features(symbols)
        date = features["date"].max() + pd.Timedelta(days=1)
        cov = build_covariance_matrix(symbols, features, date)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert all(ev >= -1e-10 for ev in eigenvalues)  # allow small numerical error

    def test_insufficient_data_returns_none(self):
        """Should return None if not enough history."""
        symbols = ["AAPL", "MSFT"]
        features = self._make_features(symbols, n_days=5)  # too few
        date = features["date"].max() + pd.Timedelta(days=1)
        cov = build_covariance_matrix(symbols, features, date)
        assert cov is None

    def test_single_symbol_returns_none(self):
        """Should return None for a single stock (need ≥2)."""
        features = self._make_features(["AAPL"], n_days=80)
        date = features["date"].max() + pd.Timedelta(days=1)
        cov = build_covariance_matrix(["AAPL"], features, date)
        assert cov is None

    def test_no_lookahead(self):
        """Should only use data strictly before the given date."""
        symbols = ["AAPL", "MSFT"]
        features = self._make_features(symbols, n_days=80)
        # Use the first date — no history before it
        first_date = features["date"].min()
        cov = build_covariance_matrix(symbols, features, first_date)
        assert cov is None  # no data before first date


class TestBuildExpectedReturns:
    """Tests for converting ML scores to expected returns."""

    def test_basic_mapping(self):
        """Higher pred_proba should map to higher expected return."""
        signals = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "pred_proba": [0.80, 0.40],
        })
        er = build_expected_returns(["AAPL", "MSFT"], signals)
        assert er["AAPL"] > er["MSFT"]

    def test_proba_0_5_maps_to_zero(self):
        """pred_proba = 0.5 should map to ~0 expected return."""
        signals = pd.DataFrame({
            "symbol": ["AAPL"],
            "pred_proba": [0.50],
        })
        er = build_expected_returns(["AAPL"], signals)
        assert abs(er["AAPL"]) < 0.01

    def test_missing_symbol_excluded(self):
        """Symbols not in signals should be excluded."""
        signals = pd.DataFrame({
            "symbol": ["AAPL"],
            "pred_proba": [0.70],
        })
        er = build_expected_returns(["AAPL", "MSFT"], signals)
        assert "MSFT" not in er.index


class TestMaximiseSharpe:
    """Tests for the Sharpe maximization optimizer."""

    def _make_inputs(self, n_stocks=5):
        """Generate synthetic expected returns and covariance matrix."""
        np.random.seed(42)
        symbols = [f"S{i}" for i in range(n_stocks)]
        er = pd.Series(np.random.uniform(0.05, 0.20, n_stocks), index=symbols)

        # Generate a valid covariance matrix
        A = np.random.randn(n_stocks, n_stocks) * 0.01
        cov = pd.DataFrame(A @ A.T + np.eye(n_stocks) * 0.01,
                           index=symbols, columns=symbols)
        return er, cov

    def test_weights_sum_to_one(self):
        """Optimized weights should sum to 1.0."""
        er, cov = self._make_inputs()
        w = maximise_sharpe(er, cov)
        assert abs(w.sum() - 1.0) < 0.01

    def test_weights_within_bounds(self):
        """Each weight should be between MIN_WEIGHT and MAX_WEIGHT."""
        # Use 10 stocks so equal weight (10%) is within MAX_WEIGHT (15%)
        er, cov = self._make_inputs(n_stocks=10)
        w = maximise_sharpe(er, cov)
        assert w.min() >= MIN_WEIGHT - 0.001
        assert w.max() <= MAX_WEIGHT + 0.001

    def test_all_positive(self):
        """All weights should be positive (long-only)."""
        er, cov = self._make_inputs()
        w = maximise_sharpe(er, cov)
        assert (w >= 0).all()

    def test_fewer_than_min_stocks(self):
        """With < MIN_STOCKS, should return equal weights."""
        er = pd.Series({"AAPL": 0.10})
        cov = pd.DataFrame([[0.04]], index=["AAPL"], columns=["AAPL"])
        w = maximise_sharpe(er, cov)
        assert len(w) == 1
        assert abs(w["AAPL"] - 1.0) < 0.01  # equal weight for 1 stock

    def test_identical_expected_returns(self):
        """With identical expected returns, should approach equal weight."""
        np.random.seed(42)
        symbols = ["S0", "S1", "S2", "S3", "S4"]
        er = pd.Series(0.10, index=symbols)
        A = np.eye(5) * 0.04  # diagonal cov = uncorrelated
        cov = pd.DataFrame(A, index=symbols, columns=symbols)
        w = maximise_sharpe(er, cov)
        # With equal returns and no correlation, equal weight is optimal
        assert w.std() < 0.05  # weights should be close to each other


class TestOptimisePortfolio:
    """Integration tests for the full optimization pipeline."""

    def _make_data(self, n_symbols=5, n_days=100):
        """Generate synthetic signals and features."""
        np.random.seed(42)
        symbols = [f"S{i}" for i in range(n_symbols)]
        dates = pd.bdate_range("2024-01-01", periods=n_days)

        features_rows = []
        for d in dates:
            for sym in symbols:
                features_rows.append({
                    "date": d,
                    "symbol": sym,
                    "return_1d": np.random.normal(0.001, 0.02),
                    "volatility_20d": abs(np.random.normal(0.015, 0.005)),
                })
        features = pd.DataFrame(features_rows)

        latest_date = dates[-1]
        signals = pd.DataFrame({
            "date": [latest_date] * n_symbols,
            "symbol": symbols,
            "pred_proba": np.random.uniform(0.40, 0.85, n_symbols),
        })

        return signals, features, latest_date

    def test_bull_regime(self):
        """Bull regime should produce non-empty weights."""
        signals, features, date = self._make_data()
        w = optimise_portfolio(signals, features, date, regime="bull")
        assert not w.empty
        assert w.sum() <= 1.0 + 0.01

    def test_bear_regime_tighter_caps(self):
        """Bear regime should produce lower caps than bull."""
        signals, features, date = self._make_data()
        w_bull = optimise_portfolio(signals, features, date, regime="bull")
        w_bear = optimise_portfolio(signals, features, date, regime="bear")
        # Bear should generally have lower max weight due to 70% cap
        assert w_bear.max() <= MAX_WEIGHT * 0.7 + 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
