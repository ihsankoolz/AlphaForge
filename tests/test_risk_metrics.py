"""
tests/test_risk_metrics.py
==========================
Unit tests for analytics/risk_metrics.py — VaR, CVaR, concentration,
Calmar, beta, skewness, kurtosis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from analytics.risk_metrics import (
    compute_var, compute_cvar, herfindahl_index,
    effective_number_of_bets, calmar_ratio, portfolio_beta,
    return_skewness, return_kurtosis, max_drawdown, risk_summary,
)


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------

class TestComputeVar:
    """Tests for historical Value at Risk."""

    def test_basic_var(self):
        """VaR at 95% should be the 5th percentile of returns."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        var = compute_var(returns, 0.95)
        assert var < 0  # VaR should be negative (a loss)
        # Should be approximately -0.032 for these params
        assert -0.06 < var < -0.01

    def test_empty_returns_zero(self):
        """Empty returns should give VaR = 0."""
        assert compute_var([], 0.95) == 0.0

    def test_all_positive_var(self):
        """If all returns are positive, VaR is still positive (no loss)."""
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        var = compute_var(returns, 0.95)
        assert var > 0

    def test_nan_handling(self):
        """NaN values should be filtered out."""
        returns = [0.01, -0.02, np.nan, 0.03, -0.01]
        var = compute_var(returns, 0.95)
        assert not np.isnan(var)

    def test_higher_confidence_more_extreme(self):
        """99% VaR should be more extreme (more negative) than 95% VaR."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 1000)
        var_95 = compute_var(returns, 0.95)
        var_99 = compute_var(returns, 0.99)
        assert var_99 < var_95


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------

class TestComputeCvar:
    """Tests for Conditional Value at Risk (Expected Shortfall)."""

    def test_cvar_worse_than_var(self):
        """CVaR should be at least as extreme as VaR."""
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 1000)
        var = compute_var(returns, 0.95)
        cvar = compute_cvar(returns, 0.95)
        assert cvar <= var

    def test_empty_returns_zero(self):
        """Empty returns should give CVaR = 0."""
        assert compute_cvar([], 0.95) == 0.0

    def test_single_return(self):
        """Single return: CVaR = that return."""
        cvar = compute_cvar([-0.05], 0.95)
        assert cvar == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

class TestHerfindahlIndex:
    """Tests for portfolio concentration."""

    def test_equal_weight(self):
        """N equal-weight positions: HHI = 1/N."""
        weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        hhi = herfindahl_index(weights)
        assert hhi == pytest.approx(0.25, abs=1e-10)

    def test_single_stock(self):
        """Single stock: HHI = 1.0."""
        hhi = herfindahl_index({"AAPL": 1.0})
        assert hhi == pytest.approx(1.0)

    def test_concentrated(self):
        """Concentrated portfolio has higher HHI."""
        equal = herfindahl_index({"A": 0.5, "B": 0.5})
        concentrated = herfindahl_index({"A": 0.9, "B": 0.1})
        assert concentrated > equal

    def test_empty_portfolio(self):
        """Empty portfolio: HHI = 0."""
        assert herfindahl_index({}) == 0.0

    def test_array_input(self):
        """Should accept array input as well as dict."""
        hhi = herfindahl_index([0.5, 0.5])
        assert hhi == pytest.approx(0.5)


class TestEffectiveNumberOfBets:
    """Tests for 1/HHI metric."""

    def test_equal_weight_n(self):
        """5 equal-weight positions → ENB = 5."""
        weights = {f"S{i}": 0.2 for i in range(5)}
        enb = effective_number_of_bets(weights)
        assert enb == pytest.approx(5.0, abs=0.01)

    def test_single_stock(self):
        """Single stock → ENB = 1."""
        enb = effective_number_of_bets({"AAPL": 1.0})
        assert enb == pytest.approx(1.0)

    def test_empty(self):
        """Empty portfolio → ENB = 0."""
        assert effective_number_of_bets({}) == 0.0

    def test_concentrated_fewer_bets(self):
        """A concentrated portfolio has fewer effective bets."""
        equal_enb = effective_number_of_bets({"A": 0.5, "B": 0.5})
        conc_enb = effective_number_of_bets({"A": 0.9, "B": 0.1})
        assert conc_enb < equal_enb


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    """Tests for annualised return / max drawdown."""

    def test_positive_with_dips(self):
        """Returns with some dips → positive Calmar (has drawdowns)."""
        np.random.seed(42)
        returns = np.random.normal(0.002, 0.01, 252)  # positive drift + noise
        calmar = calmar_ratio(returns)
        assert calmar > 0

    def test_empty_returns(self):
        """Empty returns → Calmar = 0."""
        assert calmar_ratio([]) == 0.0

    def test_no_drawdown_zero(self):
        """Monotonically increasing returns → no drawdown → Calmar = 0."""
        # All positive returns of same magnitude → running_max always equals
        # cumulative → drawdown = 0 → calmar = 0
        returns = [0.01] * 100
        calmar = calmar_ratio(returns)
        assert calmar == 0.0


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class TestPortfolioBeta:
    """Tests for portfolio beta to benchmark."""

    def test_perfect_correlation(self):
        """Portfolio identical to benchmark → beta ≈ 1."""
        np.random.seed(42)
        benchmark = np.random.normal(0.001, 0.01, 252)
        beta = portfolio_beta(benchmark, benchmark)
        assert beta == pytest.approx(1.0, abs=0.01)

    def test_double_exposure(self):
        """Portfolio = 2× benchmark → beta ≈ 2."""
        np.random.seed(42)
        benchmark = np.random.normal(0.001, 0.01, 252)
        portfolio = benchmark * 2
        beta = portfolio_beta(portfolio, benchmark)
        assert beta == pytest.approx(2.0, abs=0.01)

    def test_zero_variance_benchmark(self):
        """Constant benchmark → beta = 0 (can't compute)."""
        portfolio = [0.01, 0.02, -0.01, 0.005]
        benchmark = [0.0, 0.0, 0.0, 0.0]
        assert portfolio_beta(portfolio, benchmark) == 0.0

    def test_empty_inputs(self):
        """Empty inputs → beta = 0."""
        assert portfolio_beta([], []) == 0.0

    def test_uncorrelated(self):
        """Uncorrelated returns → beta ≈ 0."""
        np.random.seed(42)
        portfolio = np.random.normal(0, 0.01, 1000)
        np.random.seed(99)
        benchmark = np.random.normal(0, 0.01, 1000)
        beta = portfolio_beta(portfolio, benchmark)
        assert abs(beta) < 0.15  # should be close to 0


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------

class TestReturnSkewness:
    """Tests for skewness calculation."""

    def test_symmetric_near_zero(self):
        """Large symmetric sample → skewness ≈ 0."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 10000)
        skew = return_skewness(returns)
        assert abs(skew) < 0.1

    def test_right_skewed(self):
        """Right-skewed data → positive skewness."""
        # Exponential distribution is right-skewed
        np.random.seed(42)
        returns = np.random.exponential(0.01, 1000)
        skew = return_skewness(returns)
        assert skew > 0

    def test_too_few_observations(self):
        """Fewer than 3 observations → skewness = 0."""
        assert return_skewness([0.01, -0.01]) == 0.0
        assert return_skewness([]) == 0.0


class TestReturnKurtosis:
    """Tests for excess kurtosis calculation."""

    def test_normal_near_zero(self):
        """Large normal sample → excess kurtosis ≈ 0."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 50000)
        kurt = return_kurtosis(returns)
        assert abs(kurt) < 0.15

    def test_fat_tails_positive(self):
        """T-distribution with few df has fat tails → positive kurtosis."""
        np.random.seed(42)
        returns = np.random.standard_t(df=3, size=10000) * 0.01
        kurt = return_kurtosis(returns)
        assert kurt > 1.0  # heavy tails

    def test_too_few_observations(self):
        """Fewer than 4 observations → kurtosis = 0."""
        assert return_kurtosis([0.01, -0.01, 0.005]) == 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    """Tests for max drawdown utility."""

    def test_known_drawdown(self):
        """A known 10% drop should produce ~10% drawdown."""
        # Go up 10%, then drop 10% from peak
        returns = [0.10, -0.10]
        dd = max_drawdown(returns)
        # After +10%: cumulative = 1.1, After -10%: cumulative = 0.99
        # Drawdown = (0.99 - 1.1) / 1.1 = -0.10
        assert dd == pytest.approx(-0.10, abs=0.01)

    def test_no_drawdown(self):
        """All positive returns → drawdown = 0."""
        dd = max_drawdown([0.01, 0.01, 0.01])
        assert dd == 0.0

    def test_empty(self):
        """Empty returns → drawdown = 0."""
        assert max_drawdown([]) == 0.0


# ---------------------------------------------------------------------------
# Composite summary
# ---------------------------------------------------------------------------

class TestRiskSummary:
    """Tests for the risk_summary composite function."""

    def test_includes_all_base_metrics(self):
        """Summary should have all base metrics."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        summary = risk_summary(returns)
        assert "var_95" in summary
        assert "cvar_95" in summary
        assert "var_99" in summary
        assert "cvar_99" in summary
        assert "max_drawdown" in summary
        assert "calmar_ratio" in summary
        assert "skewness" in summary
        assert "kurtosis" in summary

    def test_includes_concentration_when_weights(self):
        """Summary with weights should have HHI and effective bets."""
        returns = [0.01, -0.01, 0.005]
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        summary = risk_summary(returns, weights=weights)
        assert "herfindahl" in summary
        assert "effective_bets" in summary

    def test_includes_beta_when_benchmark(self):
        """Summary with benchmark should have beta."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100)
        benchmark = np.random.normal(0, 0.01, 100)
        summary = risk_summary(returns, benchmark_returns=benchmark)
        assert "beta" in summary

    def test_no_concentration_without_weights(self):
        """Summary without weights should NOT have HHI."""
        summary = risk_summary([0.01, -0.01])
        assert "herfindahl" not in summary
        assert "effective_bets" not in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
