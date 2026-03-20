"""
Tests for analytics/walk_forward.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.walk_forward import (
    rolling_window_metrics,
    pnl_attribution_by_regime,
    pnl_attribution_by_period,
    performance_stability,
    oos_degradation,
    walk_forward_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns():
    """500 days of synthetic returns (~8% annual, ~15% vol)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0003, 0.01, size=500)


@pytest.fixture
def dated_returns():
    """Returns with dates spanning ~2 years."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=500)
    rets = rng.normal(0.0003, 0.01, size=500)
    return rets, dates


@pytest.fixture
def regime_labels():
    """Regime labels for 500 days."""
    rng = np.random.default_rng(42)
    return rng.choice(["bull", "bear", "choppy"], size=500, p=[0.5, 0.2, 0.3])


# ---------------------------------------------------------------------------
# rolling_window_metrics
# ---------------------------------------------------------------------------

class TestRollingWindowMetrics:
    def test_correct_number_of_windows(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        expected_windows = 500 // 63
        assert len(result) == expected_windows

    def test_window_has_required_keys(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        keys = {"window", "total_return", "sharpe", "max_drawdown",
                "volatility", "win_rate", "num_days"}
        assert set(result[0].keys()) == keys

    def test_window_numbering_starts_at_1(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        assert result[0]["window"] == 1

    def test_each_window_has_correct_days(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        for w in result:
            assert w["num_days"] == 63

    def test_empty_returns(self):
        assert rolling_window_metrics(np.array([]), window_days=63) == []

    def test_too_few_returns(self):
        assert rolling_window_metrics(np.array([0.01, 0.02]), window_days=63) == []

    def test_max_drawdown_non_positive(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        for w in result:
            assert w["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        for w in result:
            assert 0.0 <= w["win_rate"] <= 1.0

    def test_volatility_non_negative(self, daily_returns):
        result = rolling_window_metrics(daily_returns, window_days=63)
        for w in result:
            assert w["volatility"] >= 0

    def test_smaller_window_more_windows(self, daily_returns):
        r1 = rolling_window_metrics(daily_returns, window_days=63)
        r2 = rolling_window_metrics(daily_returns, window_days=21)
        assert len(r2) > len(r1)


# ---------------------------------------------------------------------------
# pnl_attribution_by_regime
# ---------------------------------------------------------------------------

class TestPnlAttributionByRegime:
    def test_all_regimes_present(self, daily_returns, regime_labels):
        result = pnl_attribution_by_regime(daily_returns, regime_labels)
        assert "bull" in result
        assert "bear" in result
        assert "choppy" in result

    def test_required_keys(self, daily_returns, regime_labels):
        result = pnl_attribution_by_regime(daily_returns, regime_labels)
        keys = {"total_return", "mean_daily", "sharpe", "num_days", "pct_of_total_pnl"}
        for regime_data in result.values():
            assert set(regime_data.keys()) == keys

    def test_days_sum_to_total(self, daily_returns, regime_labels):
        result = pnl_attribution_by_regime(daily_returns, regime_labels)
        total_days = sum(r["num_days"] for r in result.values())
        assert total_days == len(daily_returns)

    def test_pct_of_total_sums_to_approximately_one(self, daily_returns, regime_labels):
        result = pnl_attribution_by_regime(daily_returns, regime_labels)
        total_pct = sum(r["pct_of_total_pnl"] for r in result.values())
        assert abs(total_pct - 1.0) < 0.01

    def test_empty_returns(self):
        assert pnl_attribution_by_regime(np.array([]), np.array([])) == {}

    def test_single_regime(self, daily_returns):
        regimes = np.full(len(daily_returns), "bull")
        result = pnl_attribution_by_regime(daily_returns, regimes)
        assert len(result) == 1
        assert "bull" in result
        assert result["bull"]["num_days"] == len(daily_returns)


# ---------------------------------------------------------------------------
# pnl_attribution_by_period
# ---------------------------------------------------------------------------

class TestPnlAttributionByPeriod:
    def test_quarterly_periods(self, dated_returns):
        rets, dates = dated_returns
        result = pnl_attribution_by_period(rets, dates, period="quarter")
        assert len(result) > 0
        # ~500 days / ~63 per quarter = ~8 quarters
        assert len(result) >= 7

    def test_monthly_periods(self, dated_returns):
        rets, dates = dated_returns
        result = pnl_attribution_by_period(rets, dates, period="month")
        assert len(result) >= 20  # ~500 days / 21 per month

    def test_yearly_periods(self, dated_returns):
        rets, dates = dated_returns
        result = pnl_attribution_by_period(rets, dates, period="year")
        assert len(result) >= 2  # spans 2022-2023

    def test_required_keys(self, dated_returns):
        rets, dates = dated_returns
        result = pnl_attribution_by_period(rets, dates)
        keys = {"period", "total_return", "sharpe", "num_days", "cumulative_return"}
        assert set(result[0].keys()) == keys

    def test_cumulative_return_monotonic_check(self, dated_returns):
        """Cumulative return should be computed correctly."""
        rets, dates = dated_returns
        result = pnl_attribution_by_period(rets, dates)
        # Last period's cumulative should equal total return
        total = float(np.prod(1 + rets) - 1)
        last_cumulative = result[-1]["cumulative_return"]
        assert abs(last_cumulative - total) < 0.01

    def test_empty_returns(self):
        assert pnl_attribution_by_period(np.array([]), np.array([])) == []


# ---------------------------------------------------------------------------
# performance_stability
# ---------------------------------------------------------------------------

class TestPerformanceStability:
    def test_returns_required_keys(self, daily_returns):
        result = performance_stability(daily_returns)
        keys = {"pct_profitable_windows", "sharpe_std", "sharpe_mean",
                "worst_window_return", "best_window_return",
                "return_consistency", "n_windows"}
        assert set(result.keys()) == keys

    def test_pct_profitable_between_0_and_1(self, daily_returns):
        result = performance_stability(daily_returns)
        assert 0.0 <= result["pct_profitable_windows"] <= 1.0

    def test_worst_worse_than_best(self, daily_returns):
        result = performance_stability(daily_returns)
        assert result["worst_window_return"] <= result["best_window_return"]

    def test_n_windows_matches(self, daily_returns):
        result = performance_stability(daily_returns, window_days=63)
        expected = 500 // 63
        assert result["n_windows"] == expected

    def test_empty_returns(self):
        result = performance_stability(np.array([]))
        assert result["n_windows"] == 0
        assert result["pct_profitable_windows"] == 0.0

    def test_all_positive_returns(self):
        rets = np.full(126, 0.001)  # consistently positive
        result = performance_stability(rets, window_days=63)
        assert result["pct_profitable_windows"] == 1.0

    def test_all_negative_returns(self):
        rets = np.full(126, -0.001)  # consistently negative
        result = performance_stability(rets, window_days=63)
        assert result["pct_profitable_windows"] == 0.0


# ---------------------------------------------------------------------------
# oos_degradation
# ---------------------------------------------------------------------------

class TestOosDegradation:
    def test_returns_required_keys(self, daily_returns):
        result = oos_degradation(daily_returns)
        keys = {"first_half_sharpe", "second_half_sharpe", "sharpe_change",
                "first_half_return", "second_half_return", "degradation_detected"}
        assert set(result.keys()) == keys

    def test_sharpe_change_is_difference(self, daily_returns):
        result = oos_degradation(daily_returns)
        expected = result["second_half_sharpe"] - result["first_half_sharpe"]
        assert abs(result["sharpe_change"] - expected) < 1e-10

    def test_degradation_detected_for_declining_returns(self):
        """First half positive, second half negative → degradation."""
        first_half = np.full(250, 0.002)
        second_half = np.full(250, -0.002)
        rets = np.concatenate([first_half, second_half])
        result = oos_degradation(rets)
        assert result["degradation_detected"] is True
        assert result["sharpe_change"] < 0

    def test_no_degradation_for_consistent_returns(self, daily_returns):
        result = oos_degradation(daily_returns)
        # Random returns should not show strong degradation
        assert isinstance(result["degradation_detected"], bool)

    def test_empty_returns(self):
        result = oos_degradation(np.array([]))
        assert result["degradation_detected"] is False
        assert result["first_half_sharpe"] == 0.0

    def test_custom_split(self, daily_returns):
        result = oos_degradation(daily_returns, split_pct=0.7)
        assert isinstance(result["sharpe_change"], float)


# ---------------------------------------------------------------------------
# walk_forward_report (composite)
# ---------------------------------------------------------------------------

class TestWalkForwardReport:
    def test_all_sections_present(self, daily_returns, regime_labels, dated_returns):
        rets, dates = dated_returns
        result = walk_forward_report(
            rets, dates=dates, regimes=regime_labels[:len(rets)]
        )
        assert "rolling_windows" in result
        assert "stability" in result
        assert "degradation" in result
        assert "regime_attribution" in result
        assert "quarterly_attribution" in result

    def test_minimal_call(self, daily_returns):
        result = walk_forward_report(daily_returns)
        assert "rolling_windows" in result
        assert result["regime_attribution"] == {}
        assert result["quarterly_attribution"] == []

    def test_custom_window(self, daily_returns):
        result = walk_forward_report(daily_returns, window_days=21)
        # 500 / 21 = 23 windows
        assert len(result["rolling_windows"]) == 500 // 21
