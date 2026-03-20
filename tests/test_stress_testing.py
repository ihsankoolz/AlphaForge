"""
Tests for analytics/stress_testing.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.stress_testing import (
    historical_crisis_replay,
    parametric_stress,
    monte_carlo_var,
    correlation_stress,
    sector_concentration_stress,
    stress_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns():
    """250 days of synthetic returns (~8% annual, ~15% vol)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0003, 0.01, size=250)


@pytest.fixture
def dated_returns():
    """Returns with a DatetimeIndex spanning 2020-2023."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=800)
    rets = rng.normal(0.0003, 0.01, size=800)
    # Inject a drawdown during COVID window
    covid_mask = (dates >= "2020-02-19") & (dates <= "2020-03-23")
    rets[covid_mask] = rng.normal(-0.02, 0.03, size=covid_mask.sum())
    return pd.Series(rets, index=dates)


@pytest.fixture
def sample_weights():
    return {"AAPL": 0.15, "MSFT": 0.12, "JPM": 0.10, "XOM": 0.08, "JNJ": 0.05}


@pytest.fixture
def sample_vols():
    return {"AAPL": 0.25, "MSFT": 0.22, "JPM": 0.30, "XOM": 0.35, "JNJ": 0.18}


@pytest.fixture
def sample_sector_map():
    return {
        "AAPL": "Technology", "MSFT": "Technology",
        "JPM": "Financials", "XOM": "Energy", "JNJ": "Healthcare",
    }


# ---------------------------------------------------------------------------
# historical_crisis_replay
# ---------------------------------------------------------------------------

class TestHistoricalCrisisReplay:
    def test_returns_crisis_stats_for_matching_dates(self, dated_returns):
        result = historical_crisis_replay(dated_returns)
        assert "COVID crash (Feb-Mar 2020)" in result
        covid = result["COVID crash (Feb-Mar 2020)"]
        assert covid["trading_days"] > 0
        assert covid["total_return"] < 0  # we injected negative returns
        assert covid["max_drawdown"] <= 0
        assert covid["worst_day"] < 0

    def test_empty_returns_give_empty_result(self):
        assert historical_crisis_replay(np.array([])) == {}

    def test_plain_array_with_dates(self):
        dates = pd.bdate_range("2020-02-01", periods=60)
        rets = np.full(60, -0.01)
        result = historical_crisis_replay(rets, dates=dates)
        assert "COVID crash (Feb-Mar 2020)" in result

    def test_no_overlap_gives_empty(self):
        dates = pd.bdate_range("2018-01-01", periods=100)
        rets = np.zeros(100)
        result = historical_crisis_replay(rets, dates=dates)
        # No crisis windows overlap with 2018
        assert len(result) == 0

    def test_best_day_present(self, dated_returns):
        result = historical_crisis_replay(dated_returns)
        for crisis_data in result.values():
            assert "best_day" in crisis_data


# ---------------------------------------------------------------------------
# parametric_stress
# ---------------------------------------------------------------------------

class TestParametricStress:
    def test_default_scenarios_present(self, daily_returns):
        result = parametric_stress(daily_returns)
        assert "base (1x)" in result
        assert "severe stress (2x)" in result
        assert len(result) == 5

    def test_2x_stress_doubles_volatility(self, daily_returns):
        result = parametric_stress(daily_returns)
        base_vol = result["base (1x)"]["volatility"]
        stress_vol = result["severe stress (2x)"]["volatility"]
        assert abs(stress_vol / base_vol - 2.0) < 0.01

    def test_custom_shocks(self, daily_returns):
        result = parametric_stress(daily_returns, shocks={"test": 1.5})
        assert "test" in result
        assert result["test"]["multiplier"] == 1.5

    def test_empty_returns(self):
        assert parametric_stress(np.array([])) == {}

    def test_stress_worsens_var(self, daily_returns):
        result = parametric_stress(daily_returns)
        base_var = result["base (1x)"]["var_95"]
        stress_var = result["severe stress (2x)"]["var_95"]
        assert stress_var < base_var  # more negative = worse

    def test_max_drawdown_worsens_under_stress(self, daily_returns):
        result = parametric_stress(daily_returns)
        base_dd = result["base (1x)"]["max_drawdown"]
        stress_dd = result["extreme stress (3x)"]["max_drawdown"]
        assert stress_dd <= base_dd


# ---------------------------------------------------------------------------
# monte_carlo_var
# ---------------------------------------------------------------------------

class TestMonteCarloVar:
    def test_returns_all_keys(self, daily_returns):
        result = monte_carlo_var(daily_returns)
        expected_keys = {
            "mc_var", "mc_cvar", "mean_terminal", "median_terminal",
            "worst_path", "best_path", "pct_negative",
        }
        assert set(result.keys()) == expected_keys

    def test_var_is_negative_for_reasonable_returns(self, daily_returns):
        result = monte_carlo_var(daily_returns)
        assert result["mc_var"] < 0

    def test_cvar_worse_than_var(self, daily_returns):
        result = monte_carlo_var(daily_returns)
        assert result["mc_cvar"] <= result["mc_var"]

    def test_worst_path_worse_than_var(self, daily_returns):
        result = monte_carlo_var(daily_returns)
        assert result["worst_path"] <= result["mc_var"]

    def test_best_path_positive(self, daily_returns):
        result = monte_carlo_var(daily_returns)
        assert result["best_path"] > 0

    def test_reproducible_with_same_seed(self, daily_returns):
        r1 = monte_carlo_var(daily_returns, seed=123)
        r2 = monte_carlo_var(daily_returns, seed=123)
        assert r1["mc_var"] == r2["mc_var"]

    def test_different_seeds_give_different_results(self, daily_returns):
        r1 = monte_carlo_var(daily_returns, seed=1)
        r2 = monte_carlo_var(daily_returns, seed=2)
        assert r1["mc_var"] != r2["mc_var"]

    def test_insufficient_data(self):
        result = monte_carlo_var(np.array([0.01, 0.02]))
        assert result["mc_var"] == 0.0

    def test_longer_horizon_wider_distribution(self, daily_returns):
        r1 = monte_carlo_var(daily_returns, horizon_days=1)
        r5 = monte_carlo_var(daily_returns, horizon_days=5)
        # 5-day horizon should have more extreme worst path
        assert r5["worst_path"] < r1["worst_path"]

    def test_pct_negative_between_0_and_1(self, daily_returns):
        result = monte_carlo_var(daily_returns)
        assert 0.0 <= result["pct_negative"] <= 1.0


# ---------------------------------------------------------------------------
# correlation_stress
# ---------------------------------------------------------------------------

class TestCorrelationStress:
    def test_higher_correlation_increases_vol(self, sample_weights, sample_vols):
        result = correlation_stress(sample_weights, sample_vols)
        assert result[0.3]["portfolio_vol"] < result[0.9]["portfolio_vol"]

    def test_perfect_correlation_highest_vol(self, sample_weights, sample_vols):
        result = correlation_stress(sample_weights, sample_vols)
        assert result[1.0]["portfolio_vol"] >= result[0.9]["portfolio_vol"]

    def test_vol_increase_pct_positive_above_base(self, sample_weights, sample_vols):
        result = correlation_stress(
            sample_weights, sample_vols, base_correlation=0.3
        )
        assert result[0.9]["vol_increase_pct"] > 0

    def test_custom_stress_levels(self, sample_weights, sample_vols):
        result = correlation_stress(
            sample_weights, sample_vols,
            stress_correlations=[0.0, 0.5, 1.0],
        )
        assert len(result) == 3
        assert 0.0 in result

    def test_empty_weights(self):
        result = correlation_stress({}, {})
        assert result == {}

    def test_single_asset(self):
        result = correlation_stress(
            {"AAPL": 1.0}, {"AAPL": 0.25},
            stress_correlations=[0.5, 1.0],
        )
        # Single asset: correlation doesn't matter, vol = weight * asset_vol
        vol_50 = result[0.5]["portfolio_vol"]
        vol_100 = result[1.0]["portfolio_vol"]
        assert abs(vol_50 - vol_100) < 1e-10

    def test_array_inputs(self):
        result = correlation_stress(
            np.array([0.5, 0.5]),
            np.array([0.20, 0.20]),
            stress_correlations=[0.0, 1.0],
        )
        # At corr=0: vol = sqrt(0.5^2*0.2^2 + 0.5^2*0.2^2) = 0.1*sqrt(2) ≈ 0.1414
        # At corr=1: vol = 0.5*0.2 + 0.5*0.2 = 0.2
        assert abs(result[0.0]["portfolio_vol"] - 0.1 * np.sqrt(2)) < 1e-6
        assert abs(result[1.0]["portfolio_vol"] - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# sector_concentration_stress
# ---------------------------------------------------------------------------

class TestSectorConcentrationStress:
    def test_default_scenarios(self, sample_weights, sample_sector_map):
        result = sector_concentration_stress(sample_weights, sample_sector_map)
        assert "Tech sector -20%" in result
        tech = result["Tech sector -20%"]
        # AAPL(0.15) + MSFT(0.12) = 0.27 tech weight
        assert abs(tech["affected_weight"] - 0.27) < 1e-10
        # Impact = 0.27 * -0.20 = -0.054
        assert abs(tech["portfolio_impact"] - (-0.054)) < 1e-10

    def test_energy_shock(self, sample_weights, sample_sector_map):
        result = sector_concentration_stress(sample_weights, sample_sector_map)
        energy = result["Energy -25%"]
        assert abs(energy["affected_weight"] - 0.08) < 1e-10

    def test_empty_weights(self):
        assert sector_concentration_stress({}, {"AAPL": "Tech"}) == {}

    def test_empty_sector_map(self):
        assert sector_concentration_stress({"AAPL": 0.5}, {}) == {}

    def test_custom_shocks(self, sample_weights, sample_sector_map):
        result = sector_concentration_stress(
            sample_weights, sample_sector_map,
            sector_shocks={"Custom": ("Technology", -0.50)},
        )
        assert "Custom" in result
        assert result["Custom"]["shock"] == -0.50

    def test_unmatched_sector_zero_impact(self, sample_weights, sample_sector_map):
        result = sector_concentration_stress(
            sample_weights, sample_sector_map,
            sector_shocks={"Utilities crash": ("Utilities", -0.30)},
        )
        assert result["Utilities crash"]["affected_weight"] == 0.0
        assert result["Utilities crash"]["portfolio_impact"] == 0.0


# ---------------------------------------------------------------------------
# stress_report (composite)
# ---------------------------------------------------------------------------

class TestStressReport:
    def test_all_sections_present(self, dated_returns, sample_weights, sample_vols, sample_sector_map):
        result = stress_report(
            dated_returns,
            weights=sample_weights,
            individual_vols=sample_vols,
            sector_map=sample_sector_map,
        )
        assert "historical_crises" in result
        assert "parametric" in result
        assert "monte_carlo" in result
        assert "correlation" in result
        assert "sector" in result

    def test_minimal_call_no_weights(self, daily_returns):
        result = stress_report(daily_returns)
        assert "parametric" in result
        assert result["correlation"] == {}
        assert result["sector"] == {}

    def test_historical_crises_populated_with_dates(self, dated_returns):
        result = stress_report(dated_returns)
        assert len(result["historical_crises"]) > 0
