"""
analytics/stress_testing.py
============================
Stress testing and scenario analysis for portfolio risk assessment.

Provides:
    - Historical crisis replay — how would the portfolio perform during
      known crisis periods (2008 GFC, 2020 COVID, 2022 rate hikes)?
    - Parametric stress scenarios — shock returns/vol/correlation by
      user-defined multipliers
    - Monte Carlo VaR — simulate future returns from fitted distribution
    - Correlation stress — what happens when all correlations go to 1?
    - Sector concentration stress — test exposure to sector drawdowns

All functions are pure — they accept arrays/DataFrames and return dicts.
No I/O, no side effects.

Usage:
    from analytics.stress_testing import (
        historical_crisis_replay,
        parametric_stress,
        monte_carlo_var,
        correlation_stress,
        sector_concentration_stress,
        stress_report,
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Historical crisis periods (approximate trading-day date ranges)
# ---------------------------------------------------------------------------

CRISIS_PERIODS = {
    "COVID crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate hike selloff (Jan-Jun 2022)": ("2022-01-03", "2022-06-16"),
    "2022 Oct bear low (Aug-Oct 2022)": ("2022-08-16", "2022-10-12"),
    "SVB banking crisis (Mar 2023)": ("2023-03-08", "2023-03-15"),
    "Aug 2024 unwind (Jul-Aug 2024)": ("2024-07-16", "2024-08-05"),
}


# ---------------------------------------------------------------------------
# Historical crisis replay
# ---------------------------------------------------------------------------

def historical_crisis_replay(returns, dates=None):
    """
    Measure portfolio performance during known historical crisis windows.

    If ``dates`` is provided, returns are filtered to the crisis windows.
    If not, the function uses the full return series and matches date
    ranges where possible.

    Parameters
    ----------
    returns : array-like or pd.Series
        Daily portfolio returns.  If a Series with a DatetimeIndex, dates
        are used to filter crisis windows.  If a plain array, ``dates``
        must be provided separately.
    dates : array-like of datetime-like, optional
        Dates corresponding to each return.  Ignored if ``returns`` is
        already a DatetimeIndex Series.

    Returns
    -------
    dict
        {crisis_name: {"total_return": float, "max_drawdown": float,
         "worst_day": float, "trading_days": int}}
        Only includes crises that overlap with the data.
    """
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        ret_series = returns.copy()
    elif dates is not None:
        ret_series = pd.Series(
            np.asarray(returns, dtype=float),
            index=pd.to_datetime(dates),
        )
    else:
        return {}

    ret_series.index = ret_series.index.tz_localize(None)
    ret_series = ret_series.dropna()

    results = {}
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = (ret_series.index >= pd.Timestamp(start)) & (
            ret_series.index <= pd.Timestamp(end)
        )
        crisis_rets = ret_series[mask]
        if len(crisis_rets) < 2:
            continue

        cumulative = (1 + crisis_rets).cumprod()
        total_return = float(cumulative.iloc[-1] / cumulative.iloc[0] - 1)

        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(drawdowns.min())

        results[name] = {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "worst_day": float(crisis_rets.min()),
            "best_day": float(crisis_rets.max()),
            "trading_days": len(crisis_rets),
        }

    return results


# ---------------------------------------------------------------------------
# Parametric stress scenarios
# ---------------------------------------------------------------------------

def parametric_stress(returns, shocks=None):
    """
    Apply parametric shocks to portfolio returns and measure impact.

    Each shock multiplies every return by a factor, simulating amplified
    market moves.  Useful for "what if the market was 2x more volatile?"

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    shocks : dict, optional
        {scenario_name: multiplier}.  Default scenarios provided if None.

    Returns
    -------
    dict
        {scenario_name: {"mean_return": float, "var_95": float,
         "cvar_95": float, "max_drawdown": float, "volatility": float}}
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return {}

    if shocks is None:
        shocks = {
            "base (1x)": 1.0,
            "moderate stress (1.5x)": 1.5,
            "severe stress (2x)": 2.0,
            "extreme stress (3x)": 3.0,
            "mild recovery (0.5x)": 0.5,
        }

    results = {}
    for name, multiplier in shocks.items():
        stressed = returns * multiplier

        cumulative = np.cumprod(1 + stressed)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        percentile_5 = float(np.percentile(stressed, 5))
        tail = stressed[stressed <= percentile_5]
        cvar = float(np.mean(tail)) if len(tail) > 0 else percentile_5

        results[name] = {
            "mean_return": float(np.mean(stressed)),
            "volatility": float(np.std(stressed, ddof=1)) if len(stressed) > 1 else 0.0,
            "var_95": percentile_5,
            "cvar_95": cvar,
            "max_drawdown": float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0,
            "multiplier": multiplier,
        }

    return results


# ---------------------------------------------------------------------------
# Monte Carlo VaR
# ---------------------------------------------------------------------------

def monte_carlo_var(returns, n_simulations=10000, horizon_days=5,
                    confidence=0.95, seed=42):
    """
    Estimate VaR via Monte Carlo simulation.

    Fits a normal distribution to historical returns, then simulates
    ``n_simulations`` paths of ``horizon_days`` length.  Reports the
    loss threshold at the given confidence level.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    n_simulations : int
        Number of Monte Carlo paths.
    horizon_days : int
        Forward horizon in trading days.
    confidence : float
        Confidence level (e.g. 0.95 for 5% VaR).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {"mc_var": float, "mc_cvar": float, "mean_terminal": float,
         "median_terminal": float, "worst_path": float, "best_path": float,
         "pct_negative": float}
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return {
            "mc_var": 0.0, "mc_cvar": 0.0, "mean_terminal": 0.0,
            "median_terminal": 0.0, "worst_path": 0.0, "best_path": 0.0,
            "pct_negative": 0.0,
        }

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    rng = np.random.default_rng(seed)
    simulated = rng.normal(mu, sigma, size=(n_simulations, horizon_days))

    # Terminal cumulative return for each path
    terminal = np.prod(1 + simulated, axis=1) - 1

    percentile = (1 - confidence) * 100
    mc_var = float(np.percentile(terminal, percentile))
    tail = terminal[terminal <= mc_var]
    mc_cvar = float(np.mean(tail)) if len(tail) > 0 else mc_var

    return {
        "mc_var": mc_var,
        "mc_cvar": mc_cvar,
        "mean_terminal": float(np.mean(terminal)),
        "median_terminal": float(np.median(terminal)),
        "worst_path": float(np.min(terminal)),
        "best_path": float(np.max(terminal)),
        "pct_negative": float(np.mean(terminal < 0)),
    }


# ---------------------------------------------------------------------------
# Correlation stress
# ---------------------------------------------------------------------------

def correlation_stress(weights, individual_vols, base_correlation=0.3,
                       stress_correlations=None):
    """
    Estimate portfolio volatility under different correlation assumptions.

    In a crisis, correlations spike toward 1.0. This function shows how
    portfolio risk changes as pairwise correlations increase.

    Parameters
    ----------
    weights : array-like or dict
        Portfolio weights.
    individual_vols : array-like or dict
        Per-asset annualised volatilities (same order as weights).
    base_correlation : float
        Normal-state average pairwise correlation.
    stress_correlations : list of float, optional
        Correlation levels to test.  Defaults to [0.3, 0.5, 0.7, 0.9, 1.0].

    Returns
    -------
    dict
        {correlation_level: {"portfolio_vol": float, "vol_increase_pct": float}}
    """
    if isinstance(weights, dict):
        w = np.array(list(weights.values()), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    if isinstance(individual_vols, dict):
        v = np.array(list(individual_vols.values()), dtype=float)
    else:
        v = np.asarray(individual_vols, dtype=float)

    n = min(len(w), len(v))
    if n == 0:
        return {}
    w, v = w[:n], v[:n]

    w = w[~np.isnan(w)]
    v = v[~np.isnan(v)]
    n = min(len(w), len(v))
    if n == 0:
        return {}
    w, v = w[:n], v[:n]

    if stress_correlations is None:
        stress_correlations = [0.3, 0.5, 0.7, 0.9, 1.0]

    def _portfolio_vol(corr):
        # Build correlation matrix with uniform off-diagonal correlation
        corr_matrix = np.full((n, n), corr)
        np.fill_diagonal(corr_matrix, 1.0)
        # Covariance = diag(vol) @ corr @ diag(vol)
        cov = np.outer(v, v) * corr_matrix
        port_var = w @ cov @ w
        return float(np.sqrt(max(port_var, 0.0)))

    base_vol = _portfolio_vol(base_correlation)

    results = {}
    for corr in stress_correlations:
        port_vol = _portfolio_vol(corr)
        increase = ((port_vol / base_vol) - 1) if base_vol > 0 else 0.0
        results[corr] = {
            "portfolio_vol": port_vol,
            "vol_increase_pct": float(increase),
        }

    return results


# ---------------------------------------------------------------------------
# Sector concentration stress
# ---------------------------------------------------------------------------

def sector_concentration_stress(weights, sector_map, sector_shocks=None):
    """
    Estimate portfolio loss from sector-specific drawdowns.

    Parameters
    ----------
    weights : dict
        {symbol: weight}.
    sector_map : dict
        {symbol: sector_name}.
    sector_shocks : dict, optional
        {sector_name: drawdown_pct}.  Default shocks provided if None.

    Returns
    -------
    dict
        {scenario_name: {"sector": str, "shock": float,
         "portfolio_impact": float, "affected_weight": float}}
    """
    if not weights or not sector_map:
        return {}

    if sector_shocks is None:
        sector_shocks = {
            "Tech sector -20%": ("Technology", -0.20),
            "Financials -15%": ("Financials", -0.15),
            "Energy -25%": ("Energy", -0.25),
            "Healthcare -10%": ("Healthcare", -0.10),
            "Consumer -15%": ("Consumer", -0.15),
        }

    results = {}
    for scenario_name, (sector, shock) in sector_shocks.items():
        affected_weight = sum(
            w for sym, w in weights.items()
            if sector_map.get(sym, "") == sector
        )
        portfolio_impact = affected_weight * shock

        results[scenario_name] = {
            "sector": sector,
            "shock": shock,
            "portfolio_impact": float(portfolio_impact),
            "affected_weight": float(affected_weight),
        }

    return results


# ---------------------------------------------------------------------------
# Composite stress report
# ---------------------------------------------------------------------------

def stress_report(returns, weights=None, dates=None,
                  individual_vols=None, sector_map=None):
    """
    Run all stress tests and return a consolidated report.

    Parameters
    ----------
    returns : array-like or pd.Series
        Daily portfolio returns.
    weights : dict, optional
        Current portfolio weights.
    dates : array-like, optional
        Dates for historical replay (ignored if returns has DatetimeIndex).
    individual_vols : dict, optional
        Per-asset vols for correlation stress.
    sector_map : dict, optional
        {symbol: sector} for sector stress.

    Returns
    -------
    dict
        {"historical_crises": ..., "parametric": ..., "monte_carlo": ...,
         "correlation": ..., "sector": ...}
    """
    report = {}

    # Historical crisis replay
    report["historical_crises"] = historical_crisis_replay(returns, dates)

    # Parametric stress
    ret_array = np.asarray(returns, dtype=float) if not isinstance(
        returns, np.ndarray
    ) else returns
    report["parametric"] = parametric_stress(ret_array)

    # Monte Carlo VaR
    report["monte_carlo"] = monte_carlo_var(ret_array)

    # Correlation stress (requires weights + vols)
    if weights is not None and individual_vols is not None:
        report["correlation"] = correlation_stress(weights, individual_vols)
    else:
        report["correlation"] = {}

    # Sector concentration stress
    if weights is not None and sector_map is not None:
        report["sector"] = sector_concentration_stress(weights, sector_map)
    else:
        report["sector"] = {}

    return report
