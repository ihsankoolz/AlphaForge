"""
analytics/risk_metrics.py
=========================
Portfolio risk metrics beyond Sharpe/Sortino/max-drawdown.

Provides:
    - Value at Risk (VaR) — historical simulation
    - Conditional VaR (CVaR / Expected Shortfall)
    - Herfindahl concentration index + effective number of bets
    - Calmar ratio (annualised return / max drawdown)
    - Portfolio beta to benchmark (SPY)
    - Return distribution moments (skewness, kurtosis)

All functions are pure — they accept arrays/DataFrames and return numbers.
No I/O, no side effects.

Usage:
    from analytics.risk_metrics import (
        compute_var, compute_cvar, herfindahl_index,
        effective_number_of_bets, calmar_ratio, portfolio_beta,
        return_skewness, return_kurtosis, risk_summary,
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Value at Risk (historical simulation)
# ---------------------------------------------------------------------------

def compute_var(returns, confidence=0.95):
    """
    Historical Value at Risk.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns (e.g. [0.01, -0.02, 0.005, ...]).
    confidence : float
        Confidence level, e.g. 0.95 for 5% VaR.

    Returns
    -------
    float
        The VaR value (a negative number representing the loss threshold).
        For example -0.023 means "on the worst 5% of days, you lose at
        least 2.3%."  Returns 0.0 if input is empty.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    percentile = (1 - confidence) * 100
    return float(np.percentile(returns, percentile))


def compute_cvar(returns, confidence=0.95):
    """
    Conditional Value at Risk (Expected Shortfall).

    The average loss on days that exceed the VaR threshold.  Captures
    tail severity better than VaR alone.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        Mean of returns that fall at or below VaR.  Returns 0.0 if input
        is empty or no returns breach VaR.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    var = compute_var(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(np.mean(tail))


# ---------------------------------------------------------------------------
# Concentration / diversification
# ---------------------------------------------------------------------------

def herfindahl_index(weights):
    """
    Herfindahl-Hirschman Index of portfolio concentration.

    HHI = sum(w_i^2).  Ranges from 1/N (perfectly diversified) to 1.0
    (single stock).

    Parameters
    ----------
    weights : dict or array-like
        Portfolio weights.  If dict, values are used.

    Returns
    -------
    float
        HHI value.  0.0 for empty portfolio.
    """
    if isinstance(weights, dict):
        w = np.array(list(weights.values()), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    w = w[~np.isnan(w)]
    if len(w) == 0 or np.sum(np.abs(w)) == 0:
        return 0.0
    return float(np.sum(w ** 2))


def effective_number_of_bets(weights):
    """
    Effective number of independent bets = 1 / HHI.

    Tells you "this portfolio behaves like N equal-weight positions".
    A 10-stock portfolio where one stock dominates might have ENB = 3.

    Parameters
    ----------
    weights : dict or array-like
        Portfolio weights.

    Returns
    -------
    float
        Effective number of bets.  0.0 for empty portfolio.
    """
    hhi = herfindahl_index(weights)
    if hhi == 0:
        return 0.0
    return float(1.0 / hhi)


# ---------------------------------------------------------------------------
# Performance ratios
# ---------------------------------------------------------------------------

def calmar_ratio(returns):
    """
    Calmar ratio = annualised return / abs(max drawdown).

    A return-to-risk ratio that penalises deep drawdowns.  Higher is better.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.

    Returns
    -------
    float
        Calmar ratio.  0.0 if max drawdown is zero or returns are empty.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    if max_dd == 0:
        return 0.0

    total_return = cumulative[-1] / cumulative[0] - 1
    years = len(returns) / 252
    if years == 0:
        return 0.0
    annualised = (1 + total_return) ** (1 / years) - 1

    return float(annualised / abs(max_dd))


# ---------------------------------------------------------------------------
# Market exposure
# ---------------------------------------------------------------------------

def portfolio_beta(portfolio_returns, benchmark_returns):
    """
    Portfolio beta to a benchmark (typically SPY).

    beta = Cov(portfolio, benchmark) / Var(benchmark)

    Parameters
    ----------
    portfolio_returns : array-like
        Daily portfolio returns.
    benchmark_returns : array-like
        Daily benchmark returns (same dates).

    Returns
    -------
    float
        Beta.  0.0 if benchmark has zero variance or inputs are empty.
    """
    pr = np.asarray(portfolio_returns, dtype=float)
    br = np.asarray(benchmark_returns, dtype=float)

    # Align lengths
    n = min(len(pr), len(br))
    if n == 0:
        return 0.0
    pr, br = pr[:n], br[:n]

    # Remove NaNs pairwise
    mask = ~(np.isnan(pr) | np.isnan(br))
    pr, br = pr[mask], br[mask]
    if len(pr) < 2:
        return 0.0

    var_b = np.var(br, ddof=1)
    if var_b == 0:
        return 0.0
    cov = np.cov(pr, br, ddof=1)[0, 1]
    return float(cov / var_b)


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------

def return_skewness(returns):
    """
    Skewness of return distribution.

    Negative skew = fatter left tail = more frequent large losses.
    Equity returns are typically negatively skewed.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.

    Returns
    -------
    float
        Skewness.  0.0 if fewer than 3 observations.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 3:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    n = len(returns)
    # Adjusted Fisher-Pearson skewness
    skew = (n / ((n - 1) * (n - 2))) * np.sum(((returns - mean) / std) ** 3)
    return float(skew)


def return_kurtosis(returns):
    """
    Excess kurtosis of return distribution.

    >0 = leptokurtic (fat tails, more extreme events than normal).
    Equity returns typically have positive excess kurtosis.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.

    Returns
    -------
    float
        Excess kurtosis.  0.0 if fewer than 4 observations.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 4:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    m4 = np.mean((returns - mean) ** 4)
    # Excess kurtosis = m4/std^4 - 3
    kurt = (m4 / (std ** 4)) - 3.0
    return float(kurt)


# ---------------------------------------------------------------------------
# Max drawdown (standalone utility)
# ---------------------------------------------------------------------------

def max_drawdown(returns):
    """
    Maximum peak-to-trough drawdown from a return series.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.

    Returns
    -------
    float
        Max drawdown as a negative number (e.g. -0.15 = 15% drawdown).
        0.0 if returns are empty or never negative.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))


# ---------------------------------------------------------------------------
# Composite summary
# ---------------------------------------------------------------------------

def risk_summary(returns, weights=None, benchmark_returns=None):
    """
    Compute all risk metrics in a single call.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    weights : dict or array-like, optional
        Current portfolio weights (for concentration metrics).
    benchmark_returns : array-like, optional
        Daily benchmark returns (for beta calculation).

    Returns
    -------
    dict
        All metrics keyed by name.
    """
    summary = {
        "var_95": compute_var(returns, 0.95),
        "cvar_95": compute_cvar(returns, 0.95),
        "var_99": compute_var(returns, 0.99),
        "cvar_99": compute_cvar(returns, 0.99),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "skewness": return_skewness(returns),
        "kurtosis": return_kurtosis(returns),
    }

    if weights is not None:
        summary["herfindahl"] = herfindahl_index(weights)
        summary["effective_bets"] = effective_number_of_bets(weights)

    if benchmark_returns is not None:
        summary["beta"] = portfolio_beta(returns, benchmark_returns)

    return summary
