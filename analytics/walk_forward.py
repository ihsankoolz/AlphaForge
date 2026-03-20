"""
analytics/walk_forward.py
==========================
Walk-forward rolling-window backtest analysis with proper P&L attribution.

Provides:
    - Rolling-window performance metrics (Sharpe, return, drawdown per window)
    - P&L attribution by regime, sector, and time period
    - Out-of-sample degradation detection (does performance decay over time?)
    - Rolling stability metrics (is strategy performance consistent?)

Why walk-forward analysis matters:
    A single backtest number (e.g. "183% total return") hides whether the
    strategy was consistently profitable or just had one lucky quarter.
    Walk-forward breaks performance into non-overlapping windows and checks
    that each window contributes positively.

All functions are pure — they accept arrays/DataFrames and return dicts.
No I/O, no side effects.

Usage:
    from analytics.walk_forward import (
        rolling_window_metrics,
        pnl_attribution_by_regime,
        pnl_attribution_by_period,
        performance_stability,
        oos_degradation,
        walk_forward_report,
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling-window performance metrics
# ---------------------------------------------------------------------------

def rolling_window_metrics(returns, window_days=63):
    """
    Compute performance metrics over non-overlapping windows.

    Default window = 63 trading days (~ 1 quarter).

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    window_days : int
        Window size in trading days.

    Returns
    -------
    list of dict
        Each dict: {"window": int, "total_return": float, "sharpe": float,
         "max_drawdown": float, "volatility": float, "win_rate": float,
         "num_days": int}
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < window_days:
        return []

    n_windows = len(returns) // window_days
    results = []

    for i in range(n_windows):
        start = i * window_days
        end = start + window_days
        window_rets = returns[start:end]

        cumulative = np.cumprod(1 + window_rets)
        total_ret = float(cumulative[-1] - 1)

        vol = float(np.std(window_rets, ddof=1)) if len(window_rets) > 1 else 0.0
        sharpe = 0.0
        if vol > 0:
            annualised_mean = np.mean(window_rets) * 252
            annualised_vol = vol * np.sqrt(252)
            sharpe = float(annualised_mean / annualised_vol)

        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        win_rate = float(np.mean(window_rets > 0))

        results.append({
            "window": i + 1,
            "total_return": total_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "volatility": vol,
            "win_rate": win_rate,
            "num_days": len(window_rets),
        })

    return results


# ---------------------------------------------------------------------------
# P&L attribution by regime
# ---------------------------------------------------------------------------

def pnl_attribution_by_regime(returns, regimes):
    """
    Break down P&L contribution by market regime.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    regimes : array-like
        Regime labels for each day (same length as returns).
        E.g. ["bull", "bull", "bear", "choppy", ...].

    Returns
    -------
    dict
        {regime: {"total_return": float, "mean_daily": float,
         "sharpe": float, "num_days": int, "pct_of_total_pnl": float}}
    """
    returns = np.asarray(returns, dtype=float)
    regimes = np.asarray(regimes)

    n = min(len(returns), len(regimes))
    if n == 0:
        return {}
    returns, regimes = returns[:n], regimes[:n]

    mask = ~np.isnan(returns)
    returns, regimes = returns[mask], regimes[mask]

    total_pnl = float(np.sum(returns))

    results = {}
    for regime in np.unique(regimes):
        regime_mask = regimes == regime
        regime_rets = returns[regime_mask]

        if len(regime_rets) == 0:
            continue

        regime_pnl = float(np.sum(regime_rets))
        mean_daily = float(np.mean(regime_rets))
        vol = float(np.std(regime_rets, ddof=1)) if len(regime_rets) > 1 else 0.0
        sharpe = 0.0
        if vol > 0:
            sharpe = float((mean_daily / vol) * np.sqrt(252))

        pct_of_total = (regime_pnl / total_pnl) if total_pnl != 0 else 0.0

        results[str(regime)] = {
            "total_return": regime_pnl,
            "mean_daily": mean_daily,
            "sharpe": sharpe,
            "num_days": int(regime_mask.sum()),
            "pct_of_total_pnl": float(pct_of_total),
        }

    return results


# ---------------------------------------------------------------------------
# P&L attribution by time period
# ---------------------------------------------------------------------------

def pnl_attribution_by_period(returns, dates, period="quarter"):
    """
    Break down P&L by calendar period (month, quarter, or year).

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    dates : array-like of datetime-like
        Date for each return.
    period : str
        "month", "quarter", or "year".

    Returns
    -------
    list of dict
        Each dict: {"period": str, "total_return": float,
         "sharpe": float, "num_days": int, "cumulative_return": float}
    """
    returns = np.asarray(returns, dtype=float)
    dates = pd.to_datetime(dates)

    n = min(len(returns), len(dates))
    if n == 0:
        return []
    returns, dates = returns[:n], dates[:n]

    df = pd.DataFrame({"return": returns, "date": dates})
    df = df.dropna(subset=["return"])

    if period == "month":
        df["period"] = df["date"].dt.to_period("M").astype(str)
    elif period == "quarter":
        df["period"] = df["date"].dt.to_period("Q").astype(str)
    elif period == "year":
        df["period"] = df["date"].dt.to_period("Y").astype(str)
    else:
        df["period"] = df["date"].dt.to_period("Q").astype(str)

    results = []
    cumulative = 1.0

    for period_name, group in df.groupby("period", sort=True):
        rets = group["return"].values
        period_return = float(np.prod(1 + rets) - 1)
        cumulative *= (1 + period_return)

        vol = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        sharpe = 0.0
        if vol > 0:
            sharpe = float((np.mean(rets) / vol) * np.sqrt(252))

        results.append({
            "period": str(period_name),
            "total_return": period_return,
            "sharpe": sharpe,
            "num_days": len(rets),
            "cumulative_return": float(cumulative - 1),
        })

    return results


# ---------------------------------------------------------------------------
# Performance stability
# ---------------------------------------------------------------------------

def performance_stability(returns, window_days=63):
    """
    Measure how consistent strategy performance is across windows.

    A stable strategy has:
        - High % of profitable windows
        - Low standard deviation of per-window Sharpe
        - Positive worst-window return

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    window_days : int
        Window size for rolling analysis.

    Returns
    -------
    dict
        {"pct_profitable_windows": float, "sharpe_std": float,
         "sharpe_mean": float, "worst_window_return": float,
         "best_window_return": float, "return_consistency": float,
         "n_windows": int}
    """
    windows = rolling_window_metrics(returns, window_days)
    if not windows:
        return {
            "pct_profitable_windows": 0.0,
            "sharpe_std": 0.0,
            "sharpe_mean": 0.0,
            "worst_window_return": 0.0,
            "best_window_return": 0.0,
            "return_consistency": 0.0,
            "n_windows": 0,
        }

    total_returns = [w["total_return"] for w in windows]
    sharpes = [w["sharpe"] for w in windows]

    pct_profitable = float(np.mean(np.array(total_returns) > 0))
    sharpe_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    sharpe_mean = float(np.mean(sharpes))

    # Return consistency = mean / std of window returns (like a Sharpe of Sharpes)
    ret_std = float(np.std(total_returns, ddof=1)) if len(total_returns) > 1 else 0.0
    ret_mean = float(np.mean(total_returns))
    consistency = (ret_mean / ret_std) if ret_std > 0 else 0.0

    return {
        "pct_profitable_windows": pct_profitable,
        "sharpe_std": sharpe_std,
        "sharpe_mean": sharpe_mean,
        "worst_window_return": float(min(total_returns)),
        "best_window_return": float(max(total_returns)),
        "return_consistency": float(consistency),
        "n_windows": len(windows),
    }


# ---------------------------------------------------------------------------
# Out-of-sample degradation detection
# ---------------------------------------------------------------------------

def oos_degradation(returns, split_pct=0.5):
    """
    Compare first-half vs second-half performance to detect degradation.

    If the second half is significantly worse, the strategy may be
    overfitting to earlier data or the market regime may have shifted.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    split_pct : float
        Where to split (0.5 = midpoint).

    Returns
    -------
    dict
        {"first_half_sharpe": float, "second_half_sharpe": float,
         "sharpe_change": float, "first_half_return": float,
         "second_half_return": float, "degradation_detected": bool}
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    if len(returns) < 20:
        return {
            "first_half_sharpe": 0.0, "second_half_sharpe": 0.0,
            "sharpe_change": 0.0, "first_half_return": 0.0,
            "second_half_return": 0.0, "degradation_detected": False,
        }

    split = int(len(returns) * split_pct)
    first = returns[:split]
    second = returns[split:]

    def _sharpe(r):
        if len(r) < 2:
            return 0.0
        vol = np.std(r, ddof=1)
        if vol == 0:
            return 0.0
        return float((np.mean(r) / vol) * np.sqrt(252))

    def _total_return(r):
        return float(np.prod(1 + r) - 1)

    first_sharpe = _sharpe(first)
    second_sharpe = _sharpe(second)
    sharpe_change = second_sharpe - first_sharpe

    # Degradation = second half Sharpe drops by more than 0.3
    degraded = sharpe_change < -0.3

    return {
        "first_half_sharpe": first_sharpe,
        "second_half_sharpe": second_sharpe,
        "sharpe_change": float(sharpe_change),
        "first_half_return": _total_return(first),
        "second_half_return": _total_return(second),
        "degradation_detected": bool(degraded),
    }


# ---------------------------------------------------------------------------
# Composite walk-forward report
# ---------------------------------------------------------------------------

def walk_forward_report(returns, dates=None, regimes=None, window_days=63):
    """
    Run all walk-forward analyses and return a consolidated report.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    dates : array-like, optional
        Dates for period attribution.
    regimes : array-like, optional
        Regime labels for regime attribution.
    window_days : int
        Window size for rolling analysis.

    Returns
    -------
    dict
        {"rolling_windows": ..., "stability": ..., "degradation": ...,
         "regime_attribution": ..., "quarterly_attribution": ...}
    """
    report = {}

    report["rolling_windows"] = rolling_window_metrics(returns, window_days)
    report["stability"] = performance_stability(returns, window_days)
    report["degradation"] = oos_degradation(returns)

    if regimes is not None:
        report["regime_attribution"] = pnl_attribution_by_regime(returns, regimes)
    else:
        report["regime_attribution"] = {}

    if dates is not None:
        report["quarterly_attribution"] = pnl_attribution_by_period(
            returns, dates, period="quarter"
        )
    else:
        report["quarterly_attribution"] = []

    return report
