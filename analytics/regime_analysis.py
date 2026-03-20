"""
analytics/regime_analysis.py
=============================
Analyse HMM regime predictions — transition patterns, regime-conditional
performance, and persistence statistics.

Answers questions like:
    - How often does bull transition to bear vs stay bull?
    - What is the Sharpe ratio *within* each regime?
    - How many days does the average regime persist?
    - Are regime predictions accurate (predicted vs realised)?

All functions are pure — they accept arrays/DataFrames and return
numbers or dicts.  No I/O, no side effects.

Usage:
    from analytics.regime_analysis import (
        transition_matrix, regime_conditional_performance,
        regime_persistence, regime_accuracy, regime_report,
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def transition_matrix(regimes, labels=None):
    """
    Compute the empirical regime transition matrix.

    T[i, j] = P(regime tomorrow = j | regime today = i).

    Parameters
    ----------
    regimes : array-like
        Sequence of regime labels (e.g. [0, 0, 1, 2, 1, 0, ...]).
    labels : list, optional
        Names for regimes (e.g. ["bull", "choppy", "bear"]).
        If None, uses unique sorted values from regimes.

    Returns
    -------
    dict
        ``matrix`` (2D list of floats, row = from, col = to),
        ``labels`` (list of str/int), ``counts`` (2D list of ints).
    """
    regimes = np.asarray(regimes)
    regimes = regimes[~pd.isna(regimes)]

    if len(regimes) < 2:
        return {"matrix": [], "labels": [], "counts": []}

    if labels is None:
        labels = sorted(set(regimes))

    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    counts = np.zeros((n, n), dtype=int)

    for t in range(len(regimes) - 1):
        curr = regimes[t]
        nxt = regimes[t + 1]
        if curr in label_to_idx and nxt in label_to_idx:
            counts[label_to_idx[curr], label_to_idx[nxt]] += 1

    # Normalise rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(row_sums > 0, counts / row_sums, 0.0)

    return {
        "matrix": matrix.tolist(),
        "labels": [str(l) for l in labels],
        "counts": counts.tolist(),
    }


# ---------------------------------------------------------------------------
# Regime-conditional performance
# ---------------------------------------------------------------------------

def regime_conditional_performance(returns, regimes):
    """
    Compute performance metrics per regime.

    Parameters
    ----------
    returns : array-like
        Daily portfolio returns.
    regimes : array-like
        Regime label for each day (same length as returns).

    Returns
    -------
    dict
        Keyed by regime label.  Each value is a dict with ``count``,
        ``mean_return``, ``std_return``, ``sharpe`` (annualised),
        ``max_drawdown``, ``win_rate``.
    """
    returns = np.asarray(returns, dtype=float)
    regimes = np.asarray(regimes)

    n = min(len(returns), len(regimes))
    if n == 0:
        return {}
    returns, regimes = returns[:n], regimes[:n]

    # Remove NaN returns
    mask = ~np.isnan(returns)
    returns, regimes = returns[mask], regimes[mask]

    unique_regimes = sorted(set(regimes))
    result = {}

    for regime in unique_regimes:
        idx = regimes == regime
        r = returns[idx]

        if len(r) == 0:
            continue

        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        # Max drawdown within this regime's days
        cumulative = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_rate = float(np.mean(r > 0))

        result[str(regime)] = {
            "count": int(np.sum(idx)),
            "mean_return": mean_r,
            "std_return": std_r,
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
            "win_rate": win_rate,
        }

    return result


# ---------------------------------------------------------------------------
# Regime persistence
# ---------------------------------------------------------------------------

def regime_persistence(regimes):
    """
    Compute average duration (in days) of each regime.

    Parameters
    ----------
    regimes : array-like
        Daily regime labels.

    Returns
    -------
    dict
        Keyed by regime label.  Each value is a dict with
        ``avg_duration`` (float days), ``max_duration`` (int),
        ``min_duration`` (int), ``n_episodes`` (int).
    """
    regimes = np.asarray(regimes)
    regimes = regimes[~pd.isna(regimes)]

    if len(regimes) == 0:
        return {}

    # Find runs
    episodes = []
    current = regimes[0]
    length = 1

    for i in range(1, len(regimes)):
        if regimes[i] == current:
            length += 1
        else:
            episodes.append((current, length))
            current = regimes[i]
            length = 1
    episodes.append((current, length))

    # Aggregate by regime
    from collections import defaultdict
    durations = defaultdict(list)
    for regime, dur in episodes:
        durations[regime].append(dur)

    result = {}
    for regime, durs in durations.items():
        result[str(regime)] = {
            "avg_duration": float(np.mean(durs)),
            "max_duration": int(np.max(durs)),
            "min_duration": int(np.min(durs)),
            "n_episodes": len(durs),
        }

    return result


# ---------------------------------------------------------------------------
# Regime accuracy (predicted vs realised)
# ---------------------------------------------------------------------------

def regime_accuracy(predicted_regimes, realised_returns,
                    bull_threshold=0.0, bear_threshold=0.0):
    """
    Measure whether regime predictions match realised market behaviour.

    "Bull" should have positive returns, "bear" should have negative.
    This checks if the HMM is actually useful.

    Parameters
    ----------
    predicted_regimes : array-like
        Regime labels.  Expected values: 0 (bull), 1 (choppy), 2 (bear)
        or strings "bull", "choppy", "bear".
    realised_returns : array-like
        Actual daily market returns for the same days.
    bull_threshold : float
        Return above this is considered "truly bullish" (default 0).
    bear_threshold : float
        Return below this is considered "truly bearish" (default 0).

    Returns
    -------
    dict
        ``overall_accuracy`` (float), per-regime accuracy, and
        ``regime_distribution`` (how often each regime was predicted).
    """
    predicted = np.asarray(predicted_regimes)
    returns = np.asarray(realised_returns, dtype=float)

    n = min(len(predicted), len(returns))
    if n == 0:
        return {"overall_accuracy": 0.0, "per_regime": {},
                "regime_distribution": {}}

    predicted, returns = predicted[:n], returns[:n]
    mask = ~np.isnan(returns)
    predicted, returns = predicted[mask], returns[mask]

    if len(predicted) == 0:
        return {"overall_accuracy": 0.0, "per_regime": {},
                "regime_distribution": {}}

    # Map string labels to ints if needed
    label_map = {"bull": 0, "choppy": 1, "bear": 2}
    mapped = []
    for p in predicted:
        if isinstance(p, str) and p.lower() in label_map:
            mapped.append(label_map[p.lower()])
        else:
            mapped.append(p)
    predicted = np.array(mapped)

    # Classify actual return
    correct = 0
    per_regime_correct = {}
    per_regime_total = {}

    for i in range(len(predicted)):
        regime = predicted[i]
        ret = returns[i]

        regime_key = str(regime)
        per_regime_total[regime_key] = per_regime_total.get(regime_key, 0) + 1

        is_correct = False
        if regime == 0:  # bull
            is_correct = ret > bull_threshold
        elif regime == 2:  # bear
            is_correct = ret < bear_threshold
        elif regime == 1:  # choppy — correct if absolute return is small
            is_correct = abs(ret) < 0.01  # within 1% is "choppy"

        if is_correct:
            correct += 1
            per_regime_correct[regime_key] = \
                per_regime_correct.get(regime_key, 0) + 1

    overall = correct / len(predicted) if len(predicted) > 0 else 0.0

    # Per-regime accuracy
    per_regime = {}
    for key in per_regime_total:
        total = per_regime_total[key]
        hits = per_regime_correct.get(key, 0)
        per_regime[key] = {
            "accuracy": float(hits / total) if total > 0 else 0.0,
            "count": total,
        }

    # Distribution
    unique, counts = np.unique(predicted, return_counts=True)
    distribution = {str(u): int(c) for u, c in zip(unique, counts)}

    return {
        "overall_accuracy": float(overall),
        "per_regime": per_regime,
        "regime_distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------

def regime_report(regimes, returns, benchmark_returns=None):
    """
    Generate a comprehensive regime analysis report.

    Parameters
    ----------
    regimes : array-like
        Daily regime labels.
    returns : array-like
        Daily portfolio returns (same length as regimes).
    benchmark_returns : array-like, optional
        Market returns for regime accuracy check.

    Returns
    -------
    dict
        Full report: transitions, conditional performance, persistence,
        and accuracy.
    """
    report = {}

    report["transition_matrix"] = transition_matrix(regimes)
    report["conditional_performance"] = regime_conditional_performance(
        returns, regimes)
    report["persistence"] = regime_persistence(regimes)

    if benchmark_returns is not None:
        report["accuracy"] = regime_accuracy(regimes, benchmark_returns)
    else:
        report["accuracy"] = regime_accuracy(regimes, returns)

    return report
