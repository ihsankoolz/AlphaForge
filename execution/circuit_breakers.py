"""
execution/circuit_breakers.py
=============================
Sanity checks that run BEFORE orders are placed.

If any check fails, the pipeline halts and holds cash for the day.
These catch situations where the model is producing garbage but no
Python error is raised — the most dangerous class of bugs in a
trading system.

Why circuit breakers matter:
    A model that predicts 0.99 for every stock places max-position
    orders across the board. No exception is thrown — XGBoost happily
    returns probabilities. Without circuit breakers, you'd fully invest
    based on a broken model and only discover it when you check P&L.
"""

import logging
from typing import Optional

import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    CB_MAX_SIGNAL_RATIO,
    CB_MAX_ALLOCATION_JUMP,
    CB_MAX_DAILY_TURNOVER,
    CB_MIN_EXPECTED_SIGNALS,
    MIN_PROB,
)

log = logging.getLogger("circuit_breakers")


class CircuitBreakerTripped(Exception):
    """Raised when a hard circuit breaker is tripped — halts trading."""
    pass


def check_signal_sanity(
    signals_df: pd.DataFrame,
    total_symbols: int,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    HARD BREAKER: If >80% of stocks are above the signal threshold,
    something is wrong with the model — it's not discriminating.

    A well-functioning model should flag 30-60% of stocks as above
    threshold. If it flags 80%+, it's likely:
    - A feature is NaN/constant across all stocks (model defaults to high)
    - The model pkl is corrupted or from a different training run
    - A data pipeline bug produced identical features for all stocks

    Returns True if safe, raises CircuitBreakerTripped if breached.
    """
    _log = logger or log

    above = signals_df[signals_df["pred_proba"] >= MIN_PROB]
    ratio = len(above) / max(total_symbols, 1)

    if ratio > CB_MAX_SIGNAL_RATIO:
        msg = (
            f"CIRCUIT BREAKER: {ratio:.0%} of stocks above threshold "
            f"(limit: {CB_MAX_SIGNAL_RATIO:.0%}). "
            f"Model may be producing indiscriminate signals. "
            f"Halting trading for safety."
        )
        _log.error(msg)
        raise CircuitBreakerTripped(msg)

    if len(above) < CB_MIN_EXPECTED_SIGNALS and len(above) > 0:
        _log.warning(
            f"Only {len(above)} signals above threshold "
            f"(expected ≥{CB_MIN_EXPECTED_SIGNALS}). "
            f"Model may be too conservative today."
        )

    _log.info(
        f"Signal sanity check OK: {len(above)}/{total_symbols} "
        f"({ratio:.0%}) above threshold"
    )
    return True


def check_allocation_jump(
    target_weights: dict,
    previous_weights: dict,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    HARD BREAKER: If total allocation changes by more than 30% in a
    single day, something unusual is happening.

    Example: yesterday invested 85%, today suddenly 20% (or vice versa).
    This could indicate:
    - Regime detection flipped unexpectedly
    - Risk layer bug produced extreme weights
    - Data gap caused stale features

    Returns True if safe, raises CircuitBreakerTripped if breached.
    """
    _log = logger or log

    current_total  = sum(target_weights.values()) if target_weights else 0
    previous_total = sum(previous_weights.values()) if previous_weights else 0

    jump = abs(current_total - previous_total)

    if jump > CB_MAX_ALLOCATION_JUMP and previous_total > 0:
        msg = (
            f"CIRCUIT BREAKER: Allocation jumped {jump:.0%} in one day "
            f"({previous_total:.0%} → {current_total:.0%}, "
            f"limit: {CB_MAX_ALLOCATION_JUMP:.0%}). "
            f"Halting trading — investigate before proceeding."
        )
        _log.error(msg)
        raise CircuitBreakerTripped(msg)

    _log.info(
        f"Allocation jump check OK: {previous_total:.0%} → {current_total:.0%} "
        f"(Δ {jump:.0%})"
    )
    return True


def check_turnover(
    target_weights: dict,
    current_weights: dict,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    SOFT WARNING: If portfolio turnover exceeds 50%, log a warning.
    Does NOT halt trading — high turnover is sometimes correct (e.g.,
    regime change from bull to bear).

    Turnover = sum of absolute weight changes / 2
    (divide by 2 because each trade has a buy side and sell side)

    Returns True always (soft check).
    """
    _log = logger or log

    all_symbols = set(list(target_weights.keys()) + list(current_weights.keys()))
    total_change = sum(
        abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0))
        for s in all_symbols
    )
    turnover = total_change / 2

    if turnover > CB_MAX_DAILY_TURNOVER:
        _log.warning(
            f"High turnover: {turnover:.0%} of portfolio changing today "
            f"(threshold: {CB_MAX_DAILY_TURNOVER:.0%}). "
            f"This may be correct (e.g., regime change) but will incur "
            f"higher transaction costs."
        )
    else:
        _log.info(f"Turnover check OK: {turnover:.0%}")

    return True


def check_weight_sum(
    target_weights: dict,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    HARD BREAKER: Target weights must sum to ≤ 1.0 (can't invest more
    than 100% of capital without leverage, which we don't use).

    Returns True if safe, raises CircuitBreakerTripped if breached.
    """
    _log = logger or log
    total = sum(target_weights.values()) if target_weights else 0

    if total > 1.05:  # small tolerance for floating point
        msg = (
            f"CIRCUIT BREAKER: Target weights sum to {total:.2%} "
            f"(>100%). Cannot invest more than total capital without leverage. "
            f"Halting trading."
        )
        _log.error(msg)
        raise CircuitBreakerTripped(msg)

    _log.info(f"Weight sum check OK: {total:.1%} total allocation")
    return True


def run_all_checks(
    signals_df: pd.DataFrame,
    target_weights: dict,
    previous_weights: dict,
    current_weights: dict,
    total_symbols: int,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Run all circuit breaker checks in sequence.

    Hard breakers raise CircuitBreakerTripped (caller should catch and
    go to cash). Soft breakers log warnings but don't halt.

    Args:
        signals_df:       ML signal output
        target_weights:   Today's proposed portfolio
        previous_weights: Yesterday's portfolio (from logs or cache)
        current_weights:  Current Alpaca positions
        total_symbols:    Number of symbols in universe

    Returns True if all checks pass.
    Raises CircuitBreakerTripped if any hard check fails.
    """
    _log = logger or log
    _log.info("Running circuit breaker checks...")

    check_signal_sanity(signals_df, total_symbols, _log)
    check_weight_sum(target_weights, _log)
    check_allocation_jump(target_weights, previous_weights, _log)
    check_turnover(target_weights, current_weights, _log)

    _log.info("All circuit breaker checks passed")
    return True
