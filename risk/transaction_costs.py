"""
risk/transaction_costs.py
─────────────────────────
Realistic transaction cost model replacing the flat 0.1% assumption.

Real trading costs have three components:
  1. Half-spread — you buy at ask and sell at bid. The spread depends on
     liquidity (large-cap tech ≈ 1bp, mid-cap ≈ 5bp, small-cap ≈ 15bp).
  2. Market impact — your order moves the price against you. Larger orders
     in volatile stocks cause more impact. Modeled as: impact ∝ sqrt(order_fraction) × volatility.
  3. Broker commission — Alpaca charges $0 for stocks, so this is 0 for us.

Why this matters:
  - Flat 0.1% overestimates costs for large-cap liquid stocks (AAPL spread ≈ 0.01%)
    and underestimates costs for volatile mid-caps during high-turnover days.
  - The optimizer should know that trading volatile stocks is expensive, so it
    can penalize unnecessary turnover in those names.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MIN_ORDER_VALUE


# ── SPREAD ESTIMATES BY LIQUIDITY TIER ──────────────────────────────────────
# Half-spread in basis points (1 bp = 0.01%).
# These are conservative estimates for US large/mid-cap equities during
# regular trading hours. Real spreads are tighter for mega-caps and
# wider near open/close.

# Tier 1: Mega-cap, extremely liquid (AAPL, MSFT, AMZN, GOOGL, etc.)
# Tier 2: Large-cap, liquid (most of our universe)
# Tier 3: Mid-cap or lower volume (some Batch 1 additions)

SPREAD_TIERS = {
    # Tier 1: ~1bp half-spread
    "AAPL": 1, "MSFT": 1, "AMZN": 1, "GOOGL": 1, "NVDA": 1, "META": 1,
    "TSLA": 2, "SPY": 0.5, "QQQ": 0.5,
    # Tier 2: ~2-3bp (default for most stocks)
    # Tier 3: ~4-5bp (assigned below)
    "NEM": 5, "DUK": 4, "NEE": 3, "AMT": 3, "PLD": 3,
    "COF": 4, "BK": 4, "SCHW": 3, "MDT": 3,
}

DEFAULT_HALF_SPREAD_BP = 2.5  # basis points — reasonable for large-cap US equities


def get_half_spread(symbol: str) -> float:
    """
    Return the estimated half-spread for a symbol as a decimal fraction.
    E.g., 2.5 bp → 0.00025.
    """
    bp = SPREAD_TIERS.get(symbol, DEFAULT_HALF_SPREAD_BP)
    return bp / 10_000


def estimate_market_impact(trade_fraction: float, daily_volatility: float) -> float:
    """
    Estimate market impact cost as a fraction of trade value.

    Uses a square-root impact model (standard in execution research):
        impact = sigma × sqrt(participation_rate)

    Since we don't know exact participation rate (our order vs daily volume),
    we use a simplified version where trade_fraction is the position weight
    change, and we assume our order is small relative to daily volume
    (true for a $100K paper trading account).

    For our portfolio size, market impact is negligible — but including it
    means the model scales correctly if we ever increase capital.

    Args:
        trade_fraction: absolute weight change (e.g., 0.10 = 10% of portfolio)
        daily_volatility: stock's recent daily volatility (e.g., 0.02 = 2%)

    Returns:
        Estimated impact cost as a decimal fraction of trade value.
    """
    if trade_fraction <= 0 or daily_volatility <= 0:
        return 0.0

    # For a $100K portfolio trading 10% = $10K in a stock with $1B daily volume,
    # participation rate ≈ 0.001%. Impact is negligible.
    # Scale factor calibrated so that a 15% position change in a 3% vol stock
    # costs ~1bp of impact (realistic for our scale).
    scale = 0.01
    impact = scale * daily_volatility * np.sqrt(trade_fraction)

    return impact


def estimate_transaction_cost(
    symbol: str,
    weight_change: float,
    daily_volatility: float = 0.015,
) -> float:
    """
    Estimate the total transaction cost for a single trade.

    Total cost = half_spread + market_impact

    Args:
        symbol:           Stock ticker
        weight_change:    Absolute change in portfolio weight (e.g., 0.10)
        daily_volatility: Stock's recent daily vol (default 1.5%)

    Returns:
        Total cost as a fraction of the trade value.
        E.g., 0.0003 means 3 basis points.
    """
    spread = get_half_spread(symbol)
    impact = estimate_market_impact(weight_change, daily_volatility)
    return spread + impact


def compute_portfolio_cost(
    weight_changes: dict,
    volatilities: dict = None,
) -> float:
    """
    Compute total portfolio transaction cost for a rebalance.

    This is the function used by the backtest and optimizer. It replaces
    the old `position_changes.sum() * 0.001` with per-stock cost estimates.

    Args:
        weight_changes: {symbol: abs_weight_change} for each stock being traded
        volatilities:   {symbol: daily_vol} — optional, defaults to 1.5% per stock

    Returns:
        Total cost as a fraction of portfolio value.
        E.g., 0.002 means 20 basis points total.
    """
    if not weight_changes:
        return 0.0

    total_cost = 0.0
    for symbol, change in weight_changes.items():
        if abs(change) < 1e-6:
            continue
        vol = 0.015  # default
        if volatilities and symbol in volatilities:
            vol = volatilities[symbol]
        cost = estimate_transaction_cost(symbol, abs(change), vol)
        # Cost is per-dollar-traded, so multiply by the trade size (weight change)
        total_cost += cost * abs(change)

    return total_cost


def compute_backtest_cost(
    positions: pd.Series,
    prev_positions: pd.Series,
    features_df: pd.DataFrame = None,
    date=None,
) -> float:
    """
    Drop-in replacement for the backtest's flat cost calculation.

    Old: cost = position_changes.sum() * 0.001
    New: cost = sum of per-stock costs based on spread + impact + volatility

    Args:
        positions:      Current day's position weights (Series, index=symbols)
        prev_positions: Previous day's position weights
        features_df:    Optional — used to look up volatility per stock
        date:           Optional — current date for vol lookup

    Returns:
        Total transaction cost as a fraction of portfolio value.
    """
    changes = (positions - prev_positions).abs()
    changes = changes[changes > 1e-6]

    if changes.empty:
        return 0.0

    # Look up per-stock volatility if features available
    volatilities = {}
    if features_df is not None and date is not None:
        hist = features_df[features_df["date"] < date]
        for symbol in changes.index:
            stock_hist = hist[hist["symbol"] == symbol]
            if not stock_hist.empty and "volatility_20d" in stock_hist.columns:
                vol = stock_hist["volatility_20d"].iloc[-1]
                if not pd.isna(vol) and vol > 0:
                    volatilities[symbol] = vol

    weight_changes = changes.to_dict()
    return compute_portfolio_cost(weight_changes, volatilities)
