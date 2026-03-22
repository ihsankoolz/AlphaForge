"""
risk/position_sizer.py
─────────────────────
Converts raw ML probability scores into position sizes using:

  1. Half-Kelly Criterion — optimal bet sizing based on model edge
  2. Volatility adjustment — scale down positions in high-vol stocks

Why this order?
  Kelly gives us the right SIZE given our edge.
  Vol adjustment gives us the right RISK given the size.
  Together: high confidence + low volatility = largest positions.
             low confidence + high volatility = smallest positions.

Output: a weight (0.0 to 1.0) per stock per day that the backtester
        uses to allocate capital. Weights sum to ≤ 1.0 (remainder = cash).
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.settings as _cfg
from config.settings import (
    KELLY_FRACTION, KELLY_ODDS, MIN_PROB, MAX_POSITION,
    VOL_LOOKBACK, VOL_TARGET, MAX_POSITIONS,
    SHORT_MAX_POSITION, SHORT_MAX_TOTAL, SHORT_MAX_POSITIONS,
    SHORT_MIN_PROB, SHORT_KELLY_FRACTION,
)


# ── 1. KELLY SIZING ──────────────────────────────────────────────────────────

def kelly_weights(pred_proba: pd.Series) -> pd.Series:
    """
    Convert ML probability scores into Kelly-sized weights for one day.

    Kelly formula:
        f = (p × b - q) / b
        where:
            p = probability of winning (our pred_proba)
            q = probability of losing (1 - p)
            b = odds (avg win / avg loss)

    We estimate b = 2.0 as a reasonable prior for a top-quartile stock
    over 5 days — historically winners beat losers ~2:1 in our universe.
    This will be calibrated from actual backtest data in production.

    Half-Kelly: multiply raw Kelly fraction by KELLY_FRACTION (0.5).
    This halves the position size, accepting lower theoretical return
    in exchange for much better drawdown protection.

    Args:
        pred_proba: Series of ML probability scores for one trading day,
                    index = stock symbols, values in [0, 1]

    Returns:
        Series of raw Kelly weights (before vol adjustment or normalisation)
    """
    b = KELLY_ODDS  # odds: winners vs losers ratio (calibrated from data)

    # Only compute Kelly for stocks above the signal threshold
    active = pred_proba[pred_proba > MIN_PROB].copy()

    if active.empty:
        return pd.Series(dtype=float)

    p = active          # probability of winning
    q = 1 - active      # probability of losing

    # Raw Kelly fraction
    kelly_raw = (p * b - q) / b

    # Clip to [0, 1] — Kelly can produce negative values (meaning "short")
    # but we're long-only so floor at 0
    kelly_raw = kelly_raw.clip(lower=0)

    # Apply half-Kelly scaling
    kelly_half = kelly_raw * KELLY_FRACTION

    return kelly_half


# ── 1b. SHORT SELLING KELLY SIZING ───────────────────────────────────────────

def short_kelly_weights(pred_proba: pd.Series) -> pd.Series:
    """
    Compute Kelly-sized SHORT weights for stocks with very low pred_proba.

    Short selling logic: if the model is confident a stock will
    underperform (pred_proba < SHORT_MIN_PROB), we can profit by
    shorting it. The Kelly formula naturally produces negative fractions
    for these — we take the absolute value and apply a more conservative
    quarter-Kelly (SHORT_KELLY_FRACTION = 0.25) because:
      1. Short losses are theoretically unlimited (stock can go up forever)
      2. Our universe is mostly large-cap winners — shorting has structural
         headwind (universe composition bias)
      3. Short squeezes can cause sudden extreme losses

    This function is only called when _cfg.ALLOW_SHORT_SELLING is True.
    Disabled by default — enable after universe expands to 129+ stocks
    where there are enough mid-caps for short signals to have edge.

    Args:
        pred_proba: Series of ML probability scores, index = symbols

    Returns:
        Series of NEGATIVE weights for short positions.
        Empty series if no stocks qualify or shorting is disabled.
    """
    if not _cfg.ALLOW_SHORT_SELLING:
        return pd.Series(dtype=float)

    # Only short stocks with very low conviction
    short_candidates = pred_proba[pred_proba < SHORT_MIN_PROB].copy()

    if short_candidates.empty:
        return pd.Series(dtype=float)

    b = KELLY_ODDS
    # For shorts, the "probability of winning" is (1 - pred_proba)
    # i.e., probability the stock goes DOWN
    p_short = 1 - short_candidates   # prob of short winning
    q_short = short_candidates        # prob of short losing

    kelly_raw = (p_short * b - q_short) / b
    kelly_raw = kelly_raw.clip(lower=0)

    # Apply quarter-Kelly (more conservative than half-Kelly for longs)
    kelly_short = kelly_raw * SHORT_KELLY_FRACTION

    # Cap individual short positions
    kelly_short = kelly_short.clip(upper=SHORT_MAX_POSITION)

    # Return as negative weights to indicate short positions
    return -kelly_short


# ── 2. VOLATILITY ADJUSTMENT ─────────────────────────────────────────────────

def volatility_adjust(kelly_w: pd.Series,
                      features_df: pd.DataFrame,
                      date: pd.Timestamp) -> pd.Series:
    """
    Scale each Kelly weight down proportionally to that stock's recent volatility.

    Logic:
        weight_final = kelly_weight × (VOL_TARGET / stock_volatility)

    If a stock's 20-day volatility is 2% per day (double our 1% target),
    we halve its position. If it's 0.5% (half our target), we double it
    (subject to MAX_POSITION cap).

    This is called "volatility targeting" or "vol parity" — ensuring each
    position contributes roughly equal risk to the portfolio regardless of
    how volatile the underlying stock is.

    Args:
        kelly_w:     Kelly weights for the active stocks today
        features_df: Full features DataFrame with volatility columns
        date:        Current trading day (to look up vol estimates)

    Returns:
        Vol-adjusted weights, still un-normalised
    """
    if kelly_w.empty:
        return kelly_w

    # Get volatility for the stocks we're trading today
    # Use the most recent available vol estimate STRICTLY BEFORE today
    # to avoid lookahead bias (today's vol uses today's close which
    # isn't known at order time — we trade at market open)
    hist = features_df[features_df["date"] < date]

    adjusted = {}
    for symbol in kelly_w.index:
        stock_hist = hist[hist["symbol"] == symbol]
        if stock_hist.empty:
            adjusted[symbol] = kelly_w[symbol]
            continue

        # Use volatility_20d — already computed in feature engineering
        vol = stock_hist["volatility_20d"].iloc[-1]

        if vol <= 0 or pd.isna(vol):
            adjusted[symbol] = kelly_w[symbol]
            continue

        # Scale: if vol is exactly VOL_TARGET, weight is unchanged
        # Higher vol → smaller weight; lower vol → larger weight
        scale = VOL_TARGET / vol
        adjusted[symbol] = kelly_w[symbol] * scale

    return pd.Series(adjusted)


# ── 3. NORMALISE & CAP ───────────────────────────────────────────────────────

def normalise_weights(weights: pd.Series,
                      max_positions: int = MAX_POSITIONS) -> pd.Series:
    """
    Take raw weights and produce final allocations that:
      - Cap any single position at MAX_POSITION (15%)
      - Keep top N positions by weight (concentration control)
      - Sum to ≤ 1.0 (remainder stays as cash)

    We do NOT force weights to sum to exactly 1.0. If the model is
    low-confidence across all stocks today, the portfolio should hold
    more cash — forced full-investment would over-ride the model's
    implicit uncertainty signal.

    Args:
        weights:       Raw vol-adjusted Kelly weights
        max_positions: Maximum concurrent positions

    Returns:
        Final normalised weights summing to ≤ 1.0
    """
    if weights.empty:
        return weights

    # Keep only the top N positions by weight
    weights = weights.nlargest(max_positions)

    # Cap individual positions
    weights = weights.clip(upper=MAX_POSITION)

    # Normalise so they sum to at most 1.0
    total = weights.sum()
    if total > 1.0:
        weights = weights / total

    return weights


# ── 4. MAIN FUNCTION — SIZE ONE DAY ──────────────────────────────────────────

def compute_positions(signals_today: pd.DataFrame,
                      features_df: pd.DataFrame,
                      date: pd.Timestamp,
                      max_positions: int = 10) -> pd.Series:
    """
    Full pipeline for one trading day:
        ML proba → Kelly weights → vol adjustment → normalise → final weights

    Args:
        signals_today:  DataFrame rows for today, must have 'symbol' and
                        'pred_proba' columns
        features_df:    Full historical features (for vol lookup)
        date:           Today's date
        max_positions:  Max concurrent positions

    Returns:
        Series of final position weights, index = symbol, values in [0, 1]
        Weights sum to ≤ 1.0. Missing symbols = 0 (not held).
    """
    if signals_today.empty:
        return pd.Series(dtype=float)

    pred_proba = signals_today.set_index("symbol")["pred_proba"]

    # Step 1: Kelly sizing (long positions)
    kelly_w = kelly_weights(pred_proba)

    if kelly_w.empty and not _cfg.ALLOW_SHORT_SELLING:
        return pd.Series(dtype=float)

    # Step 2: Volatility adjustment (long positions)
    if not kelly_w.empty:
        vol_w = volatility_adjust(kelly_w, features_df, date)
        final_long = normalise_weights(vol_w, max_positions)
    else:
        final_long = pd.Series(dtype=float)

    # Step 3: Short positions (if enabled)
    if _cfg.ALLOW_SHORT_SELLING:
        short_w = short_kelly_weights(pred_proba)
        if not short_w.empty:
            # Vol-adjust shorts the same way (using absolute values)
            short_abs = short_w.abs()
            short_vol = volatility_adjust(short_abs, features_df, date)
            # Cap individual positions
            short_vol = short_vol.clip(upper=SHORT_MAX_POSITION)
            # Keep only top N short candidates by weight
            short_vol = short_vol.nlargest(SHORT_MAX_POSITIONS)
            # Cap total short exposure
            total_short = short_vol.sum()
            if total_short > SHORT_MAX_TOTAL:
                short_vol = short_vol * (SHORT_MAX_TOTAL / total_short)
            short_final = -short_vol
            # Combine: long weights are positive, short weights are negative
            final_w = pd.concat([final_long, short_final])
            return final_w

    return final_long


# ── 5. DIAGNOSTIC — SEE WHAT THE SIZER IS DOING ──────────────────────────────

def diagnose_sizer(signals_df: pd.DataFrame,
                   features_df: pd.DataFrame,
                   n_sample_days: int = 5):
    """
    Print a sample of daily position weights so you can see the sizer in action.
    Run this after building the sizer to verify it's working as expected.

    What to look for:
      - High pred_proba stocks should get larger weights
      - High volatility stocks should get smaller weights than equally-
        confident but lower-volatility stocks
      - Total weight per day should be ≤ 1.0 (the rest is cash)
      - No single position should exceed MAX_POSITION (0.15)
    """
    print("\n========== POSITION SIZER DIAGNOSTIC ==========")
    print(f"  Kelly fraction:  {KELLY_FRACTION} (half-Kelly)")
    print(f"  Vol target:      {VOL_TARGET:.1%} daily vol per position")
    print(f"  Max position:    {MAX_POSITION:.1%} of portfolio")
    print(f"  Min probability: {MIN_PROB}")
    print("=" * 48)

    all_dates = sorted(signals_df["date"].unique())
    # Sample evenly from the date range
    step = max(1, len(all_dates) // n_sample_days)
    sample_dates = all_dates[::step][:n_sample_days]

    for date in sample_dates:
        today_signals = signals_df[signals_df["date"] == date]
        weights = compute_positions(today_signals, features_df, date)

        print(f"\n  Date: {date.date()} | Regime context:")
        print(f"  {'Symbol':<8} {'Pred Proba':>11} {'Kelly Raw':>10} "
              f"{'Final Weight':>13} {'Cash':>6}")
        print("  " + "-" * 52)

        for symbol, w in weights.sort_values(ascending=False).items():
            row = today_signals[today_signals["symbol"] == symbol]
            if row.empty:
                continue
            prob  = row["pred_proba"].iloc[0]
            # Recompute raw kelly for display
            b = KELLY_ODDS
            kelly_raw = ((prob * b - (1 - prob)) / b) * KELLY_FRACTION
            print(f"  {symbol:<8} {prob:>11.4f} {kelly_raw:>10.4f} {w:>12.4f}")

        cash = max(0, 1.0 - weights.sum())
        print(f"  {'':8} {'':11} {'':10} {'TOTAL:':>10} {weights.sum():>6.4f}")
        print(f"  {'':8} {'':11} {'':10} {'CASH:':>10} {cash:>6.4f}")

    print("\n========== END DIAGNOSTIC ==========\n")


def compute_kelly_weights(
    signals_df:  pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict:
    """
    Live inference wrapper called by execution/run_daily.py.

    Extracts the latest date from features_df, filters signals to that date,
    and runs the full Kelly + vol adjustment pipeline via compute_positions().

    Args:
        signals_df:  DataFrame with columns [symbol, pred_proba] — output of
                     generate_signals() for today.
        features_df: Full historical features DataFrame (flat, tz-naive) —
                     needed for volatility lookback in volatility_adjust().

    Returns:
        dict of {symbol: weight} for symbols above MIN_PROB threshold.
        Empty dict if no symbols qualify.
    """
    # Get the latest date available in features (= yesterday's close)
    latest_date = pd.to_datetime(features_df["time"]).max()

    # compute_positions() expects a 'date' column in features_df
    df = features_df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

    weights_series = compute_positions(
        signals_today=signals_df,
        features_df=df,
        date=latest_date,
    )

    return weights_series.to_dict()

# ── 6. EMPIRICAL KELLY ODDS CALIBRATION ──────────────────────────────────────

def calibrate_kelly_odds(signals_df: pd.DataFrame) -> float:
    """
    Compute Kelly odds empirically from actual signal quality data.

    Kelly odds = avg_win / avg_loss among stocks above the signal threshold.

    The signal quality table showed:
        Very High bucket: +4.42% avg 5d return
        Very Low bucket:  -3.53% avg 5d return

    But the correct calibration uses ACTUAL win/loss ratios from the
    pred_proba > MIN_PROB population, not extreme buckets.

    Args:
        signals_df: DataFrame with columns [pred_proba, forward_return_5d]

    Returns:
        Calibrated odds ratio (avg_win / avg_loss). Falls back to
        KELLY_ODDS default if insufficient data.
    """
    above = signals_df[signals_df["pred_proba"] > MIN_PROB].copy()

    if len(above) < 100:
        print(f"  Insufficient data for calibration ({len(above)} rows) "
              f"— using default odds {KELLY_ODDS}")
        return KELLY_ODDS

    winners = above[above["forward_return_5d"] > 0]["forward_return_5d"]
    losers  = above[above["forward_return_5d"] < 0]["forward_return_5d"].abs()

    if losers.empty or losers.mean() == 0:
        print(f"  No losing trades found — using default odds {KELLY_ODDS}")
        return KELLY_ODDS

    empirical_odds = winners.mean() / losers.mean()

    # Sanity bounds: odds below 0.5 or above 5.0 are suspect
    empirical_odds = max(0.5, min(5.0, empirical_odds))

    win_rate = len(winners) / len(above)
    print(f"  Kelly odds calibration:")
    print(f"    Avg winner: +{winners.mean():.2%}")
    print(f"    Avg loser:  -{losers.mean():.2%}")
    print(f"    Win rate:   {win_rate:.1%}")
    print(f"    Empirical odds: {empirical_odds:.2f} (default was {KELLY_ODDS})")

    return empirical_odds


# ── 7. MAIN — RUN STANDALONE TO TEST ─────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("Loading data for position sizer diagnostic...")

    signals  = pd.read_parquet("data/processed/ml_signals.parquet")
    features = pd.read_parquet("data/processed/features_daily.parquet")

    # Normalise dates
    signals["date"]  = pd.to_datetime(signals["date"]).dt.normalize().dt.tz_localize(None)
    features = features.reset_index()
    features["date"] = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None)

    print(f"  Signals:  {len(signals):,} rows | "
          f"{signals['date'].min().date()} → {signals['date'].max().date()}")
    print(f"  Features: {len(features):,} rows")

    # Run the diagnostic on 5 sample days spread across the full period
    diagnose_sizer(signals, features, n_sample_days=5)

    # Summary statistics across all days
    print("Computing summary statistics across all trading days...")
    all_dates   = sorted(signals["date"].unique())
    all_weights = []

    for date in all_dates:
        today = signals[signals["date"] == date]
        w     = compute_positions(today, features, date)
        if not w.empty:
            all_weights.append({
                "date":          date,
                "n_positions":   len(w),
                "total_invested":w.sum(),
                "cash":          max(0, 1.0 - w.sum()),
                "max_position":  w.max(),
                "avg_position":  w.mean(),
            })

    summary = pd.DataFrame(all_weights)
    print(f"\n========== SIZER SUMMARY ({len(summary)} trading days) ==========")
    print(f"  Avg positions per day:    {summary['n_positions'].mean():.1f}")
    print(f"  Avg capital invested:     {summary['total_invested'].mean():.1%}")
    print(f"  Avg cash held:            {summary['cash'].mean():.1%}")
    print(f"  Avg largest position:     {summary['max_position'].mean():.1%}")
    print(f"  Avg position size:        {summary['avg_position'].mean():.1%}")
    print(f"  Days at max positions:    "
          f"{(summary['n_positions'] >= 10).sum()} "
          f"({(summary['n_positions'] >= 10).mean():.1%})")
    print("=" * 52)
    print("\nPosition sizer working correctly.")
    print("Next step: run risk/portfolio_optimizer.py")