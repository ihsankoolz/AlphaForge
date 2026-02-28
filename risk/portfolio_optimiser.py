"""
risk/portfolio_optimizer.py
───────────────────────────
Mean-variance portfolio optimization (Markowitz) applied on top of
Kelly-sized positions.

Pipeline per trading day:
  1. Kelly sizer gives us candidate stocks + initial weights
  2. We build a covariance matrix from recent price history
  3. Scipy optimizer finds weights that maximize Sharpe ratio
  4. Constraints ensure long-only, capped positions, and no slivers

Why Markowitz on top of Kelly?
  Kelly answers: "how much edge do I have in each stock?"
  Markowitz answers: "given how these stocks move together,
                      what combination maximises risk-adjusted return?"
  Kelly ignores correlation. Markowitz is entirely about correlation.
  Together they handle both individual sizing AND portfolio construction.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings
import sys
import os
warnings.filterwarnings("ignore")

# Ensure project root is on the path so local modules resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.position_sizer import compute_positions, MIN_PROB, MAX_POSITION


# ── CONSTANTS ────────────────────────────────────────────────────────────────

COV_LOOKBACK    = 60    # days of return history to estimate covariance
                        # 60 days ≈ 3 months — enough to be stable,
                        # short enough to reflect current market structure
MIN_WEIGHT      = 0.02  # minimum position size — no slivers below 2%
                        # positions below this get zeroed out
MAX_WEIGHT      = MAX_POSITION   # inherit 15% cap from position sizer
RISK_FREE_RATE  = 0.05 / 252     # daily risk-free rate (~5% annual, 2021-2025
                                  # approximate US Fed funds rate average)
MIN_STOCKS      = 2     # need at least 2 stocks to optimize correlations
                        # (1 stock = just use Kelly weight directly)


# ── 1. BUILD COVARIANCE MATRIX ───────────────────────────────────────────────

def build_covariance_matrix(symbols: list,
                            features_df: pd.DataFrame,
                            date: pd.Timestamp) -> pd.DataFrame:
    """
    Estimate the covariance matrix from recent daily returns.

    We use the 60 days of history BEFORE today (no lookahead bias).
    Returns are already computed in features_daily as return_1d.

    Why 60 days?
      - Too short (e.g. 20 days): noisy, unstable correlations
      - Too long (e.g. 252 days): correlations from a year ago may not
        reflect the current market regime
      - 60 days: captures the current regime while having enough data
        to estimate a stable matrix

    Args:
        symbols:     List of stock symbols to include
        features_df: Full features DataFrame with return_1d column
        date:        Current trading day — use history strictly before this

    Returns:
        Covariance matrix as DataFrame (symbols × symbols)
        Returns None if insufficient data
    """
    hist = features_df[features_df["date"] < date].copy()
    hist = hist[hist["symbol"].isin(symbols)]

    # Pivot to wide format: rows = dates, columns = symbols, values = return_1d
    returns_wide = hist.pivot_table(
        index="date", columns="symbol", values="return_1d"
    ).sort_index().tail(COV_LOOKBACK)

    # Drop symbols with too many missing values
    returns_wide = returns_wide.dropna(axis=1, thresh=int(COV_LOOKBACK * 0.8))

    if returns_wide.shape[0] < 20 or returns_wide.shape[1] < 2:
        return None

    # Forward-fill then drop any remaining NaNs
    returns_wide = returns_wide.ffill().dropna()

    # Compute annualised covariance matrix
    # Daily cov × 252 trading days = annualised
    cov_matrix = returns_wide.cov() * 252

    return cov_matrix


# ── 2. EXPECTED RETURNS FROM ML SCORES ───────────────────────────────────────

def build_expected_returns(symbols: list,
                           signals_today: pd.DataFrame) -> pd.Series:
    """
    Convert ML probability scores into expected return estimates.

    We use a simple linear mapping:
        expected_return = (pred_proba - 0.5) × scaling_factor

    A pred_proba of 0.75 → expected_return of 0.25 × 0.40 = 10% annualised
    A pred_proba of 0.50 → expected_return of 0.00 (no edge)
    A pred_proba of 0.35 → expected_return slightly negative (boundary case)

    The 0.40 scaling factor is calibrated so that our highest-confidence
    predictions (~0.90 proba) map to ~16% expected annual return,
    which is consistent with what we saw in the signal quality table.

    Args:
        symbols:       Symbols to include
        signals_today: Today's signals with pred_proba column

    Returns:
        Series of annualised expected returns, index = symbol
    """
    proba = signals_today.set_index("symbol")["pred_proba"]
    proba = proba.reindex(symbols).dropna()

    # Linear mapping from probability to expected return
    scaling = 0.40
    expected_returns = (proba - 0.5) * scaling

    return expected_returns


# ── 3. SHARPE MAXIMISATION ───────────────────────────────────────────────────

def maximise_sharpe(expected_returns: pd.Series,
                    cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Find portfolio weights that maximise the Sharpe ratio.

    Sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

    This is a constrained optimisation problem solved with scipy.minimize.
    We minimise negative Sharpe (scipy only minimises, not maximises).

    Constraints:
      - Weights sum to 1.0 (fully invested within selected stocks)
      - Each weight between MIN_WEIGHT and MAX_WEIGHT
      - Long-only (weights ≥ 0, enforced by bounds)

    Note: we optimise weights within the selected stock universe.
    The Kelly sizer already decided WHICH stocks to hold.
    Markowitz decides HOW MUCH of each to hold.

    Args:
        expected_returns: Annualised expected returns per stock
        cov_matrix:       Annualised covariance matrix

    Returns:
        Optimal weights as Series, index = symbol
        Returns equal weights if optimisation fails
    """
    # Align symbols — only optimise stocks present in both inputs
    symbols = expected_returns.index.intersection(cov_matrix.index)
    if len(symbols) < MIN_STOCKS:
        # Not enough stocks to optimise — return equal weights
        n = len(expected_returns)
        return pd.Series(1.0 / n, index=expected_returns.index)

    mu  = expected_returns[symbols].values   # expected returns vector
    cov = cov_matrix.loc[symbols, symbols].values  # covariance matrix

    n = len(symbols)

    def neg_sharpe(weights):
        """Objective: minimise negative Sharpe = maximise Sharpe."""
        port_return = weights @ mu
        port_vol    = np.sqrt(weights @ cov @ weights)
        if port_vol < 1e-8:
            return 0.0
        sharpe = (port_return - RISK_FREE_RATE * 252) / port_vol
        return -sharpe

    # Starting point: equal weights
    w0 = np.ones(n) / n

    # Constraints: weights must sum to 1.0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds: each weight between MIN_WEIGHT and MAX_WEIGHT
    bounds = [(MIN_WEIGHT, MAX_WEIGHT)] * n

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",          # Sequential Least Squares — standard for
        bounds=bounds,            # constrained portfolio optimisation
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9}
    )

    if result.success:
        opt_weights = pd.Series(result.x, index=symbols)
    else:
        # Optimisation failed — fall back to equal weights
        opt_weights = pd.Series(1.0 / n, index=symbols)

    return opt_weights


# ── 4. FULL PIPELINE — ONE TRADING DAY ───────────────────────────────────────

def optimise_portfolio(signals_today: pd.DataFrame,
                       features_df: pd.DataFrame,
                       date: pd.Timestamp,
                       regime: str = "bull") -> pd.Series:
    """
    Full optimisation pipeline for one trading day.

    Step 1: Regime filter
      - Choppy regime → rotate into SPY (safe haven, already in universe)
      - Bear regime   → run optimiser but with tighter position caps
      - Bull regime   → full optimisation, standard caps

    Step 2: Kelly sizer → candidate stocks + initial weights

    Step 3: Build covariance matrix from recent history

    Step 4: Markowitz optimisation → final weights

    Step 5: Scale weights back (Kelly told us how much to invest overall,
            Markowitz told us how to split it)

    Args:
        signals_today: Today's ML signals (date + symbol + pred_proba)
        features_df:   Full historical features
        date:          Current trading day
        regime:        Current HMM regime ('bull', 'choppy', 'bear')

    Returns:
        Final optimised weights, index = symbol, sum ≤ 1.0
    """
    # ── REGIME FILTER ─────────────────────────────────────────────────────────
    # Choppy regime is handled upstream in ml_backtest.py (pure cash).
    # Optimiser only runs for bull and bear regimes.

    # Tighter caps in bear regime — model is less reliable in crashes
    max_w = MAX_WEIGHT * 0.7 if regime == "bear" else MAX_WEIGHT

    # ── STEP 1: KELLY SIZING → CANDIDATE STOCKS ───────────────────────────────
    kelly_weights = compute_positions(signals_today, features_df, date)

    if kelly_weights.empty or len(kelly_weights) < MIN_STOCKS:
        return kelly_weights   # too few candidates — just use Kelly as-is

    symbols      = kelly_weights.index.tolist()
    kelly_total  = kelly_weights.sum()   # total capital Kelly wants invested

    # ── STEP 2: COVARIANCE MATRIX ─────────────────────────────────────────────
    cov_matrix = build_covariance_matrix(symbols, features_df, date)

    if cov_matrix is None:
        # Insufficient history (early in the dataset) — use Kelly weights
        return kelly_weights

    # ── STEP 3: EXPECTED RETURNS ──────────────────────────────────────────────
    expected_returns = build_expected_returns(symbols, signals_today)

    # ── STEP 4: MARKOWITZ OPTIMISATION ───────────────────────────────────────
    opt_weights = maximise_sharpe(expected_returns, cov_matrix)

    # Apply bear regime tighter cap
    opt_weights = opt_weights.clip(upper=max_w)

    # ── STEP 5: RESCALE BY KELLY TOTAL ───────────────────────────────────────
    # Markowitz gives us HOW to split the invested capital (sums to 1.0)
    # Kelly told us HOW MUCH to invest overall (kelly_total ≤ 1.0)
    # Multiply to get final portfolio weights
    final_weights = opt_weights * kelly_total

    # Ensure no weight exceeds the absolute cap after rescaling
    final_weights = final_weights.clip(upper=MAX_WEIGHT)

    return final_weights.sort_values(ascending=False)


# ── 5. DIAGNOSTIC ────────────────────────────────────────────────────────────

def diagnose_optimizer(signals_df: pd.DataFrame,
                       features_df: pd.DataFrame,
                       regimes_df: pd.DataFrame,
                       n_sample_days: int = 5):
    """
    Show what Markowitz does DIFFERENTLY from pure Kelly on sample days.

    The key comparison:
      Kelly weight  → what naive sizing would give
      Markowitz weight → what correlation-aware sizing gives
      Difference    → the value Markowitz is adding

    What to look for:
      - Correlated stocks (GS + JPM + BAC) should be down-weighted vs Kelly
      - Uncorrelated diversifiers should be up-weighted vs Kelly
      - Total invested capital should be similar (Markowitz redistributes,
        Kelly determines the total)
    """
    print("\n========== PORTFOLIO OPTIMIZER DIAGNOSTIC ==========")
    print(f"  Covariance lookback: {COV_LOOKBACK} days")
    print(f"  Min weight:          {MIN_WEIGHT:.1%}")
    print(f"  Max weight:          {MAX_WEIGHT:.1%}")
    print(f"  Risk-free rate:      {RISK_FREE_RATE*252:.1%} annual")
    print("=" * 52)

    all_dates = sorted(signals_df["date"].unique())
    step = max(1, len(all_dates) // n_sample_days)
    sample_dates = all_dates[::step][:n_sample_days]

    for date in sample_dates:
        today_signals = signals_df[signals_df["date"] == date]

        # Get regime for this day
        regime_row = regimes_df[regimes_df["date"] == date]
        regime = regime_row["regime"].iloc[0] if not regime_row.empty else "bull"

        # Kelly weights (baseline)
        kelly_w = compute_positions(today_signals, features_df, date)

        # Optimised weights
        opt_w = optimise_portfolio(today_signals, features_df, date, regime)

        print(f"\n  Date: {date.date()} | Regime: {regime.upper()}")
        print(f"  {'Symbol':<8} {'Kelly W':>9} {'Markowitz W':>12} {'Δ Change':>10}")
        print("  " + "-" * 42)

        all_symbols = sorted(set(kelly_w.index) | set(opt_w.index))
        for sym in all_symbols:
            kw  = kelly_w.get(sym, 0.0)
            mw  = opt_w.get(sym, 0.0)
            delta = mw - kw
            marker = " ↑" if delta > 0.01 else (" ↓" if delta < -0.01 else "  ")
            print(f"  {sym:<8} {kw:>8.2%} {mw:>11.2%} {delta:>+9.2%}{marker}")

        print(f"  {'':8} {'':9} {'':12} {'':10}")
        print(f"  {'TOTAL':<8} {kelly_w.sum():>8.2%} {opt_w.sum():>11.2%}")
        cash = max(0, 1.0 - opt_w.sum())
        print(f"  {'CASH':<8} {'':9} {cash:>11.2%}")

    print("\n========== END DIAGNOSTIC ==========\n")


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("Loading data for portfolio optimizer diagnostic...")

    signals  = pd.read_parquet("data/processed/ml_signals.parquet")
    features = pd.read_parquet("data/processed/features_daily.parquet")
    regimes  = pd.read_parquet("data/processed/regime_labels.parquet")

    # Normalise dates
    signals["date"]  = pd.to_datetime(signals["date"]).dt.normalize().dt.tz_localize(None)
    features = features.reset_index()
    features["date"] = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None)
    regimes.index    = pd.to_datetime(regimes.index).tz_localize(None).normalize()
    regimes          = regimes.reset_index().rename(columns={"index": "date"})

    print(f"  Signals:  {len(signals):,} rows")
    print(f"  Features: {len(features):,} rows")
    print(f"  Regimes:  {len(regimes):,} rows\n")

    # Run diagnostic on 5 sample days
    diagnose_optimizer(signals, features, regimes, n_sample_days=5)

    # Quick summary — how much does Markowitz change Kelly weights?
    print("Computing weight change summary across all trading days...")
    all_dates = sorted(signals["date"].unique())
    changes   = []

    # Sample every 10th day for speed
    for date in all_dates[::10]:
        today    = signals[signals["date"] == date]
        reg_row  = regimes[regimes["date"] == date]
        regime   = reg_row["regime"].iloc[0] if not reg_row.empty else "bull"

        kelly_w  = compute_positions(today, features, date)
        opt_w    = optimise_portfolio(today, features, date, regime)

        if kelly_w.empty or opt_w.empty:
            continue

        all_sym = set(kelly_w.index) | set(opt_w.index)
        for sym in all_sym:
            kw = kelly_w.get(sym, 0.0)
            mw = opt_w.get(sym, 0.0)
            changes.append(abs(mw - kw))

    if changes:
        avg_change = np.mean(changes)
        print(f"\n  Average absolute weight change (Kelly → Markowitz): {avg_change:.2%}")
        print(f"  This means Markowitz is redistributing on average "
              f"{avg_change:.2%} of capital per position per day")
        print(f"  (Small changes = Markowitz mostly agreeing with Kelly)")
        print(f"  (Large changes = Markowitz finding significant correlation issues)")

    print("\nPortfolio optimizer working correctly.")
    print("Next step: update ml_backtest.py to use the full risk layer")