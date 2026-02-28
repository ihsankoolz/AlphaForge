import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.backtest import calculate_metrics

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────

def load_data():
    features = pd.read_parquet("data/processed/features_daily.parquet")
    signals  = pd.read_parquet("data/processed/ml_signals.parquet")
    regimes  = pd.read_parquet("data/processed/regime_labels.parquet")

    # Normalize all indexes to tz-naive date
    features = features.reset_index()
    features["date"] = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None)

    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize().dt.tz_localize(None)

    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None).normalize()
    regimes = regimes.reset_index().rename(columns={"index": "date"})

    return features, signals, regimes

# ── 2. RUN ML BACKTEST ───────────────────────────────────────────────────────

def run_ml_backtest(features, signals, regimes,
                    initial_capital=100_000, transaction_cost=0.001,
                    top_n=10, min_signal=0.35):
    """
    Trade using ML predicted probabilities as signals.

    Key differences from rule-based backtest:
    - Signal strength = predicted probability of being top quartile
    - Position sizing is SIGNAL-WEIGHTED not equal weight
      → higher confidence positions get more capital
    - Still apply regime filter: skip choppy days (no edge)

    Signal-weighted sizing means if we hold 3 stocks with signals
    0.8, 0.6, 0.4 the weights are proportional: 44%, 33%, 22%.
    This is more sophisticated than naive equal weighting.
    """
    # Build returns matrix (date × symbol)
    returns_wide = features.pivot_table(
        index="date", columns="symbol", values="return_1d"
    )

    # Build signal matrix from ML predictions
    signals_wide = signals.pivot_table(
        index="date", columns="symbol", values="signal"
    ).fillna(0)

    # Normalize regime index
    regimes = regimes.set_index("date")["regime"]

    # Align all to tz-naive dates
    returns_wide.index = pd.to_datetime(returns_wide.index).tz_localize(None).normalize()
    signals_wide.index = pd.to_datetime(signals_wide.index).tz_localize(None).normalize()
    regimes.index      = pd.to_datetime(regimes.index).tz_localize(None).normalize()

    # CRITICAL: shift signals forward 1 day — same as rule-based backtest
    signals_wide = signals_wide.shift(1)

    common_dates = (
        returns_wide.index
        .intersection(signals_wide.index)
        .intersection(regimes.index)
    )

    returns_wide = returns_wide.loc[common_dates].fillna(0)
    signals_wide = signals_wide.loc[common_dates].fillna(0)

    portfolio_value = initial_capital
    portfolio_history = []
    prev_positions = pd.Series(0.0, index=returns_wide.columns)

    for date in common_dates:
        regime     = regimes.get(date, "bull")
        day_sigs   = signals_wide.loc[date]
        day_rets   = returns_wide.loc[date]

        # Skip choppy regime — no edge, preserve capital
        if regime == "choppy":
            portfolio_history.append({
                "date": date, "portfolio_value": portfolio_value,
                "daily_return": 0.0, "regime": regime,
                "num_positions": 0
            })
            prev_positions = pd.Series(0.0, index=returns_wide.columns)
            continue

        # Select top N signals above minimum threshold
        candidates = day_sigs[day_sigs > min_signal].nlargest(top_n)

        positions = pd.Series(0.0, index=returns_wide.columns)

        if len(candidates) > 0:
            # Signal-weighted position sizing
            # Stocks with higher ML confidence get proportionally more capital
            total_signal = candidates.sum()
            for symbol, sig_strength in candidates.items():
                positions[symbol] = sig_strength / total_signal

        # Transaction costs
        position_changes = (positions - prev_positions).abs()
        cost = position_changes.sum() * transaction_cost

        pnl = (positions * day_rets).sum()
        portfolio_value = portfolio_value * (1 + pnl - cost)

        portfolio_history.append({
            "date":            date,
            "portfolio_value": portfolio_value,
            "daily_return":    pnl - cost,
            "regime":          regime,
            "num_positions":   len(candidates)
        })

        prev_positions = positions

    return pd.DataFrame(portfolio_history).set_index("date")

# ── 3. BENCHMARK COMPARISON ──────────────────────────────────────────────────

def compare_strategies(ml_portfolio):
    """
    Load all prior strategy results and print a side-by-side comparison.
    This is the table you'll put in your portfolio/README.
    """
    strategies = {
        "ML Signal (XGBoost)":     ml_portfolio,
        "Regime Switcher":         pd.read_parquet("data/processed/backtest_regime_switcher.parquet"),
        "Momentum (standalone)":   pd.read_parquet("data/processed/backtest_momentum.parquet"),
        "Mean Reversion":          pd.read_parquet("data/processed/backtest_mean_reversion.parquet"),
    }

    print(f"\n{'='*75}")
    print(f"  {'STRATEGY COMPARISON':^71}")
    print(f"{'='*75}")
    print(f"  {'Strategy':<28} {'Total Ret':>10} {'Ann Ret':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    print(f"  {'-'*71}")

    results = {}
    for name, port in strategies.items():
        ret    = port["daily_return"]
        start  = port["portfolio_value"].iloc[0]
        end    = port["portfolio_value"].iloc[-1]

        total  = (end / start) - 1
        years  = len(ret) / 252
        ann    = (1 + total) ** (1 / years) - 1
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0

        cum    = (1 + ret).cumprod()
        dd     = ((cum - cum.cummax()) / cum.cummax()).min()
        wr     = (ret > 0).mean()

        print(f"  {name:<28} {total:>9.2%} {ann:>7.2%} {sharpe:>8.2f} {dd:>7.2%} {wr:>7.2%}")
        results[name] = {"total": total, "ann": ann, "sharpe": sharpe, "max_dd": dd}

    # SPY benchmark — buy and hold
    print(f"  {'-'*71}")
    print(f"  {'SPY Buy & Hold (approx)':<28} {'~85-90%':>10} {'~14%':>8} {'~0.80':>8} {'~-34%':>8} {'  N/A':>8}")
    print(f"{'='*75}\n")

    return results

# ── 4. REGIME BREAKDOWN ──────────────────────────────────────────────────────

def regime_breakdown(ml_portfolio):
    """How did the ML strategy perform within each regime it traded in?"""
    print(f"\n{'='*55}")
    print(f"  ML Strategy — Performance by Regime")
    print(f"{'='*55}")

    for regime in ["bull", "bear", "choppy"]:
        subset = ml_portfolio[ml_portfolio["regime"] == regime]
        if len(subset) < 5:
            continue
        ret = subset["daily_return"]
        total = (ret + 1).prod() - 1
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
        print(f"  {regime.upper()} ({len(subset)} days):")
        print(f"    Total Return: {total:.2%}  |  Sharpe: {sharpe:.2f}")
    print(f"{'='*55}\n")

# ── 5. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    features, signals, regimes = load_data()
    print(f"  ML signals: {len(signals):,} rows covering "
          f"{signals['date'].min().date()} → {signals['date'].max().date()}\n")

    print("Running ML backtest...")
    ml_portfolio = run_ml_backtest(features, signals, regimes)

    print(f"  Trading days: {len(ml_portfolio)}")
    print(f"  Days with positions: {(ml_portfolio['num_positions'] > 0).sum()}")
    print(f"  Avg positions per day: {ml_portfolio['num_positions'].mean():.1f}\n")

    regime_breakdown(ml_portfolio)
    results = compare_strategies(ml_portfolio)

    ml_portfolio.to_parquet("data/processed/backtest_ml.parquet")
    print("Saved: data/processed/backtest_ml.parquet")