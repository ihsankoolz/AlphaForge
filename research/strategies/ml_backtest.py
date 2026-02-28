import pandas as pd
import numpy as np
import sys
import os

# Project root — needed for risk.portfolio_optimiser
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# research/ folder — needed for strategies.backtest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.backtest import calculate_metrics
from risk.portfolio_optimiser import optimise_portfolio

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────

def load_data():
    features = pd.read_parquet("data/processed/features_daily.parquet")
    signals  = pd.read_parquet("data/processed/ml_signals.parquet")
    regimes  = pd.read_parquet("data/processed/regime_labels.parquet")

    features = features.reset_index()
    features["date"] = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None)

    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize().dt.tz_localize(None)

    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None).normalize()
    regimes = regimes.reset_index().rename(columns={"index": "date"})

    return features, signals, regimes

# ── 2. RUN ML BACKTEST (with Risk Layer) ─────────────────────────────────────

def run_ml_backtest(features, signals, regimes,
                    initial_capital=100_000, transaction_cost=0.001,
                    top_n=10, min_signal=0.35):
    """
    Trade using ML signals routed through the full risk layer:
        ML probability scores
            → Kelly Criterion sizing (position_sizer.py)
            → Mean-variance optimization (portfolio_optimiser.py)
            → Final portfolio weights

    Regime rules (same as original regime switcher that produced 186%):
      Bull   → full Kelly + Markowitz optimization
      Bear   → Kelly + Markowitz with tighter 70% position caps
      Choppy → pure cash (0% invested, 100% cash)
               SPY rotation was tested and destroyed returns in 2022
               The regime switcher's 186% came from pure cash in choppy
    """
    # Build returns matrix (date x symbol)
    returns_wide = features.pivot_table(
        index="date", columns="symbol", values="return_1d"
    )
    returns_wide.index = pd.to_datetime(returns_wide.index).tz_localize(None).normalize()
    returns_wide = returns_wide.fillna(0)

    # Regime index
    regimes_idx = regimes.set_index("date")["regime"]
    regimes_idx.index = pd.to_datetime(regimes_idx.index).tz_localize(None).normalize()

    # Signals — keep as long-format for the optimizer
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize().dt.tz_localize(None)
    all_signal_dates = sorted(signals["date"].unique())

    # Only trade during ML signal period (starts Aug 2021 after 18-month warmup)
    # Running before this period adds ~383 dead days with no signals
    signal_start = signals["date"].min()
    common_dates = sorted(
        returns_wide.index
        .intersection(regimes_idx.index)
    )
    common_dates = [d for d in common_dates if d >= signal_start]

    portfolio_value   = initial_capital
    portfolio_history = []
    prev_positions    = pd.Series(0.0, index=returns_wide.columns)

    for date in common_dates:
        regime   = regimes_idx.get(date, "bull")
        day_rets = returns_wide.loc[date]

        # CHOPPY: pure cash — verified best behavior from regime switcher
        if regime == "choppy":
            portfolio_history.append({
                "date": date, "portfolio_value": portfolio_value,
                "daily_return": 0.0, "regime": regime, "num_positions": 0
            })
            prev_positions = pd.Series(0.0, index=returns_wide.columns)
            continue

        # LOOKAHEAD BIAS PREVENTION: use signals from strictly before today
        prior_signal_dates = [d for d in all_signal_dates if d < date]
        if not prior_signal_dates:
            portfolio_history.append({
                "date": date, "portfolio_value": portfolio_value,
                "daily_return": 0.0, "regime": regime, "num_positions": 0
            })
            continue

        signal_date   = prior_signal_dates[-1]
        signals_today = signals[signals["date"] == signal_date].copy()

        # KELLY + MARKOWITZ RISK LAYER
        final_weights = optimise_portfolio(
            signals_today = signals_today,
            features_df   = features,
            date          = date,
            regime        = regime
        )

        # Map weights back to full symbol universe
        positions = pd.Series(0.0, index=returns_wide.columns)
        for symbol, weight in final_weights.items():
            if symbol in positions.index:
                positions[symbol] = weight

        # Transaction costs
        position_changes = (positions - prev_positions).abs()
        cost = position_changes.sum() * transaction_cost

        # P&L
        pnl = (positions * day_rets).sum()
        portfolio_value = portfolio_value * (1 + pnl - cost)

        portfolio_history.append({
            "date":            date,
            "portfolio_value": portfolio_value,
            "daily_return":    pnl - cost,
            "regime":          regime,
            "num_positions":   len(final_weights)
        })

        prev_positions = positions

    return pd.DataFrame(portfolio_history).set_index("date")

# ── 3. BENCHMARK COMPARISON ──────────────────────────────────────────────────

def compare_strategies(ml_portfolio):
    strategies = {
        "ML + Risk Layer (v3)":    ml_portfolio,
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
        ret   = port["daily_return"]
        start = port["portfolio_value"].iloc[0]
        end   = port["portfolio_value"].iloc[-1]

        total  = (end / start) - 1
        years  = len(ret) / 252
        ann    = (1 + total) ** (1 / years) - 1
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0

        cum = (1 + ret).cumprod()
        dd  = ((cum - cum.cummax()) / cum.cummax()).min()
        wr  = (ret > 0).mean()

        print(f"  {name:<28} {total:>9.2%} {ann:>7.2%} {sharpe:>8.2f} {dd:>7.2%} {wr:>7.2%}")
        results[name] = {"total": total, "ann": ann, "sharpe": sharpe, "max_dd": dd}

    print(f"  {'-'*71}")
    print(f"  {'SPY Buy & Hold (approx)':<28} {'~85-90%':>10} {'~14%':>8} {'~0.80':>8} {'~-34%':>8} {'  N/A':>8}")
    print(f"{'='*75}\n")

    return results

# ── 4. REGIME BREAKDOWN ──────────────────────────────────────────────────────

def regime_breakdown(ml_portfolio):
    print(f"\n{'='*55}")
    print(f"  ML + Risk Layer — Performance by Regime")
    print(f"{'='*55}")

    for regime in ["bull", "bear", "choppy"]:
        subset = ml_portfolio[ml_portfolio["regime"] == regime]
        if len(subset) < 5:
            continue
        ret    = subset["daily_return"]
        total  = (ret + 1).prod() - 1
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

    print("Running ML backtest with full risk layer (Kelly + Markowitz)...")
    print("  Note: this is slower than v2 — optimizer runs once per trading day\n")
    ml_portfolio = run_ml_backtest(features, signals, regimes)

    print(f"  Trading days:        {len(ml_portfolio)}")
    print(f"  Days with positions: {(ml_portfolio['num_positions'] > 0).sum()}")
    print(f"  Avg positions/day:   {ml_portfolio['num_positions'].mean():.1f}\n")

    regime_breakdown(ml_portfolio)
    results = compare_strategies(ml_portfolio)

    ml_portfolio.to_parquet("data/processed/backtest_ml.parquet")
    print("Saved: data/processed/backtest_ml.parquet")
    print("\nWeek 7 complete. Next: Week 8 — Execution Layer & Paper Trading")