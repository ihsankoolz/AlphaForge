import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum import generate_signals as momentum_signals
from strategies.mean_reversion import generate_signals as mr_signals
from strategies.backtest import run_backtest, calculate_metrics

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────

def load_all_data():
    features  = pd.read_parquet("data/processed/features_daily.parquet")
    regimes   = pd.read_parquet("data/processed/regime_labels.parquet")
    mom_port  = pd.read_parquet("data/processed/backtest_momentum.parquet")
    mr_port   = pd.read_parquet("data/processed/backtest_mean_reversion.parquet")
    return features, regimes, mom_port, mr_port

# ── 2. ANALYSE STRATEGY PERFORMANCE BY REGIME ────────────────────────────────

def analyse_by_regime(portfolio, regimes, strategy_name):
    """
    Join daily portfolio returns with regime labels and break down
    performance metrics for each regime separately.

    This answers: does momentum actually work better in bull markets?
    Does mean reversion work better in choppy markets?
    """
    # Normalize both indexes to plain dates for joining
    portfolio = portfolio.copy()
    portfolio.index = pd.to_datetime(portfolio.index).tz_localize(None).normalize()

    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None).normalize()

    joined = portfolio.join(regimes[["regime"]], how="inner")
    
    if joined["regime"].isna().all():
        print(f"WARNING: No regime data joined for {strategy_name}")
        return

    print(f"\n{'='*55}")
    print(f"  {strategy_name} — Performance by Regime")
    print(f"{'='*55}")

    overall_sharpe = _sharpe(joined["daily_return"])
    overall_return = (joined["daily_return"] + 1).prod() - 1
    print(f"  Overall: Total Return {overall_return:.2%}, Sharpe {overall_sharpe:.2f}")
    print(f"{'-'*55}")

    for regime in ["bull", "choppy", "bear"]:
        subset = joined[joined["regime"] == regime]
        if len(subset) < 10:
            print(f"  {regime.upper()}: insufficient data ({len(subset)} days)")
            continue

        returns = subset["daily_return"]
        total_ret  = (returns + 1).prod() - 1
        sharpe     = _sharpe(returns)
        win_rate   = (returns > 0).mean()
        max_dd     = _max_drawdown(returns)
        avg_daily  = returns.mean()

        print(f"  {regime.upper()} ({len(subset)} days):")
        print(f"    Total Return:   {total_ret:.2%}")
        print(f"    Avg Daily:      {avg_daily:.4%}")
        print(f"    Sharpe:         {sharpe:.2f}")
        print(f"    Max Drawdown:   {max_dd:.2%}")
        print(f"    Win Rate:       {win_rate:.2%}")

    print(f"{'='*55}\n")
    return joined

# ── 3. BUILD REGIME-AWARE COMBINED STRATEGY ──────────────────────────────────

def run_regime_switcher(features, regimes, initial_capital=100_000, transaction_cost=0.001):
    """
    The regime-aware strategy:
    - Bull regime   → run momentum signals
    - Choppy regime → run mean reversion signals  
    - Bear regime   → go to cash (no positions)

    Going to cash in bear regimes is the key risk management insight.
    Momentum strategies get destroyed in high-volatility crashes.
    Better to miss some gains than to take a -30% drawdown.
    """
    print("Generating signals for regime switcher...")
    mom_signals_df = momentum_signals(features)
    mr_signals_df  = mr_signals(features)

    # Reshape to wide format (date × symbol)
    returns_wide = features["return_1d"].unstack(level="symbol")
    returns_wide.index = pd.to_datetime(returns_wide.index).tz_localize(None).normalize()  # ← add this

    mom_wide = mom_signals_df["signal"].unstack(level="symbol").fillna(0).shift(1)
    mom_wide.index = pd.to_datetime(mom_wide.index).tz_localize(None).normalize()          # ← add this

    mr_wide  = mr_signals_df["signal"].unstack(level="symbol").fillna(0).shift(1)
    mr_wide.index = pd.to_datetime(mr_wide.index).tz_localize(None).normalize()            # ← add this

    # Normalize regime index to plain date
    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None).normalize()  

    # Align all data to common dates
    common_dates = (
        returns_wide.index
        .intersection(mom_wide.index)
        .intersection(mr_wide.index)
        .intersection(regimes.index)
    )
    returns_wide = returns_wide.loc[common_dates].fillna(0)
    mom_wide     = mom_wide.loc[common_dates].fillna(0)
    mr_wide      = mr_wide.loc[common_dates].fillna(0)

    portfolio_value = initial_capital
    portfolio_history = []
    prev_positions = pd.Series(0.0, index=returns_wide.columns)

    for date in common_dates:
        regime = regimes.loc[date, "regime"] if date in regimes.index else "bull"

        # Select signals based on current regime
        if regime == "bull":
            day_signals = mom_wide.loc[date]
            active_strategy = "momentum"
        elif regime == "bear":
            day_signals = mom_wide.loc[date]   # keep momentum in high-vol periods
            active_strategy = "momentum"
        else:  # choppy → cash, neither strategy works here
            day_signals = pd.Series(0.0, index=returns_wide.columns)
            active_strategy = "cash"

        day_returns = returns_wide.loc[date]

        # Build positions — top 10 buy signals, equal weight
        buy_candidates = day_signals[day_signals > 0].nlargest(10)
        positions = pd.Series(0.0, index=returns_wide.columns)
        if len(buy_candidates) > 0:
            weight = 1.0 / len(buy_candidates)
            for symbol in buy_candidates.index:
                positions[symbol] = weight

        # Transaction costs on position changes
        position_changes = (positions - prev_positions).abs()
        cost = position_changes.sum() * transaction_cost

        pnl = (positions * day_returns).sum()
        portfolio_value = portfolio_value * (1 + pnl - cost)

        portfolio_history.append({
            "date":            date,
            "portfolio_value": portfolio_value,
            "daily_return":    pnl - cost,
            "regime":          regime,
            "strategy":        active_strategy,
            "num_positions":   len(buy_candidates)
        })

        prev_positions = positions

    return pd.DataFrame(portfolio_history).set_index("date")

# ── 4. REGIME SWITCHER METRICS ────────────────────────────────────────────────

def summarise_switcher(portfolio):
    returns = portfolio["daily_return"]
    start   = portfolio["portfolio_value"].iloc[0]
    end     = portfolio["portfolio_value"].iloc[-1]

    total_ret  = (end / start) - 1
    years      = len(returns) / 252
    ann_ret    = (1 + total_ret) ** (1 / years) - 1
    sharpe     = _sharpe(returns)
    max_dd     = _max_drawdown(returns)
    win_rate   = (returns > 0).mean()

    # How many days in each regime/strategy
    regime_counts = portfolio["strategy"].value_counts()

    print(f"\n{'='*55}")
    print(f"  REGIME-AWARE SWITCHER STRATEGY")
    print(f"{'='*55}")
    print(f"  Start Value:       ${start:,.0f}")
    print(f"  End Value:         ${end:,.0f}")
    print(f"  Total Return:      {total_ret:.2%}")
    print(f"  Annualised Return: {ann_ret:.2%}")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_dd:.2%}")
    print(f"  Win Rate:          {win_rate:.2%}")
    print(f"  Trading Days:      {len(returns)}")
    print(f"{'-'*55}")
    print(f"  Days in Momentum:      {regime_counts.get('momentum', 0)}")
    print(f"  Days in Mean Reversion:{regime_counts.get('mean_reversion', 0)}")
    print(f"  Days in Cash:          {regime_counts.get('cash', 0)}")
    print(f"  Days in Momentum:      {regime_counts.get('momentum', 0)}")
    print(f"  Days in Cash (choppy): {regime_counts.get('cash', 0)}")
    print(f"{'='*55}\n")

# ── 5. HELPERS ────────────────────────────────────────────────────────────────

def _sharpe(returns, periods=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods)

def _max_drawdown(returns):
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

# ── 6. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    features, regimes, mom_port, mr_port = load_all_data()

    # Step 1 — how did each standalone strategy perform within each regime?
    print("\nAnalysing individual strategy performance by regime...")
    analyse_by_regime(mom_port,  regimes, "Momentum Strategy")
    analyse_by_regime(mr_port,   regimes, "Mean Reversion Strategy")

    # Step 2 — run the regime-aware switcher
    print("\nRunning regime-aware switcher strategy...")
    switcher_portfolio = run_regime_switcher(features, regimes)

    # Step 3 — summarise switcher results
    summarise_switcher(switcher_portfolio)

    # Save for use in Week 5 ML model
    switcher_portfolio.to_parquet("data/processed/backtest_regime_switcher.parquet")
    print("Saved: data/processed/backtest_regime_switcher.parquet")