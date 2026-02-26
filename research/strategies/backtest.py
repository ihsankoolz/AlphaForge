import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum import generate_signals as momentum_signals
from strategies.mean_reversion import generate_signals as mr_signals

def run_backtest(signals_df, features_df, initial_capital=100000, transaction_cost=0.001):
    """
    Simulate trading based on signals.
    
    Key rule: signal generated on day T is executed at day T+1 open.
    This prevents lookahead bias — you can't trade on information
    you only have at end of day until the next day.
    """
    # Get daily returns and reshape to wide format (dates as rows, symbols as columns)
    returns = features_df["return_1d"].unstack(level="symbol")
    signals_wide = signals_df["signal"].unstack(level="symbol").fillna(0)

    # CRITICAL: shift signals forward by 1 day
    # Signal on Monday → trade executes Tuesday
    # Without this shift, you're trading on information you don't have yet
    signals_wide = signals_wide.shift(1)

    # Align to common dates after the shift
    common_dates = returns.index.intersection(signals_wide.index)
    returns = returns.loc[common_dates].fillna(0)
    signals_wide = signals_wide.loc[common_dates].fillna(0)

    portfolio_value = initial_capital
    portfolio_history = []
    prev_positions = pd.Series(0.0, index=signals_wide.columns)

    for date in common_dates:
        day_signals = signals_wide.loc[date]
        day_returns = returns.loc[date]

        # Select top 5 buy signals and top 5 sell signals
        buy_candidates = day_signals[day_signals > 0].nlargest(10)

        # Build position vector — equal weight across active positions
        positions = pd.Series(0.0, index=signals_wide.columns)
        
        if len(buy_candidates) > 0:
            weight = 1.0 / len(buy_candidates)
            for symbol in buy_candidates.index:
                positions[symbol] = weight

        # Transaction cost — charged on every position change
        position_changes = (positions - prev_positions).abs()
        cost = position_changes.sum() * transaction_cost

        # Daily P&L = sum of (position weight × that stock's return)
        pnl = (positions * day_returns).sum()
        
        # Portfolio grows by daily P&L minus costs
        portfolio_value = portfolio_value * (1 + pnl - cost)

        portfolio_history.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "daily_return": pnl - cost,
            "num_positions": len(buy_candidates)
        })

        prev_positions = positions

    return pd.DataFrame(portfolio_history).set_index("date")

def calculate_metrics(portfolio_history, strategy_name="Strategy"):
    """Calculate standard performance metrics."""
    returns = portfolio_history["daily_return"]
    start_value = portfolio_history["portfolio_value"].iloc[0]
    end_value = portfolio_history["portfolio_value"].iloc[-1]

    total_return = (end_value / start_value) - 1
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 else 0
    win_rate = (returns > 0).sum() / len(returns)

    # Annualised return
    years = len(returns) / 252
    annualised_return = (1 + total_return) ** (1 / years) - 1

    print(f"\n=============================")
    print(f"  {strategy_name}")
    print(f"=============================")
    print(f"  Start Value:       ${start_value:,.0f}")
    print(f"  End Value:         ${end_value:,.0f}")
    print(f"  Total Return:      {total_return:.2%}")
    print(f"  Annualised Return: {annualised_return:.2%}")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Sortino Ratio:     {sortino:.2f}")
    print(f"  Max Drawdown:      {max_drawdown:.2%}")
    print(f"  Win Rate:          {win_rate:.2%}")
    print(f"  Trading Days:      {len(returns)}")
    print(f"=============================\n")

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }

def diagnose_signals(signals_df, features_df, strategy_name="Strategy"):
    """
    Check if signals actually predict next-day returns correctly.
    This is the most basic sanity check for any strategy.
    """
    returns = features_df["return_1d"].unstack(level="symbol")
    signals_wide = signals_df["signal"].unstack(level="symbol").fillna(0)
    
    # Shift returns back by 1 to align with signals
    # i.e. does today's signal predict tomorrow's return?
    future_returns = returns.shift(-1)
    
    common_dates = signals_wide.index.intersection(future_returns.index)
    
    results = []
    for date in common_dates:
        for symbol in signals_wide.columns:
            sig = signals_wide.loc[date, symbol]
            ret = future_returns.loc[date, symbol] if symbol in future_returns.columns else np.nan
            if sig != 0 and not np.isnan(ret):
                results.append({"signal": sig, "next_return": ret})
    
    df = pd.DataFrame(results)
    
    buy_signals = df[df["signal"] > 0]
    sell_signals = df[df["signal"] < 0]
    
    print(f"\n--- Signal Diagnostic: {strategy_name} ---")
    print(f"Total signals generated: {len(df)}")
    print(f"Buy signals: {len(buy_signals)}")
    if len(buy_signals) > 0:
        print(f"  Avg next-day return after buy signal:  {buy_signals['next_return'].mean():.4%}")
        print(f"  % of buy signals that were correct:    {(buy_signals['next_return'] > 0).mean():.2%}")
    print(f"Sell signals: {len(sell_signals)}")
    if len(sell_signals) > 0:
        print(f"  Avg next-day return after sell signal: {sell_signals['next_return'].mean():.4%}")
        print(f"  % of sell signals that were correct:   {(sell_signals['next_return'] < 0).mean():.2%}")
    print(f"------------------------------------------")

if __name__ == "__main__":
    print("Loading features...")
    features = pd.read_parquet("data/processed/features_daily.parquet")

    print("Running Momentum strategy...")
    mom_signals = momentum_signals(features)
    diagnose_signals(mom_signals, features, "Momentum")
    mom_portfolio = run_backtest(mom_signals, features)
    mom_metrics = calculate_metrics(mom_portfolio, "Momentum Strategy")

    print("Running Mean Reversion strategy...")
    mr_signals_df = mr_signals(features)
    diagnose_signals(mr_signals_df, features, "Mean Reversion")
    mr_portfolio = run_backtest(mr_signals_df, features)
    mr_metrics = calculate_metrics(mr_portfolio, "Mean Reversion Strategy")

    mom_portfolio.to_parquet("data/processed/backtest_momentum.parquet")
    mr_portfolio.to_parquet("data/processed/backtest_mean_reversion.parquet")
    print("Results saved.")