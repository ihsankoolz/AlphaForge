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
    
    Assumptions:
    - We trade at next day's open after signal is generated (realistic)
    - Transaction cost of 0.1% per trade each way (covers spread + commission)
    - Equal capital allocated to each active position
    - Maximum 10 positions at once
    
    Returns daily portfolio value and trade log.
    """
    # Get daily returns for each stock
    returns = features_df["return_1d"].unstack(level="symbol")
    opens = features_df["open"].unstack(level="symbol")

    signals_wide = signals_df["signal"].unstack(level="symbol").fillna(0)
    
    # Align all dataframes to same dates
    common_dates = returns.index.intersection(signals_wide.index)
    returns = returns.loc[common_dates]
    signals_wide = signals_wide.loc[common_dates]

    portfolio_value = initial_capital
    portfolio_history = []
    prev_positions = pd.Series(0.0, index=signals_wide.columns)

    for date in common_dates:
        day_signals = signals_wide.loc[date]
        day_returns = returns.loc[date]

        # Only take top signals — max 10 positions
        active = day_signals[day_signals != 0].nlargest(5).index.tolist() + \
                 day_signals[day_signals != 0].nsmallest(5).index.tolist()
        
        # Build position vector — equal weight among active signals
        positions = pd.Series(0.0, index=signals_wide.columns)
        if active:
            weight = 1.0 / len(active)
            for symbol in active:
                positions[symbol] = np.sign(day_signals[symbol]) * weight

        # Calculate transaction costs for position changes
        position_changes = (positions - prev_positions).abs()
        cost = position_changes.sum() * transaction_cost
        
        # Calculate daily P&L
        pnl = (positions * day_returns).sum()
        portfolio_value = portfolio_value * (1 + pnl - cost)

        portfolio_history.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "daily_return": pnl - cost,
            "num_positions": len(active)
        })

        prev_positions = positions

    return pd.DataFrame(portfolio_history).set_index("date")

def calculate_metrics(portfolio_history):
    """Calculate standard performance metrics."""
    returns = portfolio_history["daily_return"]
    
    total_return = (portfolio_history["portfolio_value"].iloc[-1] / 
                    portfolio_history["portfolio_value"].iloc[0]) - 1
    
    # Sharpe ratio — annualised (252 trading days)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Sortino ratio — like Sharpe but only penalises downside volatility
    downside_returns = returns[returns < 0]
    sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    print("\n=============================")
    print("  Backtest Performance Report")
    print("=============================")
    print(f"  Total Return:    {total_return:.2%}")
    print(f"  Sharpe Ratio:    {sharpe:.2f}")
    print(f"  Sortino Ratio:   {sortino:.2f}")
    print(f"  Max Drawdown:    {max_drawdown:.2%}")
    print(f"  Win Rate:        {win_rate:.2%}")
    print(f"  Trading Days:    {len(returns)}")
    print("=============================\n")
    
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }

if __name__ == "__main__":
    print("Loading features...")
    features = pd.read_parquet("data/processed/features_daily.parquet")

    print("Running Momentum strategy...")
    mom_signals = momentum_signals(features)
    mom_portfolio = run_backtest(mom_signals, features)
    print("Momentum Results:")
    mom_metrics = calculate_metrics(mom_portfolio)

    print("Running Mean Reversion strategy...")
    mr_signals_df = mr_signals(features)
    mr_portfolio = run_backtest(mr_signals_df, features)
    print("Mean Reversion Results:")
    mr_metrics = calculate_metrics(mr_portfolio)

    # Save results
    mom_portfolio.to_parquet("data/processed/backtest_momentum.parquet")
    mr_portfolio.to_parquet("data/processed/backtest_mean_reversion.parquet")
    print("Results saved to data/processed/")