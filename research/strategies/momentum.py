import pandas as pd
import numpy as np

def generate_signals(features_df):
    """
    Momentum Strategy — buy recent winners, sell recent losers.
    
    Logic:
    - Every day, rank all stocks by their 20-day return
    - Top 20% get a buy signal (+1)
    - Bottom 20% get a sell signal (-1)
    - Everyone else gets 0 (no position)
    - Confirm the signal with MACD histogram (momentum must be in same direction)
    
    Returns a DataFrame with columns: date, symbol, signal
    """
    signals = []

    # Work day by day
    for date, day_data in features_df.groupby(level="time"):
        
        # Drop stocks with missing data on this day
        day_data = day_data.dropna(subset=["return_20d", "macd_hist"])
        
        if len(day_data) < 5:
            continue

        # Rank stocks by 20-day return (0 = worst, 1 = best)
        day_data = day_data.copy()
        day_data["return_rank"] = day_data["return_20d"].rank(pct=True)

        for (time, symbol), row in day_data.iterrows():
            
            raw_signal = 0.0

            # Top 20% by return + MACD confirming upward momentum
            if row["return_rank"] >= 0.8 and row["macd_hist"] > 0:
                raw_signal = 1.0

            # Bottom 20% by return + MACD confirming downward momentum
            elif row["return_rank"] <= 0.2 and row["macd_hist"] < 0:
                raw_signal = -1.0

            # Scale signal strength by how extreme the return rank is
            # e.g. rank 0.95 is stronger than rank 0.82
            if raw_signal == 1.0:
                raw_signal = (row["return_rank"] - 0.8) / 0.2  # scales 0.8-1.0 → 0-1
            elif raw_signal == -1.0:
                raw_signal = -((0.2 - row["return_rank"]) / 0.2)  # scales 0-0.2 → 0 to -1

            signals.append({
                "date": time,
                "symbol": symbol,
                "signal": raw_signal,
                "return_20d": row["return_20d"],
                "return_rank": row["return_rank"],
                "macd_hist": row["macd_hist"]
            })

    return pd.DataFrame(signals).set_index(["date", "symbol"])