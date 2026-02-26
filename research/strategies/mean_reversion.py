import pandas as pd
import numpy as np

def generate_signals(features_df):
    """
    Mean Reversion Strategy — buy oversold stocks, sell overbought stocks.
    
    Logic:
    - RSI below 35 and price near lower Bollinger Band → oversold → buy
    - RSI above 65 and price near upper Bollinger Band → overbought → sell
    - Signal strength scales with how extreme the oversold/overbought condition is
    - Volume confirmation — only trade if volume is at least 80% of normal
      (avoids trading on thin, meaningless moves)
    
    Returns a DataFrame with columns: date, symbol, signal
    """
    signals = []

    for date, day_data in features_df.groupby(level="time"):

        day_data = day_data.dropna(subset=["rsi_14", "bb_pct", "volume_ratio"])

        if len(day_data) < 5:
            continue

        for (time, symbol), row in day_data.iterrows():

            signal = 0.0

            # Skip if volume is too low — thin markets make signals unreliable
            if row["volume_ratio"] < 0.8:
                signals.append({"date": time, "symbol": symbol, "signal": 0.0,
                                 "rsi_14": row["rsi_14"], "bb_pct": row["bb_pct"],
                                 "volume_ratio": row["volume_ratio"]})
                continue

            # Oversold condition — potential buy
            if row["rsi_14"] < 35 and row["bb_pct"] < 0.25:
                # How oversold? RSI 20 is more oversold than RSI 34
                rsi_strength = (35 - row["rsi_14"]) / 35        # 0 to 1
                bb_strength = (0.25 - row["bb_pct"]) / 0.25     # 0 to 1
                signal = (rsi_strength + bb_strength) / 2        # average the two

            # Overbought condition — potential sell
            elif row["rsi_14"] > 65 and row["bb_pct"] > 0.75:
                rsi_strength = (row["rsi_14"] - 65) / 35        # 0 to 1
                bb_strength = (row["bb_pct"] - 0.75) / 0.25     # 0 to 1
                signal = -((rsi_strength + bb_strength) / 2)     # negative = sell

            signals.append({
                "date": time,
                "symbol": symbol,
                "signal": signal,
                "rsi_14": row["rsi_14"],
                "bb_pct": row["bb_pct"],
                "volume_ratio": row["volume_ratio"]
            })

    return pd.DataFrame(signals).set_index(["date", "symbol"])