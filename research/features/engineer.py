import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sqlalchemy as sa

load_dotenv()

def get_engine():
    return sa.create_engine(os.getenv("DATABASE_URL"))

def load_daily_data():
    """Load all daily OHLCV data from TimescaleDB into a DataFrame."""
    engine = get_engine()
    df = pd.read_sql("""
        SELECT time, symbol, open, high, low, close, volume
        FROM ohlcv
        ORDER BY symbol, time
    """, engine)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["time", "symbol"]).sort_index()
    return df

def add_returns(df):
    """Compute daily percentage returns for each symbol."""
    df["return_1d"] = df.groupby("symbol")["close"].pct_change(1)
    df["return_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["return_10d"] = df.groupby("symbol")["close"].pct_change(10)
    df["return_20d"] = df.groupby("symbol")["close"].pct_change(20)
    return df

def add_volatility(df):
    """Compute rolling volatility (standard deviation of returns)."""
    df["volatility_10d"] = (
        df.groupby("symbol")["return_1d"]
        .transform(lambda x: x.rolling(10).std())
    )
    df["volatility_20d"] = (
        df.groupby("symbol")["return_1d"]
        .transform(lambda x: x.rolling(20).std())
    )
    return df

def add_rsi(df, period=14):
    """Compute Relative Strength Index."""
    def compute_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["rsi_14"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: compute_rsi(x, period))
    )
    return df

def add_macd(df):
    """Compute MACD line, signal line, and histogram."""
    def compute_macd(series):
        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        result = pd.DataFrame({
            "macd_line": macd_line.values,
            "macd_signal": signal_line.values,
            "macd_hist": histogram.values
        }, index=series.index)
        return result

    macd_parts = []
    for symbol, group in df.groupby("symbol"):
        result = compute_macd(group["close"])
        macd_parts.append(result)
    
    macd_all = pd.concat(macd_parts)
    df = df.join(macd_all)
    return df

def add_bollinger_bands(df, window=20, num_std=2):
    """Compute Bollinger Bands and percentage bandwidth."""
    def compute_bb(series):
        mid = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        pct_b = (series - lower) / (upper - lower)
        bandwidth = (upper - lower) / mid
        return pd.DataFrame({
            "bb_mid": mid.values,
            "bb_upper": upper.values,
            "bb_lower": lower.values,
            "bb_pct": pct_b.values,
            "bb_bandwidth": bandwidth.values
        }, index=series.index)

    bb_parts = []
    for symbol, group in df.groupby("symbol"):
        result = compute_bb(group["close"])
        bb_parts.append(result)
    
    bb_all = pd.concat(bb_parts)
    df = df.join(bb_all)
    return df

def add_volume_signals(df):
    """Compute volume relative to its recent average."""
    df["volume_ratio"] = (
        df.groupby("symbol")["volume"]
        .transform(lambda x: x / x.rolling(20).mean())
    )
    return df

def add_market_correlation(df):
    """Compute rolling 60-day correlation of each stock with SPY."""
    spy_returns = df.xs("SPY", level="symbol")["return_1d"]
    
    corr_parts = []
    for symbol, group in df.groupby("symbol"):
        stock_returns = group["return_1d"].droplevel("symbol")
        
        if symbol == "SPY":
            corr = pd.Series(1.0, index=stock_returns.index)
        else:
            spy_aligned = spy_returns.reindex(stock_returns.index)
            corr = stock_returns.rolling(60).corr(spy_aligned)
        
        corr.index = pd.MultiIndex.from_arrays(
            [corr.index, [symbol] * len(corr)],
            names=["time", "symbol"]
        )
        corr_parts.append(corr.rename("spy_correlation"))
    
    corr_all = pd.concat(corr_parts)
    df["spy_correlation"] = corr_all
    return df

def engineer_features(save=True):
    """Load data, compute all features, and optionally save to file."""
    print("Loading data from database...")
    df = load_daily_data()
    
    print("Computing returns...")
    df = add_returns(df)
    
    print("Computing volatility...")
    df = add_volatility(df)
    
    print("Computing RSI...")
    df = add_rsi(df)
    
    print("Computing MACD...")
    df = add_macd(df)
    
    print("Computing Bollinger Bands...")
    df = add_bollinger_bands(df)
    
    print("Computing volume signals...")
    df = add_volume_signals(df)
    
    print("Computing market correlation...")
    df = add_market_correlation(df)
    
    # Drop rows where we don't have enough history to compute features
    df = df.dropna(subset=["return_20d", "volatility_20d", "rsi_14", "macd_line", "bb_pct"])
    
    print(f"\nFeature engineering complete. Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    if save:
        output_path = "data/processed/features_daily.parquet"
        df.to_parquet(output_path)
        print(f"Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    df = engineer_features()
    print("\nSample for AAPL:")
    print(df.xs("AAPL", level="symbol").tail())
