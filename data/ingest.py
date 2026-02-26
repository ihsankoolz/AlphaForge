import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
import sqlalchemy as sa

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ---- CUSTOMISE THESE ----
SYMBOLS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
    # Finance
    "JPM", "GS", "BAC", "MS", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Energy
    "XOM", "CVX", "COP",
    # Consumer
    "MCD", "NKE", "SBUX", "WMT", "COST",
    # Industrial
    "CAT", "BA", "HON", "GE",
    # ETFs (useful for regime detection later)
    "SPY", "QQQ", "VIX"
]
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 12, 31)
# --------------------------

def fetch_ohlcv(symbols, start, end):
    """Pull daily OHLCV data from Alpaca for given symbols."""
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end
    )
    
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df.rename(columns={"timestamp": "time"}, inplace=True)
    print(f"Fetched {len(df)} rows from Alpaca")
    return df

def fetch_ohlcv_hourly(symbols, start, end):
    """Pull hourly OHLCV data from Alpaca for given symbols."""
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end
    )
    
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df.rename(columns={"timestamp": "time"}, inplace=True)
    print(f"Fetched {len(df)} hourly rows from Alpaca")
    return df

def setup_table(conn):
    """Create the ohlcv table and hypertable if they don't exist."""
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            time        TIMESTAMPTZ NOT NULL,
            symbol      TEXT NOT NULL,
            open        DOUBLE PRECISION,
            high        DOUBLE PRECISION,
            low         DOUBLE PRECISION,
            close       DOUBLE PRECISION,
            volume      DOUBLE PRECISION
        );
    """))
    conn.execute(sa.text("""
        SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);
    """))
    # Add unique constraint to prevent duplicates
    conn.execute(sa.text("""
        ALTER TABLE ohlcv 
        DROP CONSTRAINT IF EXISTS ohlcv_time_symbol_unique;
    """))
    conn.execute(sa.text("""
        ALTER TABLE ohlcv 
        ADD CONSTRAINT ohlcv_time_symbol_unique UNIQUE (time, symbol);
    """))
    conn.commit()



def save_to_timescale(df):
    """Store OHLCV data in TimescaleDB, skipping duplicates."""
    DB_URL = os.getenv("DATABASE_URL")
    engine = sa.create_engine(DB_URL)
    
    with engine.connect() as conn:
        setup_table(conn)
        
        df_to_save = df[["time", "symbol", "open", "high", "low", "close", "volume"]]
        inserted = 0
        
        for _, row in df_to_save.iterrows():
            result = conn.execute(sa.text("""
                INSERT INTO ohlcv (time, symbol, open, high, low, close, volume)
                VALUES (:time, :symbol, :open, :high, :low, :close, :volume)
                ON CONFLICT (time, symbol) DO NOTHING
            """), row.to_dict())
            inserted += result.rowcount
        
        conn.commit()
    
    print(f"Inserted {inserted} new rows ({len(df_to_save) - inserted} duplicates skipped)")

def setup_hourly_table(conn):
    """Create the ohlcv_hourly table if it doesn't exist."""
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS ohlcv_hourly (
            time        TIMESTAMPTZ NOT NULL,
            symbol      TEXT NOT NULL,
            open        DOUBLE PRECISION,
            high        DOUBLE PRECISION,
            low         DOUBLE PRECISION,
            close       DOUBLE PRECISION,
            volume      DOUBLE PRECISION
        );
    """))
    conn.execute(sa.text("""
        SELECT create_hypertable('ohlcv_hourly', 'time', if_not_exists => TRUE);
    """))
    conn.execute(sa.text("""
        ALTER TABLE ohlcv_hourly
        DROP CONSTRAINT IF EXISTS ohlcv_hourly_time_symbol_unique;
    """))
    conn.execute(sa.text("""
        ALTER TABLE ohlcv_hourly
        ADD CONSTRAINT ohlcv_hourly_time_symbol_unique UNIQUE (time, symbol);
    """))
    conn.commit()

def save_hourly_to_timescale(df):
    """Store hourly OHLCV data in TimescaleDB, skipping duplicates."""
    DB_URL = os.getenv("DATABASE_URL")
    engine = sa.create_engine(DB_URL)
    
    with engine.connect() as conn:
        setup_hourly_table(conn)
        
        df_to_save = df[["time", "symbol", "open", "high", "low", "close", "volume"]]
        inserted = 0
        
        for _, row in df_to_save.iterrows():
            result = conn.execute(sa.text("""
                INSERT INTO ohlcv_hourly (time, symbol, open, high, low, close, volume)
                VALUES (:time, :symbol, :open, :high, :low, :close, :volume)
                ON CONFLICT (time, symbol) DO NOTHING
            """), row.to_dict())
            inserted += result.rowcount
        
        conn.commit()
    
    print(f"Inserted {inserted} new hourly rows ({len(df_to_save) - inserted} duplicates skipped)")

def print_summary():
    """Print a summary of what's currently stored in the database."""
    DB_URL = os.getenv("DATABASE_URL")
    engine = sa.create_engine(DB_URL)
    
    with engine.connect() as conn:
        result = conn.execute(sa.text("""
            SELECT symbol, COUNT(*) as rows, MIN(time) as from_date, MAX(time) as to_date
            FROM ohlcv
            GROUP BY symbol
            ORDER BY symbol
        """))
        rows = result.fetchall()
    
    print("\n--- Database Summary ---")
    for row in rows:
        print(f"{row[0]}: {row[1]} rows | {row[2].date()} to {row[3].date()}")
    print("------------------------\n")

if __name__ == "__main__":
    # Daily data
    print("=== Fetching Daily Data ===")
    df_daily = fetch_ohlcv(SYMBOLS, START_DATE, END_DATE)
    save_to_timescale(df_daily)
    print_summary()
    
    # Hourly data
    print("=== Fetching Hourly Data ===")
    df_hourly = fetch_ohlcv_hourly(SYMBOLS, START_DATE, END_DATE)
    save_hourly_to_timescale(df_hourly)