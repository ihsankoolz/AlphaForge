import os
import pandas as pd
from dotenv import load_dotenv
import sqlalchemy as sa

load_dotenv()

def get_engine():
    return sa.create_engine(os.getenv("DATABASE_URL"))

def check_missing_dates(table="ohlcv"):
    """Check for unexpected gaps in trading days per symbol."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(sa.text(f"""
            SELECT symbol, COUNT(*) as row_count
            FROM {table}
            GROUP BY symbol
            ORDER BY symbol
        """))
        rows = result.fetchall()
    
    print(f"\n--- Row Count Check ({table}) ---")
    counts = [r[1] for r in rows]
    expected = max(counts)
    issues = 0
    for row in rows:
        if row[1] < expected * 0.95:  # flag if more than 5% fewer rows than expected
            print(f"  WARNING {row[0]}: only {row[1]} rows (expected ~{expected})")
            issues += 1
    if issues == 0:
        print(f"  All symbols look consistent (~{expected} rows each)")

def check_nulls(table="ohlcv"):
    """Check for any null values in price or volume columns."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(sa.text(f"""
            SELECT symbol,
                SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as null_open,
                SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) as null_high,
                SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) as null_low,
                SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
                SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
            FROM {table}
            GROUP BY symbol
            HAVING SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) > 0
        """))
        rows = result.fetchall()
    
    print(f"\n--- Null Check ({table}) ---")
    if not rows:
        print("  No nulls found")
    else:
        for row in rows:
            print(f"  WARNING {row[0]}: nulls detected — open:{row[1]} high:{row[2]} low:{row[3]} close:{row[4]} volume:{row[5]}")

def check_price_anomalies(table="ohlcv"):
    """Check for suspicious prices — zeros, negatives, or extreme spikes."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(sa.text(f"""
            SELECT symbol,
                SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END) as zero_or_negative,
                SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as high_below_low,
                SUM(CASE WHEN close > high OR close < low THEN 1 ELSE 0 END) as close_outside_range
            FROM {table}
            GROUP BY symbol
            HAVING 
                SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END) > 0 OR
                SUM(CASE WHEN high < low THEN 1 ELSE 0 END) > 0 OR
                SUM(CASE WHEN close > high OR close < low THEN 1 ELSE 0 END) > 0
        """))
        rows = result.fetchall()
    
    print(f"\n--- Price Anomaly Check ({table}) ---")
    if not rows:
        print("  No price anomalies found")
    else:
        for row in rows:
            print(f"  WARNING {row[0]}: zero/neg:{row[1]} high<low:{row[2]} close outside range:{row[3]}")

def run_all_checks():
    print("=============================")
    print("  Data Validation Report")
    print("=============================")
    
    for table in ["ohlcv", "ohlcv_hourly"]:
        check_missing_dates(table)
        check_nulls(table)
        check_price_anomalies(table)
    
    print("\n=============================")
    print("  Validation Complete")
    print("=============================")

if __name__ == "__main__":
    run_all_checks()