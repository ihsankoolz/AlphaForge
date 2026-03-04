"""
scripts/expand_universe_batch1.py
==================================
One-time universe expansion — Batch 1 (50 new symbols).

Stages:
    1. OHLCV backfill  — fetch 2020-2025 daily data for new symbols → TimescaleDB
    2. News backfill   — fetch 2020-2025 news, score with FinBERT → news_scored.parquet
    3. Recompute       — rebuild features_daily.parquet + sentiment_daily.parquet
                         on full 79-symbol universe

Checkpointing: each stage writes a .done file so reruns skip completed stages.
    data/processed/.batch1_ohlcv.done
    data/processed/.batch1_news.done
    data/processed/.batch1_recompute.done

Usage:
    python scripts/expand_universe_batch1.py

    Estimated runtime (CPU):
        Stage 1 — ~15 minutes  (OHLCV fetch, fast)
        Stage 2 — ~5 hours     (news fetch ~90min + FinBERT scoring ~3.5hrs)
        Stage 3 — ~2 minutes   (feature recompute)
    Run overnight. Safe to interrupt and resume.
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Batch 1 — 50 new symbols ──────────────────────────────────────────────────
BATCH1_SYMBOLS = [
    # Tech — mega caps missing from current universe
    "TSLA", "AVGO", "AMD", "ORCL", "ADBE", "CRM", "NFLX", "QCOM", "TXN", "CSCO",
    "IBM", "NOW", "AMAT", "MU", "INTC",
    # Finance — V and MA are larger than GS by market cap
    "V", "MA", "C", "WFC", "AXP", "SCHW", "COF", "BK",
    # Healthcare — LLY is now top-10 S&P constituent
    "LLY", "MRK", "AMGN", "BMY", "GILD", "TMO", "MDT", "ISRG",
    # Consumer
    "HD", "TGT", "LOW", "BKNG", "TJX", "MAR",
    # Industrial
    "RTX", "LMT", "UPS", "DE", "ETN",
    # Utilities — negative beta, Markowitz loves this
    "NEE", "DUK",
    # REITs — different return profile entirely
    "AMT", "PLD",
    # Materials
    "LIN", "NEM",
    # Communications
    "DIS", "VZ",
]

# Full universe after this batch
EXISTING_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN',
    'JPM', 'GS', 'BAC', 'MS', 'BLK',
    'JNJ', 'UNH', 'PFE', 'ABBV',
    'XOM', 'CVX', 'COP',
    'MCD', 'NKE', 'SBUX', 'WMT', 'COST',
    'CAT', 'BA', 'HON', 'GE',
    'SPY', 'QQQ',
]
FULL_UNIVERSE = EXISTING_SYMBOLS + BATCH1_SYMBOLS

print(f"Batch 1: {len(BATCH1_SYMBOLS)} new symbols")
print(f"Full universe after expansion: {len(FULL_UNIVERSE)} symbols")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — OHLCV BACKFILL
# ─────────────────────────────────────────────────────────────────────────────

def stage1_ohlcv():
    done_flag = os.path.join(DATA_DIR, '.batch1_ohlcv.done')
    if os.path.exists(done_flag):
        print("\n[Stage 1] Already complete — skipping OHLCV backfill")
        return

    print("\n" + "=" * 60)
    print("  STAGE 1 — OHLCV BACKFILL")
    print("  Fetching 2020-2025 daily bars for 50 new symbols")
    print("  Estimated time: ~15 minutes")
    print("=" * 60)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    import sqlalchemy as sa

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    db_url     = os.getenv("DATABASE_URL")

    client = StockHistoricalDataClient(api_key, secret_key)
    engine = sa.create_engine(db_url)

    start = datetime(2020, 1, 1)
    end   = datetime(2025, 12, 31)

    # Fetch in batches of 10 to avoid API timeouts
    total_inserted = 0
    for i in range(0, len(BATCH1_SYMBOLS), 10):
        chunk = BATCH1_SYMBOLS[i:i+10]
        print(f"\n  Fetching {chunk} ({i+1}-{min(i+10, len(BATCH1_SYMBOLS))}/{len(BATCH1_SYMBOLS)})...")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX
            )
            bars = client.get_stock_bars(request)
            df   = bars.df.reset_index()
            df.rename(columns={"timestamp": "time"}, inplace=True)
            df   = df[["time", "symbol", "open", "high", "low", "close", "volume"]]

            # Upsert into TimescaleDB
            inserted = 0
            with engine.connect() as conn:
                for _, row in df.iterrows():
                    result = conn.execute(sa.text("""
                        INSERT INTO ohlcv (time, symbol, open, high, low, close, volume)
                        VALUES (:time, :symbol, :open, :high, :low, :close, :volume)
                        ON CONFLICT (time, symbol) DO NOTHING
                    """), row.to_dict())
                    inserted += result.rowcount
                conn.commit()

            total_inserted += inserted
            print(f"  Inserted {inserted:,} rows ({len(df) - inserted} duplicates skipped)")

        except Exception as e:
            print(f"  ERROR fetching {chunk}: {e}")
            print("  Continuing with next chunk...")

        time.sleep(1)  # be polite to the API

    print(f"\n  Stage 1 complete — {total_inserted:,} total rows inserted")
    open(done_flag, 'w').close()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — NEWS BACKFILL + FINBERT SCORING
# ─────────────────────────────────────────────────────────────────────────────

def stage2_news():
    done_flag = os.path.join(DATA_DIR, '.batch1_news.done')
    if os.path.exists(done_flag):
        print("\n[Stage 2] Already complete — skipping news backfill")
        return

    print("\n" + "=" * 60)
    print("  STAGE 2 — NEWS BACKFILL + FINBERT SCORING")
    print("  Fetching 2020-2025 news for 50 new symbols")
    print("  Estimated time: ~5 hours total (run overnight)")
    print("=" * 60)

    # Import sentiment functions from existing module
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models'))
    from sentiment import fetch_news_alpaca, expand_by_symbol, load_finbert, score_sentiment

    scored_cache = os.path.join(DATA_DIR, 'news_scored.parquet')

    # ── Step 2a: Fetch raw news ───────────────────────────────────────────────
    raw_cache = os.path.join(DATA_DIR, 'batch1_news_raw.parquet')

    if os.path.exists(raw_cache):
        print("\n  Loading cached raw news for batch 1...")
        news_df = pd.read_parquet(raw_cache)
        print(f"  Loaded {len(news_df):,} cached articles")
    else:
        print("\n  Fetching news from Alpaca (2020-01-01 → 2025-12-31)...")
        print("  This takes ~90 minutes — do not interrupt\n")
        news_df = fetch_news_alpaca(
            BATCH1_SYMBOLS,
            start_date="2020-01-01",
            end_date="2025-12-31"
        )
        news_df.to_parquet(raw_cache, index=False)
        print(f"\n  Saved {len(news_df):,} raw articles to {raw_cache}")

    # ── Step 2b: Expand by symbol ─────────────────────────────────────────────
    print("\n  Expanding articles by symbol...")
    expanded = expand_by_symbol(news_df, BATCH1_SYMBOLS)
    print(f"  {len(expanded):,} (symbol, article) pairs")

    if expanded.empty:
        print("  No articles found — skipping scoring")
        open(done_flag, 'w').close()
        return

    # ── Step 2c: Deduplicate against existing scored cache ────────────────────
    if os.path.exists(scored_cache):
        existing = pd.read_parquet(scored_cache)
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize().dt.tz_localize(None)
        existing_keys = set(
            zip(existing["date"].astype(str),
                existing["symbol"],
                existing["headline"])
        )
        mask = ~expanded.apply(
            lambda r: (str(r["date"]), r["symbol"], r["headline"]) in existing_keys,
            axis=1
        )
        expanded = expanded[mask]
        print(f"  {len(expanded):,} genuinely new articles after dedup against existing cache")
    else:
        existing = pd.DataFrame()
        print("  No existing cache — scoring all articles")

    if expanded.empty:
        print("  All articles already scored — nothing to do")
        open(done_flag, 'w').close()
        return

    # ── Step 2d: Score with FinBERT ───────────────────────────────────────────
    # Score in checkpointed chunks of 5,000 articles so we can resume
    # if the process is interrupted partway through
    chunk_size    = 5000
    chunks_dir    = os.path.join(DATA_DIR, 'batch1_scored_chunks')
    os.makedirs(chunks_dir, exist_ok=True)

    n_chunks = (len(expanded) // chunk_size) + 1
    print(f"\n  Scoring {len(expanded):,} articles in {n_chunks} chunks of {chunk_size}")
    print(f"  Each chunk takes ~{chunk_size // 32 * 1.5 / 60:.0f} minutes on CPU")
    print(f"  Loading FinBERT...")

    finbert = load_finbert()
    scored_chunks = []

    for chunk_idx in range(n_chunks):
        chunk_file = os.path.join(chunks_dir, f'chunk_{chunk_idx:04d}.parquet')

        if os.path.exists(chunk_file):
            print(f"  Chunk {chunk_idx+1}/{n_chunks} already scored — loading from cache")
            scored_chunks.append(pd.read_parquet(chunk_file))
            continue

        start_i = chunk_idx * chunk_size
        end_i   = min(start_i + chunk_size, len(expanded))
        chunk   = expanded.iloc[start_i:end_i].copy()

        if chunk.empty:
            continue

        print(f"\n  Chunk {chunk_idx+1}/{n_chunks} — scoring {len(chunk):,} articles "
              f"({start_i:,} to {end_i:,})...")

        t0     = time.time()
        scores = score_sentiment(chunk["text"].tolist(), finbert)
        elapsed = time.time() - t0

        chunk["label"]           = [s["label"]           for s in scores]
        chunk["confidence"]      = [s["confidence"]      for s in scores]
        chunk["sentiment_score"] = [s["sentiment_score"] for s in scores]

        chunk.to_parquet(chunk_file, index=False)
        scored_chunks.append(chunk)

        remaining = n_chunks - chunk_idx - 1
        print(f"  Chunk {chunk_idx+1} done in {elapsed/60:.1f}min — "
              f"~{remaining * elapsed / 60:.0f}min remaining")

    # ── Step 2e: Merge with existing cache and save ───────────────────────────
    print("\n  Merging all scored chunks...")
    new_scored = pd.concat(scored_chunks, ignore_index=True)
    new_scored["date"] = pd.to_datetime(new_scored["date"]).dt.normalize().dt.tz_localize(None)

    if not existing.empty:
        combined = pd.concat([existing, new_scored], ignore_index=True)
    else:
        combined = new_scored

    combined.to_parquet(scored_cache, index=False)
    print(f"  Updated scored cache: {len(combined):,} total articles")
    print(f"  Saved to {scored_cache}")

    open(done_flag, 'w').close()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — RECOMPUTE FEATURES + SENTIMENT ON FULL UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def stage3_recompute():
    done_flag = os.path.join(DATA_DIR, '.batch1_recompute.done')
    if os.path.exists(done_flag):
        print("\n[Stage 3] Already complete — skipping recompute")
        return

    print("\n" + "=" * 60)
    print("  STAGE 3 — RECOMPUTE ON FULL 79-SYMBOL UNIVERSE")
    print("  Rebuilding features_daily.parquet + sentiment_daily.parquet")
    print("  Estimated time: ~2-3 minutes")
    print("=" * 60)

    # ── 3a: Recompute features ────────────────────────────────────────────────
    print("\n  Recomputing features_daily.parquet...")
    from research.features.engineer import engineer_features
    df_features = engineer_features(save=True)
    print(f"  Features: {len(df_features):,} rows, latest: "
          f"{df_features.index.get_level_values('time').max().date()}")

    # ── 3b: Recompute daily sentiment on full universe ────────────────────────
    print("\n  Recomputing sentiment_daily.parquet on full universe...")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models'))
    from sentiment import aggregate_daily_sentiment

    scored_cache = os.path.join(DATA_DIR, 'news_scored.parquet')
    scored_df    = pd.read_parquet(scored_cache)
    scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.normalize().dt.tz_localize(None)

    features_flat = df_features.reset_index()
    all_dates     = pd.to_datetime(features_flat["time"]).dt.normalize().dt.tz_localize(None).unique()
    all_dates     = sorted(all_dates)

    daily_sentiment = aggregate_daily_sentiment(scored_df, FULL_UNIVERSE, all_dates)
    daily_sentiment.to_parquet(
        os.path.join(DATA_DIR, 'sentiment_daily.parquet'), index=False
    )

    coverage = (daily_sentiment['article_count'] > 0).mean()
    print(f"  Sentiment: {len(daily_sentiment):,} rows, coverage: {coverage:.1%}")

    print("\n  Stage 3 complete")
    open(done_flag, 'w').close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AlphaForge — Universe Expansion Batch 1")
    print(f"  Adding {len(BATCH1_SYMBOLS)} symbols → {len(FULL_UNIVERSE)} total")
    print("=" * 60)

    t_start = time.time()

    stage1_ohlcv()
    stage2_news()
    stage3_recompute()

    elapsed = (time.time() - t_start) / 3600
    print("\n" + "=" * 60)
    print("  Batch 1 expansion COMPLETE")
    print(f"  Total elapsed: {elapsed:.1f} hours")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Update SYMBOLS list in execution/run_daily.py to FULL_UNIVERSE")
    print("  2. Retrain HMM: python models/regime_hmm.py")
    print("  3. Retrain XGBoost: python models/ml_signal.py")
    print("  4. Run dry run to verify: python execution/run_daily.py")