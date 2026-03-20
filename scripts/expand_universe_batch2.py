"""
scripts/expand_universe_batch2.py
==================================
Universe expansion — Batch 2 (50 new symbols → 129 total).

Focus: sector depth — more names within existing sectors, plus new
sub-industries (biotech, fintech, cybersecurity, clean energy).

Stages (same pattern as Batch 1):
    1. OHLCV backfill  — fetch 2020-2026 daily data for new symbols → TimescaleDB
    2. News backfill   — fetch 2020-2026 news, score with FinBERT → news_scored.parquet
    3. Recompute       — rebuild features_daily.parquet + sentiment_daily.parquet
                         on full 129-symbol universe
    4. Retrain models  — retrain HMM + XGBoost on new universe

Checkpointing: each stage writes a .done file so reruns skip completed stages.
    data/processed/.batch2_ohlcv.done
    data/processed/.batch2_news.done
    data/processed/.batch2_recompute.done
    data/processed/.batch2_retrain.done

Usage:
    python scripts/expand_universe_batch2.py

    Estimated runtime (CPU):
        Stage 1 — ~15 minutes  (OHLCV fetch, fast)
        Stage 2 — ~5 hours     (news fetch ~90min + FinBERT scoring ~3.5hrs)
        Stage 3 — ~3 minutes   (feature recompute)
        Stage 4 — ~20 minutes  (HMM + XGBoost retrain)
    Run overnight. Safe to interrupt and resume.

Prerequisites:
    - Batch 1 must be complete (79 symbols active)
    - Paper trading confirmed stable on 79-symbol universe
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

# ── Batch 2 — 50 new symbols ──────────────────────────────────────────────────
# Selection criteria:
#   - Sector depth: add more names within existing sectors
#   - New sub-industries: biotech, fintech, cybersecurity, clean energy
#   - Minimum $10B market cap (liquid enough for paper trading)
#   - All must have 2020-2026 data available on Alpaca/IEX

BATCH2_SYMBOLS = [
    # Tech — semiconductors & cloud depth
    "MRVL", "KLAC", "LRCX", "SNPS", "CDNS",
    "PANW", "CRWD", "ZS",                    # cybersecurity
    "DDOG", "SNOW", "TEAM",                   # cloud/SaaS
    # Finance — fintech & insurance
    "SQ", "PYPL", "FIS", "FISV",             # fintech / payments
    "PGR", "TRV", "MET", "AIG",              # insurance
    # Healthcare — biotech depth
    "VRTX", "REGN", "MRNA", "BIIB",          # biotech
    "SYK", "ZBH", "EW", "DXCM",             # medtech
    # Consumer — e-commerce & restaurants
    "ABNB", "DASH", "CMG", "YUM", "DPZ",     # restaurants & delivery
    "LULU", "ROST",                           # retail
    # Industrial — defense & aerospace depth
    "NOC", "GD", "HWM", "TDG",
    # Energy — clean energy + midstream
    "ENPH", "FSLR",                           # solar
    "WMB", "KMI",                             # midstream
    # Materials & mining
    "FCX", "APD", "SHW",
    # Communications & media
    "CMCSA", "T", "CHTR",
    # REITs & real estate depth
    "CCI", "EQIX",                            # data center REITs
]

# Import pre-Batch-2 universe (Original + Batch 1 only) to detect true duplicates.
# We check against Original + Batch1 specifically — NOT SYMBOLS (which may already
# include Batch 2 if settings.py was updated before this script ran).
from config.settings import SYMBOLS_ORIGINAL, SYMBOLS_BATCH1

PRE_BATCH2_UNIVERSE = SYMBOLS_ORIGINAL + SYMBOLS_BATCH1

# Only flag actual duplicates against pre-Batch-2 universe
dupes = set(PRE_BATCH2_UNIVERSE) & set(BATCH2_SYMBOLS)
if dupes:
    print(f"WARNING: {len(dupes)} duplicate symbols with Original+Batch1: {dupes}")
    print("Removing duplicates from Batch 2...")
    BATCH2_SYMBOLS = [s for s in BATCH2_SYMBOLS if s not in set(PRE_BATCH2_UNIVERSE)]

FULL_UNIVERSE = PRE_BATCH2_UNIVERSE + BATCH2_SYMBOLS

print(f"Batch 2: {len(BATCH2_SYMBOLS)} new symbols")
print(f"Pre-Batch-2 universe: {len(PRE_BATCH2_UNIVERSE)} symbols")
print(f"Full universe after expansion: {len(FULL_UNIVERSE)} symbols")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — OHLCV BACKFILL
# ─────────────────────────────────────────────────────────────────────────────

def stage1_ohlcv():
    done_flag = os.path.join(DATA_DIR, '.batch2_ohlcv.done')
    if os.path.exists(done_flag):
        print("\n[Stage 1] Already complete — skipping OHLCV backfill")
        return

    print("\n" + "=" * 60)
    print("  STAGE 1 — OHLCV BACKFILL")
    print(f"  Fetching 2020-2026 daily bars for {len(BATCH2_SYMBOLS)} new symbols")
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
    end   = datetime(2026, 12, 31)

    # Fetch in batches of 10 to avoid API timeouts
    total_inserted = 0
    for i in range(0, len(BATCH2_SYMBOLS), 10):
        chunk = BATCH2_SYMBOLS[i:i+10]
        print(f"\n  Fetching {chunk} ({i+1}-{min(i+10, len(BATCH2_SYMBOLS))}/{len(BATCH2_SYMBOLS)})...")

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
    done_flag = os.path.join(DATA_DIR, '.batch2_news.done')
    if os.path.exists(done_flag):
        print("\n[Stage 2] Already complete — skipping news backfill")
        return

    print("\n" + "=" * 60)
    print("  STAGE 2 — NEWS BACKFILL + FINBERT SCORING")
    print(f"  Fetching 2020-2026 news for {len(BATCH2_SYMBOLS)} new symbols")
    print("  Estimated time: ~5 hours total (run overnight)")
    print("=" * 60)

    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models'))
    from sentiment import fetch_news_alpaca, expand_by_symbol, load_finbert, score_sentiment

    scored_cache = os.path.join(DATA_DIR, 'news_scored.parquet')

    # ── Step 2a: Fetch raw news ───────────────────────────────────────────────
    raw_cache = os.path.join(DATA_DIR, 'batch2_news_raw.parquet')

    if os.path.exists(raw_cache) and os.path.getsize(raw_cache) > 1000:
        news_df = pd.read_parquet(raw_cache)
        if len(news_df) > 0:
            print(f"\n  Loading cached raw news for batch 2... {len(news_df):,} articles")
        else:
            print("\n  Cached news file is empty — re-fetching...")
            os.remove(raw_cache)
    if not os.path.exists(raw_cache):
        print("\n  Fetching news from Alpaca (2020-01-01 → 2026-12-31)...")
        print("  This takes ~90 minutes — do not interrupt\n")
        news_df = fetch_news_alpaca(
            BATCH2_SYMBOLS,
            start_date="2020-01-01",
            end_date="2026-12-31"
        )
        news_df.to_parquet(raw_cache, index=False)
        print(f"\n  Saved {len(news_df):,} raw articles to {raw_cache}")

    # ── Step 2b: Expand by symbol ─────────────────────────────────────────────
    print("\n  Expanding articles by symbol...")
    expanded = expand_by_symbol(news_df, BATCH2_SYMBOLS)
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
    chunk_size    = 5000
    chunks_dir    = os.path.join(DATA_DIR, 'batch2_scored_chunks')
    os.makedirs(chunks_dir, exist_ok=True)

    n_chunks = (len(expanded) // chunk_size) + 1
    print(f"\n  Scoring {len(expanded):,} articles in {n_chunks} chunks of {chunk_size}")
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

    open(done_flag, 'w').close()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — RECOMPUTE FEATURES + SENTIMENT ON FULL UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def stage3_recompute():
    done_flag = os.path.join(DATA_DIR, '.batch2_recompute.done')
    if os.path.exists(done_flag):
        print("\n[Stage 3] Already complete — skipping recompute")
        return

    print("\n" + "=" * 60)
    print(f"  STAGE 3 — RECOMPUTE ON FULL {len(FULL_UNIVERSE)}-SYMBOL UNIVERSE")
    print("  Rebuilding features_daily.parquet + sentiment_daily.parquet")
    print("  Estimated time: ~3 minutes")
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

    open(done_flag, 'w').close()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — RETRAIN MODELS ON NEW UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

def stage4_retrain():
    done_flag = os.path.join(DATA_DIR, '.batch2_retrain.done')
    if os.path.exists(done_flag):
        print("\n[Stage 4] Already complete — skipping retrain")
        return

    print("\n" + "=" * 60)
    print("  STAGE 4 — RETRAIN MODELS ON NEW UNIVERSE")
    print("  Retraining HMM + XGBoost on 129 symbols")
    print("  Estimated time: ~20 minutes")
    print("=" * 60)

    # ── 4a: Version existing models before overwriting ────────────────────────
    print("\n  Versioning current models before retrain...")
    try:
        from models.versioning import save_model_version
        import pickle

        models_dir = os.path.join(PROJECT_ROOT, 'models')

        for model_file in ['hmm_model.pkl', 'xgb_model.pkl']:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                save_model_version(
                    model_data, model_file.replace('.pkl', ''),
                    metadata={"reason": "pre-batch2-retrain", "universe_size": 79}
                )
                print(f"  Versioned: {model_file}")
    except Exception as e:
        print(f"  Warning: could not version models ({e}) — continuing anyway")

    # ── 4b: Retrain HMM ─────────────────────────────────────────────────────
    print("\n  Retraining HMM on 129-symbol features...")
    try:
        from models.regime_hmm import load_market_features, train_hmm, label_regimes
        import pickle
        market_features = load_market_features()
        print(f"  Market features: {len(market_features)} days")
        model, scaler, feature_cols = train_hmm(market_features)
        regime_df, label_map = label_regimes(market_features, model, scaler, feature_cols)
        # Save model + labels
        model_path = os.path.join(PROJECT_ROOT, 'models', 'hmm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, f)
        regime_df.to_parquet(os.path.join(DATA_DIR, 'regime_labels.parquet'))
        print(f"  HMM retrained — saved to {model_path}")
        print(f"  Regime labels: {len(regime_df)} days, label_map: {label_map}")
    except ImportError:
        print("  models.regime_hmm.train_hmm() not found — run manually:")
        print("    python models/regime_hmm.py")
    except Exception as e:
        print(f"  HMM retrain error: {e}")
        print("  Run manually: python models/regime_hmm.py")

    # ── 4c: Retrain XGBoost ──────────────────────────────────────────────────
    print("\n  Retraining XGBoost on 129-symbol features + sentiment...")
    try:
        from models.ml_signal import load_data, walk_forward_validation
        print("  Loading data...")
        df = load_data()
        print(f"  Loaded {len(df):,} rows | {df['symbol'].nunique()} symbols")
        print("  Running walk-forward validation...")
        predictions, scores = walk_forward_validation(df)
        print(f"  Walk-forward complete: {len(predictions):,} predictions")
        print(f"  Mean AUC: {scores['auc'].mean():.3f}")

        # Save predictions for backtesting
        predictions.to_parquet(
            os.path.join(DATA_DIR, 'ml_signals.parquet'), index=False
        )
        print("  Saved updated ml_signals.parquet")
    except ImportError:
        print("  Could not import ml_signal functions — run manually:")
        print("    python models/ml_signal.py")
    except Exception as e:
        print(f"  XGBoost retrain error: {e}")
        print("  Run manually: python models/ml_signal.py")

    open(done_flag, 'w').close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AlphaForge — Universe Expansion Batch 2")
    print(f"  Adding {len(BATCH2_SYMBOLS)} symbols → {len(FULL_UNIVERSE)} total")
    print("=" * 60)

    t_start = time.time()

    stage1_ohlcv()
    stage2_news()
    stage3_recompute()
    stage4_retrain()

    elapsed = (time.time() - t_start) / 3600
    print("\n" + "=" * 60)
    print("  Batch 2 expansion COMPLETE")
    print(f"  Total elapsed: {elapsed:.1f} hours")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Update config/settings.py — add SYMBOLS_BATCH2 to SYMBOLS")
    print("  2. Verify dry run: python execution/run_daily.py")
    print("  3. Monitor paper trading for 5 days on 129-symbol universe")
    print("  4. If stable, consider enabling ALLOW_SHORT_SELLING=True")
    print("     (universe now large enough for short signals to have edge)")
