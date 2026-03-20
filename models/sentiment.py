import pandas as pd
import numpy as np
import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    SENTIMENT_DECAY_HALFLIFE_HOURS, SENTIMENT_MAX_AGE_HOURS,
)

# ── 1. FETCH NEWS FROM ALPACA ─────────────────────────────────────────────────

def fetch_news_alpaca(symbols, start_date="2020-01-01", end_date="2025-12-31",
                      limit_per_request=200):
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client     = NewsClient(api_key=api_key, secret_key=secret_key)

    all_articles = []
    seen_ids     = set()  # deduplicate articles that mention multiple symbols

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    for sym_idx, symbol in enumerate(symbols):
        print(f"  Fetching news for {symbol} ({sym_idx+1}/{len(symbols)})...")
        current = start

        while current < end:
            chunk_end = min(current + timedelta(days=90), end)  # quarterly chunks per symbol

            try:
                request = NewsRequest(
                    symbols=symbol,
                    start=current.strftime("%Y-%m-%dT00:00:00Z"),
                    end=chunk_end.strftime("%Y-%m-%dT00:00:00Z"),
                    limit=limit_per_request,
                    sort="DESC"
                )
                news = client.get_news(request)

                articles = []
                for key, val in news:
                    if key == "data" and isinstance(val, dict) and "news" in val:
                        articles = val["news"]
                        break

                for article in articles:
                    article_id = getattr(article, "id", None)
                    if article_id and article_id in seen_ids:
                        continue  # skip duplicates
                    if article_id:
                        seen_ids.add(article_id)

                    article_symbols = getattr(article, "symbols", []) or []
                    created = getattr(article, "created_at", None)
                    if created is None:
                        continue

                    all_articles.append({
                        "date":     pd.to_datetime(created).normalize().tz_localize(None),
                        "headline": getattr(article, "headline", "") or "",
                        "summary":  getattr(article, "summary",  "") or "",
                        "symbols":  article_symbols,
                        "source":   getattr(article, "source",   "") or "",
                    })

            except Exception as e:
                print(f"\n    Warning: {symbol} chunk failed: {e}")

            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)

    print(f"\n  Fetched {len(all_articles):,} articles total ({len(seen_ids):,} unique)")
    return pd.DataFrame(all_articles)


def expand_by_symbol(news_df, symbols):
    """
    Each article can mention multiple stocks.
    Expand so there's one row per (date, symbol) pair.
    This lets us compute per-stock daily sentiment.
    """
    rows = []
    for _, row in news_df.iterrows():
        article_syms = row["symbols"] if isinstance(row["symbols"], list) else []
        # Keep only symbols in our universe
        relevant = [s for s in article_syms if s in symbols]
        if not relevant:
            # Article has no symbol tags — skip
            continue
        for sym in relevant:
            rows.append({
                "date":     row["date"],
                "symbol":   sym,
                "headline": row["headline"],
                "summary":  row["summary"],
                "text":     f"{row['headline']}. {row['summary']}".strip()
            })

    return pd.DataFrame(rows)


# ── 2. FINBERT SENTIMENT SCORING ─────────────────────────────────────────────

def load_finbert():
    """
    FinBERT is a version of BERT fine-tuned specifically on financial text.
    Regular sentiment models trained on movie reviews or Twitter don't 
    understand financial language — "the stock fell sharply" is negative
    but "the company beat earnings estimates" is positive in ways a 
    general model might miss.
    
    Output: positive, negative, or neutral probability for each text.
    """
    from transformers import pipeline
    print("  Loading FinBERT model (downloads ~500MB on first run)...")
    
    finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        max_length=512,
        truncation=True,
        device=-1  # CPU — change to 0 if you have a GPU
    )
    print("  FinBERT loaded.")
    return finbert


def score_sentiment(texts, finbert, batch_size=32):
    """
    Run FinBERT on a list of texts in batches.
    Returns a list of dicts with 'label' and 'score'.
    
    Labels: 'positive', 'negative', 'neutral'
    Score: confidence in that label (0 to 1)
    
    We convert to a single sentiment score:
    +score if positive, -score if negative, 0 if neutral
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            preds = finbert(batch)
            for pred in preds:
                label = pred["label"].lower()
                score = pred["score"]
                if label == "positive":
                    sentiment_score = score
                elif label == "negative":
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0
                results.append({
                    "label": label,
                    "confidence": score,
                    "sentiment_score": sentiment_score
                })
        except Exception as e:
            # If a batch fails, fill with neutral
            for _ in batch:
                results.append({"label": "neutral", "confidence": 0.0,
                                 "sentiment_score": 0.0})

        if i % (batch_size * 10) == 0 and i > 0:
            print(f"    Scored {i:,} / {len(texts):,} articles...", end="\r")

    return results


# ── 3. AGGREGATE TO DAILY SCORES ─────────────────────────────────────────────

def _compute_staleness_weight(created_at, trade_date):
    """
    Compute exponential decay weight for an article based on its age.

    Articles published right before market open are freshest and most
    relevant. As articles age, their sentiment signal degrades — by the
    time we trade at 9:30 AM, yesterday's pre-market article is 18+ hours
    old. The decay function down-weights stale articles so the aggregated
    sentiment reflects what the market hasn't yet priced in.

    Uses half-life model: weight = 2^(-age_hours / halflife)
      - At 0 hours old:  weight = 1.0
      - At 18 hours old: weight = 0.5 (default halflife)
      - At 36 hours old: weight = 0.25
      - At 72+ hours old: weight = 0.0 (discarded)

    Args:
        created_at: article publication timestamp (or date)
        trade_date: the trading day this article is being aggregated for

    Returns:
        float weight in [0, 1]
    """
    try:
        if pd.isna(created_at) or pd.isna(trade_date):
            return 1.0

        created_ts = pd.Timestamp(created_at)
        trade_ts = pd.Timestamp(trade_date)

        # Assume trading happens at 9:30 AM ET on the trade date
        # Articles are aggregated for the day they were published
        # Age = hours between article creation and end of that trading day
        # For daily aggregation: age relative to the trade_date's market close
        trade_close = trade_ts + pd.Timedelta(hours=16)  # 4 PM proxy
        age_hours = (trade_close - created_ts).total_seconds() / 3600.0

        if age_hours < 0:
            return 1.0  # future article (shouldn't happen, full weight)
        if age_hours > SENTIMENT_MAX_AGE_HOURS:
            return 0.0  # too old, discard

        weight = 2.0 ** (-age_hours / SENTIMENT_DECAY_HALFLIFE_HOURS)
        return weight
    except Exception:
        return 1.0  # fallback: full weight if timestamp parsing fails


def aggregate_daily_sentiment(scored_df, symbols, all_dates):
    """
    Convert per-article sentiment into per-stock per-day features.

    Features we create:
    - sentiment_mean:    weighted avg sentiment score across articles that day
    - sentiment_pos_pct: % of articles that were positive
    - sentiment_neg_pct: % of articles that were negative
    - article_count:     number of articles (high = more news attention)
    - sentiment_std:     disagreement between articles (high = uncertainty)
    - sentiment_freshness: avg staleness weight (1.0 = all fresh, 0.5 = aging)

    Staleness degradation: articles are weighted by an exponential decay
    based on their age. Recent articles count more than stale ones.
    sentiment_mean uses these weights; other features remain unweighted
    since they measure distribution properties (% positive, count, std).

    For days with no news we fill with 0 (neutral) — absence of news
    is itself a signal worth preserving as a zero rather than NaN.
    """
    df = scored_df.copy()

    # Compute staleness weights if timestamp information is available
    has_timestamps = "created_at" in df.columns
    if has_timestamps:
        df["_decay_weight"] = df.apply(
            lambda r: _compute_staleness_weight(r.get("created_at"), r["date"]),
            axis=1,
        )
    else:
        # No sub-day timestamps — all articles on same day get equal weight
        df["_decay_weight"] = 1.0

    # Weighted sentiment mean
    df["_weighted_score"] = df["sentiment_score"] * df["_decay_weight"]

    daily = df.groupby(["date", "symbol"]).agg(
        _weighted_score_sum = ("_weighted_score", "sum"),
        _decay_weight_sum   = ("_decay_weight", "sum"),
        sentiment_std       = ("sentiment_score", "std"),
        sentiment_pos_pct   = ("label", lambda x: (x == "positive").mean()),
        sentiment_neg_pct   = ("label", lambda x: (x == "negative").mean()),
        article_count       = ("sentiment_score", "count"),
        sentiment_freshness = ("_decay_weight", "mean"),
    ).reset_index()

    # Compute decay-weighted sentiment mean from aggregated sums
    daily["sentiment_mean"] = np.where(
        daily["_decay_weight_sum"] > 0,
        daily["_weighted_score_sum"] / daily["_decay_weight_sum"],
        0.0,
    )
    daily = daily.drop(columns=["_weighted_score_sum", "_decay_weight_sum"])

    # Build full grid of all date × symbol combinations
    date_sym = pd.MultiIndex.from_product(
        [all_dates, symbols], names=["date", "symbol"]
    ).to_frame(index=False)

    full = date_sym.merge(daily, on=["date", "symbol"], how="left")
    full = full.fillna({
        "sentiment_mean":    0.0,
        "sentiment_std":     0.0,
        "sentiment_pos_pct": 0.0,
        "sentiment_neg_pct": 0.0,
        "article_count":     0.0,
        "sentiment_freshness": 0.0,
    })

    # Binary indicator: did this stock have ANY news today?
    # Absence of news for a stock that normally has 3-5 articles/day is
    # informative — may signal pre-earnings quiet period, trading halt, etc.
    # Treating it as a separate feature lets the model distinguish "no news
    # and neutral" from "news existed and was perfectly neutral".
    full["has_news"] = (full["article_count"] > 0).astype(float)

    # Rolling 3-day sentiment — smooths out single-day noise
    full = full.sort_values(["symbol", "date"])
    full["sentiment_3d"] = (
        full.groupby("symbol")["sentiment_mean"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return full


def run_sentiment_pipeline(
    symbols:    list,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Live inference wrapper called by execution/run_daily.py each morning.

    Fetches only NEW articles (start_date → end_date), scores only those
    with FinBERT, appends to the existing scored cache, then recomputes
    and overwrites sentiment_daily.parquet.

    Why incremental?
      FinBERT takes ~2-3 minutes on the full 2020-2025 history.
      Each morning we only have ~7 days of new articles (~50-200 total).
      Scoring those takes <10 seconds. Appending keeps the pipeline fast.
    """
    # ── Normalise date args ───────────────────────────────────────────────────
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
    end_str   = end_date.strftime("%Y-%m-%d")   if hasattr(end_date,   "strftime") else str(end_date)

    print(f"  Sentiment pipeline: {start_str} → {end_str} for {len(symbols)} symbols")

    scored_cache = "data/processed/news_scored.parquet"

    # ── Load existing scored cache ────────────────────────────────────────────
    if os.path.exists(scored_cache):
        existing_scored = pd.read_parquet(scored_cache)
        existing_scored["date"] = pd.to_datetime(existing_scored["date"]).dt.normalize().dt.tz_localize(None)
        print(f"  Loaded {len(existing_scored):,} existing scored articles")
        # Determine the actual fetch window — only fetch what we don't have yet
        latest_cached = existing_scored["date"].max()
        # Start from day after latest cached, not from start_date
        # (start_date from run_daily is just a safety window, not a hard start)
        fetch_from = (latest_cached + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"  Cache covers up to {latest_cached.date()} — fetching from {fetch_from}")
    else:
        existing_scored = pd.DataFrame()
        fetch_from      = start_str
        print(f"  No cache found — fetching from {fetch_from}")

    # ── Fetch new articles ────────────────────────────────────────────────────
    # Check if there's actually a gap to fill
    fetch_from_dt = datetime.strptime(fetch_from, "%Y-%m-%d")
    end_dt        = datetime.strptime(end_str,    "%Y-%m-%d")

    new_scored = pd.DataFrame()

    if fetch_from_dt <= end_dt:
        print(f"  Fetching new articles: {fetch_from} → {end_str}")
        new_raw = fetch_news_alpaca(symbols, start_date=fetch_from, end_date=end_str)

        if new_raw.empty:
            print("  No new articles found")
        else:
            # Expand by symbol
            new_expanded = expand_by_symbol(new_raw, symbols)
            print(f"  {len(new_expanded):,} new (symbol, article) pairs")

            if not new_expanded.empty:
                # Deduplicate against existing cache by (date, symbol, headline)
                if not existing_scored.empty:
                    existing_keys = set(
                        zip(existing_scored["date"].astype(str),
                            existing_scored["symbol"],
                            existing_scored["headline"])
                    )
                    mask = ~new_expanded.apply(
                        lambda r: (str(r["date"]), r["symbol"], r["headline"]) in existing_keys,
                        axis=1
                    )
                    new_expanded = new_expanded[mask]
                    print(f"  {len(new_expanded):,} genuinely new articles after dedup")

                if not new_expanded.empty:
                    # Score with FinBERT
                    print(f"  Scoring {len(new_expanded):,} articles with FinBERT...")
                    finbert = load_finbert()
                    scores  = score_sentiment(new_expanded["text"].tolist(), finbert)

                    new_scored = new_expanded.copy()
                    new_scored["label"]           = [s["label"]           for s in scores]
                    new_scored["confidence"]      = [s["confidence"]      for s in scores]
                    new_scored["sentiment_score"] = [s["sentiment_score"] for s in scores]
                    print(f"  Scored {len(new_scored):,} new articles")
    else:
        print(f"  Cache already up to date — no new articles to fetch")

    # ── Merge new into existing cache and save ────────────────────────────────
    if not new_scored.empty:
        combined = pd.concat([existing_scored, new_scored], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"]).dt.normalize().dt.tz_localize(None)
        combined.to_parquet(scored_cache, index=False)
        print(f"  Updated cache: {len(combined):,} total scored articles")
    else:
        combined = existing_scored

    if combined.empty:
        raise ValueError("No scored articles available — cannot compute sentiment")

    # ── Recompute daily sentiment over full history ───────────────────────────
    # Always recompute from the full scored cache so that sentiment_3d
    # rolling windows are correct at the edges of the new data.
    print("  Aggregating to daily sentiment features...")
    features  = pd.read_parquet("data/processed/features_daily.parquet")
    features  = features.reset_index()
    all_dates = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None).unique()
    all_dates = sorted(all_dates)

    daily_sentiment = aggregate_daily_sentiment(combined, symbols, all_dates)

    # ── Save and return ───────────────────────────────────────────────────────
    daily_sentiment.to_parquet("data/processed/sentiment_daily.parquet", index=False)

    latest = daily_sentiment["date"].max()
    coverage = (daily_sentiment["article_count"] > 0).mean()
    print(f"  Saved sentiment_daily.parquet — latest: {latest.date()}, coverage: {coverage:.1%}")

    return daily_sentiment

# ── 4. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
        "JPM", "GS", "BAC", "MS", "BLK",
        "JNJ", "UNH", "PFE", "ABBV",
        "XOM", "CVX", "COP",
        "MCD", "NKE", "SBUX", "WMT", "COST",
        "CAT", "BA", "HON", "GE",
        "SPY", "QQQ"
    ]

    # ── Step 1: Fetch news ────────────────────────────────────────────────────
    cache_path = "data/processed/news_raw.parquet"

    if os.path.exists(cache_path):
        print("Loading cached news data...")
        news_df = pd.read_parquet(cache_path)
        print(f"  Loaded {len(news_df):,} cached articles")
    else:
        print("Fetching news from Alpaca API (this takes ~5-10 minutes)...")
        news_df = fetch_news_alpaca(SYMBOLS, start_date="2020-01-01", end_date="2025-12-31")
        os.makedirs("data/processed", exist_ok=True)
        news_df.to_parquet(cache_path)
        print(f"  Cached to {cache_path}")

    # ── Step 2: Expand by symbol ──────────────────────────────────────────────
    print("\nExpanding articles by symbol...")
    expanded = expand_by_symbol(news_df, SYMBOLS)
    print(f"  {len(expanded):,} (symbol, article) pairs from "
          f"{expanded['symbol'].nunique()} symbols")
    print(f"  Date range: {expanded['date'].min().date()} → "
          f"{expanded['date'].max().date()}")

    # ── Step 3: Run FinBERT ───────────────────────────────────────────────────
    scored_cache = "data/processed/news_scored.parquet"

    if os.path.exists(scored_cache):
        print("\nLoading cached sentiment scores...")
        scored_df = pd.read_parquet(scored_cache)
        print(f"  Loaded {len(scored_df):,} scored articles")
    else:
        print(f"\nRunning FinBERT on {len(expanded):,} articles...")
        finbert = load_finbert()

        texts = expanded["text"].tolist()
        scores = score_sentiment(texts, finbert)

        scored_df = expanded.copy()
        scored_df["label"]           = [s["label"]           for s in scores]
        scored_df["confidence"]      = [s["confidence"]      for s in scores]
        scored_df["sentiment_score"] = [s["sentiment_score"] for s in scores]

        scored_df.to_parquet(scored_cache)
        print(f"\n  Saved scored articles to {scored_cache}")

    # ── Step 4: Aggregate to daily ────────────────────────────────────────────
    print("\nAggregating to daily sentiment scores...")
    features = pd.read_parquet("data/processed/features_daily.parquet")
    features = features.reset_index()
    all_dates = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None).unique()
    all_dates = sorted(all_dates)

    daily_sentiment = aggregate_daily_sentiment(scored_df, SYMBOLS, all_dates)

    print(f"  Daily sentiment shape: {daily_sentiment.shape}")
    print(f"  Coverage: {(daily_sentiment['article_count'] > 0).mean():.1%} "
          f"of stock-days have at least one article")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    daily_sentiment.to_parquet("data/processed/sentiment_daily.parquet", index=False)
    print("\nSaved: data/processed/sentiment_daily.parquet")

    # Quick sanity check
    print("\nSample sentiment scores:")
    sample = daily_sentiment[daily_sentiment["article_count"] > 0].head(10)
    print(sample[["date", "symbol", "sentiment_mean", "article_count",
                  "sentiment_pos_pct"]].to_string(index=False))

    print("\nDone. Next step: retrain ml_signal.py with sentiment features.")