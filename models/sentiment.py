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

# ── 1. FETCH NEWS FROM ALPACA ─────────────────────────────────────────────────

def fetch_news_alpaca(symbols, start_date="2020-01-01", end_date="2025-12-31",
                      limit_per_request=50):
    """
    Fetch financial news articles from Alpaca News API.
    
    Alpaca returns news with headline, summary, and full content.
    We'll use headline + summary as input to FinBERT — full content
    is often too long and adds noise.
    
    Rate limiting: Alpaca allows ~200 requests/minute on free tier.
    We add a small sleep between requests to stay safe.
    """
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    client = NewsClient(api_key=api_key, secret_key=secret_key)

    all_articles = []
    
    # Fetch in monthly chunks to avoid hitting request limits
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    current = start
    chunk_num = 1
    
    while current < end:
        chunk_end = min(current + timedelta(days=30), end)
        
        print(f"  Fetching news chunk {chunk_num}: "
              f"{current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}", 
              end="\r")

        try:
            request = NewsRequest(
                symbols=",".join(symbols),
                start=current.strftime("%Y-%m-%dT00:00:00Z"),
                end=chunk_end.strftime("%Y-%m-%dT00:00:00Z"),
                limit=limit_per_request,
                sort="DESC"
            )
            news = client.get_news(request)

            # Response is a NewsSet — data lives in news.data['news']
            articles = []
            for key, val in news:
                if key == "data" and isinstance(val, dict) and "news" in val:
                    articles = val["news"]
                    break

            for article in articles:
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
            print(f"\n  Warning: chunk {chunk_num} failed: {e}")

        current = chunk_end + timedelta(days=1)
        chunk_num += 1
        time.sleep(0.3)  # stay within rate limits

    print(f"\n  Fetched {len(all_articles):,} articles total")
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

def aggregate_daily_sentiment(scored_df, symbols, all_dates):
    """
    Convert per-article sentiment into per-stock per-day features.
    
    Features we create:
    - sentiment_mean:    avg sentiment score across all articles that day
    - sentiment_pos_pct: % of articles that were positive
    - sentiment_neg_pct: % of articles that were negative  
    - article_count:     number of articles (high = more news attention)
    - sentiment_std:     disagreement between articles (high = uncertainty)
    
    For days with no news we fill with 0 (neutral) — absence of news
    is itself a signal worth preserving as a zero rather than NaN.
    """
    daily = scored_df.groupby(["date", "symbol"]).agg(
        sentiment_mean    = ("sentiment_score", "mean"),
        sentiment_std     = ("sentiment_score", "std"),
        sentiment_pos_pct = ("label", lambda x: (x == "positive").mean()),
        sentiment_neg_pct = ("label", lambda x: (x == "negative").mean()),
        article_count     = ("sentiment_score", "count")
    ).reset_index()

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
        "article_count":     0.0
    })

    # Rolling 3-day sentiment — smooths out single-day noise
    full = full.sort_values(["symbol", "date"])
    full["sentiment_3d"] = (
        full.groupby("symbol")["sentiment_mean"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return full


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