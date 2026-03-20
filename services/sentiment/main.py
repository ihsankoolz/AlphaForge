"""
services/sentiment/main.py
==========================
Sentiment scoring microservice.

Fetches recent news from Alpaca, scores via FinBERT, and updates
sentiment_daily.parquet.  Non-fatal — downstream services handle None.

Can run standalone:
    python -m services.sentiment.main
"""

import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from services.common import get_logger, publish_event, report_status
from config.settings import SYMBOLS, SENTIMENT_LOOKBACK_DAYS, DATA_DIR

logger = get_logger("sentiment")


def run():
    """Execute the sentiment scoring stage."""
    logger.info(f"Starting sentiment pipeline for {len(SYMBOLS)} symbols")
    report_status("sentiment", "running")

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=SENTIMENT_LOOKBACK_DAYS)

    # Try live pipeline
    try:
        from models.sentiment import run_sentiment_pipeline
        sent = run_sentiment_pipeline(SYMBOLS, start_date=start_dt, end_date=end_dt)
        logger.info(f"Sentiment OK — {len(sent):,} rows")
        publish_event("alphaforge:pipeline", {"stage": "sentiment", "status": "done", "rows": len(sent)})
        report_status("sentiment", "done", {"rows": len(sent)})
        return
    except ImportError:
        logger.warning("run_sentiment_pipeline not found — falling back to cache")
    except Exception as e:
        logger.warning(f"Sentiment pipeline error: {e}")

    # Fallback
    try:
        path = os.path.join(DATA_DIR, "sentiment_daily.parquet")
        sent = pd.read_parquet(path)
        logger.info(f"Loaded cached sentiment — {len(sent):,} rows")
        publish_event("alphaforge:pipeline", {"stage": "sentiment", "status": "cached"})
        report_status("sentiment", "cached")
    except Exception:
        logger.warning("No sentiment data available — continuing without")
        publish_event("alphaforge:pipeline", {"stage": "sentiment", "status": "skipped"})
        report_status("sentiment", "skipped")


if __name__ == "__main__":
    run()
