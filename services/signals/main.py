"""
services/signals/main.py
========================
ML signal generation microservice.

Loads the ensemble model (XGBoost + LightGBM) and generates pred_proba
for every symbol in the universe. Publishes signal counts to Redis.

Can run standalone:
    python -m services.signals.main
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from services.common import get_logger, publish_event, report_status
from config.settings import MODELS_DIR, DATA_DIR

logger = get_logger("signals")


def run():
    """Execute the ML signal generation stage."""
    logger.info("Starting ML signal generation")
    report_status("signals", "running")

    # Load features
    try:
        features_df = pd.read_parquet(os.path.join(DATA_DIR, "features_daily.parquet"))
        if isinstance(features_df.index, pd.MultiIndex):
            features_df = features_df.reset_index()
    except Exception as e:
        logger.error(f"Could not load features: {e}")
        report_status("signals", "error")
        return

    # Load sentiment (optional)
    sentiment_df = None
    try:
        sentiment_df = pd.read_parquet(os.path.join(DATA_DIR, "sentiment_daily.parquet"))
        if isinstance(sentiment_df.index, pd.MultiIndex):
            sentiment_df = sentiment_df.reset_index()
    except Exception:
        logger.info("No sentiment data — using base features only")

    # Generate signals
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    try:
        from models.ml_signal import generate_signals
        signals_df = generate_signals(
            features_df=features_df,
            sentiment_df=sentiment_df,
            model_path=model_path,
        )
        above = int((signals_df["pred_proba"] >= 0.35).sum())
        logger.info(f"Signals generated — {len(signals_df)} symbols, {above} above threshold")

        publish_event("alphaforge:pipeline", {
            "stage": "signals",
            "status": "done",
            "total": len(signals_df),
            "above_threshold": above,
        })
        report_status("signals", "done", {"total": len(signals_df), "above_threshold": above})
    except Exception as e:
        logger.error(f"Signal generation failed: {e}", exc_info=True)
        publish_event("alphaforge:pipeline", {"stage": "signals", "status": "error"})
        report_status("signals", "error", {"error": str(e)})


if __name__ == "__main__":
    run()
