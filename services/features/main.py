"""
services/features/main.py
=========================
Feature engineering microservice.

Recomputes the 22-feature matrix from TimescaleDB OHLCV data.
Publishes 'features:done' event with row count when complete.

Can run standalone:
    python -m services.features.main
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from services.common import get_logger, publish_event, report_status
from config.settings import SYMBOLS, DATA_DIR

logger = get_logger("features")


def run():
    """Execute the feature engineering stage."""
    logger.info(f"Starting feature computation for {len(SYMBOLS)} symbols")
    report_status("features", "running")

    features_df = None

    # Try live recomputation
    try:
        from research.features.engineer import compute_features
        raw = compute_features(SYMBOLS)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index()
        features_df = raw
        logger.info(f"Features computed — {len(features_df):,} rows")
    except ImportError:
        logger.warning("compute_features not found — falling back to cache")
    except Exception as e:
        logger.error(f"Feature computation error: {e}", exc_info=True)

    # Fallback to cached parquet
    if features_df is None:
        try:
            path = os.path.join(DATA_DIR, "features_daily.parquet")
            features_df = pd.read_parquet(path)
            if isinstance(features_df.index, pd.MultiIndex):
                features_df = features_df.reset_index()
            logger.info(f"Loaded cached features — {len(features_df):,} rows")
        except Exception as e:
            logger.error(f"FATAL — could not load features: {e}")
            report_status("features", "error", {"error": str(e)})
            publish_event("alphaforge:pipeline", {"stage": "features", "status": "error"})
            return

    publish_event("alphaforge:pipeline", {
        "stage": "features",
        "status": "done",
        "rows": len(features_df),
    })
    report_status("features", "done", {"rows": len(features_df)})


if __name__ == "__main__":
    run()
