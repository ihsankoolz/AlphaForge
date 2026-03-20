"""
services/regime/main.py
=======================
Regime detection microservice.

Loads hmm_model.pkl and predicts today's market regime (bull/choppy/bear).
Publishes the regime to Redis so signals and risk services can consume it.

Can run standalone:
    python -m services.regime.main
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from services.common import get_logger, publish_event, report_status
from config.settings import MODELS_DIR, DATA_DIR

logger = get_logger("regime")


def run():
    """Execute the regime detection stage."""
    logger.info("Starting regime detection")
    report_status("regime", "running")

    # Load features (needed for regime prediction)
    try:
        path = os.path.join(DATA_DIR, "features_daily.parquet")
        features_df = pd.read_parquet(path)
        if isinstance(features_df.index, pd.MultiIndex):
            features_df = features_df.reset_index()
    except Exception as e:
        logger.error(f"Could not load features for regime detection: {e}")
        regime = "choppy"
        publish_event("alphaforge:pipeline", {"stage": "regime", "status": "done", "regime": regime})
        report_status("regime", "fallback", {"regime": regime})
        return

    model_path = os.path.join(MODELS_DIR, "hmm_model.pkl")

    try:
        from models.regime_hmm import predict_regime
        regime = predict_regime(features_df, model_path=model_path)
        regime = regime.lower().strip()
        assert regime in ("bull", "choppy", "bear")
        logger.info(f"Regime: {regime.upper()}")
    except Exception as e:
        logger.warning(f"Regime detection failed: {e} — defaulting to choppy")
        regime = "choppy"

    publish_event("alphaforge:pipeline", {"stage": "regime", "status": "done", "regime": regime})
    report_status("regime", "done", {"regime": regime})


if __name__ == "__main__":
    run()
