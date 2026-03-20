"""
services/risk/main.py
=====================
Risk layer microservice.

Runs Kelly Criterion + Markowitz portfolio optimization to convert
ML signal probabilities into target portfolio weights.

Can run standalone:
    python -m services.risk.main
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from services.common import get_logger, publish_event, report_status
from config.settings import DATA_DIR, MIN_STOCKS

logger = get_logger("risk")


def run():
    """Execute the risk layer stage."""
    logger.info("Starting risk layer (Kelly + Markowitz)")
    report_status("risk", "running")

    # Load latest signals
    try:
        signals_path = os.path.join(DATA_DIR, "latest_signals.parquet")
        signals_df = pd.read_parquet(signals_path)
    except Exception as e:
        logger.error(f"Could not load signals: {e}")
        report_status("risk", "error")
        return

    # Load features for volatility
    try:
        features_df = pd.read_parquet(os.path.join(DATA_DIR, "features_daily.parquet"))
        if isinstance(features_df.index, pd.MultiIndex):
            features_df = features_df.reset_index()
    except Exception as e:
        logger.error(f"Could not load features: {e}")
        report_status("risk", "error")
        return

    # Kelly sizing
    try:
        from risk.position_sizer import compute_kelly_weights
        kelly_weights = compute_kelly_weights(signals_df, features_df)
        logger.info(f"Kelly — {len(kelly_weights)} positions")
    except Exception as e:
        logger.error(f"Kelly sizing failed: {e}")
        publish_event("alphaforge:pipeline", {"stage": "risk", "status": "error"})
        report_status("risk", "error")
        return

    if len(kelly_weights) < MIN_STOCKS:
        logger.info(f"Only {len(kelly_weights)} positions — skipping Markowitz")
        final_weights = kelly_weights
    else:
        try:
            from risk.portfolio_optimiser import optimize_weights
            final_weights = optimize_weights(
                kelly_weights=kelly_weights,
                features_df=features_df,
                regime="bull",
            )
            logger.info(f"Markowitz — {len(final_weights)} final positions")
        except Exception as e:
            logger.warning(f"Markowitz failed: {e} — using Kelly weights")
            final_weights = kelly_weights

    invested = sum(final_weights.values()) * 100
    logger.info(f"Target: {len(final_weights)} positions, {invested:.1f}% invested")

    publish_event("alphaforge:pipeline", {
        "stage": "risk",
        "status": "done",
        "positions": len(final_weights),
        "invested_pct": round(invested, 1),
    })
    report_status("risk", "done", {"positions": len(final_weights)})


if __name__ == "__main__":
    run()
