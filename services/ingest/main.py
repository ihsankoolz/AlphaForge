"""
services/ingest/main.py
=======================
Data ingestion microservice.

Fetches OHLCV data from Alpaca and upserts into TimescaleDB.
Publishes 'ingest:done' event when complete so downstream services can start.

Can run standalone:
    python -m services.ingest.main
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.common import get_logger, publish_event, report_status
from config.settings import SYMBOLS

logger = get_logger("ingest")


def run():
    """Execute the ingest stage."""
    logger.info(f"Starting ingest for {len(SYMBOLS)} symbols")
    report_status("ingest", "running")

    try:
        from data.ingest import ingest_latest
        ingest_latest(SYMBOLS, days_back=5)
        logger.info("Ingest complete")
        publish_event("alphaforge:pipeline", {"stage": "ingest", "status": "done"})
        report_status("ingest", "done")
    except ImportError:
        logger.warning("data.ingest.ingest_latest not found — skipping")
        publish_event("alphaforge:pipeline", {"stage": "ingest", "status": "skipped"})
        report_status("ingest", "skipped")
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        publish_event("alphaforge:pipeline", {"stage": "ingest", "status": "error", "error": str(e)})
        report_status("ingest", "error", {"error": str(e)})


if __name__ == "__main__":
    run()
