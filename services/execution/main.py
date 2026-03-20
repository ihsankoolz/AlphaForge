"""
services/execution/main.py
==========================
Order execution microservice.

Receives target weights (from risk service) and executes trades via
Alpaca's paper trading API. Includes circuit breaker checks.

Can run standalone:
    python -m services.execution.main
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.common import get_logger, publish_event, report_status

logger = get_logger("execution")


def run(dry_run: bool = True):
    """Execute the order placement stage."""
    mode = "DRY RUN" if dry_run else "LIVE"
    logger.info(f"Starting order execution ({mode})")
    report_status("execution", "running", {"mode": mode})

    # In a full deployment, target_weights would come from Redis or a shared store.
    # For now, this service wraps the existing OrderManager interface.
    try:
        from execution.order_manager import OrderManager
        manager = OrderManager(dry_run=dry_run)

        # Target weights would be read from Redis/shared state in production.
        # This entrypoint demonstrates the service boundary.
        target_weights = {}  # populated by pipeline orchestrator

        success = manager.rebalance(target_weights)
        status = "done" if success else "partial"
        logger.info(f"Execution {status}")

        publish_event("alphaforge:pipeline", {
            "stage": "execution",
            "status": status,
            "mode": mode,
        })
        report_status("execution", status)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        publish_event("alphaforge:pipeline", {"stage": "execution", "status": "error"})
        report_status("execution", "error", {"error": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run(dry_run=not args.live)
