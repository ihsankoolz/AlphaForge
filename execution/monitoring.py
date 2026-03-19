"""
execution/monitoring.py
=======================
Basic monitoring and alerting for the AlphaForge daily pipeline.

Provides:
    1. Email alerts on pipeline failure (optional — requires SMTP config)
    2. Daily summary report written to logs/
    3. Pipeline health tracking (consecutive failures, drift detection)

Configuration:
    Set ALERT_EMAIL_ENABLED=True and configure SMTP settings in
    config/settings.py to enable email alerts.
"""

import os
import sys
import json
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    LOGS_DIR,
    ALERT_EMAIL_ENABLED,
    ALERT_EMAIL_TO,
    ALERT_SMTP_HOST,
    ALERT_SMTP_PORT,
    ALERT_SMTP_USER,
    ALERT_SMTP_PASS,
)

log = logging.getLogger("monitoring")

# File to track pipeline state across runs
STATE_FILE = os.path.join(LOGS_DIR, "pipeline_state.json")


def _load_state() -> dict:
    """Load persistent pipeline state from disk."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "consecutive_failures": 0,
        "last_success": None,
        "last_failure": None,
        "last_weights": {},
        "last_regime": None,
    }


def _save_state(state: dict):
    """Persist pipeline state to disk."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def send_alert(subject: str, body: str):
    """
    Send an email alert. Only fires if ALERT_EMAIL_ENABLED is True
    and SMTP settings are configured in config/settings.py.

    Fails silently if email cannot be sent — alerting should never
    crash the pipeline.
    """
    if not ALERT_EMAIL_ENABLED:
        log.info(f"Alert (email disabled): {subject}")
        return

    if not all([ALERT_EMAIL_TO, ALERT_SMTP_HOST, ALERT_SMTP_USER]):
        log.warning("Email alerting enabled but SMTP not configured")
        return

    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[AlphaForge] {subject}"
        msg["From"]    = ALERT_SMTP_USER
        msg["To"]      = ALERT_EMAIL_TO

        with smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT) as server:
            server.starttls()
            server.login(ALERT_SMTP_USER, ALERT_SMTP_PASS)
            server.send_message(msg)

        log.info(f"Alert email sent: {subject}")
    except Exception as e:
        log.warning(f"Failed to send alert email: {e}")


def record_success(
    regime: str,
    target_weights: dict,
    n_positions: int,
    elapsed_seconds: float,
):
    """
    Record a successful pipeline run. Resets the consecutive failure counter.
    """
    state = _load_state()
    state["consecutive_failures"] = 0
    state["last_success"] = datetime.now().isoformat()
    state["last_weights"] = target_weights
    state["last_regime"]  = regime
    _save_state(state)

    log.info(
        f"Pipeline success recorded — regime={regime}, "
        f"{n_positions} positions, {elapsed_seconds:.0f}s"
    )


def record_failure(stage: str, error: str):
    """
    Record a pipeline failure. Sends alert if consecutive failures exceed 2.
    """
    state = _load_state()
    state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
    state["last_failure"] = datetime.now().isoformat()
    state["last_failure_stage"] = stage
    state["last_failure_error"] = error
    _save_state(state)

    n_failures = state["consecutive_failures"]
    log.error(
        f"Pipeline failure #{n_failures} — stage={stage}, error={error}"
    )

    if n_failures >= 2:
        send_alert(
            subject=f"Pipeline failed {n_failures}x consecutively",
            body=(
                f"AlphaForge pipeline has failed {n_failures} times in a row.\n\n"
                f"Latest failure:\n"
                f"  Stage: {stage}\n"
                f"  Error: {error}\n"
                f"  Time:  {datetime.now()}\n\n"
                f"Last successful run: {state.get('last_success', 'never')}\n\n"
                f"Action required: check logs at {LOGS_DIR}"
            ),
        )


def get_previous_weights() -> dict:
    """
    Retrieve yesterday's target weights for circuit breaker comparison.
    Returns empty dict if no previous state exists.
    """
    state = _load_state()
    return state.get("last_weights", {})


def write_daily_summary(
    regime: str,
    target_weights: dict,
    signals_count: int,
    elapsed_seconds: float,
    execution_ok: bool,
    dry_run: bool,
):
    """
    Write a structured daily summary to logs/summary_YYYYMMDD.json.

    This file can be read by the dashboard or external monitoring tools
    to track pipeline health over time.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")
    summary_path = os.path.join(LOGS_DIR, f"summary_{today_str}.json")

    summary = {
        "date": date.today().isoformat(),
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run" if dry_run else "live",
        "regime": regime,
        "signals_above_threshold": signals_count,
        "n_positions": len(target_weights),
        "total_invested_pct": round(sum(target_weights.values()) * 100, 1),
        "top_positions": dict(
            sorted(target_weights.items(), key=lambda x: -x[1])[:5]
        ),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "execution_ok": execution_ok,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Daily summary written to {summary_path}")
