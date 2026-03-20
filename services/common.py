"""
services/common.py
==================
Shared utilities for AlphaForge microservices.

Provides:
    - Redis connection helper (pub/sub for inter-service events)
    - TimescaleDB connection helper
    - Structured logging setup
    - Service heartbeat / health check
"""

import os
import sys
import json
import logging
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(service_name: str) -> logging.Logger:
    """Create a structured logger for a service."""
    logger = logging.getLogger(f"alphaforge.{service_name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = f"%(asctime)s [{service_name.upper()}] %(levelname)-8s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
def get_redis():
    """Return a Redis client. Falls back gracefully if redis is unavailable."""
    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        return redis.Redis(host=host, port=port, decode_responses=True)
    except Exception:
        return None


def publish_event(channel: str, data: dict):
    """Publish a JSON event to a Redis channel."""
    r = get_redis()
    if r is not None:
        try:
            r.publish(channel, json.dumps(data))
        except Exception:
            pass


def wait_for_event(channel: str, timeout: int = 300) -> dict | None:
    """Block until a message arrives on a Redis channel (or timeout)."""
    r = get_redis()
    if r is None:
        return None
    try:
        pubsub = r.pubsub()
        pubsub.subscribe(channel)
        for message in pubsub.listen():
            if message["type"] == "message":
                return json.loads(message["data"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TimescaleDB
# ---------------------------------------------------------------------------
def get_db_url() -> str:
    """Build the TimescaleDB connection URL from env vars."""
    user = os.getenv("POSTGRES_USER", "alphaforge")
    pw = os.getenv("POSTGRES_PASSWORD", "alphaforge")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "alphaforge")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------
def report_status(service_name: str, status: str, details: dict | None = None):
    """Publish a service health status event to Redis."""
    event = {
        "service": service_name,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if details:
        event.update(details)
    publish_event("alphaforge:health", event)
