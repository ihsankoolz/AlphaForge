"""
services/dashboard/main.py
==========================
Dashboard microservice launcher.

Wraps the existing Streamlit app (dashboard/app.py) for Docker deployment.
In production, Streamlit is launched directly via docker-compose CMD.

Can run standalone:
    python -m services.dashboard.main
    # or: streamlit run dashboard/app.py
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.common import get_logger, report_status

logger = get_logger("dashboard")


def run():
    """Launch the Streamlit dashboard."""
    logger.info("Starting Streamlit dashboard")
    report_status("dashboard", "running")

    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dashboard", "app.py",
    )

    if not os.path.exists(app_path):
        logger.error(f"Dashboard app not found at {app_path}")
        report_status("dashboard", "error")
        return

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ])


if __name__ == "__main__":
    run()
