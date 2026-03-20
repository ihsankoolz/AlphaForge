###############################################################################
# AlphaForge — Base image for all services
###############################################################################
# Single Dockerfile shared by every service in docker-compose.yml.
# Each service overrides the CMD to run its own entrypoint.
###############################################################################

FROM python:3.11-slim

WORKDIR /app

# System deps for numpy, pandas, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        xgboost lightgbm hmmlearn transformers torch \
        streamlit plotly redis psycopg2-binary \
    && rm -rf /root/.cache/pip

# Copy project source
COPY . .

# Default: run the full daily pipeline
CMD ["python", "-m", "execution.run_daily"]
