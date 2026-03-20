"""
config/settings.py
==================
Single source of truth for all AlphaForge constants, thresholds, and
symbol lists.

Every module imports from here instead of defining its own copy.
This prevents silent divergence when a constant is changed in one file
but not another.

Usage:
    from config.settings import SYMBOLS, KELLY_FRACTION, MAX_POSITION
"""

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Stock universe — THE canonical list.  Update ONLY here.
# ---------------------------------------------------------------------------
SYMBOLS_ORIGINAL = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
    "JPM",  "GS",   "BAC",   "MS",   "BLK",
    "JNJ",  "UNH",  "PFE",   "ABBV",
    "XOM",  "CVX",  "COP",
    "MCD",  "NKE",  "SBUX",  "WMT",  "COST",
    "CAT",  "BA",   "HON",   "GE",
    "SPY",  "QQQ",
]

SYMBOLS_BATCH1 = [
    "TSLA", "AVGO", "AMD",  "ORCL", "ADBE", "CRM",  "NFLX", "QCOM", "TXN",  "CSCO",
    "IBM",  "NOW",  "AMAT", "MU",   "INTC",
    "V",    "MA",   "C",    "WFC",  "AXP",  "SCHW", "COF",  "BK",
    "LLY",  "MRK",  "AMGN", "BMY",  "GILD", "TMO",  "MDT",  "ISRG",
    "HD",   "TGT",  "LOW",  "BKNG", "TJX",  "MAR",
    "RTX",  "LMT",  "UPS",  "DE",   "ETN",
    "NEE",  "DUK",
    "AMT",  "PLD",
    "LIN",  "NEM",
    "DIS",  "VZ",
]

# Full active universe — modules should import this, not build their own list
SYMBOLS = SYMBOLS_ORIGINAL + SYMBOLS_BATCH1

# ---------------------------------------------------------------------------
# Kelly Criterion / Position sizing
# ---------------------------------------------------------------------------
KELLY_FRACTION = 0.5       # half-Kelly safety buffer
KELLY_ODDS     = 2.0       # default prior; overridden by empirical calibration
MIN_PROB       = 0.35      # ML threshold — below this = no position
MAX_POSITION   = 0.15      # single-stock cap (15% of portfolio)
VOL_LOOKBACK   = 20        # days to estimate stock volatility
VOL_TARGET     = 0.01      # target 1% daily vol per position
MAX_POSITIONS  = 10        # max concurrent positions

# ---------------------------------------------------------------------------
# Markowitz / Portfolio optimisation
# ---------------------------------------------------------------------------
COV_LOOKBACK   = 60        # days of return history for covariance matrix
MIN_WEIGHT     = 0.02      # minimum position (no slivers below 2%)
MAX_WEIGHT     = MAX_POSITION  # inherit from position sizer
RISK_FREE_RATE = 0.05 / 252   # daily risk-free (~5% annual)
MIN_STOCKS     = 2         # need ≥2 stocks to optimise correlations

# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------
MAX_POSITION_PCT   = MAX_POSITION  # alias for order_manager
DAILY_LOSS_LIMIT   = 0.03          # halt if portfolio down >3% today
MIN_ORDER_VALUE    = 50            # skip orders below $50
MIN_WEIGHT_CHANGE  = 0.01          # skip rebalance if delta < 1%

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
REGIME_MAP = {"bull": 0, "choppy": 1, "bear": 2}
HMM_FEATURE_COLS = [
    "mean_return", "vol_return", "mean_volatility",
    "mean_rsi", "mean_macd_hist", "mean_bb_pct", "mean_volume_ratio",
]

# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------
SENTIMENT_LOOKBACK_DAYS = 7
SENTIMENT_FEATURE_COLS = [
    "sentiment_mean", "sentiment_std",
    "sentiment_pos_pct", "sentiment_neg_pct",
    "article_count", "sentiment_3d",
    "sentiment_freshness",
]
SENTIMENT_FILL_VALUES = {
    "sentiment_mean":    0.0,
    "sentiment_std":     0.0,
    "sentiment_pos_pct": 0.0,
    "sentiment_neg_pct": 0.0,
    "article_count":     0.0,
    "sentiment_3d":      0.0,
    "sentiment_freshness": 0.0,
}

# ---------------------------------------------------------------------------
# ML signal
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Raw price features
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    # Technical indicators
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_pct", "bb_bandwidth",
    "volume_ratio", "spy_correlation",
    # Regime context
    "regime_encoded",
    # Rule-based signals
    "momentum_signal", "mr_signal",
    # Sentiment features
    "sentiment_mean", "sentiment_std",
    "sentiment_pos_pct", "sentiment_neg_pct",
    "article_count", "sentiment_3d",
    # News presence indicator (added in code review improvements)
    "has_news",
]

# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------
COV_SHRINKAGE_METHOD = "ledoit_wolf"  # "sample" (raw) or "ledoit_wolf" (shrinkage)

# ---------------------------------------------------------------------------
# Short selling
# ---------------------------------------------------------------------------
ALLOW_SHORT_SELLING  = False   # enable after universe expands to 129+ stocks
SHORT_MAX_POSITION   = 0.10   # max short position (10% of portfolio)
SHORT_MIN_PROB       = 0.20   # short stocks with pred_proba below this
SHORT_KELLY_FRACTION = 0.25   # quarter-Kelly for shorts (more conservative)

# ---------------------------------------------------------------------------
# Sentiment staleness
# ---------------------------------------------------------------------------
SENTIMENT_DECAY_HALFLIFE_HOURS = 18.0  # article weight halves every 18 hours
SENTIMENT_MAX_AGE_HOURS        = 72.0  # discard articles older than 72 hours

# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
DEFAULT_HALF_SPREAD_BP = 2.5    # basis points — default spread for large-cap
MARKET_IMPACT_SCALE    = 0.01   # impact model scale factor
TURNOVER_PENALTY_LAMBDA = 0.005 # Markowitz turnover penalty strength

# ---------------------------------------------------------------------------
# Pipeline circuit breakers
# ---------------------------------------------------------------------------
CB_MAX_SIGNAL_RATIO     = 0.80   # halt if >80% of stocks above threshold
CB_MAX_ALLOCATION_JUMP  = 0.30   # halt if total allocation jumps >30% vs yesterday
CB_MAX_DAILY_TURNOVER   = 0.50   # warn if portfolio turnover exceeds 50%
CB_MIN_EXPECTED_SIGNALS = 3      # warn if fewer signals than expected

# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------
MODEL_VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
MAX_MODEL_VERSIONS = 10  # keep last N model versions

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
ALERT_EMAIL_ENABLED = False       # set True + configure SMTP to enable
ALERT_EMAIL_TO      = ""          # recipient email
ALERT_SMTP_HOST     = ""          # SMTP server
ALERT_SMTP_PORT     = 587
ALERT_SMTP_USER     = ""
ALERT_SMTP_PASS     = ""
