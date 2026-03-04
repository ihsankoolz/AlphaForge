#!/usr/bin/env python3
"""
execution/run_daily.py
======================
AlphaForge daily pipeline orchestrator.

Runs once each morning, scheduled via Windows Task Scheduler:
    9:25 AM ET  → script starts, ingest + compute begin
    9:30 AM ET  → market opens, orders placed
    ~9:35 AM ET → script exits normally

Pipeline stages:
    1. Ingest      — fetch yesterday's OHLCV into TimescaleDB
    2. Features    — recompute features_daily.parquet
    3. Sentiment   — fetch/score recent news, update sentiment_daily.parquet
    4. Regime      — predict today's regime via HMM (bull / choppy / bear)
    5. ML Signals  — generate XGBoost v3 pred_proba per symbol (skipped if choppy)
    6. Risk Layer  — Kelly + Markowitz → target weights    (skipped if choppy)
    7. Execute     — rebalance portfolio via order_manager

Usage:
    python execution/run_daily.py           # dry run — logs only, no real orders
    python execution/run_daily.py --live    # live paper trading

Scheduling (Windows Task Scheduler):
    Program  : python
    Arguments: C:\\...\\AlphaForge\\execution\\run_daily.py --live
    Trigger  : Daily, 9:25 AM ET (weekdays only)

-------------------------------------------------------------------------------
EXPECTED MODULE INTERFACES
-------------------------------------------------------------------------------
Each module below needs to expose a live-prediction function in addition to
its existing batch/backtest functions. Add these if they don't already exist:

data/ingest.py
    ingest_latest(symbols: list[str], days_back: int = 5) -> None
        Fetch the last `days_back` trading days for each symbol and upsert
        into TimescaleDB. Handles weekends/holidays gracefully.

research/features/engineer.py
    compute_features(symbols: list[str]) -> pd.DataFrame
        Recompute all 22 features from TimescaleDB data.
        Returns DataFrame with MultiIndex [time, symbol] (UTC).
        Also overwrites data/processed/features_daily.parquet.

models/regime_hmm.py
    predict_regime(features_df: pd.DataFrame, model_path: str) -> str
        Load hmm_model.pkl, aggregate features_df to market-level for the
        most recent date, predict and return 'bull', 'choppy', or 'bear'.

models/sentiment.py
    run_sentiment_pipeline(symbols, start_date, end_date) -> pd.DataFrame
        Fetch news from Alpaca API, score via FinBERT, return daily sentiment
        features per symbol. Columns: [date, symbol, sentiment_mean,
        sentiment_std, sentiment_pos_pct, sentiment_neg_pct, article_count,
        sentiment_3d].
        Also updates data/processed/sentiment_daily.parquet.

models/ml_signal.py
    generate_signals(
        features_df:   pd.DataFrame,
        sentiment_df:  pd.DataFrame | None,
        model_path:    str,
    ) -> pd.DataFrame
        Load xgb_model.pkl. Build the v3 feature vector for the latest date
        (base features + regime label + rule signals + sentiment). Return
        DataFrame with columns: [symbol, pred_proba]. One row per symbol.

risk/position_sizer.py
    compute_kelly_weights(
        signals_df:  pd.DataFrame,   # [symbol, pred_proba]
        features_df: pd.DataFrame,   # for volatility lookback
    ) -> dict[str, float]
        Apply Kelly Criterion (half-Kelly) + volatility adjustment.
        Return {symbol: weight} for symbols with pred_proba >= MIN_PROB (0.35).

risk/portfolio_optimiser.py
    optimize_weights(
        kelly_weights: dict[str, float],
        features_df:   pd.DataFrame,
        regime:        str,
    ) -> dict[str, float]
        Markowitz Sharpe maximization over the Kelly-selected stocks.
        Applies regime-specific caps (bull: 15%, bear: 70% total).
        Return {symbol: final_weight}.

execution/order_manager.py
    class OrderManager:
        def __init__(self, dry_run: bool = True): ...
        def rebalance(self, target_weights: dict[str, float]) -> bool:
            Compute position deltas vs current holdings, place orders.
            Handles: market hours check, daily loss limit, min order $50,
            sells before buys. Returns True on success.
-------------------------------------------------------------------------------
"""

import sys
import os
import logging
import argparse
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import pytz

# ---------------------------------------------------------------------------
# sys.path — make all AlphaForge modules importable from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, 'research')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOLS = [
    # Original 29
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN',
    'JPM',  'GS',   'BAC',   'MS',   'BLK',
    'JNJ',  'UNH',  'PFE',   'ABBV',
    'XOM',  'CVX',  'COP',
    'MCD',  'NKE',  'SBUX',  'WMT',  'COST',
    'CAT',  'BA',   'HON',   'GE',
    'SPY',  'QQQ',
    # Batch 1 — 50 new symbols
    'TSLA', 'AVGO', 'AMD',  'ORCL', 'ADBE', 'CRM',  'NFLX', 'QCOM', 'TXN',  'CSCO',
    'IBM',  'NOW',  'AMAT', 'MU',   'INTC',
    'V',    'MA',   'C',    'WFC',  'AXP',  'SCHW', 'COF',  'BK',
    'LLY',  'MRK',  'AMGN', 'BMY',  'GILD', 'TMO',  'MDT',  'ISRG',
    'HD',   'TGT',  'LOW',  'BKNG', 'TJX',  'MAR',
    'RTX',  'LMT',  'UPS',  'DE',   'ETN',
    'NEE',  'DUK',
    'AMT',  'PLD',
    'LIN',  'NEM',
    'DIS',  'VZ',
]

DATA_DIR   = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR   = os.path.join(PROJECT_ROOT, 'logs')

ET_TZ = pytz.timezone('America/New_York')

# Sentiment fetch window: last N days (covers weekends + any API gaps)
SENTIMENT_LOOKBACK_DAYS = 7

# Minimum number of Kelly-selected positions to bother running Markowitz.
# With <2 stocks there's nothing to correlate.
MIN_POSITIONS_FOR_MARKOWITZ = 2


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(dry_run: bool) -> logging.Logger:
    """
    Configure file + console logging.
    Log file: logs/trading_YYYYMMDD.log
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    today_str = date.today().strftime('%Y%m%d')
    log_file  = os.path.join(LOGS_DIR, f'trading_{today_str}.log')
    mode_tag  = 'DRY_RUN' if dry_run else 'LIVE'

    fmt = f'%(asctime)s [{mode_tag}] %(levelname)-8s | %(message)s'

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger('run_daily')


def divider(logger: logging.Logger, title: str) -> None:
    logger.info('─' * 60)
    logger.info(f'  {title}')
    logger.info('─' * 60)


# ---------------------------------------------------------------------------
# Helper: normalise timezone-aware timestamps to tz-naive dates
# ---------------------------------------------------------------------------
def _normalise_timestamps(df: pd.DataFrame, col: str = 'time') -> pd.DataFrame:
    """
    Strip UTC timezone from a timestamp column so it can be joined cleanly
    with tz-naive DataFrames. Always: .normalize() THEN .tz_localize(None).
    """
    if col in df.columns:
        ts = pd.to_datetime(df[col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        df[col] = ts.dt.normalize()
    return df


# ---------------------------------------------------------------------------
# STEP 1 — INGEST
# ---------------------------------------------------------------------------
def run_ingest(logger: logging.Logger) -> bool:
    """
    Fetch the last DAYS_BACK trading days of OHLCV data from Alpaca and
    upsert into TimescaleDB. Fetching 5 days covers weekends and US holidays
    without creating gaps.

    Returns True on success, False on failure.
    Failure here is non-fatal — downstream steps fall back to cached parquet.
    """
    divider(logger, 'STEP 1 — INGEST')
    days_back = 5
    try:
        from data.ingest import ingest_latest   # type: ignore
        ingest_latest(SYMBOLS, days_back=days_back)
        logger.info(f'Ingest OK — last {days_back} days, {len(SYMBOLS)} symbols')
        return True
    except ImportError:
        logger.warning('data.ingest.ingest_latest not found — add this function to data/ingest.py')
        logger.warning('Continuing with cached data')
        return False
    except Exception as exc:
        logger.error(f'Ingest error: {exc}', exc_info=True)
        logger.warning('Continuing with cached data')
        return False


# ---------------------------------------------------------------------------
# STEP 2 — FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def run_features(logger: logging.Logger) -> pd.DataFrame | None:
    """
    Recompute all 22 features from TimescaleDB and update features_daily.parquet.

    Falls back to cached features_daily.parquet if live recomputation fails.
    Returns a flat DataFrame with columns [time, symbol, <22 features>].
    Returns None only if both live and cached load fail (fatal).
    """
    divider(logger, 'STEP 2 — FEATURE ENGINEERING')

    # --- Try live recomputation first ---
    try:
        from research.features.engineer import compute_features   # type: ignore
        raw = compute_features(SYMBOLS)
        # Flatten MultiIndex [time, symbol] if present
        if isinstance(raw.index, pd.MultiIndex) and raw.index.names == ['time', 'symbol']:
            raw = raw.reset_index()
        features_df = _normalise_timestamps(raw, col='time')
        latest = features_df['time'].max().date()
        logger.info(f'Features computed — {len(features_df):,} rows, latest date: {latest}')
        return features_df
    except ImportError:
        logger.warning('research.features.engineer.compute_features not found — falling back to cache')
    except Exception as exc:
        logger.error(f'Feature engineering error: {exc}', exc_info=True)
        logger.warning('Falling back to cached features_daily.parquet')

    # --- Fallback: load cached parquet ---
    try:
        path = os.path.join(DATA_DIR, 'features_daily.parquet')
        raw = pd.read_parquet(path)
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.reset_index()
        features_df = _normalise_timestamps(raw, col='time')
        latest = features_df['time'].max().date()
        logger.info(f'Loaded cached features — {len(features_df):,} rows, latest: {latest}')
        return features_df
    except Exception as exc:
        logger.error(f'FATAL — could not load features: {exc}', exc_info=True)
        return None


# ---------------------------------------------------------------------------
# STEP 3 — SENTIMENT
# ---------------------------------------------------------------------------
def run_sentiment(logger: logging.Logger) -> pd.DataFrame | None:
    """
    Fetch last SENTIMENT_LOOKBACK_DAYS of news from Alpaca, score via FinBERT,
    and update sentiment_daily.parquet.

    Sentiment is NON-FATAL. XGBoost v3 was trained with sentiment features but
    can produce predictions on the base feature set (pred_proba will be slightly
    lower quality, still directionally correct).

    Returns DataFrame with columns:
        [date, symbol, sentiment_mean, sentiment_std, sentiment_pos_pct,
         sentiment_neg_pct, article_count, sentiment_3d]
    or None if both live and cached load fail.
    """
    divider(logger, 'STEP 3 — SENTIMENT')
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=SENTIMENT_LOOKBACK_DAYS)

    # --- Try live pipeline ---
    try:
        from models.sentiment import run_sentiment_pipeline   # type: ignore
        sent = run_sentiment_pipeline(SYMBOLS, start_date=start_dt, end_date=end_dt)
        sent = _normalise_timestamps(sent, col='date')
        latest = sent['date'].max().date()
        logger.info(f'Sentiment OK — {len(sent):,} rows, latest: {latest}')
        return sent
    except ImportError:
        logger.warning('models.sentiment.run_sentiment_pipeline not found — falling back to cache')
    except Exception as exc:
        logger.warning(f'Sentiment pipeline error (non-fatal): {exc}', exc_info=True)
        logger.info('Falling back to cached sentiment_daily.parquet')

    # --- Fallback: load cached parquet ---
    try:
        path = os.path.join(DATA_DIR, 'sentiment_daily.parquet')
        sent = pd.read_parquet(path)
        if isinstance(sent.index, pd.MultiIndex):
            sent = sent.reset_index()
        sent = _normalise_timestamps(sent, col='date')
        latest = sent['date'].max().date()
        staleness = (date.today() - latest).days
        logger.info(f'Loaded cached sentiment — latest: {latest} ({staleness}d stale)')
        if staleness > 3:
            logger.warning(f'Sentiment data is {staleness} days stale — signals may degrade')
        return sent
    except Exception as exc:
        logger.warning(f'Could not load cached sentiment: {exc}')
        logger.warning('Continuing without sentiment — XGBoost will use base features only')
        return None


# ---------------------------------------------------------------------------
# STEP 4 — REGIME DETECTION
# ---------------------------------------------------------------------------
def run_regime(features_df: pd.DataFrame, logger: logging.Logger) -> str:
    """
    Load hmm_model.pkl, compute market-level aggregate features for the most
    recent date, and return today's regime.

    Returns 'bull', 'choppy', or 'bear'.
    On ANY failure: returns 'choppy' — the safest fallback (goes to cash).
    Never raises.
    """
    divider(logger, 'STEP 4 — REGIME DETECTION')
    model_path = os.path.join(MODELS_DIR, 'hmm_model.pkl')

    try:
        from models.regime_hmm import predict_regime   # type: ignore
        regime = predict_regime(features_df, model_path=model_path)
        regime = regime.lower().strip()
        assert regime in ('bull', 'choppy', 'bear'), f'Unexpected regime: {regime!r}'
        logger.info(f'Regime: {regime.upper()}')
        return regime
    except ImportError:
        logger.warning('models.regime_hmm.predict_regime not found — add this function to models/regime_hmm.py')
    except AssertionError as exc:
        logger.error(f'Regime returned unexpected value: {exc}')
    except Exception as exc:
        logger.error(f'Regime detection error: {exc}', exc_info=True)

    logger.warning("Defaulting to 'choppy' regime — pipeline will hold pure cash today")
    return 'choppy'


# ---------------------------------------------------------------------------
# STEP 5 — ML SIGNAL GENERATION
# ---------------------------------------------------------------------------
def run_ml_signals(
    features_df:  pd.DataFrame,
    sentiment_df: pd.DataFrame | None,
    logger:       logging.Logger,
    regime,
) -> pd.DataFrame | None:
    """
    Load xgb_model.pkl and generate pred_proba for every symbol for today.

    The model was trained with:
        - 22 base features (from features_daily.parquet)
        - Regime label (categorical: bull/choppy/bear)
        - Rule-based signals (momentum rank, RSI/BB mean-reversion)
        - 6 sentiment features (set to 0 if sentiment_df is None)

    generate_signals() should:
        1. Filter features_df to the most recent available date
        2. Join sentiment_df on (date, symbol)
        3. Construct the exact feature vector the model was trained on
        4. Return DataFrame with columns [symbol, pred_proba]

    Returns DataFrame or None (on failure — will trigger pure cash fallback).
    """
    divider(logger, 'STEP 5 — ML SIGNAL GENERATION')
    model_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')

    try:
        from models.ml_signal import generate_signals   # type: ignore
        signals_df = generate_signals(
            features_df=features_df,
            sentiment_df=sentiment_df,
            model_path=model_path,
            regime=regime,
        )

        # Validate output shape
        required_cols = {'symbol', 'pred_proba'}
        if not required_cols.issubset(signals_df.columns):
            raise ValueError(f'generate_signals must return columns {required_cols}, got {signals_df.columns.tolist()}')

        above_threshold = signals_df[signals_df['pred_proba'] >= 0.35]
        logger.info(
            f'Signals generated — {len(signals_df)} symbols scored, '
            f'{len(above_threshold)} above 0.35 threshold'
        )

        if len(above_threshold) > 0:
            top5 = above_threshold.nlargest(min(5, len(above_threshold)), 'pred_proba')
            for _, row in top5.iterrows():
                logger.info(f'  {row["symbol"]:<6} pred_proba={row["pred_proba"]:.4f}')
        else:
            logger.warning('No symbols above 0.35 threshold — pipeline will hold cash today')

        return signals_df

    except ImportError:
        logger.error('models.ml_signal.generate_signals not found — add this function to models/ml_signal.py')
    except Exception as exc:
        logger.error(f'ML signal generation error: {exc}', exc_info=True)

    logger.warning('ML signals unavailable — defaulting to pure cash')
    return None


# ---------------------------------------------------------------------------
# STEP 6 — RISK LAYER
# ---------------------------------------------------------------------------
def run_risk_layer(
    signals_df:  pd.DataFrame,
    features_df: pd.DataFrame,
    regime:      str,
    logger:      logging.Logger,
) -> dict[str, float]:
    """
    Translate pred_proba scores into final portfolio weights.

    Flow:
        Kelly Criterion (half-Kelly + vol targeting)
            → compute_kelly_weights() → {symbol: kelly_weight}
        Markowitz mean-variance optimization
            → optimize_weights()      → {symbol: final_weight}

    Regime-specific behavior:
        'choppy' → This function should never be called in choppy.
                   Handled upstream in main pipeline.
        'bull'   → Full Kelly + Markowitz, 15% max cap
        'bear'   → Kelly + Markowitz with 70% total cap (model less reliable)

    On any failure: returns {} (pure cash). Never raises.
    """
    divider(logger, 'STEP 6 — RISK LAYER')

    try:
        from risk.position_sizer    import compute_kelly_weights   # type: ignore
        from risk.portfolio_optimiser import optimize_weights      # type: ignore
    except ImportError as exc:
        logger.error(f'Risk layer import error: {exc}')
        logger.warning('Risk layer unavailable — holding pure cash')
        return {}

    # ── 6a: Kelly Criterion ─────────────────────────────────────────────────
    try:
        kelly_weights: dict[str, float] = compute_kelly_weights(signals_df, features_df)
    except Exception as exc:
        logger.error(f'Kelly sizing error: {exc}', exc_info=True)
        logger.warning('Kelly sizing failed — holding pure cash')
        return {}

    n_positions  = len(kelly_weights)
    total_invested = sum(kelly_weights.values())
    logger.info(
        f'Kelly — {n_positions} positions selected, '
        f'{total_invested*100:.1f}% invested, '
        f'{(1-total_invested)*100:.1f}% cash'
    )

    if n_positions == 0:
        logger.info('No positions with sufficient edge — holding pure cash')
        return {}

    if n_positions < MIN_POSITIONS_FOR_MARKOWITZ:
        logger.info(
            f'Only {n_positions} position(s) — skipping Markowitz '
            f'(need ≥{MIN_POSITIONS_FOR_MARKOWITZ} to compute covariance)'
        )
        return kelly_weights

    # ── 6b: Markowitz Portfolio Optimization ────────────────────────────────
    try:
        final_weights: dict[str, float] = optimize_weights(
            kelly_weights=kelly_weights,
            features_df=features_df,
            regime=regime,
        )
    except Exception as exc:
        logger.error(f'Markowitz optimizer error: {exc}', exc_info=True)
        logger.warning('Falling back to Kelly weights (no Markowitz)')
        return kelly_weights

    # Log average absolute weight change (Kelly → Markowitz)
    all_symbols = set(list(kelly_weights.keys()) + list(final_weights.keys()))
    avg_shift = float(np.mean([
        abs(final_weights.get(s, 0.0) - kelly_weights.get(s, 0.0))
        for s in all_symbols
    ]))
    logger.info(
        f'Markowitz — {len(final_weights)} final positions, '
        f'avg weight shift: {avg_shift*100:.2f}%'
    )

    # Log top positions
    sorted_pos = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
    logger.info('Target weights (top positions):')
    for sym, w in sorted_pos[:10]:
        kelly_w = kelly_weights.get(sym, 0.0)
        arrow   = '↑' if w > kelly_w else ('↓' if w < kelly_w else '=')
        logger.info(f'  {sym:<6} {w*100:5.1f}%  [{arrow} from Kelly {kelly_w*100:.1f}%]')

    return final_weights


# ---------------------------------------------------------------------------
# STEP 7 — ORDER EXECUTION
# ---------------------------------------------------------------------------
def run_execution(
    target_weights: dict[str, float],
    dry_run:        bool,
    logger:         logging.Logger,
) -> bool:
    """
    Instantiate OrderManager and rebalance the portfolio to target_weights.

    If target_weights is empty (choppy regime or upstream failure), the order
    manager will liquidate all positions (go to full cash).

    Returns True on success, False on any order placement failure.
    """
    divider(logger, 'STEP 7 — ORDER EXECUTION')

    try:
        from execution.order_manager import OrderManager   # type: ignore
        manager = OrderManager(dry_run=dry_run)
        success = manager.rebalance(target_weights)
        if success:
            logger.info('Order execution complete')
        else:
            logger.warning('OrderManager.rebalance() returned False — check order_manager logs')
        return success
    except ImportError as exc:
        logger.error(f'Could not import OrderManager: {exc}')
        return False
    except Exception as exc:
        logger.error(f'Order execution error: {exc}', exc_info=True)
        return False


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(dry_run: bool = True) -> None:
    """
    Orchestrate the full AlphaForge morning pipeline.

    Stage failures are handled gracefully:
    - Ingest failure     → continue with cached data
    - Feature failure    → FATAL (abort — no data to trade on)
    - Sentiment failure  → continue with cached or None
    - Regime failure     → default to 'choppy' (pure cash)
    - ML signal failure  → default to pure cash
    - Risk layer failure → default to pure cash
    - Execution failure  → logged, script exits normally
    """
    logger     = setup_logging(dry_run)
    t_start    = datetime.now()
    now_et     = datetime.now(ET_TZ)

    logger.info('=' * 60)
    logger.info('  AlphaForge — Daily Pipeline')
    logger.info('=' * 60)
    logger.info(f'  Time (ET) : {now_et.strftime("%Y-%m-%d %H:%M:%S %Z")}')
    logger.info(f'  Mode      : {"DRY RUN (no real orders)" if dry_run else "LIVE PAPER TRADING"}')
    logger.info(f'  Universe  : {len(SYMBOLS)} symbols')
    logger.info('=' * 60)

    # ── 1: Ingest ────────────────────────────────────────────────────────────
    run_ingest(logger)  # non-fatal

    # ── 2: Features ──────────────────────────────────────────────────────────
    features_df = run_features(logger)
    if features_df is None:
        logger.error('FATAL: Features unavailable — cannot generate signals. Aborting.')
        return

    # ── 3: Sentiment ─────────────────────────────────────────────────────────
    sentiment_df = run_sentiment(logger)  # non-fatal; None is handled downstream

    # ── 4: Regime ────────────────────────────────────────────────────────────
    regime = run_regime(features_df, logger)

    # ── 5 + 6: ML Signals + Risk Layer ───────────────────────────────────────
    # Choppy = pure cash. Do NOT call ML model or optimizer.
    # Rationale: SPY rotation in choppy produced -40.89% return due to 2022
    # rate hike selloff. Pure cash is the empirically proven correct rule.
    if regime == 'choppy':
        divider(logger, 'STEP 5 — ML SIGNAL GENERATION (skipped — choppy regime)')
        logger.info('Choppy regime → holding pure cash. Skipping ML + Risk Layer.')
        divider(logger, 'STEP 6 — RISK LAYER (skipped — choppy regime)')
        target_weights: dict[str, float] = {}

    else:
        signals_df = run_ml_signals(features_df, sentiment_df, logger,regime=regime)

        if signals_df is None or signals_df[signals_df['pred_proba'] >= 0.35].empty:
            # No actionable signals today — go to cash
            divider(logger, 'STEP 6 — RISK LAYER (skipped — no signals above threshold)')
            logger.info('No signals above 0.35 threshold — holding pure cash today')
            target_weights = {}
        else:
            target_weights = run_risk_layer(signals_df, features_df, regime, logger)

    # ── 7: Execute ───────────────────────────────────────────────────────────
    execution_ok = run_execution(target_weights, dry_run, logger)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - t_start).total_seconds()
    logger.info('=' * 60)
    logger.info('  Pipeline Complete')
    logger.info('=' * 60)
    logger.info(f'  Regime      : {regime.upper()}')
    logger.info(f'  Positions   : {len(target_weights)}')
    logger.info(f'  Invested    : {sum(target_weights.values())*100:.1f}%')
    logger.info(f'  Execution   : {"OK" if execution_ok else "FAILED"}')
    logger.info(f'  Elapsed     : {elapsed:.1f}s')
    logger.info(f'  Mode        : {"DRY RUN" if dry_run else "LIVE PAPER TRADING"}')
    logger.info('=' * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AlphaForge daily pipeline orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python execution/run_daily.py           # dry run — safe to test anytime
  python execution/run_daily.py --live    # live paper trading (weekday market hours only)
        """,
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live paper trading orders (default: dry run)',
    )
    args = parser.parse_args()

    run_pipeline(dry_run=not args.live)