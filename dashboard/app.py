"""
AlphaForge — Live Trading Dashboard
====================================
Streamlit dashboard for monitoring the AlphaForge quantitative trading platform.

Run:  streamlit run dashboard/app.py
From: AlphaForge/ project root

Pages:
  1. Overview      — account value, positions, regime, P&L
  2. Signals       — ML predictions, sentiment, regime history
  3. Backtests     — historical strategy performance curves & comparison
  4. Risk          — sector exposure, concentration, drawdown, weight shifts
  5. Pipeline      — execution logs, health checks, stage timing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import glob
import json
import re

# ── PROJECT ROOT ──────────────────────────────────────────────────────────────
# Dashboard lives in AlphaForge/dashboard/ — go one level up for project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── ALPACA CLIENT ─────────────────────────────────────────────────────────────

def get_alpaca_client():
    """Lazy-load Alpaca trading client from .env credentials."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        from alpaca.trading.client import TradingClient
        api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret  = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not api_key or not secret:
            return None
        return TradingClient(api_key, secret, paper=True)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADERS — cached with st.cache_data for performance
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

@st.cache_data(ttl=300)  # refresh every 5 min
def load_account_info():
    """Fetch live account info from Alpaca."""
    client = get_alpaca_client()
    if not client:
        return None
    try:
        account = client.get_account()
        return {
            "equity":          float(account.equity),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
            "portfolio_value": float(account.portfolio_value or account.equity),
            "pnl_today":       float(account.equity) - float(account.last_equity),
            "pnl_today_pct":   (float(account.equity) - float(account.last_equity))
                               / float(account.last_equity) * 100
                               if float(account.last_equity) > 0 else 0,
            "last_equity":     float(account.last_equity),
        }
    except Exception as e:
        st.warning(f"Alpaca connection error: {e}")
        return None

@st.cache_data(ttl=300)
def load_positions():
    """Fetch current positions from Alpaca."""
    client = get_alpaca_client()
    if not client:
        return pd.DataFrame()
    try:
        positions = client.get_all_positions()
        if not positions:
            return pd.DataFrame()
        rows = []
        for p in positions:
            rows.append({
                "symbol":       p.symbol,
                "qty":          float(p.qty),
                "market_value": float(p.market_value),
                "cost_basis":   float(p.cost_basis),
                "unrealised_pl": float(p.unrealized_pl),
                "unrealised_pct": float(p.unrealized_plpc) * 100,
                "current_price": float(p.current_price),
                "avg_entry":    float(p.avg_entry_price),
                "side":         p.side,
            })
        df = pd.DataFrame(rows)
        total_mv = df["market_value"].sum()
        df["weight_pct"] = (df["market_value"] / total_mv * 100).round(2) if total_mv > 0 else 0
        return df.sort_values("market_value", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not load positions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_order_history(days=7):
    """Fetch recent order history from Alpaca."""
    client = get_alpaca_client()
    if not client:
        return pd.DataFrame()
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=datetime.now() - timedelta(days=days),
            limit=100,
        )
        orders = client.get_orders(req)
        if not orders:
            return pd.DataFrame()
        rows = []
        for o in orders:
            rows.append({
                "submitted_at": o.submitted_at,
                "symbol":       o.symbol,
                "side":         o.side.value if hasattr(o.side, 'value') else str(o.side),
                "notional":     float(o.notional) if o.notional else None,
                "qty":          float(o.qty) if o.qty else None,
                "filled_qty":   float(o.filled_qty) if o.filled_qty else None,
                "status":       o.status.value if hasattr(o.status, 'value') else str(o.status),
                "filled_avg":   float(o.filled_avg_price) if o.filled_avg_price else None,
                "order_type":   o.type.value if hasattr(o.type, 'value') else str(o.type),
            })
        return pd.DataFrame(rows).sort_values("submitted_at", ascending=False).reset_index(drop=True)
    except Exception as e:
        st.warning(f"Could not load orders: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_parquet(filename):
    """Load a parquet file from data/processed/. Returns empty DataFrame on error."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def _safe_normalize_dates(series):
    """Normalize a date Series: strip timezone then normalize to midnight."""
    s = pd.to_datetime(series)
    if hasattr(s.dt, 'tz') and s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()

def _safe_normalize_index(index):
    """Normalize a DatetimeIndex: strip timezone then normalize to midnight."""
    idx = pd.to_datetime(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()

@st.cache_data(ttl=600)
def load_regime_labels():
    df = load_parquet("regime_labels.parquet")
    if df.empty:
        return df
    df = df.copy()
    if "regime" in df.columns and df.index.name != "date":
        df.index = _safe_normalize_index(df.index)
    df = df.reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    if "date" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "date"})
    if "date" in df.columns:
        df["date"] = _safe_normalize_dates(df["date"])
    return df

@st.cache_data(ttl=600)
def load_ml_signals():
    df = load_parquet("ml_signals.parquet")
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = _safe_normalize_dates(df["date"])
    return df

@st.cache_data(ttl=600)
def load_sentiment():
    df = load_parquet("sentiment_daily.parquet")
    if df.empty:
        return df
    df = df.copy()
    # Handle MultiIndex (time, symbol) like features_daily.parquet
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name in ("time", "date"):
        df = df.reset_index()
    # Normalize column names
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    if "date" in df.columns:
        df["date"] = _safe_normalize_dates(df["date"])
    return df

@st.cache_data(ttl=600)
def load_features():
    df = load_parquet("features_daily.parquet")
    if df.empty:
        return df
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "time" in df.columns:
        df["date"] = _safe_normalize_dates(df["time"])
    return df

def load_latest_log():
    """Load the most recent trading log file."""
    if not os.path.exists(LOGS_DIR):
        return None, None
    log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "trading_*.log")))
    if not log_files:
        return None, None
    latest = log_files[-1]
    try:
        with open(latest, "r") as f:
            return os.path.basename(latest), f.read()
    except Exception:
        return None, None

def parse_log_stages(log_text):
    """Extract stage timings and status from the LAST pipeline run in a log file."""
    if not log_text:
        return []

    # Find the start of the LAST pipeline run
    run_markers = [m.start() for m in re.finditer(r"AlphaForge — Daily Pipeline", log_text)]
    if run_markers:
        log_text = log_text[run_markers[-1]:]  # only parse from last run

    stages = []
    stage_pattern = re.compile(r"STEP (\d+) — (\w+)")
    time_pattern  = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    lines = log_text.strip().split("\n")
    current_stage = None
    current_start = None

    for line in lines:
        stage_match = stage_pattern.search(line)
        time_match  = time_pattern.match(line)
        ts = None
        if time_match:
            try:
                ts = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        if stage_match:
            if current_stage and ts and current_start:
                stages.append({
                    "stage":    current_stage,
                    "start":    current_start,
                    "end":      ts,
                    "duration": (ts - current_start).total_seconds(),
                    "status":   "OK",
                })
            current_stage = f"Step {stage_match.group(1)} — {stage_match.group(2)}"
            current_start = ts

    # Close final stage with last timestamp
    if current_stage and current_start:
        last_ts = None
        for line in reversed(lines):
            tm = time_pattern.match(line)
            if tm:
                try:
                    last_ts = datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
                break
        if last_ts:
            stages.append({
                "stage":    current_stage,
                "start":    current_start,
                "end":      last_ts,
                "duration": (last_ts - current_start).total_seconds(),
                "status":   "OK",
            })
    return stages


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR MAPPING — for portfolio exposure analysis
# ══════════════════════════════════════════════════════════════════════════════

SECTOR_MAP = {
    # Tech
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "NVDA": "Tech",
    "META": "Tech", "AMZN": "Tech", "TSLA": "Tech", "AVGO": "Tech",
    "AMD": "Tech", "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech",
    "NFLX": "Tech", "QCOM": "Tech", "TXN": "Tech", "CSCO": "Tech",
    "IBM": "Tech", "NOW": "Tech", "AMAT": "Tech", "MU": "Tech", "INTC": "Tech",
    # Finance
    "JPM": "Finance", "GS": "Finance", "BAC": "Finance", "MS": "Finance",
    "BLK": "Finance", "V": "Finance", "MA": "Finance", "C": "Finance",
    "WFC": "Finance", "AXP": "Finance", "SCHW": "Finance", "COF": "Finance", "BK": "Finance",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "LLY": "Healthcare", "MRK": "Healthcare", "AMGN": "Healthcare", "BMY": "Healthcare",
    "GILD": "Healthcare", "TMO": "Healthcare", "MDT": "Healthcare", "ISRG": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    # Consumer
    "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer", "WMT": "Consumer",
    "COST": "Consumer", "HD": "Consumer", "TGT": "Consumer", "LOW": "Consumer",
    "BKNG": "Consumer", "TJX": "Consumer", "MAR": "Consumer",
    # Industrial
    "CAT": "Industrial", "BA": "Industrial", "HON": "Industrial", "GE": "Industrial",
    "RTX": "Industrial", "LMT": "Industrial", "UPS": "Industrial", "DE": "Industrial",
    "ETN": "Industrial",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities",
    # REITs
    "AMT": "REITs", "PLD": "REITs",
    # Materials
    "LIN": "Materials", "NEM": "Materials",
    # Communications
    "DIS": "Communications", "VZ": "Communications",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF",
}

SECTOR_COLORS = {
    "Tech":           "#4A90D9",
    "Finance":        "#50C878",
    "Healthcare":     "#E06C75",
    "Energy":         "#D19A66",
    "Consumer":       "#C678DD",
    "Industrial":     "#56B6C2",
    "Utilities":      "#E5C07B",
    "REITs":          "#BE5046",
    "Materials":      "#98C379",
    "Communications": "#61AFEF",
    "ETF":            "#ABB2BF",
}

REGIME_COLORS = {"bull": "#50C878", "bear": "#E06C75", "choppy": "#E5C07B"}
REGIME_ICONS  = {"bull": "🟢", "bear": "🔴", "choppy": "🟡"}


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE LAYOUT CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AlphaForge Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }

    /* Dataframe styling */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #21262D; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 6px;
        padding: 8px 16px;
        border: 1px solid #21262D;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F6FEB !important;
        border-color: #1F6FEB !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Regime badge */
    .regime-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    .regime-bull   { background: #16341F; color: #50C878; border: 1px solid #50C878; }
    .regime-bear   { background: #3B1A1E; color: #E06C75; border: 1px solid #E06C75; }
    .regime-choppy { background: #3B3416; color: #E5C07B; border: 1px solid #E5C07B; }
</style>
""", unsafe_allow_html=True)


# ── PLOTLY DEFAULTS ───────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="Consolas, monospace", color="#C9D1D9"),
    margin=dict(l=40, r=20, t=40, b=30),
    xaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
    yaxis=dict(gridcolor="#21262D", zerolinecolor="#21262D"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ AlphaForge")
    st.caption("Quantitative Trading Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🎯 Signals", "📈 Backtests", "🛡️ Risk", "🔧 Pipeline"],
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)
    if auto_refresh:
        import time as _time
        st.caption("⏱️ Auto-refresh active")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.markdown("# Portfolio Overview")

    # ── Account metrics row ───────────────────────────────────────────────
    account = load_account_info()
    positions_df = load_positions()
    regimes = load_regime_labels()

    # Current regime
    current_regime = "unknown"
    if not regimes.empty and "date" in regimes.columns:
        latest = regimes.sort_values("date").iloc[-1]
        current_regime = latest.get("regime", "unknown")

    if account:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Portfolio Value", f"${account['equity']:,.2f}")
        c2.metric("Daily P&L",
                  f"${account['pnl_today']:+,.2f}",
                  delta=f"{account['pnl_today_pct']:+.2f}%")
        c3.metric("Cash", f"${account['cash']:,.2f}")
        c4.metric("Positions", len(positions_df) if not positions_df.empty else 0)

        regime_class = f"regime-{current_regime}" if current_regime in REGIME_COLORS else "regime-bull"
        c5.markdown(
            f'<div style="text-align:center;margin-top:8px;">'
            f'<span class="regime-badge {regime_class}">'
            f'{REGIME_ICONS.get(current_regime, "⚪")} {current_regime.upper()}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("📡 Connect Alpaca API keys in `.env` to see live account data.")

    st.divider()

    # ── Positions table + sector exposure side by side ────────────────────
    col_pos, col_sector = st.columns([3, 2])

    with col_pos:
        st.markdown("### Current Positions")
        if not positions_df.empty:
            display_df = positions_df[[
                "symbol", "weight_pct", "market_value", "unrealised_pl",
                "unrealised_pct", "current_price", "avg_entry"
            ]].copy()
            display_df.columns = [
                "Symbol", "Weight %", "Market Value", "Unrealised P&L",
                "Unrealised %", "Price", "Avg Entry"
            ]
            st.dataframe(
                display_df.style.format({
                    "Weight %":      "{:.1f}%",
                    "Market Value":  "${:,.2f}",
                    "Unrealised P&L": "${:+,.2f}",
                    "Unrealised %":  "{:+.2f}%",
                    "Price":         "${:.2f}",
                    "Avg Entry":     "${:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open positions.")

    with col_sector:
        st.markdown("### Sector Exposure")
        if not positions_df.empty:
            positions_df["sector"] = positions_df["symbol"].map(SECTOR_MAP).fillna("Other")
            sector_weights = positions_df.groupby("sector")["weight_pct"].sum().sort_values(ascending=False)

            fig = go.Figure(go.Pie(
                labels=sector_weights.index,
                values=sector_weights.values,
                hole=0.55,
                marker=dict(colors=[SECTOR_COLORS.get(s, "#ABB2BF") for s in sector_weights.index]),
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions to display.")

    # ── Regime history chart ──────────────────────────────────────────────
    st.markdown("### Regime History (Last 90 Days)")
    if not regimes.empty and "date" in regimes.columns:
        recent = regimes.sort_values("date").tail(90)
        fig = go.Figure()
        for regime in ["bull", "bear", "choppy"]:
            mask = recent["regime"] == regime
            subset = recent[mask]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset["date"], y=[regime.upper()] * len(subset),
                    mode="markers",
                    marker=dict(color=REGIME_COLORS.get(regime), size=10, symbol="square"),
                    name=regime.capitalize(),
                ))
        fig.update_layout(**PLOTLY_LAYOUT, height=180,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=["BULL", "CHOPPY", "BEAR"],
                       gridcolor="#21262D"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent orders ─────────────────────────────────────────────────────
    st.markdown("### Recent Orders")
    orders_df = load_order_history(days=7)
    if not orders_df.empty:
        display_orders = orders_df[[
            "submitted_at", "symbol", "side", "notional", "status", "filled_avg"
        ]].head(20).copy()
        display_orders.columns = ["Time", "Symbol", "Side", "Notional $", "Status", "Fill Price"]
        st.dataframe(display_orders, use_container_width=True, hide_index=True)
    else:
        st.info("No recent orders.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — SIGNALS & SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Signals":
    st.markdown("# Signals & Sentiment")

    signals = load_ml_signals()
    sentiment = load_sentiment()

    # ── Latest ML signals table ───────────────────────────────────────────
    st.markdown("### ML Signal Scores (Latest Day)")

    if not signals.empty and "date" in signals.columns:
        latest_date = signals["date"].max()
        latest_signals = signals[signals["date"] == latest_date].copy()

        if "pred_proba" in latest_signals.columns:
            latest_signals = latest_signals.sort_values("pred_proba", ascending=False)
            latest_signals["rank"] = range(1, len(latest_signals) + 1)
            latest_signals["above_threshold"] = latest_signals["pred_proba"] >= 0.35

            st.caption(f"Signal date: **{latest_date.strftime('%Y-%m-%d')}** · "
                       f"{latest_signals['above_threshold'].sum()} / {len(latest_signals)} above 0.35 threshold")

            col_top, col_bottom = st.columns(2)

            with col_top:
                st.markdown("**🟢 Top Signals**")
                top = latest_signals.head(15)[["rank", "symbol", "pred_proba"]].copy()
                top.columns = ["#", "Symbol", "Pred Proba"]
                st.dataframe(
                    top.style.format({"Pred Proba": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )

            with col_bottom:
                st.markdown("**🔴 Bottom Signals**")
                bottom = latest_signals.tail(15)[["rank", "symbol", "pred_proba"]].copy()
                bottom.columns = ["#", "Symbol", "Pred Proba"]
                st.dataframe(
                    bottom.style.format({"Pred Proba": "{:.4f}"}),
                    use_container_width=True, hide_index=True,
                )

            # Signal distribution histogram
            st.markdown("### Signal Distribution")
            fig = go.Figure(go.Histogram(
                x=latest_signals["pred_proba"],
                nbinsx=30,
                marker_color="#4A90D9",
                marker_line=dict(color="#1F6FEB", width=1),
            ))
            fig.add_vline(x=0.35, line_dash="dash", line_color="#E5C07B",
                          annotation_text="Threshold (0.35)")
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="Predicted Probability",
                              yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No ML signals found. Run the pipeline to generate signals.")

    st.divider()

    # ── Sentiment overview ────────────────────────────────────────────────
    st.markdown("### Sentiment Overview (Latest Day)")

    if not sentiment.empty and "date" in sentiment.columns:
        latest_sent_date = sentiment["date"].max()
        latest_sent = sentiment[sentiment["date"] == latest_sent_date].copy()

        if "sentiment_mean" in latest_sent.columns and "symbol" in latest_sent.columns:
            latest_sent = latest_sent.sort_values("sentiment_mean", ascending=False)

            st.caption(f"Sentiment date: **{latest_sent_date.strftime('%Y-%m-%d')}** · "
                       f"{len(latest_sent)} symbols with data")

            # Sentiment bar chart — top and bottom
            top_pos = latest_sent.head(10)
            top_neg = latest_sent.tail(10)
            chart_data = pd.concat([top_pos, top_neg]).drop_duplicates(subset="symbol")
            chart_data = chart_data.sort_values("sentiment_mean")

            fig = go.Figure(go.Bar(
                x=chart_data["sentiment_mean"],
                y=chart_data["symbol"],
                orientation="h",
                marker_color=[
                    "#50C878" if v >= 0 else "#E06C75"
                    for v in chart_data["sentiment_mean"]
                ],
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=400,
                              xaxis_title="Sentiment Score",
                              yaxis_title="",
                              title="Most Positive & Negative Sentiment")
            st.plotly_chart(fig, use_container_width=True)

            # Article count by symbol
            if "article_count" in latest_sent.columns:
                st.markdown("### News Volume (Article Count)")
                vol = latest_sent.nlargest(20, "article_count")[["symbol", "article_count"]]
                fig = go.Figure(go.Bar(
                    x=vol["symbol"], y=vol["article_count"],
                    marker_color="#61AFEF",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=300,
                                  xaxis_title="Symbol", yaxis_title="Articles")
                st.plotly_chart(fig, use_container_width=True)
    else:
        if sentiment.empty:
            st.info("No sentiment data found. Check that `sentiment_daily.parquet` exists.")
        else:
            st.warning(f"Sentiment data loaded ({len(sentiment)} rows) but missing expected columns. "
                       f"Columns found: `{list(sentiment.columns)}`")

    # ── Signal Quality Analytics ───────────────────────────────────────────
    st.divider()
    st.markdown("### Signal Quality Monitoring")

    _ml_bt = load_parquet("backtest_ml.parquet")
    if (not _ml_bt.empty and not signals.empty
            and "pred_proba" in signals.columns and "date" in signals.columns):
        from analytics.signal_quality import (
            bucket_accuracy, signal_distribution_shift, _manual_auc,
        )

        # Bucket calibration: split all signals into 5 buckets
        # Need to match predictions with outcomes — use features for forward returns
        _feats = load_features()
        _has_calibration = False

        if (not _feats.empty and "return_1d" in _feats.columns
                and "symbol" in _feats.columns and "date" in _feats.columns
                and "symbol" in signals.columns):
            # Build 5-day forward returns per (date, symbol) from features
            _ret_pivot = _feats.pivot_table(index="date", columns="symbol", values="return_1d")
            _fwd5 = _ret_pivot.rolling(5).sum().shift(-5)  # 5-day forward return
            _fwd5_stacked = _fwd5.stack().reset_index()
            _fwd5_stacked.columns = ["date", "symbol", "fwd_return_5d"]

            _merged = signals.merge(_fwd5_stacked, on=["date", "symbol"], how="inner")
            _merged = _merged.dropna(subset=["pred_proba", "fwd_return_5d"])

            if len(_merged) > 100:
                # Binary label: top quartile of 5-day returns = 1
                _threshold = _merged["fwd_return_5d"].quantile(0.75)
                _y_true = (_merged["fwd_return_5d"] >= _threshold).astype(float).values
                _y_score = _merged["pred_proba"].values

                # Overall AUC
                if len(np.unique(_y_true)) >= 2:
                    _auc = _manual_auc(_y_true, _y_score)
                    st.metric("Overall AUC (non-overlapping)", f"{_auc:.4f}",
                              help="AUC-ROC: 0.5 = random, 1.0 = perfect. "
                                   "Measures how well the model ranks winners above losers.")

                # Bucket calibration chart
                buckets = bucket_accuracy(_y_true, _y_score, n_buckets=5)
                if buckets:
                    _has_calibration = True
                    st.markdown("#### Prediction Calibration (5 Buckets)")
                    st.caption("Do stocks predicted at high probability actually win more often?")

                    _bk_df = pd.DataFrame(buckets)
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(
                        x=_bk_df["bucket"], y=_bk_df["actual_win_rate"] * 100,
                        name="Actual Win Rate %",
                        marker_color="#50C878",
                    ), secondary_y=False)
                    fig.add_trace(go.Scatter(
                        x=_bk_df["bucket"], y=_bk_df["avg_pred"] * 100,
                        name="Avg Prediction %",
                        mode="lines+markers",
                        line=dict(color="#E5C07B", width=2),
                    ), secondary_y=True)
                    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    fig.update_yaxes(title_text="Actual Win Rate %", secondary_y=False)
                    fig.update_yaxes(title_text="Avg Prediction %", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)

                # Recalibration check (use recent vs older signal distributions)
                _dates_sorted = sorted(_merged["date"].unique())
                _mid = len(_dates_sorted) // 2
                if _mid > 50:
                    _old_scores = _merged[_merged["date"].isin(_dates_sorted[:_mid])]["pred_proba"].values
                    _new_scores = _merged[_merged["date"].isin(_dates_sorted[_mid:])]["pred_proba"].values
                    _kl = signal_distribution_shift(_old_scores, _new_scores)

                    st.markdown("#### Signal Distribution Shift")
                    c_kl1, c_kl2 = st.columns(2)
                    c_kl1.metric("KL Divergence", f"{_kl:.4f}",
                                 help="0 = identical distributions. "
                                      ">0.1 = meaningful shift in model output.")
                    _status = "Low drift" if _kl < 0.05 else "Moderate drift" if _kl < 0.15 else "High drift"
                    _color = "#50C878" if _kl < 0.05 else "#E5C07B" if _kl < 0.15 else "#E06C75"
                    c_kl2.markdown(
                        f'<div style="margin-top:8px;padding:8px 16px;border-radius:8px;'
                        f'border:1px solid {_color};color:{_color};display:inline-block;">'
                        f'{_status}</div>',
                        unsafe_allow_html=True,
                    )

        if not _has_calibration:
            st.info("Need ML signals + features with forward returns for calibration analysis.")
    else:
        st.info("Signal quality analytics require ML signals and backtest data.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — BACKTESTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Backtests":
    st.markdown("# Backtest Results")

    # Load all backtest parquets
    bt_files = {
        "Momentum":          "backtest_momentum.parquet",
        "Mean Reversion":    "backtest_mean_reversion.parquet",
        "Regime Switcher":   "backtest_regime_switcher.parquet",
        "ML + Risk Layer":   "backtest_ml.parquet",
    }

    portfolios = {}
    for name, fname in bt_files.items():
        df = load_parquet(fname)
        if not df.empty:
            df = df.copy()
            df.index = _safe_normalize_index(df.index)
            portfolios[name] = df

    if not portfolios:
        st.info("No backtest results found. Run the backtest scripts first.")
    else:
        # ── Equity curves ─────────────────────────────────────────────────
        st.markdown("### Equity Curves")
        strategy_colors = {
            "Momentum":        "#61AFEF",
            "Mean Reversion":  "#C678DD",
            "Regime Switcher": "#E5C07B",
            "ML + Risk Layer": "#50C878",
        }

        fig = go.Figure()
        for name, df in portfolios.items():
            fig.add_trace(go.Scatter(
                x=df.index, y=df["portfolio_value"],
                mode="lines",
                name=name,
                line=dict(color=strategy_colors.get(name, "#ABB2BF"), width=2),
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=450,
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Performance comparison table ──────────────────────────────────
        st.markdown("### Strategy Comparison")
        rows = []
        for name, df in portfolios.items():
            ret    = df["daily_return"]
            start  = df["portfolio_value"].iloc[0]
            end    = df["portfolio_value"].iloc[-1]
            total  = (end / start) - 1
            years  = len(ret) / 252
            ann    = (1 + total) ** (1 / years) - 1 if years > 0 else 0
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
            cum    = (1 + ret).cumprod()
            dd     = ((cum - cum.cummax()) / cum.cummax()).min()
            wr     = (ret > 0).mean()
            sortino_denom = ret[ret < 0].std()
            sortino = (ret.mean() / sortino_denom) * np.sqrt(252) if sortino_denom > 0 else 0

            rows.append({
                "Strategy":     name,
                "Total Return": total,
                "Ann. Return":  ann,
                "Sharpe":       sharpe,
                "Sortino":      sortino,
                "Max Drawdown": dd,
                "Win Rate":     wr,
                "Days":         len(ret),
                "Period":       f"{df.index.min().strftime('%Y-%m')} → {df.index.max().strftime('%Y-%m')}",
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(
            comp_df.style.format({
                "Total Return": "{:.2%}",
                "Ann. Return":  "{:.2%}",
                "Sharpe":       "{:.2f}",
                "Sortino":      "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Win Rate":     "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── Drawdown chart ────────────────────────────────────────────────
        st.markdown("### Drawdown Comparison")
        fig = go.Figure()
        for name, df in portfolios.items():
            ret = df["daily_return"]
            cum = (1 + ret).cumprod()
            dd  = (cum - cum.cummax()) / cum.cummax()
            fig.add_trace(go.Scatter(
                x=df.index, y=dd * 100,
                mode="lines", name=name, fill="tozeroy",
                line=dict(color=strategy_colors.get(name, "#ABB2BF"), width=1),
                fillcolor=strategy_colors.get(name, "#ABB2BF").replace(")", ",0.15)").replace("rgb", "rgba")
                          if "rgb" in strategy_colors.get(name, "") else None,
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=350,
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-regime breakdown (ML + Risk Layer) ────────────────────────
        if "ML + Risk Layer" in portfolios:
            ml_df = portfolios["ML + Risk Layer"]
            if "regime" in ml_df.columns:
                st.markdown("### ML + Risk Layer — Performance by Regime")
                regime_rows = []
                for regime in ["bull", "bear", "choppy"]:
                    subset = ml_df[ml_df["regime"] == regime]
                    if len(subset) < 5:
                        continue
                    ret = subset["daily_return"]
                    total = (ret + 1).prod() - 1
                    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
                    regime_rows.append({
                        "Regime":       f"{REGIME_ICONS.get(regime, '')} {regime.upper()}",
                        "Days":         len(subset),
                        "Total Return": total,
                        "Sharpe":       sharpe,
                    })
                if regime_rows:
                    st.dataframe(
                        pd.DataFrame(regime_rows).style.format({
                            "Total Return": "{:.2%}", "Sharpe": "{:.2f}"
                        }),
                        use_container_width=True, hide_index=True,
                    )

        # ── Regime Transition Analysis ─────────────────────────────────────
        regimes = load_regime_labels()
        if not regimes.empty and "regime" in regimes.columns:
            from analytics.regime_analysis import (
                transition_matrix, regime_persistence, regime_accuracy,
            )

            regime_seq = regimes.sort_values("date")["regime"].values

            st.divider()
            st.markdown("### Regime Transition Analysis")

            # Transition matrix heatmap
            _labels = ["bull", "choppy", "bear"]
            tm = transition_matrix(regime_seq, labels=_labels)
            if tm["matrix"]:
                col_tm, col_persist = st.columns([3, 2])

                with col_tm:
                    st.markdown("#### Transition Probabilities")
                    st.caption("P(tomorrow's regime | today's regime)")
                    _matrix = np.array(tm["matrix"])
                    fig = go.Figure(go.Heatmap(
                        z=_matrix,
                        x=[l.upper() for l in _labels],
                        y=[l.upper() for l in _labels],
                        colorscale=[[0, "#0E1117"], [1, "#4A90D9"]],
                        zmin=0, zmax=1,
                        text=np.round(_matrix, 3),
                        texttemplate="%{text:.1%}",
                        textfont=dict(size=13),
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                                      xaxis_title="To Regime",
                                      yaxis_title="From Regime")
                    fig.update_layout(margin=dict(l=80, r=20, t=30, b=60))
                    st.plotly_chart(fig, use_container_width=True)

                with col_persist:
                    st.markdown("#### Regime Persistence")
                    st.caption("Average duration per regime (trading days)")
                    persist = regime_persistence(regime_seq)
                    persist_rows = []
                    for regime in _labels:
                        p = persist.get(regime, {})
                        if p:
                            persist_rows.append({
                                "Regime":   f"{REGIME_ICONS.get(regime, '')} {regime.upper()}",
                                "Avg Days": p["avg_duration"],
                                "Max Days": p["max_duration"],
                                "Episodes": p["n_episodes"],
                            })
                    if persist_rows:
                        st.dataframe(
                            pd.DataFrame(persist_rows).style.format({
                                "Avg Days": "{:.1f}",
                            }),
                            use_container_width=True, hide_index=True,
                        )

            # Regime accuracy (using ML backtest returns if available)
            if "ML + Risk Layer" in portfolios:
                _ml = portfolios["ML + Risk Layer"]
                if "regime" in _ml.columns and "daily_return" in _ml.columns:
                    _acc_regimes = _ml["regime"].values
                    _acc_returns = _ml["daily_return"].values
                    acc = regime_accuracy(_acc_regimes, _acc_returns)

                    st.markdown("#### Regime Prediction Accuracy")
                    st.caption("Did bull predictions actually have positive returns?")
                    acc_cols = st.columns(len(acc.get("per_regime", {})) + 1)
                    acc_cols[0].metric("Overall Accuracy",
                                       f"{acc['overall_accuracy']:.1%}")
                    for i, (reg, info) in enumerate(acc.get("per_regime", {}).items()):
                        _reg_name = {"bull": "Bull", "bear": "Bear",
                                     "choppy": "Choppy"}.get(reg, reg.upper())
                        acc_cols[i + 1].metric(
                            f"{_reg_name} Accuracy",
                            f"{info['accuracy']:.1%}",
                            help=f"{info['count']} days in this regime",
                        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RISK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🛡️ Risk":
    st.markdown("# Risk & Analytics")

    positions_df = load_positions()
    features = load_features()

    # ── Risk metrics from backtest returns ─────────────────────────────────
    # Try to compute advanced risk metrics from the best available return series
    _bt_ml = load_parquet("backtest_ml.parquet")
    _has_returns = not _bt_ml.empty and "daily_return" in _bt_ml.columns

    if _has_returns:
        from analytics.risk_metrics import (
            compute_var, compute_cvar, calmar_ratio, portfolio_beta,
            return_skewness, return_kurtosis, max_drawdown,
            herfindahl_index, effective_number_of_bets,
        )
        _ret = _bt_ml["daily_return"].dropna().values

        st.markdown("### Portfolio Risk Metrics")
        st.caption("Computed from ML + Risk Layer backtest returns")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR (95%)", f"{compute_var(_ret, 0.95):.2%}",
                  help="On the worst 5% of days, you lose at least this much")
        r2.metric("CVaR (95%)", f"{compute_cvar(_ret, 0.95):.2%}",
                  help="Average loss on the worst 5% of days")
        r3.metric("VaR (99%)", f"{compute_var(_ret, 0.99):.2%}",
                  help="On the worst 1% of days, you lose at least this much")
        r4.metric("CVaR (99%)", f"{compute_cvar(_ret, 0.99):.2%}",
                  help="Average loss on the worst 1% of days (tail severity)")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Max Drawdown", f"{max_drawdown(_ret):.2%}")
        r6.metric("Calmar Ratio", f"{calmar_ratio(_ret):.2f}",
                  help="Annualised return / max drawdown. Higher = better risk-adjusted")
        r7.metric("Skewness", f"{return_skewness(_ret):.3f}",
                  help="Negative = fatter left tail (more large losses)")
        r8.metric("Excess Kurtosis", f"{return_kurtosis(_ret):.3f}",
                  help=">0 = fat tails (more extreme events than normal)")

        # Beta to SPY if features available
        if not features.empty and "date" in features.columns and "symbol" in features.columns:
            spy_data = features[features["symbol"] == "SPY"].copy()
            if not spy_data.empty and "return_1d" in spy_data.columns:
                spy_returns = spy_data.sort_values("date")["return_1d"].dropna().values
                # Align lengths (take most recent overlap)
                n = min(len(_ret), len(spy_returns))
                if n > 20:
                    beta = portfolio_beta(_ret[-n:], spy_returns[-n:])
                    st.metric("Portfolio Beta (vs SPY)", f"{beta:.3f}",
                              help="1.0 = moves with market. <1 = defensive. >1 = aggressive")

        # Return distribution histogram
        st.markdown("### Return Distribution")
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Histogram(
            x=_ret * 100, nbinsx=60,
            marker_color="#4A90D9",
            marker_line=dict(color="#1F6FEB", width=0.5),
            name="Daily Returns",
        ))
        _var95 = compute_var(_ret, 0.95) * 100
        fig.add_vline(x=_var95, line_dash="dash", line_color="#E06C75",
                      annotation_text=f"VaR 95% ({_var95:.2f}%)")
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          xaxis_title="Daily Return (%)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # ── Position concentration ────────────────────────────────────────────
    st.markdown("### Position Concentration")
    if not positions_df.empty:
        col_conc, col_tree = st.columns([1, 2])

        with col_conc:
            weights_dict = dict(zip(positions_df["symbol"],
                                    positions_df["weight_pct"] / 100))
            try:
                from analytics.risk_metrics import (
                    herfindahl_index as _hhi, effective_number_of_bets as _enb,
                )
                hhi = _hhi(weights_dict)
                n_eff = _enb(weights_dict)
            except ImportError:
                weights = positions_df["weight_pct"].values / 100
                hhi = (weights ** 2).sum()
                n_eff = 1 / hhi if hhi > 0 else 0
            max_pos = positions_df["weight_pct"].max()

            st.metric("HHI (Herfindahl)", f"{hhi:.4f}")
            st.metric("Effective # Positions", f"{n_eff:.1f}")
            st.metric("Largest Position", f"{max_pos:.1f}%")
            st.caption(
                "HHI ranges 0→1. Lower = more diversified. "
                "Equal-weight 10 positions → HHI = 0.10."
            )

        with col_tree:
            positions_df["sector"] = positions_df["symbol"].map(SECTOR_MAP).fillna("Other")
            fig = px.treemap(
                positions_df, path=["sector", "symbol"],
                values="market_value",
                color="unrealised_pct",
                color_continuous_scale=["#E06C75", "#21262D", "#50C878"],
                color_continuous_midpoint=0,
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=400)
            fig.update_layout(margin=dict(l=5, r=5, t=30, b=5))
            fig.update_traces(textinfo="label+percent parent")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No positions to analyse.")

    st.divider()

    # ── Portfolio weight bar chart ────────────────────────────────────────
    st.markdown("### Current Weights")
    if not positions_df.empty:
        fig = go.Figure(go.Bar(
            x=positions_df["symbol"],
            y=positions_df["weight_pct"],
            marker_color=[SECTOR_COLORS.get(SECTOR_MAP.get(s, "Other"), "#ABB2BF")
                          for s in positions_df["symbol"]],
            text=[f"{w:.1f}%" for w in positions_df["weight_pct"]],
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                          yaxis_title="Weight (%)",
                          xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Correlation heatmap of current holdings ───────────────────────────
    st.markdown("### Correlation Matrix (Current Holdings, 60-day)")
    if not positions_df.empty and not features.empty and "date" in features.columns:
        symbols = positions_df["symbol"].tolist()
        recent = features[features["symbol"].isin(symbols)].copy()
        if "return_1d" in recent.columns:
            pivot = recent.pivot_table(index="date", columns="symbol", values="return_1d")
            pivot = pivot.dropna(axis=1, how="all").tail(60)
            if len(pivot.columns) >= 2:
                corr = pivot.corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns, y=corr.index,
                    colorscale=[[0, "#E06C75"], [0.5, "#0E1117"], [1, "#50C878"]],
                    zmid=0, zmin=-1, zmax=1,
                    text=corr.values.round(2),
                    texttemplate="%{text}",
                    textfont=dict(size=10),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=450)
                fig.update_layout(margin=dict(l=60, r=20, t=30, b=60))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 holdings with return data for correlation matrix.")
        else:
            st.info("Return data not available in features.")
    else:
        st.info("Need positions + features data for correlation analysis.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — PIPELINE HEALTH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔧 Pipeline":
    st.markdown("# Pipeline Health")

    log_name, log_text = load_latest_log()

    if log_name:
        st.markdown(f"### Latest Log: `{log_name}`")

        # Parse stages
        stages = parse_log_stages(log_text)
        if stages:
            st.markdown("#### Stage Timing")
            stage_df = pd.DataFrame(stages)
            stage_df["duration_str"] = stage_df["duration"].apply(
                lambda s: f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"
            )

            # Gantt-style bar chart
            fig = go.Figure(go.Bar(
                x=stage_df["duration"],
                y=stage_df["stage"],
                orientation="h",
                marker_color="#4A90D9",
                text=stage_df["duration_str"],
                textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=300,
                              xaxis_title="Duration (seconds)")
            st.plotly_chart(fig, use_container_width=True)

            total_time = stage_df["duration"].sum()
            st.caption(f"Total pipeline time: **{total_time:.1f}s** ({total_time/60:.1f} min)")

        # Key metrics extracted from log
        st.markdown("#### Pipeline Summary")
        summary_patterns = {
            "Universe":   r"Universe\s*:\s*(\d+)\s*symbols",
            "Regime":     r"Regime\s*:\s*(\w+)",
            "Positions":  r"Positions\s*:\s*(\d+)",
            "Invested":   r"Invested\s*:\s*([\d.]+%)",
            "Elapsed":    r"Elapsed\s*:\s*([\d.]+s)",
            "Mode":       r"Mode\s*:\s*(.+)",
        }
        cols = st.columns(len(summary_patterns))
        for i, (label, pattern) in enumerate(summary_patterns.items()):
            matches = re.findall(pattern, log_text)
            value = matches[-1] if matches else "—"
            cols[i].metric(label, value)

        # Raw log (expandable)
        with st.expander("📄 Full Log Output", expanded=False):
            st.code(log_text, language="log")

    else:
        st.info("No log files found in `logs/` directory.")

    st.divider()

    # ── Data freshness check ──────────────────────────────────────────────
    st.markdown("### Data Freshness")
    freshness_files = {
        "Features":     "features_daily.parquet",
        "ML Signals":   "ml_signals.parquet",
        "Regime Labels": "regime_labels.parquet",
        "Sentiment":    "sentiment_daily.parquet",
    }
    fresh_rows = []
    for label, fname in freshness_files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            size_mb = os.path.getsize(path) / (1024 * 1024)
            fresh_rows.append({
                "File":         label,
                "Last Modified": mtime.strftime("%Y-%m-%d %H:%M"),
                "Size":         f"{size_mb:.1f} MB",
                "Age":          f"{(datetime.now() - mtime).total_seconds() / 3600:.1f}h ago",
            })
        else:
            fresh_rows.append({"File": label, "Last Modified": "Not found", "Size": "—", "Age": "—"})

    st.dataframe(pd.DataFrame(fresh_rows), use_container_width=True, hide_index=True)

    # ── All log files ─────────────────────────────────────────────────────
    st.markdown("### Log History")
    if os.path.exists(LOGS_DIR):
        log_files = sorted(glob.glob(os.path.join(LOGS_DIR, "trading_*.log")), reverse=True)
        if log_files:
            log_rows = []
            for lf in log_files[:20]:
                mtime = datetime.fromtimestamp(os.path.getmtime(lf))
                size_kb = os.path.getsize(lf) / 1024
                log_rows.append({
                    "File":     os.path.basename(lf),
                    "Date":     mtime.strftime("%Y-%m-%d %H:%M"),
                    "Size":     f"{size_kb:.1f} KB",
                })
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No trading logs found.")


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-REFRESH (if enabled)
# ══════════════════════════════════════════════════════════════════════════════

if auto_refresh:
    import time as _time
    _time.sleep(300)  # 5 minutes
    st.rerun()