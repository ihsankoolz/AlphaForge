# AlphaForge — Project Reference Guide

## What Is This Project?

AlphaForge is an end-to-end quantitative trading platform. The goal is to build a system that:
1. Ingests and stores real market data
2. Engineers meaningful signals from that data
3. Tests trading strategies against historical data (backtesting)
4. Detects what kind of market environment we're in (regime detection)
5. Uses machine learning to generate smarter trading signals
6. Incorporates news/sentiment as an alternative data source
7. Manages risk and sizes positions properly
8. Executes live paper trades automatically via Alpaca
9. Displays everything on a real-time dashboard

This is not a research notebook. It is a system — each component has a specific job and feeds into the next.

---

## The 10-Week Roadmap

| Week | Focus | Status |
|------|-------|--------|
| 1 | Foundation & Environment Setup | ✅ Done |
| 2 | Data Layer Completion | ✅ Done |
| 3 | Feature Engineering & Traditional Strategies | 🔄 In Progress |
| 4 | Regime Detection with Hidden Markov Models | ⏳ Upcoming |
| 5 | ML Signal Generation (XGBoost) | ⏳ Upcoming |
| 6 | Sentiment Layer as Alternative Data | ⏳ Upcoming |
| 7 | Risk & Portfolio Layer | ⏳ Upcoming |
| 8 | Execution Layer & Paper Trading | ⏳ Upcoming |
| 9 | Dashboard & Visualization | ⏳ Upcoming |
| 10 | Polish, Documentation & Write-Up | ⏳ Upcoming |

---

## System Architecture — How Everything Fits Together

```
[Alpaca API] 
     │
     ▼
[data/ingest.py] ──────────────────► [TimescaleDB]
     │                                     │
     │                                     ▼
     │                          [data/validate.py]
     │                          (checks for gaps, nulls, anomalies)
     │                                     │
     ▼                                     ▼
[research/features/engineer.py] ◄── loads from TimescaleDB
     │
     ▼
[data/processed/features_daily.parquet]
     │
     ├──► [research/strategies/momentum.py]
     ├──► [research/strategies/mean_reversion.py]
     │         │
     │         ▼
     │    Signals (-1 to +1 per stock per day)
     │         │
     ├──► [models/] ← Week 5: ML model trained on features
     ├──► [sentiment pipeline] ← Week 6: FinBERT scores
     │
     ▼
[risk/] ← Week 7: position sizing, portfolio optimization
     │
     ▼
[execution/] ← Week 8: order management, Alpaca paper trading
     │
     ▼
[dashboard/] ← Week 9: Streamlit live dashboard
```

---

## Data Layer

### What We Store

**TimescaleDB** — a PostgreSQL extension optimised for time-series data. Faster than regular PostgreSQL for queries like "give me all prices between these two dates."

Two tables:

**`ohlcv`** — daily price data
| Column | Description |
|--------|-------------|
| time | Date of the trading day (UTC) |
| symbol | Stock ticker e.g. AAPL |
| open | Price at market open (9:30 AM ET) |
| high | Highest price reached during the day |
| low | Lowest price reached during the day |
| close | Price at market close (4:00 PM ET) |
| volume | Total shares traded that day |

**`ohlcv_hourly`** — same structure but one row per hour per stock

### Our Stock Universe (29 stocks)

| Sector | Stocks |
|--------|--------|
| Tech | AAPL, MSFT, GOOGL, NVDA, META, AMZN |
| Finance | JPM, GS, BAC, MS, BLK |
| Healthcare | JNJ, UNH, PFE, ABBV |
| Energy | XOM, CVX, COP |
| Consumer | MCD, NKE, SBUX, WMT, COST |
| Industrial | CAT, BA, HON, GE |
| ETFs | SPY, QQQ |

Date range: **January 2020 to December 2025**

Why this range? It captures very different market environments — COVID crash (2020), bull run (2021), rate hike selloff (2022), AI rally (2023-2024). This makes your regime detection in Week 4 much more meaningful.

### Why Historical Data First?

You never run a live strategy without first proving it works historically. The flow is:
- **Now (Weeks 1-7):** Use historical data to build and validate strategies
- **Week 8:** Switch to live mode — system runs daily, ingests real prices, trades automatically
- **The `ingest.py` script we built now will also be the script that runs daily in Week 8** — we just add a scheduler

---

## Feature Engineering

Features are computed in `research/features/engineer.py` and saved to `data/processed/features_daily.parquet`.

**What is a feature?** A transformed version of raw price/volume data that captures a meaningful pattern — things like "this stock has been trending up for 2 weeks" or "volume is unusually high today." Features are the inputs your strategies and ML model use to make decisions.

Current feature set: **22 columns** across 43,123 rows (29 stocks × ~1,490 trading days)

---

### Returns

**Columns:** `return_1d`, `return_5d`, `return_10d`, `return_20d`

**How it's calculated:** `pct_change(n)` — how much the closing price changed over the last n trading days as a percentage.

**Example:** `return_5d = 0.03` means the stock is up 3% over the last 5 trading days.

**Why it matters:** Returns are the most fundamental signal in finance. Every momentum strategy is built on the observation that recent returns predict near-future returns — stocks that have gone up tend to keep going up over short horizons (this is called the momentum premium and is one of the most well-documented patterns in academic finance).

---

### Volatility

**Columns:** `volatility_10d`, `volatility_20d`

**How it's calculated:** Rolling standard deviation of `return_1d` over 10 and 20 days.

**Example:** `volatility_20d = 0.025` means the stock has been moving about ±2.5% per day on average over the last 20 days.

**Why it matters:** Volatility appears in almost every part of this project:
- **Week 4 (Regime Detection):** The HMM uses volatility to identify whether the market is in a calm trending state or a choppy volatile state
- **Week 7 (Position Sizing):** You allocate less capital to high-volatility stocks to keep overall portfolio risk balanced
- It's your measure of "how dangerous is this stock right now"

---

### RSI — Relative Strength Index

**Column:** `rsi_14`

**How it's calculated:** Compares average gains on up days versus average losses on down days over the last 14 days, expressed as a number from 0 to 100.

**How to read it:**
- **RSI > 70** → Overbought. The stock has gone up too far too fast and may be due for a pullback
- **RSI < 30** → Oversold. The stock has dropped too far and may be due for a bounce
- **RSI = 50** → Neutral

**Example:** `rsi_14 = 28` means the stock has been falling hard for 2 weeks — more selling than buying. A mean reversion trader would see this as a potential buy opportunity.

**Why it matters:** RSI is the core signal for your **mean reversion strategy**. The intuition is statistical — if a stock has gone up 13 out of the last 14 days, it's likely stretched beyond its fair value and the rubber band will snap back.

---

### MACD — Moving Average Convergence Divergence

**Columns:** `macd_line`, `macd_signal`, `macd_hist`

**How it's calculated:**
1. Compute a fast 12-day exponential moving average (EMA) of price
2. Compute a slow 26-day EMA of price
3. `macd_line` = fast EMA minus slow EMA (measures momentum)
4. `macd_signal` = 9-day EMA of the MACD line (a smoothed version)
5. `macd_hist` = `macd_line` minus `macd_signal` (the gap between them)

**How to read it:**
- When `macd_line` crosses **above** `macd_signal` → momentum turning upward → potential buy
- When `macd_line` crosses **below** `macd_signal` → momentum slowing → potential sell
- `macd_hist > 0` means upward momentum, `macd_hist < 0` means downward momentum

**Example:** `macd_hist = 0.85` and rising means short-term momentum is accelerating upward.

**Why it matters:** MACD is your **momentum signal**. It's one of the most widely used indicators in professional trading and will be a key input to your ML model in Week 5.

---

### Bollinger Bands

**Columns:** `bb_mid`, `bb_upper`, `bb_lower`, `bb_pct`, `bb_bandwidth`

**How it's calculated:**
1. `bb_mid` = 20-day simple moving average of close price
2. `bb_upper` = `bb_mid` + (2 × 20-day standard deviation)
3. `bb_lower` = `bb_mid` - (2 × 20-day standard deviation)
4. `bb_pct` = (close - bb_lower) / (bb_upper - bb_lower)
5. `bb_bandwidth` = (bb_upper - bb_lower) / bb_mid

**How to read `bb_pct`** (the most important one):
- `bb_pct = 1.0` → price is touching the upper band (potentially overbought)
- `bb_pct = 0.0` → price is touching the lower band (potentially oversold)
- `bb_pct = 0.5` → price is right at the middle band (neutral)

**Statistical meaning:** By construction, about 95% of all price action falls within the bands. When price touches the upper or lower band, it's a statistically unusual event.

**How to read `bb_bandwidth`:**
- Narrow bands (small number) → low volatility, market is quiet → often precedes a big breakout
- Wide bands (large number) → high volatility, market is moving a lot

**Example from AAPL (Dec 2025):** `bb_pct = 0.20` means AAPL's price is sitting near the lower 20% of its band — slightly oversold territory.

**Why it matters:** `bb_pct` is the other core signal for your **mean reversion strategy** alongside RSI. `bb_bandwidth` feeds into **regime detection** in Week 4 — when it's very narrow, the market is coiling up for a move.

---

### Volume Ratio

**Column:** `volume_ratio`

**How it's calculated:** Today's volume divided by the 20-day average volume.

**How to read it:**
- `volume_ratio = 1.0` → normal volume day
- `volume_ratio = 2.5` → volume is 2.5x higher than usual — something significant is happening
- `volume_ratio = 0.4` → very quiet day, low conviction

**Example from AAPL (Dec 24, 2025):** `volume_ratio = 0.41` — Christmas Eve, barely anyone trading. Any price move that day means very little.

**Why it matters:** Volume confirms price moves. A 3% rally on 0.5x normal volume is likely noise. A 3% rally on 3x normal volume means real buyers are stepping in with conviction. Your ML model will learn to weight price signals differently based on volume confirmation.

---

### SPY Correlation

**Column:** `spy_correlation`

**How it's calculated:** Rolling 60-day Pearson correlation between each stock's daily returns and SPY's (S&P 500 ETF) daily returns.

**How to read it:**
- `spy_correlation = 0.95` → the stock moves almost in lockstep with the market. Hard to generate alpha from market-driven moves.
- `spy_correlation = 0.20` → the stock moves fairly independently of the market. More interesting for stock-specific strategies.
- `spy_correlation = -0.30` → the stock tends to move opposite to the market (rare — defensive assets, inverse ETFs)

**Example from AAPL (Dec 2025):** `spy_correlation = 0.58` — moderately correlated with the market. About half of AAPL's daily moves can be explained by overall market direction.

**Why it matters:** This feature helps distinguish whether a stock move is driven by company-specific news or just the whole market moving. In Week 7, when building your portfolio, you'll use correlations to ensure your positions aren't all just bets on the same thing (market direction). Low-correlation stocks that are moving strongly are the most interesting signals.

---

## What's a Trading Signal?

A signal is a number between **-1 and +1** assigned to each stock each day:
- **+1** → Strong buy
- **-1** → Strong sell (or short)
- **0** → No position

Your strategies produce signals. Your risk layer (Week 7) converts those signals into actual position sizes (how many dollars to allocate). Your execution layer (Week 8) converts position sizes into real orders sent to Alpaca.

---

## Key Finance Concepts

### What is Backtesting?
Testing a strategy against historical data to see how it would have performed. This is how you validate an idea before risking real (or paper) money. The critical danger is **overfitting** — accidentally building a strategy that only works on past data because you tuned it too specifically to that data.

### What is the Momentum Premium?
The academic observation (documented since the 1990s by Jegadeesh and Titman) that stocks which have performed well over the past 3-12 months tend to continue performing well over the next 3-12 months. One of the most robust and persistent patterns in financial markets.

### What is Mean Reversion?
The idea that extreme price moves tend to snap back toward average. A stock that drops 15% in a week has likely overshot its fair value — scared sellers pushed it too far. Eventually buyers step in and it reverts. RSI and Bollinger Bands are tools for identifying these extremes.

### What is a Hypertable?
TimescaleDB's special table structure optimised for time-series data. It automatically partitions data by time intervals under the hood, making date range queries extremely fast. This is why we use TimescaleDB instead of plain PostgreSQL.

### What is Walk-Forward Validation?
The correct way to validate ML models on financial data. Instead of randomly splitting data into train/test sets (which causes data leakage in time-series), you train on a window of historical data, test on the next out-of-sample period, then roll the window forward and repeat. This gives honest performance estimates that reflect real-world conditions.

### What is a Market Regime?
Financial markets don't behave the same way all the time. Sometimes they trend strongly upward or downward. Sometimes they chop sideways. Sometimes they're extremely volatile. These different environments are called regimes. A momentum strategy works brilliantly in trending regimes but loses badly in choppy ones. Week 4 builds a Hidden Markov Model to detect which regime the market is currently in, so your strategies can adjust their behavior accordingly.

### What is Alternative Data?
Any data source beyond traditional price/volume data — news sentiment, Reddit posts, satellite imagery, credit card transactions, web traffic. Hedge funds pay millions for unique alternative datasets to gain an edge. In Week 6, you'll use FinBERT to process news/Reddit text into sentiment scores as an alternative data signal.

### What is the Sharpe Ratio?
The most common measure of risk-adjusted return. Calculated as (average return - risk-free rate) / standard deviation of returns. A Sharpe of 1.0 is decent. Above 1.5 is good. Above 2.0 is excellent. It answers the question: "How much return are you getting per unit of risk taken?"

---

## File Structure Reference

```
AlphaForge/
├── data/
│   ├── ingest.py              ← pulls OHLCV from Alpaca, stores in TimescaleDB
│   ├── validate.py            ← checks data quality before strategies use it
│   ├── raw/                   ← unused for now, for raw downloaded files
│   └── processed/
│       └── features_daily.parquet  ← computed features, ready for strategies
├── research/
│   ├── features/
│   │   └── engineer.py        ← computes all 22 features from raw OHLCV
│   ├── strategies/            ← Week 3: momentum + mean reversion strategies
│   └── notebooks/             ← for exploratory analysis
├── risk/                      ← Week 7: position sizing, portfolio optimization
├── execution/                 ← Week 8: order management, Alpaca trading
├── dashboard/                 ← Week 9: Streamlit live dashboard
├── models/                    ← Week 5: trained ML models
├── tests/                     ← unit tests
├── config/                    ← configuration files
├── .env                       ← API keys and DB credentials (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Current Dependencies

```
alpaca-py       ← market data and trading API
pandas          ← data manipulation
numpy           ← numerical computing
sqlalchemy      ← database connection
psycopg2        ← PostgreSQL driver
python-dotenv   ← loads .env file
pytz            ← timezone handling
pyarrow         ← parquet file format
```

---

*Last updated: Week 3 — Feature Engineering*