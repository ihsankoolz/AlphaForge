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
| 3 | Feature Engineering & Traditional Strategies | ✅ Done |
| 4 | Regime Detection with Hidden Markov Models | ⏳ Next |
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
     ├──► [research/strategies/momentum.py]       → signal (0 to +1)
     ├──► [research/strategies/mean_reversion.py] → signal (0 to +1)
     │         │
     │         ▼
     │    [research/strategies/backtest.py]
     │    (simulates trading, measures performance)
     │
     ├──► [models/regime_detector.py] ← Week 4: HMM detects market state
     ├──► [models/ml_signal.py]       ← Week 5: XGBoost signal generator
     ├──► [models/sentiment.py]       ← Week 6: FinBERT sentiment scores
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

Why this range? It captures very different market environments — COVID crash (2020), bull run (2021), rate hike selloff (2022), AI rally (2023-2024). This makes regime detection in Week 4 much more meaningful.

### Why Historical Data First?

You never run a live strategy without first proving it works historically. The flow is:
- **Weeks 1-7:** Use historical data to build and validate strategies
- **Week 8:** Switch to live mode — system runs daily, ingests real prices, trades automatically
- The `ingest.py` script we built now will also be the script that runs daily in Week 8 — we just add a scheduler

---

## Feature Engineering

Features are computed in `research/features/engineer.py` and saved to `data/processed/features_daily.parquet`.

**What is a feature?** A transformed version of raw price/volume data that captures a meaningful pattern. Features are the inputs your strategies and ML model use to make decisions.

Current feature set: **22 columns** across 43,123 rows (29 stocks × ~1,490 trading days)

---

### Returns

**Columns:** `return_1d`, `return_5d`, `return_10d`, `return_20d`

**How it's calculated:** `pct_change(n)` — how much the closing price changed over the last n trading days as a percentage.

**Example:** `return_5d = 0.03` means the stock is up 3% over the last 5 trading days.

**Why it matters:** The most fundamental signal in finance. Every momentum strategy is built on the observation that recent returns predict near-future returns — stocks that have gone up tend to keep going up over short horizons (the momentum premium, documented since the 1990s).

---

### Volatility

**Columns:** `volatility_10d`, `volatility_20d`

**How it's calculated:** Rolling standard deviation of `return_1d` over 10 and 20 days.

**Example:** `volatility_20d = 0.025` means the stock has been moving about ±2.5% per day on average over the last 20 days.

**Why it matters:** Appears in almost every part of the project. Week 4 HMM uses it to detect market regimes. Week 7 position sizing allocates less capital to high-volatility stocks. It's your measure of "how dangerous is this stock right now."

---

### RSI — Relative Strength Index

**Column:** `rsi_14`

**How it's calculated:** Compares average gains on up days versus average losses on down days over the last 14 days, expressed as 0 to 100.

**How to read it:**
- **RSI > 70** → Overbought. Gone up too far too fast.
- **RSI < 30** → Oversold. Dropped too far.
- **RSI = 50** → Neutral.

**Why it matters:** Core signal for mean reversion. When combined with Bollinger Band percentage, identifies statistically extreme moves that tend to snap back.

---

### MACD — Moving Average Convergence Divergence

**Columns:** `macd_line`, `macd_signal`, `macd_hist`

**How it's calculated:**
1. Fast 12-day exponential moving average (EMA) of price
2. Slow 26-day EMA of price
3. `macd_line` = fast EMA minus slow EMA
4. `macd_signal` = 9-day EMA of the MACD line
5. `macd_hist` = `macd_line` minus `macd_signal`

**How to read it:**
- `macd_hist > 0` and rising → upward momentum accelerating → buy confirmation
- `macd_hist < 0` → downward momentum

**Why it matters:** Used as momentum confirmation. Prevents buying stocks that went up 20 days ago but have already started reversing. If MACD histogram is negative, recent momentum has stalled — don't buy.

---

### Bollinger Bands

**Columns:** `bb_mid`, `bb_upper`, `bb_lower`, `bb_pct`, `bb_bandwidth`

**How it's calculated:**
1. `bb_mid` = 20-day simple moving average
2. `bb_upper` = `bb_mid` + (2 × 20-day standard deviation)
3. `bb_lower` = `bb_mid` - (2 × 20-day standard deviation)
4. `bb_pct` = (close - bb_lower) / (bb_upper - bb_lower)
5. `bb_bandwidth` = (bb_upper - bb_lower) / bb_mid

**How to read `bb_pct`:**
- `1.0` → price at upper band (stretched high)
- `0.0` → price at lower band (stretched low)
- `0.5` → price at midpoint (neutral)

**Statistical meaning:** ~95% of all price action falls within the bands by construction. Touching them is statistically unusual.

**How to read `bb_bandwidth`:**
- Narrow → quiet market, often precedes a big breakout
- Wide → volatile market

**Why it matters:** `bb_pct` is the second core signal in mean reversion alongside RSI. `bb_bandwidth` feeds into regime detection in Week 4.

---

### Volume Ratio

**Column:** `volume_ratio`

**How it's calculated:** Today's volume divided by 20-day average volume.

**How to read it:**
- `1.0` → normal volume day
- `2.5` → 2.5x higher than usual, significant activity
- `0.4` → very quiet, low conviction

**Why it matters:** Volume confirms price moves. A 3% rally on 0.5x normal volume is likely noise. A 3% rally on 3x normal volume means real buyers are stepping in. Mean reversion filters out signals when volume is below 80% of normal.

---

### SPY Correlation

**Column:** `spy_correlation`

**How it's calculated:** Rolling 60-day Pearson correlation between each stock's daily returns and SPY's daily returns.

**How to read it:**
- `0.95` → moves almost in lockstep with the market
- `0.20` → moves fairly independently
- Negative → tends to move opposite to the market

**Why it matters:** Distinguishes stock-specific moves from broad market moves. Used in portfolio construction (Week 7) to ensure positions aren't all just bets on market direction.

---

## Strategies

### What is a Strategy?

A set of rules that takes features as input and outputs a **signal** — a number between -1 and +1 for each stock each day. The strategy doesn't allocate capital or place orders — that's the risk layer (Week 7) and execution layer (Week 8). The strategy just says "I think this stock is going up" with some level of conviction.

### How Are Strategies Created?

Three ways professionals do it:
1. **Academic research** — documented market patterns backed by decades of peer-reviewed papers
2. **Observation** — a trader notices a pattern, formalises it into rules, tests it historically
3. **ML/data mining** — feed a model thousands of features and let it find patterns (Week 5)

### Why Only Two Strategies Right Now?

More strategies only help if they're genuinely uncorrelated. Momentum and mean reversion are natural opposites:
- Momentum works in **trending markets**
- Mean reversion works in **choppy, range-bound markets**

This directly sets up Week 4 — the regime detector will identify which environment we're in and the system will know which strategy to trust. In Week 5, the ML model becomes a third smarter strategy that learns to combine both signals.

### Why Long-Only?

Our universe is 29 blue-chip companies. Over 6 years including the biggest bull run in history, even the "weakest" stocks went up. Shorting the relative losers in a universe of winners is structurally broken — confirmed by signal diagnostics which showed sell signals averaged +0.088% next-day return (the stocks we were shorting kept going up).

This is not overfitting. It's recognising a structural mismatch between strategy design and universe composition. Short selling becomes viable in Week 7 when we expand the universe to include genuinely weak companies alongside the blue chips.

---

### Momentum Strategy (`research/strategies/momentum.py`)

**Core idea:** Every day, rank all 29 stocks by their 20-day return relative to each other. Buy the top 20% if MACD confirms momentum is still active.

**Why cross-sectional ranking?** Compares stocks against each other, not just their own history. Automatically adjusts to market conditions — in a bull market where everything is up, you still only buy the strongest relative performers.

**Signal scaling:** A stock ranked top 5% gets a stronger signal than one ranked top 21%.

**Backtest results (2020-2025):**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Return | +91.85% | $100k → $191k |
| Annualised Return | +11.67% | Slightly below SPY's ~14% |
| Sharpe Ratio | 0.56 | Decent for a first-pass rule-based strategy |
| Max Drawdown | -30.24% | Gets hurt badly in crashes |
| Win Rate | 51.24% | Slightly better than a coin flip |

---

### Mean Reversion Strategy (`research/strategies/mean_reversion.py`)

**Core idea:** Buy stocks that are statistically oversold — RSI below 35 AND price near the lower Bollinger Band. Both signals must agree.

**Why require both RSI and Bollinger Band?** A stock can have low RSI just because it's a fundamentally bad company in long-term decline — that's a falling knife, not mean reversion. Requiring both signals reduces false positives.

**Volume filter:** Only trade if volume ≥ 80% of normal. Thin market moves are unreliable.

**Backtest results (2020-2025):**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Return | +8.71% | Weak — barely above cash |
| Annualised Return | +1.42% | Well below SPY |
| Sharpe Ratio | 0.19 | Very low |
| Max Drawdown | -45.16% | Worse than momentum |
| Win Rate | 28.31% | Looks bad but misleading — mostly cash days |

**Important nuance:** The signal quality is actually good — average next-day return after a buy signal is +0.21%, stronger than momentum's +0.077%. The problem is frequency — RSI < 35 AND bb_pct < 0.25 is a rare combination, so the strategy spends most of its time in cash. The ML model in Week 5 will learn to find subtler versions of the same pattern and generate signals more often.

---

## Backtesting Engine (`research/strategies/backtest.py`)

### What is Backtesting?

Simulating how a strategy would have performed historically. Output is a daily portfolio value curve showing how $100,000 would have grown or shrunk day by day.

### Key Design Decisions

**Signal shift (most critical):** Signals on day T execute on day T+1. You can't trade at today's close the moment your signal fires. Without this shift we saw a fantasy 146,600% return — that's called **lookahead bias** and it's the most common dangerous mistake in amateur quant research.

**Transaction cost 0.1% per trade:** Models bid-ask spread plus broker commission. Without this, a frequently-trading strategy can look profitable but lose money in practice.

**Equal weight positions:** Capital split equally across all active positions. Simple and robust.

**Maximum 10 positions:** Prevents over-concentration.

### Signal Diagnostic Tool

Before trusting backtest results, run `diagnose_signals()` to check: does today's signal actually predict tomorrow's return in the right direction?

- Buy signals followed by positive returns on average → signal has genuine edge
- Buy signals followed by negative returns → signal is backwards or broken

This diagnostic revealed our sell signals were pointing the wrong direction (sold stocks kept going up), leading to the long-only redesign.

### Common Backtesting Traps

**Lookahead bias** — using future information. Fixed by shifting signals one day forward.

**Survivorship bias** — only testing companies that still exist today. Our fixed 29-stock universe has mild survivorship bias — worth acknowledging when presenting results.

**Overfitting** — tuning parameters so specifically to historical data they stop working on new data. We kept parameters simple and round (RSI < 35, top 20%) rather than optimising precisely.

**Transaction cost neglect** — ignoring trading costs. A strategy that trades every day and looks profitable before costs can easily lose money after costs.

---

## Current Backtest Results Summary

| Strategy | Total Return | Ann. Return | Sharpe | Max Drawdown | Notes |
|----------|-------------|-------------|--------|--------------|-------|
| Momentum | +91.85% | +11.67% | 0.56 | -30.24% | Slightly above SPY total return |
| Mean Reversion | +8.71% | +1.42% | 0.19 | -45.16% | Good signal, too infrequent |
| SPY Benchmark | ~85-90% | ~14% | ~0.8 | ~-34% | Buy and hold comparison |

Both strategies are long-only due to universe composition. Short selling revisited in Week 7.

---

## What's Coming Next

### Week 4 — Regime Detection (Hidden Markov Models)

Build a model that classifies each day into one of three market states: trending bull, trending bear, or high-volatility/choppy. Then route signals — momentum fires in trending regimes, mean reversion fires in choppy regimes.

**Why this is the most impressive addition:** Real hedge funds use regime detection. Almost no student projects include it. It also demonstrates understanding of a core quant finance concept — markets are non-stationary, meaning the same rules don't work all the time.

**Inputs from what we've built:** `volatility_20d` and `return_20d` features are the primary HMM inputs.

### Week 5 — ML Signal Generation (XGBoost + MLflow)

Train XGBoost on all 22 features to predict next-day direction. Use walk-forward validation for honest out-of-sample results. Log all experiments in MLflow.

**Why walk-forward not k-fold:** Standard k-fold randomly shuffles data causing data leakage in time series. Walk-forward always tests on the next unseen period — reflects real-world conditions honestly.

**Expected improvement over rule-based strategies:** The model will learn subtler patterns and generate mean-reversion-style signals more frequently, fixing the "barely ever invested" problem.

### Week 6 — Sentiment as Alternative Data (FinBERT)

Pull financial news/Reddit posts for your stocks, run through FinBERT, aggregate into daily sentiment scores, add as features to the ML model. Directly connects to your TEMPO project experience with NLP and transformers.

### Week 7 — Risk & Portfolio Layer

Kelly Criterion for position sizing, mean-variance optimization for portfolio weights, proper transaction cost modeling. Also revisit universe expansion to 200+ stocks including mid/small caps — makes short selling viable.

### Week 8 — Live Paper Trading

Connect to Alpaca paper trading API. Schedule pipeline to run daily at market open. Add safeguards (max position size, daily loss limits). `ingest.py` becomes a live daily system.

### Week 9 — Streamlit Dashboard

Live P&L curve, current positions, strategy performance by regime, sentiment signals, risk metrics updating daily.

### Week 10 — Polish & Write-Up

Clean README, demo video, Medium article on one interesting finding, docstrings throughout.

---

## Key Finance Concepts

**Momentum Premium** — stocks that performed well over the past 3-12 months tend to continue performing well. Academic finding since 1993 (Jegadeesh and Titman), one of the most robust patterns in finance.

**Mean Reversion** — extreme price moves tend to snap back toward average. Fear and greed cause prices to overshoot fair value. RSI and Bollinger Bands measure this overshoot.

**Cross-Sectional vs Time-Series Signals**
- Cross-sectional: compare stocks against each other (momentum — rank all 29 stocks)
- Time-series: compare a stock against its own history (mean reversion — is this stock oversold vs its own recent prices?)

**Universe Composition Bias** — a strategy must be designed for its universe. Shorting losers in a universe of blue-chip winners is structurally broken. This is different from overfitting.

**Lookahead Bias** — using future information in a backtest. Causes unrealistically good results. Fixed by shifting signals one day forward.

**Spread** — the difference between buy price and sell price. On a $100 stock with a $0.05 spread, you lose $0.05 immediately on purchase. Our 0.1% transaction cost models this plus broker commission.

**Walk-Forward Validation** — train on a rolling window, test on the next unseen period, roll forward, repeat. The honest way to validate ML models on time-series data.

**Market Regime** — markets behave differently in different environments: trending bull, trending bear, high-volatility/choppy. Week 4 builds a Hidden Markov Model to detect these states automatically.

**Alternative Data** — any data beyond price/volume: news sentiment, Reddit posts, satellite imagery. Week 6 uses FinBERT to turn financial text into trading signals.

**Sharpe Ratio** — (average return - risk free rate) / volatility, annualised. Return per unit of risk. Above 1.0 decent, above 1.5 good, above 2.0 excellent.

**Max Drawdown** — worst peak-to-trough decline during the backtest. The number that determines whether real investors would stay invested or panic. Reducing it is a primary goal of Week 7.

**Hypertable** — TimescaleDB's table structure that auto-partitions data by time, making date-range queries extremely fast.

**Survivorship Bias** — only testing on companies that still exist today skews results upward. Our fixed 29-stock universe has mild survivorship bias.

---

## File Structure Reference

```
AlphaForge/
├── data/
│   ├── ingest.py                           ← pulls OHLCV from Alpaca, stores in TimescaleDB
│   ├── validate.py                         ← checks data quality
│   ├── raw/
│   └── processed/
│       ├── features_daily.parquet          ← 43,123 rows, 22 features
│       ├── backtest_momentum.parquet       ← momentum strategy daily P&L
│       └── backtest_mean_reversion.parquet ← mean reversion daily P&L
├── research/
│   ├── features/
│   │   └── engineer.py                    ← computes all 22 features
│   ├── strategies/
│   │   ├── momentum.py                    ← cross-sectional momentum, long-only
│   │   ├── mean_reversion.py              ← RSI + Bollinger Band, long-only
│   │   └── backtest.py                    ← simulation engine + performance metrics
│   └── notebooks/
├── risk/                                  ← Week 7
├── execution/                             ← Week 8
├── dashboard/                             ← Week 9
├── models/                                ← Weeks 4, 5, 6
├── tests/
├── config/
├── .env                                   ← API keys + DB credentials (never commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Current Dependencies

```
alpaca-py        ← market data and trading API
pandas           ← data manipulation
numpy            ← numerical computing
sqlalchemy       ← database connection
psycopg2         ← PostgreSQL driver
python-dotenv    ← loads .env file
pytz             ← timezone handling
pyarrow          ← parquet file format
```

---

*Last updated: End of Week 3 — Feature Engineering, Strategies & Backtesting complete*