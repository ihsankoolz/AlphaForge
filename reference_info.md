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
| 4 | Regime Detection with Hidden Markov Models | ✅ Done |
| 5 | ML Signal Generation (XGBoost) | ✅ Done |
| 6 | Sentiment Layer as Alternative Data | ✅ Done (pipeline complete, integration pending) |
| 7 | Risk & Portfolio Layer | ⏳ Next |
| 8 | Execution Layer & Paper Trading | ⏳ Upcoming |
| 9 | Dashboard & Visualization | ⏳ Upcoming |
| 10 | Polish, Documentation & Write-Up | ⏳ Upcoming |

---

## System Architecture — How Everything Fits Together

```
[Alpaca API]
     │
     ├──► [Alpaca News API] ──► [models/sentiment.py] ──► [sentiment_daily.parquet]
     │                               (FinBERT scoring)              │
     ▼                                                               │
[data/ingest.py] ──────────────────► [TimescaleDB]                  │
     │                                     │                        │
     │                                     ▼                        │
     │                          [data/validate.py]                  │
     │                                     │                        │
     ▼                                     ▼                        │
[research/features/engineer.py] ◄── loads from TimescaleDB          │
     │                                                              │
     ▼                                                              │
[data/processed/features_daily.parquet]                             │
     │                                                              │
     ├──► [research/strategies/momentum.py]    → signal (0 to +1)  │
     ├──► [research/strategies/mean_reversion.py] → signal (0 to +1)│
     │         │                                                    │
     │         ▼                                                    │
     │    [research/strategies/backtest.py]                         │
     │                                                              │
     ├──► [models/regime_hmm.py] ──► [regime_labels.parquet]       │
     │         │                                                    │
     │         ▼                                                    │
     │    [research/strategies/regime_switcher.py]                  │
     │                                                              │
     └──► [models/ml_signal.py] ◄── regime labels + rule signals ◄─┘
               │                 ◄── sentiment features (Week 7)
               ▼
     [data/processed/ml_signals.parquet]
               │
               ▼
     [research/strategies/ml_backtest.py]
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

---

## Feature Engineering

Features are computed in `research/features/engineer.py` and saved to `data/processed/features_daily.parquet`.

**What is a feature?** A transformed version of raw price/volume data that captures a meaningful pattern. Features are the inputs your strategies and ML model use to make decisions.

Current feature set: **22 columns** across 43,123 rows (29 stocks × ~1,490 trading days)

### Returns
**Columns:** `return_1d`, `return_5d`, `return_10d`, `return_20d`
How much the closing price changed over the last n trading days. `return_5d = 0.03` means up 3% over 5 days. The most fundamental signal in finance — the backbone of every momentum strategy.

### Volatility
**Columns:** `volatility_10d`, `volatility_20d`
Rolling standard deviation of daily returns. `volatility_20d = 0.025` means the stock moves ±2.5% per day on average. Used by HMM for regime detection and will be used by Week 7 position sizing.

### RSI — Relative Strength Index
**Column:** `rsi_14`
Compares average gains vs losses over 14 days, expressed as 0-100. Above 70 = overbought, below 30 = oversold, 50 = neutral. Core mean reversion signal. We used thresholds of RSI < 35 (buy) and RSI > 65 (sell) in our rule-based strategy.

### MACD — Moving Average Convergence Divergence
**Columns:** `macd_line`, `macd_signal`, `macd_hist`
Difference between fast (12-day) and slow (26-day) exponential moving averages. `macd_hist` = macd_line − signal line. Positive histogram = upward momentum building. Used as confirmation signal in momentum strategy.

### Bollinger Bands
**Columns:** `bb_mid`, `bb_upper`, `bb_lower`, `bb_pct`, `bb_bandwidth`
Price envelope 2 standard deviations above/below a 20-day moving average. `bb_pct` = where price sits within the band (0 = lower band, 1 = upper band). Below 0 = statistically oversold. Second most important feature in the XGBoost model (16% importance).

### Volume Ratio
**Column:** `volume_ratio`
Today's volume divided by 20-day average volume. Ratio > 1.5 means unusually high trading activity, often signals a significant move.

### SPY Correlation
**Column:** `spy_correlation`
Rolling 20-day correlation with SPY (S&P 500 ETF). Measures how much a stock moves with the overall market. Used by XGBoost as the 4th most important feature.

---

## Backtesting Engine (`research/strategies/backtest.py`)

### What is Backtesting?
Simulating how a strategy would have performed historically. Output is a daily portfolio value curve showing how $100,000 would have grown or shrunk day by day.

### Key Design Decisions
**Signal shift (most critical):** Signals on day T execute on day T+1. Without this we saw a fantasy 146,600% return — that's lookahead bias, the most common dangerous mistake in amateur quant research.

**Transaction cost 0.1% per trade:** Models bid-ask spread plus broker commission.

**Equal weight positions:** Capital split equally across all active positions.

**Maximum 10 positions:** Prevents over-concentration.

### What Went Wrong and What We Learned

**First run results were completely broken:**
- Momentum showed 146,600% total return — impossible, caused by lookahead bias (signal and execution on same day)
- Mean reversion showed -100% — went completely bankrupt because it was buying overbought stocks and selling oversold ones (inverted logic)
- Fix: Added `signals_wide = signals_wide.shift(1)` — one line that makes the entire backtest realistic

**Short selling destroyed both strategies:**
- Signal diagnostic showed sell signals generated positive returns on average — we were shorting stocks that kept going up
- Root cause: our universe is 29 blue-chip winners. Even the "losers" in a universe of winners go up over time
- Fix: Made both strategies long-only. This is not overfitting — it's recognising universe composition bias
- Plan: Revisit short selling in Week 7 when universe potentially expands to 200+ stocks

**Mean reversion was too selective:**
- RSI < 35 AND bb_pct < 0.25 simultaneously is rare — strategy barely invested
- Win rate of 28% looked bad but was because most days had zero positions (counted as losses)
- Signal quality was actually good (+0.21% avg next-day return after buy signal)
- Fix: This will be addressed when ML model learns to fire the signal more intelligently

### Signal Diagnostic Tool
Before trusting backtest results, always run `diagnose_signals()`:
- Buy signals followed by positive avg returns → signal has genuine edge
- Buy signals followed by negative avg returns → signal is backwards or broken
- Check that higher confidence corresponds to better returns

---

## Current Backtest Results Summary

| Strategy | Total Return | Ann. Return | Sharpe | Max Drawdown | Notes |
|----------|-------------|-------------|--------|--------------|-------|
| Momentum (standalone) | +91.85% | +11.67% | 0.56 | -30.24% | Long-only, 29 stocks, 2020-2025 |
| Mean Reversion (standalone) | +8.71% | +1.42% | 0.19 | -45.16% | Good signal, too infrequent |
| Regime Switcher | +186.65% | +19.54% | 0.96 | -23.25% | Best performer to date |
| ML Signal (XGBoost) | +38.02% | +7.63% | 0.52 | -26.43% | Shorter history (2021-2025 only) |
| SPY Benchmark | ~85-90% | ~14% | ~0.80 | ~-34% | Buy and hold comparison |

**Important note on ML result:** The ML strategy only covers 2021-2025 (4 years) because walk-forward validation needed 18 months of training before first prediction. The regime switcher covers the full 6 years including the strong 2020-2021 bull run. This is an unfair comparison — the ML Sharpe of 0.52 is roughly equivalent to standalone momentum (0.56) over the same period.

---

## Week 4 — Regime Detection with Hidden Markov Models

### What We Built
`models/regime_hmm.py` — trains a 3-state Gaussian HMM on daily market-level features to classify each trading day into bull, choppy, or bear regime.

### How the HMM Works
- "Hidden" because regimes are not directly observable — only their effects (returns, volatility) are
- "Gaussian" because each hidden state emits observations drawn from a normal distribution
- Each regime has its own mean and variance of the 7 input features
- The model learns: (1) transition matrix — probability of moving between regimes, (2) emission parameters — what each regime looks like

### Input Features (market-level, aggregated across all 29 stocks per day)
- `mean_return` — avg daily return across universe
- `vol_return` — cross-sectional dispersion of returns
- `mean_volatility` — avg rolling volatility
- `mean_rsi` — avg RSI (is the market overbought overall?)
- `mean_macd_hist` — avg MACD momentum
- `mean_bb_pct` — avg Bollinger Band position
- `mean_volume_ratio` — unusual volume activity

### What Went Wrong and What We Learned

**First HMM attempt — one state captured a single outlier day:**
- The model dedicated an entire state to one extreme day (likely March 24, 2020, COVID bounce ~+9%)
- Labelling by mean return caused this — one extreme day stole the "bull" label
- Fix 1: Switched labelling to use volatility instead of return (lowest vol = bull, highest vol = bear)
- Fix 2: Winsorized features at 1st and 99th percentile before training — clips extremes so outliers can't consume a state
- Lesson: Always winsorize before training HMMs on financial data

**Bear regime is actually "high volatility" not "trending down":**
- Bear regime showed positive average return (+0.20%) with highest volatility (0.0320)
- This is correct and actually more useful — high volatility periods include both crashes AND violent recoveries
- Momentum strategy actually performed well in bear regime (Sharpe 0.91) because it caught those recoveries

### Final Regime Characteristics

| Regime | Days | % of Time | Avg Daily Return | Avg Volatility | Avg RSI |
|--------|------|-----------|-----------------|----------------|---------|
| Bull | 810 | 54.5% | +0.14% | 0.0148 | 54.7 |
| Choppy | 357 | 24.0% | -0.17% | 0.0180 | 48.8 |
| Bear | 320 | 21.5% | +0.20% | 0.0320 | 51.2 |

**Transition probabilities:**
- Bull: 93.5% chance of staying bull next day
- Bear: 97.2% chance of staying bear next day (crashes are persistent)
- Choppy: 83.2% chance of staying choppy (most transient regime)

### Regime Switcher Results — The Key Insight

Original hypothesis: momentum in bull, mean reversion in choppy, cash in bear.

**Per-regime diagnostic blew up the hypothesis:**

| Strategy | Bull Sharpe | Choppy Sharpe | Bear Sharpe |
|----------|-------------|---------------|-------------|
| Momentum | 1.57 | -0.96 | 0.91 |
| Mean Reversion | 1.52 | -1.28 | -0.08 |

Mean reversion underperformed momentum in every single regime. The correct switching rule, discovered from data:
- Bull → momentum ✓
- Bear → momentum (high vol doesn't mean down — catches recoveries)
- Choppy → cash (neither strategy has edge, preserve capital)

**Result: Sharpe improved from 0.56 → 0.96, total return from 91.85% → 186.65%, max drawdown reduced from -30.24% → -23.25%**

Just sitting in cash during 357 choppy days doubled the return and improved risk metrics. This is the entire value of regime detection in one number.

### Files Produced
- `data/processed/regime_labels.parquet` — daily regime label for each trading day
- `models/hmm_model.pkl` — trained HMM model + scaler for live inference

---

## Week 5 — ML Signal Generation (XGBoost)

### What We Built
`models/ml_signal.py` — XGBoost classifier trained via walk-forward validation to predict which stocks will be top-quartile performers over the next 5 days. Output is a probability score per stock per day used as trading signal strength.

`research/strategies/ml_backtest.py` — backtests trading using ML probability scores as signals, with signal-weighted position sizing.

### Design Decisions and Why

**Target variable — top quartile over 5 days (not binary up/down tomorrow):**
- Predicting next-day direction for large-cap stocks is nearly impossible — EMH means professional quants have already arbitraged away most 1-day patterns
- Predicting relative rank within universe over 5 days is more tractable
- "Will this stock be in the top 25% of performers this week?" is a question the model can actually learn to answer
- Mean AUC of 0.83 validated this choice vs 0.506 with 1-day binary target

**Walk-forward validation (18-month initial window, quarterly retraining):**
- Train on 2020→mid-2021, predict Q3 2021
- Train on 2020→Q3 2021, predict Q4 2021
- ...continues quarterly to end of data
- 18 months initial window (not 12) because 5-day targets have fewer signal events
- This gives honest out-of-sample predictions — no future data ever touches training

**Rule-based signals as ML features:**
- `momentum_signal` and `mr_signal` from Weeks 3 added as input features
- ML model learns when to trust momentum signals and when to ignore them
- ML model learns to combine domain knowledge with raw features in ways we couldn't hand-code

**Class imbalance correction:**
- Top quartile only occurs 25% of the time → model would lazily predict "not top quartile" for everything
- `scale_pos_weight` tells XGBoost to upweight the positive class by ~3x
- Fixed the model learning to always predict the majority class

### What Went Wrong and What We Learned

**First ML attempt (v1) — results were essentially random:**
- AUC: 0.506 (barely above 0.5 coinflip baseline)
- Accuracy: 0.511 (worse than just always predicting "up" at 52.3%)
- Signal quality table was non-monotonic — high confidence didn't predict better returns
- Root cause: predicting binary up/down tomorrow is too hard for large-cap liquid stocks
- All three fixes applied: better target (5-day rank), rule-based signals as features, class imbalance correction

**v2 results after fixes:**
- AUC: 0.829 (excellent for financial prediction)
- Signal quality perfectly monotonic — Very Low bucket averages -3.58%, Very High averages +4.43%
- Very High confidence stocks hit top quartile 67.84% of the time

### Feature Importance (XGBoost v2)

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| return_5d | 35.2% | 5-day momentum dominates — recent performance predicts near-future rank |
| bb_pct | 16.3% | Bollinger Band position — second most important technical signal |
| return_10d | 6.4% | Medium-term momentum |
| spy_correlation | 5.9% | How much stock tracks market |
| return_1d | 5.4% | Very recent momentum |
| momentum_signal | 4.8% | Rule-based momentum validated by ML |
| regime_encoded | 4.1% | HMM regime context matters |
| rsi_14 | 3.5% | RSI still contributes |
| mr_signal | 1.7% | Mean reversion signal adds minor value |

Key insight: return_5d dominating confirms the momentum premium is real. RSI and MACD (backbone of rule-based strategies) are at the bottom, explaining why those strategies needed ML enhancement.

### Why ML Backtest Underperforms Regime Switcher

The standalone ML backtest shows +38% vs regime switcher's +186%. This is NOT because ML is worse. Three reasons:
1. ML signals only available from August 2021 (18-month training warmup), missing the best 2020-2021 bull run
2. Fewer positions per day (avg 5.4 vs 10) — model is more selective
3. The comparison is unfair — different time periods

**The correct role of the ML model:** It is not a standalone strategy. It is the brain that all other components plug into. In Week 7, ML probability scores become inputs to the portfolio optimizer. The regime switcher was a hand-coded proof of concept — the ML model is the production replacement that does the same thing but smarter, incorporating all 22 features + regime + sentiment.

### Signal-Weighted Position Sizing
Unlike rule-based strategies (equal weight), ML backtest allocates capital proportional to signal strength. If AAPL has probability 0.80 and NVDA 0.60 and BAC 0.40, weights are 44%/33%/22% respectively. Higher model confidence = more capital.

### Files Produced
- `data/processed/ml_signals.parquet` — probability scores for every stock on every out-of-sample day
- `data/processed/backtest_ml.parquet` — ML strategy daily P&L
- `models/xgb_model.pkl` — final trained XGBoost model for live inference

---

## Week 6 — Sentiment Layer (FinBERT)

### What We Built
`models/sentiment.py` — fetches financial news from Alpaca News API, runs through FinBERT, aggregates into daily per-stock sentiment scores.

### How FinBERT Works
FinBERT is BERT fine-tuned specifically on financial text. Regular sentiment models trained on movie reviews or Twitter don't understand financial language — "the company missed estimates" is negative but "the stock fell on heavy volume" requires financial context a general model would miss. Output is positive/negative/neutral classification with confidence score. We convert to a single sentiment score: +score if positive, -score if negative, 0 if neutral.

### What Went Wrong and What We Learned

**Alpaca NewsRequest expects a string, not a list:**
- `symbols=['AAPL', 'MSFT']` → validation error
- Fix: `symbols=','.join(symbols)` — comma-separated string

**Response structure was not what we expected:**
- `news.news` returned None — the attribute doesn't exist
- Actual data lives at `news.data['news']` — accessed by iterating the NewsSet object
- Articles are `News` objects (use `getattr(article, 'headline')`) not dicts (don't use `article.get('headline')`)

**First fetch attempt — 2.7% coverage (only 3,550 articles):**
- Fetching all 29 symbols together with limit=50 per monthly chunk = 50 articles shared across all stocks
- Popular stocks like AAPL dominated, smaller stocks got almost nothing
- Fix: Fetch each symbol separately with quarterly chunks — every stock gets its own 200-article budget
- Result: 72,193 articles, 67% coverage — 20x improvement

**Per-symbol deduplication:**
- Same article can appear when fetching AAPL and when fetching MSFT (if article mentions both)
- Use a `seen_ids` set to deduplicate by article ID
- 72,193 unique articles → 105,542 (symbol, article) pairs after expansion (one article counted once per symbol it mentions)

### Daily Sentiment Features Produced

| Feature | Description |
|---------|-------------|
| `sentiment_mean` | Average FinBERT score across all articles that day (-1 to +1) |
| `sentiment_std` | Disagreement between articles — high std = uncertainty |
| `sentiment_pos_pct` | % of articles classified positive |
| `sentiment_neg_pct` | % of articles classified negative |
| `article_count` | Number of articles — high count = more news attention |
| `sentiment_3d` | Rolling 3-day average sentiment — smooths single-day noise |

### Coverage Statistics
- 67% of stock-days have at least one article
- 3.4 articles per stock-day on average
- Sentiment range: -0.977 to +0.960 (full scale used)
- Mean sentiment: +0.008 (slightly positive — financial news is naturally optimistic)
- Median: 0.000 (half the days neutral — financial news is often factual)

### Important: Caching
First run takes ~15-20 minutes (API fetching + FinBERT inference). Both raw news and scored articles are cached as parquet files. Subsequent runs are instant. If you need to refetch, delete `data/processed/news_raw.parquet` and `data/processed/news_scored.parquet`.

### Next Step (Not Yet Done)
Sentiment features need to be added to `ml_signal.py` as additional input features and the model retrained. Expected to improve AUC above 0.83. This is the first thing to do in the next session.

### Files Produced
- `data/processed/news_raw.parquet` — raw article text with dates and symbols
- `data/processed/news_scored.parquet` — articles with FinBERT labels and scores
- `data/processed/sentiment_daily.parquet` — daily aggregated sentiment per stock

---

## What's Coming Next

### Immediate Next Step — Integrate Sentiment into ML Model
Add `sentiment_mean`, `sentiment_3d`, `sentiment_pos_pct`, `sentiment_neg_pct`, `article_count` as features to `ml_signal.py`. Retrain with walk-forward validation. Measure if AUC improves above 0.83.

### Week 7 — Risk & Portfolio Layer
- **Kelly Criterion** for position sizing — mathematically optimal bet sizing given edge and odds
- **Mean-variance optimization** for portfolio weights — classic Markowitz framework
- **Universe expansion** to 200+ stocks including mid/small caps — makes short selling viable (small caps can be genuine losers unlike our blue-chip universe)
- **Cash alternative** — instead of pure cash in choppy regime, rotate into bonds or low-vol ETF
- Replace equal-weight positions with signal-weighted or volatility-adjusted weights throughout

### Week 8 — Live Paper Trading
Connect to Alpaca paper trading API. Schedule pipeline to run daily at market open. Add safeguards (max position size, daily loss limits). `ingest.py` becomes a live daily system.

### Week 9 — Streamlit Dashboard
Live P&L curve, current positions, strategy performance by regime, sentiment signals, risk metrics updating daily.

### Week 10 — Polish & Write-Up
Clean README, demo video, docstrings throughout, Medium article on one interesting finding (regime detection value-add is a compelling story).

---

## OOP vs Functional Design

The current codebase is written functionally (plain functions, no classes). This is intentional for research code — linear flow, easier to debug, faster to iterate.

**Where OOP would add genuine value:**
- `Strategy` base class with `generate_signals()` and `calculate_positions()` methods — momentum and mean reversion both inherit from it
- `Portfolio` object that tracks positions, cash, P&L as state (needed in Week 7)
- `DataFeed` class managing WebSocket connections for live data (needed in Week 8)
- `MLSignalModel` class with `train()`, `predict()`, `save()`, `load()` methods

**Rule of thumb:** Use functions when data flows in one direction (research). Use classes when you need to maintain state over time (live trading). As the project moves from research into execution, OOP will naturally appear.

---

## Key Technical Gotchas to Remember

**Timezone issues are everywhere in this project:**
- `features_daily.parquet` stores timestamps with UTC timezone (`+00:00`)
- Backtest portfolio history uses plain dates (tz-naive)
- Always normalize with `.dt.normalize().dt.tz_localize(None)` before joining DataFrames
- Order matters: `.normalize()` THEN `.tz_localize(None)` — doing it in reverse raises AttributeError

**Parquet index structure:**
- `features_daily.parquet` has a MultiIndex of `['time', 'symbol']` — NOT columns
- Always `df.reset_index()` before trying to access `time` or `symbol` as columns
- Momentum and mean reversion strategy files expect the original `['time', 'symbol']` index — pass them the raw parquet, not the reset version

**Alpaca News API:**
- `symbols` parameter must be a comma-separated string, not a list
- Response is a `NewsSet` object — data at `news.data['news']`, NOT `news.news`
- Articles are `News` objects — use `getattr(article, 'headline')` NOT `article.get('headline')`
- Rate limit: add `time.sleep(0.3)` between requests

**HMM training:**
- Always winsorize features at 1st/99th percentile before training — one extreme day can consume an entire state
- Label regimes by volatility (stable) not by mean return (fragile to outliers)
- Log-likelihood scores are NOT comparable across models trained on different data

**XGBoost for financial data:**
- `scale_pos_weight` is critical when target class is rare (top quartile = 25% of data)
- `min_child_weight=30` prevents overfitting on small leaf nodes
- `max_depth=4` keeps trees shallow — financial data has weak signals, deep trees overfit
- Signal threshold for top-quartile prediction should be 0.35 not 0.5 (probabilities cluster below 0.5 when positive class is rare)

---

## Key Finance & Technical Concepts

**Momentum Premium** — stocks that performed well over the past 3-12 months tend to continue performing well. Academic finding since 1993 (Jegadeesh and Titman), one of the most robust patterns in finance. Our XGBoost confirmed this — return_5d is the single most important feature at 35%.

**Mean Reversion** — extreme price moves tend to snap back toward average. Fear and greed cause prices to overshoot fair value. RSI and Bollinger Bands measure this overshoot.

**Cross-Sectional vs Time-Series Signals**
- Cross-sectional: compare stocks against each other (momentum — rank all 29 stocks by recent return)
- Time-series: compare a stock against its own history (mean reversion — is AAPL oversold vs its own recent prices?)

**Universe Composition Bias** — a strategy must be designed for its universe. Shorting "losers" in a universe of blue-chip winners is structurally broken because even the worst large-cap stock tends to go up over time. This is different from overfitting — it's a mismatch between strategy design and universe design.

**Lookahead Bias** — using future information in a backtest. Causes unrealistically good results. Fixed by shifting signals one day forward (`signals.shift(1)`). This was the cause of the 146,600% fantasy return we saw early on.

**Spread** — the difference between buy price and sell price. On a $100 stock with a $0.05 spread, you lose $0.05 immediately on purchase. Our 0.1% transaction cost models this plus broker commission.

**Walk-Forward Validation** — train on all data up to today, predict tomorrow, move forward and repeat. The honest way to validate ML models on time-series data. Standard k-fold cross-validation is wrong for time series because it leaks future data into training.

**Market Regime** — markets behave differently in different environments: trending bull, trending bear, high-volatility/choppy. An HMM detects these states automatically from price and volume features.

**Hidden Markov Model (HMM)** — statistical model where you can't directly observe the states (bull/bear/choppy) but you can infer them from observable data (returns, volatility). "Hidden" = unobservable states, "Markov" = next state depends only on current state (memoryless).

**Transition Matrix** — in an HMM, the probability of moving from one regime to another. Our bear regime has 97.2% probability of staying bear — crashes are persistent, they don't resolve in one day.

**Winsorization** — clipping extreme values at defined percentiles (e.g. 1st and 99th) before feeding into a model. Prevents outlier days from dominating model training. Essential for financial data which has fat tails.

**Fat Tails** — financial return distributions have more extreme events than a normal distribution would predict. Crashes and rallies happen more often than normal bell curve math suggests. This is why winsorization is necessary.

**Alternative Data** — any data beyond price/volume: news sentiment, Reddit posts, satellite imagery, credit card transactions. Week 6 uses FinBERT to turn financial text into trading signals.

**FinBERT** — BERT (Bidirectional Encoder Representations from Transformers) fine-tuned specifically on financial text. Understands financial language better than general-purpose sentiment models. Classifies text as positive/negative/neutral with confidence score.

**Sharpe Ratio** — (average return − risk-free rate) / volatility, annualised. Return per unit of risk. Above 1.0 decent, above 1.5 good, above 2.0 excellent. Most important single metric for comparing strategies.

**Sortino Ratio** — like Sharpe but only counts downside volatility as "risk." Big upward swings aren't really risk — you only care about drops. A higher Sortino than Sharpe means your losses are smaller than your gains.

**Max Drawdown** — worst peak-to-trough decline during the backtest. The number that determines whether real investors would stay invested or panic. Reducing it is a primary goal of Week 7.

**Win Rate** — percentage of trading days where the portfolio made money. Can be misleading in isolation — a strategy can have a low win rate but still be profitable if wins are larger than losses. Mean reversion's 28% win rate looked alarming but was because most days had zero positions.

**Signal-Weighted Position Sizing** — allocate capital proportional to signal strength rather than equally. If model is twice as confident about AAPL as NVDA, AAPL gets twice the capital. More sophisticated than equal weighting.

**AUC (Area Under the ROC Curve)** — measures how well a classifier ranks predictions. 0.5 = random coinflip, 1.0 = perfect. In financial ML, above 0.54 is considered meaningful edge. Our XGBoost achieved 0.83 — excellent.

**Log-Likelihood** — how well an HMM explains observed data. The log of the probability that the model would generate those observations. More negative = worse fit, less negative = better fit. Only comparable across models trained on identical data.

**Scale_pos_weight** — XGBoost parameter that upweights the minority class in imbalanced datasets. If negative class is 3x more common than positive, set scale_pos_weight=3 to compensate.

**Kelly Criterion** — mathematical formula for optimal position sizing given your edge (expected return) and odds. Tells you exactly what fraction of capital to risk on each trade. Will be implemented in Week 7.

**Mean-Variance Optimization** — Markowitz's classic framework for building portfolios. Given expected returns and correlations between assets, find the combination of weights that maximises return for a given level of risk. Will be implemented in Week 7.

**Efficient Market Hypothesis (EMH)** — theory that all available information is already priced into stock prices, making it impossible to consistently beat the market. In practice, large-cap liquid stocks are close to efficient — which is why predicting 1-day direction is nearly impossible and why we switched to 5-day relative rank prediction.

**Hypertable** — TimescaleDB's table structure that auto-partitions data by time, making date-range queries extremely fast.

**Survivorship Bias** — only testing on companies that still exist today skews results upward. Our fixed 29-stock universe has mild survivorship bias — worth acknowledging when presenting results.

**Coverage Rate** — in the context of sentiment data, the percentage of stock-days that have at least one news article. 67% is good. 2.7% (our first attempt) is too sparse to be useful as a feature.

---

## File Structure Reference

```
AlphaForge/
├── data/
│   ├── ingest.py                               ← pulls OHLCV from Alpaca, stores in TimescaleDB
│   ├── validate.py                             ← checks data quality
│   ├── raw/
│   └── processed/
│       ├── features_daily.parquet              ← 43,123 rows, 22 features, index: [time, symbol]
│       ├── regime_labels.parquet               ← daily regime label (bull/choppy/bear)
│       ├── backtest_momentum.parquet           ← momentum strategy daily P&L
│       ├── backtest_mean_reversion.parquet     ← mean reversion daily P&L
│       ├── backtest_regime_switcher.parquet    ← regime switcher daily P&L (best so far)
│       ├── backtest_ml.parquet                 ← ML strategy daily P&L
│       ├── ml_signals.parquet                  ← XGBoost probability scores per stock per day
│       ├── news_raw.parquet                    ← raw articles from Alpaca News API (cached)
│       ├── news_scored.parquet                 ← articles with FinBERT sentiment scores (cached)
│       └── sentiment_daily.parquet             ← daily sentiment features per stock
├── research/
│   ├── features/
│   │   └── engineer.py                        ← computes all 22 features
│   ├── strategies/
│   │   ├── momentum.py                        ← cross-sectional momentum, long-only
│   │   ├── mean_reversion.py                  ← RSI + Bollinger Band, long-only
│   │   ├── backtest.py                        ← simulation engine + performance metrics
│   │   ├── regime_switcher.py                 ← HMM-driven strategy selection
│   │   └── ml_backtest.py                     ← backtests ML probability signals
│   └── notebooks/
├── models/
│   ├── regime_hmm.py                          ← trains HMM, labels regimes, saves model
│   ├── hmm_model.pkl                          ← trained HMM + scaler (pickle)
│   ├── ml_signal.py                           ← XGBoost walk-forward training + signal generation
│   ├── xgb_model.pkl                          ← final trained XGBoost model (pickle)
│   └── sentiment.py                           ← Alpaca news fetch + FinBERT scoring pipeline
├── risk/                                      ← Week 7
├── execution/                                 ← Week 8
├── dashboard/                                 ← Week 9
├── tests/
├── config/
├── .env                                       ← API keys + DB credentials (NEVER commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Current Dependencies

```
alpaca-py        ← market data, news API, and trading API
pandas           ← data manipulation
numpy            ← numerical computing
sqlalchemy       ← database connection
psycopg2         ← PostgreSQL driver
python-dotenv    ← loads .env file
pytz             ← timezone handling
pyarrow          ← parquet file format
hmmlearn         ← Hidden Markov Model implementation
scikit-learn     ← StandardScaler, metrics (accuracy, AUC)
xgboost          ← gradient boosting ML model
transformers     ← HuggingFace library for FinBERT
torch            ← PyTorch backend for FinBERT
```

---

*Last updated: End of Week 6 — Regime Detection (HMM), ML Signal Generation (XGBoost), and Sentiment Pipeline (FinBERT) complete. Next: integrate sentiment features into ML model, then Week 7 Risk & Portfolio Layer.*