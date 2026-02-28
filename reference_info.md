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
| 6 | Sentiment Layer as Alternative Data | ✅ Done |
| 6.5 | Sentiment Integration into ML Model (v3) | ✅ Done |
| 7 | Risk & Portfolio Layer (Kelly + Markowitz) | ✅ Done |
| 8 | Execution Layer & Paper Trading | 🔄 In Progress |
| 8.5 | Universe Expansion (200+ stocks) | ⏳ Upcoming |
| 9 | Dashboard & Visualization | ⏳ Upcoming |
| 10 | Polish, Documentation & Write-Up | ⏳ Upcoming |

**Note on universe expansion:** Deliberately deferred until after the execution layer is working. Reason: get the risk layer proven on 29 stocks first, then expansion is just plugging more data into an already-tested system. Will slot between Week 8 and Week 9.

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
               │                 ◄── sentiment features (v3)
               ▼
     [data/processed/ml_signals.parquet]
               │
               ▼
     [research/strategies/ml_backtest.py]
               │
               ▼
     [risk/position_sizer.py] ← Kelly Criterion + vol adjustment
               │
               ▼
     [risk/portfolio_optimiser.py] ← Markowitz mean-variance
               │
               ▼
     [execution/order_manager.py] ← Alpaca paper trading orders
               │
               ▼
     [execution/run_daily.py] ← daily pipeline orchestrator
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
Rolling standard deviation of daily returns. `volatility_20d = 0.025` means the stock moves ±2.5% per day on average. Used by HMM for regime detection and by Week 7 position sizing.

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
- Plan: Revisit short selling in Week 8.5 when universe expands to 200+ stocks

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

| Strategy | Total Return | Ann. Return | Sharpe | Max Drawdown | Period | Notes |
|----------|-------------|-------------|--------|--------------|--------|-------|
| Momentum (standalone) | +91.85% | +11.67% | 0.56 | -30.24% | 2020-2025 | Long-only, 29 stocks |
| Mean Reversion (standalone) | +8.71% | +1.42% | 0.19 | -45.16% | 2020-2025 | Good signal, too infrequent |
| Regime Switcher | +186.65% | +19.54% | 0.96 | -23.25% | 2020-2025 | Best full-period performer |
| ML Signal v2 (XGBoost only) | +24.87% | +5.20% | 0.38 | -28.62% | 2021-2025 | Naive signal-weighting |
| ML + Risk Layer v3 | +22.52% | +4.74% | 0.49 | -15.54% | 2021-2025 | Kelly + Markowitz |
| SPY Benchmark | ~55-60% | ~12% | ~0.70 | ~-34% | 2021-2025 | Buy and hold comparison |

**Critical note on comparisons:** The regime switcher covers 2020-2025 including the COVID crash recovery (the best 18-month period in recent market history). The ML strategies only cover Aug 2021-2025 — a much harder period that started right before the 2022 rate hike selloff. Comparing 186% to 22% across different time periods is unfair. Over the SAME 2021-2025 period, the regime switcher's advantage shrinks substantially.

**What the Risk Layer actually achieved (v2 → v3 comparison, same period):**
- Sharpe: 0.38 → 0.49 (+29% improvement)
- Max Drawdown: -28.62% → -15.54% (cut nearly in half)
- Total Return: 24.87% → 22.52% (slightly lower — expected, Kelly is conservative by design)

The risk layer's job is not to maximize return — it's to maximize risk-adjusted return. Halving the drawdown while improving Sharpe by 29% is exactly correct behavior.

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

### Why ML Backtest Underperforms Regime Switcher

The standalone ML backtest shows +38% vs regime switcher's +186%. This is NOT because ML is worse. Three reasons:
1. ML signals only available from August 2021 (18-month training warmup), missing the best 2020-2021 bull run
2. Fewer positions per day (avg 5.4 vs 10) — model is more selective
3. The comparison is unfair — different time periods

**The correct role of the ML model:** It is not a standalone strategy. It is the brain that all other components plug into. In Week 7, ML probability scores become inputs to the portfolio optimizer.

### Key Gotchas
- Signal threshold for top-quartile prediction should be 0.35 not 0.5 (probabilities cluster below 0.5 when positive class is rare)

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

### Files Produced
- `data/processed/news_raw.parquet` — raw article text with dates and symbols
- `data/processed/news_scored.parquet` — articles with FinBERT labels and scores
- `data/processed/sentiment_daily.parquet` — daily aggregated sentiment per stock

---

## Week 6.5 — Sentiment Integration into ML Model (v3)

### What We Did
Added 6 sentiment features to `models/ml_signal.py` as additional XGBoost inputs. Retrained via walk-forward validation. This is called v3 of the ML model.

### How Sentiment Was Merged
- Loaded `sentiment_daily.parquet` and normalized dates to tz-naive (same as features_daily)
- Left-merged on `[date, symbol]` — keeps all feature rows, adds sentiment where available
- Missing days (no news coverage) filled with 0.0 — neutral sentiment, zero article count
- This fill is intentional: absence of news is information in itself (no coverage = neutral signal)

### v3 Results

**Walk-forward AUC by fold:**
All 18 folds between 0.753 and 0.859. Mean AUC: 0.829 (identical to v2 baseline).

**Signal quality table (v3):**
| Bucket | Avg 5d Return | Top Quartile % |
|--------|-------------|----------------|
| Very Low | -3.53% | 2.72% |
| Low | -1.55% | 11.34% |
| Mid | +0.11% | 19.68% |
| High | +1.72% | 36.40% |
| Very High | +4.42% | 67.80% |

Perfectly monotonic — model still clearly separates winners from losers.

**Feature importance (v3):**
| Feature | Importance |
|---------|------------|
| return_5d | 29.2% |
| bb_pct | 17.6% |
| return_10d | 7.1% |
| ... | ... |
| article_count | 1.6% |
| sentiment_std | 1.3% |
| sentiment_pos_pct | 1.3% |
| sentiment_3d | 1.2% |
| sentiment_mean | 1.1% |
| sentiment_neg_pct | 0.8% |
| **Sentiment total** | **7.3%** |

### How to Interpret the Sentiment Results

**AUC didn't improve (0.829 → 0.829). Does that mean sentiment failed?**

No. The correct interpretation: sentiment added genuine information (7.3% combined importance) but didn't move the AUC needle because price features already captured most of what sentiment knows. For large-cap blue-chip stocks, news gets priced in almost immediately — by the time FinBERT reads the headline, `return_5d` has already absorbed the market's reaction. Sentiment is a slower, noisier version of a signal the price data already contains more cleanly.

**Key insight from feature importance ranking:**
- `article_count` (1.6%) was the MOST important sentiment feature — more than `sentiment_mean` (1.1%)
- High news volume is a mild contrarian indicator: stocks getting a lot of attention tend to mean-revert rather than continue momentum
- `sentiment_std` (1.3%) ranked above `sentiment_mean` — disagreement between articles matters more than their average direction. High uncertainty in the news → risk signal.
- `sentiment_mean` at the bottom confirms the "news already priced in" story for large-caps

**Why 7.3% combined importance is not nothing:**
Sentiment collectively contributes more than `macd_hist` (1.4%), `bb_bandwidth` (1.5%), or `return_20d` (1.25%). It's a legitimate feature that belongs in the model.

**Why sentiment will matter more after universe expansion:**
The 29-stock universe is the most efficiently priced set of companies on earth — thousands of analysts cover them. Mid and small caps have 2-3 analysts, news doesn't get priced in within minutes. After expanding to 200+ stocks, sentiment edge will grow substantially. The 7.3% importance is a floor, not a ceiling.

**Coverage note:** Actual measured coverage was 49.3% (not the expected 67%). This is because many legitimately neutral days score exactly 0.0 after FinBERT averaging, which gets counted as "no coverage." The article fetch was correct — it's a measurement artifact.

### What to Say About Sentiment in Interviews/Presentations
"Sentiment contributes 7.3% feature importance but provides diminishing marginal value over price signals in a blue-chip universe, consistent with the Efficient Market Hypothesis for large-caps. We expect this to increase significantly after universe expansion to mid/small-cap stocks where news is less efficiently priced."

---

## Week 7 — Risk & Portfolio Layer

### Overview
Week 7 adds two files that sit between the ML signals and the backtest:
- `risk/position_sizer.py` — Kelly Criterion + volatility adjustment (individual position sizing)
- `risk/portfolio_optimiser.py` — Markowitz mean-variance optimization (portfolio-level construction)

These answer different questions:
- **Kelly:** "How much capital should I risk on each individual stock given my edge?"
- **Markowitz:** "Given how all these stocks move relative to each other, what's the optimal combination?"

---

## Week 7 — Position Sizing: Kelly Criterion

### What is Position Sizing?
Position sizing answers: you have $100,000 and 5 stocks to buy — how much goes in each? The signal tells you WHICH stocks to buy. Position sizing tells you HOW MUCH. It's arguably more important than signal generation — great signal with bad sizing can still blow up a portfolio.

### Position Sizing Methods — All Alternatives

| Method | How it works | Best for |
|--------|-------------|----------|
| Equal weight | Split capital evenly | Simple strategies, low signal confidence |
| Signal weight | Weight ∝ signal strength | What ML v2 backtest used |
| Kelly Criterion | Optimal bet size given edge and odds | When you have reliable probability estimates |
| Volatility parity | Each position contributes equal risk, not capital | Risk-conscious portfolios, standard at quant funds |
| Mean-variance (Markowitz) | Optimize weights to maximize Sharpe as portfolio | When correlations between positions matter |
| Max drawdown targeting | Size so worst-case stays within loss limit | Very risk-averse mandates |

**Why we chose Kelly + Markowitz together:** Kelly handles individual sizing, Markowitz handles correlations between positions. Each solves a different problem. Together they're more powerful than either alone.

### Kelly Criterion Explained

Kelly answers: given that my model has a certain edge, what fraction of capital should I bet?

```
Kelly fraction = (p × b - q) / b

where:
  p = probability of winning (our pred_proba)
  q = probability of losing (1 - p)
  b = odds = avg_win / avg_loss
```

**What are "odds" in trading?** If your winners average +3% and losers average -1.5%, odds = 2.0. You win $2 for every $1 risked.

**Why half-Kelly (multiply by 0.5)?** Raw Kelly assumes your probability estimates are perfect. Our pred_proba of 0.70 isn't exactly 70% likely — it's the model's best guess. Half-Kelly cuts position sizes in half, sacrificing some theoretical return for much better drawdown protection. Standard practice at real quant funds.

**Volatility adjustment (on top of Kelly):**
```
final_weight = kelly_weight × (target_vol / stock_volatility)
```
If NVDA has 3% daily vol and our target is 1%, NVDA's position is cut by ⅔. Same signal confidence, less capital at risk. This ensures every position contributes roughly equal risk regardless of how volatile the underlying stock is.

### Constants Used
```python
KELLY_FRACTION = 0.5    # half-Kelly safety buffer
MIN_PROB       = 0.35   # below this = no position
MAX_POSITION   = 0.15   # hard cap at 15% per stock
VOL_LOOKBACK   = 20     # days to estimate volatility
VOL_TARGET     = 0.01   # target 1% daily vol per position
```

### Position Sizer Diagnostic Results

```
Avg positions per day:    9.2
Avg capital invested:     88.4%
Avg cash held:            11.6%
Avg largest position:     12.0%
Avg position size:        9.6%
Days at max positions:    871 (79.0%)
```

**How to interpret:**
- 88.4% invested means the model almost always finds high-confidence trades
- 79% of days hit 10-position maximum — normalization dominates Kelly on busy days, positions converge toward equal weight
- 11.6% cash is the model's implicit uncertainty signal — Kelly holding back capital says "don't bet big today"
- 12% avg largest position — 15% cap is working, nothing dominates dangerously

**Sample day (2023-05-08) showing Kelly working clearly:**
- Only 8 positions (not 10) — model selective
- 23% cash held
- AAPL (15%) vs MCD (4.6%) — vol adjustment differentiating positions with similar signals

---

## Week 7 — Portfolio Optimization: Markowitz Mean-Variance

### The Problem Kelly Ignores

On 2024-03-22, Kelly gave GS, BAC, JPM, and META roughly equal weights. But GS, BAC, and JPM are all large-cap US financials that move almost identically. When the Fed makes an announcement, all three drop together. You haven't diversified — you've taken the same financial sector bet three times in different jerseys.

Markowitz's 1952 insight: don't optimize stocks individually, optimize the portfolio as a whole. The question isn't "is this stock good?" but "does adding this stock make the whole portfolio better?"

### The Two Ingredients

**Expected returns** — we use ML probability scores as proxy. Higher pred_proba = higher expected return. Linear mapping: `expected_return = (pred_proba - 0.5) × 0.40`

**Covariance matrix** — how do stocks move relative to each other? Built from 60 days of historical returns (before the current date — no lookahead bias). Diagonal = each stock's own variance. Off-diagonal = co-movement between pairs.

Why 60 days? Too short (20 days) = noisy unstable correlations. Too long (252 days) = stale, doesn't reflect current regime. 60 days captures current structure while having enough data.

### What the Optimizer Does

Searches through every possible weight combination and finds the one maximizing the Sharpe ratio. Solved via scipy `minimize` with SLSQP method (Sequential Least Squares Programming — standard for constrained portfolio optimization).

Constraints:
- Weights sum to 1.0 (fully invested within selected stocks)
- Each weight between MIN_WEIGHT (2%) and MAX_WEIGHT (15%)
- Long-only (no negative weights)

### The Efficient Frontier

```
Return
  ↑
  |                    ● Maximum Return portfolio
  |                 ●
  |              ●  ← Maximum Sharpe (what we target)
  |           ●
  |        ●
  |     ● Minimum Variance portfolio
  |
  └──────────────────────────→ Risk (volatility)
```

Every point on the curve is optimal — you can't do better without taking more risk. We target maximum Sharpe.

### How Kelly and Markowitz Work Together

```
Kelly → "here are the stocks worth holding and how confident I am in each"
           ↓
Markowitz → "given how these stocks correlate, here's the optimal split"
           ↓
final_weight = markowitz_proportion × kelly_total
```

Kelly determines the TOTAL capital to invest. Markowitz determines HOW to split it. Rescaling at the end preserves Kelly's implicit cash signal.

### Markowitz Diagnostic Results

**Average absolute weight change (Kelly → Markowitz): 7.21%**

This means Markowitz is meaningfully redistributing capital — not rubber-stamping Kelly weights. It's actively restructuring based on correlations.

**Key observations from sample days:**

2025-02-07 (BULL): BAC and GS both slashed to 2% while JPM bumped to 15%. All three are large-cap US financials. Markowitz said: "JPM alone gives you financial sector exposure, BAC and GS are redundant risk." Meanwhile COST, META, SBUX, JPM went to 15% — genuinely uncorrelated businesses that diversify well.

2024-03-22 (BULL): Kelly gave everyone exactly 10% (normalization washed out differences). Markowitz restructured significantly — rewarded genuine diversifiers, penalized correlated clusters.

2022-06-22 (BEAR): Total invested dropped to 70.4% — optimizer held more cash automatically through tighter bear regime caps. Correct risk behavior.

**The 2% minimum floor:** Several stocks get pushed to the minimum. This means Markowitz wanted to remove them entirely but the `MIN_WEIGHT = 0.02` constraint forced a minimum holding. In a real production system you might let the optimizer zero these out completely.

### Regime Behavior in the Optimizer

- **Bull:** Full Kelly + Markowitz, standard 15% caps
- **Bear:** Kelly + Markowitz with tighter 70% position caps (model less reliable in crashes)
- **Choppy:** Pure cash — handled BEFORE calling the optimizer in `ml_backtest.py`

**Why pure cash in choppy (not SPY rotation):**
SPY rotation was initially coded and tested. It destroyed returns — -40.89% in the choppy regime, -2.44 Sharpe, bringing overall portfolio to -30.14% total. Root cause: the 2022 rate hike selloff hit during many choppy days, and 80% of capital in SPY during a 20% drawdown is catastrophic. The regime switcher's 186% came from pure cash in choppy. Reverted immediately.

**Lesson:** Empirical testing overrides theoretical elegance. Pure cash in uncertainty is the correct rule.

### Constants Used
```python
COV_LOOKBACK   = 60     # days of history for covariance matrix
MIN_WEIGHT     = 0.02   # minimum position — no slivers below 2%
MAX_WEIGHT     = 0.15   # inherits from position sizer
RISK_FREE_RATE = 0.05/252  # daily risk-free (~5% annual, 2021-2025 avg)
MIN_STOCKS     = 2      # need at least 2 stocks to optimize correlations
```

### Known Limitation of Markowitz
Covariance matrix is estimated from historical returns — assumes future correlation structure looks like the past. In a crisis, correlations spike (everything moves together). This is a known weakness. More sophisticated approaches (Black-Litterman, robust optimization) address this but are out of scope for AlphaForge.

### Files Added
- `risk/__init__.py` — empty file making risk/ a Python module
- `risk/position_sizer.py` — Kelly + vol adjustment
- `risk/portfolio_optimiser.py` — Markowitz Sharpe maximization

---

## Week 7 — Final Backtest Results

After fixing two bugs (SPY rotation in choppy, date range starting before ML signal period):

**v3 Performance by Regime (Aug 2021 - Dec 2025):**
| Regime | Days | Total Return | Sharpe |
|--------|------|-------------|--------|
| BULL | 659 | +16.62% | 0.56 |
| BEAR | 174 | +5.05% | 0.57 |
| CHOPPY | 272 | 0.00% | 0.00 |

**Overall: +22.52% total, 0.49 Sharpe, -15.54% max drawdown**

**What the bug journey taught us:**

Bug 1 — SPY rotation: Coded, tested, produced -30.14% total return. Root cause: SPY lost 20% in 2022 during choppy regime days. Fix: pure cash.

Bug 2 — Wrong date range: Backtest was running from Jan 2020 even though ML signals only start Aug 2021. This added 383 dead days where no signals existed, artificially weighting the denominator. Fix: filter `common_dates` to start from `signal_start`.

**The honest assessment:**
- Max drawdown cut from -28.62% to -15.54% — real investors would stay in this strategy
- Sharpe improved 29% — better risk-adjusted return
- Raw return slightly lower — expected, Kelly is conservative by design
- The risk layer does its job correctly

---

## Week 8 — Execution Layer & Paper Trading

### What Changes in Week 8

Everything before Week 8 runs historically — feed it 5 years of past data, it tells you what would have happened. Week 8 makes the system run live — every day, automatically, with real paper orders.

### What "Paper Trading" Means

Alpaca offers a paper trading account — simulated brokerage with $100,000 fake money executing against real live market prices. No real money at risk, mechanics identical to a live account. Industry standard way to validate a strategy before going live.

### AlphaForge Trading Style — Daily Rebalancing (NOT Day Trading)

**Day trading** = opening and closing positions within the same day, profiting from intraday price moves. Requires tick-level data, extremely fast execution, very hard to profit from.

**AlphaForge** = daily rebalancing system. Every morning, look at current portfolio, model generates new target weights based on yesterday's closing prices, rebalance to targets. You might hold AAPL for weeks if the model keeps liking it. The 5-day forward return target means thinking in week-long horizons, not minutes. Profits compound over months.

### How Long Does run_daily.py Run?

It runs once per day and exits — it is NOT a while loop. Flow:
```
9:25 AM → run_daily.py starts
9:30 AM → market opens, orders placed
9:35 AM → script finishes and exits
```

Scheduled via Windows Task Scheduler to run automatically each weekday morning. The 3% daily loss limit doesn't keep the script alive — it prevents orders from being placed that day, then the script exits normally.

### The Four Components of Week 8

**1. `execution/order_manager.py`** — the core new piece. Translates target portfolio weights into actual Alpaca orders. Handles: connection, market hours check, daily loss limit, computing buy/sell deltas, placing orders.

**2. `execution/run_daily.py`** — orchestrates everything in sequence each morning: fetch yesterday's prices → recompute features → run HMM → run ML model → run risk layer → place orders.

**3. Safeguards (hard limits):**
- Max position size: 15% (inherits from risk layer)
- Daily loss limit: halt all trading if portfolio down >3% today
- Market hours: only place orders 9:30–15:55 ET (soft close 5 min before end)
- Min order value: skip orders below $50
- Dry run mode: log orders without actually placing them

**4. Logging** — every order, every decision, every error written to `logs/trading_YYYYMMDD.log`. When something goes wrong in a live system (and it will), the log is how you debug it.

### Key Design Decisions in Order Manager

**Target vs Current (the core concept):**
```
Target:  AAPL 12%, MSFT 8%, NVDA 0%
Current: AAPL 10%, MSFT 8%, NVDA 5%
Action:  Buy 2% AAPL, Hold MSFT, Sell all NVDA
```
Only trade the DELTA. Minimizes transaction costs and unnecessary turnover.

**Sells before buys:** Order list sorted so sells execute first. Selling NVDA frees up cash before buying AAPL — prevents "insufficient funds" failures.

**Notional orders (dollars, not shares):** "Buy $1,200 of AAPL" not "buy 5 shares." Much easier to work with portfolio percentages. Alpaca supports fractional shares via notional orders.

**1% minimum weight change threshold:** If optimizer shifts a position from 10.2% to 10.6%, ignore it. Transaction cost would exceed the benefit.

**`dry_run=True` by default:** System will never place a real order unless you explicitly set `dry_run=False`. Critical safeguard during development.

### Order Manager Test Results

Running `python execution/order_manager.py` on a Saturday produced:
```
Connected to Alpaca (PAPER) account  ← SUCCESS
Market is closed — no orders will be placed  ← CORRECT
```

This is expected and correct behavior. The Alpaca connection worked. Market hours check worked. Test properly on a weekday between 9:30-3:55 PM ET (Singapore time: 9:30 PM - 3:55 AM next day).

### Files Added
- `execution/__init__.py` — empty module marker
- `execution/order_manager.py` — full execution pipeline

---

## Questions Asked This Session & Answers

### "Is Kelly Criterion the only and best way to handle position sizing?"

No. See the position sizing methods table above. Kelly is theoretically optimal when your probability estimates are accurate, but it's aggressive. Real quant funds use fractional Kelly (half-Kelly) or combine Kelly with volatility parity. We use half-Kelly + volatility adjustment because: (1) our probability estimates aren't perfect, (2) volatility adjustment ensures equal risk contribution per position regardless of stock volatility.

### "What is 'odds' in Kelly Criterion?"

Odds = average winning trade / average losing trade. If winners average +3% and losers average -1.5%, odds = 2.0. Combined with win rate, Kelly tells you the optimal bet fraction. We use b=2.0 as a prior calibrated from the signal quality table (Very High bucket +4.42% vs Very Low -3.53% ≈ 3:1 ratio, conservatively estimated at 2:1).

### "Why is the ML strategy so weak compared to the regime switcher?"

The comparison is unfair — different time periods. The regime switcher covers 2020-2025 including the COVID bull run (best 18 months in recent market history). ML only covers Aug 2021-2025, starting right before the 2022 rate hike selloff where SPY dropped 20%. Over the SAME period, the regime switcher's advantage shrinks substantially. The key improvement is max drawdown: -15.54% vs -28.62%, which is what the risk layer was designed to achieve.

### "Is this a day trading system?"

No. AlphaForge is a daily rebalancing system that thinks in week-long horizons (5-day forward return target). Day trading operates on seconds/milliseconds and requires completely different infrastructure. See the Day Trading section below.

### "Should I add universe expansion now or later?"

Later. Rationale: get the risk layer proven on 29 stocks first, then expansion is just plugging more data into an already-tested system. Scheduled between Week 8 and Week 9.

### "Should choppy regime rotate into SPY or hold pure cash?"

We tested SPY rotation and it produced -40.89% in the choppy regime due to 2022 rate hike selloff exposure. Reverted to pure cash. The regime switcher's 186% was built on pure cash in choppy — empirically proven to be the right rule.

---

## Day Trading vs AlphaForge — Key Differences

This is a completely different project with a different technical stack. Summary:

**Why Python isn't used for real day trading:**
Python is too slow for anything under the minute timeframe. Professional firms measure execution speed in microseconds. C++ compiles directly to machine code with no interpreter overhead — a well-written C++ trading system executes an order in under 10 microseconds. Python operates in milliseconds — 100x slower.

**What day trading additionally requires:**
- Direct market data feed (ITCH, OPRA) delivering tick-by-tick order book data — millions of messages per second
- Co-location — physically placing servers inside exchange data centers to minimize network round-trip latency
- Order book modeling — modeling the limit order book in real time (bid-ask spread, queue depth, iceberg orders)
- Market microstructure knowledge — how matching engines prioritize orders, price-time priority, pro-rata matching, order types (IOC, FOK, pegged)
- Pre-trade risk systems in C++ — hard limits executing before every single order
- FIX protocol — industry standard messaging format
- Low-latency C++ — lock-free data structures, SIMD instructions, kernel bypass networking (DPDK), CPU affinity

**Technical concepts needed beyond AlphaForge:**
- Market Microstructure Theory (O'Hara) — standard text
- Trading and Exchanges (Larry Harris) — more practical
- Order flow imbalance as predictive signal
- Statistical arbitrage at high frequency — pairs trading, ETF arbitrage at millisecond level
- TCP vs UDP, multicast, kernel bypass networking
- Co-location and network latency optimization

**Realistic career path:**
Finish AlphaForge → quant internship (Python research) → learn C++ in parallel → specialize into execution/microstructure in 2nd or 3rd role. Very few people jump straight into HFT infrastructure — typically 3-5 year progression.

---

## Key Finance & Technical Concepts

**Momentum Premium** — stocks that performed well over the past 3-12 months tend to continue performing well. Academic finding since 1993 (Jegadeesh and Titman), one of the most robust patterns in finance. Our XGBoost confirmed this — return_5d is the single most important feature at 35%.

**Mean Reversion** — extreme price moves tend to snap back toward average. Fear and greed cause prices to overshoot fair value. RSI and Bollinger Bands measure this overshoot.

**Cross-Sectional vs Time-Series Signals**
- Cross-sectional: compare stocks against each other (momentum — rank all 29 stocks by recent return)
- Time-series: compare a stock against its own history (mean reversion — is AAPL oversold vs its own recent prices?)

**Universe Composition Bias** — a strategy must be designed for its universe. Shorting "losers" in a universe of blue-chip winners is structurally broken because even the worst large-cap stock tends to go up over time.

**Lookahead Bias** — using future information in a backtest. Causes unrealistically good results. Fixed by shifting signals one day forward. This caused the 146,600% fantasy return early on.

**Spread** — the difference between buy price and sell price. Our 0.1% transaction cost models this plus broker commission.

**Walk-Forward Validation** — train on all data up to today, predict tomorrow, move forward and repeat. The honest way to validate ML models on time-series data.

**Market Regime** — markets behave differently in different environments: trending bull, trending bear, high-volatility/choppy.

**Hidden Markov Model (HMM)** — statistical model where states (bull/bear/choppy) are unobservable but can be inferred from observable data (returns, volatility).

**Transition Matrix** — probability of moving from one regime to another. Our bear regime has 97.2% probability of staying bear — crashes are persistent.

**Winsorization** — clipping extreme values at defined percentiles before feeding into a model. Prevents outlier days from dominating training.

**Fat Tails** — financial return distributions have more extreme events than a normal distribution predicts. Why winsorization is necessary.

**Alternative Data** — any data beyond price/volume: news sentiment, Reddit posts, satellite imagery, credit card transactions.

**FinBERT** — BERT fine-tuned on financial text. Classifies text as positive/negative/neutral with confidence score.

**Sharpe Ratio** — (avg return − risk-free rate) / volatility, annualised. Return per unit of risk. Above 1.0 decent, above 1.5 good, above 2.0 excellent.

**Sortino Ratio** — like Sharpe but only counts downside volatility as risk. Big upward swings aren't risk — you only care about drops.

**Max Drawdown** — worst peak-to-trough decline. The number that determines whether real investors would stay invested or panic.

**Win Rate** — percentage of trading days the portfolio made money. Can be misleading — a strategy can have low win rate but still be profitable if wins are larger than losses.

**Signal-Weighted Position Sizing** — allocate capital proportional to signal strength. What ML v2 used before being replaced by Kelly + Markowitz.

**AUC (Area Under the ROC Curve)** — measures how well a classifier ranks predictions. 0.5 = random, 1.0 = perfect. Above 0.54 is meaningful edge in finance. Our XGBoost achieved 0.83.

**Scale_pos_weight** — XGBoost parameter upweighting the minority class. Top quartile occurs 25% of time → set scale_pos_weight ≈ 3.

**Kelly Criterion** — optimal position sizing given edge and odds. f = (p×b - q) / b. We use half-Kelly (multiply by 0.5) for safety.

**Half-Kelly** — multiply raw Kelly fraction by 0.5. Halves position size, sacrifices some theoretical return for much better drawdown protection. Standard at real quant funds.

**Volatility Parity / Vol Targeting** — size each position so it contributes equal risk to the portfolio. `weight = kelly_weight × (target_vol / stock_vol)`. High-vol stocks get smaller positions for the same signal.

**Mean-Variance Optimization (Markowitz)** — find portfolio weights maximizing return for a given risk level (or equivalently maximizing Sharpe). Uses expected returns + covariance matrix as inputs.

**Covariance Matrix** — square matrix showing how each pair of stocks moves together. Diagonal = own variance. Off-diagonal = co-movement. Core input to Markowitz.

**Efficient Frontier** — the curve of optimal portfolios where you can't improve return without increasing risk. We target the maximum Sharpe point on this curve.

**SLSQP (Sequential Least Squares Programming)** — optimization algorithm used by scipy to solve the constrained Sharpe maximization problem. Standard choice for portfolio optimization.

**Notional Orders** — orders denominated in dollar amount rather than share quantity. "Buy $1,200 of AAPL" instead of "buy 5 shares." Easier to work with portfolio percentages.

**Rebalancing** — adjusting current portfolio positions toward target weights. Only trading the delta (difference) between current and target — minimizes transaction costs and turnover.

**Daily Loss Limit** — safeguard that halts all trading if the portfolio drops more than X% in a single day. Prevents a buggy signal from causing a runaway loss spiral.

**Paper Trading** — simulated trading with fake money against real live market prices. Industry standard validation step before risking real capital.

**EMH (Efficient Market Hypothesis)** — theory that all available information is already priced in. Large-caps are close to efficient — which is why predicting 1-day direction is nearly impossible and why we predict 5-day relative rank instead.

**Hypertable** — TimescaleDB's table structure that auto-partitions data by time for fast date-range queries.

**Survivorship Bias** — only testing on companies that still exist skews results upward. Our 29-stock universe has mild survivorship bias worth acknowledging.

**Coverage Rate** — percentage of stock-days with at least one news article. 67% is good. 2.7% (first attempt) is too sparse to be useful.

**FIX Protocol** — Financial Information eXchange. Industry standard messaging format for order management used by all brokers and exchanges.

**Co-location** — physically placing trading servers inside exchange data centers to minimize network latency. Used by high-frequency trading firms.

**Order Book** — list of all outstanding buy and sell orders for a stock at different price levels. Day trading strategies model this in real time.

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

**Alpaca News API:**
- `symbols` parameter must be a comma-separated string, not a list
- Response is a `NewsSet` object — data at `news.data['news']`, NOT `news.news`
- Articles are `News` objects — use `getattr(article, 'headline')` NOT `article.get('headline')`
- Rate limit: add `time.sleep(0.3)` between requests

**HMM training:**
- Always winsorize features at 1st/99th percentile before training
- Label regimes by volatility (stable) not by mean return (fragile to outliers)

**XGBoost for financial data:**
- `scale_pos_weight` is critical when target class is rare
- `min_child_weight=30` prevents overfitting on small leaf nodes
- `max_depth=4` keeps trees shallow — financial data has weak signals, deep trees overfit

**ML signal threshold:**
- Use 0.35 not 0.5 — probabilities cluster below 0.5 when positive class is rare (25% of data)

**sys.path for multi-level imports:**
- `ml_backtest.py` lives in `research/strategies/` and needs BOTH:
  - `dirname × 3` → project root (for `risk/` module)
  - `dirname × 2` → `research/` (for `strategies/` module)
- Always add both when a file needs to import from multiple levels

**`__init__.py` files required:**
- Every folder you import from as a module needs an empty `__init__.py`
- Created for `risk/` and `execution/` during Week 7/8

**Never `pip install` a package with the same name as your local module:**
- `pip install risk` installs a random unrelated PyPI package
- Use `__init__.py` + `sys.path` to make local modules importable

**ML backtest date range:**
- Always filter `common_dates` to start from `signals["date"].min()` (Aug 2021)
- Running from 2020 adds 383 dead days with no signals, distorting metrics

**Portfolio optimiser — choppy regime:**
- Handle choppy (pure cash) in `ml_backtest.py` BEFORE calling the optimiser
- The optimiser should only ever receive bull or bear regime calls
- SPY rotation in choppy was tested and destroyed returns due to 2022 selloff

**Alpaca order manager:**
- Market is closed on weekends — order_manager.py will halt immediately on Saturday/Sunday
- This is correct behavior, not a bug
- Test properly on a weekday 9:30 AM - 3:55 PM ET (9:30 PM - 3:55 AM SGT)

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
│       ├── backtest_regime_switcher.parquet    ← regime switcher daily P&L
│       ├── backtest_ml.parquet                 ← ML + Risk Layer v3 daily P&L
│       ├── ml_signals.parquet                  ← XGBoost v3 probability scores per stock per day
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
│   │   └── ml_backtest.py                     ← ML + Risk Layer backtest (v3)
│   └── notebooks/
├── models/
│   ├── regime_hmm.py                          ← trains HMM, labels regimes, saves model
│   ├── hmm_model.pkl                          ← trained HMM + scaler (pickle)
│   ├── ml_signal.py                           ← XGBoost v3 (with sentiment), walk-forward
│   ├── xgb_model.pkl                          ← final trained XGBoost v3 model (pickle)
│   └── sentiment.py                           ← Alpaca news fetch + FinBERT scoring pipeline
├── risk/
│   ├── __init__.py                            ← makes risk/ importable as a module
│   ├── position_sizer.py                      ← Kelly Criterion + volatility adjustment
│   └── portfolio_optimiser.py                 ← Markowitz mean-variance Sharpe maximization
├── execution/
│   ├── __init__.py                            ← makes execution/ importable as a module
│   ├── order_manager.py                       ← Alpaca paper trading order execution
│   └── run_daily.py                           ← daily pipeline orchestrator (Week 8, pending)
├── dashboard/                                 ← Week 9
├── logs/
│   └── trading_YYYYMMDD.log                   ← daily execution logs
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
scipy            ← portfolio optimization (SLSQP solver) ← NEW Week 7
zoneinfo         ← timezone handling for market hours check ← NEW Week 8
```

---

## What's Coming Next

### Immediate Next Step — run_daily.py (Week 8 completion)
Build the daily pipeline orchestrator that wires all existing components together into one automated morning script. Test with order_manager.py in dry_run=True mode on a weekday during market hours. Then switch to dry_run=False for live paper trading.

### Week 8.5 — Universe Expansion
Expand from 29 → 200+ stocks. Steps: update `data/ingest.py` to fetch new tickers, re-run `engineer.py` for feature computation, retrain ML model (walk-forward will take longer), update regime HMM on expanded universe. This unlocks viable short selling and makes sentiment features more powerful.

### Week 9 — Streamlit Dashboard
Live P&L curve, current positions, strategy performance by regime, sentiment signals, risk metrics updating daily.

### Week 10 — Polish & Write-Up
Clean README, demo video, docstrings throughout, Medium article. Best story: the regime detection value-add (186% → how pure cash during choppy doubled returns). Second story: sentiment's role in efficient vs inefficient markets.

---

## Areas for Improvement

1. **Universe expansion** — the biggest single lever. Mid/small cap stocks have less efficient pricing, making sentiment and ML signals much more powerful. Also enables viable short selling.

2. **Covariance estimation** — the 60-day lookback assumes stable correlations. In crises, correlations spike (everything moves together). Black-Litterman or robust optimization would handle this better.

3. **Kelly odds calibration** — we hardcoded b=2.0 based on the signal quality table. In production, this should be estimated from rolling historical data and updated quarterly.

4. **Sentiment coverage** — 49% effective coverage. After universe expansion to stocks with fewer analysts, coverage will improve and sentiment will show more edge.

5. **Transaction cost model** — our 0.1% flat cost is a simplification. Real costs depend on order size, volatility, and time of day. A more realistic model would improve backtest accuracy.

6. **Short selling** — currently long-only due to universe composition bias. After expanding to 200+ stocks including genuine losers, short selling becomes viable and should improve Sharpe.

7. **Turnover constraint** — currently no explicit limit on how much the portfolio changes day-to-day. High turnover = high transaction costs. Adding a turnover penalty to the Markowitz objective would reduce costs.

---

*Last updated: End of Week 7 / Start of Week 8 — Sentiment integration (v3 ML model, 0.829 AUC, 7.3% sentiment importance), Risk Layer complete (Kelly + Markowitz, Sharpe 0.49, max drawdown -15.54%), Order Manager built and connection tested. Next: run_daily.py to complete Week 8, then universe expansion and dashboard.*