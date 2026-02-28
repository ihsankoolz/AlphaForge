import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.strategies.momentum import generate_signals as momentum_signals
from research.strategies.mean_reversion import generate_signals as mr_signals

# ── 1. LOAD & PREPARE DATA ───────────────────────────────────────────────────

def load_data():
    features = pd.read_parquet("data/processed/features_daily.parquet")
    regimes  = pd.read_parquet("data/processed/regime_labels.parquet")

    features = features.reset_index()
    features["date"] = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None)

    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None).normalize()
    regimes = regimes.reset_index().rename(columns={"index": "date"})

    # ── ADD RULE-BASED SIGNALS AS FEATURES ───────────────────────────────────
    print("  Generating rule-based signals to use as ML features...")
    features_raw = pd.read_parquet("data/processed/features_daily.parquet")

    mom_sig = momentum_signals(features_raw).reset_index()[["date", "symbol", "signal"]].rename(columns={"signal": "momentum_signal"})
    mr_sig  = mr_signals(features_raw).reset_index()[["date", "symbol", "signal"]].rename(columns={"signal": "mr_signal"})

    mom_sig["date"] = pd.to_datetime(mom_sig["date"]).dt.normalize().dt.tz_localize(None)
    mr_sig["date"]  = pd.to_datetime(mr_sig["date"]).dt.normalize().dt.tz_localize(None)

    df = features.merge(regimes[["date", "regime"]], on="date", how="left")
    df = df.merge(mom_sig, on=["date", "symbol"], how="left")
    df = df.merge(mr_sig,  on=["date", "symbol"], how="left")

    df["momentum_signal"] = df["momentum_signal"].fillna(0)
    df["mr_signal"]       = df["mr_signal"].fillna(0)

    # Encode regime
    regime_map = {"bull": 0, "choppy": 1, "bear": 2}
    df["regime_encoded"] = df["regime"].map(regime_map).fillna(0)

    # ── ADD SENTIMENT FEATURES ────────────────────────────────────────────────
    print("  Loading sentiment features...")
    sentiment = pd.read_parquet("data/processed/sentiment_daily.parquet")

    # Normalise the sentiment date index to tz-naive dates so the merge aligns
    # with features_daily regardless of how sentiment.py stored timestamps
    sentiment = sentiment.reset_index()
    date_col = "date" if "date" in sentiment.columns else sentiment.columns[0]
    sentiment[date_col] = pd.to_datetime(sentiment[date_col]).dt.normalize().dt.tz_localize(None)
    sentiment = sentiment.rename(columns={date_col: "date"})

    # Keep only the columns we want as features (drop extras like article_ids etc.)
    sentiment_feature_cols = [
        "sentiment_mean", "sentiment_std",
        "sentiment_pos_pct", "sentiment_neg_pct",
        "article_count", "sentiment_3d"
    ]
    # Only keep columns that actually exist in the file
    sentiment_feature_cols = [c for c in sentiment_feature_cols if c in sentiment.columns]
    sentiment = sentiment[["date", "symbol"] + sentiment_feature_cols]

    df = df.merge(sentiment, on=["date", "symbol"], how="left")

    # Days with no news coverage → neutral sentiment, zero counts
    # This is intentional: absence of news is information in itself
    sentiment_fill = {
        "sentiment_mean":    0.0,   # neutral
        "sentiment_std":     0.0,   # no disagreement
        "sentiment_pos_pct": 0.0,
        "sentiment_neg_pct": 0.0,
        "article_count":     0.0,   # no coverage
        "sentiment_3d":      0.0,   # neutral rolling average
    }
    for col, fill_val in sentiment_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    coverage = df["sentiment_mean"].astype(bool).mean() if "sentiment_mean" in df.columns else 0
    print(f"  Sentiment coverage: {coverage:.1%} of stock-days have news data")

    # ── TARGET: 5-DAY FORWARD RELATIVE RANK ──────────────────────────────────
    df = df.sort_values(["symbol", "date"])
    df["forward_return_5d"] = df.groupby("symbol")["return_1d"].transform(
        lambda x: x.shift(-1).rolling(5).sum()
    )
    df["forward_rank"] = df.groupby("date")["forward_return_5d"].rank(pct=True)
    df["target"] = (df["forward_rank"] > 0.75).astype(int)
    df = df.dropna(subset=["forward_return_5d", "forward_rank"])

    return df

# ── 2. FEATURE COLUMNS ───────────────────────────────────────────────────────

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
    # Rule-based signals — domain knowledge baked in
    "momentum_signal", "mr_signal",
    # Sentiment features (Week 6) — FinBERT scores from Alpaca News
    # Missing days filled with 0 (neutral) — 33% of stock-days have no coverage
    "sentiment_mean",    # avg FinBERT score that day (-1 to +1)
    "sentiment_std",     # disagreement between articles — high = uncertainty
    "sentiment_pos_pct", # % of articles classified positive
    "sentiment_neg_pct", # % of articles classified negative
    "article_count",     # news volume — high count = more attention
    "sentiment_3d",      # 3-day rolling avg sentiment — smooths single-day noise
]

# ── 3. WALK-FORWARD VALIDATION ───────────────────────────────────────────────

def walk_forward_validation(df, initial_train_months=18, retrain_every_months=3):
    """
    18-month initial window instead of 12 — gives the model more patterns
    to learn from before its first real prediction, especially important
    since we changed to a 5-day forward target which has fewer signal events.
    """
    # Only keep feature cols that actually exist in df (guards against
    # sentiment columns being absent if sentiment file is missing)
    active_features = [c for c in FEATURE_COLS if c in df.columns]
    if len(active_features) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(active_features)
        print(f"  Warning: {len(missing)} feature(s) not found and will be skipped: {missing}")

    df = df.sort_values("date")
    all_dates = sorted(df["date"].unique())

    initial_cutoff = all_dates[0] + pd.DateOffset(months=initial_train_months)
    retrain_delta  = pd.DateOffset(months=retrain_every_months)

    all_predictions = []
    model_scores    = []
    current_cutoff  = initial_cutoff
    fold = 1

    while current_cutoff < pd.Timestamp(all_dates[-1]):
        next_cutoff = current_cutoff + retrain_delta

        train = df[df["date"] <  current_cutoff]
        test  = df[(df["date"] >= current_cutoff) & (df["date"] < next_cutoff)]

        if len(train) < 500 or len(test) < 50:
            current_cutoff = next_cutoff
            continue

        X_train = train[active_features].fillna(0)
        y_train = train["target"]
        X_test  = test[active_features].fillna(0)
        y_test  = test["target"]

        # Class imbalance — top quartile is 25% of data so we
        # tell XGBoost to upweight the positive class 3x
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,       # slower learning = better generalisation
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=30,      # even more conservative leaf splits
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_class = (pred_proba > 0.5).astype(int)

        acc = accuracy_score(y_test, pred_class)
        try:
            auc = roc_auc_score(y_test, pred_proba)
        except:
            auc = 0.5

        print(f"  Fold {fold:2d} | Train: {train['date'].min().date()} → "
              f"{train['date'].max().date()} "
              f"| Test: {test['date'].min().date()} → {test['date'].max().date()} "
              f"| Acc: {acc:.3f} | AUC: {auc:.3f}")

        model_scores.append({
            "fold": fold, "accuracy": acc, "auc": auc,
            "test_start": current_cutoff, "n_train": len(train)
        })

        test_preds = test[["date", "symbol", "forward_return_5d",
                           "forward_rank", "target"]].copy()
        test_preds["pred_proba"] = pred_proba
        test_preds["pred_class"] = pred_class
        all_predictions.append(test_preds)

        if next_cutoff >= pd.Timestamp(all_dates[-1]):
            os.makedirs("models", exist_ok=True)
            with open("models/xgb_model.pkl", "wb") as f:
                pickle.dump({"model": model, "feature_cols": active_features}, f)
            print(f"\n  → Final model saved to models/xgb_model.pkl")

        current_cutoff = next_cutoff
        fold += 1

    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(model_scores)

# ── 4. EVALUATE SIGNAL QUALITY ───────────────────────────────────────────────

def evaluate_signals(predictions_df):
    """
    Key question: does higher predicted probability actually correspond
    to stocks that perform better over the next 5 days?
    We want a clean monotonic pattern — each bucket better than the last.
    """
    df = predictions_df.copy()
    df["prob_bucket"] = pd.qcut(df["pred_proba"], q=5,
                                labels=["Very Low", "Low", "Mid", "High", "Very High"])

    print("\n========== ML SIGNAL QUALITY (v3 — with sentiment) ==========")
    print("Avg 5-day forward return by predicted probability bucket:")
    print(f"  {'Bucket':<12} {'Avg 5d Return':>14} {'Top Quartile %':>15} {'Count':>8}")
    print("  " + "-" * 52)

    bucket_stats = df.groupby("prob_bucket", observed=True).agg(
        avg_return  = ("forward_return_5d", "mean"),
        pct_top_q   = ("target",            "mean"),
        count       = ("target",            "count")
    )
    for bucket, row in bucket_stats.iterrows():
        print(f"  {str(bucket):<12} {row['avg_return']:>13.4%} "
              f"{row['pct_top_q']:>14.2%} {row['count']:>8}")

    print("  " + "-" * 52)
    print(f"\n  Overall AUC target: >0.83 (beat v2 baseline)")
    print(f"  Baseline top-quartile rate: {df['target'].mean():.3f} (should be ~0.25)")
    print("==============================================================\n")

# ── 5. FEATURE IMPORTANCE ────────────────────────────────────────────────────

def print_feature_importance(df):
    active_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[active_features].fillna(0)
    y = df["target"]

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=30,
        eval_metric="auc", random_state=42, verbosity=0
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=active_features)
    importance = importance.sort_values(ascending=False)

    print("\n========== FEATURE IMPORTANCE (v3 — with sentiment) ==========")
    for feat, score in importance.items():
        bar = "█" * int(score * 300)
        print(f"  {feat:<22} {score:.4f}  {bar}")

    # Highlight sentiment contribution
    sentiment_cols = ["sentiment_mean", "sentiment_std", "sentiment_pos_pct",
                      "sentiment_neg_pct", "article_count", "sentiment_3d"]
    sentiment_importance = sum(importance.get(c, 0) for c in sentiment_cols)
    print(f"\n  → Sentiment features combined importance: {sentiment_importance:.4f} "
          f"({sentiment_importance:.1%} of model)")
    print("===============================================================\n")

# ── 6. CONVERT TO SIGNALS ────────────────────────────────────────────────────

def predictions_to_signals(predictions_df):
    """
    High predicted probability = high confidence this stock will be a
    top quartile performer = strong buy signal.
    Threshold at 0.35 (not 0.5) because top quartile only occurs 25%
    of the time — the model's probabilities will naturally cluster lower.
    """
    df = predictions_df.copy()
    df["signal"] = 0.0
    buy_mask = df["pred_proba"] > 0.35
    df.loc[buy_mask, "signal"] = (df.loc[buy_mask, "pred_proba"] - 0.35) / 0.65
    return df

# ── 7. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data and generating features (including sentiment)...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Target rate (top quartile): {df['target'].mean():.3f} "
          f"(expected ~0.25)\n")
    print(f"  Total features: {len([c for c in FEATURE_COLS if c in df.columns])} "
          f"({len([c for c in FEATURE_COLS if c in df.columns]) - 17} sentiment features added)\n")

    print("Running walk-forward validation (v3 — with sentiment)...")
    predictions, scores = walk_forward_validation(df)

    print(f"\nWalk-forward complete: {len(predictions):,} out-of-sample predictions")
    print(f"Mean Accuracy: {scores['accuracy'].mean():.3f}")
    print(f"Mean AUC:      {scores['auc'].mean():.3f}  (v2 baseline: 0.829)")

    evaluate_signals(predictions)
    print_feature_importance(df)

    signals = predictions_to_signals(predictions)
    signals.to_parquet("data/processed/ml_signals.parquet")
    print("Saved: data/processed/ml_signals.parquet")
    print("\nNext step: run ml_backtest.py to trade using v3 ML signals")
    print("Then: begin Week 7 — Risk & Portfolio Layer")