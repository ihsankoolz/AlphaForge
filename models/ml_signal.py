import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD & PREPARE DATA ───────────────────────────────────────────────────

def load_data():
    """Pulls in features parquet and regime labels, merges them together,
        then create the target variable by shifting next day's return onto today's 
        row.  So each row ends up as "here are today's conditions, did the stock go up tomorrow?"""
    features = pd.read_parquet("data/processed/features_daily.parquet")
    regimes  = pd.read_parquet("data/processed/regime_labels.parquet")

    features = features.reset_index()
    features["date"] = pd.to_datetime(features["time"]).dt.normalize().dt.tz_localize(None)

    regimes = regimes.copy()
    regimes.index = pd.to_datetime(regimes.index).tz_localize(None).normalize()
    regimes = regimes.reset_index().rename(columns={"index": "date"})

    # Merge regime label onto features
    df = features.merge(regimes[["date", "regime"]], on="date", how="left")

    # Encode regime as integer — XGBoost needs numbers not strings
    regime_map = {"bull": 0, "choppy": 1, "bear": 2}
    df["regime_encoded"] = df["regime"].map(regime_map).fillna(0)

    # ── TARGET VARIABLE ──────────────────────────────────────────────────────
    # We want to predict tomorrow's return using today's features
    # So we shift return_1d back by 1 within each symbol group
    df = df.sort_values(["symbol", "date"])
    df["target_return"] = df.groupby("symbol")["return_1d"].shift(-1)
    df["target"] = (df["target_return"] > 0).astype(int)  # 1 = up, 0 = down

    # Drop rows where target is unknown (last day of each symbol)
    df = df.dropna(subset=["target_return"])

    return df

# ── 2. DEFINE FEATURES ───────────────────────────────────────────────────────

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_pct", "bb_bandwidth",
    "volume_ratio", "spy_correlation",
    "regime_encoded"
]

# ── 3. WALK-FORWARD VALIDATION ───────────────────────────────────────────────

def walk_forward_validation(df, initial_train_months=12, retrain_every_months=3):
    """
    Simulates how the model would perform in live trading.

    Each quarter:
    1. Train on all data up to today
    2. Predict the next quarter
    3. Move forward and repeat

    This gives us honest out-of-sample predictions for every single day,
    which we can then use as signals in the backtest.

    Why expanding window (not rolling)?
    More historical data generally helps tree models. We never throw away
    old data — we just keep adding to the training set.
    """
    df = df.sort_values("date")
    all_dates = sorted(df["date"].unique())

    start_date = all_dates[0]
    initial_cutoff = all_dates[0] + pd.DateOffset(months=initial_train_months)
    retrain_delta  = pd.DateOffset(months=retrain_every_months)

    all_predictions = []
    model_scores    = []

    current_cutoff = initial_cutoff
    fold = 1

    while current_cutoff < pd.Timestamp(all_dates[-1]):
        next_cutoff = current_cutoff + retrain_delta

        train = df[df["date"] <  current_cutoff]
        test  = df[(df["date"] >= current_cutoff) & (df["date"] < next_cutoff)]

        if len(train) < 500 or len(test) < 50:
            current_cutoff = next_cutoff
            continue

        X_train = train[FEATURE_COLS].fillna(0)
        y_train = train["target"]
        X_test  = test[FEATURE_COLS].fillna(0)
        y_test  = test["target"]

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,           # shallow trees reduce overfitting
            learning_rate=0.05,
            subsample=0.8,         # use 80% of rows per tree
            colsample_bytree=0.8,  # use 80% of features per tree
            min_child_weight=20,   # require 20+ samples per leaf
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        # Predict probability of going up (class 1)
        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_class = (pred_proba > 0.5).astype(int)

        acc  = accuracy_score(y_test, pred_class)
        try:
            auc = roc_auc_score(y_test, pred_proba)
        except:
            auc = 0.5

        print(f"  Fold {fold:2d} | Train: {train['date'].min().date()} → {train['date'].max().date()} "
              f"| Test: {test['date'].min().date()} → {test['date'].max().date()} "
              f"| Acc: {acc:.3f} | AUC: {auc:.3f}")

        model_scores.append({"fold": fold, "accuracy": acc, "auc": auc,
                              "test_start": current_cutoff, "n_train": len(train)})

        # Store predictions with metadata
        test_preds = test[["date", "symbol", "target_return", "target"]].copy()
        test_preds["pred_proba"] = pred_proba
        test_preds["pred_class"] = pred_class
        all_predictions.append(test_preds)

        # Save last model (used for live inference later)
        if next_cutoff >= pd.Timestamp(all_dates[-1]):
            os.makedirs("models", exist_ok=True)
            with open("models/xgb_model.pkl", "wb") as f:
                pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)
            print(f"\n  → Final model saved to models/xgb_model.pkl")

        current_cutoff = next_cutoff
        fold += 1

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    scores_df      = pd.DataFrame(model_scores)

    return predictions_df, scores_df

# ── 4. CONVERT PREDICTIONS TO SIGNALS ────────────────────────────────────────

def predictions_to_signals(predictions_df):
    """
    Convert ML predicted probabilities into trading signals.

    The predicted probability of going up (pred_proba) becomes our signal:
    - pred_proba > 0.55 → buy signal (strength = how far above 0.55)
    - pred_proba < 0.45 → sell signal (we'll ignore this, long-only)
    - Between 0.45-0.55 → no signal (model isn't confident enough)

    This is better than the rule-based approach because:
    - Signal strength is continuous, not binary
    - It incorporates ALL 22 features simultaneously
    - It learned the relationships from data rather than hand-coded rules
    """
    df = predictions_df.copy()

    # Signal = how confident the model is minus the neutral threshold
    df["signal"] = 0.0
    buy_mask = df["pred_proba"] > 0.55
    df.loc[buy_mask, "signal"] = (df.loc[buy_mask, "pred_proba"] - 0.55) / 0.45

    return df

# ── 5. EVALUATE SIGNAL QUALITY ───────────────────────────────────────────────

def evaluate_signals(predictions_df):
    """
    Same diagnostic we ran on rule-based signals in Week 3.
    Does a higher predicted probability actually mean better next-day returns?
    """
    df = predictions_df.copy()

    # Bucket predictions into quintiles
    df["prob_bucket"] = pd.qcut(df["pred_proba"], q=5,
                                labels=["Very Low", "Low", "Mid", "High", "Very High"])

    print("\n========== ML SIGNAL QUALITY ==========")
    print("Avg next-day return by predicted probability bucket:")
    print(f"{'Bucket':<12} {'Avg Return':>12} {'% Correct':>12} {'Count':>8}")
    print("-" * 46)

    bucket_stats = df.groupby("prob_bucket", observed=True).agg(
        avg_return=("target_return", "mean"),
        pct_correct=("target", "mean"),
        count=("target", "count")
    )
    for bucket, row in bucket_stats.iterrows():
        print(f"  {str(bucket):<10} {row['avg_return']:>11.4%} {row['pct_correct']:>11.2%} {row['count']:>8}")

    print("-" * 46)
    print(f"\nOverall accuracy: {(df['pred_class'] == df['target']).mean():.3f}")
    print(f"Baseline (always predict up): {df['target'].mean():.3f}")
    print("========================================\n")

# ── 6. FEATURE IMPORTANCE ────────────────────────────────────────────────────

def print_feature_importance(predictions_df, df_full):
    """
    Retrain on full dataset just to inspect which features matter most.
    This is for insight only — not used in the actual backtest.
    """
    X = df_full[FEATURE_COLS].fillna(0)
    y = df_full["target"]

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        eval_metric="logloss", random_state=42, verbosity=0
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=False)

    print("\n========== FEATURE IMPORTANCE ==========")
    for feat, score in importance.items():
        bar = "█" * int(score * 200)
        print(f"  {feat:<20} {score:.4f}  {bar}")
    print("=========================================\n")

# ── 7. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['date'].min().date()} → {df['date'].max().date()}\n")

    print("Running walk-forward validation (this takes ~1-2 minutes)...")
    predictions, scores = walk_forward_validation(df)

    print(f"\nWalk-forward complete: {len(predictions):,} out-of-sample predictions")
    print(f"Mean Accuracy: {scores['accuracy'].mean():.3f}")
    print(f"Mean AUC:      {scores['auc'].mean():.3f}")

    evaluate_signals(predictions)
    print_feature_importance(predictions, df)

    # Convert to signals and save
    signals = predictions_to_signals(predictions)
    signals.to_parquet("data/processed/ml_signals.parquet")
    print("Saved: data/processed/ml_signals.parquet")
    print("\nNext step: run ml_backtest.py to compare ML signals vs rule-based strategies")