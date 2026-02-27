import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ── 1. LOAD & PREPARE FEATURES ───────────────────────────────────────────────

def load_market_features(path="data/processed/features_daily.parquet"):
    """
    We need market-level (not stock-level) features for regime detection.
    The HMM should describe what the overall MARKET is doing each day,
    not what any individual stock is doing.

    Strategy: aggregate across all 29 stocks per day.
    """
    df = pd.read_parquet(path)
    
    df = df.reset_index()
    
    df["date"] = pd.to_datetime(df["time"]).dt.date

    daily = df.groupby("date").agg(
        mean_return     = ("return_1d",   "mean"),   # avg return across universe
        vol_return      = ("return_1d",   "std"),    # cross-sectional dispersion
        mean_volatility = ("volatility_20d", "mean"),# avg rolling volatility
        mean_rsi        = ("rsi_14",      "mean"),   # avg RSI — market overbought?
        mean_macd_hist  = ("macd_hist",   "mean"),   # avg MACD momentum
        mean_bb_pct     = ("bb_pct",      "mean"),   # where in Bollinger Band range
        mean_volume_ratio = ("volume_ratio", "mean") # unusual volume activity
    ).dropna()

    daily.index = pd.to_datetime(daily.index)
    
    return daily

# ── 2. TRAIN HMM ─────────────────────────────────────────────────────────────

def train_hmm(features_df, n_states=3, n_iter=200, random_state=42):
    """
    Train a Gaussian Hidden Markov Model.

    Why Gaussian HMM?
    - 'Hidden' because we can't directly observe regimes — only their effects
    - 'Gaussian' because each hidden state emits observations drawn from a
      normal distribution. Each regime has its own mean and variance of features.
    - n_states=3: we expect bull (trending up), bear (trending down), choppy

    The model learns:
    - Transition matrix: probability of moving from one regime to another
    - Emission params: what returns/volatility look like in each regime
    """
    feature_cols = [
        "mean_return", "vol_return", "mean_volatility",
        "mean_rsi", "mean_macd_hist", "mean_bb_pct", "mean_volume_ratio"
    ]

    X = features_df[feature_cols].values

    # Clip outliers at 1st and 99th percentile before scaling
    # Without this, one extreme day (e.g. COVID bounce March 2020) 
    # consumes an entire HMM state on its own
    lower = np.percentile(X, 1, axis=0)
    upper = np.percentile(X, 99, axis=0)
    X = np.clip(X, lower, upper)

    # Standardise — HMM is sensitive to feature scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",   # each state has its own full covariance matrix
        n_iter=n_iter,
        random_state=random_state
    )
    model.fit(X_scaled)

    return model, scaler, feature_cols

# ── 3. LABEL REGIMES ─────────────────────────────────────────────────────────

def label_regimes(features_df, model, scaler, feature_cols):
    """
    Predict which regime each trading day belongs to.
    Then map the raw HMM state numbers (0, 1, 2) to meaningful labels
    by looking at the average return in each state.
    """
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)

    raw_states = model.predict(X_scaled)
    features_df = features_df.copy()
    features_df["raw_state"] = raw_states

    # Label by volatility — much more stable than return
    # Lowest volatility → bull, highest → bear, middle → choppy
    state_vols = (
        features_df.groupby("raw_state")["mean_volatility"].mean().sort_values()
    )
    label_map = {
        state_vols.index[0]: "bull",
        state_vols.index[1]: "choppy",
        state_vols.index[2]: "bear",
    }
    features_df["regime"] = features_df["raw_state"].map(label_map)

    return features_df, label_map

# ── 4. ANALYSE REGIMES ───────────────────────────────────────────────────────

def analyse_regimes(regime_df):
    """
    Print a breakdown of each regime's characteristics.
    This is how we validate the HMM actually learned meaningful states.
    """
    print("\n========== REGIME ANALYSIS ==========")
    print(f"Total trading days: {len(regime_df)}")
    print(f"Date range: {regime_df.index.min().date()} → {regime_df.index.max().date()}\n")

    for regime in ["bull", "bear", "choppy"]:
        subset = regime_df[regime_df["regime"] == regime]
        pct = len(subset) / len(regime_df) * 100
        print(f"  {regime.upper()} regime: {len(subset)} days ({pct:.1f}%)")
        print(f"    Avg daily return:    {subset['mean_return'].mean():.4%}")
        print(f"    Avg volatility:      {subset['mean_volatility'].mean():.4f}")
        print(f"    Avg RSI:             {subset['mean_rsi'].mean():.1f}")
        print(f"    Avg MACD hist:       {subset['mean_macd_hist'].mean():.4f}")
        print()

    # Transition matrix — how sticky are regimes?
    print("  Transition probabilities (how likely to stay in same regime):")
    for regime in ["bull", "bear", "choppy"]:
        days_in = regime_df[regime_df["regime"] == regime]
        if len(days_in) > 1:
            # Proportion of days where next day is same regime
            next_regime = regime_df["regime"].shift(-1)
            same = (regime_df["regime"] == regime) & (next_regime == regime)
            stay_prob = same.sum() / (regime_df["regime"] == regime).sum()
            print(f"    {regime.upper()}: {stay_prob:.1%} chance of staying")
    print("=====================================\n")

# ── 5. SAVE OUTPUTS ──────────────────────────────────────────────────────────

def save_outputs(regime_df, model, scaler, label_map):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save regime labels — backtest and ML model will use this
    regime_df[["mean_return", "mean_volatility", "mean_rsi",
               "mean_macd_hist", "regime"]].to_parquet(
        "data/processed/regime_labels.parquet"
    )

    # Save trained model for inference in later weeks
    with open("models/hmm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "label_map": label_map}, f)

    print("Saved: data/processed/regime_labels.parquet")
    print("Saved: models/hmm_model.pkl")

# ── 6. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading market features...")
    features = load_market_features()
    print(f"Loaded {len(features)} trading days\n")

    print("Training Hidden Markov Model (3 states)...")
    model, scaler, feature_cols = train_hmm(features)
    print(f"HMM trained. Log-likelihood: {model.score(scaler.transform(features[feature_cols].values)):.2f}\n")

    print("Labelling regimes...")
    regime_df, label_map = label_regimes(features, model, scaler, feature_cols)
    print(f"State → label mapping: {label_map}\n")

    analyse_regimes(regime_df)
    save_outputs(regime_df, model, scaler, label_map)

    print("Done. Next step: run regime_switcher.py to see strategy performance by regime.")