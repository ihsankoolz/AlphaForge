"""
analytics/signal_quality.py
============================
Monitor ML model signal quality over time.

Detects model drift, tracks prediction accuracy, and triggers
recalibration alerts — so you know when to retrain before the
quarterly schedule.

Provides:
    - Rolling AUC on non-overlapping windows
    - Prediction accuracy by probability bucket
    - Signal distribution shift detection (KL divergence)
    - Recalibration trigger (AUC below threshold)
    - Feature importance drift (cosine similarity between epochs)

All functions are pure — they accept arrays/DataFrames and return
numbers or dicts.  No I/O, no side effects.

Usage:
    from analytics.signal_quality import (
        rolling_auc, bucket_accuracy, signal_distribution_shift,
        needs_recalibration, feature_importance_drift,
        signal_quality_report,
    )
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling AUC (non-overlapping windows)
# ---------------------------------------------------------------------------

def rolling_auc(y_true, y_score, window=60, step=5):
    """
    Compute AUC on rolling non-overlapping windows.

    Every ``step`` days, compute AUC using the most recent ``window``
    days of predictions.  The ``step`` parameter avoids temporal
    correlation between adjacent 5-day forward return targets.

    Parameters
    ----------
    y_true : array-like
        Binary labels (1 = top quartile return, 0 = not).
    y_score : array-like
        Predicted probabilities from the model.
    window : int
        Number of days in each evaluation window.
    step : int
        Skip between evaluation points (default 5 for non-overlapping
        5-day targets).

    Returns
    -------
    list[dict]
        Each dict has keys ``end_idx`` (int) and ``auc`` (float).
        Returns empty list if inputs are too short.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Remove paired NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]

    n = len(y_true)
    if n < window:
        return []

    results = []
    for end in range(window, n + 1, step):
        start = end - window
        yt = y_true[start:end]
        ys = y_score[start:end]

        # AUC needs both classes present
        if len(np.unique(yt)) < 2:
            continue

        auc = _manual_auc(yt, ys)
        results.append({"end_idx": end, "auc": auc})

    return results


def _manual_auc(y_true, y_score):
    """
    Compute AUC-ROC without sklearn dependency.

    Uses the Wilcoxon-Mann-Whitney statistic:
    AUC = P(score(positive) > score(negative)).
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Count how many positive scores beat negative scores
    count = 0
    ties = 0
    for p in pos:
        count += np.sum(neg < p)
        ties += np.sum(neg == p)
    auc = (count + 0.5 * ties) / (len(pos) * len(neg))
    return float(auc)


# ---------------------------------------------------------------------------
# Bucket accuracy
# ---------------------------------------------------------------------------

def bucket_accuracy(y_true, y_score, n_buckets=5):
    """
    Split predictions into buckets and measure accuracy per bucket.

    Useful for checking calibration: do stocks predicted at 0.70 actually
    win ~70% of the time?  And do low-probability stocks actually
    underperform?

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_score : array-like
        Predicted probabilities.
    n_buckets : int
        Number of equal-frequency buckets.

    Returns
    -------
    list[dict]
        Each dict: ``bucket`` (str), ``count`` (int),
        ``avg_pred`` (float), ``actual_win_rate`` (float).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true, y_score = y_true[mask], y_score[mask]

    if len(y_true) == 0:
        return []

    # Sort by score
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    bucket_size = max(1, len(y_true) // n_buckets)
    results = []

    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < n_buckets - 1 else len(y_true)
        if start >= len(y_true):
            break

        bt = y_true[start:end]
        bs = y_score[start:end]

        results.append({
            "bucket": f"Q{i+1}",
            "count": len(bt),
            "avg_pred": float(np.mean(bs)),
            "actual_win_rate": float(np.mean(bt)),
        })

    return results


# ---------------------------------------------------------------------------
# Signal distribution shift (KL divergence)
# ---------------------------------------------------------------------------

def signal_distribution_shift(scores_old, scores_new, n_bins=20):
    """
    Detect distribution shift between two sets of model predictions
    using KL divergence.

    A large KL divergence means the model's output distribution has
    changed — either the market changed or the model is drifting.

    Parameters
    ----------
    scores_old : array-like
        Predicted probabilities from an earlier period.
    scores_new : array-like
        Predicted probabilities from the current period.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        KL divergence (bits).  0 means identical distributions.
        Returns 0.0 if either input is empty.
    """
    old = np.asarray(scores_old, dtype=float)
    new = np.asarray(scores_new, dtype=float)
    old = old[~np.isnan(old)]
    new = new[~np.isnan(new)]

    if len(old) == 0 or len(new) == 0:
        return 0.0

    # Common bin edges
    lo = min(old.min(), new.min())
    hi = max(old.max(), new.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)

    hist_old, _ = np.histogram(old, bins=edges, density=True)
    hist_new, _ = np.histogram(new, bins=edges, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_old = hist_old + eps
    hist_new = hist_new + eps

    # Normalise to proper probability distributions
    hist_old = hist_old / hist_old.sum()
    hist_new = hist_new / hist_new.sum()

    # KL(new || old) — how much new diverges from old
    kl = float(np.sum(hist_new * np.log(hist_new / hist_old)))
    return max(0.0, kl)


# ---------------------------------------------------------------------------
# Recalibration trigger
# ---------------------------------------------------------------------------

def needs_recalibration(auc_values, threshold=0.55, min_periods=3):
    """
    Check if the model needs retraining based on recent AUC.

    Parameters
    ----------
    auc_values : list[float]
        Recent rolling AUC values (most recent last).
    threshold : float
        If the average of the last ``min_periods`` AUC values falls
        below this, trigger recalibration.
    min_periods : int
        Number of recent AUC values to average.

    Returns
    -------
    dict
        ``trigger`` (bool), ``recent_avg_auc`` (float),
        ``threshold`` (float).
    """
    auc_values = [v for v in auc_values if not np.isnan(v)]
    if len(auc_values) < min_periods:
        return {
            "trigger": False,
            "recent_avg_auc": None,
            "threshold": threshold,
            "reason": "insufficient_data",
        }

    recent = auc_values[-min_periods:]
    avg = float(np.mean(recent))

    return {
        "trigger": avg < threshold,
        "recent_avg_auc": avg,
        "threshold": threshold,
        "reason": "auc_below_threshold" if avg < threshold else "ok",
    }


# ---------------------------------------------------------------------------
# Feature importance drift
# ---------------------------------------------------------------------------

def feature_importance_drift(importance_old, importance_new):
    """
    Measure how much feature importance has shifted between two training
    epochs using cosine similarity.

    Parameters
    ----------
    importance_old : dict
        Feature name → importance score from previous training.
    importance_new : dict
        Feature name → importance score from current training.

    Returns
    -------
    dict
        ``cosine_similarity`` (float, 1.0 = identical ranking),
        ``top_changes`` (list of features that changed most).
    """
    if not importance_old or not importance_new:
        return {"cosine_similarity": 1.0, "top_changes": []}

    # Union of all features
    all_features = sorted(set(importance_old) | set(importance_new))

    vec_old = np.array([importance_old.get(f, 0.0) for f in all_features])
    vec_new = np.array([importance_new.get(f, 0.0) for f in all_features])

    # Cosine similarity
    dot = np.dot(vec_old, vec_new)
    norm_old = np.linalg.norm(vec_old)
    norm_new = np.linalg.norm(vec_new)

    if norm_old == 0 or norm_new == 0:
        return {"cosine_similarity": 0.0, "top_changes": []}

    cos_sim = float(dot / (norm_old * norm_new))

    # Top changes: absolute difference in normalised importance
    if vec_old.sum() > 0:
        vec_old = vec_old / vec_old.sum()
    if vec_new.sum() > 0:
        vec_new = vec_new / vec_new.sum()

    diffs = [(all_features[i], abs(vec_new[i] - vec_old[i]))
             for i in range(len(all_features))]
    diffs.sort(key=lambda x: x[1], reverse=True)

    top_changes = [{"feature": f, "abs_change": float(d)}
                   for f, d in diffs[:5]]

    return {
        "cosine_similarity": cos_sim,
        "top_changes": top_changes,
    }


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------

def signal_quality_report(y_true, y_score, scores_old=None,
                          importance_old=None, importance_new=None,
                          auc_threshold=0.55):
    """
    Generate a comprehensive signal quality report.

    Parameters
    ----------
    y_true : array-like
        Binary labels for the evaluation period.
    y_score : array-like
        Predicted probabilities for the evaluation period.
    scores_old : array-like, optional
        Predictions from a prior period (for distribution shift).
    importance_old : dict, optional
        Prior feature importance (for drift detection).
    importance_new : dict, optional
        Current feature importance (for drift detection).
    auc_threshold : float
        Recalibration trigger threshold.

    Returns
    -------
    dict
        Full report with rolling AUC, bucket accuracy, shift, drift,
        and recalibration status.
    """
    report = {}

    # Rolling AUC
    auc_series = rolling_auc(y_true, y_score, window=60, step=5)
    report["rolling_auc"] = auc_series

    # Overall AUC
    y_t = np.asarray(y_true, dtype=float)
    y_s = np.asarray(y_score, dtype=float)
    mask = ~(np.isnan(y_t) | np.isnan(y_s))
    y_t, y_s = y_t[mask], y_s[mask]
    if len(np.unique(y_t)) >= 2:
        report["overall_auc"] = _manual_auc(y_t, y_s)
    else:
        report["overall_auc"] = None

    # Bucket accuracy
    report["bucket_accuracy"] = bucket_accuracy(y_true, y_score)

    # Recalibration check
    auc_vals = [r["auc"] for r in auc_series]
    report["recalibration"] = needs_recalibration(auc_vals, auc_threshold)

    # Distribution shift
    if scores_old is not None:
        report["distribution_shift_kl"] = signal_distribution_shift(
            scores_old, y_score)
    else:
        report["distribution_shift_kl"] = None

    # Feature importance drift
    if importance_old is not None and importance_new is not None:
        report["feature_drift"] = feature_importance_drift(
            importance_old, importance_new)
    else:
        report["feature_drift"] = None

    return report
