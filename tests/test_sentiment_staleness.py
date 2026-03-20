"""
tests/test_sentiment_staleness.py
=================================
Unit tests for sentiment staleness degradation (time-decay weighting).

Tests that:
  - Recent articles get higher weight than old ones
  - Articles beyond max age get zero weight
  - Decay-weighted mean differs from unweighted mean
  - Edge cases: no timestamps, NaN values, future articles
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from models.sentiment import _compute_staleness_weight, aggregate_daily_sentiment
from config.settings import SENTIMENT_DECAY_HALFLIFE_HOURS, SENTIMENT_MAX_AGE_HOURS


class TestComputeStalenessWeight:
    """Tests for the per-article decay weight function."""

    def test_fresh_article_full_weight(self):
        """Article published at market close should have weight ~1.0."""
        trade_date = pd.Timestamp("2024-01-10")
        created_at = pd.Timestamp("2024-01-10 15:30:00")  # 30 min before close
        w = _compute_staleness_weight(created_at, trade_date)
        assert w > 0.95

    def test_old_article_low_weight(self):
        """Article from 36 hours ago should have weight ~0.25."""
        trade_date = pd.Timestamp("2024-01-10")
        # 36 hours before 4 PM = previous day 4 AM
        created_at = pd.Timestamp("2024-01-09 04:00:00")
        w = _compute_staleness_weight(created_at, trade_date)
        assert w < 0.5

    def test_beyond_max_age_zero_weight(self):
        """Article older than SENTIMENT_MAX_AGE_HOURS gets weight 0."""
        trade_date = pd.Timestamp("2024-01-10")
        # 5 days ago — well beyond 72h max
        created_at = pd.Timestamp("2024-01-05 10:00:00")
        w = _compute_staleness_weight(created_at, trade_date)
        assert w == 0.0

    def test_future_article_full_weight(self):
        """Article from the future should get full weight (edge case)."""
        trade_date = pd.Timestamp("2024-01-10")
        created_at = pd.Timestamp("2024-01-11 09:00:00")
        w = _compute_staleness_weight(created_at, trade_date)
        assert w == 1.0

    def test_nan_inputs_full_weight(self):
        """NaN inputs should return fallback weight of 1.0."""
        assert _compute_staleness_weight(pd.NaT, pd.Timestamp("2024-01-10")) == 1.0
        assert _compute_staleness_weight(pd.Timestamp("2024-01-10"), pd.NaT) == 1.0

    def test_halflife_correct(self):
        """At exactly one halflife, weight should be ~0.5."""
        trade_date = pd.Timestamp("2024-01-10")
        # 4 PM - halflife hours = the created_at that gives exactly 1 halflife age
        halflife_h = SENTIMENT_DECAY_HALFLIFE_HOURS
        created_at = pd.Timestamp("2024-01-10 16:00:00") - pd.Timedelta(hours=halflife_h)
        w = _compute_staleness_weight(created_at, trade_date)
        assert abs(w - 0.5) < 0.05

    def test_weight_decreases_with_age(self):
        """Older articles should always have lower weight than newer ones."""
        trade_date = pd.Timestamp("2024-01-10")
        w_fresh = _compute_staleness_weight(
            pd.Timestamp("2024-01-10 14:00:00"), trade_date)
        w_medium = _compute_staleness_weight(
            pd.Timestamp("2024-01-10 02:00:00"), trade_date)
        w_old = _compute_staleness_weight(
            pd.Timestamp("2024-01-09 10:00:00"), trade_date)
        assert w_fresh > w_medium > w_old


class TestAggregateDailySentimentWithDecay:
    """Tests for decay-weighted aggregation."""

    def _make_scored_df(self, with_timestamps=False):
        """Create a simple scored DataFrame for testing."""
        rows = [
            {"date": pd.Timestamp("2024-01-10"), "symbol": "AAPL",
             "headline": "Apple surges", "sentiment_score": 0.8,
             "label": "positive", "text": "Apple surges"},
            {"date": pd.Timestamp("2024-01-10"), "symbol": "AAPL",
             "headline": "Old negative news", "sentiment_score": -0.6,
             "label": "negative", "text": "Old negative news"},
            {"date": pd.Timestamp("2024-01-10"), "symbol": "MSFT",
             "headline": "MSFT neutral", "sentiment_score": 0.0,
             "label": "neutral", "text": "MSFT neutral"},
        ]
        if with_timestamps:
            # First article is fresh, second is old
            rows[0]["created_at"] = pd.Timestamp("2024-01-10 14:00:00")
            rows[1]["created_at"] = pd.Timestamp("2024-01-08 10:00:00")  # 2 days old
            rows[2]["created_at"] = pd.Timestamp("2024-01-10 12:00:00")
        return pd.DataFrame(rows)

    def test_freshness_feature_exists(self):
        """Output should include sentiment_freshness column."""
        scored = self._make_scored_df()
        symbols = ["AAPL", "MSFT"]
        dates = [pd.Timestamp("2024-01-10")]
        result = aggregate_daily_sentiment(scored, symbols, dates)
        assert "sentiment_freshness" in result.columns

    def test_no_timestamps_equal_weight(self):
        """Without created_at column, all articles get weight 1.0."""
        scored = self._make_scored_df(with_timestamps=False)
        symbols = ["AAPL", "MSFT"]
        dates = [pd.Timestamp("2024-01-10")]
        result = aggregate_daily_sentiment(scored, symbols, dates)
        aapl = result[result["symbol"] == "AAPL"].iloc[0]
        # Without decay: mean of 0.8 and -0.6 = 0.1
        assert abs(aapl["sentiment_mean"] - 0.1) < 0.05

    def test_with_timestamps_decay_applied(self):
        """With timestamps, old articles should have less influence."""
        scored = self._make_scored_df(with_timestamps=True)
        symbols = ["AAPL", "MSFT"]
        dates = [pd.Timestamp("2024-01-10")]
        result = aggregate_daily_sentiment(scored, symbols, dates)
        aapl = result[result["symbol"] == "AAPL"].iloc[0]
        # With decay: the fresh positive article (0.8) dominates over the
        # 2-day-old negative article (-0.6). Weighted mean should be > 0.1
        assert aapl["sentiment_mean"] > 0.1

    def test_has_news_still_works(self):
        """has_news feature should still be present and correct."""
        scored = self._make_scored_df()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        dates = [pd.Timestamp("2024-01-10")]
        result = aggregate_daily_sentiment(scored, symbols, dates)
        aapl = result[result["symbol"] == "AAPL"].iloc[0]
        googl = result[result["symbol"] == "GOOGL"].iloc[0]
        assert aapl["has_news"] == 1.0
        assert googl["has_news"] == 0.0

    def test_empty_scored_df(self):
        """Empty scored DataFrame should produce all-zero features."""
        scored = pd.DataFrame(columns=[
            "date", "symbol", "headline", "sentiment_score", "label", "text"
        ])
        symbols = ["AAPL"]
        dates = [pd.Timestamp("2024-01-10")]
        result = aggregate_daily_sentiment(scored, symbols, dates)
        assert len(result) == 1
        assert result.iloc[0]["sentiment_mean"] == 0.0
        assert result.iloc[0]["sentiment_freshness"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
