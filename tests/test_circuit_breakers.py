"""
tests/test_circuit_breakers.py
==============================
Unit tests for pipeline circuit breakers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from execution.circuit_breakers import (
    check_signal_sanity,
    check_allocation_jump,
    check_turnover,
    check_weight_sum,
    CircuitBreakerTripped,
)
from config.settings import CB_MAX_SIGNAL_RATIO, MIN_PROB


class TestSignalSanity:
    """Tests for the signal ratio circuit breaker."""

    def test_normal_ratio_passes(self):
        """50% of stocks above threshold should pass."""
        signals = pd.DataFrame({
            "pred_proba": [0.50, 0.50, 0.20, 0.20],
        })
        assert check_signal_sanity(signals, total_symbols=4)

    def test_high_ratio_trips(self):
        """90% of stocks above threshold should trip the breaker."""
        signals = pd.DataFrame({
            "pred_proba": [0.50] * 9 + [0.20],
        })
        with pytest.raises(CircuitBreakerTripped):
            check_signal_sanity(signals, total_symbols=10)

    def test_zero_symbols_no_crash(self):
        """Zero total symbols should not cause division by zero."""
        signals = pd.DataFrame({"pred_proba": []})
        assert check_signal_sanity(signals, total_symbols=0)


class TestAllocationJump:
    """Tests for the allocation jump circuit breaker."""

    def test_small_change_passes(self):
        """10% change should pass."""
        target = {"AAPL": 0.10, "MSFT": 0.10}
        previous = {"AAPL": 0.12, "MSFT": 0.12}
        assert check_allocation_jump(target, previous)

    def test_large_jump_trips(self):
        """50% jump should trip the breaker."""
        target = {"AAPL": 0.40, "MSFT": 0.40}  # 80% total
        previous = {"AAPL": 0.10}  # 10% total
        with pytest.raises(CircuitBreakerTripped):
            check_allocation_jump(target, previous)

    def test_first_run_no_previous(self):
        """First run with no previous weights should pass."""
        target = {"AAPL": 0.50}
        previous = {}
        assert check_allocation_jump(target, previous)


class TestWeightSum:
    """Tests for the weight sum circuit breaker."""

    def test_normal_sum_passes(self):
        """Weights summing to 0.85 should pass."""
        assert check_weight_sum({"AAPL": 0.40, "MSFT": 0.45})

    def test_over_one_trips(self):
        """Weights summing to 1.5 should trip the breaker."""
        with pytest.raises(CircuitBreakerTripped):
            check_weight_sum({"AAPL": 0.80, "MSFT": 0.80})

    def test_empty_weights_pass(self):
        """Empty weights (cash) should pass."""
        assert check_weight_sum({})


class TestTurnover:
    """Tests for the turnover warning (soft check)."""

    def test_low_turnover_passes(self):
        """10% turnover should pass without warning."""
        target = {"AAPL": 0.12, "MSFT": 0.10}
        current = {"AAPL": 0.10, "MSFT": 0.10}
        assert check_turnover(target, current)

    def test_high_turnover_still_passes(self):
        """High turnover is a soft warning, not a hard breaker."""
        target = {"AAPL": 0.50}
        current = {"MSFT": 0.50}  # completely different portfolio
        assert check_turnover(target, current)  # should still return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
