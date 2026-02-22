# tests/unit/core/test_data_layer_ohlcv.py
# MASP v1.2.0-G -- STRICT MODE
# Target: jarvis.core.data_layer.OHLCV
# No mocks. No side effects. All tests independent. Pure pytest.

from __future__ import annotations

import math

import pytest

from jarvis.core.data_layer import OHLCV, NumericalInstabilityError


# =============================================================================
# Helpers
# =============================================================================

def _valid() -> OHLCV:
    """Canonical valid OHLCV instance. Satisfies all invariants."""
    return OHLCV(open=1.00, high=1.50, low=0.80, close=1.20, volume=1000.0)


# =============================================================================
# 1. Successful construction with valid values
# =============================================================================

class TestOHLCVValidConstruction:
    def test_basic_valid(self):
        ohlcv = _valid()
        assert ohlcv.open == 1.00
        assert ohlcv.high == 1.50
        assert ohlcv.low == 0.80
        assert ohlcv.close == 1.20
        assert ohlcv.volume == 1000.0

    def test_high_equals_low(self):
        # high == low is a valid doji candle
        ohlcv = OHLCV(open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        assert ohlcv.high == ohlcv.low

    def test_high_equals_open(self):
        ohlcv = OHLCV(open=1.5, high=1.5, low=1.0, close=1.2, volume=500.0)
        assert ohlcv.high == ohlcv.open

    def test_high_equals_close(self):
        ohlcv = OHLCV(open=1.1, high=1.5, low=1.0, close=1.5, volume=300.0)
        assert ohlcv.high == ohlcv.close

    def test_small_positive_values(self):
        ohlcv = OHLCV(open=1e-9, high=2e-9, low=1e-9, close=1.5e-9, volume=1e-9)
        assert ohlcv.volume > 0

    def test_large_values(self):
        ohlcv = OHLCV(open=1e10, high=2e10, low=5e9, close=1.5e10, volume=1e12)
        assert ohlcv.open > 0

    def test_is_frozen(self):
        ohlcv = _valid()
        with pytest.raises(Exception):
            ohlcv.open = 99.0  # type: ignore[misc]

    def test_integer_inputs_accepted(self):
        ohlcv = OHLCV(open=1, high=2, low=1, close=2, volume=100)
        assert ohlcv.open == 1
        assert ohlcv.volume == 100


# =============================================================================
# 2. NaN in any field -> NumericalInstabilityError
# =============================================================================

_NAN_FIELD_CASES = [
    ("open",   dict(open=float("nan"), high=1.5, low=0.8, close=1.2, volume=100.0)),
    ("high",   dict(open=1.0, high=float("nan"), low=0.8, close=1.2, volume=100.0)),
    ("low",    dict(open=1.0, high=1.5, low=float("nan"), close=1.2, volume=100.0)),
    ("close",  dict(open=1.0, high=1.5, low=0.8, close=float("nan"), volume=100.0)),
    ("volume", dict(open=1.0, high=1.5, low=0.8, close=1.2, volume=float("nan"))),
]

_INF_FIELD_CASES = [
    ("open",   dict(open=float("inf"), high=float("inf"), low=0.8, close=1.2, volume=100.0)),
    ("high",   dict(open=1.0, high=float("inf"), low=0.8, close=1.2, volume=100.0)),
    ("low",    dict(open=1.0, high=1.5, low=float("inf"), close=1.2, volume=100.0)),
    ("close",  dict(open=1.0, high=1.5, low=0.8, close=float("inf"), volume=100.0)),
    ("volume", dict(open=1.0, high=1.5, low=0.8, close=1.2, volume=float("inf"))),
]

_NEG_INF_FIELD_CASES = [
    ("open",   dict(open=float("-inf"), high=1.5, low=0.8, close=1.2, volume=100.0)),
    ("high",   dict(open=1.0, high=float("-inf"), low=0.8, close=1.2, volume=100.0)),
    ("low",    dict(open=1.0, high=1.5, low=float("-inf"), close=1.2, volume=100.0)),
    ("close",  dict(open=1.0, high=1.5, low=0.8, close=float("-inf"), volume=100.0)),
    ("volume", dict(open=1.0, high=1.5, low=0.8, close=1.2, volume=float("-inf"))),
]


class TestOHLCVNumericalInstability:
    @pytest.mark.parametrize("field_name,kwargs", _NAN_FIELD_CASES, ids=[c[0] for c in _NAN_FIELD_CASES])
    def test_nan_raises(self, field_name, kwargs):
        with pytest.raises(NumericalInstabilityError):
            OHLCV(**kwargs)

    @pytest.mark.parametrize("field_name,kwargs", _INF_FIELD_CASES, ids=["inf_" + c[0] for c in _INF_FIELD_CASES])
    def test_pos_inf_raises(self, field_name, kwargs):
        with pytest.raises(NumericalInstabilityError):
            OHLCV(**kwargs)

    @pytest.mark.parametrize("field_name,kwargs", _NEG_INF_FIELD_CASES, ids=["neg_inf_" + c[0] for c in _NEG_INF_FIELD_CASES])
    def test_neg_inf_raises(self, field_name, kwargs):
        with pytest.raises(NumericalInstabilityError):
            OHLCV(**kwargs)

    def test_nan_raises_numerical_instability_not_value_error_subtype_check(self):
        # NumericalInstabilityError IS a ValueError (subclass)
        with pytest.raises(ValueError):
            OHLCV(open=float("nan"), high=1.5, low=0.8, close=1.2, volume=100.0)

    def test_nan_error_message_contains_field_name(self):
        with pytest.raises(NumericalInstabilityError, match="open"):
            OHLCV(open=float("nan"), high=1.5, low=0.8, close=1.2, volume=100.0)

    def test_nan_checked_before_positivity(self):
        # NaN is not > 0 either; NumericalInstabilityError must be raised, not ValueError
        with pytest.raises(NumericalInstabilityError):
            OHLCV(open=float("nan"), high=1.5, low=0.8, close=1.2, volume=100.0)


# =============================================================================
# 3. Negative volume -> ValueError
# =============================================================================

class TestOHLCVNegativeVolume:
    def test_negative_volume_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=0.8, close=1.2, volume=-1.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=0.8, close=1.2, volume=0.0)

    def test_very_small_negative_volume_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=0.8, close=1.2, volume=-1e-15)

    @pytest.mark.parametrize("vol", [-1000.0, -0.001, -1e-10])
    def test_parametrized_negative_volume(self, vol):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=0.8, close=1.2, volume=vol)


# =============================================================================
# 4. high < low -> ValueError
# =============================================================================

class TestOHLCVHighLessThanLow:
    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=0.7, low=0.8, close=0.75, volume=100.0)

    def test_high_just_below_low_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=0.9999, low=1.0, close=1.0, volume=100.0)

    @pytest.mark.parametrize("high,low", [(0.5, 1.0), (0.1, 0.9), (1.0, 2.0)])
    def test_parametrized_high_less_than_low(self, high, low):
        close = high
        open_ = high
        with pytest.raises(ValueError):
            OHLCV(open=open_, high=high, low=low, close=close, volume=100.0)


# =============================================================================
# 5. high < open -> ValueError
# =============================================================================

class TestOHLCVHighLessThanOpen:
    def test_high_less_than_open_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=2.0, high=1.5, low=1.0, close=1.2, volume=100.0)

    def test_high_just_below_open_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0001, high=1.0, low=0.9, close=0.95, volume=100.0)

    @pytest.mark.parametrize("open_,high", [(3.0, 2.9), (10.0, 9.99), (1.5, 1.4)])
    def test_parametrized_high_less_than_open(self, open_, high):
        # low <= high, close <= high
        with pytest.raises(ValueError):
            OHLCV(open=open_, high=high, low=high - 0.05, close=high, volume=100.0)


# =============================================================================
# 6. high < close -> ValueError
# =============================================================================

class TestOHLCVHighLessThanClose:
    def test_high_less_than_close_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=0.8, close=1.6, volume=100.0)

    def test_high_just_below_close_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.4999, low=0.9, close=1.5, volume=100.0)

    @pytest.mark.parametrize("high,close", [(1.0, 1.1), (2.0, 3.0), (0.5, 0.6)])
    def test_parametrized_high_less_than_close(self, high, close):
        # open <= high so open = high; low <= high
        with pytest.raises(ValueError):
            OHLCV(open=high, high=high, low=high * 0.9, close=close, volume=100.0)


# =============================================================================
# 7. low > open: not a separately enforced invariant in this implementation.
#    The code only checks high < low, high < open, high < close.
#    A case where low > open is structurally possible without violating
#    any check (e.g. open=1.0, high=3.0, low=2.0, close=2.5).
#    Tests below confirm the actual enforced boundary behavior.
# =============================================================================

class TestOHLCVLowOpenRelationship:
    def test_low_greater_than_open_no_error_when_high_valid(self):
        # low(2.0) > open(1.0), but high(3.0) satisfies all three high-checks.
        # The implementation does NOT raise here -- this is the actual contract.
        ohlcv = OHLCV(open=1.0, high=3.0, low=2.0, close=2.5, volume=100.0)
        assert ohlcv.low > ohlcv.open

    def test_low_greater_than_open_raises_when_high_violated(self):
        # If additionally high < open, ValueError is raised (high constraint).
        with pytest.raises(ValueError):
            OHLCV(open=2.5, high=2.0, low=2.0, close=2.0, volume=100.0)


# =============================================================================
# 8. low > close: same reasoning as above.
# =============================================================================

class TestOHLCVLowCloseRelationship:
    def test_low_greater_than_close_no_error_when_high_valid(self):
        # low(2.0) > close(1.5), but high(3.0) satisfies all high-checks.
        ohlcv = OHLCV(open=2.5, high=3.0, low=2.0, close=1.5, volume=100.0)
        # No error raised -- low > close is not an enforced invariant.
        # close(1.5) < low(2.0) is permitted by the OHLCV implementation.
        assert ohlcv.low > ohlcv.close

    def test_low_greater_than_close_raises_when_high_violated(self):
        # high < close triggers ValueError regardless of low relationship.
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.0, low=1.0, close=1.5, volume=100.0)


# =============================================================================
# 9. open <= 0 -> ValueError
# =============================================================================

class TestOHLCVOpenNonPositive:
    def test_zero_open_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=0.0, high=1.5, low=0.0, close=1.2, volume=100.0)

    def test_negative_open_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=-1.0, high=1.5, low=-2.0, close=1.2, volume=100.0)

    def test_very_small_negative_open_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=-1e-15, high=1.0, low=-1e-15, close=1.0, volume=100.0)

    @pytest.mark.parametrize("open_val", [0.0, -0.001, -1.0, -1e6])
    def test_parametrized_non_positive_open(self, open_val):
        # high must be >= all other values to isolate only the open violation;
        # use a large high so that open <= 0 is the first positivity failure.
        with pytest.raises(ValueError):
            OHLCV(open=open_val, high=max(abs(open_val) + 1, 1.0),
                  low=open_val, close=0.5, volume=100.0)


# =============================================================================
# 10. close <= 0 -> ValueError
# =============================================================================

class TestOHLCVCloseNonPositive:
    def test_zero_close_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=0.0, close=0.0, volume=100.0)

    def test_negative_close_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=-1.0, close=-0.5, volume=100.0)

    def test_very_small_negative_close_raises(self):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=-1e-15, close=-1e-15, volume=100.0)

    @pytest.mark.parametrize("close_val", [0.0, -0.001, -1.0, -1e6])
    def test_parametrized_non_positive_close(self, close_val):
        with pytest.raises(ValueError):
            OHLCV(open=1.0, high=1.5, low=close_val, close=close_val, volume=100.0)


# =============================================================================
# Priority ordering: NumericalInstabilityError before ValueError
# =============================================================================

class TestOHLCVValidationOrder:
    def test_nan_takes_priority_over_positivity(self):
        # NaN in open is caught in Phase 1 before Phase 2 positivity check.
        with pytest.raises(NumericalInstabilityError):
            OHLCV(open=float("nan"), high=1.5, low=0.8, close=1.2, volume=100.0)

    def test_nan_takes_priority_over_structural(self):
        # NaN in high -- caught before high < low structural check.
        with pytest.raises(NumericalInstabilityError):
            OHLCV(open=1.0, high=float("nan"), low=0.8, close=1.2, volume=100.0)

    def test_positivity_checked_before_structural(self):
        # open=0 triggers positivity ValueError; high >= low is also satisfied.
        # Phase 2 runs before Phase 3.
        with pytest.raises(ValueError):
            OHLCV(open=0.0, high=1.0, low=0.0, close=0.5, volume=100.0)
