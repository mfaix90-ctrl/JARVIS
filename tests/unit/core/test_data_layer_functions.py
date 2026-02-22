# tests/unit/core/test_data_layer_functions.py
# MASP v1.2.0-G -- STRICT MODE
# Target: jarvis.core.data_layer -- pure functions and DataCache
# No mocks. No fixtures. Pure pytest. Deterministic. No side effects.

from __future__ import annotations

import math

import pytest

from jarvis.core.data_layer import (
    DataCache,
    DataQualityError,
    EnhancedMarketData,
    NumericalInstabilityError,
    OHLCV,
    QUALITY_HARD_GATE,
    SequenceError,
    ValidationResult,
    check_sequence,
    classify_session,
    compute_gap,
    estimate_quality_score,
    validate_enhanced_market_data,
    validate_numeric_field,
    validate_ohlcv,
    GAP_THRESHOLDS,
)


# =============================================================================
# Shared builder helpers
# =============================================================================

def _ohlcv() -> OHLCV:
    return OHLCV(open=1.00, high=1.50, low=0.80, close=1.20, volume=1000.0)


def _emd(**overrides) -> EnhancedMarketData:
    kwargs = dict(
        symbol="SPX",
        asset_class="indices",
        timeframe="D1",
        timestamp_utc=1_700_000_000,
        ohlcv=_ohlcv(),
        quality_score=0.9,
        sequence_id=1,
        data_source="historical",
        provider_id="provider_x",
        gap_detected=False,
        gap_size=None,
        session_tag="NEW_YORK",
        spread_bps=1.0,
        is_stale=False,
        liquidity_regime="normal",
    )
    kwargs.update(overrides)
    return EnhancedMarketData(**kwargs)


# =============================================================================
# 1) validate_numeric_field
# =============================================================================

class TestValidateNumericField:
    def test_normal_float_returned_unchanged(self):
        assert validate_numeric_field("x", 1.23) == 1.23

    def test_zero_is_valid(self):
        assert validate_numeric_field("x", 0.0) == 0.0

    def test_negative_finite_is_valid(self):
        assert validate_numeric_field("x", -99.9) == -99.9

    def test_large_float_is_valid(self):
        assert validate_numeric_field("x", 1e308) == 1e308

    def test_nan_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            validate_numeric_field("f", float("nan"))

    def test_pos_inf_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            validate_numeric_field("f", float("inf"))

    def test_neg_inf_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            validate_numeric_field("f", float("-inf"))

    def test_error_message_contains_field_name(self):
        with pytest.raises(NumericalInstabilityError, match="my_field"):
            validate_numeric_field("my_field", float("nan"))

    def test_numerical_instability_is_value_error(self):
        with pytest.raises(ValueError):
            validate_numeric_field("f", float("nan"))

    @pytest.mark.parametrize("val", [0.0, 1.0, -1.0, 1e-15, 1e15])
    def test_parametrized_finite_values(self, val):
        assert validate_numeric_field("v", val) == val


# =============================================================================
# 2) validate_ohlcv
# =============================================================================

class TestValidateOhlcv:
    def test_valid_ohlcv_passes(self):
        validate_ohlcv(_ohlcv())  # must not raise

    def test_nan_field_raises_numerical_instability(self):
        import types as _t
        # Build a namespace that mimics an OHLCV but with NaN in open.
        # validate_ohlcv uses getattr, so SimpleNamespace works.
        obj = _t.SimpleNamespace(open=float("nan"), high=1.5, low=0.8, close=1.2, volume=100.0)
        with pytest.raises(NumericalInstabilityError):
            validate_ohlcv(obj)  # type: ignore[arg-type]

    def test_zero_volume_raises_value_error(self):
        import types as _t
        obj = _t.SimpleNamespace(open=1.0, high=1.5, low=0.8, close=1.2, volume=0.0)
        with pytest.raises(ValueError):
            validate_ohlcv(obj)  # type: ignore[arg-type]

    def test_high_less_than_low_raises(self):
        import types as _t
        obj = _t.SimpleNamespace(open=1.0, high=0.7, low=1.0, close=0.75, volume=100.0)
        with pytest.raises(ValueError):
            validate_ohlcv(obj)  # type: ignore[arg-type]

    def test_high_less_than_open_raises(self):
        import types as _t
        obj = _t.SimpleNamespace(open=2.0, high=1.5, low=1.0, close=1.2, volume=100.0)
        with pytest.raises(ValueError):
            validate_ohlcv(obj)  # type: ignore[arg-type]

    def test_high_less_than_close_raises(self):
        import types as _t
        obj = _t.SimpleNamespace(open=1.0, high=1.5, low=0.8, close=2.0, volume=100.0)
        with pytest.raises(ValueError):
            validate_ohlcv(obj)  # type: ignore[arg-type]

    def test_valid_doji_candle_passes(self):
        ohlcv = OHLCV(open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        validate_ohlcv(ohlcv)  # must not raise


# =============================================================================
# 3) compute_gap
# =============================================================================

class TestComputeGap:
    def test_no_gap_same_price(self):
        detected, size = compute_gap(100.0, 100.0, "forex")
        assert detected is False
        assert size == 0.0

    def test_gap_below_threshold_forex(self):
        # forex threshold = 0.02 (2%). Gap of 1% should not trigger.
        detected, size = compute_gap(100.0, 101.0, "forex")
        assert detected is False
        assert math.isclose(size, 0.01, rel_tol=1e-9)

    def test_gap_above_threshold_forex(self):
        # forex threshold = 0.02. Gap of 3% must trigger.
        detected, size = compute_gap(100.0, 103.0, "forex")
        assert detected is True
        assert math.isclose(size, 0.03, rel_tol=1e-9)

    def test_gap_exactly_at_threshold_not_triggered(self):
        # gap_size > threshold, not >=. Exactly at threshold -> not detected.
        threshold = GAP_THRESHOLDS["forex"]  # 0.02
        current_open = 100.0 * (1.0 + threshold)
        detected, size = compute_gap(100.0, current_open, "forex")
        assert detected is False

    def test_gap_just_above_threshold_triggered(self):
        threshold = GAP_THRESHOLDS["forex"]
        current_open = 100.0 * (1.0 + threshold + 1e-9)
        detected, size = compute_gap(100.0, current_open, "forex")
        assert detected is True

    @pytest.mark.parametrize("asset_class,threshold", list(GAP_THRESHOLDS.items()))
    def test_gap_above_threshold_all_asset_classes(self, asset_class, threshold):
        current_open = 100.0 * (1.0 + threshold + 0.01)
        detected, size = compute_gap(100.0, current_open, asset_class)
        assert detected is True
        assert size > threshold

    @pytest.mark.parametrize("asset_class", ["crypto", "forex", "indices", "commodities", "rates"])
    def test_no_gap_all_asset_classes(self, asset_class):
        detected, size = compute_gap(100.0, 100.0, asset_class)
        assert detected is False
        assert size == 0.0

    def test_downward_gap_detected(self):
        # Gap is computed as abs(), so direction doesn't matter.
        detected, size = compute_gap(100.0, 94.0, "forex")
        assert detected is True
        assert math.isclose(size, 0.06, rel_tol=1e-9)

    def test_prev_close_zero_raises(self):
        with pytest.raises(ValueError, match="prev_close"):
            compute_gap(0.0, 100.0, "forex")

    def test_prev_close_negative_raises(self):
        with pytest.raises(ValueError, match="prev_close"):
            compute_gap(-1.0, 100.0, "forex")

    def test_current_open_zero_raises(self):
        with pytest.raises(ValueError, match="current_open"):
            compute_gap(100.0, 0.0, "forex")

    def test_current_open_negative_raises(self):
        with pytest.raises(ValueError, match="current_open"):
            compute_gap(100.0, -5.0, "forex")

    def test_invalid_asset_class_raises(self):
        with pytest.raises(ValueError):
            compute_gap(100.0, 101.0, "equities")

    def test_empty_asset_class_raises(self):
        with pytest.raises(ValueError):
            compute_gap(100.0, 101.0, "")

    def test_nan_prev_close_raises(self):
        with pytest.raises(NumericalInstabilityError):
            compute_gap(float("nan"), 100.0, "forex")

    def test_nan_current_open_raises(self):
        with pytest.raises(NumericalInstabilityError):
            compute_gap(100.0, float("nan"), "forex")

    def test_inf_prev_close_raises(self):
        with pytest.raises(NumericalInstabilityError):
            compute_gap(float("inf"), 100.0, "forex")

    def test_inf_current_open_raises(self):
        with pytest.raises(NumericalInstabilityError):
            compute_gap(100.0, float("inf"), "forex")

    def test_gap_size_always_non_negative(self):
        _, size = compute_gap(100.0, 90.0, "crypto")
        assert size >= 0.0

    def test_returns_tuple_of_bool_and_float(self):
        result = compute_gap(100.0, 100.0, "forex")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)


# =============================================================================
# 4) classify_session
# =============================================================================

class TestClassifySession:
    @pytest.mark.parametrize("tag", [
        "LONDON", "NEW_YORK", "TOKYO", "SYDNEY",
        "CRYPTO_24_7", "PRE_MARKET", "POST_MARKET", "AUCTION", "UNKNOWN",
    ])
    def test_valid_tags_returned_unchanged(self, tag):
        assert classify_session(tag) == tag

    @pytest.mark.parametrize("bad_tag", [
        "london", "new_york", "EUROPE", "NY", "ASIA", "", "FRANKFURT",
        "open", "CLOSE", "random_string",
    ])
    def test_unknown_tags_return_unknown(self, bad_tag):
        assert classify_session(bad_tag) == "UNKNOWN"

    def test_empty_string_returns_unknown(self):
        assert classify_session("") == "UNKNOWN"

    def test_never_raises(self):
        # classify_session must not raise for any input.
        for tag in ["xyz", "", "123", "LONDON", "london"]:
            result = classify_session(tag)
            assert isinstance(result, str)


# =============================================================================
# 5) check_sequence
# =============================================================================

class TestCheckSequence:
    def test_correct_monotonic_increment(self):
        check_sequence(0, 1)  # must not raise

    def test_large_jump_is_valid(self):
        check_sequence(0, 1_000_000)  # must not raise

    def test_increment_by_one_from_large(self):
        check_sequence(999, 1000)  # must not raise

    def test_regression_raises_sequence_error(self):
        with pytest.raises(SequenceError):
            check_sequence(5, 4)

    def test_large_regression_raises(self):
        with pytest.raises(SequenceError):
            check_sequence(100, 1)

    def test_equality_raises_sequence_error(self):
        with pytest.raises(SequenceError):
            check_sequence(3, 3)

    def test_zero_equality_raises(self):
        with pytest.raises(SequenceError):
            check_sequence(0, 0)

    def test_negative_prev_id_raises_value_error(self):
        with pytest.raises(ValueError, match="prev_id"):
            check_sequence(-1, 0)

    def test_negative_current_id_raises_value_error(self):
        with pytest.raises(ValueError, match="current_id"):
            check_sequence(0, -1)

    def test_both_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            check_sequence(-5, -3)

    def test_sequence_error_is_value_error(self):
        with pytest.raises(ValueError):
            check_sequence(5, 4)

    def test_returns_none_on_success(self):
        result = check_sequence(0, 1)
        assert result is None

    @pytest.mark.parametrize("prev,curr", [(0, 1), (10, 11), (0, 100), (999, 1000)])
    def test_parametrized_valid_sequences(self, prev, curr):
        check_sequence(prev, curr)  # must not raise

    @pytest.mark.parametrize("prev,curr", [(1, 0), (5, 5), (100, 99), (1000, 0)])
    def test_parametrized_invalid_sequences(self, prev, curr):
        with pytest.raises(SequenceError):
            check_sequence(prev, curr)


# =============================================================================
# 6) estimate_quality_score
# =============================================================================

class TestEstimateQualityScore:
    def test_all_ones_returns_one(self):
        score = estimate_quality_score(
            nan_ratio=0.0, completeness=1.0,
            session_factor=1.0, liquidity_factor=1.0,
        )
        assert math.isclose(score, 1.0)

    def test_normal_case(self):
        score = estimate_quality_score(
            nan_ratio=0.1, completeness=0.9,
            session_factor=0.8, liquidity_factor=0.9,
        )
        expected = 0.9 * (1.0 - 0.1) * 0.8 * 0.9
        assert math.isclose(score, expected, rel_tol=1e-9)

    def test_nan_ratio_one_returns_zero(self):
        score = estimate_quality_score(
            nan_ratio=1.0, completeness=1.0,
            session_factor=1.0, liquidity_factor=1.0,
        )
        assert score == 0.0

    def test_completeness_zero_returns_zero(self):
        score = estimate_quality_score(
            nan_ratio=0.0, completeness=0.0,
            session_factor=1.0, liquidity_factor=1.0,
        )
        assert score == 0.0

    def test_session_factor_zero_returns_zero(self):
        score = estimate_quality_score(
            nan_ratio=0.0, completeness=1.0,
            session_factor=0.0, liquidity_factor=1.0,
        )
        assert score == 0.0

    def test_liquidity_factor_zero_returns_zero(self):
        score = estimate_quality_score(
            nan_ratio=0.0, completeness=1.0,
            session_factor=1.0, liquidity_factor=0.0,
        )
        assert score == 0.0

    def test_output_clamped_at_one(self):
        # All 1.0 inputs -> raw = 1.0 -> clamped = 1.0
        score = estimate_quality_score(
            nan_ratio=0.0, completeness=1.0,
            session_factor=1.0, liquidity_factor=1.0,
        )
        assert score <= 1.0

    def test_output_clamped_at_zero(self):
        score = estimate_quality_score(
            nan_ratio=1.0, completeness=0.0,
            session_factor=0.0, liquidity_factor=0.0,
        )
        assert score >= 0.0

    def test_nan_nan_ratio_raises(self):
        with pytest.raises(NumericalInstabilityError):
            estimate_quality_score(
                nan_ratio=float("nan"), completeness=1.0,
                session_factor=1.0, liquidity_factor=1.0,
            )

    def test_nan_completeness_raises(self):
        with pytest.raises(NumericalInstabilityError):
            estimate_quality_score(
                nan_ratio=0.0, completeness=float("nan"),
                session_factor=1.0, liquidity_factor=1.0,
            )

    def test_nan_session_factor_raises(self):
        with pytest.raises(NumericalInstabilityError):
            estimate_quality_score(
                nan_ratio=0.0, completeness=1.0,
                session_factor=float("nan"), liquidity_factor=1.0,
            )

    def test_nan_liquidity_factor_raises(self):
        with pytest.raises(NumericalInstabilityError):
            estimate_quality_score(
                nan_ratio=0.0, completeness=1.0,
                session_factor=1.0, liquidity_factor=float("nan"),
            )

    def test_inf_raises(self):
        with pytest.raises(NumericalInstabilityError):
            estimate_quality_score(
                nan_ratio=float("inf"), completeness=1.0,
                session_factor=1.0, liquidity_factor=1.0,
            )

    def test_nan_ratio_above_one_raises(self):
        with pytest.raises(ValueError):
            estimate_quality_score(
                nan_ratio=1.1, completeness=1.0,
                session_factor=1.0, liquidity_factor=1.0,
            )

    def test_completeness_below_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_quality_score(
                nan_ratio=0.0, completeness=-0.1,
                session_factor=1.0, liquidity_factor=1.0,
            )

    def test_session_factor_above_one_raises(self):
        with pytest.raises(ValueError):
            estimate_quality_score(
                nan_ratio=0.0, completeness=1.0,
                session_factor=1.01, liquidity_factor=1.0,
            )

    def test_liquidity_factor_below_zero_raises(self):
        with pytest.raises(ValueError):
            estimate_quality_score(
                nan_ratio=0.0, completeness=1.0,
                session_factor=1.0, liquidity_factor=-0.01,
            )

    def test_output_in_zero_one_range(self):
        score = estimate_quality_score(
            nan_ratio=0.3, completeness=0.7,
            session_factor=0.9, liquidity_factor=0.85,
        )
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        a = estimate_quality_score(0.1, 0.9, 0.8, 0.9)
        b = estimate_quality_score(0.1, 0.9, 0.8, 0.9)
        assert a == b


# =============================================================================
# 7) validate_enhanced_market_data
# =============================================================================

class TestValidateEnhancedMarketData:
    def test_valid_returns_validation_result(self):
        result = validate_enhanced_market_data(_emd())
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.errors == []

    def test_quality_score_below_hard_gate_raises(self):
        # QUALITY_HARD_GATE = 0.5. Use 0.49 to trigger.
        data = _emd(quality_score=0.5)
        # Exactly 0.5 is NOT below 0.5 -> must not raise.
        validate_enhanced_market_data(data)

    def test_quality_score_just_below_hard_gate_raises(self):
        data = _emd(quality_score=0.49)
        with pytest.raises(DataQualityError):
            validate_enhanced_market_data(data)

    def test_quality_score_zero_raises(self):
        data = _emd(quality_score=0.0)
        with pytest.raises(DataQualityError):
            validate_enhanced_market_data(data)

    # -- indices: large gap -> warning --

    def test_indices_large_gap_produces_warning(self):
        # indices threshold = 0.03. gap_size = 0.05 > 0.03.
        data = _emd(
            asset_class="indices",
            gap_detected=True,
            gap_size=0.05,
        )
        result = validate_enhanced_market_data(data)
        assert result.valid is True
        assert result.gap_adjusted is True
        assert any("gap" in w.lower() for w in result.warnings)

    def test_indices_small_gap_no_warning(self):
        # gap_size = 0.01 < 0.03 threshold -> gap_adjusted True, no warning text
        data = _emd(
            asset_class="indices",
            gap_detected=True,
            gap_size=0.01,
        )
        result = validate_enhanced_market_data(data)
        assert result.gap_adjusted is True
        assert not any("Large gap" in w for w in result.warnings)

    def test_indices_no_gap_gap_adjusted_false(self):
        data = _emd(asset_class="indices", gap_detected=False, gap_size=None)
        result = validate_enhanced_market_data(data)
        assert result.gap_adjusted is False

    # -- forex + TOKYO + low liquidity -> warning --

    def test_forex_tokyo_low_liquidity_warning(self):
        data = _emd(
            asset_class="forex",
            session_tag="TOKYO",
            liquidity_regime="low",
            gap_detected=False,
            gap_size=None,
        )
        result = validate_enhanced_market_data(data)
        assert result.valid is True
        assert result.liquidity_adjusted is True
        assert any("liquidity" in w.lower() for w in result.warnings)

    def test_forex_tokyo_normal_liquidity_no_warning(self):
        data = _emd(
            asset_class="forex",
            session_tag="TOKYO",
            liquidity_regime="normal",
            gap_detected=False,
            gap_size=None,
        )
        result = validate_enhanced_market_data(data)
        assert result.liquidity_adjusted is False

    def test_forex_london_low_liquidity_no_forex_specific_warning(self):
        # LONDON + low does not trigger the TOKYO-specific warning,
        # but the cross-asset liquidity check still fires.
        data = _emd(
            asset_class="forex",
            session_tag="LONDON",
            liquidity_regime="low",
            gap_detected=False,
            gap_size=None,
        )
        result = validate_enhanced_market_data(data)
        assert result.liquidity_adjusted is True

    # -- crypto: large gap -> warning --

    def test_crypto_large_gap_produces_warning(self):
        # crypto threshold = 0.05. gap_size = 0.10.
        data = _emd(
            asset_class="crypto",
            gap_detected=True,
            gap_size=0.10,
            session_tag="CRYPTO_24_7",
        )
        result = validate_enhanced_market_data(data)
        assert result.gap_adjusted is True
        assert any("crypto" in w.lower() or "gap" in w.lower() for w in result.warnings)

    def test_crypto_small_gap_no_warning_text(self):
        data = _emd(
            asset_class="crypto",
            gap_detected=True,
            gap_size=0.01,
            session_tag="CRYPTO_24_7",
        )
        result = validate_enhanced_market_data(data)
        assert result.gap_adjusted is True
        assert not any("Unexpected gap" in w for w in result.warnings)

    # -- commodities: large gap -> warning --

    def test_commodities_large_gap_produces_warning(self):
        # commodities threshold = 0.04. gap_size = 0.08.
        data = _emd(
            asset_class="commodities",
            gap_detected=True,
            gap_size=0.08,
        )
        result = validate_enhanced_market_data(data)
        assert result.gap_adjusted is True
        assert any("commodities" in w.lower() or "gap" in w.lower() for w in result.warnings)

    # -- rates: large gap -> warning --

    def test_rates_large_gap_produces_warning(self):
        # rates threshold = 0.02. gap_size = 0.05.
        data = _emd(
            asset_class="rates",
            gap_detected=True,
            gap_size=0.05,
        )
        result = validate_enhanced_market_data(data)
        assert result.gap_adjusted is True
        assert any("rates" in w.lower() or "gap" in w.lower() for w in result.warnings)

    # -- liquidity_regime="low" -> liquidity_adjusted True --

    @pytest.mark.parametrize("asset_class,session_tag", [
        ("indices", "NEW_YORK"),
        ("crypto", "CRYPTO_24_7"),
        ("commodities", "LONDON"),
        ("rates", "NEW_YORK"),
    ])
    def test_low_liquidity_regime_sets_liquidity_adjusted(self, asset_class, session_tag):
        data = _emd(
            asset_class=asset_class,
            session_tag=session_tag,
            liquidity_regime="low",
            gap_detected=False,
            gap_size=None,
        )
        result = validate_enhanced_market_data(data)
        assert result.liquidity_adjusted is True

    def test_normal_liquidity_regime_not_adjusted(self):
        data = _emd(liquidity_regime="normal")
        result = validate_enhanced_market_data(data)
        assert result.liquidity_adjusted is False

    def test_high_liquidity_regime_not_adjusted(self):
        data = _emd(liquidity_regime="high")
        result = validate_enhanced_market_data(data)
        assert result.liquidity_adjusted is False

    # -- is_stale True -> stale warning --

    def test_is_stale_true_produces_warning(self):
        data = _emd(is_stale=True)
        result = validate_enhanced_market_data(data)
        assert result.valid is True
        assert any("stale" in w.lower() for w in result.warnings)

    def test_is_stale_false_no_stale_warning(self):
        data = _emd(is_stale=False)
        result = validate_enhanced_market_data(data)
        assert not any("stale" in w.lower() for w in result.warnings)

    def test_no_warnings_clean_data(self):
        data = _emd(
            asset_class="indices",
            gap_detected=False,
            gap_size=None,
            liquidity_regime="high",
            is_stale=False,
        )
        result = validate_enhanced_market_data(data)
        assert result.warnings == []
        assert result.errors == []

    def test_valid_field_always_true(self):
        # validate_enhanced_market_data always returns valid=True when no exception.
        data = _emd(is_stale=True, liquidity_regime="low")
        result = validate_enhanced_market_data(data)
        assert result.valid is True


# =============================================================================
# 8) DataCache
# =============================================================================

class TestDataCache:

    # -- store + retrieve before expiry --

    def test_store_and_retrieve_before_expiry(self):
        cache = DataCache()
        cache.store("k", "value", ttl_steps=10, now_step=0)
        result = cache.retrieve("k", now_step=5)
        assert result == "value"

    def test_retrieve_at_step_just_before_expiry(self):
        cache = DataCache()
        cache.store("k", 42, ttl_steps=5, now_step=0)
        # expiry_step = 5. now_step=4 < 5 -> still valid.
        assert cache.retrieve("k", now_step=4) == 42

    def test_store_any_type(self):
        cache = DataCache()
        cache.store("list_key", [1, 2, 3], ttl_steps=1, now_step=0)
        assert cache.retrieve("list_key", now_step=0) == [1, 2, 3]

    def test_overwrite_existing_key(self):
        cache = DataCache()
        cache.store("k", "first", ttl_steps=10, now_step=0)
        cache.store("k", "second", ttl_steps=10, now_step=0)
        assert cache.retrieve("k", now_step=0) == "second"

    # -- retrieve after expiry -> None --

    def test_retrieve_at_expiry_returns_none(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=5, now_step=0)
        # expiry_step = 5. now_step=5 >= 5 -> expired.
        result = cache.retrieve("k", now_step=5)
        assert result is None

    def test_retrieve_after_expiry_returns_none(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=5, now_step=0)
        result = cache.retrieve("k", now_step=100)
        assert result is None

    def test_retrieve_missing_key_returns_none(self):
        cache = DataCache()
        assert cache.retrieve("nonexistent", now_step=0) is None

    # -- lazy eviction --

    def test_lazy_eviction_removes_entry_after_expiry(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=3, now_step=0)
        assert cache.size() == 1
        # Trigger expiry via retrieve.
        cache.retrieve("k", now_step=3)
        assert cache.size() == 0

    def test_expired_entry_not_returned_after_eviction(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=2, now_step=0)
        cache.retrieve("k", now_step=5)  # triggers eviction
        # Second retrieve must also return None.
        assert cache.retrieve("k", now_step=6) is None

    def test_live_entry_not_evicted_by_other_retrieval(self):
        cache = DataCache()
        cache.store("a", 1, ttl_steps=10, now_step=0)
        cache.store("b", 2, ttl_steps=1, now_step=0)
        cache.retrieve("b", now_step=5)  # evicts "b"
        assert cache.retrieve("a", now_step=5) == 1

    # -- purge_expired --

    def test_purge_expired_removes_stale_entries(self):
        cache = DataCache()
        cache.store("a", 1, ttl_steps=2, now_step=0)
        cache.store("b", 2, ttl_steps=10, now_step=0)
        count = cache.purge_expired(now_step=2)
        assert count == 1
        assert cache.retrieve("b", now_step=2) == 2

    def test_purge_expired_returns_zero_when_nothing_expired(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=10, now_step=0)
        assert cache.purge_expired(now_step=5) == 0

    def test_purge_expired_all_entries(self):
        cache = DataCache()
        for i in range(5):
            cache.store(str(i), i, ttl_steps=1, now_step=0)
        count = cache.purge_expired(now_step=10)
        assert count == 5
        assert cache.size() == 0

    def test_purge_expired_invalid_now_step_raises(self):
        cache = DataCache()
        with pytest.raises(ValueError):
            cache.purge_expired(now_step=-1)

    # -- invalidate --

    def test_invalidate_removes_entry(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=10, now_step=0)
        cache.invalidate("k")
        assert cache.retrieve("k", now_step=0) is None

    def test_invalidate_nonexistent_key_no_error(self):
        cache = DataCache()
        cache.invalidate("does_not_exist")  # must not raise

    def test_invalidate_only_removes_target_key(self):
        cache = DataCache()
        cache.store("a", 1, ttl_steps=10, now_step=0)
        cache.store("b", 2, ttl_steps=10, now_step=0)
        cache.invalidate("a")
        assert cache.retrieve("b", now_step=0) == 2
        assert cache.retrieve("a", now_step=0) is None

    # -- size --

    def test_size_empty_cache(self):
        cache = DataCache()
        assert cache.size() == 0

    def test_size_after_store(self):
        cache = DataCache()
        cache.store("a", 1, ttl_steps=5, now_step=0)
        cache.store("b", 2, ttl_steps=5, now_step=0)
        assert cache.size() == 2

    def test_size_includes_logically_expired_before_eviction(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=1, now_step=0)
        # Expired at now_step=1 but not yet lazily evicted.
        assert cache.size() == 1

    def test_size_decreases_after_invalidate(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=10, now_step=0)
        cache.invalidate("k")
        assert cache.size() == 0

    # -- invalid inputs: store --

    def test_store_empty_key_raises(self):
        cache = DataCache()
        with pytest.raises(ValueError):
            cache.store("", "v", ttl_steps=5, now_step=0)

    def test_store_zero_ttl_raises(self):
        cache = DataCache()
        with pytest.raises(ValueError):
            cache.store("k", "v", ttl_steps=0, now_step=0)

    def test_store_negative_ttl_raises(self):
        cache = DataCache()
        with pytest.raises(ValueError):
            cache.store("k", "v", ttl_steps=-1, now_step=0)

    def test_store_negative_now_step_raises(self):
        cache = DataCache()
        with pytest.raises(ValueError):
            cache.store("k", "v", ttl_steps=5, now_step=-1)

    def test_store_float_ttl_raises(self):
        cache = DataCache()
        with pytest.raises((ValueError, TypeError)):
            cache.store("k", "v", ttl_steps=1.5, now_step=0)  # type: ignore[arg-type]

    # -- invalid inputs: retrieve --

    def test_retrieve_negative_now_step_raises(self):
        cache = DataCache()
        with pytest.raises(ValueError):
            cache.retrieve("k", now_step=-1)

    # -- isolation between instances --

    def test_two_cache_instances_are_independent(self):
        cache1 = DataCache()
        cache2 = DataCache()
        cache1.store("k", "from_cache1", ttl_steps=10, now_step=0)
        assert cache2.retrieve("k", now_step=0) is None

    # -- ttl boundary semantics --

    def test_ttl_semantics_expiry_at_now_plus_ttl(self):
        cache = DataCache()
        cache.store("k", "v", ttl_steps=3, now_step=10)
        # expiry_step = 13
        assert cache.retrieve("k", now_step=12) == "v"   # 12 < 13
        assert cache.retrieve("k", now_step=13) is None  # 13 >= 13 (expired)
