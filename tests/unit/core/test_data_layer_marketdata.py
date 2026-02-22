# tests/unit/core/test_data_layer_marketdata.py
# MASP v1.2.0-G -- STRICT MODE
# Target: jarvis.core.data_layer.MarketData / EnhancedMarketData
# No mocks. No side effects. All tests independent. Pure pytest.

from __future__ import annotations

import math

import pytest

from jarvis.core.data_layer import (
    DataQualityError,
    EnhancedMarketData,
    MarketData,
    OHLCV,
    NumericalInstabilityError,
)


# =============================================================================
# Shared fixtures (plain functions -- no pytest fixtures, no side effects)
# =============================================================================

def _valid_ohlcv() -> OHLCV:
    return OHLCV(open=1.00, high=1.50, low=0.80, close=1.20, volume=1000.0)


def _valid_market_data(**overrides) -> MarketData:
    kwargs = dict(
        symbol="BTC/USDT",
        asset_class="crypto",
        timeframe="H1",
        timestamp_utc=1_700_000_000,
        ohlcv=_valid_ohlcv(),
        quality_score=0.9,
        sequence_id=0,
        data_source="historical",
        provider_id="exchange_a",
    )
    kwargs.update(overrides)
    return MarketData(**kwargs)


def _valid_enhanced_market_data(**overrides) -> EnhancedMarketData:
    kwargs = dict(
        symbol="EUR/USD",
        asset_class="forex",
        timeframe="M15",
        timestamp_utc=1_700_000_000,
        ohlcv=_valid_ohlcv(),
        quality_score=0.85,
        sequence_id=1,
        data_source="live",
        provider_id="broker_b",
        gap_detected=False,
        gap_size=None,
        session_tag="LONDON",
        spread_bps=0.5,
        is_stale=False,
        liquidity_regime="normal",
    )
    kwargs.update(overrides)
    return EnhancedMarketData(**kwargs)


# =============================================================================
# MarketData -- 1. Successful construction with valid values
# =============================================================================

class TestMarketDataValidConstruction:
    def test_basic_valid(self):
        md = _valid_market_data()
        assert md.symbol == "BTC/USDT"
        assert md.asset_class == "crypto"
        assert md.timeframe == "H1"
        assert md.quality_score == 0.9
        assert md.sequence_id == 0
        assert md.data_source == "historical"
        assert md.provider_id == "exchange_a"

    def test_is_frozen(self):
        md = _valid_market_data()
        with pytest.raises(Exception):
            md.symbol = "ETH/USDT"  # type: ignore[misc]

    def test_quality_score_zero(self):
        md = _valid_market_data(quality_score=0.0)
        assert md.quality_score == 0.0

    def test_quality_score_one(self):
        md = _valid_market_data(quality_score=1.0)
        assert md.quality_score == 1.0

    def test_quality_score_boundary_half(self):
        md = _valid_market_data(quality_score=0.5)
        assert md.quality_score == 0.5

    def test_sequence_id_zero(self):
        md = _valid_market_data(sequence_id=0)
        assert md.sequence_id == 0

    def test_sequence_id_large(self):
        md = _valid_market_data(sequence_id=10_000_000)
        assert md.sequence_id == 10_000_000

    def test_timestamp_utc_zero(self):
        md = _valid_market_data(timestamp_utc=0)
        assert md.timestamp_utc == 0

    @pytest.mark.parametrize("asset_class", ["crypto", "forex", "indices", "commodities", "rates"])
    def test_all_valid_asset_classes(self, asset_class):
        md = _valid_market_data(asset_class=asset_class)
        assert md.asset_class == asset_class

    @pytest.mark.parametrize("timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"])
    def test_all_valid_timeframes(self, timeframe):
        md = _valid_market_data(timeframe=timeframe)
        assert md.timeframe == timeframe

    @pytest.mark.parametrize("data_source", ["historical", "live", "hybrid_backfill", "hybrid_live"])
    def test_all_valid_data_sources(self, data_source):
        md = _valid_market_data(data_source=data_source)
        assert md.data_source == data_source


# =============================================================================
# MarketData -- 2. invalid asset_class -> ValueError
# =============================================================================

class TestMarketDataInvalidAssetClass:
    @pytest.mark.parametrize("bad_asset", [
        "CRYPTO", "Crypto", "equities", "bonds", "", "unknown", "fx",
    ])
    def test_invalid_asset_class_raises(self, bad_asset):
        with pytest.raises(ValueError):
            _valid_market_data(asset_class=bad_asset)

    def test_empty_asset_class_raises(self):
        with pytest.raises(ValueError):
            _valid_market_data(asset_class="")

    def test_error_message_contains_field(self):
        with pytest.raises(ValueError, match="asset_class"):
            _valid_market_data(asset_class="equities")


# =============================================================================
# MarketData -- 3. invalid timeframe -> ValueError
# =============================================================================

class TestMarketDataInvalidTimeframe:
    @pytest.mark.parametrize("bad_tf", [
        "1H", "h1", "m15", "1m", "daily", "weekly", "", "H2", "M60",
    ])
    def test_invalid_timeframe_raises(self, bad_tf):
        with pytest.raises(ValueError):
            _valid_market_data(timeframe=bad_tf)

    def test_empty_timeframe_raises(self):
        with pytest.raises(ValueError):
            _valid_market_data(timeframe="")

    def test_error_message_contains_field(self):
        with pytest.raises(ValueError, match="timeframe"):
            _valid_market_data(timeframe="1H")


# =============================================================================
# MarketData -- 4. quality_score < 0 -> DataQualityError
# =============================================================================

class TestMarketDataQualityScoreBelowZero:
    def test_negative_quality_score_raises(self):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=-0.01)

    def test_quality_score_minus_one_raises(self):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=-1.0)

    def test_quality_score_very_small_negative_raises(self):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=-1e-15)

    @pytest.mark.parametrize("score", [-0.001, -0.5, -1.0, -100.0])
    def test_parametrized_negative_quality_score(self, score):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=score)

    def test_data_quality_error_is_value_error(self):
        with pytest.raises(ValueError):
            _valid_market_data(quality_score=-0.1)


# =============================================================================
# MarketData -- 5. quality_score > 1 -> DataQualityError
# =============================================================================

class TestMarketDataQualityScoreAboveOne:
    def test_quality_score_above_one_raises(self):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=1.001)

    def test_quality_score_two_raises(self):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=2.0)

    def test_quality_score_very_slightly_above_one_raises(self):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=1.0 + 1e-15)

    @pytest.mark.parametrize("score", [1.001, 1.5, 2.0, 100.0])
    def test_parametrized_quality_score_above_one(self, score):
        with pytest.raises(DataQualityError):
            _valid_market_data(quality_score=score)


# =============================================================================
# MarketData -- 6. quality_score NaN -> NumericalInstabilityError
#               (NaN is non-finite; raises NumericalInstabilityError per code)
# =============================================================================

class TestMarketDataQualityScoreNaN:
    def test_nan_quality_score_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            _valid_market_data(quality_score=float("nan"))

    def test_pos_inf_quality_score_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            _valid_market_data(quality_score=float("inf"))

    def test_neg_inf_quality_score_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            _valid_market_data(quality_score=float("-inf"))

    def test_numerical_instability_is_value_error(self):
        # NumericalInstabilityError inherits from ValueError
        with pytest.raises(ValueError):
            _valid_market_data(quality_score=float("nan"))


# =============================================================================
# MarketData -- 7. sequence_id < 0 -> ValueError
# =============================================================================

class TestMarketDataSequenceId:
    def test_negative_sequence_id_raises(self):
        with pytest.raises(ValueError):
            _valid_market_data(sequence_id=-1)

    def test_large_negative_sequence_id_raises(self):
        with pytest.raises(ValueError):
            _valid_market_data(sequence_id=-1_000_000)

    @pytest.mark.parametrize("sid", [-1, -10, -999])
    def test_parametrized_negative_sequence_id(self, sid):
        with pytest.raises(ValueError):
            _valid_market_data(sequence_id=sid)

    def test_error_message_contains_field(self):
        with pytest.raises(ValueError, match="sequence_id"):
            _valid_market_data(sequence_id=-1)


# =============================================================================
# MarketData -- 8. invalid data_source -> ValueError
# =============================================================================

class TestMarketDataInvalidDataSource:
    @pytest.mark.parametrize("bad_source", [
        "HISTORICAL", "Historic", "realtime", "backfill", "", "api", "live_feed",
    ])
    def test_invalid_data_source_raises(self, bad_source):
        with pytest.raises(ValueError):
            _valid_market_data(data_source=bad_source)

    def test_empty_data_source_raises(self):
        with pytest.raises(ValueError):
            _valid_market_data(data_source="")

    def test_error_message_contains_field(self):
        with pytest.raises(ValueError, match="data_source"):
            _valid_market_data(data_source="realtime")


# =============================================================================
# EnhancedMarketData -- 1. Successful construction with valid values
# =============================================================================

class TestEnhancedMarketDataValidConstruction:
    def test_basic_valid_no_gap(self):
        emd = _valid_enhanced_market_data()
        assert emd.gap_detected is False
        assert emd.gap_size is None
        assert emd.session_tag == "LONDON"
        assert emd.spread_bps == 0.5
        assert emd.is_stale is False
        assert emd.liquidity_regime == "normal"

    def test_valid_with_gap(self):
        emd = _valid_enhanced_market_data(gap_detected=True, gap_size=0.03)
        assert emd.gap_detected is True
        assert emd.gap_size == 0.03

    def test_is_frozen(self):
        emd = _valid_enhanced_market_data()
        with pytest.raises(Exception):
            emd.session_tag = "TOKYO"  # type: ignore[misc]

    def test_spread_bps_zero(self):
        emd = _valid_enhanced_market_data(spread_bps=0.0)
        assert emd.spread_bps == 0.0

    def test_is_stale_true(self):
        emd = _valid_enhanced_market_data(is_stale=True)
        assert emd.is_stale is True

    @pytest.mark.parametrize("regime", ["high", "normal", "low", "unknown"])
    def test_all_valid_liquidity_regimes(self, regime):
        emd = _valid_enhanced_market_data(liquidity_regime=regime)
        assert emd.liquidity_regime == regime

    @pytest.mark.parametrize("tag", [
        "LONDON", "NEW_YORK", "TOKYO", "SYDNEY",
        "CRYPTO_24_7", "PRE_MARKET", "POST_MARKET", "AUCTION", "UNKNOWN",
    ])
    def test_all_valid_session_tags(self, tag):
        emd = _valid_enhanced_market_data(session_tag=tag)
        assert emd.session_tag == tag

    def test_inherits_marketdata_validation(self):
        # Parent validation still runs: invalid asset_class raises ValueError.
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(asset_class="equities")

    def test_gap_size_zero_when_gap_detected(self):
        emd = _valid_enhanced_market_data(gap_detected=True, gap_size=0.0)
        assert emd.gap_size == 0.0


# =============================================================================
# EnhancedMarketData -- 2. gap_detected=True + gap_size=None -> ValueError
# =============================================================================

class TestEnhancedMarketDataGapDetectedNoneSize:
    def test_gap_detected_true_gap_size_none_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(gap_detected=True, gap_size=None)

    def test_error_message_contains_gap_size(self):
        with pytest.raises(ValueError, match="gap_size"):
            _valid_enhanced_market_data(gap_detected=True, gap_size=None)


# =============================================================================
# EnhancedMarketData -- 3. gap_detected=False + gap_size!=None -> ValueError
# =============================================================================

class TestEnhancedMarketDataNoGapButSizeSet:
    def test_gap_detected_false_gap_size_set_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(gap_detected=False, gap_size=0.01)

    def test_gap_detected_false_gap_size_zero_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(gap_detected=False, gap_size=0.0)

    @pytest.mark.parametrize("gap_size", [0.001, 0.05, 1.0, 99.9])
    def test_parametrized_no_gap_but_size_set(self, gap_size):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(gap_detected=False, gap_size=gap_size)

    def test_error_message_contains_gap_size(self):
        with pytest.raises(ValueError, match="gap_size"):
            _valid_enhanced_market_data(gap_detected=False, gap_size=0.05)


# =============================================================================
# EnhancedMarketData -- 4. spread_bps < 0 -> ValueError
# =============================================================================

class TestEnhancedMarketDataNegativeSpread:
    def test_negative_spread_bps_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(spread_bps=-0.01)

    def test_large_negative_spread_bps_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(spread_bps=-100.0)

    def test_very_small_negative_spread_bps_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(spread_bps=-1e-15)

    @pytest.mark.parametrize("bps", [-0.001, -1.0, -50.0])
    def test_parametrized_negative_spread_bps(self, bps):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(spread_bps=bps)

    def test_error_message_contains_spread_bps(self):
        with pytest.raises(ValueError, match="spread_bps"):
            _valid_enhanced_market_data(spread_bps=-1.0)

    def test_nan_spread_bps_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            _valid_enhanced_market_data(spread_bps=float("nan"))

    def test_inf_spread_bps_raises_numerical_instability(self):
        with pytest.raises(NumericalInstabilityError):
            _valid_enhanced_market_data(spread_bps=float("inf"))


# =============================================================================
# EnhancedMarketData -- 5. invalid liquidity_regime -> ValueError
# =============================================================================

class TestEnhancedMarketDataInvalidLiquidityRegime:
    @pytest.mark.parametrize("bad_regime", [
        "HIGH", "Normal", "LOW", "medium", "none", "", "extreme", "liquid",
    ])
    def test_invalid_liquidity_regime_raises(self, bad_regime):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(liquidity_regime=bad_regime)

    def test_empty_liquidity_regime_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(liquidity_regime="")

    def test_error_message_contains_liquidity_regime(self):
        with pytest.raises(ValueError, match="liquidity_regime"):
            _valid_enhanced_market_data(liquidity_regime="medium")


# =============================================================================
# EnhancedMarketData -- 6. invalid session_tag -> ValueError
# =============================================================================

class TestEnhancedMarketDataInvalidSessionTag:
    @pytest.mark.parametrize("bad_tag", [
        "london", "new_york", "NY", "EUROPE", "ASIA", "", "FRANKFURT", "OPEN",
    ])
    def test_invalid_session_tag_raises(self, bad_tag):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(session_tag=bad_tag)

    def test_empty_session_tag_raises(self):
        with pytest.raises(ValueError):
            _valid_enhanced_market_data(session_tag="")

    def test_error_message_contains_session_tag(self):
        with pytest.raises(ValueError, match="session_tag"):
            _valid_enhanced_market_data(session_tag="EUROPE")


# =============================================================================
# EnhancedMarketData -- validation order: parent before child
# =============================================================================

class TestEnhancedMarketDataValidationOrder:
    def test_parent_validation_runs_before_child(self):
        # Invalid asset_class (parent check) takes priority over gap inconsistency (child check).
        with pytest.raises(ValueError, match="asset_class"):
            _valid_enhanced_market_data(
                asset_class="equities",
                gap_detected=True,
                gap_size=None,
            )

    def test_quality_score_nan_raises_before_gap_check(self):
        # Parent NaN check on quality_score fires before child gap consistency check.
        with pytest.raises(NumericalInstabilityError):
            _valid_enhanced_market_data(
                quality_score=float("nan"),
                gap_detected=True,
                gap_size=None,
            )
