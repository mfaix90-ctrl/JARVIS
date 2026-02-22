# jarvis/tests/test_pipeline_contract.py
# Version: 1.0.0
# Contract tests for the JARVIS pipeline.
# FAS v6.1.0 -- External test layer.
#
# CONSTRAINTS:
#   No business logic.
#   No reimplementation of any canonical function.
#   Only imports and assertions.
#   Pure deterministic contract tests.
#   No private internals imported.
#   ASCII only.
#
# Standard import pattern:
#   from jarvis.orchestrator import run_full_pipeline
#   from jarvis.backtest import run_backtest
#   from jarvis.core.regime import GlobalRegimeState

import pytest

from jarvis.orchestrator import run_full_pipeline
from jarvis.backtest import run_backtest
from jarvis.core.regime import GlobalRegimeState


# ---------------------------------------------------------------------------
# SHARED FIXTURES
# ---------------------------------------------------------------------------

# Minimal valid returns history satisfying RiskEngine minimum of 20 periods.
_RETURNS_20: list[float] = [0.01, -0.02, 0.015, -0.005, 0.02,
                             0.01, -0.01, 0.005, -0.015, 0.02,
                             0.01, -0.02, 0.015, -0.005, 0.02,
                             0.01, -0.01, 0.005, -0.015, 0.02]

# Longer series for backtest tests.
_RETURNS_40: list[float] = _RETURNS_20 + _RETURNS_20

# Matching price series for backtest tests (all strictly positive).
_PRICES_40: list[float] = [100.0 + float(i) for i in range(40)]

_REGIME_RISK_ON:  GlobalRegimeState = GlobalRegimeState.RISK_ON
_REGIME_RISK_OFF: GlobalRegimeState = GlobalRegimeState.RISK_OFF
_REGIME_TRANSITION: GlobalRegimeState = GlobalRegimeState.TRANSITION

_CAPITAL: float = 100_000.0
_ASSET_PRICES: dict[str, float] = {"ASSET": 500.0}
_META_UNCERTAINTY: float = 0.1


# ---------------------------------------------------------------------------
# CONTRACT: run_full_pipeline -- RETURN TYPE
# ---------------------------------------------------------------------------

class TestRunFullPipelineReturnType:

    def test_returns_dict(self) -> None:
        result = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert isinstance(result, dict)

    def test_keys_match_asset_prices_keys(self) -> None:
        result = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert set(result.keys()) == set(_ASSET_PRICES.keys())

    def test_all_values_are_float(self) -> None:
        result = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        for value in result.values():
            assert isinstance(value, float)

    def test_all_position_sizes_are_finite(self) -> None:
        import math
        result = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        for value in result.values():
            assert math.isfinite(value)

    def test_all_position_sizes_are_non_negative(self) -> None:
        result = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        for value in result.values():
            assert value >= 0.0


# ---------------------------------------------------------------------------
# CONTRACT: run_full_pipeline -- DETERMINISM
# ---------------------------------------------------------------------------

class TestRunFullPipelineDeterminism:

    def test_identical_inputs_produce_identical_output_risk_on(self) -> None:
        result_a = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        result_b = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert result_a == result_b

    def test_identical_inputs_produce_identical_output_risk_off(self) -> None:
        result_a = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_OFF,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        result_b = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_OFF,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert result_a == result_b

    def test_identical_inputs_produce_identical_output_transition(self) -> None:
        result_a = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_TRANSITION,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        result_b = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_TRANSITION,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert result_a == result_b

    def test_repeated_calls_do_not_mutate_input_returns(self) -> None:
        returns_copy: list[float] = list(_RETURNS_20)
        run_full_pipeline(
            returns_history=returns_copy,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert returns_copy == _RETURNS_20

    def test_repeated_calls_do_not_mutate_input_asset_prices(self) -> None:
        prices_copy: dict[str, float] = dict(_ASSET_PRICES)
        run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=prices_copy,
        )
        assert prices_copy == _ASSET_PRICES

    def test_different_regime_produces_different_or_equal_output(self) -> None:
        # Determinism contract: same regime -> same output.
        # Different regime is permitted to produce a different output
        # but must never raise an exception.
        result_on = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        result_off = run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=_REGIME_RISK_OFF,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )
        assert isinstance(result_on, dict)
        assert isinstance(result_off, dict)


# ---------------------------------------------------------------------------
# CONTRACT: run_full_pipeline -- REGIME COVERAGE
# ---------------------------------------------------------------------------

class TestRunFullPipelineRegimeCoverage:

    def test_risk_on_does_not_raise(self) -> None:
        run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=GlobalRegimeState.RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )

    def test_risk_off_does_not_raise(self) -> None:
        run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=GlobalRegimeState.RISK_OFF,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )

    def test_transition_does_not_raise(self) -> None:
        run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=GlobalRegimeState.TRANSITION,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )

    def test_crisis_does_not_raise(self) -> None:
        run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=GlobalRegimeState.CRISIS,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )

    def test_unknown_does_not_raise(self) -> None:
        run_full_pipeline(
            returns_history=_RETURNS_20,
            current_regime=GlobalRegimeState.UNKNOWN,
            meta_uncertainty=_META_UNCERTAINTY,
            total_capital=_CAPITAL,
            asset_prices=_ASSET_PRICES,
        )


# ---------------------------------------------------------------------------
# CONTRACT: run_full_pipeline -- INVALID INPUTS
# ---------------------------------------------------------------------------

class TestRunFullPipelineInvalidInputs:

    def test_returns_history_too_short_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=[0.01] * 19,
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=_CAPITAL,
                asset_prices=_ASSET_PRICES,
            )

    def test_empty_returns_history_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=[],
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=_CAPITAL,
                asset_prices=_ASSET_PRICES,
            )

    def test_zero_total_capital_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=_RETURNS_20,
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=0.0,
                asset_prices=_ASSET_PRICES,
            )

    def test_negative_total_capital_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=_RETURNS_20,
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=-1.0,
                asset_prices=_ASSET_PRICES,
            )

    def test_empty_asset_prices_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=_RETURNS_20,
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=_CAPITAL,
                asset_prices={},
            )

    def test_non_positive_asset_price_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=_RETURNS_20,
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=_CAPITAL,
                asset_prices={"ASSET": 0.0},
            )

    def test_negative_asset_price_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_full_pipeline(
                returns_history=_RETURNS_20,
                current_regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
                total_capital=_CAPITAL,
                asset_prices={"ASSET": -100.0},
            )


# ---------------------------------------------------------------------------
# CONTRACT: run_backtest -- RETURN TYPE
# ---------------------------------------------------------------------------

class TestRunBacktestReturnType:

    def test_returns_list(self) -> None:
        result = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        assert isinstance(result, list)

    def test_all_values_are_float(self) -> None:
        result = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        for value in result:
            assert isinstance(value, float)

    def test_equity_curve_length(self) -> None:
        result = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        assert len(result) == len(_RETURNS_40) - 20

    def test_all_equity_values_are_finite(self) -> None:
        import math
        result = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        for value in result:
            assert math.isfinite(value)

    def test_all_equity_values_are_positive(self) -> None:
        result = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        for value in result:
            assert value > 0.0


# ---------------------------------------------------------------------------
# CONTRACT: run_backtest -- DETERMINISM
# ---------------------------------------------------------------------------

class TestRunBacktestDeterminism:

    def test_identical_inputs_produce_identical_equity_curve(self) -> None:
        result_a = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        result_b = run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        assert result_a == result_b

    def test_repeated_calls_do_not_mutate_returns_series(self) -> None:
        returns_copy: list[float] = list(_RETURNS_40)
        run_backtest(
            returns_series=returns_copy,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        assert returns_copy == _RETURNS_40

    def test_repeated_calls_do_not_mutate_price_series(self) -> None:
        prices_copy: list[float] = list(_PRICES_40)
        run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=prices_copy,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )
        assert prices_copy == _PRICES_40


# ---------------------------------------------------------------------------
# CONTRACT: run_backtest -- INVALID WINDOW LENGTH
# ---------------------------------------------------------------------------

class TestRunBacktestInvalidWindow:

    def test_window_below_20_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=19,
                initial_capital=_CAPITAL,
                asset_price_series=_PRICES_40,
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )

    def test_window_of_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=0,
                initial_capital=_CAPITAL,
                asset_price_series=_PRICES_40,
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )

    def test_window_of_one_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=1,
                initial_capital=_CAPITAL,
                asset_price_series=_PRICES_40,
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )

    def test_window_of_exactly_20_does_not_raise(self) -> None:
        run_backtest(
            returns_series=_RETURNS_40,
            window=20,
            initial_capital=_CAPITAL,
            asset_price_series=_PRICES_40,
            regime=_REGIME_RISK_ON,
            meta_uncertainty=_META_UNCERTAINTY,
        )


# ---------------------------------------------------------------------------
# CONTRACT: run_backtest -- INVALID CAPITAL
# ---------------------------------------------------------------------------

class TestRunBacktestInvalidCapital:

    def test_zero_initial_capital_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=20,
                initial_capital=0.0,
                asset_price_series=_PRICES_40,
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )

    def test_negative_initial_capital_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=20,
                initial_capital=-1.0,
                asset_price_series=_PRICES_40,
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )


# ---------------------------------------------------------------------------
# CONTRACT: run_backtest -- MISMATCHED SERIES LENGTH
# ---------------------------------------------------------------------------

class TestRunBacktestMismatchedLength:

    def test_price_series_longer_than_returns_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=20,
                initial_capital=_CAPITAL,
                asset_price_series=_PRICES_40 + [999.0],
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )

    def test_price_series_shorter_than_returns_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=20,
                initial_capital=_CAPITAL,
                asset_price_series=_PRICES_40[:-1],
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )

    def test_empty_price_series_with_non_empty_returns_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            run_backtest(
                returns_series=_RETURNS_40,
                window=20,
                initial_capital=_CAPITAL,
                asset_price_series=[],
                regime=_REGIME_RISK_ON,
                meta_uncertainty=_META_UNCERTAINTY,
            )
