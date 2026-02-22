import dataclasses
import sys

import pytest

from jarvis.core.risk_layer import (
    RiskNumericalError,
    RiskParameterConsistencyError,
    RiskParameters,
    RiskValidationError,
)


def _valid_rp_kwargs():
    return dict(
        max_position_pct_nav=0.05,
        max_gross_exposure_pct=1.5,
        max_drawdown_hard_stop=0.10,
        max_drawdown_soft_warn=0.05,
        volatility_target_ann=0.15,
        liquidity_haircut_floor=0.2,
        max_open_positions=10,
        kelly_fraction=0.25,
    )


class TestRiskParametersHappyPath:
    def test_construction_succeeds(self):
        rp = RiskParameters(**_valid_rp_kwargs())
        assert rp.kelly_fraction == 0.25

    def test_frozen(self):
        rp = RiskParameters(**_valid_rp_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            rp.kelly_fraction = 1.0  # type: ignore

    def test_max_position_pct_nav_at_one_valid(self):
        rp = RiskParameters(**{**_valid_rp_kwargs(), "max_position_pct_nav": 1.0})
        assert rp.max_position_pct_nav == 1.0

    def test_kelly_fraction_at_one_valid(self):
        rp = RiskParameters(**{**_valid_rp_kwargs(), "kelly_fraction": 1.0})
        assert rp.kelly_fraction == 1.0

    def test_max_open_positions_one_valid(self):
        rp = RiskParameters(**{**_valid_rp_kwargs(), "max_open_positions": 1})
        assert rp.max_open_positions == 1

    def test_soft_warn_just_below_hard_stop_valid(self):
        rp = RiskParameters(**{
            **_valid_rp_kwargs(),
            "max_drawdown_soft_warn": 0.0999,
            "max_drawdown_hard_stop": 0.10,
        })
        assert rp.max_drawdown_soft_warn < rp.max_drawdown_hard_stop

    def test_gross_exposure_above_one_valid(self):
        """Leverage > 100% is permitted."""
        rp = RiskParameters(**{**_valid_rp_kwargs(), "max_gross_exposure_pct": 3.0})
        assert rp.max_gross_exposure_pct == 3.0

    def test_liquidity_haircut_floor_at_one_valid(self):
        rp = RiskParameters(**{**_valid_rp_kwargs(), "liquidity_haircut_floor": 1.0})
        assert rp.liquidity_haircut_floor == 1.0


class TestRiskParametersFiniteCheck:
    @pytest.mark.parametrize("field", [
        "max_position_pct_nav",
        "max_gross_exposure_pct",
        "max_drawdown_hard_stop",
        "max_drawdown_soft_warn",
        "volatility_target_ann",
        "liquidity_haircut_floor",
        "kelly_fraction",
    ])
    def test_nan_raises_numerical_error(self, field):
        with pytest.raises(RiskNumericalError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), field: float("nan")})
        assert exc_info.value.field_name == field

    @pytest.mark.parametrize("field", [
        "max_position_pct_nav",
        "max_gross_exposure_pct",
        "volatility_target_ann",
    ])
    def test_inf_raises_numerical_error(self, field):
        with pytest.raises(RiskNumericalError):
            RiskParameters(**{**_valid_rp_kwargs(), field: float("inf")})


class TestRiskParametersRangeConstraints:
    def test_max_position_pct_nav_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), "max_position_pct_nav": 0.0})
        assert exc_info.value.field_name == "max_position_pct_nav"

    def test_max_position_pct_nav_above_one_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "max_position_pct_nav": 1.0001})

    def test_max_gross_exposure_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), "max_gross_exposure_pct": 0.0})
        assert exc_info.value.field_name == "max_gross_exposure_pct"

    def test_max_gross_exposure_negative_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "max_gross_exposure_pct": -1.0})

    @pytest.mark.parametrize("field", ["max_drawdown_hard_stop", "max_drawdown_soft_warn"])
    def test_drawdown_zero_raises(self, field):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), field: 0.0})
        assert exc_info.value.field_name == field

    @pytest.mark.parametrize("field", ["max_drawdown_hard_stop", "max_drawdown_soft_warn"])
    def test_drawdown_one_raises(self, field):
        """1.0 is excluded from the open interval (0, 1)."""
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), field: 1.0})

    def test_volatility_target_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), "volatility_target_ann": 0.0})
        assert exc_info.value.field_name == "volatility_target_ann"

    def test_volatility_target_negative_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "volatility_target_ann": -0.01})

    def test_liquidity_haircut_floor_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), "liquidity_haircut_floor": 0.0})
        assert exc_info.value.field_name == "liquidity_haircut_floor"

    def test_kelly_fraction_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), "kelly_fraction": 0.0})
        assert exc_info.value.field_name == "kelly_fraction"

    def test_kelly_fraction_above_one_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "kelly_fraction": 1.0001})


class TestRiskParametersIntConstraints:
    def test_max_open_positions_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            RiskParameters(**{**_valid_rp_kwargs(), "max_open_positions": 0})
        assert exc_info.value.field_name == "max_open_positions"

    def test_max_open_positions_negative_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "max_open_positions": -5})

    def test_max_open_positions_float_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "max_open_positions": 10.0})  # type: ignore

    def test_max_open_positions_bool_raises(self):
        with pytest.raises(RiskValidationError):
            RiskParameters(**{**_valid_rp_kwargs(), "max_open_positions": True})


class TestRiskParametersCrossField:
    """INV-RP-10: soft_warn < hard_stop."""

    @pytest.mark.parametrize("soft,hard,label", [
        (0.10, 0.10, "equal"),
        (0.11, 0.10, "soft greater than hard"),
        (0.99, 0.01, "wildly inverted"),
    ])
    def test_soft_gte_hard_raises(self, soft, hard, label):
        with pytest.raises(RiskParameterConsistencyError) as exc_info:
            RiskParameters(**{
                **_valid_rp_kwargs(),
                "max_drawdown_soft_warn": soft,
                "max_drawdown_hard_stop": hard,
            })
        exc = exc_info.value
        assert exc.field_a == "max_drawdown_soft_warn"
        assert exc.field_b == "max_drawdown_hard_stop"
        assert exc.value_a == soft
        assert exc.value_b == hard

    def test_soft_just_less_than_hard_valid(self):
        eps = sys.float_info.epsilon * 0.05
        rp = RiskParameters(**{
            **_valid_rp_kwargs(),
            "max_drawdown_soft_warn": 0.05 - eps,
            "max_drawdown_hard_stop": 0.05,
        })
        assert rp.max_drawdown_soft_warn < rp.max_drawdown_hard_stop
