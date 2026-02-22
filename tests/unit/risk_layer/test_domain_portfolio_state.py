import dataclasses

import pytest

from jarvis.core.risk_layer import (
    PortfolioState,
    RiskNumericalError,
    RiskParameterConsistencyError,
    RiskValidationError,
)


def _valid_portfolio_kwargs():
    return dict(
        nav=1_000_000.0,
        gross_exposure_usd=200_000.0,
        net_exposure_usd=-50_000.0,
        open_positions=5,
        peak_nav=1_200_000.0,
        realized_drawdown_pct=0.0,
        current_step=0,
    )


class TestPortfolioStateHappyPath:
    def test_construction_succeeds(self):
        pf = PortfolioState(**_valid_portfolio_kwargs())
        assert pf.nav == 1_000_000.0

    def test_frozen(self):
        pf = PortfolioState(**_valid_portfolio_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            pf.nav = 999.0  # type: ignore

    def test_zero_open_positions_valid(self):
        pf = PortfolioState(**{**_valid_portfolio_kwargs(), "open_positions": 0})
        assert pf.open_positions == 0

    def test_zero_current_step_valid(self):
        pf = PortfolioState(**{**_valid_portfolio_kwargs(), "current_step": 0})
        assert pf.current_step == 0

    def test_nav_equals_peak_nav_valid(self):
        pf = PortfolioState(**{
            **_valid_portfolio_kwargs(),
            "nav": 1_000_000.0,
            "peak_nav": 1_000_000.0,
        })
        assert pf.nav == pf.peak_nav

    def test_zero_gross_exposure_valid(self):
        pf = PortfolioState(**{**_valid_portfolio_kwargs(), "gross_exposure_usd": 0.0})
        assert pf.gross_exposure_usd == 0.0

    def test_negative_net_exposure_valid(self):
        pf = PortfolioState(**{**_valid_portfolio_kwargs(), "net_exposure_usd": -500_000.0})
        assert pf.net_exposure_usd == -500_000.0

    def test_drawdown_pct_at_maximum_boundary(self):
        """realized_drawdown_pct == 1.0 is permitted (complete wipeout)."""
        pf = PortfolioState(**{**_valid_portfolio_kwargs(), "realized_drawdown_pct": 1.0})
        assert pf.realized_drawdown_pct == 1.0


class TestPortfolioStateFloatValidation:
    @pytest.mark.parametrize("field,value", [
        ("nav",                   float("nan")),
        ("gross_exposure_usd",    float("nan")),
        ("net_exposure_usd",      float("inf")),
        ("peak_nav",              float("-inf")),
        ("realized_drawdown_pct", float("nan")),
    ])
    def test_nan_inf_raises_numerical_error(self, field, value):
        with pytest.raises(RiskNumericalError) as exc_info:
            PortfolioState(**{**_valid_portfolio_kwargs(), field: value})
        assert exc_info.value.field_name == field

    def test_nav_zero_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PortfolioState(**{**_valid_portfolio_kwargs(), "nav": 0.0})
        assert exc_info.value.field_name == "nav"

    def test_nav_negative_raises(self):
        with pytest.raises(RiskValidationError):
            PortfolioState(**{**_valid_portfolio_kwargs(), "nav": -1.0})

    def test_gross_exposure_negative_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PortfolioState(**{**_valid_portfolio_kwargs(), "gross_exposure_usd": -0.01})
        assert exc_info.value.field_name == "gross_exposure_usd"

    def test_peak_nav_zero_raises(self):
        with pytest.raises(RiskValidationError):
            PortfolioState(**{**_valid_portfolio_kwargs(), "peak_nav": 0.0})

    def test_drawdown_pct_above_one_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PortfolioState(**{**_valid_portfolio_kwargs(), "realized_drawdown_pct": 1.0001})
        assert exc_info.value.field_name == "realized_drawdown_pct"

    def test_drawdown_pct_negative_raises(self):
        with pytest.raises(RiskValidationError):
            PortfolioState(**{**_valid_portfolio_kwargs(), "realized_drawdown_pct": -0.001})


class TestPortfolioStateIntValidation:
    def test_negative_open_positions_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PortfolioState(**{**_valid_portfolio_kwargs(), "open_positions": -1})
        assert exc_info.value.field_name == "open_positions"

    def test_float_open_positions_raises(self):
        with pytest.raises(RiskValidationError):
            PortfolioState(**{**_valid_portfolio_kwargs(), "open_positions": 1.0})  # type: ignore

    def test_bool_open_positions_raises(self):
        with pytest.raises(RiskValidationError):
            PortfolioState(**{**_valid_portfolio_kwargs(), "open_positions": True})

    def test_negative_current_step_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PortfolioState(**{**_valid_portfolio_kwargs(), "current_step": -1})
        assert exc_info.value.field_name == "current_step"


class TestPortfolioStateCrossField:
    """INV-PF-09: peak_nav >= nav."""

    def test_peak_nav_less_than_nav_raises(self):
        with pytest.raises(RiskParameterConsistencyError) as exc_info:
            PortfolioState(**{
                **_valid_portfolio_kwargs(),
                "nav": 1_100_000.0,
                "peak_nav": 1_000_000.0,
            })
        exc = exc_info.value
        assert exc.field_a == "peak_nav"
        assert exc.field_b == "nav"

    def test_peak_nav_equal_nav_valid(self):
        pf = PortfolioState(**{
            **_valid_portfolio_kwargs(),
            "nav": 1_000_000.0,
            "peak_nav": 1_000_000.0,
        })
        assert pf.peak_nav == pf.nav

    def test_peak_nav_greater_than_nav_valid(self):
        pf = PortfolioState(**{
            **_valid_portfolio_kwargs(),
            "nav": 900_000.0,
            "peak_nav": 1_000_000.0,
        })
        assert pf.peak_nav > pf.nav
