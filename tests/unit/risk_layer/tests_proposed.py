# =============================================================================
# JARVIS v6.1.0 -- PHASE 7: RISK LAYER
# File:   tests/risk_layer/unit/test_exceptions.py
#         tests/risk_layer/unit/test_domain.py
#         tests/risk_layer/conftest.py
#
# PROPOSED UNIT TEST STRUCTURE
# Coverage target: >= 90% line, 100% branch on all validation paths.
# =============================================================================
#
# DIRECTORY LAYOUT
# ----------------
#   tests/
#     risk_layer/
#       __init__.py
#       conftest.py                        <-- shared fixtures
#       unit/
#         __init__.py
#         test_exceptions.py              <-- exceptions.py coverage
#         test_domain_position_spec.py    <-- PositionSpec coverage
#         test_domain_portfolio_state.py  <-- PortfolioState coverage
#         test_domain_risk_parameters.py  <-- RiskParameters coverage
#         test_domain_enums.py            <-- Side, RiskVerdict
#
# =============================================================================


# =============================================================================
# FILE: tests/risk_layer/conftest.py
# =============================================================================

"""
Shared fixtures for Phase 7 unit tests.

All fixtures produce minimal-valid instances -- i.e. the simplest inputs
that pass all validation without triggering any exception. Tests deviate
from these baselines by overriding individual fields via dataclasses.replace()
or by constructing directly with the specific invalid value under test.

No mocks. No monkeypatching of domain objects.
All fixtures are pure functions (no state, no IO).
"""

import pytest
from jarvis.risk_layer.domain import (
    Side,
    PositionSpec,
    PortfolioState,
    RiskParameters,
)


@pytest.fixture
def valid_position_spec() -> PositionSpec:
    """Minimal valid PositionSpec. All fields at their smallest valid values."""
    return PositionSpec(
        symbol="BTC-USD",
        asset_class="crypto",
        side=Side.LONG,
        entry_price=50_000.0,
        current_price=51_000.0,
        quantity=0.1,
        max_position_usd=10_000.0,
    )


@pytest.fixture
def valid_portfolio_state() -> PortfolioState:
    """Minimal valid PortfolioState with no drawdown (nav == peak_nav)."""
    return PortfolioState(
        nav=1_000_000.0,
        gross_exposure_usd=200_000.0,
        net_exposure_usd=150_000.0,
        open_positions=3,
        peak_nav=1_000_000.0,
        realized_drawdown_pct=0.0,
        current_step=100,
    )


@pytest.fixture
def valid_portfolio_state_with_drawdown() -> PortfolioState:
    """PortfolioState with a 5% drawdown from peak."""
    return PortfolioState(
        nav=950_000.0,
        gross_exposure_usd=100_000.0,
        net_exposure_usd=100_000.0,
        open_positions=1,
        peak_nav=1_000_000.0,
        realized_drawdown_pct=0.05,
        current_step=500,
    )


@pytest.fixture
def valid_risk_parameters() -> RiskParameters:
    """
    Conservative but valid RiskParameters.
    soft_warn=0.05, hard_stop=0.10 (5% gap -- clear ordering).
    """
    return RiskParameters(
        max_position_pct_nav=0.05,
        max_gross_exposure_pct=1.5,
        max_drawdown_hard_stop=0.10,
        max_drawdown_soft_warn=0.05,
        volatility_target_ann=0.15,
        liquidity_haircut_floor=0.2,
        max_open_positions=10,
        kelly_fraction=0.25,
    )


# =============================================================================
# FILE: tests/risk_layer/unit/test_exceptions.py
# =============================================================================

"""
Unit tests for jarvis/risk_layer/exceptions.py.

Coverage objectives:
  - RiskError: construction, attributes, __repr__, __eq__.
  - RiskNumericalError: message format, field_name, value, empty field_name rejection.
  - RiskValidationError: message format, constraint attribute, edge cases.
  - RiskParameterConsistencyError: message format, both field attributes,
    invariant_description, __repr__, __eq__.
  - Exception hierarchy: isinstance checks.
  - str(exception) matches expected message for each type.

Test count target: ~40 test functions (100% branch on all __init__ paths).
"""

import math
import pytest
from jarvis.risk_layer.exceptions import (
    RiskError,
    RiskNumericalError,
    RiskValidationError,
    RiskParameterConsistencyError,
)


class TestRiskErrorBase:
    """RiskError base class -- construction and attributes."""

    def test_construction_stores_message(self):
        exc = RiskError(message="test message")
        assert exc.message == "test message"
        assert str(exc) == "test message"

    def test_construction_default_field_name_is_empty_string(self):
        exc = RiskError(message="msg")
        assert exc.field_name == ""

    def test_construction_default_value_is_none(self):
        exc = RiskError(message="msg")
        assert exc.value is None

    def test_construction_with_all_args(self):
        exc = RiskError(message="msg", field_name="foo", value=42)
        assert exc.field_name == "foo"
        assert exc.value == 42

    def test_is_exception_subclass(self):
        assert issubclass(RiskError, Exception)

    def test_empty_message_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty string"):
            RiskError(message="")

    def test_non_string_message_raises_value_error(self):
        with pytest.raises(ValueError):
            RiskError(message=None)  # type: ignore[arg-type]

    def test_non_string_field_name_raises_value_error(self):
        with pytest.raises(ValueError):
            RiskError(message="msg", field_name=123)  # type: ignore[arg-type]

    def test_equality_same_type_same_values(self):
        a = RiskError(message="msg", field_name="f", value=1.0)
        b = RiskError(message="msg", field_name="f", value=1.0)
        assert a == b

    def test_equality_different_message(self):
        a = RiskError(message="msg1", field_name="f", value=1.0)
        b = RiskError(message="msg2", field_name="f", value=1.0)
        assert a != b

    def test_equality_different_type(self):
        a = RiskError(message="msg")
        assert a != "not an exception"

    def test_repr_contains_class_name(self):
        exc = RiskError(message="msg", field_name="f", value=0)
        assert "RiskError" in repr(exc)
        assert "field_name" in repr(exc)


class TestRiskNumericalError:
    """RiskNumericalError -- NaN/Inf field violations."""

    def test_nan_message_format(self):
        exc = RiskNumericalError(field_name="nav", value=float("nan"))
        assert "nav" in exc.message
        assert "non-finite" in exc.message
        assert "NaN" in exc.message or "nan" in exc.message.lower()

    def test_positive_inf_message_format(self):
        exc = RiskNumericalError(field_name="price", value=float("inf"))
        assert "price" in exc.message
        assert "inf" in exc.message.lower()

    def test_negative_inf_message_format(self):
        exc = RiskNumericalError(field_name="qty", value=float("-inf"))
        assert "qty" in exc.message

    def test_field_name_attribute(self):
        exc = RiskNumericalError(field_name="entry_price", value=float("nan"))
        assert exc.field_name == "entry_price"

    def test_value_attribute_is_nan(self):
        val = float("nan")
        exc = RiskNumericalError(field_name="f", value=val)
        assert math.isnan(exc.value)

    def test_value_attribute_is_inf(self):
        exc = RiskNumericalError(field_name="f", value=float("inf"))
        assert exc.value == float("inf")

    def test_is_risk_error_subclass(self):
        exc = RiskNumericalError(field_name="f", value=float("nan"))
        assert isinstance(exc, RiskError)

    def test_is_exception_subclass(self):
        exc = RiskNumericalError(field_name="f", value=float("nan"))
        assert isinstance(exc, Exception)

    def test_empty_field_name_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            RiskNumericalError(field_name="", value=float("nan"))

    def test_message_is_deterministic(self):
        """Identical inputs must produce identical messages."""
        a = RiskNumericalError(field_name="nav", value=float("inf"))
        b = RiskNumericalError(field_name="nav", value=float("inf"))
        assert a.message == b.message

    def test_can_be_raised_and_caught_as_risk_error(self):
        with pytest.raises(RiskError):
            raise RiskNumericalError(field_name="f", value=float("nan"))

    def test_str_equals_message(self):
        exc = RiskNumericalError(field_name="nav", value=float("nan"))
        assert str(exc) == exc.message


class TestRiskValidationError:
    """RiskValidationError -- range / sign / type / membership violations."""

    def test_message_contains_field_name(self):
        exc = RiskValidationError(field_name="nav", value=-1.0, constraint="must be > 0")
        assert "nav" in exc.message

    def test_message_contains_value(self):
        exc = RiskValidationError(field_name="nav", value=-1.0, constraint="must be > 0")
        assert "-1.0" in exc.message or repr(-1.0) in exc.message

    def test_message_contains_constraint(self):
        exc = RiskValidationError(field_name="nav", value=-1.0, constraint="must be > 0")
        assert "must be > 0" in exc.message

    def test_constraint_attribute_stored(self):
        exc = RiskValidationError(field_name="f", value=99, constraint="must be in [0,1]")
        assert exc.constraint == "must be in [0,1]"

    def test_field_name_attribute(self):
        exc = RiskValidationError(field_name="qty", value=0.0, constraint="must be > 0")
        assert exc.field_name == "qty"

    def test_value_attribute(self):
        exc = RiskValidationError(field_name="f", value="bad", constraint="must be ascii")
        assert exc.value == "bad"

    def test_is_risk_error_subclass(self):
        exc = RiskValidationError(field_name="f", value=0, constraint="c")
        assert isinstance(exc, RiskError)

    def test_empty_field_name_raises(self):
        with pytest.raises(ValueError):
            RiskValidationError(field_name="", value=0, constraint="c")

    def test_empty_constraint_raises(self):
        with pytest.raises(ValueError):
            RiskValidationError(field_name="f", value=0, constraint="")

    def test_non_string_constraint_raises(self):
        with pytest.raises(ValueError):
            RiskValidationError(field_name="f", value=0, constraint=None)  # type: ignore

    def test_message_is_deterministic(self):
        a = RiskValidationError(field_name="nav", value=-5.0, constraint="must be > 0")
        b = RiskValidationError(field_name="nav", value=-5.0, constraint="must be > 0")
        assert a.message == b.message

    def test_str_equals_message(self):
        exc = RiskValidationError(field_name="f", value=0, constraint="c")
        assert str(exc) == exc.message

    def test_can_be_raised_and_caught_as_risk_error(self):
        with pytest.raises(RiskError):
            raise RiskValidationError(field_name="f", value=0, constraint="c")


class TestRiskParameterConsistencyError:
    """RiskParameterConsistencyError -- cross-field violations."""

    def test_message_contains_both_field_names(self):
        exc = RiskParameterConsistencyError(
            field_a="soft_warn", value_a=0.10,
            field_b="hard_stop", value_b=0.05,
            invariant_description="soft_warn must be < hard_stop",
        )
        assert "soft_warn" in exc.message
        assert "hard_stop" in exc.message

    def test_message_contains_both_values(self):
        exc = RiskParameterConsistencyError(
            field_a="soft_warn", value_a=0.10,
            field_b="hard_stop", value_b=0.05,
            invariant_description="soft_warn must be < hard_stop",
        )
        assert repr(0.10) in exc.message
        assert repr(0.05) in exc.message

    def test_message_contains_invariant_description(self):
        desc = "soft_warn must be strictly less than hard_stop"
        exc = RiskParameterConsistencyError(
            field_a="a", value_a=1, field_b="b", value_b=0,
            invariant_description=desc,
        )
        assert desc in exc.message

    def test_field_a_attribute(self):
        exc = RiskParameterConsistencyError(
            field_a="peak_nav", value_a=900_000.0,
            field_b="nav", value_b=1_000_000.0,
            invariant_description="peak_nav must be >= nav",
        )
        assert exc.field_a == "peak_nav"

    def test_field_b_attribute(self):
        exc = RiskParameterConsistencyError(
            field_a="peak_nav", value_a=900_000.0,
            field_b="nav", value_b=1_000_000.0,
            invariant_description="peak_nav must be >= nav",
        )
        assert exc.field_b == "nav"

    def test_value_a_attribute(self):
        exc = RiskParameterConsistencyError(
            field_a="a", value_a=0.10, field_b="b", value_b=0.05,
            invariant_description="desc",
        )
        assert exc.value_a == 0.10

    def test_value_b_attribute(self):
        exc = RiskParameterConsistencyError(
            field_a="a", value_a=0.10, field_b="b", value_b=0.05,
            invariant_description="desc",
        )
        assert exc.value_b == 0.05

    def test_invariant_description_attribute(self):
        exc = RiskParameterConsistencyError(
            field_a="a", value_a=1, field_b="b", value_b=2,
            invariant_description="a must be > b",
        )
        assert exc.invariant_description == "a must be > b"

    def test_base_field_name_is_field_a(self):
        """Base class field_name should be the primary (field_a) field."""
        exc = RiskParameterConsistencyError(
            field_a="soft_warn", value_a=0.10,
            field_b="hard_stop", value_b=0.05,
            invariant_description="desc",
        )
        assert exc.field_name == "soft_warn"

    def test_is_risk_error_subclass(self):
        exc = RiskParameterConsistencyError(
            field_a="a", value_a=1, field_b="b", value_b=2,
            invariant_description="desc",
        )
        assert isinstance(exc, RiskError)

    def test_empty_field_a_raises(self):
        with pytest.raises(ValueError):
            RiskParameterConsistencyError(
                field_a="", value_a=1, field_b="b", value_b=2,
                invariant_description="desc",
            )

    def test_empty_field_b_raises(self):
        with pytest.raises(ValueError):
            RiskParameterConsistencyError(
                field_a="a", value_a=1, field_b="", value_b=2,
                invariant_description="desc",
            )

    def test_empty_invariant_description_raises(self):
        with pytest.raises(ValueError):
            RiskParameterConsistencyError(
                field_a="a", value_a=1, field_b="b", value_b=2,
                invariant_description="",
            )

    def test_equality(self):
        kwargs = dict(
            field_a="a", value_a=1, field_b="b", value_b=2,
            invariant_description="desc",
        )
        assert (
            RiskParameterConsistencyError(**kwargs)
            == RiskParameterConsistencyError(**kwargs)
        )

    def test_repr_contains_all_fields(self):
        exc = RiskParameterConsistencyError(
            field_a="a", value_a=1, field_b="b", value_b=2,
            invariant_description="desc",
        )
        r = repr(exc)
        assert "field_a" in r
        assert "field_b" in r
        assert "invariant_description" in r

    def test_message_is_deterministic(self):
        kwargs = dict(
            field_a="a", value_a=0.10, field_b="b", value_b=0.05,
            invariant_description="a must be < b",
        )
        a = RiskParameterConsistencyError(**kwargs)
        b = RiskParameterConsistencyError(**kwargs)
        assert a.message == b.message


class TestExceptionHierarchy:
    """Verify the inheritance chain."""

    def test_numerical_is_risk_error(self):
        assert issubclass(RiskNumericalError, RiskError)

    def test_validation_is_risk_error(self):
        assert issubclass(RiskValidationError, RiskError)

    def test_consistency_is_risk_error(self):
        assert issubclass(RiskParameterConsistencyError, RiskError)

    def test_all_are_exceptions(self):
        for cls in (RiskError, RiskNumericalError, RiskValidationError,
                    RiskParameterConsistencyError):
            assert issubclass(cls, Exception)

    def test_numerical_is_not_validation(self):
        exc = RiskNumericalError(field_name="f", value=float("nan"))
        assert not isinstance(exc, RiskValidationError)

    def test_validation_is_not_numerical(self):
        exc = RiskValidationError(field_name="f", value=0, constraint="c")
        assert not isinstance(exc, RiskNumericalError)


# =============================================================================
# FILE: tests/risk_layer/unit/test_domain_enums.py
# =============================================================================

"""
Tests for Side and RiskVerdict enumerations.
"""

from jarvis.risk_layer.domain import Side, RiskVerdict


class TestSide:
    def test_long_value(self):
        assert Side.LONG == "LONG"

    def test_short_value(self):
        assert Side.SHORT == "SHORT"

    def test_exactly_two_members(self):
        assert set(Side) == {Side.LONG, Side.SHORT}

    def test_str_subclass(self):
        assert isinstance(Side.LONG, str)

    def test_from_string(self):
        assert Side("LONG") is Side.LONG
        assert Side("SHORT") is Side.SHORT

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            Side("INVALID")


class TestRiskVerdict:
    def test_all_five_members_present(self):
        members = {v.value for v in RiskVerdict}
        assert members == {"APPROVE", "REDUCE", "HOLD", "HALT", "REJECT"}

    def test_str_subclass(self):
        assert isinstance(RiskVerdict.APPROVE, str)


# =============================================================================
# FILE: tests/risk_layer/unit/test_domain_position_spec.py
# =============================================================================

"""
Tests for PositionSpec frozen dataclass.

Covers every invariant in INV-PS-01 through INV-PS-08.
Each invalid value is tested in isolation to confirm fail-fast ordering.
"""

import math
import dataclasses
import pytest
from jarvis.risk_layer.domain import Side, PositionSpec
from jarvis.risk_layer.exceptions import (
    RiskNumericalError,
    RiskValidationError,
)

# --- Helpers ---

def _valid_kwargs():
    return dict(
        symbol="ETH-USD",
        asset_class="crypto",
        side=Side.SHORT,
        entry_price=2000.0,
        current_price=1900.0,
        quantity=5.0,
        max_position_usd=20_000.0,
    )


class TestPositionSpecHappyPath:
    def test_construction_succeeds(self):
        spec = PositionSpec(**_valid_kwargs())
        assert spec.symbol == "ETH-USD"
        assert spec.side is Side.SHORT

    def test_frozen(self):
        spec = PositionSpec(**_valid_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.quantity = 99.0  # type: ignore

    def test_all_asset_classes_accepted(self):
        from jarvis.core.data_layer import VALID_ASSET_CLASSES
        for ac in VALID_ASSET_CLASSES:
            kwargs = _valid_kwargs()
            kwargs["asset_class"] = ac
            spec = PositionSpec(**kwargs)
            assert spec.asset_class == ac

    def test_both_sides_accepted(self):
        for side in Side:
            kwargs = _valid_kwargs()
            kwargs["side"] = side
            spec = PositionSpec(**kwargs)
            assert spec.side is side


class TestPositionSpecSymbolValidation:
    def test_empty_symbol_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), "symbol": ""})
        assert exc_info.value.field_name == "symbol"

    def test_non_string_symbol_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "symbol": 123})  # type: ignore

    def test_non_ascii_symbol_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "symbol": "BTC\u20ac"})


class TestPositionSpecAssetClassValidation:
    def test_unknown_asset_class_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), "asset_class": "real_estate"})
        assert exc_info.value.field_name == "asset_class"

    def test_empty_asset_class_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "asset_class": ""})

    def test_uppercase_asset_class_raises(self):
        # VALID_ASSET_CLASSES are lowercase; "CRYPTO" must not sneak through
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "asset_class": "CRYPTO"})


class TestPositionSpecSideValidation:
    def test_string_side_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), "side": "LONG"})  # type: ignore
        assert exc_info.value.field_name == "side"

    def test_none_side_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "side": None})  # type: ignore


class TestPositionSpecFloatValidation:
    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd"
    ])
    def test_nan_raises_numerical_error(self, field):
        with pytest.raises(RiskNumericalError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), field: float("nan")})
        assert exc_info.value.field_name == field

    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd"
    ])
    def test_inf_raises_numerical_error(self, field):
        with pytest.raises(RiskNumericalError):
            PositionSpec(**{**_valid_kwargs(), field: float("inf")})

    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd"
    ])
    def test_zero_raises_validation_error(self, field):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), field: 0.0})
        assert exc_info.value.field_name == field

    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd"
    ])
    def test_negative_raises_validation_error(self, field):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), field: -0.001})

    def test_very_small_positive_quantity_is_valid(self):
        spec = PositionSpec(**{**_valid_kwargs(), "quantity": 1e-10})
        assert spec.quantity == 1e-10

    def test_very_large_price_is_valid(self):
        spec = PositionSpec(**{**_valid_kwargs(), "entry_price": 1e15})
        assert spec.entry_price == 1e15


# =============================================================================
# FILE: tests/risk_layer/unit/test_domain_portfolio_state.py
# =============================================================================

"""
Tests for PortfolioState frozen dataclass.
Covers INV-PF-01 through INV-PF-09, including the cross-field invariant.
"""

import dataclasses
import pytest
from jarvis.risk_layer.domain import PortfolioState
from jarvis.risk_layer.exceptions import (
    RiskNumericalError,
    RiskValidationError,
    RiskParameterConsistencyError,
)


def _valid_portfolio_kwargs():
    return dict(
        nav=1_000_000.0,
        gross_exposure_usd=200_000.0,
        net_exposure_usd=-50_000.0,   # negative is valid (net short)
        open_positions=5,
        peak_nav=1_200_000.0,
        realized_drawdown_pct=0.0,    # 0 is valid
        current_step=0,               # 0 is valid
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
        """No drawdown: nav == peak_nav is permitted."""
        pf = PortfolioState(**{**_valid_portfolio_kwargs(),
                                "nav": 1_000_000.0, "peak_nav": 1_000_000.0})
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
        # bool is a subclass of int; booleans must be rejected
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
                "peak_nav": 1_000_000.0,  # peak < nav -- invalid
            })
        exc = exc_info.value
        assert exc.field_a == "peak_nav"
        assert exc.field_b == "nav"

    def test_peak_nav_equal_nav_valid(self):
        """Equality is the no-drawdown case -- must succeed."""
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


# =============================================================================
# FILE: tests/risk_layer/unit/test_domain_risk_parameters.py
# =============================================================================

"""
Tests for RiskParameters frozen dataclass.
Covers INV-RP-01 through INV-RP-10, including the cross-field soft/hard
drawdown invariant.

Key boundary conditions tested:
  - soft_warn == hard_stop raises (not strictly less)
  - soft_warn > hard_stop raises
  - soft_warn just below hard_stop succeeds
  - max_position_pct_nav == 1.0 is valid (at most 100% of NAV)
  - max_position_pct_nav == 0.0 is invalid (must be strictly positive)
  - max_drawdown_hard_stop == 0.0 and == 1.0 are both invalid (open interval)
  - kelly_fraction == 1.0 is valid (full Kelly)
  - max_open_positions == 1 is valid (minimum)
  - max_open_positions == 0 raises
"""

import dataclasses
import pytest
from jarvis.risk_layer.domain import RiskParameters
from jarvis.risk_layer.exceptions import (
    RiskNumericalError,
    RiskValidationError,
    RiskParameterConsistencyError,
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
        rp = RiskParameters(**{**_valid_rp_kwargs(),
                                "max_drawdown_soft_warn": 0.0999,
                                "max_drawdown_hard_stop": 0.10})
        assert rp.max_drawdown_soft_warn < rp.max_drawdown_hard_stop

    def test_gross_exposure_above_one_valid(self):
        """Leverage > 100% is permitted."""
        rp = RiskParameters(**{**_valid_rp_kwargs(), "max_gross_exposure_pct": 3.0})
        assert rp.max_gross_exposure_pct == 3.0


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

    def test_liquidity_haircut_floor_at_one_valid(self):
        rp = RiskParameters(**{**_valid_rp_kwargs(), "liquidity_haircut_floor": 1.0})
        assert rp.liquidity_haircut_floor == 1.0

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
        # 0.0499... < 0.05: the smallest representable gap
        import sys
        eps = sys.float_info.epsilon * 0.05
        rp = RiskParameters(**{
            **_valid_rp_kwargs(),
            "max_drawdown_soft_warn": 0.05 - eps,
            "max_drawdown_hard_stop": 0.05,
        })
        assert rp.max_drawdown_soft_warn < rp.max_drawdown_hard_stop


# =============================================================================
# COVERAGE STRATEGY NOTES (not executable -- documentation only)
# =============================================================================

"""
COVERAGE TARGETS
----------------
exceptions.py:
  Target: 100% line, 100% branch.
  The only branches are the guard conditions in each __init__.
  Every guard has a passing test (valid input) and a failing test (invalid input).

domain.py:
  Target: >= 90% line, 100% branch on all __post_init__ methods.

  V1 gate (finiteness):
    Each float field gets a NaN test and an Inf test.
    This ensures the math.isfinite branch fires at least twice per field.

  V2 gate (range/sign):
    Boundary values are always tested:
      - At the boundary (e.g. exactly 0.0 when > 0.0 is required)
      - Just inside the boundary (e.g. 1e-15 when > 0.0 is required)
      - Just outside the boundary (e.g. -1e-15 when >= 0.0 is required)
      - Deep inside invalid territory (e.g. -999.0)

  V3 gate (enum/set):
    Unknown values, empty strings, wrong types.

  V4 gate (cross-field):
    All three sub-cases: equal, inverted, and just-valid.

PARAMETRIZE STRATEGY
--------------------
Use @pytest.mark.parametrize for:
  - All float fields sharing the same constraint (e.g. all "finite, > 0" fields
    in PositionSpec share the same two parametrize tests).
  - drawdown boundary values (the truth table in conftest design doc).
  - All valid asset classes (verified in a single parametrize loop).

This keeps the test count manageable (~150 test functions total) while
achieving complete branch coverage.

MUTATION TESTING
----------------
After the standard suite passes at >= 90%, run:
  mutmut run --paths-to-mutate=jarvis/risk_layer/exceptions.py,jarvis/risk_layer/domain.py
Target mutation score: >= 85%.
Focus areas: comparison operators (< vs <=, > vs >=), string formatting,
field_name attribute assignment, cross-field ordering.
"""
