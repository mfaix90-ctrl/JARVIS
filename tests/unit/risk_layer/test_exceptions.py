import math

import pytest

from jarvis.core.risk_layer import (
    RiskError,
    RiskNumericalError,
    RiskParameterConsistencyError,
    RiskValidationError,
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
        for cls in (
            RiskError,
            RiskNumericalError,
            RiskValidationError,
            RiskParameterConsistencyError,
        ):
            assert issubclass(cls, Exception)

    def test_numerical_is_not_validation(self):
        exc = RiskNumericalError(field_name="f", value=float("nan"))
        assert not isinstance(exc, RiskValidationError)

    def test_validation_is_not_numerical(self):
        exc = RiskValidationError(field_name="f", value=0, constraint="c")
        assert not isinstance(exc, RiskNumericalError)
