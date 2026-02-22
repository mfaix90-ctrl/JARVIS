# =============================================================================
# JARVIS v6.1.0 -- PHASE 7: RISK & CAPITAL MANAGEMENT LAYER
# File:   jarvis/risk_layer/exceptions.py
# Authority: JARVIS FAS v6.1.0 -- Phase 7, Risk Engine
# =============================================================================
#
# SCOPE
# -----
# Defines the exception hierarchy for the Risk & Capital Management Layer.
# All exceptions are pure value objects: no side effects, no logging,
# no external references, no I/O of any kind.
#
# EXCEPTION HIERARCHY
# -------------------
#   RiskError(Exception)                         -- base; never raised directly
#     RiskNumericalError(RiskError)              -- NaN / Inf in a numeric field
#     RiskValidationError(RiskError)             -- range / sign / type violation
#     RiskParameterConsistencyError(RiskError)   -- cross-field logic violation
#
# DETERMINISM GUARANTEES
# ----------------------
# DET-01  No stochastic operations.
# DET-02  All message content is derived exclusively from constructor arguments.
# DET-03  No side effects. Exception construction is a pure value operation.
# DET-04  No module-level mutable state.
# DET-05  No datetime / time / random / uuid.
# DET-06  No file I/O, no network I/O.
#
# PROHIBITED ACTIONS CONFIRMED ABSENT
# ------------------------------------
#   No logging module
#   No numpy / scipy
#   No datetime.now() / time.time()
#   No random / secrets / uuid
#   No file IO
#   No domain imports (exceptions must remain leaf dependencies)
#
# MESSAGE CONTRACT
# ----------------
# Every exception message is:
#   - Deterministic: identical inputs -> identical message string.
#   - Explicit: field name and violating value always included.
#   - ASCII-safe: no Unicode outside the basic Latin block.
#   - Non-empty: always contains enough context to identify the violation site.
#
# =============================================================================

from __future__ import annotations

from typing import Any


# =============================================================================
# BASE EXCEPTION
# =============================================================================

class RiskError(Exception):
    """
    Base class for all Risk & Capital Management Layer exceptions.

    Never raised directly. Use a concrete subclass.

    Attributes:
        field_name:  Name of the offending field, or empty string if not
                     applicable (e.g. cross-field consistency errors report
                     both fields via the message instead).
        value:       The offending value at the time of validation,
                     or None if the violation is relational rather than
                     field-local.
        message:     Human-readable description of the violation.
                     Always non-empty. Always deterministic.
    """

    def __init__(
        self,
        message:    str,
        field_name: str = "",
        value:      Any = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError(
                "RiskError: message must be a non-empty string"
            )
        if not isinstance(field_name, str):
            raise ValueError(
                "RiskError: field_name must be a string"
            )
        super().__init__(message)
        self.field_name: str = field_name
        self.value:      Any = value
        self.message:    str = message

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(field_name=" + repr(self.field_name)
            + ", value=" + repr(self.value)
            + ", message=" + repr(self.message)
            + ")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RiskError):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.field_name == other.field_name
            and self.value == other.value
            and self.message == other.message
        )


# =============================================================================
# CONCRETE EXCEPTIONS
# =============================================================================

class RiskNumericalError(RiskError):
    """
    Raised when a numeric field contains NaN or Inf.

    This is the first validation gate. NaN and Inf are unconditionally
    rejected; no downstream validation, clipping, or recovery is attempted
    after this error is raised.

    Message format:
        "RiskNumericalError: field '<field_name>' contains non-finite
         value: <value>. NaN and Inf are not permitted."

    Args:
        field_name:  Name of the field that contains the non-finite value.
                     Must be a non-empty string.
        value:       The non-finite value (float NaN, float Inf, or
                     float -Inf). Stored on the exception for programmatic
                     inspection.

    Raises:
        ValueError if field_name is empty.
    """

    def __init__(self, field_name: str, value: float) -> None:
        if not field_name:
            raise ValueError(
                "RiskNumericalError: field_name must be a non-empty string"
            )
        message = (
            "RiskNumericalError: field '"
            + field_name
            + "' contains non-finite value: "
            + repr(value)
            + ". NaN and Inf are not permitted."
        )
        super().__init__(message=message, field_name=field_name, value=value)


class RiskValidationError(RiskError):
    """
    Raised when a field value violates a range, sign, type, or membership
    constraint.

    This covers:
      - Values outside a required numeric range (e.g. pct not in [0.0, 1.0]).
      - Non-positive values where strictly positive is required.
      - Enum / set membership failures (unknown asset_class, unknown Side).
      - Integer constraint failures (e.g. open_positions < 0).

    Message format:
        "RiskValidationError: field '<field_name>' violates constraint
         '<constraint>': got <value>."

    Args:
        field_name:  Name of the offending field. Must be non-empty.
        value:       The offending value.
        constraint:  Human-readable constraint description.
                     Examples:
                       "must be > 0"
                       "must be in (0.0, 1.0]"
                       "must be in VALID_ASSET_CLASSES"
                       "must be a valid Side enum member"
                     Must be non-empty.

    Raises:
        ValueError if field_name or constraint is empty.
    """

    def __init__(
        self,
        field_name:  str,
        value:       Any,
        constraint:  str,
    ) -> None:
        if not field_name:
            raise ValueError(
                "RiskValidationError: field_name must be a non-empty string"
            )
        if not isinstance(constraint, str) or not constraint:
            raise ValueError(
                "RiskValidationError: constraint must be a non-empty string"
            )
        message = (
            "RiskValidationError: field '"
            + field_name
            + "' violates constraint '"
            + constraint
            + "': got "
            + repr(value)
            + "."
        )
        super().__init__(message=message, field_name=field_name, value=value)
        self.constraint: str = constraint


class RiskParameterConsistencyError(RiskError):
    """
    Raised when two or more fields are individually valid but together
    violate a cross-field consistency invariant.

    This covers:
      - max_drawdown_soft_warn >= max_drawdown_hard_stop
      - peak_nav < nav (portfolio state incoherence)
      - Any other relational constraint between two distinct fields.

    Message format:
        "RiskParameterConsistencyError: cross-field invariant violated --
         <invariant_description>. Field '<field_a>' = <value_a>,
         field '<field_b>' = <value_b>."

    Args:
        field_a:               Name of the first field involved.
        value_a:               Value of the first field.
        field_b:               Name of the second field involved.
        value_b:               Value of the second field.
        invariant_description: Human-readable description of the required
                               relationship between the two fields.
                               Example:
                                 "max_drawdown_soft_warn must be strictly
                                  less than max_drawdown_hard_stop"
                               Must be non-empty.

    Raises:
        ValueError if field_a, field_b, or invariant_description is empty.

    Note on field_name / value attributes:
        The base class field_name is set to field_a and value to value_a.
        field_b and value_b are available as additional attributes.
        This is a pragmatic choice: callers that handle RiskError generically
        see the primary offending field; callers that handle
        RiskParameterConsistencyError specifically can inspect both.
    """

    def __init__(
        self,
        field_a:               str,
        value_a:               Any,
        field_b:               str,
        value_b:               Any,
        invariant_description: str,
    ) -> None:
        if not field_a:
            raise ValueError(
                "RiskParameterConsistencyError: field_a must be non-empty"
            )
        if not field_b:
            raise ValueError(
                "RiskParameterConsistencyError: field_b must be non-empty"
            )
        if not isinstance(invariant_description, str) or not invariant_description:
            raise ValueError(
                "RiskParameterConsistencyError: invariant_description must be non-empty"
            )
        message = (
            "RiskParameterConsistencyError: cross-field invariant violated -- "
            + invariant_description
            + ". Field '"
            + field_a
            + "' = "
            + repr(value_a)
            + ", field '"
            + field_b
            + "' = "
            + repr(value_b)
            + "."
        )
        super().__init__(message=message, field_name=field_a, value=value_a)
        self.field_a:               str = field_a
        self.value_a:               Any = value_a
        self.field_b:               str = field_b
        self.value_b:               Any = value_b
        self.invariant_description: str = invariant_description

    def __repr__(self) -> str:
        return (
            "RiskParameterConsistencyError("
            + "field_a=" + repr(self.field_a)
            + ", value_a=" + repr(self.value_a)
            + ", field_b=" + repr(self.field_b)
            + ", value_b=" + repr(self.value_b)
            + ", invariant_description=" + repr(self.invariant_description)
            + ")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RiskParameterConsistencyError):
            return NotImplemented
        return (
            self.field_a               == other.field_a
            and self.value_a           == other.value_a
            and self.field_b           == other.field_b
            and self.value_b           == other.value_b
            and self.invariant_description == other.invariant_description
        )


# =============================================================================
# MODULE __all__
# =============================================================================

__all__ = [
    "RiskError",
    "RiskNumericalError",
    "RiskValidationError",
    "RiskParameterConsistencyError",
]
