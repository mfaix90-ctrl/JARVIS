# =============================================================================
# JARVIS v6.1.0 -- PHASE 7: RISK & CAPITAL MANAGEMENT LAYER
# File:   jarvis/risk_layer/domain.py
# Authority: JARVIS FAS v6.1.0 -- Phase 7, Risk Engine
# =============================================================================
#
# SCOPE
# -----
# Frozen domain dataclasses for Phase 7 inputs.
# Implements exactly three public types: PositionSpec, PortfolioState,
# RiskParameters.
#
# No sizing logic. No caps. No verdict logic. No evaluate_risk.
#
# DEPENDENCIES
# ------------
#   stdlib:    math, dataclasses, enum, typing
#   internal:  VALID_ASSET_CLASSES from jarvis.core.data_layer
#   PROHIBITED: numpy, logging, datetime.now(), random, file IO, network IO
#
# VALIDATION PHILOSOPHY
# ---------------------
# Validation is fail-fast and layered, in this fixed order per dataclass:
#
#   V1  Finiteness   -- math.isfinite on every float field.
#                       Raises RiskNumericalError(field_name, value).
#                       No further validation is attempted after this fires.
#   V2  Sign / Range -- field-local constraints (> 0, [0,1], etc.).
#                       Raises RiskValidationError(field_name, value, constraint).
#   V3  Enum / Set   -- Side membership, VALID_ASSET_CLASSES membership.
#                       Raises RiskValidationError with an appropriate constraint.
#   V4  Cross-field  -- relational invariants between two fields.
#                       Raises RiskParameterConsistencyError.
#
# There is NO silent coercion anywhere in this module.
# No field is clipped, clamped, floored, or defaulted silently.
# Every violation raises an exception with the violating field name and value.
#
# DETERMINISM CONSTRAINTS
# -----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly. No module-level mutable reads.
# DET-03  No side effects. All dataclasses are frozen; no mutation path exists.
# DET-04  All numeric checks are deterministic (math.isfinite, comparisons).
# DET-05  No datetime.now() / time.time().
# DET-06  No random / secrets / uuid.
#
# INVARIANTS ENFORCED
# -------------------
# PositionSpec
#   INV-PS-01  All float fields finite (V1 gate).
#   INV-PS-02  entry_price > 0.
#   INV-PS-03  current_price > 0.
#   INV-PS-04  quantity > 0.
#   INV-PS-05  max_position_usd > 0.
#   INV-PS-06  symbol is a non-empty ASCII string.
#   INV-PS-07  asset_class in VALID_ASSET_CLASSES.
#   INV-PS-08  side is a valid Side member.
#
# PortfolioState
#   INV-PF-01  All float fields finite.
#   INV-PF-02  nav > 0.
#   INV-PF-03  gross_exposure_usd >= 0.
#   INV-PF-04  net_exposure_usd is finite (any sign -- no additional constraint).
#   INV-PF-05  open_positions >= 0.
#   INV-PF-06  peak_nav > 0.
#   INV-PF-07  realized_drawdown_pct in [0.0, 1.0].
#   INV-PF-08  current_step >= 0.
#   INV-PF-09  peak_nav >= nav  (cross-field: peak is always the high-water mark).
#
# RiskParameters
#   INV-RP-01  All float fields finite.
#   INV-RP-02  max_position_pct_nav in (0.0, 1.0].
#   INV-RP-03  max_gross_exposure_pct > 0.
#   INV-RP-04  max_drawdown_hard_stop in (0.0, 1.0).
#   INV-RP-05  max_drawdown_soft_warn in (0.0, 1.0).
#   INV-RP-06  volatility_target_ann > 0.
#   INV-RP-07  liquidity_haircut_floor in (0.0, 1.0].
#   INV-RP-08  max_open_positions >= 1.
#   INV-RP-09  kelly_fraction in (0.0, 1.0].
#   INV-RP-10  max_drawdown_soft_warn < max_drawdown_hard_stop  (cross-field).
#
# PROHIBITED ACTIONS CONFIRMED ABSENT
# ------------------------------------
#   No numpy / scipy
#   No logging module
#   No datetime.now() / time.time()
#   No random / secrets / uuid
#   No file IO
#   No network IO
#   No silent coercion of any field
#   No S06+ imports beyond data_layer
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import FrozenSet

from jarvis.core.data_layer import VALID_ASSET_CLASSES
from .exceptions import (
    RiskNumericalError,
    RiskParameterConsistencyError,
    RiskValidationError,
)


# =============================================================================
# SECTION 1 -- ENUMERATIONS
# =============================================================================

class Side(str, Enum):
    """
    Trade side. Inherits from str for clean serialisation.

    LONG  -- buying / long exposure.
    SHORT -- selling / short exposure.

    Using str inheritance means Side.LONG == "LONG" is True,
    which simplifies downstream serialisation without importing enum machinery.
    """
    LONG  = "LONG"
    SHORT = "SHORT"


class RiskVerdict(str, Enum):
    """
    Risk engine decision. Defined here for import convenience.
    Populated by evaluate_risk (Phase 7B); declared here as part of the
    domain contract so RiskDecision can reference it.

    APPROVE  -- proceed at approved_quantity (no caps applied, or caps
                applied but quantity unchanged).
    REDUCE   -- proceed at a reduced approved_quantity (at least one cap
                applied and quantity lowered).
    HOLD     -- do not open new positions; hold existing unchanged.
    HALT     -- hard stop; flatten or freeze all activity immediately.
    REJECT   -- this specific position is rejected; portfolio continues.
    """
    APPROVE = "APPROVE"
    REDUCE  = "REDUCE"
    HOLD    = "HOLD"
    HALT    = "HALT"
    REJECT  = "REJECT"


# =============================================================================
# SECTION 2 -- INTERNAL VALIDATION HELPERS
# =============================================================================
# All helpers are module-private (prefixed _). They take an explicit field_name
# so error messages are always self-identifying.

def _check_finite(field_name: str, value: float) -> None:
    """
    V1 gate: raise RiskNumericalError if value is not finite.

    Called before any range check. If this fires, no further validation
    is performed on the field.
    """
    if not math.isfinite(value):
        raise RiskNumericalError(field_name=field_name, value=value)


def _check_positive(field_name: str, value: float) -> None:
    """V2: value must be strictly > 0."""
    if value <= 0.0:
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be > 0",
        )


def _check_non_negative_float(field_name: str, value: float) -> None:
    """V2: value must be >= 0."""
    if value < 0.0:
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be >= 0",
        )


def _check_unit_interval_open_closed(field_name: str, value: float) -> None:
    """V2: value must be in (0.0, 1.0] -- strictly positive, at most 1."""
    if not (0.0 < value <= 1.0):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be in (0.0, 1.0]",
        )


def _check_unit_interval_closed(field_name: str, value: float) -> None:
    """V2: value must be in [0.0, 1.0]."""
    if not (0.0 <= value <= 1.0):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be in [0.0, 1.0]",
        )


def _check_unit_interval_open_open(field_name: str, value: float) -> None:
    """V2: value must be in (0.0, 1.0) -- strictly between 0 and 1."""
    if not (0.0 < value < 1.0):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be in (0.0, 1.0)",
        )


def _check_non_empty_ascii_string(field_name: str, value: str) -> None:
    """V3: value must be a non-empty string containing only ASCII characters."""
    if not isinstance(value, str) or not value:
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be a non-empty string",
        )
    try:
        value.encode("ascii")
    except (UnicodeEncodeError, AttributeError):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must contain only ASCII characters",
        )


def _check_asset_class(field_name: str, value: str) -> None:
    """V3: value must be a member of VALID_ASSET_CLASSES."""
    if value not in VALID_ASSET_CLASSES:
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be in VALID_ASSET_CLASSES: " + repr(sorted(VALID_ASSET_CLASSES)),
        )


def _check_side(field_name: str, value: object) -> None:
    """V3: value must be a valid Side enum member."""
    if not isinstance(value, Side):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be a valid Side enum member (Side.LONG or Side.SHORT)",
        )


def _check_non_negative_int(field_name: str, value: int) -> None:
    """V2: value must be an int >= 0."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be a non-negative integer",
        )
    if value < 0:
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be >= 0",
        )


def _check_positive_int(field_name: str, value: int) -> None:
    """V2: value must be an int >= 1."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be a positive integer",
        )
    if value < 1:
        raise RiskValidationError(
            field_name=field_name,
            value=value,
            constraint="must be >= 1",
        )


# =============================================================================
# SECTION 3 -- POSITION SPEC
# =============================================================================

@dataclass(frozen=True)
class PositionSpec:
    """
    Describes a single proposed position to be evaluated by the risk engine.

    All fields are immutable after construction. Validation in __post_init__
    is fail-fast: first violation encountered raises immediately.

    Invariants (see module docstring INV-PS-*):
      - symbol: non-empty ASCII string.
      - asset_class: member of VALID_ASSET_CLASSES.
      - side: a valid Side enum member.
      - entry_price:       finite, > 0.
      - current_price:     finite, > 0.
      - quantity:          finite, > 0.
      - max_position_usd:  finite, > 0.
    """

    symbol:           str
    """Instrument identifier. Non-empty ASCII string (e.g. 'BTC-USD')."""

    asset_class:      str
    """Asset class key. Must be in VALID_ASSET_CLASSES."""

    side:             Side
    """Trade direction. Must be a valid Side enum member."""

    entry_price:      float
    """Intended entry price per unit. Finite, strictly positive."""

    current_price:    float
    """Current market price per unit. Finite, strictly positive."""

    quantity:         float
    """Number of units requested. Finite, strictly positive."""

    max_position_usd: float
    """
    Hard position size cap in USD, set by the caller (e.g. from a per-symbol
    risk limit table). The risk engine will never approve a position larger
    than this. Finite, strictly positive.
    """

    def __post_init__(self) -> None:
        # --- V3: string and enum fields first (before float gate) ---
        _check_non_empty_ascii_string("symbol", self.symbol)
        _check_asset_class("asset_class", self.asset_class)
        _check_side("side", self.side)

        # --- V1 + V2: float fields in declaration order ---
        _FLOAT_FIELDS_POSITIVE = (
            ("entry_price",      self.entry_price),
            ("current_price",    self.current_price),
            ("quantity",         self.quantity),
            ("max_position_usd", self.max_position_usd),
        )
        for fname, fvalue in _FLOAT_FIELDS_POSITIVE:
            _check_finite(fname, fvalue)
            _check_positive(fname, fvalue)


# =============================================================================
# SECTION 4 -- PORTFOLIO STATE
# =============================================================================

@dataclass(frozen=True)
class PortfolioState:
    """
    Snapshot of portfolio-level metrics at the time of risk evaluation.

    Immutable after construction. Represents the current state of the portfolio
    into which the proposed PositionSpec would be placed.

    Invariants (see module docstring INV-PF-*):
      - nav:                    finite, > 0.
      - gross_exposure_usd:     finite, >= 0.
      - net_exposure_usd:       finite (any sign; long-short book).
      - open_positions:         int, >= 0.
      - peak_nav:               finite, > 0.
      - realized_drawdown_pct:  finite, in [0.0, 1.0].
      - current_step:           int, >= 0.
      - peak_nav >= nav         (cross-field invariant INV-PF-09).
    """

    nav:                   float
    """Net Asset Value in USD. Finite, strictly positive."""

    gross_exposure_usd:    float
    """
    Sum of absolute values of all open position USD values.
    Finite, non-negative.
    """

    net_exposure_usd:      float
    """
    Algebraic sum of signed open position USD values (long minus short).
    Finite. May be any sign.
    """

    open_positions:        int
    """Number of currently open positions. Non-negative integer."""

    peak_nav:              float
    """
    Highest NAV achieved since inception or last reset (high-water mark).
    Finite, strictly positive. Must be >= nav.
    """

    realized_drawdown_pct: float
    """
    Percentage drawdown from peak_nav to current nav, expressed as a fraction
    in [0.0, 1.0]. Computed by the caller as: 1.0 - (nav / peak_nav).
    Must be consistent with peak_nav and nav (checked implicitly via range).
    """

    current_step:          int
    """
    Monotonic logical step counter. Matches the caller's bar index or
    sequence number. Non-negative integer.
    """

    def __post_init__(self) -> None:
        # --- V1 + V2: float fields ---
        _check_finite("nav", self.nav)
        _check_positive("nav", self.nav)

        _check_finite("gross_exposure_usd", self.gross_exposure_usd)
        _check_non_negative_float("gross_exposure_usd", self.gross_exposure_usd)

        _check_finite("net_exposure_usd", self.net_exposure_usd)
        # net_exposure_usd: finite only; any sign is valid.

        _check_finite("peak_nav", self.peak_nav)
        _check_positive("peak_nav", self.peak_nav)

        _check_finite("realized_drawdown_pct", self.realized_drawdown_pct)
        _check_unit_interval_closed("realized_drawdown_pct", self.realized_drawdown_pct)

        # --- V2: integer fields ---
        _check_non_negative_int("open_positions", self.open_positions)
        _check_non_negative_int("current_step", self.current_step)

        # --- V4: cross-field invariant INV-PF-09 ---
        # peak_nav is the all-time high-water mark; it can never be less than
        # the current nav. Equality is permitted (nav == peak_nav means no
        # drawdown has occurred since the last reset).
        if self.peak_nav < self.nav:
            raise RiskParameterConsistencyError(
                field_a="peak_nav",
                value_a=self.peak_nav,
                field_b="nav",
                value_b=self.nav,
                invariant_description=(
                    "peak_nav must be >= nav "
                    "(peak_nav is the high-water mark; "
                    "it cannot be lower than the current nav)"
                ),
            )


# =============================================================================
# SECTION 5 -- RISK PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class RiskParameters:
    """
    Immutable configuration object for the risk engine.

    All parameters are validated in __post_init__. No defaults are provided;
    every field must be supplied explicitly by the caller. This enforces
    conscious configuration rather than accidental reliance on silent defaults.

    Invariants (see module docstring INV-RP-*):
      - max_position_pct_nav:   finite, in (0.0, 1.0].
      - max_gross_exposure_pct: finite, > 0.
      - max_drawdown_hard_stop: finite, in (0.0, 1.0).
      - max_drawdown_soft_warn: finite, in (0.0, 1.0).
      - volatility_target_ann:  finite, > 0.
      - liquidity_haircut_floor: finite, in (0.0, 1.0].
      - max_open_positions:     int, >= 1.
      - kelly_fraction:         finite, in (0.0, 1.0].
      - max_drawdown_soft_warn < max_drawdown_hard_stop  (cross-field).
    """

    max_position_pct_nav:    float
    """
    Maximum position size as a fraction of NAV.
    E.g. 0.05 means no single position may exceed 5% of NAV.
    In (0.0, 1.0].
    """

    max_gross_exposure_pct:  float
    """
    Maximum gross exposure as a fraction of NAV (i.e. gross leverage ceiling).
    E.g. 1.5 means gross exposure may not exceed 150% of NAV.
    Strictly positive (values > 1.0 are permitted to allow leverage).
    """

    max_drawdown_hard_stop:  float
    """
    Drawdown level at which all new risk-taking is halted immediately.
    E.g. 0.10 means a 10% drawdown from peak_nav triggers HALT.
    In (0.0, 1.0). Must be strictly greater than max_drawdown_soft_warn.
    """

    max_drawdown_soft_warn:  float
    """
    Drawdown level at which position sizing is reduced (REDUCE verdict).
    In (0.0, 1.0). Must be strictly less than max_drawdown_hard_stop.
    """

    volatility_target_ann:   float
    """
    Annualised volatility target used for vol-adjusted position sizing.
    E.g. 0.15 means the engine targets 15% annualised portfolio volatility.
    Strictly positive.
    """

    liquidity_haircut_floor: float
    """
    Minimum size multiplier applied by the liquidity adjustment.
    Even in the most illiquid conditions, approved_quantity will be at
    least (liquidity_haircut_floor * uncapped_quantity).
    In (0.0, 1.0].
    """

    max_open_positions:      int
    """
    Maximum number of simultaneously open positions.
    Integer >= 1.
    """

    kelly_fraction:          float
    """
    Fractional Kelly multiplier applied to the raw Kelly criterion output.
    E.g. 0.25 means quarter-Kelly sizing.
    In (0.0, 1.0].
    """

    def __post_init__(self) -> None:
        # --- V1: finiteness on all float fields ---
        _FLOAT_FIELDS: tuple = (
            ("max_position_pct_nav",    self.max_position_pct_nav),
            ("max_gross_exposure_pct",  self.max_gross_exposure_pct),
            ("max_drawdown_hard_stop",  self.max_drawdown_hard_stop),
            ("max_drawdown_soft_warn",  self.max_drawdown_soft_warn),
            ("volatility_target_ann",   self.volatility_target_ann),
            ("liquidity_haircut_floor", self.liquidity_haircut_floor),
            ("kelly_fraction",          self.kelly_fraction),
        )
        for fname, fvalue in _FLOAT_FIELDS:
            _check_finite(fname, fvalue)

        # --- V2: range and sign constraints ---
        _check_unit_interval_open_closed("max_position_pct_nav",    self.max_position_pct_nav)
        _check_positive("max_gross_exposure_pct",                   self.max_gross_exposure_pct)
        _check_unit_interval_open_open("max_drawdown_hard_stop",    self.max_drawdown_hard_stop)
        _check_unit_interval_open_open("max_drawdown_soft_warn",    self.max_drawdown_soft_warn)
        _check_positive("volatility_target_ann",                    self.volatility_target_ann)
        _check_unit_interval_open_closed("liquidity_haircut_floor", self.liquidity_haircut_floor)
        _check_unit_interval_open_closed("kelly_fraction",          self.kelly_fraction)

        # --- V2: integer fields ---
        _check_positive_int("max_open_positions", self.max_open_positions)

        # --- V4: cross-field invariant INV-RP-10 ---
        # soft_warn must be strictly less than hard_stop.
        # If they are equal or inverted, the soft warning band collapses to
        # zero width, which makes the REDUCE verdict unreachable -- a
        # configuration error that must be caught at construction time.
        if self.max_drawdown_soft_warn >= self.max_drawdown_hard_stop:
            raise RiskParameterConsistencyError(
                field_a="max_drawdown_soft_warn",
                value_a=self.max_drawdown_soft_warn,
                field_b="max_drawdown_hard_stop",
                value_b=self.max_drawdown_hard_stop,
                invariant_description=(
                    "max_drawdown_soft_warn must be strictly less than "
                    "max_drawdown_hard_stop "
                    "(the soft-warning band must have positive width)"
                ),
            )


# =============================================================================
# SECTION 6 -- MODULE __all__
# =============================================================================

__all__ = [
    "Side",
    "RiskVerdict",
    "PositionSpec",
    "PortfolioState",
    "RiskParameters",
]
