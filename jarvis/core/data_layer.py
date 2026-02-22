# jarvis/core/data_layer.py
# Version: 6.0.1
# Session: S03 -- Data Ingestion / Data Layer
# Authority: JARVIS FAS v6.0.1 -- LAYER 1 DATA INGESTION (S03)
#
# =============================================================================
# FAS COMPLIANCE DECLARATION
# =============================================================================
#
# Dependency matrix (binding, from FAS master matrix):
#   S03 -> S01  (permitted)
#   S03 -> S02  (FORBIDDEN -- no logging_layer import)
#
# This module has NO import of jarvis.core.integrity_layer.
# No S01 operation is required for data objects per FAS v6.0.1 S03 spec.
# (DEC-02, approved 2026-02-21: no IntegrityLayer import unless hash-chain
#  stamping of data objects is explicitly added to FAS scope.)
#
# WRITE PERMISSION (from FAS dependency matrix):
#   data_layer.py  WRITES TO:             EnhancedMarketData
#   data_layer.py  FORBIDDEN FROM WRITING: Any State
#
# DETERMINISM GUARANTEES:
#   DET-01  No stochastic operations. No random, no uuid, no os.urandom.
#   DET-02  All inputs passed explicitly. No module-level mutable reads.
#   DET-03  No side effects in computational functions.
#   DET-04  All branches are functions of explicit inputs only.
#   DET-05  Cache TTL enforced via caller-supplied monotonic step counter.
#           No wall-clock access (time.time, datetime.now forbidden).
#   DET-06  Timestamps are caller-supplied integers (Unix epoch seconds).
#           datetime objects are not used. (DEC-01, approved 2026-02-21.)
#
# PROHIBITED ACTIONS CONFIRMED ABSENT:
#   - No import of logging_layer (S02)
#   - No import of regime.py (S05)
#   - No import of numpy (DEC-03, approved 2026-02-21: math.isfinite only)
#   - No datetime.now(), datetime.utcnow(), time.time()
#   - No import of random, secrets, uuid
#   - No file IO (open, os.path, pathlib)
#   - No network IO (requests, urllib, socket)
#   - No OOD logic (belongs to S10)
#   - No print statements
#   - No global mutable state
#   - No state object writes
#
# OOD BOUNDARY:
#   This module raises DataQualityError when quality_score < QUALITY_HARD_GATE.
#   The CALLER (pipeline orchestrator) is responsible for setting any OOD flag.
#   This module never classifies OOD, never invokes OOD ensemble, never
#   sets HOLD or DEFENSIVE mode. Those belong to S10.
#
# CACHE DESIGN:
#   DataCache uses a caller-supplied monotonic step counter for TTL.
#   now_step is an integer representing the caller's logical clock tick
#   (e.g., bar index, sequence counter). The cache stores:
#     expiry_step = store_step + ttl_steps
#   Eviction: entry is stale when now_step >= expiry_step.
#   This is fully deterministic and reproducible in backtests.
#
# FROZEN DATACLASS INHERITANCE:
#   Both MarketData and EnhancedMarketData are @dataclass(frozen=True).
#   Python requires the parent to be frozen when the child is frozen.
#   (DEC-04, approved 2026-02-21.)
#
# ASCII COMPLIANCE:
#   All string literals, comments, and identifiers use 7-bit ASCII only.
#
# Canonical import pattern for downstream consumers:
#   from jarvis.core.data_layer import (
#       OHLCV, MarketData, EnhancedMarketData,
#       ValidationResult, DataCache,
#       validate_numeric_field, validate_ohlcv,
#       validate_enhanced_market_data, compute_gap,
#       classify_session, check_sequence,
#       NumericalInstabilityError, DataQualityError, SequenceError,
#       QUALITY_HARD_GATE, GAP_THRESHOLDS,
#       VALID_ASSET_CLASSES, VALID_TIMEFRAMES, VALID_SESSION_TAGS,
#   )
#
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# =============================================================================
# SECTION 1 -- MODULE-LEVEL CONSTANTS (IMMUTABLE)
# =============================================================================

# Quality gate threshold. FAS: "Bei quality_score < 0.5 -> Daten verwerfen."
# Raising DataQualityError is the hard-stop action. Caller sets OOD flag.
QUALITY_HARD_GATE: float = 0.5

# Minimum quality to avoid liquidity warning. FAS: "Liquidity Quality >= 0.6"
LIQUIDITY_QUALITY_MINIMUM: float = 0.6

# Per-asset gap thresholds (fractional). FAS:
#   Crypto: 5%, Forex: 2%, Indices: 3%, Commodities: 4%
# "rates" uses the forex threshold (2%) as the closest FAS analog.
GAP_THRESHOLDS: Dict[str, float] = {
    "crypto":      0.05,
    "forex":       0.02,
    "indices":     0.03,
    "commodities": 0.04,
    "rates":       0.02,
}

# Canonical asset class identifiers. All values ASCII-lowercase.
VALID_ASSET_CLASSES: FrozenSet[str] = frozenset({
    "crypto",
    "forex",
    "indices",
    "commodities",
    "rates",
})

# Canonical timeframe identifiers. FAS StandardizedMarketDataObject.timeframe.
VALID_TIMEFRAMES: FrozenSet[str] = frozenset({
    "M1", "M5", "M15", "M30",
    "H1", "H4",
    "D1", "W1",
})

# Canonical session tags. FAS StandardizedMarketDataObject.session_tag.
VALID_SESSION_TAGS: FrozenSet[str] = frozenset({
    "LONDON",
    "NEW_YORK",
    "TOKYO",
    "SYDNEY",
    "CRYPTO_24_7",
    "PRE_MARKET",
    "POST_MARKET",
    "AUCTION",
    "UNKNOWN",
})

# Canonical data source identifiers.
VALID_DATA_SOURCES: FrozenSet[str] = frozenset({
    "historical",
    "live",
    "hybrid_backfill",
    "hybrid_live",
})

# OHLCV price fields inspected during structural validation.
_OHLCV_PRICE_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")


# =============================================================================
# SECTION 2 -- EXCEPTIONS
# =============================================================================

class NumericalInstabilityError(ValueError):
    """
    Raised when any numeric field contains NaN or Inf.

    FAS mandate: "Numerische Instabilitaet in Feld '<name>': <value>".
    Never silently ignored. Always propagated to caller.

    Applicable to: OHLCV fields, quality_score, spread_bps,
                   gap_size, and any float in ValidationResult paths.
    """


class DataQualityError(ValueError):
    """
    Raised when quality_score < QUALITY_HARD_GATE (0.5).

    FAS hard-stop: "Bei quality_score < 0.5 -> Daten verwerfen, OOD-Flag setzen."
    This module raises DataQualityError. The CALLER sets the OOD flag.
    This module has no access to any OOD or state subsystem.

    Also raised when quality_score is outside [0.0, 1.0].
    """


class SequenceError(ValueError):
    """
    Raised when a sequence_id violates the monotonic-increasing invariant.

    FAS: "sequence_id: int -- Monoton steigend -- fuer Kontinuitaets-Validierung".
    Sequence regression indicates data stream corruption or out-of-order delivery.
    """


# =============================================================================
# SECTION 3 -- OHLCV DATA STRUCTURE
# =============================================================================

@dataclass(frozen=True)
class OHLCV:
    """
    Open-High-Low-Close-Volume candle. Immutable value object.

    FAS constraints enforced in __post_init__:
    - All five fields must be finite (not NaN, not Inf).
    - All five fields must be strictly positive (> 0).
    - high >= low  (structural OHLCV invariant).
    - high >= open (structural OHLCV invariant).
    - high >= close (structural OHLCV invariant).

    Raises NumericalInstabilityError on NaN/Inf.
    Raises ValueError on non-positive values or structural violations.
    """

    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

    def __post_init__(self) -> None:
        # Phase 1: NaN/Inf check on all fields.
        for name in _OHLCV_PRICE_FIELDS:
            value: float = getattr(self, name)
            if not math.isfinite(value):
                raise NumericalInstabilityError(
                    "OHLCV." + name + " contains NaN or Inf: " + repr(value)
                )

        # Phase 2: Positivity check on all fields.
        for name in _OHLCV_PRICE_FIELDS:
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(
                    "OHLCV." + name + " must be > 0, got: " + repr(value)
                )

        # Phase 3: Structural OHLCV invariants.
        if self.high < self.low:
            raise ValueError(
                "OHLCV.high (" + repr(self.high) + ") < low (" + repr(self.low) + ")"
            )
        if self.high < self.open:
            raise ValueError(
                "OHLCV.high (" + repr(self.high) + ") < open (" + repr(self.open) + ")"
            )
        if self.high < self.close:
            raise ValueError(
                "OHLCV.high (" + repr(self.high) + ") < close (" + repr(self.close) + ")"
            )


# =============================================================================
# SECTION 4 -- MARKETDATA BASE STRUCTURE
# =============================================================================

@dataclass(frozen=True)
class MarketData:
    """
    Base validated market data object.

    All fields are immutable after construction. Timestamps are caller-supplied
    integers (Unix epoch seconds). No datetime objects are used.
    (DEC-01, approved 2026-02-21: timestamp_utc: int replaces datetime.)

    FAS schema reference: StandardizedMarketDataObject (market_data_provider.py
    section of FAS v6.0.1 Session 03 Extension).

    quality_score must be in [0.0, 1.0]. Enforced in __post_init__.
    sequence_id must be >= 0. Monotonic enforcement is the caller's
    responsibility via check_sequence(); the field itself only requires >= 0.
    """

    # -- Identity --
    symbol:         str       # e.g. "BTC/USDT", "EUR/USD", "SPX"
    asset_class:    str       # one of VALID_ASSET_CLASSES
    timeframe:      str       # one of VALID_TIMEFRAMES
    timestamp_utc:  int       # Unix epoch seconds, caller-supplied

    # -- Price data --
    ohlcv:          OHLCV

    # -- Quality --
    quality_score:  float     # [0.0, 1.0]

    # -- Continuity --
    sequence_id:    int       # monotonically increasing, >= 0

    # -- Source metadata (opaque to downstream; for logging by caller only) --
    data_source:    str       # one of VALID_DATA_SOURCES
    provider_id:    str       # arbitrary identifier, not used in logic

    def __post_init__(self) -> None:
        # symbol: non-empty string
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("MarketData.symbol must be a non-empty string")

        # asset_class: canonical value
        if self.asset_class not in VALID_ASSET_CLASSES:
            raise ValueError(
                "MarketData.asset_class invalid: " + repr(self.asset_class)
                + ". Valid: " + repr(sorted(VALID_ASSET_CLASSES))
            )

        # timeframe: canonical value
        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                "MarketData.timeframe invalid: " + repr(self.timeframe)
                + ". Valid: " + repr(sorted(VALID_TIMEFRAMES))
            )

        # timestamp_utc: non-negative integer
        if not isinstance(self.timestamp_utc, int) or self.timestamp_utc < 0:
            raise ValueError(
                "MarketData.timestamp_utc must be a non-negative int, got: "
                + repr(self.timestamp_utc)
            )

        # quality_score: finite and in [0, 1]
        if not math.isfinite(self.quality_score):
            raise NumericalInstabilityError(
                "MarketData.quality_score contains NaN or Inf: "
                + repr(self.quality_score)
            )
        if not (0.0 <= self.quality_score <= 1.0):
            raise DataQualityError(
                "MarketData.quality_score must be in [0.0, 1.0], got: "
                + repr(self.quality_score)
            )

        # sequence_id: non-negative integer
        if not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValueError(
                "MarketData.sequence_id must be a non-negative int, got: "
                + repr(self.sequence_id)
            )

        # data_source: canonical value
        if self.data_source not in VALID_DATA_SOURCES:
            raise ValueError(
                "MarketData.data_source invalid: " + repr(self.data_source)
                + ". Valid: " + repr(sorted(VALID_DATA_SOURCES))
            )

        # provider_id: non-empty string
        if not isinstance(self.provider_id, str) or not self.provider_id:
            raise ValueError("MarketData.provider_id must be a non-empty string")


# =============================================================================
# SECTION 5 -- ENHANCEDMARKETDATA EXTENDED STRUCTURE
# =============================================================================

@dataclass(frozen=True)
class EnhancedMarketData(MarketData):
    """
    Extended market data object with multi-asset microstructure awareness.

    Inherits all MarketData fields and validation. Adds:
    - gap_detected / gap_size:  structural gap metadata (not OOD logic)
    - session_tag:              canonical trading session identifier
    - spread_bps:               estimated spread in basis points
    - is_stale:                 true if data exceeded caller-defined age window
    - liquidity_regime:         "high", "normal", "low", or "unknown"

    FAS reference: EnhancedMarketData(MarketData) -- Session 03 Extension,
    lines 16105-16140 of FAS v6.0.1.

    OOD BOUNDARY:
      gap_detected=True is a structural data flag. It signals a price gap
      to the feature layer (S04) and pipeline orchestrator. It does NOT
      trigger OOD classification. OOD logic belongs exclusively to S10.

    WRITE PERMISSION:
      Only data_layer.py constructs EnhancedMarketData.
      FAS: "WRITE: EnhancedMarketData object -- data_layer.py ONLY"
    """

    # -- Gap metadata --
    gap_detected:       bool
    gap_size:           Optional[float]   # fractional; None when gap_detected=False

    # -- Session metadata --
    session_tag:        str               # one of VALID_SESSION_TAGS

    # -- Spread / execution cost --
    spread_bps:         float             # >= 0, finite

    # -- Staleness --
    is_stale:           bool

    # -- Liquidity regime (structural classification, not OOD) --
    liquidity_regime:   str               # "high", "normal", "low", "unknown"

    def __post_init__(self) -> None:
        # Run parent validation first.
        # frozen=True with dataclass inheritance: super().__post_init__ must be
        # called explicitly because Python does not chain __post_init__ automatically.
        super().__post_init__()

        # gap_detected and gap_size consistency.
        if self.gap_detected:
            if self.gap_size is None:
                raise ValueError(
                    "EnhancedMarketData.gap_size must not be None when "
                    "gap_detected=True"
                )
            if not math.isfinite(self.gap_size):
                raise NumericalInstabilityError(
                    "EnhancedMarketData.gap_size contains NaN or Inf: "
                    + repr(self.gap_size)
                )
            if self.gap_size < 0.0:
                raise ValueError(
                    "EnhancedMarketData.gap_size must be >= 0.0, got: "
                    + repr(self.gap_size)
                )
        else:
            # gap_size must be None when no gap detected.
            # Supplying a non-None value when gap_detected=False is a
            # caller error; reject it to prevent silent data inconsistency.
            if self.gap_size is not None:
                raise ValueError(
                    "EnhancedMarketData.gap_size must be None when "
                    "gap_detected=False, got: " + repr(self.gap_size)
                )

        # session_tag: canonical value.
        if self.session_tag not in VALID_SESSION_TAGS:
            raise ValueError(
                "EnhancedMarketData.session_tag invalid: " + repr(self.session_tag)
                + ". Valid: " + repr(sorted(VALID_SESSION_TAGS))
            )

        # spread_bps: finite and >= 0.
        if not math.isfinite(self.spread_bps):
            raise NumericalInstabilityError(
                "EnhancedMarketData.spread_bps contains NaN or Inf: "
                + repr(self.spread_bps)
            )
        if self.spread_bps < 0.0:
            raise ValueError(
                "EnhancedMarketData.spread_bps must be >= 0.0, got: "
                + repr(self.spread_bps)
            )

        # liquidity_regime: canonical value.
        _VALID_LIQUIDITY_REGIMES: FrozenSet[str] = frozenset({
            "high", "normal", "low", "unknown"
        })
        if self.liquidity_regime not in _VALID_LIQUIDITY_REGIMES:
            raise ValueError(
                "EnhancedMarketData.liquidity_regime invalid: "
                + repr(self.liquidity_regime)
                + ". Valid: " + repr(sorted(_VALID_LIQUIDITY_REGIMES))
            )


# =============================================================================
# SECTION 6 -- VALIDATIONRESULT STRUCTURE
# =============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """
    Result of validate_enhanced_market_data().

    valid=True means the data object passed all checks and quality_score
    is above QUALITY_HARD_GATE. Warnings are non-blocking observations
    (e.g. large gap in indices, low Asian session liquidity).

    valid=False is returned only when quality_score is in [0.5, 1.0] but
    asset-class-specific structural warnings are present and the caller
    requested strict mode. Under default mode, valid=False is only produced
    by quality_score < QUALITY_HARD_GATE (which raises DataQualityError).

    FAS: "validate_market_data(data: EnhancedMarketData) -> ValidationResult"
    """

    valid:               bool
    errors:              List[str]    # empty list when valid=True
    warnings:            List[str]    # non-empty for soft observations
    gap_adjusted:        bool         # True if gap was detected and flagged
    liquidity_adjusted:  bool         # True if low-liquidity warning issued


# =============================================================================
# SECTION 7 -- PURE VALIDATION FUNCTIONS
# =============================================================================

def validate_numeric_field(name: str, value: float) -> float:
    """
    Validate a single numeric field for NaN/Inf.

    FAS: "Prueft jeden numerischen Wert vor Verwendung. Wirft Exception bei
    NaN/Inf -- niemals silentes Ignorieren."

    Args:
        name:  Field name for error message construction.
        value: Numeric value to validate.

    Returns:
        value unchanged if finite.

    Raises:
        NumericalInstabilityError if value is NaN or Inf.
    """
    if not math.isfinite(value):
        raise NumericalInstabilityError(
            "Numerical instability in field '" + name + "': " + repr(value)
        )
    return value


def validate_ohlcv(ohlcv: OHLCV) -> None:
    """
    Validate structural OHLCV invariants.

    All invariants are also enforced in OHLCV.__post_init__, so a successfully
    constructed OHLCV object is always valid. This function is provided as an
    explicit validation entry point for callers who receive OHLCV objects from
    external sources and wish to re-validate without constructing a new object.

    Args:
        ohlcv: OHLCV instance to validate.

    Raises:
        NumericalInstabilityError if any field is NaN or Inf.
        ValueError if any positivity or structural invariant is violated.
    """
    for name in _OHLCV_PRICE_FIELDS:
        value: float = getattr(ohlcv, name)
        validate_numeric_field("OHLCV." + name, value)
        if value <= 0.0:
            raise ValueError(
                "OHLCV." + name + " must be > 0, got: " + repr(value)
            )

    if ohlcv.high < ohlcv.low:
        raise ValueError(
            "OHLCV.high (" + repr(ohlcv.high) + ") < low (" + repr(ohlcv.low) + ")"
        )
    if ohlcv.high < ohlcv.open:
        raise ValueError(
            "OHLCV.high (" + repr(ohlcv.high) + ") < open (" + repr(ohlcv.open) + ")"
        )
    if ohlcv.high < ohlcv.close:
        raise ValueError(
            "OHLCV.high (" + repr(ohlcv.high) + ") < close (" + repr(ohlcv.close) + ")"
        )


def validate_enhanced_market_data(data: EnhancedMarketData) -> ValidationResult:
    """
    Asset-class-aware validation of an EnhancedMarketData object.

    Applies all structural invariants from OHLCV and MarketData (via frozen
    dataclass __post_init__), then applies asset-class-specific soft rules
    per FAS Section LAYER 1 DATA INGESTION (S03):

    Indices:
      gap_detected=True and gap_size > 0.02 (2%) -> warning, valid=True
      (FAS: "Gaps are normal for indices")

    Forex:
      session_tag indicates Asia AND liquidity_regime="low" -> warning, valid=True
      (FAS: "Low liquidity in Asian session")

    All asset classes:
      liquidity_regime="low" -> liquidity_adjusted=True in result

    Hard gate:
      quality_score < QUALITY_HARD_GATE (0.5) -> DataQualityError raised.
      This is the S03 hard-stop. The CALLER sets any OOD flag.

    Args:
        data: EnhancedMarketData instance to validate.

    Returns:
        ValidationResult with valid, errors, warnings, gap_adjusted,
        liquidity_adjusted fields populated.

    Raises:
        DataQualityError if quality_score < QUALITY_HARD_GATE.
        NumericalInstabilityError if any numeric field is NaN or Inf.
          (These are raised during EnhancedMarketData construction, but
           this function re-checks spread_bps and gap_size explicitly
           as a defence-in-depth measure for callers who deserialize
           data objects without constructing them via __init__.)
    """
    errors:   List[str] = []
    warnings: List[str] = []
    gap_adjusted:       bool = False
    liquidity_adjusted: bool = False

    # -- Hard gate: quality_score --
    # Re-validate even though __post_init__ already checked it.
    # Defence-in-depth: objects may be reconstructed outside __init__
    # in test or deserialization scenarios.
    validate_numeric_field("quality_score", data.quality_score)
    if data.quality_score < QUALITY_HARD_GATE:
        raise DataQualityError(
            "quality_score " + repr(data.quality_score)
            + " is below QUALITY_HARD_GATE " + repr(QUALITY_HARD_GATE)
            + " for symbol=" + repr(data.symbol)
            + " asset_class=" + repr(data.asset_class)
            + ". Data rejected. Caller must set OOD flag if applicable."
        )

    # -- NaN/Inf defence on spread_bps --
    validate_numeric_field("spread_bps", data.spread_bps)

    # -- NaN/Inf defence on gap_size when present --
    if data.gap_detected and data.gap_size is not None:
        validate_numeric_field("gap_size", data.gap_size)

    # -- Asset-class-specific soft rules --
    if data.asset_class == "indices":
        # FAS: "Check for gaps at market open. If gap_detected and gap_size > 0.02:
        #        return ValidationResult(valid=True, warnings=['Large gap detected
        #        at market open'], gap_adjusted=True)"
        if data.gap_detected and data.gap_size is not None:
            gap_threshold: float = GAP_THRESHOLDS.get("indices", 0.03)
            if data.gap_size > gap_threshold:
                warnings.append(
                    "Large gap detected at market open: gap_size="
                    + repr(data.gap_size)
                    + " exceeds indices threshold=" + repr(gap_threshold)
                )
            gap_adjusted = True

    elif data.asset_class == "forex":
        # FAS: "Different validation for FX. If current_session == 'asia' and
        #        liquidity_regime == 'low': return ValidationResult(valid=True,
        #        warnings=['Low liquidity in Asian session'],
        #        liquidity_adjusted=True)"
        if data.session_tag == "TOKYO" and data.liquidity_regime == "low":
            warnings.append(
                "Low liquidity in Asian session: session_tag="
                + repr(data.session_tag)
                + ", liquidity_regime=" + repr(data.liquidity_regime)
            )
            liquidity_adjusted = True

    elif data.asset_class == "crypto":
        # Crypto: 24/7, no session gaps expected. Large gap is a warning.
        if data.gap_detected and data.gap_size is not None:
            gap_threshold = GAP_THRESHOLDS.get("crypto", 0.05)
            if data.gap_size > gap_threshold:
                warnings.append(
                    "Unexpected gap in crypto market: gap_size="
                    + repr(data.gap_size)
                    + " exceeds crypto threshold=" + repr(gap_threshold)
                )
            gap_adjusted = True

    elif data.asset_class == "commodities":
        if data.gap_detected and data.gap_size is not None:
            gap_threshold = GAP_THRESHOLDS.get("commodities", 0.04)
            if data.gap_size > gap_threshold:
                warnings.append(
                    "Large gap in commodities data: gap_size="
                    + repr(data.gap_size)
                    + " exceeds commodities threshold=" + repr(gap_threshold)
                )
            gap_adjusted = True

    elif data.asset_class == "rates":
        if data.gap_detected and data.gap_size is not None:
            gap_threshold = GAP_THRESHOLDS.get("rates", 0.02)
            if data.gap_size > gap_threshold:
                warnings.append(
                    "Large gap in rates data: gap_size="
                    + repr(data.gap_size)
                    + " exceeds rates threshold=" + repr(gap_threshold)
                )
            gap_adjusted = True

    # -- Cross-asset liquidity check --
    # FAS: "Liquidity Quality >= 0.6"
    # If liquidity_regime is "low" for any asset class, flag it.
    if data.liquidity_regime == "low":
        liquidity_adjusted = True
        if "Low liquidity" not in " ".join(warnings):
            warnings.append(
                "Low liquidity regime detected: asset_class="
                + repr(data.asset_class)
                + ", session_tag=" + repr(data.session_tag)
            )

    # -- Staleness warning --
    if data.is_stale:
        warnings.append(
            "Data is stale: symbol=" + repr(data.symbol)
            + ", timestamp_utc=" + repr(data.timestamp_utc)
        )

    return ValidationResult(
        valid=True,
        errors=errors,
        warnings=warnings,
        gap_adjusted=gap_adjusted,
        liquidity_adjusted=liquidity_adjusted,
    )


# =============================================================================
# SECTION 8 -- PURE COMPUTATIONAL FUNCTIONS
# =============================================================================

def compute_gap(
    prev_close: float,
    current_open: float,
    asset_class: str,
) -> Tuple[bool, float]:
    """
    Compute gap between previous close and current open.

    FAS: "detect_gap(market_data) -> Returns gap_size, gap_detected"

    The gap is expressed as a fraction of prev_close:
        gap_size = abs(current_open - prev_close) / prev_close

    gap_detected is True when gap_size exceeds the asset-class threshold
    from GAP_THRESHOLDS.

    Args:
        prev_close:   Closing price of the preceding bar. Must be > 0, finite.
        current_open: Opening price of the current bar. Must be > 0, finite.
        asset_class:  One of VALID_ASSET_CLASSES.

    Returns:
        (gap_detected: bool, gap_size: float)
        gap_size is always >= 0.0 and finite.

    Raises:
        NumericalInstabilityError if prev_close or current_open is NaN/Inf.
        ValueError if prev_close <= 0 or current_open <= 0.
        ValueError if asset_class is not in VALID_ASSET_CLASSES.
    """
    validate_numeric_field("prev_close", prev_close)
    validate_numeric_field("current_open", current_open)

    if prev_close <= 0.0:
        raise ValueError(
            "compute_gap: prev_close must be > 0, got: " + repr(prev_close)
        )
    if current_open <= 0.0:
        raise ValueError(
            "compute_gap: current_open must be > 0, got: " + repr(current_open)
        )
    if asset_class not in VALID_ASSET_CLASSES:
        raise ValueError(
            "compute_gap: asset_class invalid: " + repr(asset_class)
        )

    gap_size: float = abs(current_open - prev_close) / prev_close
    threshold: float = GAP_THRESHOLDS.get(asset_class, 0.02)
    gap_detected: bool = gap_size > threshold

    return gap_detected, gap_size


def classify_session(session_tag: str) -> str:
    """
    Validate and return the canonical session tag.

    FAS: "detect_session(timestamp, asset_class) -> Returns current session"
    This function handles the pure classification step (tag -> canonical form).
    Timestamp-to-session mapping is the caller's responsibility; it requires
    a wall clock or bar timestamp that the caller supplies.

    Args:
        session_tag: Raw session tag string.

    Returns:
        Canonical session tag (one of VALID_SESSION_TAGS).
        Returns "UNKNOWN" if session_tag is not in VALID_SESSION_TAGS.

    Raises:
        Nothing. Unrecognised values are normalised to "UNKNOWN".
    """
    if session_tag in VALID_SESSION_TAGS:
        return session_tag
    return "UNKNOWN"


def check_sequence(prev_id: int, current_id: int) -> None:
    """
    Enforce monotonically increasing sequence_id invariant.

    FAS: "sequence_id: int -- Monoton steigend -- fuer Kontinuitaets-Validierung"

    Args:
        prev_id:    sequence_id of the preceding data object. Must be >= 0.
        current_id: sequence_id of the current data object. Must be > prev_id.

    Raises:
        SequenceError if current_id <= prev_id (regression or duplicate).
        ValueError if either argument is negative.
    """
    if prev_id < 0:
        raise ValueError(
            "check_sequence: prev_id must be >= 0, got: " + repr(prev_id)
        )
    if current_id < 0:
        raise ValueError(
            "check_sequence: current_id must be >= 0, got: " + repr(current_id)
        )
    if current_id <= prev_id:
        raise SequenceError(
            "Sequence regression detected: current_id=" + repr(current_id)
            + " <= prev_id=" + repr(prev_id)
            + ". Data stream may be corrupted or out-of-order."
        )


def estimate_quality_score(
    nan_ratio:        float,
    completeness:     float,
    session_factor:   float,
    liquidity_factor: float,
) -> float:
    """
    Compute quality_score from sub-components.

    FAS multi-asset quality formula:
      quality_score = base_score * (1 - nan_ratio) * session_factor * liquidity_factor
    where base_score = completeness (ratio of non-missing bars).

    All inputs must be in [0.0, 1.0] and finite.
    Output is clamped to [0.0, 1.0].

    Args:
        nan_ratio:        Fraction of NaN values in the data window. [0.0, 1.0]
        completeness:     Fraction of expected bars present. [0.0, 1.0]
        session_factor:   Session quality multiplier. 1.0 for primary sessions,
                          < 1.0 for off-hours or thin sessions. [0.0, 1.0]
        liquidity_factor: Liquidity quality multiplier. 1.0 for normal/high,
                          < 1.0 for low liquidity. [0.0, 1.0]

    Returns:
        quality_score in [0.0, 1.0].

    Raises:
        NumericalInstabilityError if any input is NaN or Inf.
        ValueError if any input is outside [0.0, 1.0].
    """
    for name, value in [
        ("nan_ratio",        nan_ratio),
        ("completeness",     completeness),
        ("session_factor",   session_factor),
        ("liquidity_factor", liquidity_factor),
    ]:
        validate_numeric_field(name, value)
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                "estimate_quality_score: " + name
                + " must be in [0.0, 1.0], got: " + repr(value)
            )

    raw: float = completeness * (1.0 - nan_ratio) * session_factor * liquidity_factor

    # Clamp to [0.0, 1.0] to absorb floating-point drift from multiplication.
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


# =============================================================================
# SECTION 9 -- DATACACHE (EXPLICIT-CLOCK, IN-MEMORY ONLY)
# =============================================================================

@dataclass
class _CacheEntry:
    """
    Internal cache entry. Not part of the public API.

    expiry_step: the now_step value at or after which this entry is considered
    expired and must be evicted. Expiry check: now_step >= expiry_step.
    """
    value:        Any
    expiry_step:  int


class DataCache:
    """
    In-memory key-value cache with deterministic TTL via caller-supplied
    monotonic step counter.

    FAS: "cache_data(key, data, ttl: int) -> None"
         "get_cached(key) -> Optional[Any]"

    DETERMINISM:
      No wall-clock access. now_step is always supplied by the caller.
      In backtest contexts now_step is the bar index.
      In live contexts now_step is any monotonically increasing integer
      (e.g. message sequence number).

    TTL SEMANTICS:
      store(key, value, ttl_steps=60, now_step=N) stores with expiry_step = N + 60.
      retrieve(key, now_step=M) returns value if M < N + 60, else None.
      The entry is evicted on first failed retrieve (lazy eviction).

    ISOLATION:
      Each DataCache instance has its own independent _store dict.
      No class-level or module-level shared state exists.
      Multiple DataCache instances do not interact.

    NO IO:
      All storage is in-process memory. No Redis, no disk, no network.
      FAS cache strategy (Redis, LRU) describes the production deployment
      target; this module provides the pure-computation contract.
    """

    def __init__(self) -> None:
        # Instance-level store. No module-level or class-level mutable state.
        self._store: Dict[str, _CacheEntry] = {}

    def store(
        self,
        key:        str,
        value:      Any,
        ttl_steps:  int,
        now_step:   int,
    ) -> None:
        """
        Store a value in the cache.

        Args:
            key:        Cache key. Non-empty string.
            value:      Value to cache. Any type; caller is responsible for
                        ensuring the value is serialisable if persistence is
                        required (outside this module's scope).
            ttl_steps:  Time-to-live in logical steps. Must be > 0.
            now_step:   Current logical step counter. Must be >= 0.

        Raises:
            ValueError if key is empty, ttl_steps <= 0, or now_step < 0.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("DataCache.store: key must be a non-empty string")
        if not isinstance(ttl_steps, int) or ttl_steps <= 0:
            raise ValueError(
                "DataCache.store: ttl_steps must be a positive int, got: "
                + repr(ttl_steps)
            )
        if not isinstance(now_step, int) or now_step < 0:
            raise ValueError(
                "DataCache.store: now_step must be a non-negative int, got: "
                + repr(now_step)
            )

        expiry_step: int = now_step + ttl_steps
        self._store[key] = _CacheEntry(value=value, expiry_step=expiry_step)

    def retrieve(
        self,
        key:       str,
        now_step:  int,
    ) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Returns the stored value if the entry exists and has not expired.
        Evicts the entry on expiry and returns None.
        Returns None if the key is not present.

        Expiry condition: now_step >= expiry_step.

        Args:
            key:       Cache key.
            now_step:  Current logical step counter. Must be >= 0.

        Returns:
            Cached value, or None if absent or expired.

        Raises:
            ValueError if now_step < 0.
        """
        if not isinstance(now_step, int) or now_step < 0:
            raise ValueError(
                "DataCache.retrieve: now_step must be a non-negative int, got: "
                + repr(now_step)
            )

        entry: Optional[_CacheEntry] = self._store.get(key)
        if entry is None:
            return None

        if now_step >= entry.expiry_step:
            # Lazy eviction on first expired access.
            del self._store[key]
            return None

        return entry.value

    def invalidate(self, key: str) -> None:
        """
        Explicitly evict a cache entry.

        No-op if the key does not exist.

        Args:
            key: Cache key to remove.
        """
        self._store.pop(key, None)

    def size(self) -> int:
        """
        Return the number of entries currently in the cache.

        Includes entries that may be logically expired but have not yet been
        lazily evicted. Use purge_expired() to remove all stale entries.

        Returns:
            Non-negative integer count of entries.
        """
        return len(self._store)

    def purge_expired(self, now_step: int) -> int:
        """
        Eagerly evict all entries whose expiry_step <= now_step.

        This is an optional maintenance operation. The cache is correct
        without calling it (lazy eviction handles correctness). This method
        is provided for callers that need to bound memory usage.

        Args:
            now_step: Current logical step counter. Must be >= 0.

        Returns:
            Number of entries evicted.

        Raises:
            ValueError if now_step < 0.
        """
        if not isinstance(now_step, int) or now_step < 0:
            raise ValueError(
                "DataCache.purge_expired: now_step must be a non-negative int, got: "
                + repr(now_step)
            )

        expired_keys: List[str] = [
            k for k, entry in self._store.items()
            if now_step >= entry.expiry_step
        ]
        for k in expired_keys:
            del self._store[k]

        return len(expired_keys)


# =============================================================================
# SECTION 10 -- MODULE __all__
# =============================================================================

__all__ = [
    # Exceptions
    "NumericalInstabilityError",
    "DataQualityError",
    "SequenceError",
    # Data structures
    "OHLCV",
    "MarketData",
    "EnhancedMarketData",
    "ValidationResult",
    # Cache
    "DataCache",
    # Pure functions
    "validate_numeric_field",
    "validate_ohlcv",
    "validate_enhanced_market_data",
    "compute_gap",
    "classify_session",
    "check_sequence",
    "estimate_quality_score",
    # Constants
    "QUALITY_HARD_GATE",
    "LIQUIDITY_QUALITY_MINIMUM",
    "GAP_THRESHOLDS",
    "VALID_ASSET_CLASSES",
    "VALID_TIMEFRAMES",
    "VALID_SESSION_TAGS",
    "VALID_DATA_SOURCES",
]
