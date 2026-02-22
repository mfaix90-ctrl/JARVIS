# =============================================================================
# JARVIS v6.1.0 -- PHASE 7B: RISK & CAPITAL MANAGEMENT LAYER
# File:   jarvis/core/risk_layer/evaluator.py
# Authority: JARVIS FAS v6.1.0 -- Phase 7B, Risk Evaluation Engine
# =============================================================================
#
# SCOPE
# -----
# Implements the risk evaluation entry points for Phase 7B.
# Exactly three public symbols are defined:
#
#   RiskDecision            -- frozen dataclass; output of all evaluation calls.
#   evaluate_position_risk  -- position-level risk check.
#   evaluate_portfolio_risk -- portfolio-level risk check (no position input).
#
# WHAT IS NOT IN THIS FILE
# ------------------------
#   No sizing logic.
#   No order execution logic.
#   No volatility-adjusted caps (Phase 7C+).
#   No liquidity haircuts (Phase 7C+).
#   No Kelly sizing (Phase 7C+).
#   No I/O of any kind.
#   No logging.
#   No global mutable state.
#
# DRAWDOWN THRESHOLD DERIVATION
# ------------------------------
# PortfolioState carries nav, peak_nav, and realized_drawdown_pct.
# RiskParameters carries max_drawdown_hard_stop and max_drawdown_soft_warn
# as fractions of peak_nav (e.g. 0.10 = 10% drawdown from peak).
#
# The spec references hard_stop_nav and soft_warn_nav as absolute NAV levels.
# These are not stored fields; they are derived per evaluation call:
#
#   hard_stop_nav = peak_nav * (1.0 - max_drawdown_hard_stop)
#   soft_warn_nav = peak_nav * (1.0 - max_drawdown_soft_warn)
#
# Comparison is then:
#   nav <= hard_stop_nav  ->  HALT      (hard stop; all new risk halted)
#   nav <= soft_warn_nav  ->  REDUCE    (soft warn; reduce sizing)
#   otherwise             ->  APPROVE   (within limits)
#
# Hard stop is checked before soft warn (fail-fast; most severe wins).
#
# VERDICT MAPPING
# ---------------
# The spec names HARD_STOP and SOFT_WARN, which do not match the RiskVerdict
# enum defined in domain.py. The correct mapping is:
#
#   Spec HARD_STOP  ->  RiskVerdict.HALT
#   Spec SOFT_WARN  ->  RiskVerdict.REDUCE
#   Spec OK         ->  RiskVerdict.APPROVE
#
# RiskVerdict is the canonical enum; spec shorthand labels are not exposed.
#
# DETERMINISM CONSTRAINTS
# -----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly. No module-level mutable reads.
# DET-03  No side effects. RiskDecision is frozen; inputs are never mutated.
# DET-04  All comparisons are deterministic arithmetic on validated floats.
# DET-05  No datetime.now() / time.time().
# DET-06  No random / secrets / uuid.
#
# PROHIBITED ACTIONS CONFIRMED ABSENT
# ------------------------------------
#   No numpy / scipy
#   No logging module
#   No datetime.now() / time.time()
#   No random / secrets / uuid
#   No file IO / network IO
#   No mutation of any input argument
#   No global or module-level mutable state
#   No circular imports (imports only from sibling modules and data_layer)
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from jarvis.core.data_layer import VALID_ASSET_CLASSES

from .domain import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    RiskVerdict,
)
from .exceptions import RiskValidationError


# =============================================================================
# SECTION 1 -- RISK DECISION (OUTPUT CONTRACT)
# =============================================================================

@dataclass(frozen=True)
class RiskDecision:
    """
    Immutable output of a risk evaluation call.

    Produced by evaluate_position_risk() and evaluate_portfolio_risk().
    Never constructed directly by callers of the public API; always returned
    by one of the evaluation functions.

    Attributes:
        verdict:             The risk engine's decision. Always a RiskVerdict
                             member. Never None.
        messages:            Ordered tuple of human-readable diagnostic strings.
                             Empty tuple when no messages apply. Never a list.
                             Each string is ASCII-safe and non-empty.
        max_position_size:   Optional USD cap on position size this cycle.
                             None when no explicit cap is being enforced by
                             this evaluation (e.g. cap logic is Phase 7C+).
                             When present: finite, > 0.
        requires_rebalance:  True when the portfolio state indicates that
                             existing positions should be rebalanced before
                             new risk is taken. False in the initial engine.

    Invariants:
        INV-RD-01  verdict is a RiskVerdict member (enforced by Python enum).
        INV-RD-02  messages is a tuple, never a list.
        INV-RD-03  frozen=True -- no field may be mutated after construction.
        INV-RD-04  max_position_size is None or a finite positive float.

    No validation is performed in __post_init__. RiskDecision is a pure
    value object; its correctness is the responsibility of the evaluation
    functions that construct it.
    """

    verdict:            RiskVerdict
    messages:           Tuple[str, ...]
    max_position_size:  Optional[float]
    requires_rebalance: bool


# =============================================================================
# SECTION 2 -- INTERNAL HELPERS (module-private)
# =============================================================================

def _compute_verdict(
    nav:                    float,
    peak_nav:               float,
    max_drawdown_hard_stop: float,
    max_drawdown_soft_warn: float,
) -> RiskVerdict:
    """
    Derive a RiskVerdict from current NAV and drawdown thresholds.

    All arguments are assumed to have been validated by the domain layer
    prior to this call. No re-validation is performed here.

    Hard stop is evaluated before soft warn (most severe condition wins).

    Threshold derivation:
        hard_stop_nav = peak_nav * (1.0 - max_drawdown_hard_stop)
        soft_warn_nav = peak_nav * (1.0 - max_drawdown_soft_warn)

    Args:
        nav:                    Current portfolio net asset value. > 0.
        peak_nav:               High-water mark NAV. >= nav, > 0.
        max_drawdown_hard_stop: Hard stop fraction in (0, 1). > soft_warn.
        max_drawdown_soft_warn: Soft warn fraction in (0, 1). < hard_stop.

    Returns:
        RiskVerdict.HALT    if nav <= hard_stop_nav
        RiskVerdict.REDUCE  if nav <= soft_warn_nav
        RiskVerdict.APPROVE otherwise
    """
    hard_stop_nav: float = peak_nav * (1.0 - max_drawdown_hard_stop)
    soft_warn_nav: float = peak_nav * (1.0 - max_drawdown_soft_warn)

    if nav <= hard_stop_nav:
        return RiskVerdict.HALT
    if nav <= soft_warn_nav:
        return RiskVerdict.REDUCE
    return RiskVerdict.APPROVE


def _validate_asset_class(asset_class: str) -> None:
    """
    Raise RiskValidationError if asset_class is not in VALID_ASSET_CLASSES.

    This is a belt-and-suspenders check. PositionSpec.__post_init__ already
    enforces this; the check here guards against callers that construct
    PositionSpec via object.__setattr__ bypass or future refactoring.

    Args:
        asset_class: The asset_class string from a PositionSpec.

    Raises:
        RiskValidationError if asset_class is not a known asset class.
    """
    if asset_class not in VALID_ASSET_CLASSES:
        raise RiskValidationError(
            field_name="asset_class",
            value=asset_class,
            constraint=(
                "must be in VALID_ASSET_CLASSES: "
                + repr(sorted(VALID_ASSET_CLASSES))
            ),
        )


# =============================================================================
# SECTION 3 -- EVALUATE POSITION RISK
# =============================================================================

def evaluate_position_risk(
    position:  PositionSpec,
    portfolio: PortfolioState,
    params:    RiskParameters,
) -> RiskDecision:
    """
    Evaluate risk for a proposed position against the current portfolio state
    and risk parameter configuration.

    This is a pure function: it reads its three arguments and returns a
    RiskDecision. It does not mutate any input, access global state, perform
    I/O, or produce side effects.

    Evaluation order (fail-fast; first match wins for verdict):
        1. Asset class validation (raises RiskValidationError -- not a verdict).
        2. Hard stop check: if nav <= peak_nav * (1 - hard_stop) -> HALT.
        3. Soft warn check: if nav <= peak_nav * (1 - soft_warn) -> REDUCE.
        4. No breach: -> APPROVE.

    Note on asset class check:
        An invalid asset_class raises rather than returning a REJECT verdict
        because it indicates a programming error in the caller, not a market
        condition. A verdict implies the engine processed the request; an
        exception implies the request was malformed.

    Args:
        position:  The proposed position to evaluate. Must be a valid
                   PositionSpec (already validated by its constructor).
        portfolio: Current portfolio snapshot. Must be a valid PortfolioState.
        params:    Risk configuration. Must be a valid RiskParameters.

    Returns:
        RiskDecision with:
            verdict:            HALT | REDUCE | APPROVE
            messages:           () -- empty for Phase 7B; populated in 7C+.
            max_position_size:  None -- cap logic is Phase 7C+.
            requires_rebalance: False -- rebalance logic is Phase 7C+.

    Raises:
        RiskValidationError: if position.asset_class is not in VALID_ASSET_CLASSES.
    """
    # --- Step 1: Asset class validation ---
    # Raises RiskValidationError on unknown asset class.
    # Must be first: a malformed position should never reach verdict logic.
    _validate_asset_class(position.asset_class)

    # --- Steps 2-4: Drawdown verdict ---
    verdict: RiskVerdict = _compute_verdict(
        nav=portfolio.nav,
        peak_nav=portfolio.peak_nav,
        max_drawdown_hard_stop=params.max_drawdown_hard_stop,
        max_drawdown_soft_warn=params.max_drawdown_soft_warn,
    )

    return RiskDecision(
        verdict=verdict,
        messages=(),
        max_position_size=None,
        requires_rebalance=False,
    )


# =============================================================================
# SECTION 4 -- EVALUATE PORTFOLIO RISK
# =============================================================================

def evaluate_portfolio_risk(
    portfolio: PortfolioState,
    params:    RiskParameters,
) -> RiskDecision:
    """
    Evaluate risk at the portfolio level, without reference to any specific
    proposed position.

    Used to determine the overall health of the portfolio before any position
    evaluation is attempted. Callers may check the portfolio verdict first
    and skip evaluate_position_risk entirely when verdict is HALT.

    This is a pure function with the same determinism and no-side-effect
    guarantees as evaluate_position_risk.

    Evaluation order:
        1. Hard stop check: if nav <= peak_nav * (1 - hard_stop) -> HALT.
        2. Soft warn check: if nav <= peak_nav * (1 - soft_warn) -> REDUCE.
        3. No breach: -> APPROVE.

    Args:
        portfolio: Current portfolio snapshot. Must be a valid PortfolioState.
        params:    Risk configuration. Must be a valid RiskParameters.

    Returns:
        RiskDecision with:
            verdict:            HALT | REDUCE | APPROVE
            messages:           () -- empty for Phase 7B; populated in 7C+.
            max_position_size:  None -- cap logic is Phase 7C+.
            requires_rebalance: False -- rebalance logic is Phase 7C+.

    Raises:
        Nothing beyond what PortfolioState and RiskParameters constructors
        already enforce. All inputs are pre-validated by their frozen
        dataclass constructors.
    """
    verdict: RiskVerdict = _compute_verdict(
        nav=portfolio.nav,
        peak_nav=portfolio.peak_nav,
        max_drawdown_hard_stop=params.max_drawdown_hard_stop,
        max_drawdown_soft_warn=params.max_drawdown_soft_warn,
    )

    return RiskDecision(
        verdict=verdict,
        messages=(),
        max_position_size=None,
        requires_rebalance=False,
    )


# =============================================================================
# SECTION 5 -- MODULE __all__
# =============================================================================

__all__ = [
    "RiskDecision",
    "evaluate_position_risk",
    "evaluate_portfolio_risk",
]
