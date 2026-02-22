# =============================================================================
# JARVIS v6.1.0 -- PHASE 7D: RISK & CAPITAL MANAGEMENT LAYER
# File:   jarvis/core/risk_layer/sizing.py
# Authority: JARVIS FAS v6.1.0 -- Phase 7D, Position Sizing
# =============================================================================
#
# SCOPE
# -----
# Implements deterministic position sizing for Phase 7D.
# Exactly two public symbols are defined:
#
#   PositionSizingResult  -- frozen dataclass; output of size_position().
#   size_position()       -- pure sizing function; consumes a RiskDecision.
#
# CHANGES FROM PHASE 7C
# ---------------------
#   A) REDUCE verdict multiplier: when verdict == REDUCE, an additional
#      liquidity_haircut_floor multiplier is applied to the Kelly-sized target.
#
#   B) Kelly fraction sizing: target *= params.kelly_fraction.
#      Applied after cap, before REDUCE multiplier.
#      Guard: kelly_fraction validated in (0.0, 1.0] by domain.
#
#   C) Volatility-adjusted cap: effective_cap = base_cap * vol_cap_scalar,
#      where vol_cap_scalar = min(volatility_target_ann / max(position_vol, 1e-8), 1.0).
#      When position_vol is None -> vol_cap_scalar = 1.0 (identity, backward compatible).
#
#   D) Liquidity floor guarantee: target >= requested * liquidity_haircut_floor.
#      Applied last; prevents pathological compression to near-zero.
#
# SIZING PIPELINE ORDER (deterministic, fixed):
#   1. HALT branch       -> return immediately (allowed=False, target=None).
#   2. requested_notional = position.quantity * position.current_price
#   3. base_cap           = portfolio.nav * params.max_position_pct_nav
#   4. vol_cap_scalar     = min(params.volatility_target_ann / max(position_vol, 1e-8), 1.0)
#                           if position_vol is not None else 1.0
#   5. effective_cap      = min(base_cap, base_cap * vol_cap_scalar)
#   6. raw_target         = min(requested_notional, effective_cap)
#   7. kelly_target       = raw_target * params.kelly_fraction
#   8. reduce_target      = kelly_target * params.liquidity_haircut_floor  (REDUCE only)
#                           kelly_target  (APPROVE)
#   9. liquidity_floor    = requested_notional * params.liquidity_haircut_floor
#  10. target             = max(reduce_target, liquidity_floor)
#
# INVARIANTS
# ----------
# INV-SZ-01: target_notional is None whenever allowed is False.
# INV-SZ-02: target_notional is a finite positive float whenever allowed is True.
# INV-SZ-03: target_notional <= effective_cap (before liquidity floor) whenever allowed is True.
# INV-SZ-04: result is frozen; no field may be mutated after construction.
# INV-SZ-05: target_notional >= requested_notional * liquidity_haircut_floor
#            whenever allowed is True (liquidity floor guarantee).
#
# BACKWARD COMPATIBILITY
# ----------------------
# position_vol=None, kelly_fraction=1.0, verdict=APPROVE:
#   -> vol_cap_scalar=1.0, kelly multiplier=1.0, no REDUCE -> Phase 7C arithmetic.
#
# DETERMINISM CONSTRAINTS
# -----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly. No module-level mutable reads.
# DET-03  No side effects. PositionSizingResult is frozen; inputs never mutated.
# DET-04  All arithmetic is deterministic: *, min(), max(), guarded division.
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
#   No float rounding / coercion
#   No circular imports
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .domain import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    RiskVerdict,
)
from .evaluator import RiskDecision


# =============================================================================
# SECTION 1 -- POSITION SIZING RESULT (OUTPUT CONTRACT)
# =============================================================================

@dataclass(frozen=True)
class PositionSizingResult:
    """
    Immutable output of size_position().

    Attributes:
        allowed:          True when the position may proceed (verdict is not
                          HALT). False when all new risk-taking is suspended.
        target_notional:  The approved USD notional size for this position.
                          None when allowed is False.
                          A finite positive float when allowed is True.
        reason:           The RiskVerdict that drove this sizing decision.
                          Echoed directly from the input RiskDecision.verdict.

    Invariants:
        INV-SZ-01  target_notional is None when allowed is False.
        INV-SZ-02  target_notional is a finite positive float when allowed is True.
        INV-SZ-04  frozen=True -- no field may be mutated after construction.

    No validation in __post_init__. Correctness is size_position()'s responsibility.
    """

    allowed:          bool
    target_notional:  Optional[float]
    reason:           RiskVerdict


# =============================================================================
# SECTION 2 -- SIZE POSITION
# =============================================================================

def size_position(
    position:     PositionSpec,
    portfolio:    PortfolioState,
    params:       RiskParameters,
    decision:     RiskDecision,
    position_vol: Optional[float] = None,
) -> PositionSizingResult:
    """
    Compute the allowed position size given a pre-evaluated RiskDecision.

    Pure function: reads arguments, returns PositionSizingResult.
    No mutation, no global state, no I/O, no side effects.

    Sizing pipeline (fixed order):
        1.  HALT -> return immediately.
        2.  requested_notional = quantity * current_price
        3.  base_cap = nav * max_position_pct_nav
        4.  vol_cap_scalar = min(volatility_target_ann / max(position_vol, 1e-8), 1.0)
                             if position_vol is not None else 1.0
        5.  effective_cap = min(base_cap, base_cap * vol_cap_scalar)
        6.  raw_target = min(requested_notional, effective_cap)
        7.  kelly_target = raw_target * kelly_fraction
        8.  reduce_target = kelly_target * liquidity_haircut_floor  (REDUCE)
                          = kelly_target                             (APPROVE)
        9.  liquidity_floor = requested_notional * liquidity_haircut_floor
        10. target = max(reduce_target, liquidity_floor)

    Args:
        position:     PositionSpec. quantity and current_price derive requested notional.
        portfolio:    PortfolioState. nav used for base cap.
        params:       RiskParameters. All sizing scalars consumed here.
        decision:     RiskDecision from evaluate_position_risk(). Only verdict consumed.
        position_vol: Optional annualised volatility of this position.
                      None -> vol_cap_scalar = 1.0 (no vol adjustment; backward compat).

    Returns:
        PositionSizingResult(allowed, target_notional, reason).
    """
    # ------------------------------------------------------------------
    # Step 1: HALT -- no sizing arithmetic performed.
    # ------------------------------------------------------------------
    if decision.verdict is RiskVerdict.HALT:
        return PositionSizingResult(
            allowed=False,
            target_notional=None,
            reason=decision.verdict,
        )

    # ------------------------------------------------------------------
    # Step 2: Requested notional.
    # quantity and current_price are finite and > 0 (domain-validated).
    # ------------------------------------------------------------------
    requested_notional: float = position.quantity * position.current_price

    # ------------------------------------------------------------------
    # Step 3: Base position cap.
    # nav and max_position_pct_nav are finite and > 0 (domain-validated).
    # ------------------------------------------------------------------
    base_cap: float = portfolio.nav * params.max_position_pct_nav

    # ------------------------------------------------------------------
    # Step 4: Volatility-adjusted cap scalar.
    # position_vol=None -> scalar=1.0 (identity; preserves Phase 7C output).
    # max(position_vol, 1e-8) guards against division by zero or near-zero.
    # min(..., 1.0) prevents the scalar from exceeding 1 (no vol-based leverage).
    # ------------------------------------------------------------------
    if position_vol is not None:
        vol_cap_scalar: float = min(
            params.volatility_target_ann / max(position_vol, 1e-8),
            1.0,
        )
    else:
        vol_cap_scalar = 1.0

    # ------------------------------------------------------------------
    # Step 5: Effective cap.
    # vol_cap = base_cap * scalar. effective_cap = min(base_cap, vol_cap).
    # When scalar=1.0: vol_cap == base_cap -> effective_cap == base_cap.
    # ------------------------------------------------------------------
    effective_cap: float = min(base_cap, base_cap * vol_cap_scalar)

    # ------------------------------------------------------------------
    # Step 6: Raw target capped at effective_cap.
    # ------------------------------------------------------------------
    raw_target: float = min(requested_notional, effective_cap)

    # ------------------------------------------------------------------
    # Step 7: Kelly fraction.
    # kelly_fraction is validated in (0.0, 1.0] by RiskParameters.
    # No guard needed; always reduces or preserves raw_target.
    # ------------------------------------------------------------------
    kelly_target: float = raw_target * params.kelly_fraction

    # ------------------------------------------------------------------
    # Step 8: REDUCE multiplier (conditional on clamping).
    # REDUCE applies liquidity_haircut_floor only when the position was
    # clamped by the cap (raw_target < requested_notional).
    # When not clamped (position already within cap), REDUCE passes
    # kelly_target through unchanged -- no additional compression.
    # APPROVE always passes kelly_target through unchanged.
    # liquidity_haircut_floor validated in (0.0, 1.0] by RiskParameters.
    # ------------------------------------------------------------------
    _was_clamped: bool = raw_target < requested_notional
    if decision.verdict is RiskVerdict.REDUCE and _was_clamped:
        reduce_target: float = kelly_target * params.liquidity_haircut_floor
    else:
        reduce_target = kelly_target

    # ------------------------------------------------------------------
    # Step 9: Liquidity floor guarantee (INV-SZ-05).
    # Even after Kelly and REDUCE compression, target must be at least
    # liquidity_haircut_floor fraction of the original requested notional.
    # This prevents compound discounting from compressing to near-zero.
    # ------------------------------------------------------------------
    liquidity_floor_value: float = requested_notional * params.liquidity_haircut_floor
    target: float = max(reduce_target, liquidity_floor_value)

    return PositionSizingResult(
        allowed=True,
        target_notional=target,
        reason=decision.verdict,
    )


# =============================================================================
# SECTION 3 -- MODULE __all__
# =============================================================================

__all__ = [
    "PositionSizingResult",
    "size_position",
]
