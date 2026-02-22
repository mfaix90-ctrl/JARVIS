# =============================================================================
# JARVIS v6.1.0 -- PHASE 8: EXECUTION GUARD
# File:   jarvis/core/execution_guard.py
# =============================================================================
#
# PURPOSE
# -------
# Strict boundary adapter between strategy output and the risk subsystem.
# Translates a PositionSizingResult into an ExecutionOrder, or suppresses
# the order entirely when the risk layer disallows the trade.
#
# This module contains no risk logic. It has no knowledge of drawdown
# thresholds, position caps, or sizing arithmetic. All policy decisions are
# delegated entirely to assess_trade() in jarvis.core.risk_layer.engine.
#
# WHAT IS NOT IN THIS FILE
# ------------------------
#   No cap calculations.
#   No threshold knowledge.
#   No exception wrapping.
#   No logging.
#   No mutation of inputs.
#   No reimplementation of evaluator or sizing logic.
#
# DETERMINISM CONSTRAINTS
# -----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly. No module-level mutable reads.
# DET-03  No side effects. ExecutionOrder is frozen; inputs never mutated.
# DET-04  No datetime.now() / time.time().
# DET-05  No random / secrets / uuid.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from jarvis.core.risk_layer import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    Side,
)
from jarvis.core.risk_layer.engine import assess_trade


# =============================================================================
# SECTION 1 -- EXECUTION ORDER
# =============================================================================

@dataclass(frozen=True)
class ExecutionOrder:
    """
    Immutable instruction for downstream execution infrastructure.

    Produced by build_execution_order() when the risk layer approves the trade.
    Contains only the information required to route and size the order.

    Attributes:
        symbol:           Instrument identifier. Echoed from PositionSpec.
        side:             Trade direction. Echoed from PositionSpec.
        target_notional:  Approved USD notional. Sourced from PositionSizingResult.
                          Always finite and > 0 when this object exists.
    """

    symbol:           str
    side:             Side
    target_notional:  float


# =============================================================================
# SECTION 2 -- BUILD EXECUTION ORDER
# =============================================================================

def build_execution_order(
    position:  PositionSpec,
    portfolio: PortfolioState,
    params:    RiskParameters,
) -> Optional[ExecutionOrder]:
    """
    Translate a proposed position into an ExecutionOrder, subject to risk approval.

    Delegates all risk evaluation and sizing to assess_trade(). This function
    contains no policy knowledge; it only routes the result.

    Args:
        position:  The proposed position to evaluate.
        portfolio: Current portfolio snapshot.
        params:    Risk configuration.

    Returns:
        ExecutionOrder if the trade is allowed (result.allowed is True).
        None           if the trade is blocked  (result.allowed is False).
    """
    result = assess_trade(position, portfolio, params)

    if not result.allowed:
        return None

    return ExecutionOrder(
        symbol=position.symbol,
        side=position.side,
        target_notional=result.target_notional,
    )


# =============================================================================
# SECTION 3 -- MODULE __all__
# =============================================================================

__all__ = [
    "ExecutionOrder",
    "build_execution_order",
]
