from .domain import PortfolioState, PositionSpec, RiskParameters
from .evaluator import evaluate_position_risk
from .sizing import PositionSizingResult, size_position


def assess_trade(
    position:  PositionSpec,
    portfolio: PortfolioState,
    params:    RiskParameters,
) -> PositionSizingResult:
    """
    Orchestrate a full risk assessment and position sizing in one call.

    Delegates entirely to the evaluation and sizing layers:
        1. evaluate_position_risk(position, portfolio, params) -> RiskDecision
        2. size_position(position, portfolio, params, decision) -> PositionSizingResult

    Pure, deterministic, stateless, non-mutating.
    No branching, no validation, no threshold computation.
    All logic resides in evaluator.py and sizing.py.
    """
    decision = evaluate_position_risk(position, portfolio, params)
    return size_position(position, portfolio, params, decision)


__all__ = ["assess_trade"]
