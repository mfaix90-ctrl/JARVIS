# jarvis/execution/exposure_router.py
# Version: 1.0.0
# External boundary adapter module.
# External to jarvis/core/ and jarvis/risk/ per FAS v6.1.0 architecture rules.
#
# DETERMINISM GUARANTEE:
#   No stochastic operations. No random number generation. No sampling.
#   No external state reads. No side effects. No file I/O. No logging.
#   No environment variable access. No global mutable state.
#   Output is a pure function of inputs.
#
# PURPOSE:
#   Acts as a strict boundary adapter between risk exposure output and
#   portfolio allocation. Delegates all allocation logic exclusively to
#   jarvis.portfolio.allocate_positions.
#   No allocation logic is reimplemented here.
#
# Standard import pattern:
#   from jarvis.execution.exposure_router import route_exposure_to_positions

from jarvis.portfolio import allocate_positions


def route_exposure_to_positions(
    total_capital: float,
    exposure_fraction: float,
    asset_prices: dict[str, float],
) -> dict[str, float]:
    """
    Boundary adapter between risk exposure and portfolio allocation.

    Validates exposure_fraction, then delegates to
    jarvis.portfolio.allocate_positions. Does not reimplement
    allocation logic.

    Parameters
    ----------
    total_capital : float
        Total portfolio capital available. Must be strictly positive.
        Validated downstream by allocate_positions.
    exposure_fraction : float
        Fraction of total_capital to allocate across all assets.
        Must be in [0.0, 1.0]. Validated here before delegation.
    asset_prices : dict[str, float]
        Mapping of asset symbol to current price.
        Must be non-empty. All prices must be strictly positive.
        Validated downstream by allocate_positions.

    Returns
    -------
    dict[str, float]
        Mapping of asset symbol to position size (number of units).
        Returned unchanged from allocate_positions.

    Raises
    ------
    ValueError
        If exposure_fraction < 0.0 or exposure_fraction > 1.0.
        If total_capital <= 0 (raised by allocate_positions).
        If asset_prices is empty (raised by allocate_positions).
        If any asset price <= 0 (raised by allocate_positions).

    Notes
    -----
    Deterministic. No side effects. No global state. Pure function.
    """
    if exposure_fraction < 0.0 or exposure_fraction > 1.0:
        raise ValueError(
            f"exposure_fraction must be in [0.0, 1.0]. Received: {exposure_fraction}"
        )

    positions: dict[str, float] = allocate_positions(
        total_capital=total_capital,
        exposure_fraction=exposure_fraction,
        asset_prices=asset_prices,
    )

    return positions
