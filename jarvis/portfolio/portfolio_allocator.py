# jarvis/portfolio/portfolio_allocator.py
# Version: 1.0.0
# External portfolio allocation module.
# External to jarvis/core/ and jarvis/risk/ per FAS v6.1.0 architecture rules.
#
# DETERMINISM GUARANTEE:
#   No stochastic operations. No random number generation. No sampling.
#   No external state reads. No side effects. No file I/O. No logging.
#   No environment variable access. No global mutable state.
#   Output is a pure function of inputs.
#
# Standard import pattern:
#   from jarvis.portfolio.portfolio_allocator import allocate_positions


def allocate_positions(
    total_capital: float,
    exposure_fraction: float,
    asset_prices: dict[str, float],
) -> dict[str, float]:
    """
    Compute equal-weight position sizes for a set of assets.

    Parameters
    ----------
    total_capital : float
        Total portfolio capital available. Must be strictly positive.
    exposure_fraction : float
        Fraction of total_capital to allocate across all assets.
        Must be in [0.0, 1.0].
    asset_prices : dict[str, float]
        Mapping of asset symbol to current price.
        Must be non-empty. All prices must be strictly positive.

    Returns
    -------
    dict[str, float]
        Mapping of asset symbol to position size (number of units).
        Position size = equal_weight_capital / asset_price.
        When exposure_fraction is 0.0, all position sizes are 0.0.

    Raises
    ------
    ValueError
        If total_capital <= 0.
        If exposure_fraction < 0 or exposure_fraction > 1.
        If asset_prices is empty.
        If any asset price <= 0.

    Notes
    -----
    Deterministic. No side effects. No global state. Pure function.
    """
    # ------------------------------------------------------------------
    # Input validation.
    # ------------------------------------------------------------------
    if total_capital <= 0.0:
        raise ValueError(
            f"total_capital must be strictly positive. Received: {total_capital}"
        )

    if exposure_fraction < 0.0 or exposure_fraction > 1.0:
        raise ValueError(
            f"exposure_fraction must be in [0.0, 1.0]. Received: {exposure_fraction}"
        )

    if not asset_prices:
        raise ValueError(
            "asset_prices must be non-empty. Received an empty dict."
        )

    invalid_prices: list[str] = [
        symbol for symbol, price in asset_prices.items() if price <= 0.0
    ]
    if invalid_prices:
        raise ValueError(
            f"All asset prices must be strictly positive. "
            f"Non-positive prices found for: {invalid_prices}"
        )

    # ------------------------------------------------------------------
    # Computation.
    # ------------------------------------------------------------------
    number_of_assets: int = len(asset_prices)

    allocated_capital: float = total_capital * exposure_fraction

    equal_weight_capital: float = allocated_capital / number_of_assets

    positions: dict[str, float] = {
        symbol: equal_weight_capital / price
        for symbol, price in asset_prices.items()
    }

    return positions
