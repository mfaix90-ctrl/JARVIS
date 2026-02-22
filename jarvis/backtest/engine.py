# jarvis/backtest/engine.py
# Version: 1.0.0
# External deterministic backtest layer.
# External to jarvis/core/, jarvis/risk/, jarvis/portfolio/,
# jarvis/execution/, jarvis/orchestrator/ per FAS v6.1.0 architecture rules.
#
# DETERMINISM GUARANTEE:
#   No stochastic operations. No random number generation. No sampling.
#   No external state reads. No side effects. No file I/O. No logging.
#   No environment variable access. No global mutable state.
#   Output is a pure function of inputs.
#
# PURPOSE:
#   Rolls a sliding window over returns_series and asset_price_series,
#   delegating all risk and allocation logic to run_full_pipeline() at
#   each timestep. No risk or allocation logic is reimplemented here.
#
# DESIGN NOTE -- SYMBOL KEY:
#   run_full_pipeline() requires asset_prices: dict[str, float].
#   run_backtest() receives a scalar price series with no symbol names.
#   A fixed synthetic key "ASSET" is used to wrap each scalar price into
#   the required single-entry dict at every timestep. This is the only
#   valid deterministic approach that avoids reimplementing downstream
#   logic.
#
# EQUITY UPDATE CONTRACT:
#   equity[t] = equity[t-1] * (1.0 + returns_series[t] * position_size)
#   where position_size is the scalar extracted from the single-key
#   positions dict returned by run_full_pipeline().
#
# Standard import pattern:
#   from jarvis.backtest.engine import run_backtest

from jarvis.core.regime import GlobalRegimeState
from jarvis.orchestrator import run_full_pipeline

_SYNTHETIC_SYMBOL: str = "ASSET"


def run_backtest(
    returns_series: list[float],
    window: int,
    initial_capital: float,
    asset_price_series: list[float],
    regime: GlobalRegimeState,
    meta_uncertainty: float,
) -> list[float]:
    """
    Deterministic rolling-window backtest over a single-asset return series.

    For each timestep t >= window:
      1. Slices returns_series[t - window : t] as the lookback window.
      2. Wraps asset_price_series[t] into a single-entry dict under the
         synthetic key "ASSET".
      3. Calls run_full_pipeline() to obtain position size.
      4. Extracts the scalar position size from the returned dict.
      5. Updates equity: equity * (1.0 + returns_series[t] * position_size).
      6. Appends updated equity to the equity curve.

    All risk assessment and allocation logic is delegated to
    run_full_pipeline(). Nothing is reimplemented here.

    Parameters
    ----------
    returns_series : list[float]
        Sequence of per-period returns. Must have length >= window.
        Passed as the lookback window to run_full_pipeline() at each step.
        No NaN or Inf values permitted (validated downstream by RiskEngine).
    window : int
        Lookback window size in periods. Must be >= 20 (RiskEngine minimum).
    initial_capital : float
        Starting equity value. Must be strictly positive.
        Passed as total_capital to run_full_pipeline() at each step.
    asset_price_series : list[float]
        Per-period asset prices. Must have the same length as returns_series.
        Each scalar price is wrapped into a single-entry dict at each step.
        All prices must be strictly positive (validated downstream).
    regime : GlobalRegimeState
        Canonical GlobalRegimeState instance from jarvis.core.regime.
        Passed unchanged to run_full_pipeline() at every timestep.
    meta_uncertainty : float
        Meta-uncertainty in [0.0, 1.0].
        Passed unchanged to run_full_pipeline() at every timestep.

    Returns
    -------
    list[float]
        Equity curve. One entry per timestep t where t >= window.
        Length is len(returns_series) - window.
        Returns an empty list if len(returns_series) == window
        (no full step available beyond the first window).

    Raises
    ------
    ValueError
        If window < 20.
        If initial_capital <= 0.
        If len(asset_price_series) != len(returns_series).
        Additional ValueError may propagate from run_full_pipeline()
        on invalid returns or price values.

    Notes
    -----
    Deterministic. No side effects. No global state. Pure function.
    regime and meta_uncertainty are held constant across all timesteps.
    This matches the function signature contract. Dynamic regime
    sequencing is out of scope for this module.
    """
    # ------------------------------------------------------------------
    # Input validation.
    # ------------------------------------------------------------------
    if window < 20:
        raise ValueError(
            f"window must be >= 20 (RiskEngine minimum). Received: {window}"
        )

    if initial_capital <= 0.0:
        raise ValueError(
            f"initial_capital must be strictly positive. Received: {initial_capital}"
        )

    if len(asset_price_series) != len(returns_series):
        raise ValueError(
            f"asset_price_series and returns_series must have equal length. "
            f"Received: len(returns_series)={len(returns_series)}, "
            f"len(asset_price_series)={len(asset_price_series)}"
        )

    # ------------------------------------------------------------------
    # Rolling-window backtest loop.
    # ------------------------------------------------------------------
    equity: float = initial_capital
    equity_curve: list[float] = []

    n: int = len(returns_series)

    for t in range(window, n):
        window_returns: list[float] = returns_series[t - window : t]

        asset_prices: dict[str, float] = {
            _SYNTHETIC_SYMBOL: asset_price_series[t]
        }

        positions: dict[str, float] = run_full_pipeline(
            returns_history=window_returns,
            current_regime=regime,
            meta_uncertainty=meta_uncertainty,
            total_capital=equity,
            asset_prices=asset_prices,
        )

        position_size: float = positions[_SYNTHETIC_SYMBOL]

        updated_equity: float = equity * (1.0 + returns_series[t] * position_size)
        # Capital floor: equity must never be negative or zero per FAS contract.
        # A minimum of 1e-10 preserves the strictly-positive invariant required
        # by run_full_pipeline() at the next timestep.
        equity = max(1e-10, updated_equity)

        equity_curve.append(equity)

    return equity_curve
