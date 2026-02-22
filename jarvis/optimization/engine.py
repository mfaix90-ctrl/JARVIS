# jarvis/optimization/engine.py
# Version: 1.0.0
# External optimization layer.
# External to jarvis/core/, jarvis/risk/, jarvis/utils/, jarvis/portfolio/,
# jarvis/execution/, jarvis/orchestrator/, jarvis/backtest/,
# jarvis/walkforward/, jarvis/metrics/, jarvis/report/, jarvis/strategy/
# per FAS v6.1.0 architecture rules.
#
# DETERMINISM GUARANTEE:
#   No stochastic operations. No random number generation. No sampling.
#   No external state reads. No side effects. No file I/O. No logging.
#   No environment variable access. No global mutable state.
#   Output is a pure function of inputs.
#   Iteration order over parameter combinations is fully deterministic:
#   outer loop windows, middle loop steps, inner loop meta_uncertainties.
#   No sorting is applied. Input list order is preserved exactly.
#
# PURPOSE:
#   Iterates over the Cartesian product of windows x steps x
#   meta_uncertainties and calls run_strategy() for each combination.
#   Collects structured results in a list preserving iteration order.
#   No risk logic, allocation logic, backtest logic, metric formula,
#   or strategy logic is reimplemented here.
#
# Standard import pattern:
#   from jarvis.optimization.engine import run_optimization

from jarvis.core.regime import GlobalRegimeState
from jarvis.strategy import run_strategy


def run_optimization(
    returns_series: list[float],
    asset_price_series: list[float],
    windows: list[int],
    steps: list[int],
    meta_uncertainties: list[float],
    initial_capital: float,
    regime: GlobalRegimeState,
) -> list[dict]:
    """
    Iterate over parameter combinations and collect run_strategy() results.

    FAS COMPLIANCE:
      This function is an external layer under FAS v6.1.0. It satisfies
      all determinism guarantees (DET-01 through DET-07) and all
      prohibitions defined in ARCHITECTURE.md Section 4. Specifically:

      PROHIBITED-06 (no reimplementation):
        All strategy execution is delegated entirely to
        jarvis.strategy.run_strategy(). No risk, allocation, backtest,
        walkforward, metric, or strategy logic is reproduced, inlined,
        or approximated here. This function performs only parameter
        validation and iteration over the Cartesian product of the
        input parameter lists.

      PROHIBITED-01 (no stochastic operations):
        Iteration order is fully deterministic. No shuffling, no
        random sampling of parameter combinations, no probabilistic
        pruning of the search space.

      PROHIBITED-02 (no file I/O): No file is read or written.

      PROHIBITED-03 (no logging): No logging calls are made.

      PROHIBITED-05 (no global mutable state): No module-level variable
        is written to during execution.

      DET-02 (no external state reads): All inputs are passed explicitly.

      DET-03 (no side effects): Input lists are not mutated. All
        intermediate and output containers are constructed fresh per call.

    ITERATION ORDER CONTRACT:
      The Cartesian product is iterated as:
        for window in windows:
          for step in steps:
            for meta_uncertainty in meta_uncertainties:
      This order is fixed and deterministic. No sorting is applied at
      any stage. The output list preserves this order exactly.

    VALIDATION CONTRACT:
      All three parameter lists must be non-empty.
      All windows must be >= 20 (RiskEngine minimum).
      All steps must be >= 1.
      All meta_uncertainties must be in [0.0, 1.0].
      Validation is performed before any call to run_strategy().
      Additional ValueError may propagate from run_strategy() on
      invalid series lengths or capital values.

    Parameters
    ----------
    returns_series : list[float]
        Full sequence of per-period returns.
        Passed unchanged to run_strategy() for every combination.
        Not mutated by this function.
    asset_price_series : list[float]
        Full sequence of per-period asset prices.
        Passed unchanged to run_strategy() for every combination.
        Not mutated by this function.
    windows : list[int]
        Lookback window sizes to iterate over. Must be non-empty.
        Each value must be >= 20.
    steps : list[int]
        Step sizes to iterate over. Must be non-empty.
        Each value must be >= 1.
    meta_uncertainties : list[float]
        Meta-uncertainty values to iterate over. Must be non-empty.
        Each value must be in [0.0, 1.0].
    initial_capital : float
        Starting equity passed unchanged to run_strategy() for every
        combination. Validated downstream by run_strategy().
    regime : GlobalRegimeState
        Canonical GlobalRegimeState instance from jarvis.core.regime.
        Passed unchanged to run_strategy() for every combination.

    Returns
    -------
    list[dict]
        One entry per parameter combination in iteration order.
        Each entry is a dict with keys:
          "window"           -- int: the window value for this run.
          "step"             -- int: the step value for this run.
          "meta_uncertainty" -- float: the meta_uncertainty for this run.
          "result"           -- dict[str, object]: the full output of
                                run_strategy() for this combination,
                                containing "segments", "segment_metrics",
                                and "aggregate".
        Total length equals len(windows) * len(steps) * len(meta_uncertainties).

    Raises
    ------
    ValueError
        If windows is empty.
        If steps is empty.
        If meta_uncertainties is empty.
        If any value in windows is < 20.
        If any value in steps is < 1.
        If any value in meta_uncertainties is < 0.0 or > 1.0.
        Additional ValueError may propagate from run_strategy() on
        invalid series data or capital values.

    Notes
    -----
    Deterministic. No side effects. No global state. Pure function.
    Input lists are not mutated. All output containers are fresh.
    periods_per_year is not exposed as a parameter in this layer.
    run_strategy() is called with its default periods_per_year=252.
    """
    # ------------------------------------------------------------------
    # Input validation.
    # ------------------------------------------------------------------
    if not windows:
        raise ValueError("windows must not be empty.")

    if not steps:
        raise ValueError("steps must not be empty.")

    if not meta_uncertainties:
        raise ValueError("meta_uncertainties must not be empty.")

    invalid_windows: list[int] = [w for w in windows if w < 20]
    if invalid_windows:
        raise ValueError(
            f"All windows must be >= 20 (RiskEngine minimum). "
            f"Invalid values: {invalid_windows}"
        )

    invalid_steps: list[int] = [s for s in steps if s < 1]
    if invalid_steps:
        raise ValueError(
            f"All steps must be >= 1. "
            f"Invalid values: {invalid_steps}"
        )

    invalid_mus: list[float] = [
        mu for mu in meta_uncertainties if mu < 0.0 or mu > 1.0
    ]
    if invalid_mus:
        raise ValueError(
            f"All meta_uncertainties must be in [0.0, 1.0]. "
            f"Invalid values: {invalid_mus}"
        )

    # ------------------------------------------------------------------
    # Cartesian product iteration -- order: windows x steps x meta_uncertainties.
    # ------------------------------------------------------------------
    results: list[dict] = []

    for window in windows:
        for step in steps:
            for meta_uncertainty in meta_uncertainties:
                strategy_output: dict = run_strategy(
                    returns_series=returns_series,
                    asset_price_series=asset_price_series,
                    window=window,
                    step=step,
                    initial_capital=initial_capital,
                    regime=regime,
                    meta_uncertainty=meta_uncertainty,
                )
                results.append(
                    {
                        "window": window,
                        "step": step,
                        "meta_uncertainty": meta_uncertainty,
                        "result": strategy_output,
                    }
                )

    return results
