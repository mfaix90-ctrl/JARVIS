# jarvis/report/engine.py
# Version: 1.0.0
# External report layer.
# External to jarvis/core/, jarvis/risk/, jarvis/utils/, jarvis/portfolio/,
# jarvis/execution/, jarvis/orchestrator/, jarvis/backtest/,
# jarvis/walkforward/, jarvis/metrics/ per FAS v6.1.0 architecture rules.
#
# DETERMINISM GUARANTEE:
#   No stochastic operations. No random number generation. No sampling.
#   No external state reads. No side effects. No file I/O. No logging.
#   No environment variable access. No global mutable state.
#   Output is a pure function of inputs.
#
# PURPOSE:
#   Assembles a structured report dict by delegating metric computation
#   entirely to compute_metrics(). No metric logic is reimplemented here.
#   The equity_curve is included in the output unchanged and uncopied --
#   callers must treat it as read-only per the no-mutation contract.
#
# Standard import pattern:
#   from jarvis.report.engine import generate_report

from jarvis.metrics import compute_metrics


def generate_report(
    equity_curve: list[float],
    periods_per_year: int = 252,
) -> dict[str, object]:
    """
    Assemble a structured performance report from an equity curve.

    FAS COMPLIANCE:
      This function is an external layer under FAS v6.1.0. It satisfies
      all determinism guarantees (DET-01 through DET-07) and all
      prohibitions defined in ARCHITECTURE.md Section 4. Specifically:

      PROHIBITED-06 (no reimplementation): All metric computation is
        delegated exclusively to jarvis.metrics.compute_metrics().
        No formula from that module is reproduced, inlined, or
        approximated here.

      PROHIBITED-02 (no file I/O): No file is read or written.

      PROHIBITED-03 (no logging): No logging calls are made.

      PROHIBITED-05 (no global mutable state): No module-level variable
        is written to during execution.

      DET-02 (no external state reads): All inputs are passed explicitly.
        No module-level or class-level state is read during execution.

      DET-03 (no side effects): The equity_curve argument is not mutated.
        The returned dict contains a reference to the original list, not
        a copy. Callers must treat the contained equity_curve as read-only.

    Parameters
    ----------
    equity_curve : list[float]
        Sequence of portfolio equity values.
        Must contain at least 2 values.
        All values must be strictly positive.
        Validated here (length >= 2) and downstream by compute_metrics()
        (positivity and length).
        Not mutated by this function.
    periods_per_year : int
        Annualization factor passed unchanged to compute_metrics().
        Default = 252.

    Returns
    -------
    dict[str, object]
        Structured report containing:
          "equity_curve"    -- the original equity_curve list (unmodified).
          "metrics"         -- dict[str, float] returned by compute_metrics().
                               Keys: total_return, cagr, volatility,
                                     sharpe, max_drawdown.
          "periods_per_year" -- the periods_per_year value used.

    Raises
    ------
    ValueError
        If len(equity_curve) < 2 (validated here before delegation).
        Additional ValueError may propagate from compute_metrics() if
        any equity value is <= 0 or the annualization period is invalid.

    Notes
    -----
    Deterministic. No side effects. No global state. Pure function.
    """
    if len(equity_curve) < 2:
        raise ValueError(
            f"equity_curve must contain at least 2 values. "
            f"Received: {len(equity_curve)}"
        )

    metrics: dict[str, float] = compute_metrics(
        equity_curve=equity_curve,
        periods_per_year=periods_per_year,
    )

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
        "periods_per_year": periods_per_year,
    }
