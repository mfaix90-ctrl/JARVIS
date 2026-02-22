# jarvis/robustness/engine.py
# Version: 1.0.0
# External robustness evaluation layer.
# External to jarvis/core/, jarvis/risk/, jarvis/utils/, jarvis/portfolio/,
# jarvis/execution/, jarvis/orchestrator/, jarvis/backtest/,
# jarvis/walkforward/, jarvis/metrics/, jarvis/report/, jarvis/strategy/,
# jarvis/optimization/, jarvis/selection/ per FAS v6.1.0 architecture rules.
#
# DETERMINISM GUARANTEE:
#   No stochastic operations. No random number generation. No sampling.
#   No external state reads. No side effects. No file I/O. No logging.
#   No environment variable access. No global mutable state.
#   Output is a pure function of inputs.
#
# FAS ARCHITECTURE POSITION:
#   External to all other layers.
#   Consumes output of jarvis.selection.run_selection() only.
#   No calls to any other layer are made.
#
# INPUT CONTRACT NOTE:
#   run_selection() produces ranking entries with flat keys:
#     "window", "step", "meta_uncertainty", "aggregate", "score_tuple"
#   and top-level keys: "best_params", "best_score", "ranking".
#   This module reads those keys directly. "best_params" is the
#   canonical params dict. ranking[0] is the best entry.
#
# Standard import pattern:
#   from jarvis.robustness.engine import evaluate_robustness

from typing import Any


def evaluate_robustness(
    selection_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluate parameter robustness from run_selection() output.

    FAS COMPLIANCE:
      This function is an external layer under FAS v6.1.0. It satisfies
      all determinism guarantees (DET-01 through DET-07) and all
      prohibitions defined in ARCHITECTURE.md Section 4. Specifically:

      PROHIBITED-06 (no reimplementation):
        No metric formula from any upstream layer is reimplemented here.
        All aggregate values (mean_cagr, mean_sharpe, worst_max_drawdown)
        are consumed directly from selection_output as produced by
        run_selection(). The only computations performed here are:
        population variance (a simple aggregation), top_gap (a single
        subtraction between two already-computed values), and
        robustness_score (a deterministic formula over those aggregates).
        None of these duplicate any formula owned by a canonical layer.

      PROHIBITED-02 (no file I/O): No file is read or written.
      PROHIBITED-03 (no logging): No logging calls are made.
      PROHIBITED-05 (no global mutable state): No module-level variable
        is written to during execution.
      DET-02 (no external state reads): All inputs are passed explicitly.
      DET-03 (no side effects): selection_output is not mutated.

    INPUT CONTRACT:
      selection_output must be the direct output of run_selection().
      Expected top-level keys:
        "best_params" -- dict with "window", "step", "meta_uncertainty"
        "best_score"  -- aggregate dict (not used directly; ranking[0]
                         is the authoritative best entry)
        "ranking"     -- list of dicts, each containing:
                           "window", "step", "meta_uncertainty",
                           "aggregate" (with "mean_cagr", "mean_sharpe",
                           "worst_max_drawdown"), "score_tuple"
      ranking must be non-empty. ranking[0] is the best entry.

    COMPUTED VALUES:
      top_gap:
        mean_cagr of ranking[0] minus mean_cagr of ranking[1].
        0.0 if ranking has only one entry.

      cagr_variance:
        Population variance of mean_cagr across all ranking entries.

      sharpe_variance:
        Population variance of mean_sharpe across all ranking entries.

      drawdown_variance:
        Population variance of worst_max_drawdown across all ranking
        entries.

      robustness_score:
        top_gap * (1 / (1 + cagr_variance)) * (1 / (1 + sharpe_variance))

    Parameters
    ----------
    selection_output : dict[str, Any]
        Direct output of run_selection(). Not mutated by this function.

    Returns
    -------
    dict[str, Any]
        {
            "best_params": dict -- "window", "step", "meta_uncertainty"
                                   of the top-ranked configuration.
            "robustness_score": float -- deterministic robustness score.
            "stability_metrics": {
                "top_gap":           float,
                "cagr_variance":     float,
                "sharpe_variance":   float,
                "drawdown_variance": float,
            }
        }

    Raises
    ------
    ValueError
        If ranking is empty.
    KeyError
        If required keys are absent from selection_output or its
        nested dicts. Propagated without wrapping.

    Notes
    -----
    Deterministic. No side effects. No global state. Pure function.
    selection_output is not mutated. All intermediate containers are
    constructed fresh per call.
    """
    ranking: list[dict] = selection_output["ranking"]

    if len(ranking) == 0:
        raise ValueError("ranking must not be empty.")

    best_entry: dict = ranking[0]
    best_params: dict = {
        "window":           best_entry["window"],
        "step":             best_entry["step"],
        "meta_uncertainty": best_entry["meta_uncertainty"],
    }

    # ------------------------------------------------------------------
    # Top gap: difference in mean_cagr between best and second best.
    # ------------------------------------------------------------------
    if len(ranking) > 1:
        top_gap: float = (
            best_entry["aggregate"]["mean_cagr"]
            - ranking[1]["aggregate"]["mean_cagr"]
        )
    else:
        top_gap = 0.0

    # ------------------------------------------------------------------
    # Collect aggregate values across all ranking entries.
    # ------------------------------------------------------------------
    cagr_values: list[float] = [
        entry["aggregate"]["mean_cagr"] for entry in ranking
    ]
    sharpe_values: list[float] = [
        entry["aggregate"]["mean_sharpe"] for entry in ranking
    ]
    drawdown_values: list[float] = [
        entry["aggregate"]["worst_max_drawdown"] for entry in ranking
    ]

    # ------------------------------------------------------------------
    # Population variance helper. Defined locally; not a reimplementation
    # of any canonical layer formula.
    # ------------------------------------------------------------------
    def _variance(values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        mean: float = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    cagr_variance: float     = _variance(cagr_values)
    sharpe_variance: float   = _variance(sharpe_values)
    drawdown_variance: float = _variance(drawdown_values)

    # ------------------------------------------------------------------
    # Robustness score.
    # ------------------------------------------------------------------
    robustness_score: float = (
        top_gap
        * (1.0 / (1.0 + cagr_variance))
        * (1.0 / (1.0 + sharpe_variance))
    )

    return {
        "best_params": best_params,
        "robustness_score": robustness_score,
        "stability_metrics": {
            "top_gap":           top_gap,
            "cagr_variance":     cagr_variance,
            "sharpe_variance":   sharpe_variance,
            "drawdown_variance": drawdown_variance,
        },
    }
