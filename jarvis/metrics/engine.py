# =============================================================================
# JARVIS v6.1.0 -- METRICS ENGINE
# File:   jarvis/metrics/engine.py
# Version: 1.1.0
# =============================================================================
#
# SCOPE
# -----
# Pure deterministic performance metrics. All functions are stateless,
# side-effect-free, and take no external dependencies beyond stdlib math.
#
# PUBLIC FUNCTIONS
# ----------------
#   sharpe_ratio(returns, periods_per_year, risk_free_rate) -> float
#   max_drawdown(returns) -> float
#   calmar_ratio(returns, periods_per_year) -> float
#   regime_conditional_returns(returns, regime_labels) -> dict
#
# DETERMINISM CONSTRAINTS
# -----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly.
# DET-03  No side effects.
# DET-04  No I/O, no logging, no datetime.now().
# =============================================================================

from __future__ import annotations

import math
from typing import List, Sequence, Dict


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    return sum(values) / n


def _std(values: Sequence[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    mu = _mean(values)
    variance = sum((x - mu) ** 2 for x in values) / (n - ddof)
    return math.sqrt(max(variance, 0.0))


# ---------------------------------------------------------------------------
# SHARPE RATIO
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns:          List[float],
    periods_per_year: int   = 252,
    risk_free_rate:   float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Formula:
        excess = returns - risk_free_rate / periods_per_year
        sharpe = mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)

    Returns 0.0 when std == 0 or returns is empty.
    Raises ValueError if periods_per_year < 1.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1; got {periods_per_year}")
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / periods_per_year
    excess = [r - daily_rf for r in returns]
    mu = _mean(excess)
    sd = _std(excess, ddof=1)
    if sd < 1e-15:
        return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# MAX DRAWDOWN
# ---------------------------------------------------------------------------

def max_drawdown(returns: List[float]) -> float:
    """
    Maximum drawdown from peak.

    Formula:
        cumulative = cumprod(1 + r)
        running_peak = running maximum of cumulative
        drawdown = (running_peak - cumulative) / running_peak
        max_drawdown = max(drawdown)

    Returns 0.0 for empty or single-element returns.
    """
    if len(returns) < 2:
        return 0.0
    cum = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cum *= (1.0 + r)
        if cum > peak:
            peak = cum
        dd = (peak - cum) / max(peak, 1e-15)
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# CALMAR RATIO
# ---------------------------------------------------------------------------

def calmar_ratio(
    returns:          List[float],
    periods_per_year: int = 252,
) -> float:
    """
    Calmar ratio: annualised return / max drawdown.

    Returns 0.0 when max_drawdown == 0 or returns is empty.
    Raises ValueError if periods_per_year < 1.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1; got {periods_per_year}")
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    total = 1.0
    for r in returns:
        total *= (1.0 + r)
    ann_return = total ** (periods_per_year / n) - 1.0
    mdd = max_drawdown(returns)
    if mdd < 1e-15:
        return 0.0
    return ann_return / mdd


# ---------------------------------------------------------------------------
# REGIME-CONDITIONAL RETURNS
# ---------------------------------------------------------------------------

def regime_conditional_returns(
    returns:        List[float],
    regime_labels:  List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean return and count per regime label.

    Args:
        returns:       Per-period returns. Length must equal len(regime_labels).
        regime_labels: String label per period (e.g. 'RISK_ON', 'CRISIS').

    Returns:
        dict mapping regime label -> {'mean': float, 'count': int, 'total': float}

    Raises:
        ValueError if lengths differ or either is empty.
    """
    if len(returns) != len(regime_labels):
        raise ValueError(
            f"returns and regime_labels must have equal length; "
            f"got {len(returns)} vs {len(regime_labels)}"
        )
    if len(returns) == 0:
        return {}

    buckets: Dict[str, List[float]] = {}
    for r, label in zip(returns, regime_labels):
        if label not in buckets:
            buckets[label] = []
        buckets[label].append(r)

    result: Dict[str, Dict[str, float]] = {}
    for label, vals in sorted(buckets.items()):
        result[label] = {
            "mean":  _mean(vals),
            "count": float(len(vals)),
            "total": sum(vals),
        }
    return result


__all__ = [
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "regime_conditional_returns",
]
