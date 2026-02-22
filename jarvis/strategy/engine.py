# =============================================================================
# JARVIS v6.1.0 -- STRATEGY ENGINE (SCAFFOLD)
# File:   jarvis/strategy/engine.py
# Version: 1.1.0
# =============================================================================
#
# SCOPE
# -----
# Minimal deterministic strategy scaffold. Generates directional signals
# from a returns series based on configurable momentum and mean-reversion
# lookbacks. No alpha optimisation. No model fitting. Pure functions only.
#
# PUBLIC FUNCTIONS
# ----------------
#   momentum_signal(returns, lookback) -> float  in [-1.0, 1.0]
#   mean_reversion_signal(returns, lookback) -> float  in [-1.0, 1.0]
#   combine_signals(signals, weights) -> float  in [-1.0, 1.0]
#
# DETERMINISM CONSTRAINTS
# -----------------------
# All functions are pure. No randomness, I/O, state, or side effects.
# =============================================================================

from __future__ import annotations

import math
from typing import List, Sequence


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def momentum_signal(returns: List[float], lookback: int) -> float:
    """
    Momentum signal: sign of average return over lookback window, scaled by
    magnitude relative to realised std.

    Returns value in [-1.0, 1.0].
    Returns 0.0 when len(returns) < lookback or lookback < 1.
    """
    if lookback < 1 or len(returns) < lookback:
        return 0.0
    window = returns[-lookback:]
    n = len(window)
    mu = sum(window) / n
    variance = sum((r - mu) ** 2 for r in window) / max(n - 1, 1)
    sd = math.sqrt(max(variance, 1e-15))
    signal = mu / sd
    return _clip(signal, -1.0, 1.0)


def mean_reversion_signal(returns: List[float], lookback: int) -> float:
    """
    Mean-reversion signal: negative of momentum signal.

    Returns value in [-1.0, 1.0].
    Returns 0.0 when len(returns) < lookback or lookback < 1.
    """
    return -momentum_signal(returns, lookback)


def combine_signals(signals: Sequence[float], weights: Sequence[float]) -> float:
    """
    Weighted combination of signals, normalised to [-1.0, 1.0].

    Args:
        signals: Sequence of floats, each in [-1.0, 1.0].
        weights: Corresponding non-negative weights.

    Returns:
        Weighted sum / sum(weights), clipped to [-1.0, 1.0].
        Returns 0.0 if weights sum to zero or sequences are empty.

    Raises:
        ValueError if len(signals) != len(weights).
    """
    if len(signals) != len(weights):
        raise ValueError(
            f"signals and weights must have equal length; "
            f"got {len(signals)} vs {len(weights)}"
        )
    if not signals:
        return 0.0
    total_weight = sum(weights)
    if total_weight < 1e-15:
        return 0.0
    combined = sum(s * w for s, w in zip(signals, weights)) / total_weight
    return _clip(combined, -1.0, 1.0)


__all__ = [
    "momentum_signal",
    "mean_reversion_signal",
    "combine_signals",
]
