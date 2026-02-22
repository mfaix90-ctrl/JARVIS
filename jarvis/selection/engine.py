# =============================================================================
# JARVIS v6.1.0 -- SELECTION ENGINE (SCAFFOLD)
# File:   jarvis/selection/engine.py
# Version: 1.1.0
# =============================================================================
#
# SCOPE
# -----
# Minimal deterministic instrument selection scaffold. Ranks candidates by
# a caller-supplied score function and applies configurable filters.
# No external data calls. Pure functions only.
#
# PUBLIC FUNCTIONS
# ----------------
#   rank_candidates(candidates, score_fn, descending) -> List[str]
#   filter_by_threshold(candidates, scores, threshold) -> List[str]
#   select_top_n(candidates, scores, n) -> List[str]
#
# DETERMINISM CONSTRAINTS
# -----------------------
# All functions are pure. Tie-breaking uses lexicographic sort on symbol.
# score_fn must be deterministic (caller's responsibility).
# =============================================================================

from __future__ import annotations

from typing import Callable, Dict, List, Sequence


def rank_candidates(
    candidates:  Sequence[str],
    score_fn:    Callable[[str], float],
    descending:  bool = True,
) -> List[str]:
    """
    Rank instrument symbols by a deterministic score function.

    Ties are broken lexicographically (ascending symbol name) to ensure
    deterministic output regardless of input order.

    Args:
        candidates:  Sequence of instrument symbol strings.
        score_fn:    Pure function mapping symbol -> float score.
        descending:  If True, highest score ranked first (default: True).

    Returns:
        List of symbols sorted by score, then by symbol for tie-breaking.
    """
    scored = [(score_fn(sym), sym) for sym in candidates]
    scored.sort(key=lambda x: (-x[0] if descending else x[0], x[1]))
    return [sym for _, sym in scored]


def filter_by_threshold(
    candidates:  Sequence[str],
    scores:      Dict[str, float],
    threshold:   float,
) -> List[str]:
    """
    Filter candidates to those whose score >= threshold.

    Returns list sorted by descending score, then ascending symbol (deterministic).

    Args:
        candidates:  Sequence of candidate symbols.
        scores:      Mapping of symbol -> score. Symbols not in scores receive 0.0.
        threshold:   Minimum score (inclusive).

    Returns:
        Filtered and sorted list of symbols.
    """
    qualified = [
        (scores.get(sym, 0.0), sym)
        for sym in candidates
        if scores.get(sym, 0.0) >= threshold
    ]
    qualified.sort(key=lambda x: (-x[0], x[1]))
    return [sym for _, sym in qualified]


def select_top_n(
    candidates:  Sequence[str],
    scores:      Dict[str, float],
    n:           int,
) -> List[str]:
    """
    Select top-n candidates by score.

    Args:
        candidates:  Sequence of candidate symbols.
        scores:      Mapping of symbol -> score. Missing symbols scored 0.0.
        n:           Number of candidates to select. If n >= len(candidates),
                     all candidates are returned (sorted).

    Returns:
        List of up to n symbols, sorted by descending score then ascending symbol.

    Raises:
        ValueError if n < 0.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")
    ranked = filter_by_threshold(candidates, scores, threshold=float('-inf'))
    return ranked[:n]


__all__ = [
    "rank_candidates",
    "filter_by_threshold",
    "select_top_n",
]
