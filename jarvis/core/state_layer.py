# =============================================================================
# JARVIS v6.0.1 — SESSION 05, PHASE 5.1: STATE LAYER — LatentState
# File:   jarvis/core/state_layer.py
# Authority: JARVIS FAS v6.0.1 — 02-05_CORE.md, S05 section
# Phase:  5.1 — LatentState + invariant enforcement only
# =============================================================================
#
# SCOPE (Phase 5.1)
# -----------------
# This file implements exactly one public type: LatentState.
# No Kalman filter. No HMM. No controller integration. No logging.
# No state writes. No external I/O.
#
# DEPENDENCIES
# ------------
# stdlib only: dataclasses, math, typing.
# No numpy. No third-party imports.
#
# DETERMINISM CONSTRAINTS
# -----------------------
# DET-01  No stochastic operations. No random, no uuid.
# DET-02  All inputs passed explicitly. No module-level mutable reads.
# DET-03  No side effects. LatentState is frozen; no mutation path exists.
# DET-04  All numeric operations are deterministic (math.isfinite, min, max).
# DET-05  All conditional branches are pure functions of explicit inputs.
# DET-06  No datetime.now().
#
# INVARIANTS ENFORCED (FAS INV-S05-01 through INV-S05-08)
# --------------------------------------------------------
# INV-S05-01  Exactly 12 fields. Hard assert on field count in __post_init__.
# INV-S05-02  All float fields must be finite (no NaN, no Inf).
# INV-S05-03  regime_confidence is clipped to [0.0, 1.0].
# INV-S05-04  stability is clipped to [0.0, 1.0].
# INV-S05-05  prediction_uncertainty is floored at 0.0.
# INV-S05-06  regime is an int in [0, 4]. Clamped silently.
# INV-S05-07  Frozen: no field may be mutated after construction.
# INV-S05-08  stability maps to S in D(t).
#             prediction_uncertainty contributes to sigma^2 in D(t).
#             regime_confidence maps to R in D(t).
#
# PROHIBITED ACTIONS CONFIRMED ABSENT
# ------------------------------------
#   No numpy / scipy
#   No logging module
#   No datetime.now() / time.time()
#   No random / secrets
#   No file I/O
#   No controller imports
#   No regime imports (regime field is a raw int at this phase)
#   No S06+ imports
#   No Kalman / HMM code
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List


# ---------------------------------------------------------------------------
# Constants (fixed literals — not parameterised)
# ---------------------------------------------------------------------------

#: Number of LatentState dimensions. Must never change without a version bump.
LATENT_STATE_DIMS: int = 12

#: Valid range for the HMM regime index.
REGIME_INT_MIN: int = 0
REGIME_INT_MAX: int = 4


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateError(Exception):
    """
    Raised when a LatentState field contains an invalid value.
    Specifically: NaN or Inf in any float field.
    This error must never be caught silently; it must propagate to the caller.
    """


# ---------------------------------------------------------------------------
# LatentState (FAS S05 — 12-dimensional latent state vector)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentState:
    """
    Latent state vector — exactly 12 dimensions. Immutable after construction.

    Correspondence to system contract D(t):
      regime_confidence  ->  R  (Regime confidence, always in [0, 1])
      stability          ->  S  (System stability,  always in [0, 1])
      prediction_uncertainty -> contribution to sigma^2 (always >= 0)

    INV-S05-01: Exactly 12 fields. Enforced by __post_init__ hard assert.
    INV-S05-07: frozen=True -- no mutation path exists after construction.

    Construction rules:
      - regime is clamped to [0, 4].
      - regime_confidence is clipped to [0.0, 1.0].
      - stability is clipped to [0.0, 1.0].
      - prediction_uncertainty is floored at 0.0.
      - All float fields must be finite. StateError is raised otherwise.

    Note on frozen + __post_init__:
      Because the dataclass is frozen, __post_init__ cannot assign to fields
      directly. It uses object.__setattr__ exclusively for the clamped/clipped
      fields, which is the stdlib-approved pattern for frozen dataclass
      post-construction adjustment.
    """

    # ------------------------------------------------------------------
    # Fields (order is fixed -- INV-S05-01)
    # ------------------------------------------------------------------

    regime: int
    """HMM regime index, 0-4. Clamped to [REGIME_INT_MIN, REGIME_INT_MAX]."""

    volatility: float
    """Aleatoric volatility contribution (sigma_aleatoric). Must be finite."""

    trend_strength: float
    """Trend coherence signal. Must be finite."""

    mean_reversion: float
    """Mean-reversion signal strength. Must be finite."""

    liquidity: float
    """Liquidity state estimate. Must be finite."""

    stress: float
    """Market stress indicator. Must be finite."""

    momentum: float
    """Momentum signal. Must be finite."""

    drift: float
    """Drift estimate. Must be finite."""

    noise: float
    """Noise level estimate. Must be finite."""

    regime_confidence: float
    """R in D(t). Clipped to [0.0, 1.0] on construction. Must be finite."""

    stability: float
    """S in D(t). Clipped to [0.0, 1.0] on construction. Must be finite."""

    prediction_uncertainty: float
    """Contribution to sigma^2 in D(t). Floored at 0.0 on construction. Must be finite."""

    # ------------------------------------------------------------------
    # Post-construction invariant enforcement
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # --- INV-S05-01: dimension guard ---
        # Count the annotated fields at runtime to protect against accidental
        # future additions that bypass this file's version bump requirement.
        actual_dims = len(fields(self))
        assert actual_dims == LATENT_STATE_DIMS, (
            f"LatentState dimension invariant violated: "
            f"expected {LATENT_STATE_DIMS} fields, found {actual_dims}. "
            f"A version bump and FAS update are required before adding fields."
        )

        # --- INV-S05-06: regime clamping (int) ---
        clamped_regime = max(REGIME_INT_MIN, min(REGIME_INT_MAX, int(self.regime)))
        object.__setattr__(self, "regime", clamped_regime)

        # --- Collect float field names for finiteness check ---
        _FLOAT_FIELDS: List[str] = [
            "volatility",
            "trend_strength",
            "mean_reversion",
            "liquidity",
            "stress",
            "momentum",
            "drift",
            "noise",
            "regime_confidence",
            "stability",
            "prediction_uncertainty",
        ]

        # --- INV-S05-02: finiteness check -- must precede clipping ---
        # NaN and Inf are rejected unconditionally; clipping cannot rescue them.
        for fname in _FLOAT_FIELDS:
            value = getattr(self, fname)
            if not math.isfinite(value):
                raise StateError(
                    f"LatentState.{fname} is non-finite: {value!r}. "
                    f"NaN and Inf are not permitted in any float field."
                )

        # --- INV-S05-03: regime_confidence clipped to [0.0, 1.0] ---
        clipped_rc = max(0.0, min(1.0, self.regime_confidence))
        object.__setattr__(self, "regime_confidence", clipped_rc)

        # --- INV-S05-04: stability clipped to [0.0, 1.0] ---
        clipped_stab = max(0.0, min(1.0, self.stability))
        object.__setattr__(self, "stability", clipped_stab)

        # --- INV-S05-05: prediction_uncertainty floored at 0.0 ---
        floored_pu = max(0.0, self.prediction_uncertainty)
        object.__setattr__(self, "prediction_uncertainty", floored_pu)

    # ------------------------------------------------------------------
    # Utility methods (pure; no side effects)
    # ------------------------------------------------------------------

    def as_tuple(self) -> tuple:
        """
        Return all 12 fields as an ordered tuple.
        Order matches field declaration order (INV-S05-01).
        Useful for deterministic comparison and serialisation.
        """
        return (
            self.regime,
            self.volatility,
            self.trend_strength,
            self.mean_reversion,
            self.liquidity,
            self.stress,
            self.momentum,
            self.drift,
            self.noise,
            self.regime_confidence,
            self.stability,
            self.prediction_uncertainty,
        )

    @staticmethod
    def default() -> "LatentState":
        """
        Return a conservative default LatentState for initialisation.

        regime=4 (SHOCK / UNKNOWN index) and regime_confidence=0.0 ensure
        that any downstream consumer treats the first cycle as uncertain
        and triggers FM-01 (UNDEFINED_REGIME) until real estimates arrive.
        """
        return LatentState(
            regime=4,
            volatility=0.01,
            trend_strength=0.0,
            mean_reversion=0.0,
            liquidity=0.5,
            stress=0.5,
            momentum=0.0,
            drift=0.0,
            noise=0.01,
            regime_confidence=0.0,
            stability=0.5,
            prediction_uncertainty=1.0,
        )
