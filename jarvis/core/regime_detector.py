# =============================================================================
# JARVIS v6.0.1 — SESSION 05, PHASE 5.3: REGIME DETECTOR
# File:   jarvis/core/regime_detector.py
# Authority: JARVIS FAS v6.0.1 — 02-05_CORE.md, S05 section
# Phase:  5.3 — RegimeDetector (HMM-based regime detection)
# =============================================================================
#
# SCOPE (Phase 5.3)
# -----------------
# Implements:
#   - RegimeResult  dataclass (frozen=True)
#   - RegimeDetector class
#       * detect_regime(features) -> RegimeResult
#       * transition_probability() -> List[List[float]]  (5x5 stochastic matrix)
#       * regime_confidence() -> float  in [0.0, 1.0]
#
# CONSTRAINTS
# -----------
# stdlib only: dataclasses, math, typing.
# No numpy. No scipy. No pandas. No random. No datetime.now().
# No file I/O. No logging. No global mutable state.
# Regime enums sourced exclusively from jarvis.core.regime.
# LatentState sourced from jarvis.core.state_layer.
#
# DETERMINISM GUARANTEES
# ----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly.
# DET-03  No side effects.
# DET-04  All arithmetic deterministic.
# DET-05  All branches pure functions of explicit inputs.
# DET-06  No datetime.now().
#
# HMM MODEL SPECIFICATION
# -----------------------
# 5 hidden states (indices 0-4) mapping to GlobalRegimeState:
#   0 -> RISK_ON      (Bull / trending up / low vol)
#   1 -> RISK_OFF     (Bear / trending down / elevated stress)
#   2 -> TRANSITION   (Sideways / mean-reverting)
#   3 -> RISK_OFF     (High volatility / directionally ambiguous)
#   4 -> CRISIS       (Shock / structural break)
#
# Forward algorithm: pure Python, O(T * N^2) where N=5.
# Emission probabilities: Gaussian approximation via math.exp.
# Transition matrix: ergodic (all states reachable), row-stochastic.
# Confidence: max posterior probability over current state distribution.
#
# INVARIANTS
# ----------
# INV-P53-01  transition_probability() returns a 5x5 list-of-lists.
# INV-P53-02  Each row of the transition matrix sums to 1.0.
# INV-P53-03  All entries clipped to [1e-6, 1 - 1e-6] before normalisation.
# INV-P53-04  regime_confidence() always returns a value in [0.0, 1.0].
# INV-P53-05  detect_regime() never raises on finite feature inputs.
# INV-P53-06  RegimeResult.hmm_index is always in {0, 1, 2, 3, 4}.
# INV-P53-07  RegimeResult.confidence is in [0.0, 1.0].
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from jarvis.core.regime import GlobalRegimeState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of HMM hidden states.
_N_STATES: int = 5

#: Epsilon floor for probabilities to avoid log(0).
_PROB_EPS: float = 1e-10

#: Clip bounds for transition matrix entries (before row normalisation).
_T_MIN: float = 1e-6
_T_MAX: float = 1.0 - 1e-6

#: Mapping from HMM state index -> GlobalRegimeState.
_STATE_TO_REGIME: Dict[int, GlobalRegimeState] = {
    0: GlobalRegimeState.RISK_ON,
    1: GlobalRegimeState.RISK_OFF,
    2: GlobalRegimeState.TRANSITION,
    3: GlobalRegimeState.RISK_OFF,
    4: GlobalRegimeState.CRISIS,
}

#: Default prior (uniform over 5 states).
_UNIFORM_PRIOR: List[float] = [1.0 / _N_STATES] * _N_STATES

# ---------------------------------------------------------------------------
# Default ergodic transition matrix (row-stochastic, all entries in [T_MIN, T_MAX]).
# Rows sum to 1.0. Designed to be sticky (high self-transition) while
# allowing inter-state transitions.
# State order: RISK_ON, RISK_OFF, TRANSITION, HIGH_VOL, CRISIS
# ---------------------------------------------------------------------------
_DEFAULT_TRANSITION: List[List[float]] = [
    # to:   0      1      2      3      4
    [0.70,  0.10,  0.10,  0.06,  0.04],  # from 0: RISK_ON
    [0.10,  0.65,  0.10,  0.10,  0.05],  # from 1: RISK_OFF
    [0.15,  0.15,  0.55,  0.10,  0.05],  # from 2: TRANSITION
    [0.08,  0.15,  0.12,  0.55,  0.10],  # from 3: HIGH_VOL
    [0.05,  0.10,  0.10,  0.15,  0.60],  # from 4: CRISIS
]

# ---------------------------------------------------------------------------
# Default emission means (per feature key, per state).
# Features are assumed z-scored. Emission probability modelled as
# isotropic Gaussian with unit variance (simplified for stdlib-only).
# ---------------------------------------------------------------------------
_EMISSION_MEANS: Dict[str, List[float]] = {
    # State order: RISK_ON, RISK_OFF, TRANSITION, HIGH_VOL, CRISIS
    "volatility":          [0.3,   0.8,   0.5,  1.2,  2.0],
    "trend_strength":      [0.7,  -0.5,   0.0,  0.1, -0.3],
    "mean_reversion":      [0.1,   0.2,   0.8,  0.3,  0.1],
    "stress":              [0.1,   0.6,   0.3,  0.8,  1.5],
    "momentum":            [0.6,  -0.5,   0.0,  0.1, -0.2],
    "liquidity":           [0.8,   0.3,   0.5,  0.2,  0.1],
}

# Emission sigma (shared across all features and states — simplified HMM).
_EMISSION_SIGMA: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_log_prob(x: float, mu: float, sigma: float) -> float:
    """Log probability of x under N(mu, sigma^2). sigma must be > 0."""
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def _emission_log_prob(features: Dict[str, float], state_idx: int) -> float:
    """
    Log emission probability P(features | state_idx).
    Only features present in _EMISSION_MEANS are used.
    Non-finite feature values are silently replaced with 0.0.
    """
    log_p: float = 0.0
    for feat_name, state_means in _EMISSION_MEANS.items():
        value: float = features.get(feat_name, 0.0)
        if not math.isfinite(value):
            value = 0.0
        mu: float = state_means[state_idx]
        log_p += _gaussian_log_prob(value, mu, _EMISSION_SIGMA)
    return log_p


def _log_sum_exp(log_probs: List[float]) -> float:
    """Numerically stable log-sum-exp over a list of log probabilities."""
    if not log_probs:
        return -math.inf
    max_lp: float = max(log_probs)
    if not math.isfinite(max_lp):
        return -math.inf
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))


def _normalise(probs: List[float]) -> List[float]:
    """
    Normalise a list of non-negative values to sum to 1.0.
    Any non-finite value is replaced with _PROB_EPS before normalisation.
    If all values are zero (or sum is zero), returns uniform distribution.
    """
    safe: List[float] = [
        max(_PROB_EPS, v) if math.isfinite(v) else _PROB_EPS
        for v in probs
    ]
    total: float = sum(safe)
    if total <= 0.0:
        return [1.0 / _N_STATES] * _N_STATES
    return [v / total for v in safe]


# ---------------------------------------------------------------------------
# RegimeResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeResult:
    """
    Immutable result of a single regime detection step.

    Fields
    ------
    hmm_index : int
        Index of the most probable HMM hidden state (0-4).
    regime : GlobalRegimeState
        Canonical macro regime corresponding to hmm_index.
    confidence : float
        Posterior probability of the most probable state. In [0.0, 1.0].
    posterior : tuple[float, ...]
        Full posterior distribution over all 5 states.
        Sums to 1.0 (within floating-point tolerance).
    """
    hmm_index: int
    regime: GlobalRegimeState
    confidence: float
    posterior: tuple  # tuple[float, ...] — length 5

    def __post_init__(self) -> None:
        if self.hmm_index not in _STATE_TO_REGIME:
            raise ValueError(
                f"RegimeResult.hmm_index must be in {{0,1,2,3,4}}, "
                f"got {self.hmm_index!r}."
            )
        if not isinstance(self.regime, GlobalRegimeState):
            raise TypeError(
                f"RegimeResult.regime must be a GlobalRegimeState member, "
                f"got {type(self.regime).__name__!r}."
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"RegimeResult.confidence must be in [0.0, 1.0], "
                f"got {self.confidence!r}."
            )
        if len(self.posterior) != _N_STATES:
            raise ValueError(
                f"RegimeResult.posterior must have length {_N_STATES}, "
                f"got {len(self.posterior)}."
            )


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    HMM-based regime detector.

    Maintains a running posterior distribution over 5 hidden states and
    updates it at each call to detect_regime() using the forward algorithm.

    IMMUTABILITY NOTE:
      The transition matrix and emission parameters are fixed at construction.
      The only mutable state is the current posterior (self._posterior),
      which is updated deterministically at each detect_regime() call.

    DETERMINISM:
      Given identical feature sequences, successive calls to detect_regime()
      will produce identical outputs. The posterior is reset to the uniform
      prior only by calling reset().

    STDLIB ONLY: no numpy, no scipy, no random.
    """

    def __init__(
        self,
        transition_matrix: List[List[float]] | None = None,
        initial_prior: List[float] | None = None,
    ) -> None:
        """
        Initialise the regime detector.

        Parameters
        ----------
        transition_matrix : List[List[float]] or None
            5x5 row-stochastic transition matrix. If None, the default
            ergodic matrix is used. Rows must sum to 1.0; entries are
            clipped to [T_MIN, T_MAX] and renormalised on construction.
        initial_prior : List[float] or None
            Initial state distribution over 5 states. If None, uniform
            prior [0.2, 0.2, 0.2, 0.2, 0.2] is used.
        """
        # Validate and store transition matrix.
        if transition_matrix is None:
            self._transition: List[List[float]] = [
                list(row) for row in _DEFAULT_TRANSITION
            ]
        else:
            self._transition = self._validate_transition(transition_matrix)

        # Normalise rows of the stored transition matrix.
        for i in range(_N_STATES):
            row = [max(_T_MIN, min(_T_MAX, v)) for v in self._transition[i]]
            self._transition[i] = _normalise(row)

        # Initialise posterior.
        if initial_prior is None:
            self._posterior: List[float] = list(_UNIFORM_PRIOR)
        else:
            if len(initial_prior) != _N_STATES:
                raise ValueError(
                    f"initial_prior must have length {_N_STATES}, "
                    f"got {len(initial_prior)}."
                )
            self._posterior = _normalise(list(initial_prior))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_regime(self, features: Dict[str, float]) -> RegimeResult:
        """
        Update the posterior state distribution given new feature observations
        and return the most probable regime.

        Uses the HMM forward step: prediction (transition) followed by
        update (emission likelihood weighting), then normalisation.

        Parameters
        ----------
        features : Dict[str, float]
            Feature dict. Keys matching _EMISSION_MEANS are used.
            Unknown keys are ignored. Non-finite values are replaced with 0.0.
            Must not be None.

        Returns
        -------
        RegimeResult
            Immutable result containing the most probable state, its
            canonical GlobalRegimeState, confidence, and full posterior.

        Notes
        -----
        Deterministic. Updates self._posterior as the only side effect.
        """
        if features is None:
            raise TypeError("features must be a dict, got None.")

        # --- Prediction step: propagate prior through transition matrix ---
        predicted: List[float] = [0.0] * _N_STATES
        for j in range(_N_STATES):
            predicted[j] = sum(
                self._posterior[i] * self._transition[i][j]
                for i in range(_N_STATES)
            )

        # --- Update step: weight by emission probabilities ---
        log_emit: List[float] = [
            _emission_log_prob(features, j) for j in range(_N_STATES)
        ]
        # Convert log emission to linear scale, centred for numerical stability.
        max_log_emit: float = max(log_emit)
        emit_scale: List[float] = [
            math.exp(le - max_log_emit) for le in log_emit
        ]

        unnorm: List[float] = [
            predicted[j] * emit_scale[j] for j in range(_N_STATES)
        ]

        # --- Normalise posterior ---
        self._posterior = _normalise(unnorm)

        # --- Compute result ---
        best_idx: int = max(range(_N_STATES), key=lambda j: self._posterior[j])
        confidence: float = max(0.0, min(1.0, self._posterior[best_idx]))

        return RegimeResult(
            hmm_index=best_idx,
            regime=_STATE_TO_REGIME[best_idx],
            confidence=confidence,
            posterior=tuple(self._posterior),
        )

    def transition_probability(self) -> List[List[float]]:
        """
        Return the current 5x5 row-stochastic transition matrix.

        INV-P53-01: Returns a 5x5 list-of-lists.
        INV-P53-02: Each row sums to 1.0.
        INV-P53-03: All entries in [1e-6, 1 - 1e-6] (clipped on construction).

        Returns
        -------
        List[List[float]]
            A deep copy of the internal transition matrix.
            Modifying the returned list does not affect the detector state.
        """
        return [list(row) for row in self._transition]

    def regime_confidence(self) -> float:
        """
        Return the confidence in the current most probable regime.

        This is the maximum value in the current posterior distribution,
        representing R from D(t). Always in [0.0, 1.0].

        Returns
        -------
        float
            Maximum posterior probability. In [0.0, 1.0].
        """
        return max(0.0, min(1.0, max(self._posterior)))

    def current_posterior(self) -> List[float]:
        """
        Return the current state posterior distribution as a list of 5 floats.
        Sums to 1.0 (within floating-point tolerance).
        Returns a copy; modifying it does not affect detector state.
        """
        return list(self._posterior)

    def reset(self, prior: List[float] | None = None) -> None:
        """
        Reset the posterior to the uniform prior (or a supplied prior).

        Parameters
        ----------
        prior : List[float] or None
            If None, resets to the uniform distribution.
            If supplied, must have length 5 and contain non-negative values.
        """
        if prior is None:
            self._posterior = list(_UNIFORM_PRIOR)
        else:
            if len(prior) != _N_STATES:
                raise ValueError(
                    f"prior must have length {_N_STATES}, got {len(prior)}."
                )
            self._posterior = _normalise(list(prior))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_transition(matrix: List[List[float]]) -> List[List[float]]:
        """
        Validate the shape of a transition matrix. Raises ValueError on
        wrong shape. Does NOT renormalise — that is done in __init__.
        """
        if len(matrix) != _N_STATES:
            raise ValueError(
                f"transition_matrix must have {_N_STATES} rows, "
                f"got {len(matrix)}."
            )
        for i, row in enumerate(matrix):
            if len(row) != _N_STATES:
                raise ValueError(
                    f"transition_matrix row {i} must have {_N_STATES} "
                    f"columns, got {len(row)}."
                )
        return [list(row) for row in matrix]
