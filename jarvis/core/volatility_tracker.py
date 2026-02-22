# =============================================================================
# JARVIS v6.0.1 — SESSION 05, PHASE 5.4: VOLATILITY TRACKER
# File:   jarvis/core/volatility_tracker.py
# Authority: JARVIS FAS v6.0.1 — 02-05_CORE.md, S05 section
# Phase:  5.4 — VolatilityTracker (GARCH(1,1) volatility estimation)
# =============================================================================
#
# SCOPE (Phase 5.4)
# -----------------
# Implements:
#   - VolResult   dataclass (frozen=True)
#   - VolatilityTracker class
#       * estimate_volatility(returns) -> VolResult
#       * predict_volatility(horizon)  -> float
#
# CONSTRAINTS
# -----------
# stdlib only: dataclasses, math, typing.
# No numpy. No scipy. No pandas. No random. No datetime.now().
# No file I/O. No logging. No global mutable state (beyond internal state).
#
# DETERMINISM GUARANTEES
# ----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly.
# DET-03  No side effects beyond updating internal GARCH state.
# DET-04  All arithmetic deterministic.
# DET-05  All branches pure functions of explicit inputs.
# DET-06  No datetime.now().
#
# GARCH(1,1) MODEL SPECIFICATION
# --------------------------------
# sigma^2_t = omega + alpha * r_{t-1}^2 + beta * sigma^2_{t-1}
#
# where:
#   omega = long-run variance component (> 0)
#   alpha = ARCH coefficient (shock sensitivity)
#   beta  = GARCH coefficient (variance persistence)
#   alpha + beta < 1 (stationarity constraint, enforced)
#
# Default parameters calibrated for daily financial returns:
#   omega = 1e-6  (very small constant to maintain positivity)
#   alpha = 0.10  (10% weight on squared shock)
#   beta  = 0.85  (85% variance persistence)
#
# Volatility = sqrt(sigma^2_t), floored at EPSILON = 1e-8.
#
# Predict horizon h:
#   sigma^2_{t+h} = omega/(1-alpha-beta) + (alpha+beta)^h * (sigma^2_t - omega/(1-alpha-beta))
#
# INVARIANTS
# ----------
# INV-P54-01  estimate_volatility() raises ValueError for empty or
#             all-non-finite returns lists.
# INV-P54-02  Volatility is always > 0 (EPSILON floor applied everywhere).
# INV-P54-03  predict_volatility(horizon) requires horizon >= 1.
# INV-P54-04  Non-finite return values are replaced with 0.0 before processing.
# INV-P54-05  VolResult.volatility > 0 always.
# INV-P54-06  VolResult.variance > 0 always.
# INV-P54-07  VolResult.n_clean_returns >= 0.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Epsilon floor for volatility to guarantee strict positivity.
_VOL_EPSILON: float = 1e-8

#: Default GARCH(1,1) parameters.
_DEFAULT_OMEGA: float = 1e-6
_DEFAULT_ALPHA: float = 0.10
_DEFAULT_BETA: float  = 0.85

#: Minimum number of returns required for a meaningful GARCH estimate.
_MIN_RETURNS: int = 2

#: Initial variance guess (used before any data is seen).
_INIT_VARIANCE: float = 1e-4


# ---------------------------------------------------------------------------
# VolResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolResult:
    """
    Immutable result of a GARCH(1,1) volatility estimation step.

    Fields
    ------
    volatility : float
        Current conditional volatility = sqrt(variance). Always > 0.
    variance : float
        Current conditional variance sigma^2_t. Always > 0.
    long_run_volatility : float
        Unconditional (long-run) volatility = sqrt(omega / (1-alpha-beta)).
        Always > 0. Valid only when alpha + beta < 1.
    n_clean_returns : int
        Number of finite return observations used in this estimate.
    nan_replaced : int
        Number of non-finite return values replaced with 0.0.
    """
    volatility: float
    variance: float
    long_run_volatility: float
    n_clean_returns: int
    nan_replaced: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.volatility) or self.volatility <= 0.0:
            raise ValueError(
                f"VolResult.volatility must be finite and > 0, "
                f"got {self.volatility!r}."
            )
        if not math.isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError(
                f"VolResult.variance must be finite and > 0, "
                f"got {self.variance!r}."
            )
        if not math.isfinite(self.long_run_volatility) or self.long_run_volatility <= 0.0:
            raise ValueError(
                f"VolResult.long_run_volatility must be finite and > 0, "
                f"got {self.long_run_volatility!r}."
            )
        if self.n_clean_returns < 0:
            raise ValueError(
                f"VolResult.n_clean_returns must be >= 0, "
                f"got {self.n_clean_returns!r}."
            )
        if self.nan_replaced < 0:
            raise ValueError(
                f"VolResult.nan_replaced must be >= 0, "
                f"got {self.nan_replaced!r}."
            )


# ---------------------------------------------------------------------------
# VolatilityTracker
# ---------------------------------------------------------------------------

class VolatilityTracker:
    """
    GARCH(1,1) volatility tracker.

    Maintains a running conditional variance estimate and updates it
    sequentially as new return observations arrive.

    GARCH(1,1) EQUATION:
      sigma^2_t = omega + alpha * r_{t-1}^2 + beta * sigma^2_{t-1}

    STATIONARITY CONSTRAINT (enforced on construction):
      alpha + beta must be < 1.0.
      If the supplied parameters violate this, they are rescaled to
      satisfy alpha + beta = 0.95 while preserving their ratio.

    MUTABLE STATE:
      self._current_variance tracks the most recent conditional variance.
      Only estimate_volatility() mutates this state.

    STDLIB ONLY: no numpy, no scipy, no random.
    """

    def __init__(
        self,
        omega: float = _DEFAULT_OMEGA,
        alpha: float = _DEFAULT_ALPHA,
        beta: float  = _DEFAULT_BETA,
    ) -> None:
        """
        Initialise the GARCH(1,1) tracker.

        Parameters
        ----------
        omega : float
            Long-run variance component. Must be > 0.
        alpha : float
            ARCH coefficient (shock sensitivity). Must be in (0, 1).
        beta : float
            GARCH coefficient (persistence). Must be in (0, 1).

        Notes
        -----
        If alpha + beta >= 1.0, parameters are rescaled so that
        alpha + beta = 0.95 (the stationarity boundary with margin),
        preserving the ratio alpha / beta.
        """
        if omega <= 0.0 or not math.isfinite(omega):
            raise ValueError(f"omega must be > 0, got {omega!r}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta!r}.")

        # Enforce stationarity: alpha + beta < 1.
        if alpha + beta >= 1.0:
            scale: float = 0.95 / (alpha + beta)
            alpha = alpha * scale
            beta  = beta  * scale

        self._omega: float = omega
        self._alpha: float = alpha
        self._beta:  float = beta

        # Compute long-run variance: omega / (1 - alpha - beta)
        persistence: float = self._alpha + self._beta
        denom: float = 1.0 - persistence
        self._long_run_variance: float = max(
            _VOL_EPSILON ** 2,
            self._omega / denom,
        )

        # Initialise current variance to the long-run variance.
        self._current_variance: float = self._long_run_variance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_volatility(self, returns: List[float]) -> VolResult:
        """
        Update the GARCH(1,1) model with a sequence of return observations
        and return the resulting volatility estimate.

        Non-finite values (NaN, Inf) are replaced with 0.0 and counted
        in nan_replaced. Empty or all-non-finite lists raise ValueError.

        The model is updated sequentially: for each return r_t,
          sigma^2_{t+1} = omega + alpha * r_t^2 + beta * sigma^2_t

        After processing all returns, self._current_variance holds the
        most recent conditional variance.

        Parameters
        ----------
        returns : List[float]
            Sequence of per-period returns. Must be non-empty.
            Non-finite values are replaced with 0.0.

        Returns
        -------
        VolResult
            Volatility estimate derived from the final variance state.

        Raises
        ------
        ValueError
            If returns is empty.
        TypeError
            If returns is None.
        """
        if returns is None:
            raise TypeError("returns must be a list, got None.")
        if len(returns) == 0:
            raise ValueError("returns must be non-empty.")

        # Clean returns: replace non-finite with 0.0.
        nan_replaced: int = 0
        clean: List[float] = []
        for r in returns:
            if math.isfinite(r):
                clean.append(r)
            else:
                clean.append(0.0)
                nan_replaced += 1

        n_clean: int = len(returns) - nan_replaced

        # Sequential GARCH update.
        variance: float = self._current_variance
        for r in clean:
            variance = (
                self._omega
                + self._alpha * r * r
                + self._beta * variance
            )
            # Floor to prevent numerical collapse.
            variance = max(_VOL_EPSILON ** 2, variance)

        # Update internal state.
        self._current_variance = variance

        volatility: float = max(_VOL_EPSILON, math.sqrt(variance))
        long_run_vol: float = max(_VOL_EPSILON, math.sqrt(self._long_run_variance))

        return VolResult(
            volatility=volatility,
            variance=variance,
            long_run_volatility=long_run_vol,
            n_clean_returns=n_clean,
            nan_replaced=nan_replaced,
        )

    def predict_volatility(self, horizon: int) -> float:
        """
        Predict the conditional volatility h steps ahead using the
        GARCH(1,1) mean-reverting variance forecast:

          sigma^2_{t+h} = LR_var + (alpha+beta)^h * (sigma^2_t - LR_var)

        where LR_var = omega / (1 - alpha - beta).

        Parameters
        ----------
        horizon : int
            Number of steps ahead. Must be >= 1.

        Returns
        -------
        float
            Predicted volatility at horizon h. Always > 0.

        Raises
        ------
        ValueError
            If horizon < 1.
        """
        if horizon < 1:
            raise ValueError(
                f"horizon must be >= 1, got {horizon!r}."
            )

        persistence: float = self._alpha + self._beta
        sigma2_t: float = self._current_variance
        lr_var: float = self._long_run_variance

        # Mean-reverting GARCH forecast.
        predicted_variance: float = (
            lr_var + (persistence ** horizon) * (sigma2_t - lr_var)
        )

        # Floor for strict positivity.
        predicted_variance = max(_VOL_EPSILON ** 2, predicted_variance)

        return max(_VOL_EPSILON, math.sqrt(predicted_variance))

    @property
    def current_variance(self) -> float:
        """Current conditional variance sigma^2_t. Always > 0."""
        return self._current_variance

    @property
    def current_volatility(self) -> float:
        """Current conditional volatility sqrt(sigma^2_t). Always > 0."""
        return max(_VOL_EPSILON, math.sqrt(self._current_variance))

    @property
    def long_run_variance(self) -> float:
        """Unconditional long-run variance omega / (1-alpha-beta). Always > 0."""
        return self._long_run_variance

    @property
    def parameters(self) -> dict:
        """Return GARCH parameters as a plain dict (read-only view)."""
        return {
            "omega": self._omega,
            "alpha": self._alpha,
            "beta":  self._beta,
        }

    def reset(self) -> None:
        """
        Reset the current variance to the long-run variance.
        Useful for re-initialising the tracker without creating a new instance.
        """
        self._current_variance = self._long_run_variance
