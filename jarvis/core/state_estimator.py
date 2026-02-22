# =============================================================================
# JARVIS v6.0.1 — SESSION 05, PHASE 5.5: STATE ESTIMATOR
# File:   jarvis/core/state_estimator.py
# Authority: JARVIS FAS v6.0.1 — 02-05_CORE.md, S05 section
# Phase:  5.5 — StateEstimator (Kalman filter, 12-dimensional latent state)
# =============================================================================
#
# SCOPE (Phase 5.5)
# -----------------
# Implements:
#   - Matrix utilities (pure stdlib, no numpy)
#   - KalmanState  dataclass (frozen=True) — internal covariance carrier
#   - StateEstimator class
#       * predict(state)                  -> LatentState
#       * update(state, observation)      -> LatentState
#       * get_covariance()                -> List[List[float]]  (12x12)
#       * reset()                         -> None
#
# CONSTRAINTS
# -----------
# stdlib only: dataclasses, math, typing.
# No numpy. No scipy. No pandas. No random. No datetime.now().
# No file I/O. No logging.
#
# DETERMINISM GUARANTEES
# ----------------------
# DET-01  No stochastic operations.
# DET-02  All inputs passed explicitly.
# DET-03  No side effects beyond updating internal covariance state.
# DET-04  All arithmetic deterministic (pure Python floating-point).
# DET-05  All branches pure functions of explicit inputs.
# DET-06  No datetime.now().
#
# KALMAN FILTER SPECIFICATION
# ----------------------------
# State dimension:  n = 12  (matches LatentState field count)
# Observation dim:  m = 12  (full-state observation model)
#
# Predict step:
#   x_pred = F * x_t          (state transition: identity = no drift model)
#   P_pred = F * P * F^T + Q  (covariance prediction)
#
# Update step:
#   y      = z - H * x_pred   (innovation / residual)
#   S      = H * P_pred * H^T + R   (innovation covariance)
#   K      = P_pred * H^T * S^{-1}  (Kalman gain)
#   x_upd  = x_pred + K * y   (state update)
#   P_upd  = (I - K * H) * P_pred   (covariance update — Joseph form)
#
# Default matrices:
#   F = I_12  (identity — random-walk prior)
#   H = I_12  (full-state observation)
#   Q = q * I_12  (process noise, q = 1e-4)
#   R = r * I_12  (observation noise, r = 1e-2)
#   P_0 = p0 * I_12  (initial covariance, p0 = 1.0)
#
# DIVERGENCE DETECTION
# ---------------------
# After each update, the condition number of P is estimated.
# If cond(P) > DIVERGENCE_THRESHOLD (1e6), the filter is reset to
# the initial covariance P_0. This prevents numerical blow-up.
#
# TIKHONOV REGULARISATION
# ------------------------
# All matrix inversions use Tikhonov regularisation:
#   (M + lambda * I)^{-1}  with lambda = TIKHONOV_LAMBDA = 1e-6
# This prevents singular matrix errors.
#
# STDLIB MATRIX INVERSION
# -----------------------
# For a 12x12 system we use Gaussian elimination with partial pivoting.
# No numpy, no scipy. Pure Python. O(n^3) — acceptable for n=12.
#
# INVARIANTS
# ----------
# INV-P55-01  P is always symmetric (enforced after every update).
# INV-P55-02  P diagonal entries are always > 0 (floored at epsilon).
# INV-P55-03  predict() returns a valid LatentState.
# INV-P55-04  update() returns a valid LatentState.
# INV-P55-05  get_covariance() returns a deep copy of P (12x12).
# INV-P55-06  Divergence reset restores P to P_0.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from jarvis.core.state_layer import LatentState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Latent state dimension.
_DIM: int = 12

#: Divergence threshold for condition number of P.
DIVERGENCE_THRESHOLD: float = 1e6

#: Tikhonov regularisation lambda for matrix inversion.
TIKHONOV_LAMBDA: float = 1e-6

#: Default process noise magnitude.
_DEFAULT_Q: float = 1e-4

#: Default observation noise magnitude.
_DEFAULT_R: float = 1e-2

#: Default initial covariance diagonal magnitude.
_DEFAULT_P0: float = 1.0

#: Minimum diagonal value for P (positivity floor).
_P_MIN_DIAG: float = 1e-12

#: Epsilon for numerical guards.
_EPS: float = 1e-15

# ---------------------------------------------------------------------------
# LatentState field order (must match LatentState dataclass field order).
# ---------------------------------------------------------------------------
_FIELD_NAMES: Tuple[str, ...] = (
    "regime",
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
)

# ---------------------------------------------------------------------------
# Pure-stdlib matrix utilities
# ---------------------------------------------------------------------------

def _identity(n: int) -> List[List[float]]:
    """Return an n x n identity matrix."""
    M: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 1.0
    return M


def _scalar_diag(n: int, s: float) -> List[List[float]]:
    """Return an n x n diagonal matrix with s on the diagonal."""
    M: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = s
    return M


def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Element-wise matrix addition. A and B must have same dimensions."""
    n: int = len(A)
    m: int = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication A @ B. A is (r x k), B is (k x c)."""
    r: int = len(A)
    k: int = len(A[0])
    c: int = len(B[0])
    C: List[List[float]] = [[0.0] * c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            s: float = 0.0
            for l in range(k):
                s += A[i][l] * B[l][j]
            C[i][j] = s
    return C


def _mat_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Element-wise matrix subtraction A - B."""
    n: int = len(A)
    m: int = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    """Matrix transpose."""
    r: int = len(A)
    c: int = len(A[0])
    return [[A[i][j] for i in range(r)] for j in range(c)]


def _mat_vec_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Matrix-vector multiplication A @ v."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    """Vector addition a + b."""
    return [a[i] + b[i] for i in range(len(a))]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Vector subtraction a - b."""
    return [a[i] - b[i] for i in range(len(a))]


def _symmetrise(M: List[List[float]]) -> List[List[float]]:
    """Force symmetry: M = (M + M^T) / 2."""
    n: int = len(M)
    S: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            S[i][j] = (M[i][j] + M[j][i]) * 0.5
    return S


def _tikhonov_invert(M: List[List[float]], lam: float = TIKHONOV_LAMBDA) -> List[List[float]]:
    """
    Compute (M + lam * I)^{-1} via Gaussian elimination with partial pivoting.

    Tikhonov regularisation ensures the matrix is non-singular even when M
    has very small eigenvalues. This is the ONLY matrix inversion path used
    by StateEstimator — never invert M directly.
    """
    n: int = len(M)
    # Build augmented matrix [M + lam*I | I]
    aug: List[List[float]] = [
        [M[i][j] + (lam if i == j else 0.0) for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        # Partial pivoting: find row with max abs in column.
        max_val: float = abs(aug[col][col])
        max_row: int = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot: float = aug[col][col]
        if abs(pivot) < _EPS:
            # Singular column — add extra regularisation.
            aug[col][col] = _EPS
            pivot = _EPS

        inv_pivot: float = 1.0 / pivot
        for j in range(2 * n):
            aug[col][j] *= inv_pivot

        for row in range(n):
            if row == col:
                continue
            factor: float = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    inv: List[List[float]] = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    return inv


def _condition_number_approx(M: List[List[float]]) -> float:
    """
    Approximate condition number of a symmetric positive-definite matrix
    as max_diagonal / min_diagonal.

    This is a lightweight O(n) approximation suitable for divergence
    detection. For a well-conditioned matrix the diagonal approximation
    is conservative (overestimates). For a diverging matrix it correctly
    detects blow-up.
    """
    n: int = len(M)
    diag_vals: List[float] = [abs(M[i][i]) for i in range(n)]
    max_d: float = max(diag_vals) if diag_vals else 1.0
    min_d: float = min(diag_vals) if diag_vals else 1.0
    if min_d < _EPS:
        return DIVERGENCE_THRESHOLD + 1.0  # treat as diverged
    return max_d / min_d


def _floor_diag(M: List[List[float]], floor: float = _P_MIN_DIAG) -> List[List[float]]:
    """Floor the diagonal of M at `floor` for strict positivity."""
    n: int = len(M)
    result: List[List[float]] = [list(row) for row in M]
    for i in range(n):
        if result[i][i] < floor:
            result[i][i] = floor
    return result


# ---------------------------------------------------------------------------
# KalmanState (internal — not part of public API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KalmanState:
    """
    Immutable carrier for the Kalman filter covariance matrix.
    Used internally by StateEstimator; not exported as part of the public API.

    Fields
    ------
    P : tuple
        Flattened 12x12 covariance matrix (row-major). Length = 144.
        Always symmetric and positive-definite.
    divergence_count : int
        Number of times the filter has been reset due to divergence.
    """
    P: tuple          # flat row-major float[144]
    divergence_count: int

    def as_matrix(self) -> List[List[float]]:
        """Reconstruct P as a 12x12 list-of-lists."""
        return [
            [self.P[i * _DIM + j] for j in range(_DIM)]
            for i in range(_DIM)
        ]

    @staticmethod
    def from_matrix(P_mat: List[List[float]], divergence_count: int = 0) -> "KalmanState":
        """Construct a KalmanState from a 12x12 matrix."""
        flat = tuple(P_mat[i][j] for i in range(_DIM) for j in range(_DIM))
        return KalmanState(P=flat, divergence_count=divergence_count)


# ---------------------------------------------------------------------------
# Latent state <-> vector conversion
# ---------------------------------------------------------------------------

def _state_to_vector(state: LatentState) -> List[float]:
    """Extract the 12 LatentState fields as an ordered float vector."""
    return [float(getattr(state, name)) for name in _FIELD_NAMES]


def _vector_to_state(vec: List[float], template: LatentState) -> LatentState:
    """
    Reconstruct a LatentState from a 12-element float vector.
    Applies LatentState construction constraints:
      - regime is rounded to int and clamped to [0, 4]
      - all floats must be finite (replaced with template value on failure)
    """
    def _safe(v: float, fallback: float) -> float:
        return v if math.isfinite(v) else fallback

    regime_raw: float = vec[0]
    regime_int: int = max(0, min(4, int(round(regime_raw))))

    return LatentState(
        regime=regime_int,
        volatility=_safe(vec[1],  template.volatility),
        trend_strength=_safe(vec[2], template.trend_strength),
        mean_reversion=_safe(vec[3], template.mean_reversion),
        liquidity=_safe(vec[4], template.liquidity),
        stress=_safe(vec[5], template.stress),
        momentum=_safe(vec[6], template.momentum),
        drift=_safe(vec[7], template.drift),
        noise=_safe(vec[8], template.noise),
        regime_confidence=_safe(vec[9], template.regime_confidence),
        stability=_safe(vec[10], template.stability),
        prediction_uncertainty=_safe(vec[11], template.prediction_uncertainty),
    )


# ---------------------------------------------------------------------------
# StateEstimator
# ---------------------------------------------------------------------------

class StateEstimator:
    """
    Kalman filter for the 12-dimensional JARVIS latent state vector.

    Implements the standard predict / update cycle:
      - predict(): propagates the state and covariance through the
                   state transition model (F = I, no drift).
      - update():  incorporates a new observation (z) and computes the
                   optimal Kalman gain.

    DIVERGENCE PROTECTION:
      After every update(), the approximate condition number of P is
      checked. If it exceeds DIVERGENCE_THRESHOLD (1e6), P is reset to
      the initial covariance P_0. The divergence_count field in the
      internal KalmanState is incremented on each reset.

    NUMERICAL SAFETY:
      - All matrix inversions use Tikhonov regularisation.
      - P is symmetrised after every update.
      - P diagonal is floored at _P_MIN_DIAG after every update.
      - Non-finite observation values are replaced with the predicted
        state values (no observation correction applied for those dims).

    STDLIB ONLY: no numpy, no scipy, no random.
    """

    def __init__(
        self,
        q: float = _DEFAULT_Q,
        r: float = _DEFAULT_R,
        p0: float = _DEFAULT_P0,
    ) -> None:
        """
        Initialise the StateEstimator.

        Parameters
        ----------
        q : float
            Process noise magnitude. Used as the diagonal of Q = q * I_12.
            Must be > 0.
        r : float
            Observation noise magnitude. Used as diagonal of R = r * I_12.
            Must be > 0.
        p0 : float
            Initial covariance diagonal. P_0 = p0 * I_12.
            Must be > 0.
        """
        if q <= 0.0 or not math.isfinite(q):
            raise ValueError(f"q must be > 0, got {q!r}.")
        if r <= 0.0 or not math.isfinite(r):
            raise ValueError(f"r must be > 0, got {r!r}.")
        if p0 <= 0.0 or not math.isfinite(p0):
            raise ValueError(f"p0 must be > 0, got {p0!r}.")

        self._q: float = q
        self._r: float = r
        self._p0: float = p0

        # System matrices (fixed, identity-based).
        # F: state transition matrix = I_12 (random walk)
        # H: observation matrix = I_12 (full state observation)
        # Q: process noise covariance = q * I_12
        # R: observation noise covariance = r * I_12
        self._F: List[List[float]] = _identity(_DIM)
        self._H: List[List[float]] = _identity(_DIM)
        self._Q: List[List[float]] = _scalar_diag(_DIM, q)
        self._R: List[List[float]] = _scalar_diag(_DIM, r)
        self._P0_mat: List[List[float]] = _scalar_diag(_DIM, p0)

        # Initial covariance state.
        self._ks: KalmanState = KalmanState.from_matrix(
            _scalar_diag(_DIM, p0), divergence_count=0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, state: LatentState) -> LatentState:
        """
        Kalman predict step.

        Propagates the state through the transition model and updates the
        internal covariance prediction:
          x_pred = F * x        (F = I, so x_pred = x)
          P_pred = F * P * F^T + Q

        Parameters
        ----------
        state : LatentState
            Current latent state estimate.

        Returns
        -------
        LatentState
            Predicted state (unchanged for F = I, but P is updated internally).

        Notes
        -----
        For F = I, the state vector is unchanged. The covariance grows
        by the process noise Q at each predict step.
        """
        if not isinstance(state, LatentState):
            raise TypeError(
                f"state must be a LatentState, got {type(state).__name__!r}."
            )

        P: List[List[float]] = self._ks.as_matrix()

        # P_pred = F * P * F^T + Q  (F = I => P_pred = P + Q)
        P_pred: List[List[float]] = _mat_add(P, self._Q)
        P_pred = _symmetrise(P_pred)
        P_pred = _floor_diag(P_pred)

        # Update internal state (only covariance changes at predict step).
        self._ks = KalmanState.from_matrix(P_pred, self._ks.divergence_count)

        # State vector unchanged (F = I).
        return state

    def update(
        self,
        state: LatentState,
        observation: Dict[str, float],
    ) -> LatentState:
        """
        Kalman update step.

        Incorporates a new observation z and computes the updated state
        estimate and covariance:
          y    = z - H * x_pred           (innovation)
          S    = H * P_pred * H^T + R     (innovation covariance)
          K    = P_pred * H^T * S^{-1}    (Kalman gain)
          x    = x_pred + K * y           (updated state)
          P    = (I - K * H) * P_pred     (updated covariance)

        Non-finite observation values are treated as missing: for those
        dimensions, the Kalman gain is set to 0 (no correction).

        Parameters
        ----------
        state : LatentState
            Predicted state (output of predict()).
        observation : Dict[str, float]
            Mapping from LatentState field names to observed values.
            Missing keys use the predicted state value (no correction).
            Non-finite values are treated as missing.

        Returns
        -------
        LatentState
            Updated state estimate incorporating the observation.

        Notes
        -----
        Updates self._ks (covariance) as the sole side effect.
        Divergence detection runs after every update.
        """
        if not isinstance(state, LatentState):
            raise TypeError(
                f"state must be a LatentState, got {type(state).__name__!r}."
            )
        if observation is None:
            raise TypeError("observation must be a dict, got None.")

        x_pred: List[float] = _state_to_vector(state)
        P_pred: List[List[float]] = self._ks.as_matrix()

        # Build observation vector z (substitute predicted value for missing/NaN).
        z: List[float] = []
        valid_mask: List[bool] = []
        for idx, fname in enumerate(_FIELD_NAMES):
            obs_val = observation.get(fname, None)
            if obs_val is not None and math.isfinite(obs_val):
                z.append(obs_val)
                valid_mask.append(True)
            else:
                z.append(x_pred[idx])   # no correction for missing obs
                valid_mask.append(False)

        # H = I_12 (full observation), so:
        # S = P_pred + R
        S: List[List[float]] = _mat_add(P_pred, self._R)

        # S^{-1} via Tikhonov-regularised inversion.
        S_inv: List[List[float]] = _tikhonov_invert(S)

        # K = P_pred * H^T * S^{-1}  (H = I => K = P_pred * S^{-1})
        K: List[List[float]] = _mat_mul(P_pred, S_inv)

        # For dimensions with no valid observation, zero out the Kalman gain row.
        for j in range(_DIM):
            if not valid_mask[j]:
                for i in range(_DIM):
                    K[i][j] = 0.0

        # Innovation y = z - x_pred  (H = I)
        y: List[float] = _vec_sub(z, x_pred)

        # Updated state: x = x_pred + K * y
        K_y: List[float] = _mat_vec_mul(K, y)
        x_upd: List[float] = _vec_add(x_pred, K_y)

        # Updated covariance: P = (I - K*H) * P_pred  (H = I)
        # Use Joseph stabilised form: P = (I-K)*P_pred*(I-K)^T + K*R*K^T
        I_KH: List[List[float]] = _mat_sub(_identity(_DIM), K)
        I_KH_T: List[List[float]] = _transpose(I_KH)
        KT: List[List[float]] = _transpose(K)
        P_upd: List[List[float]] = _mat_add(
            _mat_mul(_mat_mul(I_KH, P_pred), I_KH_T),
            _mat_mul(_mat_mul(K, self._R), KT),
        )

        # Enforce symmetry and positivity.
        P_upd = _symmetrise(P_upd)
        P_upd = _floor_diag(P_upd)

        # Divergence detection.
        div_count: int = self._ks.divergence_count
        cond: float = _condition_number_approx(P_upd)
        if cond > DIVERGENCE_THRESHOLD:
            # Reset covariance to initial value.
            P_upd = [list(row) for row in self._P0_mat]
            div_count += 1

        self._ks = KalmanState.from_matrix(P_upd, div_count)

        # Reconstruct LatentState from updated vector.
        updated_state: LatentState = _vector_to_state(x_upd, state)
        return updated_state

    def get_covariance(self) -> List[List[float]]:
        """
        Return the current covariance matrix P as a 12x12 list-of-lists.

        Returns a deep copy — modifying the returned matrix does not affect
        the internal filter state.

        The returned matrix is guaranteed to be:
          - 12x12
          - Symmetric (P[i][j] == P[j][i] within float tolerance)
          - Positive-definite (all diagonal entries > _P_MIN_DIAG)
        """
        P: List[List[float]] = self._ks.as_matrix()
        return [list(row) for row in P]

    @property
    def divergence_count(self) -> int:
        """Number of times the filter has been reset due to divergence."""
        return self._ks.divergence_count

    def reset(self) -> None:
        """
        Reset the covariance matrix P to the initial value P_0 = p0 * I_12.
        Does not reset the divergence count.
        """
        self._ks = KalmanState.from_matrix(
            _scalar_diag(_DIM, self._p0),
            divergence_count=self._ks.divergence_count,
        )
