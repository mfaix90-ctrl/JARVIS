# jarvis/utils/constants.py
# Version: 6.1.0
# HASH-PROTECTED -- NEVER OVERWRITE DIRECTLY AT RUNTIME.
# All constants below are hash-protected via THRESHOLD_MANIFEST.json.
# Changes require signed release + updated THRESHOLD_MANIFEST.json.
# CI blocks merge on hash mismatch.
#
# Standard import pattern:
#   from jarvis.utils.constants import (
#       JOINT_RISK_MULTIPLIER_TABLE,
#       MAX_DRAWDOWN_THRESHOLD,
#       VOL_COMPRESSION_TRIGGER,
#       SHOCK_EXPOSURE_CAP,
#       MAX_DECISION_CONTEXT,
#       BASE_SELECTIVITY_THRESHOLD,
#       DURATION_STRESS_Z_LIMIT,
#       FRAGILITY_HIGH_THRESHOLD,
#       QUALITY_SCORE_CAP_UNDER_UNCERTAINTY,
#       QUALITY_SCORE_MIN_FLOOR,
#   )

from jarvis.core.regime import GlobalRegimeState, CorrelationRegimeState


# ---------------------------------------------------------------------------
# RISK ENGINE THRESHOLDS (FAS v6.1.0)
# ---------------------------------------------------------------------------

MAX_DRAWDOWN_THRESHOLD:  float = 0.15    # 15% -- Hard Limit
VOL_COMPRESSION_TRIGGER: float = 0.30    # 30% ann. Vol -> Risk Compression
SHOCK_EXPOSURE_CAP:      float = 0.25    # Max 25% Exposure at Shock / Clip C floor


# ---------------------------------------------------------------------------
# JOINT MACRO x CORRELATION RISK MULTIPLIER TABLE (Phase 6A)
# ---------------------------------------------------------------------------
# Deterministic lookup table. No stochastic behavior.
# Hash-registered in THRESHOLD_MANIFEST.json under "joint_risk_multiplier_table".
# Applied in RiskEngine.assess() (Clip C / JRM path).
# Backward compatible: if either input is None, multiplier defaults to 1.0.
#
# Keys: GlobalRegimeState (macro regime context)
#       CorrelationRegimeState (cross-asset correlation state)
#
# No other key type is permitted. All keys are canonical enum instances
# from jarvis.core.regime (Single Authoritative Regime Source rule).
#
# Numeric values are identical to prior version. Only key types have changed
# from non-canonical strings to canonical enum instances.
#
# Canonical enum alignment:
#   CorrelationRegimeState.COUPLED    -- unchanged (already canonical)
#     (regime.py comment: "replaces CRISIS_COUPLING string from v6.0")

JOINT_RISK_MULTIPLIER_TABLE: dict = {
    GlobalRegimeState.RISK_ON: {
        CorrelationRegimeState.DIVERGENCE: 1.00,
        CorrelationRegimeState.COUPLED:    1.15,
        CorrelationRegimeState.BREAKDOWN:  1.40,
    },
    GlobalRegimeState.TRANSITION: {
        CorrelationRegimeState.DIVERGENCE: 1.15,
        CorrelationRegimeState.COUPLED:    1.35,
        CorrelationRegimeState.BREAKDOWN:  1.65,
    },
    GlobalRegimeState.RISK_OFF: {
        CorrelationRegimeState.DIVERGENCE: 1.40,
        CorrelationRegimeState.COUPLED:    1.65,
        CorrelationRegimeState.BREAKDOWN:  2.00,
    },
}


# ---------------------------------------------------------------------------
# RISK ENGINE FIXED LITERALS (FAS v6.1.0 DET-06)
# Previously implicit in risk_engine.py source; now hash-registered in
# THRESHOLD_MANIFEST.json under "crisis_damping_factor" and "vol_adjustment_cap".
# ---------------------------------------------------------------------------
CRISIS_DAMPING_FACTOR: float = 0.75   # Post-Clip-C CRISIS dampening (hash-protected)
VOL_ADJUSTMENT_CAP:    float = 3.0    # Maximum vol_adjustment scalar (hash-protected)


# ---------------------------------------------------------------------------
# DECISION MEMORY CONTEXT
# ---------------------------------------------------------------------------
MAX_DECISION_CONTEXT: int = 200


# ---------------------------------------------------------------------------
# SELECTIVITY THRESHOLD GOVERNANCE
# ---------------------------------------------------------------------------
BASE_SELECTIVITY_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# REGIME DURATION STRESS
# ---------------------------------------------------------------------------
DURATION_STRESS_Z_LIMIT: float = 2.0


# ---------------------------------------------------------------------------
# SIGNAL FRAGILITY CLASSIFICATION
# ---------------------------------------------------------------------------
FRAGILITY_HIGH_THRESHOLD: float = 0.65


# ---------------------------------------------------------------------------
# DECISION QUALITY SCORE GOVERNANCE
# ---------------------------------------------------------------------------
QUALITY_SCORE_CAP_UNDER_UNCERTAINTY: float = 0.60

QUALITY_SCORE_MIN_FLOOR: float = 0.05
