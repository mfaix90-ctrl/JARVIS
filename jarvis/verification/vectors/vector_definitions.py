# jarvis/verification/vectors/vector_definitions.py
# Version: 1.0.0
# Fixed, version-controlled input matrix for the Deterministic Verification Harness.
# Authority: DVH Implementation Blueprint v1.0.0 Section 7.
#
# NO VECTOR IS GENERATED AT RUNTIME. NO VECTOR IS SAMPLED.
# All values are rational constants with exact floating-point representations.
# The matrix is identical on every harness run for a given harness version.
#
# Execution order (Section 7.2): G-VOL, G-DD, G-MU, G-RP, G-JM, G-CR, G-BC, G-EX.
# Within each group: ascending numeric order of vector ID suffix.
#
# All macro_regime and correlation_regime values use canonical enum instances
# from jarvis.core.regime (Single Authoritative Regime Source rule).
# No plain string representations of regime values are used as data.
#
# Canonical enum alignment for JRM parameters:
#   CorrelationRegimeState.COUPLED    -- unchanged

from typing import Optional
from jarvis.core.regime import GlobalRegimeState, CorrelationRegimeState
from jarvis.verification.data_models.input_vector import InputVector


# ---------------------------------------------------------------------------
# CANONICAL RETURNS HISTORY ARRAYS (Section 5.1)
# ---------------------------------------------------------------------------

# RH-01 NORMAL: vol below VOL_COMPRESSION_TRIGGER (0.30), p95-DD below 0.15
_RH_NORMAL: tuple = tuple([0.005] * 30 + [-0.003] * 10 + [0.002] * 10)

# RH-02 HIGH-VOL: annualised EWMA vol strictly above 0.30.
_RH_HIGH_VOL: tuple = tuple(
    [0.02, -0.025, 0.022, -0.021, 0.019] * 10
)

# RH-03 HIGH-DD: p95 drawdown strictly above MAX_DRAWDOWN_THRESHOLD (0.15).
_RH_HIGH_DD: tuple = tuple(
    [-0.008, -0.006, -0.007, -0.005, -0.009] * 10
)

# RH-04 BOUNDARY: values near thresholds but not crossing them.
_RH_BOUNDARY: tuple = tuple([0.001, -0.001, 0.0015, -0.0015] * 12 + [0.001] * 2)


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def _make_vector(
    vector_id:          str,
    group_id:           str,
    returns:            tuple,
    regime_str:         str,
    meta_uncertainty:   float,
    macro_regime:       Optional[GlobalRegimeState] = None,
    correlation_regime: Optional[CorrelationRegimeState] = None,
    realized_vol:       Optional[float] = None,
    target_vol:         Optional[float] = None,
    regime_posterior:   Optional[float] = None,
    expect_exception:   bool = False,
    description:        str = "",
) -> InputVector:
    return InputVector(
        vector_id=vector_id,
        group_id=group_id,
        returns_history=returns,
        current_regime_str=regime_str,
        meta_uncertainty=meta_uncertainty,
        macro_regime=macro_regime,
        correlation_regime=correlation_regime,
        realized_vol=realized_vol,
        target_vol=target_vol,
        regime_posterior=regime_posterior,
        expect_exception=expect_exception,
        description=description,
    )


# ---------------------------------------------------------------------------
# GROUP G-VOL: Volatility boundary cases (Section 5.2)
# ---------------------------------------------------------------------------

_G_VOL = [
    _make_vector(
        "VOL-01", "G-VOL", _RH_NORMAL, "RISK_ON", 0.2,
        realized_vol=None, target_vol=None,
        description="realized_vol=None, target_vol=None -- vol_adjustment identity.",
    ),
    _make_vector(
        "VOL-02", "G-VOL", _RH_NORMAL, "RISK_ON", 0.2,
        realized_vol=0.15, target_vol=0.10,
        description="realized_vol > target_vol -- vol_adjustment < 1.0.",
    ),
    _make_vector(
        "VOL-03", "G-VOL", _RH_NORMAL, "RISK_ON", 0.2,
        realized_vol=0.10, target_vol=0.15,
        description="realized_vol < target_vol -- vol_adjustment > 1.0.",
    ),
    _make_vector(
        "VOL-04", "G-VOL", _RH_NORMAL, "RISK_ON", 0.2,
        realized_vol=0.01, target_vol=0.30,
        description="Extreme vol scaling: ratio hits 3.0 cap.",
    ),
    _make_vector(
        "VOL-05", "G-VOL", _RH_HIGH_VOL, "RISK_ON", 0.2,
        realized_vol=0.40, target_vol=0.20,
        description="High-vol returns with vol scaling active.",
    ),
]


# ---------------------------------------------------------------------------
# GROUP G-DD: Drawdown boundary cases
# ---------------------------------------------------------------------------

_G_DD = [
    _make_vector(
        "DD-01", "G-DD", _RH_NORMAL, "RISK_ON", 0.2,
        description="Normal returns: no risk compression from DD path.",
    ),
    _make_vector(
        "DD-02", "G-DD", _RH_HIGH_DD, "RISK_ON", 0.2,
        description="High-DD returns: risk compression triggered by DD path.",
    ),
    _make_vector(
        "DD-03", "G-DD", _RH_BOUNDARY, "RISK_ON", 0.2,
        description="Boundary returns: no compression expected.",
    ),
    _make_vector(
        "DD-04", "G-DD", _RH_HIGH_VOL, "RISK_ON", 0.2,
        description="High-vol returns: risk compression triggered by vol path.",
    ),
]


# ---------------------------------------------------------------------------
# GROUP G-MU: meta_uncertainty boundary cases
# ---------------------------------------------------------------------------

_G_MU = [
    _make_vector(
        "MU-01", "G-MU", _RH_NORMAL, "RISK_ON", 0.0,
        description="meta_uncertainty=0.0 -- no uncertainty penalty.",
    ),
    _make_vector(
        "MU-02", "G-MU", _RH_NORMAL, "RISK_ON", 1.0,
        description="meta_uncertainty=1.0 -- E_pre_clip forced to zero; Clip B floor active.",
    ),
    _make_vector(
        "MU-03", "G-MU", _RH_NORMAL, "RISK_ON", 0.5,
        description="meta_uncertainty=0.5 -- 50% uncertainty penalty.",
    ),
    _make_vector(
        "MU-04", "G-MU", _RH_NORMAL, "RISK_ON", 0.999,
        description="meta_uncertainty near 1.0 -- near-zero exposure.",
    ),
]


# ---------------------------------------------------------------------------
# GROUP G-RP: regime_posterior boundary cases
# ---------------------------------------------------------------------------

_G_RP = [
    _make_vector(
        "RP-01", "G-RP", _RH_NORMAL, "RISK_ON", 0.2,
        regime_posterior=None,
        description="regime_posterior=None -- posterior_confidence identity 1.0.",
    ),
    _make_vector(
        "RP-02", "G-RP", _RH_NORMAL, "RISK_ON", 0.2,
        regime_posterior=0.0,
        description="regime_posterior=0.0 -- capital_base forced to zero; Clip B floor.",
    ),
    _make_vector(
        "RP-03", "G-RP", _RH_NORMAL, "RISK_ON", 0.2,
        regime_posterior=1.0,
        description="regime_posterior=1.0 -- full posterior confidence.",
    ),
    _make_vector(
        "RP-04", "G-RP", _RH_NORMAL, "RISK_ON", 0.2,
        regime_posterior=0.5,
        description="regime_posterior=0.5 -- halves capital_base.",
    ),
    _make_vector(
        "RP-05", "G-RP", _RH_NORMAL, "RISK_ON", 0.2,
        regime_posterior=1.5,
        description="regime_posterior above 1.0 -- silent clip to 1.0.",
    ),
    _make_vector(
        "RP-06", "G-RP", _RH_NORMAL, "RISK_ON", 0.2,
        regime_posterior=-0.5,
        description="regime_posterior below 0.0 -- silent clip to 0.0.",
    ),
]


# ---------------------------------------------------------------------------
# GROUP G-JM: Joint Macro x Correlation Risk Multiplier cases
# All macro_regime and correlation_regime values are canonical enum instances.
# ---------------------------------------------------------------------------

_G_JM = [
    _make_vector(
        "JM-01", "G-JM", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=None, correlation_regime=None,
        description="JRM inactive: both None -- joint_multiplier=1.0, Clip C suppressed.",
    ),
    _make_vector(
        "JM-02", "G-JM", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=GlobalRegimeState.RISK_ON,
        correlation_regime=CorrelationRegimeState.DIVERGENCE,
        description="RISK_ON/DIVERGENCE -- joint_multiplier=1.0, Clip C suppressed.",
    ),
    _make_vector(
        "JM-03", "G-JM", _RH_NORMAL, "RISK_ON", 0.99,
        macro_regime=GlobalRegimeState.RISK_OFF,
        correlation_regime=CorrelationRegimeState.BREAKDOWN,
        description="RISK_OFF/BREAKDOWN -- multiplier=2.0, Clip C active, floor enforced.",
    ),
    _make_vector(
        "JM-04", "G-JM", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=None,
        correlation_regime=CorrelationRegimeState.COUPLED,
        description="correlation_regime set but macro_regime=None -- JRM inactive.",
    ),
    _make_vector(
        "JM-05", "G-JM", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=GlobalRegimeState.TRANSITION,
        correlation_regime=CorrelationRegimeState.COUPLED,
        description="TRANSITION/COUPLED -- multiplier=1.35, Clip C active.",
    ),
    _make_vector(
        "JM-06", "G-JM", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=GlobalRegimeState.RISK_OFF,
        correlation_regime=CorrelationRegimeState.DIVERGENCE,
        description="RISK_OFF/DIVERGENCE -- multiplier=1.40, Clip C active.",
    ),
]


# ---------------------------------------------------------------------------
# GROUP G-CR: CRISIS regime cases
# ---------------------------------------------------------------------------

_G_CR = [
    _make_vector(
        "CR-01", "G-CR", _RH_NORMAL, "CRISIS", 0.2,
        macro_regime=None, correlation_regime=None,
        description="CRISIS regime, JRM inactive: dampening applied after Clip B.",
    ),
    _make_vector(
        "CR-02", "G-CR", _RH_NORMAL, "CRISIS", 0.99,
        macro_regime=GlobalRegimeState.RISK_OFF,
        correlation_regime=CorrelationRegimeState.BREAKDOWN,
        description="CRISIS regime + JRM active: dampening applied after Clip C.",
    ),
    _make_vector(
        "CR-03", "G-CR", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=GlobalRegimeState.RISK_OFF,
        correlation_regime=CorrelationRegimeState.BREAKDOWN,
        description="Non-CRISIS with JRM active: no dampening applied.",
    ),
    _make_vector(
        "CR-04", "G-CR", _RH_HIGH_DD, "CRISIS", 0.5,
        description="CRISIS + high-DD: both CRISIS and DD compression active.",
    ),
]


# ---------------------------------------------------------------------------
# GROUP G-BC: Backward compatibility pairs
# Both vectors in a pair MUST produce bit-identical ObservedOutput (INV-08).
# ---------------------------------------------------------------------------

_G_BC = [
    # BC pair 1: No optional parameters
    _make_vector(
        "BC-01A", "G-BC", _RH_NORMAL, "RISK_ON", 0.2,
        description="BC pair 1A: no optional params.",
    ),
    _make_vector(
        "BC-01B", "G-BC", _RH_NORMAL, "RISK_ON", 0.2,
        macro_regime=None, correlation_regime=None,
        realized_vol=None, target_vol=None, regime_posterior=None,
        description="BC pair 1B: all optional params at identity defaults.",
    ),
    # BC pair 2: High-vol case
    _make_vector(
        "BC-02A", "G-BC", _RH_HIGH_VOL, "RISK_OFF", 0.3,
        description="BC pair 2A: no optional params, high vol.",
    ),
    _make_vector(
        "BC-02B", "G-BC", _RH_HIGH_VOL, "RISK_OFF", 0.3,
        macro_regime=None, correlation_regime=None,
        realized_vol=None, target_vol=None, regime_posterior=None,
        description="BC pair 2B: all optional params at identity defaults.",
    ),
    # BC pair 3: CRISIS regime
    _make_vector(
        "BC-03A", "G-BC", _RH_NORMAL, "CRISIS", 0.1,
        description="BC pair 3A: CRISIS regime, no optional params.",
    ),
    _make_vector(
        "BC-03B", "G-BC", _RH_NORMAL, "CRISIS", 0.1,
        macro_regime=None, correlation_regime=None,
        realized_vol=None, target_vol=None, regime_posterior=None,
        description="BC pair 3B: all optional params at identity defaults.",
    ),
]

# Backward compatibility pair identifiers for BIC.
BC_PAIRS: tuple = (
    ("BC-01A", "BC-01B"),
    ("BC-02A", "BC-02B"),
    ("BC-03A", "BC-03B"),
)


# ---------------------------------------------------------------------------
# GROUP G-EX: Exception boundary cases (Section 7.4)
# ---------------------------------------------------------------------------

_RH_SHORT_EMPTY: tuple = ()
_RH_SHORT_19:    tuple = tuple([0.001] * 19)

_G_EX = [
    _make_vector(
        "EX-01", "G-EX", _RH_SHORT_EMPTY, "RISK_ON", 0.2,
        expect_exception=True,
        description="Empty returns: expect ValueError (< 20 returns).",
    ),
    _make_vector(
        "EX-02", "G-EX", _RH_SHORT_19, "RISK_ON", 0.2,
        expect_exception=True,
        description="19 returns: expect ValueError (< 20 minimum).",
    ),
]


# ---------------------------------------------------------------------------
# ORDERED FULL INPUT MATRIX (Section 7.2)
# ---------------------------------------------------------------------------

INPUT_MATRIX: tuple = tuple(
    _G_VOL + _G_DD + _G_MU + _G_RP + _G_JM + _G_CR + _G_BC + _G_EX
)

VECTOR_COUNT: int = len(INPUT_MATRIX)
MATRIX_VERSION: str = "1.0.0"
