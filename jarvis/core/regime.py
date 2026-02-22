# jarvis/core/regime.py
# Version: 6.0.1
# SINGLE AUTHORITATIVE REGIME SOURCE FOR THE ENTIRE JARVIS SYSTEM.
#
# All regime types used anywhere in the system MUST be imported from this file.
# No other file may define regime enums, regime string dicts, or regime mapping logic.
#
# Standard import pattern:
#   from jarvis.core.regime import (
#       GlobalRegimeState, AssetRegimeState, AssetClass,
#       CorrelationRegimeState, HierarchicalRegime,
#       REGIME_VOL_MULTIPLIER, REGIME_VOL_MULTIPLIER_CRISIS,
#       map_s05_to_canonical, map_latent_int_to_canonical
#   )

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, unique
from typing import Dict, Optional, Tuple
import hashlib
import json


# ---------------------------------------------------------------------------
# CANONICAL ENUM DEFINITIONS
# ---------------------------------------------------------------------------

@unique
class GlobalRegimeState(Enum):
    """
    Top-level macro regime. Asset-class agnostic.
    Authoritative producer: GlobalMacroRegime (jarvis/intelligence/regime_detector.py).
    Consumers: all layers except P0 classification enforcement.
    Stored in: SystemState.regime (as .value string) for logging/monitoring only.
    Decision logic MUST use HierarchicalRegime passed as parameter, not SystemState.regime.
    """
    RISK_ON    = "RISK_ON"     # Risk appetite high; spreads tight; equities bid
    RISK_OFF   = "RISK_OFF"    # Flight to safety; spreads wide; volatility elevated
    TRANSITION = "TRANSITION"  # Regime change in progress; global_confidence < 0.5
    CRISIS     = "CRISIS"      # Systemic stress; all correlations -> 1
    UNKNOWN    = "UNKNOWN"     # Insufficient data; startup or data gap


@unique
class AssetRegimeState(Enum):
    """
    Asset-class-specific regime. Conditioned on GlobalRegimeState.
    All asset classes use this single enum.
    Replaces the S05 Regime enum (LOW_V_TREND, HIGH_V_TREND, LOW_V_REV,
    HIGH_V_REV, CRISIS) which is removed in v6.0.1.

    DIRECTIONAL DISTINCTION CONTRACT:
      TRENDING_UP and TRENDING_DOWN are strictly distinct.
      No mapping may collapse them to a single value.
      Direction is determined by the detector at detection time.

    Sub-regime strings (stored in HierarchicalRegime.sub_regime) provide
    asset-specific metadata for logging only. No module may branch on them.
    """
    TRENDING_UP     = "TRENDING_UP"     # Directional uptrend; momentum positive
    TRENDING_DOWN   = "TRENDING_DOWN"   # Directional downtrend; momentum negative
    RANGING_TIGHT   = "RANGING_TIGHT"   # Low volatility mean-reversion
    RANGING_WIDE    = "RANGING_WIDE"    # High volatility mean-reversion
    HIGH_VOLATILITY = "HIGH_VOLATILITY" # Volatility spike; no clear direction
    SHOCK           = "SHOCK"           # Extreme move; OOD likely
    RECOVERY        = "RECOVERY"        # Post-shock stabilization
    UNKNOWN         = "UNKNOWN"         # Insufficient data or model uncertainty


@unique
class AssetClass(Enum):
    """
    Canonical asset class identifiers.
    Used as keys in HierarchicalRegime.asset_regimes and asset_confidences.
    """
    CRYPTO      = "crypto"
    FOREX       = "forex"
    INDICES     = "indices"
    COMMODITIES = "commodities"
    RATES       = "rates"


@unique
class CorrelationRegimeState(Enum):
    """
    Cross-asset correlation structure.
    Replaces S21-25 local string "DECOUPLED" / "COUPLED" / "CRISIS_COUPLING".
    Authoritative producer: CorrelationRegimeDetector (jarvis/intelligence/regime_detector.py).
    """
    NORMAL     = "NORMAL"      # Correlations within historical norms
    COUPLED    = "COUPLED"     # Above-normal correlation; diversification reduced
    BREAKDOWN  = "BREAKDOWN"   # Correlations -> 1; crisis coupling; no diversification
                               # (replaces "CRISIS_COUPLING" string from v6.0)
    DIVERGENCE = "DIVERGENCE"  # Assets decoupling; regime-specific drivers dominant


# ---------------------------------------------------------------------------
# REGIME VOLATILITY MULTIPLIERS (canonical source — single definition)
# ---------------------------------------------------------------------------
# Used by ConfidenceZoneEngine. Keys are AssetRegimeState.value strings.
# Included in THRESHOLD_MANIFEST hash computation.

REGIME_VOL_MULTIPLIER: Dict[str, float] = {
    "TRENDING_UP":     1.0,
    "TRENDING_DOWN":   1.2,
    "RANGING_TIGHT":   0.8,
    "RANGING_WIDE":    1.4,
    "HIGH_VOLATILITY": 1.8,
    "SHOCK":           3.0,
    "RECOVERY":        1.6,
    "UNKNOWN":         2.5,
}
# GlobalRegimeState.CRISIS overrides all asset-level multipliers unconditionally.
REGIME_VOL_MULTIPLIER_CRISIS: float = 3.0


# ---------------------------------------------------------------------------
# CRISIS PRIORITY ORDER (DETERMINISTIC — NO EXCEPTIONS)
# ---------------------------------------------------------------------------
# Applied by HierarchicalRegimeDetector.detect() in strict order.
# First matching priority wins and sets GlobalRegimeState = CRISIS.
#
# PRIORITY 1 (HIGHEST): Hard Macro Crisis Detection
#   Condition: GlobalMacroRegime detects systemic macro stress
#              (VIX > macro_crisis_threshold AND credit_spreads > crisis_spread_threshold)
#   Action: GlobalRegimeState=CRISIS; all AssetRegimeState=SHOCK;
#           CorrelationRegimeState=BREAKDOWN
#
# PRIORITY 2: Asset-Level Capitulation Override
#   Condition: Any single asset detector returns AssetRegimeState.SHOCK
#              AND global_confidence < REGIME_CONFIDENCE_MIN
#   Action: GlobalRegimeState=CRISIS; all AssetRegimeState=SHOCK;
#           CorrelationRegimeState=BREAKDOWN
#
# PRIORITY 3: Correlation BREAKDOWN
#   Condition: CorrelationRegimeDetector returns CorrelationRegimeState.BREAKDOWN
#   Action: GlobalRegimeState=CRISIS; all AssetRegimeState=SHOCK
#
# PRIORITY 4: Confidence Threshold Force
#   Condition: global_confidence < REGIME_CONFIDENCE_CRISIS_FORCE
#   Action: GlobalRegimeState=CRISIS; all AssetRegimeState=SHOCK;
#           CorrelationRegimeState=BREAKDOWN
#
# PRIORITY 5 (LOWEST): Standard Regime Detection
#   Condition: None of priorities 1-4 triggered
#   Action: Use HMM-derived GlobalRegimeState and per-asset AssetRegimeState
#
# CRISIS OVERRIDE INVARIANT (enforced in HierarchicalRegime.create()):
#   IF GlobalRegimeState == CRISIS:
#     THEN for ALL active AssetClass: AssetRegimeState MUST be SHOCK
#     AND CorrelationRegimeState MUST be BREAKDOWN
#   Violation raises ValueError — construction is blocked.


# ---------------------------------------------------------------------------
# CANONICAL REGIME DATA STRUCTURE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchicalRegime:
    """
    Canonical regime transmission object for the entire JARVIS system.

    IMMUTABILITY CONTRACT:
      frozen=True — instances cannot be modified after construction.
      All updates produce a new HierarchicalRegime via HierarchicalRegime.create().
      Old instances are discarded or stored in history; never mutated.

    PASSING CONTRACT:
      Passed as a single parameter to any function requiring regime context.
      Functions must not extract and cache individual fields as module state.
      Functions must not cache HierarchicalRegime beyond a single pipeline cycle.

    HASH CONTRACT (v6.0.1):
      regime_hash is computed from content fields ONLY.
      timestamp is stored but EXCLUDED from hash computation.
      This ensures a deterministic hash for identical regime states regardless
      of the clock time at which the object was created.

    AUTHORSHIP:
      Created exclusively by HierarchicalRegimeDetector.
      No other class may call HierarchicalRegime.create().

    CONSUMER CONTRACT:
      All decision-logic regime reads MUST use this object.
      SystemState.regime (GlobalRegimeState.value string) is for
      logging and monitoring only — never for decision branching.
      No module may branch on sub_regime strings.
    """
    global_regime:      GlobalRegimeState
    asset_regimes:      Dict[AssetClass, AssetRegimeState]
    correlation_regime: CorrelationRegimeState
    global_confidence:  float                       # R_global in [0, 1]
    asset_confidences:  Dict[AssetClass, float]     # R_asset[class] in [0, 1]
    sub_regime:         Dict[AssetClass, str]       # metadata only — no logic branching
    timestamp:          str                         # UTC ISO-8601; excluded from hash
    sequence_id:        int                         # monotonically increasing
    regime_hash:        str                         # SHA-256[:16] of content (timestamp excluded)

    @staticmethod
    def create(
        global_regime:      GlobalRegimeState,
        asset_regimes:      Dict[AssetClass, AssetRegimeState],
        correlation_regime: CorrelationRegimeState,
        global_confidence:  float,
        asset_confidences:  Dict[AssetClass, float],
        sub_regime:         Dict[AssetClass, str],
        sequence_id:        int,
    ) -> "HierarchicalRegime":
        """
        Factory method — only valid construction path.
        Validates all fields. Enforces crisis override invariant.
        Raises ValueError on any constraint violation.
        """
        if not (0.0 <= global_confidence <= 1.0):
            raise ValueError(
                f"global_confidence out of [0,1]: {global_confidence}"
            )
        for ac, conf in asset_confidences.items():
            if not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"asset_confidence[{ac}] out of [0,1]: {conf}"
                )
        if sequence_id < 0:
            raise ValueError(f"sequence_id must be >= 0: {sequence_id}")

        # Enforce CRISIS override invariant
        if global_regime == GlobalRegimeState.CRISIS:
            for ac, ar in asset_regimes.items():
                if ar != AssetRegimeState.SHOCK:
                    raise ValueError(
                        f"CRISIS override violated: asset_regimes[{ac}]={ar.value}, "
                        f"must be SHOCK when global_regime=CRISIS"
                    )
            if correlation_regime != CorrelationRegimeState.BREAKDOWN:
                raise ValueError(
                    "CRISIS override violated: correlation_regime must be BREAKDOWN "
                    "when global_regime=CRISIS"
                )

        # Compute content-based hash — timestamp EXCLUDED
        payload = {
            "global_regime":      global_regime.value,
            "asset_regimes":      {k.value: v.value for k, v in asset_regimes.items()},
            "correlation_regime": correlation_regime.value,
            "global_confidence":  round(global_confidence, 8),
            "asset_confidences":  {k.value: round(v, 8) for k, v in asset_confidences.items()},
            "sub_regime":         {k.value: v for k, v in sub_regime.items()},
            "sequence_id":        sequence_id,
        }
        regime_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        ts = datetime.utcnow().isoformat()

        return HierarchicalRegime(
            global_regime=global_regime,
            asset_regimes=asset_regimes,
            correlation_regime=correlation_regime,
            global_confidence=global_confidence,
            asset_confidences=asset_confidences,
            sub_regime=sub_regime,
            timestamp=ts,
            sequence_id=sequence_id,
            regime_hash=regime_hash,
        )

    def get_asset_regime(self, asset_class: AssetClass) -> AssetRegimeState:
        """Safe accessor. Returns UNKNOWN if asset_class not present. Never raises KeyError."""
        return self.asset_regimes.get(asset_class, AssetRegimeState.UNKNOWN)

    def get_asset_confidence(self, asset_class: AssetClass) -> float:
        """Safe accessor. Returns 0.0 if asset_class not present."""
        return self.asset_confidences.get(asset_class, 0.0)

    def is_crisis(self) -> bool:
        """True iff global_regime == CRISIS. No other check required."""
        return self.global_regime == GlobalRegimeState.CRISIS

    def to_global_state_string(self) -> str:
        """
        Returns GlobalRegimeState.value for SystemState.regime storage.
        Used ONLY for SystemState logging — never for decision logic.
        """
        return self.global_regime.value


# ---------------------------------------------------------------------------
# BACKWARD-COMPATIBILITY MAPPING FUNCTIONS
# ---------------------------------------------------------------------------
# These functions exist solely for migration path from v6.0 to v6.0.1.
# All call sites must be migrated to native canonical types.
# Functions remain until assess_build.py confirms zero call sites remain.

_S05_TO_CANONICAL: Dict[str, Tuple[GlobalRegimeState, AssetRegimeState]] = {
    "LOW_V_TREND":  (GlobalRegimeState.RISK_ON,    AssetRegimeState.TRENDING_UP),
    "HIGH_V_TREND": (GlobalRegimeState.TRANSITION, AssetRegimeState.TRENDING_UP),
    "LOW_V_REV":    (GlobalRegimeState.RISK_ON,    AssetRegimeState.RANGING_TIGHT),
    "HIGH_V_REV":   (GlobalRegimeState.RISK_OFF,   AssetRegimeState.RANGING_WIDE),
    "CRISIS":       (GlobalRegimeState.CRISIS,      AssetRegimeState.SHOCK),
}

def map_s05_to_canonical(
    s05_regime: str,
) -> Tuple[GlobalRegimeState, AssetRegimeState]:
    """
    Deterministic mapping from removed S05 Regime string to canonical pair.
    Raises KeyError for unknown input — no silent fallback.
    """
    if s05_regime not in _S05_TO_CANONICAL:
        raise KeyError(
            f"Unknown S05 regime value: '{s05_regime}'. "
            f"Valid: {list(_S05_TO_CANONICAL.keys())}"
        )
    return _S05_TO_CANONICAL[s05_regime]


_LATENT_INT_TO_ASSET_REGIME: Dict[int, AssetRegimeState] = {
    0: AssetRegimeState.TRENDING_UP,
    1: AssetRegimeState.HIGH_VOLATILITY,
    2: AssetRegimeState.RANGING_TIGHT,
    3: AssetRegimeState.RANGING_WIDE,
    4: AssetRegimeState.SHOCK,
}

def map_latent_int_to_canonical(regime_int: int) -> AssetRegimeState:
    """
    Maps Kalman LatentState.regime integer (0-4) to AssetRegimeState.
    Internal use only: StateEstimator -> HierarchicalRegimeDetector boundary.
    Raises KeyError for out-of-range values — no silent fallback.
    """
    if regime_int not in _LATENT_INT_TO_ASSET_REGIME:
        raise KeyError(
            f"LatentState.regime integer out of range: {regime_int}. Valid: 0-4"
        )
    return _LATENT_INT_TO_ASSET_REGIME[regime_int]


# ---------------------------------------------------------------------------
# MODULE WRITE PERMISSIONS FOR REGIME FIELDS
# ---------------------------------------------------------------------------
# Field                              | Only Writer                     | All Others
# -----------------------------------|----------------------------------|------------
# HierarchicalRegime (new instance)  | HierarchicalRegimeDetector      | READ ONLY
# SystemState.regime (str value)     | GlobalSystemStateController     | READ ONLY
# LatentState.regime (int)           | StateEstimator                  | READ ONLY
# sub_regime: Dict[AssetClass, str]  | HierarchicalRegimeDetector      | READ ONLY
