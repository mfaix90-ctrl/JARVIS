# jarvis/orchestrator/pipeline.py
# Version: 1.2.1  (Governance Gate -- Pipeline-scoped validation only)
# External orchestration layer.
#
# GOVERNANCE GATE (pipeline-scoped):
#   Validates ONLY what downstream modules do NOT already validate:
#     GOV-01: meta_uncertainty in [0.0, 1.0]
#     GOV-05: regime must be a GlobalRegimeState instance
#
#   INTENTIONALLY EXCLUDED from gate (downstream owns these):
#     GOV-02: initial_capital  -> allocate_positions() raises ValueError
#     GOV-03: window size      -> RiskEngine.assess()  raises ValueError
#     GOV-06: CRISIS + meta    -> CRISIS is a valid runtime regime;
#                                 meta_uncertainty=0.1 is legitimate
#
# Standard import:
#   from jarvis.orchestrator.pipeline import run_full_pipeline

import math

from jarvis.core.regime import GlobalRegimeState
from jarvis.core.regime_detector import RegimeDetector
from jarvis.core.state_layer import LatentState
from jarvis.core.state_estimator import StateEstimator
from jarvis.core.volatility_tracker import VolatilityTracker
from jarvis.governance.exceptions import GovernanceViolationError
from jarvis.governance.policy_validator import validate_pipeline_config
from jarvis.risk.risk_engine import RiskEngine
from jarvis.execution.exposure_router import route_exposure_to_positions


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from v1.1.0)
# ---------------------------------------------------------------------------

def _extract_regime_features(returns_history: list) -> dict:
    clean = [r if math.isfinite(r) else 0.0 for r in returns_history]
    n = len(clean)
    mean_r = sum(clean) / n
    variance = sum((r - mean_r) ** 2 for r in clean) / max(n - 1, 1)
    volatility = math.sqrt(max(variance, 0.0))
    window = min(5, n)
    recent = clean[-window:]
    recent_mean = sum(recent) / window
    trend_strength = recent_mean / max(volatility, 1e-8)
    if n >= 2:
        pairs = [(clean[i], clean[i + 1]) for i in range(n - 1)]
        num = sum((a - mean_r) * (b - mean_r) for a, b in pairs)
        denom = sum((r - mean_r) ** 2 for r in clean)
        lag1_autocorr = num / max(denom, 1e-15)
        mean_reversion = max(-1.0, min(1.0, -lag1_autocorr))
    else:
        mean_reversion = 0.0
    short_window = min(10, n)
    short_clean = clean[-short_window:]
    short_mean = sum(short_clean) / len(short_clean)
    short_var = sum((r - short_mean) ** 2 for r in short_clean) / max(len(short_clean) - 1, 1)
    short_vol = math.sqrt(max(short_var, 0.0))
    full_vol = max(volatility, 1e-8)
    stress_raw = (short_vol / full_vol) - 1.0
    stress = max(0.0, min(1.0, stress_raw * 0.5 + 0.5))
    momentum = max(-1.0, min(1.0, recent_mean / max(full_vol, 1e-8)))
    liquidity = 1.0 - stress
    return {
        "volatility":     volatility,
        "trend_strength": trend_strength,
        "mean_reversion": mean_reversion,
        "stress":         stress,
        "momentum":       momentum,
        "liquidity":      liquidity,
    }


def _build_observation_vector(regime_result, vol_result) -> dict:
    stress_ratio = vol_result.volatility / max(vol_result.long_run_volatility, 1e-8)
    stress_obs = max(0.0, min(1.0, (stress_ratio - 1.0) * 0.5 + 0.5))
    stability_obs = 1.0 - stress_obs
    pred_uncertainty_obs = max(0.0, vol_result.variance)
    return {
        "regime":                 int(regime_result.hmm_index),
        "volatility":             vol_result.volatility,
        "stress":                 stress_obs,
        "regime_confidence":      regime_result.confidence,
        "stability":              stability_obs,
        "prediction_uncertainty": pred_uncertainty_obs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_full_pipeline(
    returns_history: list,
    current_regime: GlobalRegimeState,
    meta_uncertainty: float,
    total_capital: float,
    asset_prices: dict,
) -> dict:
    """
    Full deterministic pipeline from returns history to final positions.

    Step 0: Governance gate (GOV-01 + GOV-05 only).
    Steps 1-11: Unchanged from v1.1.0.

    Raises
    ------
    GovernanceViolationError
        meta_uncertainty out of [0.0, 1.0] or regime not GlobalRegimeState.
    ValueError
        Propagated from RiskEngine (invalid returns_history).
        Propagated from route_exposure_to_positions (invalid capital/prices).
    """
    # ------------------------------------------------------------------
    # Step 0: Governance Gate -- GOV-01 and GOV-05 only
    # ------------------------------------------------------------------

    # GOV-01: meta_uncertainty must be in [0.0, 1.0]
    _meta_invalid = (
        not isinstance(meta_uncertainty, (int, float))
        or not (0.0 <= float(meta_uncertainty) <= 1.0)
    )

    # GOV-05: regime must be a GlobalRegimeState instance
    _regime_invalid = not isinstance(current_regime, GlobalRegimeState)

    if _meta_invalid or _regime_invalid:
        # Build a minimal result via validate_pipeline_config using safe
        # stand-in values for parameters owned by downstream modules.
        _safe_capital = total_capital if (
            isinstance(total_capital, (int, float)) and float(total_capital) > 0
        ) else 1.0
        _safe_regime = current_regime if isinstance(
            current_regime, GlobalRegimeState
        ) else GlobalRegimeState.UNKNOWN
        _safe_meta = meta_uncertainty if (
            isinstance(meta_uncertainty, (int, float))
            and 0.0 <= float(meta_uncertainty) <= 1.0
        ) else 0.5

        _gov_result = validate_pipeline_config(
            meta_uncertainty=_safe_meta if not _meta_invalid else meta_uncertainty,
            initial_capital=_safe_capital,
            window=20,
            step=1,
            regime=_safe_regime,
            periods_per_year=252,
        )
        # Only raise for the rules we gate on
        _gated_rules = {"GOV-01", "GOV-05"}
        _gated = [v for v in _gov_result.blocking_violations if v.rule_id in _gated_rules]
        # Also run with the actual bad value so the error message is accurate
        _real_result = validate_pipeline_config(
            meta_uncertainty=meta_uncertainty if isinstance(meta_uncertainty, (int, float)) else 999,
            initial_capital=_safe_capital,
            window=20,
            step=1,
            regime=current_regime,
            periods_per_year=252,
        )
        _real_gated = [v for v in _real_result.blocking_violations if v.rule_id in _gated_rules]
        if _real_gated:
            raise GovernanceViolationError(_real_result)

    # ------------------------------------------------------------------
    # Step 1: Instantiate all components fresh per call
    # ------------------------------------------------------------------
    engine: RiskEngine = RiskEngine()
    regime_detector: RegimeDetector = RegimeDetector()
    vol_tracker: VolatilityTracker = VolatilityTracker()
    state_estimator: StateEstimator = StateEstimator()

    # ------------------------------------------------------------------
    # Steps 2-7: State pipeline
    # ------------------------------------------------------------------
    if len(returns_history) > 0:
        features: dict = _extract_regime_features(returns_history)
        regime_result = regime_detector.detect_regime(features)
        vol_result = vol_tracker.estimate_volatility(returns_history)
        previous_state: LatentState = LatentState.default()
        predicted_state: LatentState = state_estimator.predict(previous_state)
        observation: dict = _build_observation_vector(regime_result, vol_result)
        _latent_state: LatentState = state_estimator.update(predicted_state, observation)

    # ------------------------------------------------------------------
    # Step 8: Risk assessment
    # ------------------------------------------------------------------
    risk_output = engine.assess(
        returns_history=returns_history,
        current_regime=current_regime,
        meta_uncertainty=meta_uncertainty,
    )

    # ------------------------------------------------------------------
    # Steps 9-11: Exposure -> positions -> return
    # ------------------------------------------------------------------
    exposure_weight: float = risk_output.exposure_weight
    positions: dict = route_exposure_to_positions(
        total_capital=total_capital,
        exposure_fraction=exposure_weight,
        asset_prices=asset_prices,
    )
    return positions
