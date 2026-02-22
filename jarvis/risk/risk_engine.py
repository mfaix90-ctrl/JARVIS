# jarvis/risk/risk_engine.py
# Version: 6.1.0
# Formal Architecture Specification: Risk Engine FAS v6.1.0 -- FREEZE
# Authority: Risk Engine FAS v6.1.0 | Master FAS v6.1.0-G
#
# Module Boundary: This file is the sole implementation of the RiskEngine.
# Constants are defined in jarvis/utils/constants.py.
# Regime types are defined in jarvis/core/regime.py.
# No other module is in scope of this specification.
#
# DETERMINISM GUARANTEE (DET-01 through DET-07):
#   DET-01  No stochastic operations. No random number generation. No sampling.
#   DET-02  No external state reads inside assess(). All inputs passed explicitly.
#   DET-03  No side effects. assess() does not write to any external state.
#   DET-04  All arithmetic operations are deterministic floating-point.
#   DET-05  All conditional branches are deterministic functions of explicit inputs.
#   DET-06  vol_adjustment cap of 3.0 and CRISIS dampening factor of 0.75 are
#           fixed literals. Neither is parameterised.
#   DET-07  Backward compatibility is deterministic: for any call site that passes
#           no new optional parameters, output is bit-identical to prior version.
#
# CANONICAL EXPOSURE EQUATION (Section 3):
#   capital_base        = position_size * posterior_confidence
#   vol_efficiency      = vol_adjustment
#   uncertainty_penalty = 1 - meta_uncertainty
#   regime_budget       = 1.0
#   E_pre_clip          = capital_base * vol_efficiency * uncertainty_penalty * regime_budget
#
# CLIP CHAIN ORDER (INV-07): Clip A -> Clip B -> Clip C -> CRISIS dampening
#
# Standard import:
#   from jarvis.risk.risk_engine import RiskEngine, RiskOutput

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

from jarvis.core.regime import GlobalRegimeState, CorrelationRegimeState
from jarvis.utils.constants import (
    JOINT_RISK_MULTIPLIER_TABLE,
    MAX_DRAWDOWN_THRESHOLD,
    VOL_COMPRESSION_TRIGGER,
    SHOCK_EXPOSURE_CAP,
)


# ---------------------------------------------------------------------------
# OUTPUT DATA CLASS
# ---------------------------------------------------------------------------

@dataclass
class RiskOutput:
    """
    Complete Risk Assessment Output for FAS v6.1.0.

    Fields:
      expected_drawdown       -- Mean drawdown over returns history window (0-1).
      expected_drawdown_p95   -- 95th-percentile drawdown (0-1).
      volatility_forecast     -- EWMA annualised volatility forecast.
      risk_compression_active -- True when Risk Compression Mode is active.
                                 Determined solely from vol, dd_result, and
                                 current_regime (INV-10: independent of all
                                 optional parameters).
      position_size_factor    -- Adaptive position size from
                                 compute_adaptive_position_size(), post-Clip-A.
                                 This is the raw pre-equation adaptive sizing
                                 result. It is NOT the final exposure_weight
                                 (INV-09).
      exposure_weight         -- Final confidence-weighted exposure after the
                                 full clip chain (Clip B, Clip C, CRISIS
                                 dampening). Range: nominally [1e-6, 1.0];
                                 may fall below SHOCK_EXPOSURE_CAP after CRISIS
                                 dampening (specified behaviour, not a defect).
      risk_regime             -- One of: NORMAL, ELEVATED, CRITICAL, DEFENSIVE.
                                 Determined solely from vol and dd_result
                                 (INV-10: independent of all optional parameters).
    """
    expected_drawdown:       float
    expected_drawdown_p95:   float
    volatility_forecast:     float
    risk_compression_active: bool
    position_size_factor:    float
    exposure_weight:         float
    risk_regime:             str


# ---------------------------------------------------------------------------
# RISK ENGINE
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Central Risk Engine for the JARVIS Decision Quality Platform.
    Governed by Risk Engine FAS v6.1.0 (FREEZE).

    All computations are deterministic. No stochastic operations.
    assess() has no side effects and reads no external state.

    Immutable Risk Thresholds (hash-protected in THRESHOLD_MANIFEST.json):
      MAX_DRAWDOWN_THRESHOLD  = 0.15   (15%  -- Hard Limit)
      VOL_COMPRESSION_TRIGGER = 0.30   (30% ann. Vol -> Risk Compression)
      SHOCK_EXPOSURE_CAP      = 0.25   (Max 25% Exposure; Clip C floor)

    These class attributes mirror the module-level constants for direct access.
    The module-level constants in constants.py are the canonical source.
    """

    # Mirrors of module-level constants for class-level access.
    MAX_DRAWDOWN_THRESHOLD:  float = MAX_DRAWDOWN_THRESHOLD
    VOL_COMPRESSION_TRIGGER: float = VOL_COMPRESSION_TRIGGER
    SHOCK_EXPOSURE_CAP:      float = SHOCK_EXPOSURE_CAP

    # Fixed literals per FAS v6.1.0 DET-06. Not parameterised.
    _VOL_ADJUSTMENT_CAP: float = 3.0
    _CRISIS_DAMPENING:   float = 0.75
    _CLIP_B_FLOOR:       float = 1e-6

    # ---------------------------------------------------------------------------
    # COMPONENT: EXPECTED DRAWDOWN
    # ---------------------------------------------------------------------------

    def compute_expected_drawdown(
        self,
        returns_history: List[float],
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Expected Drawdown Forecast via historical simulation.
        Returns dict with keys "mean" and "p95".

        Raises ValueError if len(returns_history) < 20.
        Raises ValueError if returns_history contains NaN or Inf values.
        """
        if len(returns_history) < 20:
            raise ValueError(
                "Minimum 20 returns required for drawdown estimation. "
                f"Received: {len(returns_history)}"
            )

        arr = np.array(returns_history, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "returns_history contains NaN or Inf values. "
                "All inputs must be finite."
            )

        cum         = np.cumprod(1.0 + arr)
        running_max = np.maximum.accumulate(cum)
        drawdowns   = (running_max - cum) / np.maximum(running_max, 1e-10)

        expected_dd = float(np.mean(drawdowns))
        dd_p95      = float(np.percentile(drawdowns, confidence_level * 100))

        return {"mean": expected_dd, "p95": dd_p95}

    # ---------------------------------------------------------------------------
    # COMPONENT: VOLATILITY FORECAST
    # ---------------------------------------------------------------------------

    def forecast_volatility(
        self,
        returns: List[float],
        halflife_days: int = 20,
    ) -> float:
        """
        EWMA Volatility Forecast (annualised).
        Decay factor = exp(-log(2) / halflife_days).

        Raises ValueError if len(returns) < 5.
        Raises ValueError if returns contains NaN or Inf.
        """
        if len(returns) < 5:
            raise ValueError(
                "Minimum 5 data points required for volatility forecast. "
                f"Received: {len(returns)}"
            )

        arr = np.array(returns, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "returns contains NaN or Inf values. All inputs must be finite."
            )

        decay   = float(np.exp(-np.log(2.0) / max(halflife_days, 1)))
        weights = np.array([decay ** i for i in range(len(arr) - 1, -1, -1)])
        weights = weights / weights.sum()

        ewma_var   = float(np.sum(weights * arr ** 2))
        vol_annual = float(np.sqrt(max(ewma_var, 1e-10) * 252))
        return vol_annual

    # ---------------------------------------------------------------------------
    # COMPONENT: ADAPTIVE POSITION SIZE -- contains Clip A
    # ---------------------------------------------------------------------------

    def compute_adaptive_position_size(
        self,
        base_size:        float,
        volatility:       float,
        drawdown_p95:     float,
        regime:           str,
        meta_uncertainty: float,
    ) -> float:
        """
        Adaptive Position Sizing.

        Clip A (INV-01): bounds output to [0.0, 1.0].
        Returns a value in [0.0, 1.0].
        """
        vol_factor = float(np.clip(0.2 / max(volatility, 1e-6), 0.0, 1.0))
        dd_factor  = float(np.clip(
            1.0 - drawdown_p95 / max(self.MAX_DRAWDOWN_THRESHOLD, 1e-10),
            0.0, 1.0,
        ))
        unc_factor = float(np.clip(1.0 - meta_uncertainty, 0.0, 1.0))

        regime_cap = {
            "NORMAL":    1.0,
            "ELEVATED":  0.7,
            "CRITICAL":  0.4,
            "DEFENSIVE": 0.2,
        }.get(regime, 0.3)

        raw_factor = vol_factor * dd_factor * unc_factor * regime_cap

        # Clip A: bounds output to [0.0, 1.0] -- INV-01.
        return float(np.clip(raw_factor * base_size, 0.0, 1.0))

    # ---------------------------------------------------------------------------
    # PRIMARY ENTRY POINT: ASSESS
    # ---------------------------------------------------------------------------

    def assess(
        self,
        returns_history:    List[float],
        current_regime:     GlobalRegimeState,
        meta_uncertainty:   float,
        macro_regime:       Optional[GlobalRegimeState] = None,
        correlation_regime: Optional[CorrelationRegimeState] = None,
        realized_vol:       Optional[float] = None,
        target_vol:         Optional[float] = None,
        regime_posterior:   Optional[float] = None,
    ) -> RiskOutput:
        """
        Full Risk Assessment per FAS v6.1.0.

        Canonical Exposure Equation (Section 3):
          capital_base        = position_size * posterior_confidence
          vol_efficiency      = vol_adjustment
          uncertainty_penalty = 1 - meta_uncertainty
          regime_budget       = 1.0  (INV-05)
          E_pre_clip          = capital_base * vol_efficiency
                                * uncertainty_penalty * regime_budget

        Clip Chain (INV-07): Clip A -> Clip B -> Clip C -> CRISIS dampening

        Parameters:
          returns_history    -- List of per-period returns. Minimum length 20.
          current_regime     -- GlobalRegimeState. Used for risk_compression,
                                risk_regime determination, and CRISIS dampening.
          meta_uncertainty   -- Meta-uncertainty in [0, 1].
          macro_regime       -- Optional GlobalRegimeState for JRM macro context.
                                If None, JRM defaults to 1.0 (INV-03, INV-08).
                                Must be a GlobalRegimeState instance from
                                jarvis.core.regime -- no plain strings permitted.
          correlation_regime -- Optional CorrelationRegimeState for JRM.
                                If None, JRM defaults to 1.0 (INV-03, INV-08).
                                Must be a CorrelationRegimeState instance from
                                jarvis.core.regime -- no plain strings permitted.
          realized_vol       -- Optional. If None or if target_vol is None,
                                vol_adjustment defaults to 1.0 (INV-08).
          target_vol         -- Optional. If None or if realized_vol is None,
                                vol_adjustment defaults to 1.0 (INV-08).
          regime_posterior   -- Optional float in [0, 1]. Clipped silently (INV-11).
                                If None, posterior_confidence defaults to 1.0 (INV-06).

        Returns RiskOutput with all fields populated.

        INV-10 GUARANTEE:
          risk_compression_active and risk_regime are determined exclusively from
          vol, dd_result, and current_regime. No optional parameter affects them.

        INV-08 GUARANTEE:
          When all optional parameters are None, output is bit-identical to the
          prior version for the same required inputs.
        """
        # ------------------------------------------------------------------
        # Step 1: Upstream computations (independent of optional parameters).
        # ------------------------------------------------------------------
        dd_result = self.compute_expected_drawdown(returns_history)
        vol       = self.forecast_volatility(returns_history)

        # ------------------------------------------------------------------
        # Step 2: Risk Compression and Risk Regime (INV-10).
        # ------------------------------------------------------------------
        risk_compression = (
            vol > self.VOL_COMPRESSION_TRIGGER
            or dd_result["p95"] > self.MAX_DRAWDOWN_THRESHOLD
            or current_regime == GlobalRegimeState.CRISIS
        )

        if (current_regime == GlobalRegimeState.CRISIS
                or dd_result["p95"] > self.MAX_DRAWDOWN_THRESHOLD):
            risk_regime = "DEFENSIVE"
        elif vol > self.VOL_COMPRESSION_TRIGGER:
            risk_regime = "CRITICAL"
        elif vol > self.VOL_COMPRESSION_TRIGGER * 0.7:
            risk_regime = "ELEVATED"
        else:
            risk_regime = "NORMAL"

        # ------------------------------------------------------------------
        # Step 3: Adaptive position size -- contains Clip A (INV-01).
        # ------------------------------------------------------------------
        base_exposure = self.SHOCK_EXPOSURE_CAP if risk_compression else 1.0
        position_size = self.compute_adaptive_position_size(
            base_size=base_exposure,
            volatility=vol,
            drawdown_p95=dd_result["p95"],
            regime=risk_regime,
            meta_uncertainty=meta_uncertainty,
        )

        # ------------------------------------------------------------------
        # Step 4: Optional-parameter components (INV-08).
        # ------------------------------------------------------------------

        # posterior_confidence (INV-06):
        if regime_posterior is not None:
            posterior_confidence = float(np.clip(regime_posterior, 0.0, 1.0))
        else:
            posterior_confidence = 1.0

        # vol_adjustment (INV-08):
        if realized_vol is not None and target_vol is not None:
            raw_adj        = float(target_vol) / max(float(realized_vol), 1e-10)
            vol_adjustment = float(min(raw_adj, self._VOL_ADJUSTMENT_CAP))
            vol_adjustment = float(max(vol_adjustment, 0.0))
        else:
            vol_adjustment = 1.0

        # ------------------------------------------------------------------
        # Step 5: Canonical Exposure Equation (Section 3.1).
        # ------------------------------------------------------------------
        capital_base        = position_size * posterior_confidence
        vol_efficiency      = vol_adjustment
        uncertainty_penalty = 1.0 - meta_uncertainty
        regime_budget       = 1.0   # INV-05: 1.0 in v6.1.0

        E_pre_clip = (
            capital_base
            * vol_efficiency
            * uncertainty_penalty
            * regime_budget
        )

        # ------------------------------------------------------------------
        # Step 6: Clip B -- always applied, unconditional (INV-02).
        # ------------------------------------------------------------------
        exposure_weight = float(np.clip(E_pre_clip, self._CLIP_B_FLOOR, 1.0))

        # ------------------------------------------------------------------
        # Step 7: Joint Macro x Correlation Risk Multiplier (JRM).
        # macro_regime and correlation_regime must be canonical enum instances.
        # joint_multiplier = 1.0 iff either is None (INV-03).
        # ------------------------------------------------------------------
        if macro_regime is not None and correlation_regime is not None:
            joint_multiplier = float(
                JOINT_RISK_MULTIPLIER_TABLE[macro_regime][correlation_regime]
            )
        else:
            joint_multiplier = 1.0

        # Clip C (conditional on JRM active -- INV-03):
        if joint_multiplier != 1.0:
            exposure_weight = float(np.clip(
                exposure_weight / joint_multiplier,
                self.SHOCK_EXPOSURE_CAP,
                1.0,
            ))

        # ------------------------------------------------------------------
        # Step 8: CRISIS Dampening -- applied after Clip C (INV-04).
        # ------------------------------------------------------------------
        if current_regime == GlobalRegimeState.CRISIS:
            exposure_weight = exposure_weight * self._CRISIS_DAMPENING

        # ------------------------------------------------------------------
        # Step 9: Assemble and return output.
        # ------------------------------------------------------------------
        return RiskOutput(
            expected_drawdown=dd_result["mean"],
            expected_drawdown_p95=dd_result["p95"],
            volatility_forecast=vol,
            risk_compression_active=risk_compression,
            position_size_factor=position_size,
            exposure_weight=exposure_weight,
            risk_regime=risk_regime,
        )
