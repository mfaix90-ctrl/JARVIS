# jarvis/verification/data_models/input_vector.py
# InputVector data class for the Deterministic Verification Harness.
# Fields conform to Section 8.1 of DVH Implementation Blueprint v1.0.0.

from dataclasses import dataclass
from typing import Optional

from jarvis.core.regime import GlobalRegimeState, CorrelationRegimeState


@dataclass(frozen=True)
class InputVector:
    """
    A single immutable input vector for the Deterministic Verification Harness.

    Each InputVector corresponds to one invocation of RiskEngine.assess().
    The matrix is version-controlled and fully deterministic (Section 7).

    Fields:
      vector_id          -- Unique identifier for this vector (e.g., "VOL-01").
      group_id           -- Group to which this vector belongs
                            (G-VOL, G-DD, G-MU, G-RP, G-JM, G-CR, G-BC, G-EX).
      returns_history    -- Returns tuple passed to assess().
      current_regime_str -- String value of GlobalRegimeState (e.g., "RISK_ON").
                            Stored as string for serialization; converted to
                            GlobalRegimeState instance before assess() invocation.
      meta_uncertainty   -- meta_uncertainty float passed to assess().
      macro_regime       -- Optional GlobalRegimeState instance for JRM.
                            Must be a canonical enum from jarvis.core.regime.
                            None when JRM is inactive for this vector.
      correlation_regime -- Optional CorrelationRegimeState instance for JRM.
                            Must be a canonical enum from jarvis.core.regime.
                            None when JRM is inactive for this vector.
      realized_vol       -- Optional float for vol scaling.
      target_vol         -- Optional float for vol scaling.
      regime_posterior   -- Optional float for posterior confidence.
      expect_exception   -- If True, assess() is expected to raise ValueError.
      description        -- Human-readable description of what this vector tests.
    """
    vector_id:          str
    group_id:           str
    returns_history:    tuple                          # tuple of float, immutable
    current_regime_str: str
    meta_uncertainty:   float
    macro_regime:       Optional[GlobalRegimeState]    # canonical enum or None
    correlation_regime: Optional[CorrelationRegimeState]  # canonical enum or None
    realized_vol:       Optional[float]
    target_vol:         Optional[float]
    regime_posterior:   Optional[float]
    expect_exception:   bool
    description:        str
