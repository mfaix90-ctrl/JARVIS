# jarvis/verification/data_models/execution_record.py
# ExecutionRecord and ObservedOutput data classes.
# Conform to Section 8 of DVH Implementation Blueprint v1.0.0.

from dataclasses import dataclass
from typing import Optional

from jarvis.verification.data_models.input_vector import InputVector


@dataclass(frozen=True)
class ObservedOutput:
    """
    Observed output from one invocation of RiskEngine.assess().
    All nine fields are mandatory (BIC-C-01).

    Fields correspond to the seven RiskOutput fields plus exception metadata.
    When exception_raised is True, all float fields are set to 0.0 and
    string fields to empty string. Boolean fields are False.

    Nine fields (BIC-C-01):
      expected_drawdown       -- RiskOutput.expected_drawdown
      expected_drawdown_p95   -- RiskOutput.expected_drawdown_p95
      volatility_forecast     -- RiskOutput.volatility_forecast
      risk_compression_active -- RiskOutput.risk_compression_active
      position_size_factor    -- RiskOutput.position_size_factor
      exposure_weight         -- RiskOutput.exposure_weight
      risk_regime             -- RiskOutput.risk_regime
      exception_raised        -- True if assess() raised an exception.
      exception_type          -- Exception type name if raised; empty string otherwise.
    """
    expected_drawdown:       float
    expected_drawdown_p95:   float
    volatility_forecast:     float
    risk_compression_active: bool
    position_size_factor:    float
    exposure_weight:         float
    risk_regime:             str
    exception_raised:        bool
    exception_type:          str


@dataclass(frozen=True)
class ExecutionRecord:
    """
    Complete record for one input vector execution in the harness pipeline.
    Produced by ExecutionRecorder and ReplayEngine.
    All fields are mandatory (Section 8.3).

    Fields:
      vector_id         -- Matches InputVector.vector_id.
      group_id          -- Matches InputVector.group_id.
      input_vector      -- The InputVector that produced this record.
      observed_output   -- The ObservedOutput from the production module.
      stage             -- "ER" for initial execution; "RE" for replay.
      manifest_hash     -- Manifest hash observed by ManifestValidator at run start.
      timestamp_iso     -- UTC ISO-8601 timestamp at record creation (audit only).
      execution_id      -- UUID4 unique identifier for this record (traceability).
      harness_version   -- HARNESS_VERSION constant at time of execution.
      module_version    -- --module-version CLI argument for this run.
    """
    vector_id:       str
    group_id:        str
    input_vector:    InputVector
    observed_output: ObservedOutput
    stage:           str
    manifest_hash:   str
    timestamp_iso:   str
    execution_id:    str
    harness_version: str
    module_version:  str
