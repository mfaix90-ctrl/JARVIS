# jarvis/verification/execution_recorder.py
# ExecutionRecorder -- invokes the production Risk Engine and records results.
# Authority: DVH Implementation Blueprint v1.0.0 Section 8.
#
# NIC-06: No mock, stub, or test double is injected into the production path.
# NIC-09: No introspection or bytecode manipulation is used.
# NIC-10: Exceptions from the production module propagate naturally and are
#         observed through standard exception handling at the call site.
# EEP-01: Single-threaded. Each invocation completes before the next begins.
# EEP-02: Production Risk Engine is invoked synchronously.

import uuid
from datetime import datetime, timezone
from typing import List, Dict

from jarvis.risk.risk_engine import RiskEngine
from jarvis.core.regime import GlobalRegimeState
from jarvis.verification.data_models.input_vector import InputVector
from jarvis.verification.data_models.execution_record import ExecutionRecord, ObservedOutput
from jarvis.verification.harness_version import HARNESS_VERSION


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string. For audit timestamps only."""
    return datetime.now(timezone.utc).isoformat()


def _regime_from_string(regime_str: str) -> GlobalRegimeState:
    """Map regime string to GlobalRegimeState. Raises ValueError on unknown string."""
    try:
        return GlobalRegimeState(regime_str)
    except ValueError:
        raise ValueError(
            f"CONTRACT_VIOLATION: Unknown GlobalRegimeState string: '{regime_str}'"
        )


class ExecutionRecorder:
    """
    Invokes the production RiskEngine for each InputVector and records results.

    For each vector:
      - If vector.expect_exception is False: invokes assess(), records output.
      - If vector.expect_exception is True: invokes assess(), expects ValueError.
        If ValueError is raised: records exception_raised=True.
        If no exception or wrong exception: raises RuntimeError for FailureHandler.

    The production module is not wrapped, decorated, or instrumented (NIC-01).
    All inputs to RiskEngine are of production types and values (NIC-02).
    """

    def __init__(
        self,
        manifest_hash:  str,
        module_version: str,
        run_id:         str,
        stage:          str = "ER",
    ):
        self._manifest_hash  = manifest_hash
        self._module_version = module_version
        self._run_id         = run_id
        self._stage          = stage
        self._engine         = RiskEngine()

    def _invoke(self, vector: InputVector) -> ObservedOutput:
        """
        Invoke production RiskEngine.assess() with the given InputVector.
        Returns an ObservedOutput.

        Exceptions from the production module are caught ONLY to record them.
        They are not suppressed (NIC-10). For unexpected exceptions, the caller
        receives a re-raised RuntimeError.
        """
        regime = _regime_from_string(vector.current_regime_str)

        try:
            result = self._engine.assess(
                returns_history=list(vector.returns_history),
                current_regime=regime,
                meta_uncertainty=vector.meta_uncertainty,
                macro_regime=vector.macro_regime,
                correlation_regime=vector.correlation_regime,
                realized_vol=vector.realized_vol,
                target_vol=vector.target_vol,
                regime_posterior=vector.regime_posterior,
            )

            if vector.expect_exception:
                # Production should have raised; it did not.
                raise RuntimeError(
                    f"MISSING_EXPECTED_EXCEPTION: Vector {vector.vector_id} "
                    "expected ValueError but production module returned normally."
                )

            return ObservedOutput(
                expected_drawdown=result.expected_drawdown,
                expected_drawdown_p95=result.expected_drawdown_p95,
                volatility_forecast=result.volatility_forecast,
                risk_compression_active=result.risk_compression_active,
                position_size_factor=result.position_size_factor,
                exposure_weight=result.exposure_weight,
                risk_regime=result.risk_regime,
                exception_raised=False,
                exception_type="",
            )

        except RuntimeError:
            # Re-raise harness-generated RuntimeErrors (not from production).
            raise

        except ValueError as exc:
            if not vector.expect_exception:
                # Unexpected exception from production.
                raise RuntimeError(
                    f"EXCEPTION_TYPE_MISMATCH: Vector {vector.vector_id} raised "
                    f"unexpected ValueError: {exc}"
                ) from exc
            # Expected ValueError. Record the exception outcome.
            return ObservedOutput(
                expected_drawdown=0.0,
                expected_drawdown_p95=0.0,
                volatility_forecast=0.0,
                risk_compression_active=False,
                position_size_factor=0.0,
                exposure_weight=0.0,
                risk_regime="",
                exception_raised=True,
                exception_type="ValueError",
            )

        except Exception as exc:
            # Production raised an unexpected exception type.
            exc_name = type(exc).__name__
            if vector.expect_exception:
                raise RuntimeError(
                    f"EXCEPTION_TYPE_MISMATCH: Vector {vector.vector_id} expected "
                    f"ValueError but got {exc_name}: {exc}"
                ) from exc
            raise RuntimeError(
                f"EXCEPTION_TYPE_MISMATCH: Vector {vector.vector_id} raised "
                f"unexpected {exc_name}: {exc}"
            ) from exc

    def record_all(self, vectors: List[InputVector]) -> List[ExecutionRecord]:
        """
        Execute all vectors in order. Return a list of ExecutionRecord.
        Single-threaded. Each invocation completes before the next begins (EEP-01).
        """
        records = []
        for vector in vectors:
            observed = self._invoke(vector)
            record = ExecutionRecord(
                vector_id=vector.vector_id,
                group_id=vector.group_id,
                input_vector=vector,
                observed_output=observed,
                stage=self._stage,
                manifest_hash=self._manifest_hash,
                timestamp_iso=_now_iso(),
                execution_id=str(uuid.uuid4()),
                harness_version=HARNESS_VERSION,
                module_version=self._module_version,
            )
            records.append(record)
        return records
