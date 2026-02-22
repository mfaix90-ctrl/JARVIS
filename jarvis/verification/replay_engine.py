# jarvis/verification/replay_engine.py
# ReplayEngine -- re-executes all vectors to produce the RE-stage records.
# Authority: DVH Implementation Blueprint v1.0.0 Sections 3 and 7.
#
# The ReplayEngine is structurally identical to the ExecutionRecorder but
# produces records with stage="RE". It uses the same InputVector list and
# the same production RiskEngine. No state is shared between ER and RE passes.
# OC-05: Production results are NOT cached or reused between ER and RE stages.

from typing import List

from jarvis.verification.data_models.input_vector import InputVector
from jarvis.verification.data_models.execution_record import ExecutionRecord
from jarvis.verification.execution_recorder import ExecutionRecorder


class ReplayEngine:
    """
    Re-executes all input vectors through the production Risk Engine.
    Produces RE-stage ExecutionRecords for comparison against ER-stage records.

    The ReplayEngine re-invokes the production Risk Engine fresh for every vector.
    No ER results are reused (OC-05). The production module is invoked with
    identical inputs and must produce identical outputs (DET-P-01).

    In cross-session replay mode (DIM-05/DIM-06), the ReplayEngine still executes
    all vectors normally. The loaded prior records substitute for ER records in
    the BIC stage -- the RE stage is not skipped.
    """

    def __init__(
        self,
        manifest_hash:  str,
        module_version: str,
        run_id:         str,
    ):
        self._recorder = ExecutionRecorder(
            manifest_hash=manifest_hash,
            module_version=module_version,
            run_id=run_id,
            stage="RE",
        )

    def replay_all(self, vectors: List[InputVector]) -> List[ExecutionRecord]:
        """
        Re-execute all vectors and return RE-stage ExecutionRecords.
        Execution order is identical to the ER stage (same vector list).
        """
        return self._recorder.record_all(vectors)
