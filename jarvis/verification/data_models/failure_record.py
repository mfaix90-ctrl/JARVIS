# jarvis/verification/data_models/failure_record.py
# FailureRecord data class and failure type registry.
# Conforms to Section 13 of DVH Implementation Blueprint v1.0.0.

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# FAILURE TYPE REGISTRY (Section 13.2)
# ---------------------------------------------------------------------------
# Exit code mapping (Section 14.1 HFP-01):
#   Code 1 -- FREEZE_VIOLATION, DETERMINISM_BREACH
#   Code 2 -- INTEGRITY_FAILURE (MANIFEST_HASH_MISMATCH, MANIFEST_CONSTANT_MISMATCH,
#                                 MANIFEST_TABLEHASH_PRESENT)
#   Code 3 -- DATA_CORRUPTION, CONTRACT_VIOLATION
#   Code 4 -- Internal harness errors

FAILURE_TYPES = {
    # Exit Code 1
    "FREEZE_VIOLATION":          1,
    "DETERMINISM_BREACH":        1,
    "CLIP_A_VIOLATION":          1,    # INV-01 (VFE-05)
    "CLIP_B_FLOOR_VIOLATION":    1,    # INV-02 (VFE-05)
    "CLIP_B_RANGE_VIOLATION":    1,    # INV-02 (VFE-05)
    "CLIP_C_FLOOR_VIOLATION":    1,    # INV-03 (VFE-05)
    "CRISIS_ORDERING_VIOLATION": 1,    # INV-04 (VFE-05)
    "BACKWARD_COMPAT_VIOLATION": 1,    # INV-08 (VFE-05)
    "FIELD_NAN":                 1,    # BIC-S-02
    "FIELD_INFINITE":            1,    # BIC-S-03
    "FIELD_BOOLEAN_MISMATCH":    1,    # BIC-B-01
    # Exit Code 2
    "MANIFEST_HASH_MISMATCH":    2,    # MVI-02
    "MANIFEST_CONSTANT_MISMATCH":2,    # MVI-04
    "MANIFEST_TABLEHASH_PRESENT":2,    # MVI-05
    "INTEGRITY_FAILURE":         2,
    # Exit Code 3
    "DATA_CORRUPTION":           3,    # RSF-03, RSF-07
    "CONTRACT_VIOLATION":        3,    # DIM-04
    "EXCEPTION_TYPE_MISMATCH":   3,    # EX-01/EX-02 wrong exception
    "MISSING_EXPECTED_EXCEPTION": 3,   # EX-01/EX-02 no exception raised
    # Exit Code 4
    "HARNESS_INTERNAL_ERROR":    4,
}


@dataclass
class FailureRecord:
    """
    Failure record written to disk by the Failure Handler on any hard failure.
    Conforms to Section 13 of DVH Implementation Blueprint v1.0.0.

    All fields are mandatory. Written as JSON to the runs directory.
    The record is write-once and immutable after the run completes.

    Fields:
      failure_type_id      -- Key from FAILURE_TYPES registry.
      exit_code            -- Integer exit code (1-4).
      vector_id            -- Vector ID that triggered the failure. Empty if not applicable.
      field_name           -- Field involved in the failure. Empty if not applicable.
      detected_at_iso      -- UTC ISO-8601 timestamp of failure detection.
      run_id               -- Run identifier for this harness invocation.
      harness_version      -- HARNESS_VERSION at time of failure.
      module_version       -- --module-version argument for this run.
      manifest_hash        -- Manifest hash observed at run start.
      freeze_invariant_ref -- INV-XX reference if applicable. Empty otherwise (VFE-05).
      detail               -- Human-readable failure description.
    """
    failure_type_id:      str
    exit_code:            int
    vector_id:            str
    field_name:           str
    detected_at_iso:      str
    run_id:               str
    harness_version:      str
    module_version:       str
    manifest_hash:        str
    freeze_invariant_ref: str
    detail:               str
