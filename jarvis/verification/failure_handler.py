# jarvis/verification/failure_handler.py
# FailureHandler -- hard failure policy enforcement for the harness.
# Authority: DVH Implementation Blueprint v1.0.0 Section 14.
#
# HFP-01: Exit with non-zero exit code on any hard failure.
# HFP-03: No WARNING-level logging. All failures reported via record and exit code.
# HFP-04: No catch-and-continue. No retry. No fallback.
# HFP-06: Invoke immediately on failure detection. No further pipeline stage runs.
# HFP-07: If FailureHandler itself raises, write partial record to stderr and exit 4.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jarvis.verification.data_models.failure_record import FailureRecord, FAILURE_TYPES
from jarvis.verification.harness_version import HARNESS_VERSION

# Freeze invariant reference map (VFE-05).
_FREEZE_INV_REF = {
    "CLIP_A_VIOLATION":          "INV-01",
    "CLIP_B_FLOOR_VIOLATION":    "INV-02",
    "CLIP_B_RANGE_VIOLATION":    "INV-02",
    "CLIP_C_FLOOR_VIOLATION":    "INV-03",
    "CRISIS_ORDERING_VIOLATION": "INV-04",
    "BACKWARD_COMPAT_VIOLATION": "INV-08",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FailureHandler:
    """
    Enforces the Hard Failure Policy (Section 14).

    On any hard failure:
      1. Construct FailureRecord.
      2. Write FailureRecord JSON to runs directory.
      3. Print failure summary to stdout (HFP-05).
      4. Call sys.exit(exit_code) -- must be last operation (HFP-02).

    If FailureHandler itself raises during record writing:
      - Write partial info to stderr.
      - sys.exit(4).
    """

    def __init__(
        self,
        runs_dir:       Path,
        run_id:         str,
        module_version: str,
        manifest_hash:  str,
    ):
        self._runs_dir      = runs_dir
        self._run_id        = run_id
        self._module_version = module_version
        self._manifest_hash = manifest_hash

    def handle(
        self,
        failure_type_id: str,
        detail:          str,
        vector_id:       str = "",
        field_name:      str = "",
    ) -> None:
        """
        Execute the hard failure policy. This method does not return.

        Parses failure_type_id from RuntimeError messages if the RuntimeError
        starts with a known failure_type_id prefix.
        """
        exit_code          = FAILURE_TYPES.get(failure_type_id, 4)
        freeze_inv_ref     = _FREEZE_INV_REF.get(failure_type_id, "")
        detected_at        = _now_iso()

        record = FailureRecord(
            failure_type_id=failure_type_id,
            exit_code=exit_code,
            vector_id=vector_id,
            field_name=field_name,
            detected_at_iso=detected_at,
            run_id=self._run_id,
            harness_version=HARNESS_VERSION,
            module_version=self._module_version,
            manifest_hash=self._manifest_hash,
            freeze_invariant_ref=freeze_inv_ref,
            detail=detail,
        )

        record_dict = {
            "failure_type_id":      record.failure_type_id,
            "exit_code":            record.exit_code,
            "vector_id":            record.vector_id,
            "field_name":           record.field_name,
            "detected_at_iso":      record.detected_at_iso,
            "run_id":               record.run_id,
            "harness_version":      record.harness_version,
            "module_version":       record.module_version,
            "manifest_hash":        record.manifest_hash,
            "freeze_invariant_ref": record.freeze_invariant_ref,
            "detail":               record.detail,
        }

        try:
            self._runs_dir.mkdir(parents=True, exist_ok=True)
            ts_compact = detected_at.replace(":", "").replace("-", "").replace("+", "Z")[:16]
            filename   = f"{self._run_id}_FAIL_{ts_compact}.json"
            filepath   = self._runs_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record_dict, f, indent=4)

            # HFP-05: Stdout contains only the pass/fail summary.
            print(
                f"HARNESS RESULT: FAIL\n"
                f"Failure type:   {failure_type_id}\n"
                f"Exit code:      {exit_code}\n"
                f"Vector:         {vector_id or '(not applicable)'}\n"
                f"Detail:         {detail[:200]}\n"
                f"Record written: {filepath}"
            )

        except Exception as exc:
            # HFP-07: Write partial info to stderr and exit 4.
            sys.stderr.write(
                f"HARNESS_INTERNAL_ERROR: FailureHandler failed to write record: {exc}\n"
                f"Original failure: {failure_type_id} -- {detail}\n"
            )
            sys.exit(4)

        # HFP-02: sys.exit is the last operation.
        sys.exit(exit_code)

    def handle_from_exception(
        self,
        exc:       RuntimeError,
        vector_id: str = "",
    ) -> None:
        """
        Parse failure_type_id from RuntimeError message and invoke handle().

        Convention: RuntimeError messages from this harness start with
        FAILURE_TYPE_ID: detail
        """
        msg = str(exc)
        failure_type_id = "HARNESS_INTERNAL_ERROR"
        for known_type in FAILURE_TYPES:
            if msg.startswith(known_type + ":") or msg.startswith(known_type + " "):
                failure_type_id = known_type
                break
        self.handle(
            failure_type_id=failure_type_id,
            detail=msg,
            vector_id=vector_id,
        )
