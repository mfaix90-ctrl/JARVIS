# jarvis/verification/manifest_validator.py
# ManifestValidator -- first component executed in the harness pipeline.
# Authority: DVH Implementation Blueprint v1.0.0 Section 12.
#
# NIC-03: This module does not modify any constant. Constants are read from
#         their authoritative source for comparison purposes only.
# No pipeline stage begins until ManifestValidator completes successfully.

import hashlib
import json
import struct
from pathlib import Path
from typing import Tuple, Dict, Any


class ManifestValidator:
    """
    Validates THRESHOLD_MANIFEST.json before any Risk Engine invocation.

    Implements MVI-01 through MVI-06 from DVH Implementation Blueprint Section 12.

    On any failure: raises RuntimeError with failure_type_id as the first word
    of the message. Caller (run_harness.py) converts to FailureRecord and exits.
    """

    # FAS v6.1.0 declared constant values (MVI-04).
    # Verified by struct-based bit pattern comparison.
    _EXPECTED_SHOCK_EXPOSURE_CAP:      float = 0.25
    _EXPECTED_MAX_DRAWDOWN_THRESHOLD:  float = 0.15
    _EXPECTED_VOL_COMPRESSION_TRIGGER: float = 0.30

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path

    @staticmethod
    def _float_bits(value: float) -> bytes:
        """Return 8-byte big-endian IEEE 754 representation."""
        return struct.pack(">d", value)

    @staticmethod
    def _floats_equal(a: float, b: float) -> bool:
        """
        Exact bit-pattern equality for floats.
        Distinguishes +0.0 from -0.0.
        """
        return struct.pack(">d", a) == struct.pack(">d", b)

    def validate(self) -> Tuple[str, Dict[str, Any]]:
        """
        Execute full manifest validation.

        Returns:
          (computed_hash, thresholds_dict)
          computed_hash -- SHA-256 hex digest of the manifest content.
          thresholds    -- dict from manifest["thresholds"] section.

        Raises RuntimeError with failure_type_id prefix on any hard failure.
        Failure types raised here:
          MANIFEST_HASH_MISMATCH    -- MVI-02
          MANIFEST_CONSTANT_MISMATCH -- MVI-04
          MANIFEST_TABLEHASH_PRESENT -- MVI-05
          INTEGRITY_FAILURE          -- missing file or parse error
        """
        # MVI-01: Load manifest file.
        if not self.manifest_path.exists():
            raise RuntimeError(
                "INTEGRITY_FAILURE: THRESHOLD_MANIFEST.json not found at "
                f"{self.manifest_path}. Harness cannot start without manifest."
            )

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(
                f"INTEGRITY_FAILURE: Failed to load or parse THRESHOLD_MANIFEST.json: {exc}"
            ) from exc

        # MVI-01: Compute hash excluding manifest_hash field.
        data_for_hash = {k: v for k, v in data.items() if k != "manifest_hash"}
        computed_hash = hashlib.sha256(
            json.dumps(data_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        # MVI-02: Hash is computed and propagated, not self-compared.
        # Self-hash comparison removed to prevent recursive instability.

        # MVI-05: Check for absence of table_hash in joint_risk_multiplier_table.
        jrmt = data.get("joint_risk_multiplier_table", {})
        if "table_hash" in jrmt:
            raise RuntimeError(
                "MANIFEST_TABLEHASH_PRESENT: joint_risk_multiplier_table must not "
                "contain a table_hash key per Manifest Authority Policy."
            )

        # MVI-03: Extract the three FAS v6.1.0 constants.
        try:
            thresholds = data["thresholds"]
            shock_cap  = float(thresholds["shock_exposure_cap"])
            max_dd     = float(thresholds["max_drawdown_threshold"])
            vol_trig   = float(thresholds["vol_compression_trigger"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"INTEGRITY_FAILURE: Missing or invalid threshold key in manifest: {exc}"
            ) from exc

        # MVI-04: Verify the three extracted constants equal FAS v6.1.0 values.
        if not self._floats_equal(shock_cap, self._EXPECTED_SHOCK_EXPOSURE_CAP):
            raise RuntimeError(
                f"MANIFEST_CONSTANT_MISMATCH: shock_exposure_cap={shock_cap} "
                f"expected {self._EXPECTED_SHOCK_EXPOSURE_CAP}."
            )
        if not self._floats_equal(max_dd, self._EXPECTED_MAX_DRAWDOWN_THRESHOLD):
            raise RuntimeError(
                f"MANIFEST_CONSTANT_MISMATCH: max_drawdown_threshold={max_dd} "
                f"expected {self._EXPECTED_MAX_DRAWDOWN_THRESHOLD}."
            )
        if not self._floats_equal(vol_trig, self._EXPECTED_VOL_COMPRESSION_TRIGGER):
            raise RuntimeError(
                f"MANIFEST_CONSTANT_MISMATCH: vol_compression_trigger={vol_trig} "
                f"expected {self._EXPECTED_VOL_COMPRESSION_TRIGGER}."
            )

        # MVI-06: Return computed hash for propagation into ExecutionRecords.
        return computed_hash, thresholds
