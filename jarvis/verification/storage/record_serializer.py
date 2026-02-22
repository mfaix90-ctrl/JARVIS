# jarvis/verification/storage/record_serializer.py
# RecordSerializer -- serializes ExecutionRecord sets to JSON files.
# Authority: DVH Implementation Blueprint v1.0.0 Section 9.
#
# RSF-01: All float values serialized using float.hex() (lossless IEEE 754).
# RSF-02: +0.0, -0.0, +inf, -inf, NaN each serialize to distinct strings.
# RSF-04: File name format: {run_id}_{stage}_{timestamp}.json
# RSF-05: runs directory created if it does not exist.
#
# Canonical enum instances (GlobalRegimeState, CorrelationRegimeState) are
# serialized as their .value string. Deserialization restores enum instances.

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from jarvis.core.regime import GlobalRegimeState, CorrelationRegimeState
from jarvis.verification.data_models.execution_record import ExecutionRecord, ObservedOutput
from jarvis.verification.data_models.input_vector import InputVector
from jarvis.verification.harness_version import HARNESS_VERSION, STORAGE_FORMAT_VERSION


def _serialize_float(value: float) -> str:
    """
    RSF-01: Serialize float to lossless hexadecimal representation.
    RSF-02: Handles +0.0, -0.0, +inf, -inf, NaN distinctly.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value.hex()


def _serialize_global_regime(r: Optional[GlobalRegimeState]) -> Optional[str]:
    """Serialize GlobalRegimeState enum to its .value string, or None."""
    if r is None:
        return None
    return r.value


def _serialize_correlation_regime(r: Optional[CorrelationRegimeState]) -> Optional[str]:
    """Serialize CorrelationRegimeState enum to its .value string, or None."""
    if r is None:
        return None
    return r.value


def _serialize_input_vector(iv: InputVector) -> dict:
    return {
        "vector_id":          iv.vector_id,
        "group_id":           iv.group_id,
        "returns_history":    [_serialize_float(r) for r in iv.returns_history],
        "current_regime_str": iv.current_regime_str,
        "meta_uncertainty":   _serialize_float(iv.meta_uncertainty),
        "macro_regime":       _serialize_global_regime(iv.macro_regime),
        "correlation_regime": _serialize_correlation_regime(iv.correlation_regime),
        "realized_vol":       _serialize_float(iv.realized_vol) if iv.realized_vol is not None else None,
        "target_vol":         _serialize_float(iv.target_vol) if iv.target_vol is not None else None,
        "regime_posterior":   _serialize_float(iv.regime_posterior) if iv.regime_posterior is not None else None,
        "expect_exception":   iv.expect_exception,
        "description":        iv.description,
    }


def _serialize_observed_output(out: ObservedOutput) -> dict:
    return {
        "expected_drawdown":       _serialize_float(out.expected_drawdown),
        "expected_drawdown_p95":   _serialize_float(out.expected_drawdown_p95),
        "volatility_forecast":     _serialize_float(out.volatility_forecast),
        "risk_compression_active": out.risk_compression_active,
        "position_size_factor":    _serialize_float(out.position_size_factor),
        "exposure_weight":         _serialize_float(out.exposure_weight),
        "risk_regime":             out.risk_regime,
        "exception_raised":        out.exception_raised,
        "exception_type":          out.exception_type,
    }


def _serialize_record(rec: ExecutionRecord) -> dict:
    return {
        "vector_id":       rec.vector_id,
        "group_id":        rec.group_id,
        "input_vector":    _serialize_input_vector(rec.input_vector),
        "observed_output": _serialize_observed_output(rec.observed_output),
        "stage":           rec.stage,
        "manifest_hash":   rec.manifest_hash,
        "timestamp_iso":   rec.timestamp_iso,
        "execution_id":    rec.execution_id,
        "harness_version": rec.harness_version,
        "module_version":  rec.module_version,
    }


class RecordSerializer:
    """
    Serializes a list of ExecutionRecords to a JSON file per RSF-04/RSF-05.
    Float values serialized using float.hex() (RSF-01).
    Enum values serialized as their .value strings.
    """

    def serialize(
        self,
        records:  List[ExecutionRecord],
        runs_dir: Path,
        run_id:   str,
        stage:    str,
    ) -> Path:
        """
        Write records to a JSON file in runs_dir.
        Returns the path of the written file.
        """
        runs_dir.mkdir(parents=True, exist_ok=True)

        ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{run_id}_{stage}_{ts}.json"
        filepath = runs_dir / filename

        payload = {
            "format_version":  STORAGE_FORMAT_VERSION,
            "harness_version": HARNESS_VERSION,
            "module_version":  records[0].module_version if records else "",
            "run_id":          run_id,
            "stage":           stage,
            "vector_count":    len(records),
            "records":         [_serialize_record(r) for r in records],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return filepath
