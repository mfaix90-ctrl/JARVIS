# jarvis/verification/storage/record_loader.py
# RecordLoader -- loads and validates serialized ExecutionRecord sets.
# Authority: DVH Implementation Blueprint v1.0.0 Sections 9.4 and RSF-06/RSF-07.
#
# RSF-06: Validates format_version, harness_version, module_version, vector_count.
# RSF-07: Detects NaN in any float field during deserialization -- hard failure.
# Used for cross-session replay (DIM-05/DIM-06).
#
# Enum fields (macro_regime, correlation_regime) are deserialized from their
# .value strings back to canonical enum instances from jarvis.core.regime.

import json
import math
from pathlib import Path
from typing import List, Optional

from jarvis.core.regime import GlobalRegimeState, CorrelationRegimeState
from jarvis.verification.data_models.input_vector import InputVector
from jarvis.verification.data_models.execution_record import ExecutionRecord, ObservedOutput
from jarvis.verification.harness_version import HARNESS_VERSION, STORAGE_FORMAT_VERSION


def _deserialize_float(value: str) -> float:
    """
    RSF-01: Deserialize lossless hex float string to float.
    RSF-02: Handles nan, inf, -inf distinctly.
    """
    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float.fromhex(value)


def _deserialize_global_regime(value: Optional[str]) -> Optional[GlobalRegimeState]:
    """
    Deserialize a GlobalRegimeState .value string back to the canonical enum instance.
    Returns None if value is None.
    Raises ValueError on unrecognized string (hard failure path).
    """
    if value is None:
        return None
    try:
        return GlobalRegimeState(value)
    except ValueError:
        raise RuntimeError(
            f"DATA_CORRUPTION: Unrecognized GlobalRegimeState value string "
            f"'{value}' in serialized record. Cannot restore canonical enum."
        )


def _deserialize_correlation_regime(value: Optional[str]) -> Optional[CorrelationRegimeState]:
    """
    Deserialize a CorrelationRegimeState .value string back to the canonical enum instance.
    Returns None if value is None.
    Raises RuntimeError on unrecognized string (hard failure path).
    """
    if value is None:
        return None
    try:
        return CorrelationRegimeState(value)
    except ValueError:
        raise RuntimeError(
            f"DATA_CORRUPTION: Unrecognized CorrelationRegimeState value string "
            f"'{value}' in serialized record. Cannot restore canonical enum."
        )


def _check_nan(value: float, field: str, vector_id: str) -> None:
    """RSF-07: Detect NaN. Raise RuntimeError (hard failure)."""
    if math.isnan(value):
        raise RuntimeError(
            f"DATA_CORRUPTION: NaN detected in field '{field}' for vector "
            f"'{vector_id}' during record deserialization. "
            "Record is corrupt. Hard failure per RSF-07."
        )


def _load_input_vector(d: dict) -> InputVector:
    returns_history = tuple(_deserialize_float(r) for r in d["returns_history"])
    return InputVector(
        vector_id=d["vector_id"],
        group_id=d["group_id"],
        returns_history=returns_history,
        current_regime_str=d["current_regime_str"],
        meta_uncertainty=_deserialize_float(d["meta_uncertainty"]),
        macro_regime=_deserialize_global_regime(d["macro_regime"]),
        correlation_regime=_deserialize_correlation_regime(d["correlation_regime"]),
        realized_vol=_deserialize_float(d["realized_vol"]) if d["realized_vol"] is not None else None,
        target_vol=_deserialize_float(d["target_vol"]) if d["target_vol"] is not None else None,
        regime_posterior=_deserialize_float(d["regime_posterior"]) if d["regime_posterior"] is not None else None,
        expect_exception=d["expect_exception"],
        description=d["description"],
    )


def _load_observed_output(d: dict, vector_id: str) -> ObservedOutput:
    float_fields = [
        "expected_drawdown",
        "expected_drawdown_p95",
        "volatility_forecast",
        "position_size_factor",
        "exposure_weight",
    ]
    values = {}
    for field in float_fields:
        val = _deserialize_float(d[field])
        _check_nan(val, field, vector_id)
        values[field] = val

    return ObservedOutput(
        expected_drawdown=values["expected_drawdown"],
        expected_drawdown_p95=values["expected_drawdown_p95"],
        volatility_forecast=values["volatility_forecast"],
        risk_compression_active=bool(d["risk_compression_active"]),
        position_size_factor=values["position_size_factor"],
        exposure_weight=values["exposure_weight"],
        risk_regime=str(d["risk_regime"]),
        exception_raised=bool(d["exception_raised"]),
        exception_type=str(d["exception_type"]),
    )


def _load_record(d: dict) -> ExecutionRecord:
    vector_id = d["vector_id"]
    iv  = _load_input_vector(d["input_vector"])
    out = _load_observed_output(d["observed_output"], vector_id)
    return ExecutionRecord(
        vector_id=vector_id,
        group_id=d["group_id"],
        input_vector=iv,
        observed_output=out,
        stage=d["stage"],
        manifest_hash=d["manifest_hash"],
        timestamp_iso=d["timestamp_iso"],
        execution_id=d["execution_id"],
        harness_version=d["harness_version"],
        module_version=d["module_version"],
    )


class RecordLoader:
    """
    Loads and validates a serialized ExecutionRecord set from a JSON file.
    Implements RSF-06 and RSF-07 validation.
    Restores canonical enum instances for macro_regime and correlation_regime.
    """

    def load(
        self,
        filepath:       Path,
        module_version: str,
    ) -> List[ExecutionRecord]:
        """
        Load records from filepath. Validate per RSF-06.
        Detect NaN per RSF-07.
        Restore canonical enum instances per Single Authoritative Regime Source rule.
        """
        if not filepath.exists():
            raise RuntimeError(
                f"INTEGRITY_FAILURE: Prior record file not found: {filepath}"
            )

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"DATA_CORRUPTION: Failed to load record file {filepath}: {exc}"
            ) from exc

        # RSF-06: Validate header fields.
        if payload.get("format_version") != STORAGE_FORMAT_VERSION:
            raise RuntimeError(
                f"DATA_CORRUPTION: format_version mismatch. "
                f"File: {payload.get('format_version')}, "
                f"Expected: {STORAGE_FORMAT_VERSION}."
            )
        if payload.get("harness_version") != HARNESS_VERSION:
            raise RuntimeError(
                f"DATA_CORRUPTION: harness_version mismatch. "
                f"File: {payload.get('harness_version')}, "
                f"Expected: {HARNESS_VERSION}."
            )
        if payload.get("module_version") != module_version:
            raise RuntimeError(
                f"DATA_CORRUPTION: module_version mismatch. "
                f"File: {payload.get('module_version')}, "
                f"Expected: {module_version}."
            )

        raw_records = payload.get("records", [])
        if payload.get("vector_count") != len(raw_records):
            raise RuntimeError(
                f"DATA_CORRUPTION: vector_count={payload.get('vector_count')} "
                f"does not match actual record count={len(raw_records)}."
            )

        records = []
        for d in raw_records:
            rec = _load_record(d)
            records.append(rec)

        return records
