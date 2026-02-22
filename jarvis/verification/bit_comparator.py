# jarvis/verification/bit_comparator.py
# BitComparator -- exact IEEE 754 bit-pattern comparison of ER and RE records.
# Authority: DVH Implementation Blueprint v1.0.0 Section 10.
#
# BIC-F-02: math.isclose(), numpy.allclose(), pytest.approx(), and all other
#           tolerance-based comparison functions are PROHIBITED in this module.
# BIC-F-03: The Python equality operator applied directly to float values is
#           PROHIBITED because it considers +0.0 equal to -0.0.
# All float comparisons use struct.pack big-endian double-precision byte sequences.

import struct
import math
from typing import List, Dict, Tuple

from jarvis.verification.data_models.execution_record import ExecutionRecord, ObservedOutput
from jarvis.verification.data_models.comparison_report import ComparisonReport, FieldMismatch
from jarvis.verification.vectors.vector_definitions import BC_PAIRS


def _float_bits(value: float) -> bytes:
    """
    Return 8-byte big-endian IEEE 754 representation (BIC-F-01).
    Distinguishes +0.0 from -0.0.
    """
    return struct.pack(">d", value)


def _floats_equal(a: float, b: float) -> bool:
    """
    Exact bit-pattern equality for floats (BIC-F-01).
    +0.0 and -0.0 are NOT equal under this comparison (BIC-S-01).
    """
    return _float_bits(a) == _float_bits(b)


def _check_special(value: float, field_name: str, vector_id: str) -> None:
    """
    Check for NaN and Inf before any comparison (BIC-S-02, BIC-S-03).
    Raises RuntimeError with failure_type_id prefix on detection.
    """
    if math.isnan(value):
        raise RuntimeError(
            f"FIELD_NAN: Vector {vector_id} field {field_name} contains NaN. "
            "Hard failure before comparison."
        )
    if math.isinf(value):
        raise RuntimeError(
            f"FIELD_INFINITE: Vector {vector_id} field {field_name} contains "
            f"{'positive' if value > 0 else 'negative'} infinity. "
            "Hard failure before comparison."
        )


def _compare_outputs(
    vector_id:    str,
    er_out:       ObservedOutput,
    re_out:       ObservedOutput,
    failure_type: str,
) -> List[FieldMismatch]:
    """
    Compare all nine ObservedOutput fields (BIC-C-01).
    Returns list of FieldMismatch records. Empty list means all fields match.

    Float fields: struct-based bit comparison (BIC-F-01).
    Boolean fields: Python identity comparison (BIC-B-01).
    String fields: exact byte-for-byte equality (BIC-S-04).
    Special value pre-check applied to all float fields (BIC-S-02, BIC-S-03).
    """
    mismatches = []

    # Pre-check all float fields for NaN and Inf before any comparison.
    float_fields = [
        "expected_drawdown",
        "expected_drawdown_p95",
        "volatility_forecast",
        "position_size_factor",
        "exposure_weight",
    ]
    for field in float_fields:
        for rec_label, rec in [("ER", er_out), ("RE", re_out)]:
            val = getattr(rec, field)
            _check_special(val, f"{field}[{rec_label}]", vector_id)

    # Compare float fields: struct-based bit comparison.
    for field in float_fields:
        er_val = getattr(er_out, field)
        re_val = getattr(re_out, field)
        if not _floats_equal(er_val, re_val):
            mismatches.append(FieldMismatch(
                vector_id=vector_id,
                field_name=field,
                er_value_hex=_float_bits(er_val).hex(),
                re_value_hex=_float_bits(re_val).hex(),
                failure_type=failure_type,
            ))

    # Compare boolean: identity (BIC-B-01).
    for field in ["risk_compression_active", "exception_raised"]:
        er_val = getattr(er_out, field)
        re_val = getattr(re_out, field)
        if er_val is not re_val:
            mismatches.append(FieldMismatch(
                vector_id=vector_id,
                field_name=field,
                er_value_hex=str(er_val),
                re_value_hex=str(re_val),
                failure_type="FIELD_BOOLEAN_MISMATCH",
            ))

    # Compare string fields: exact equality (BIC-S-04).
    for field in ["risk_regime", "exception_type"]:
        er_val = getattr(er_out, field)
        re_val = getattr(re_out, field)
        if er_val != re_val:
            mismatches.append(FieldMismatch(
                vector_id=vector_id,
                field_name=field,
                er_value_hex=repr(er_val),
                re_value_hex=repr(re_val),
                failure_type=failure_type,
            ))

    return mismatches


class BitComparator:
    """
    Performs exact bit-pattern comparison of ER-stage vs RE-stage records.

    Two comparison passes are performed:
      Pass 1 (BIC-F-01): ER vs RE for every vector (determinism verification).
      Pass 2 (BIC-BC-01): Backward compatibility pair comparison.

    Any mismatch in either pass produces a FieldMismatch and is a hard failure.

    Method:
      compare(er_records, re_records) -> ComparisonReport
    """

    def compare(
        self,
        er_records: List[ExecutionRecord],
        re_records: List[ExecutionRecord],
    ) -> ComparisonReport:
        """
        Compare ER and RE record sets. Returns ComparisonReport.

        Raises RuntimeError with failure_type_id prefix on NaN or Inf detection.
        All other failures are collected into ComparisonReport.mismatches.
        """
        # Build lookup dicts by vector_id.
        er_by_id: Dict[str, ExecutionRecord] = {r.vector_id: r for r in er_records}
        re_by_id: Dict[str, ExecutionRecord] = {r.vector_id: r for r in re_records}

        all_mismatches = []

        # Pass 1: ER vs RE for every vector.
        for vector_id, er_rec in er_by_id.items():
            if vector_id not in re_by_id:
                all_mismatches.append(FieldMismatch(
                    vector_id=vector_id,
                    field_name="(record missing)",
                    er_value_hex="present",
                    re_value_hex="missing",
                    failure_type="DETERMINISM_BREACH",
                ))
                continue
            re_rec = re_by_id[vector_id]
            mismatches = _compare_outputs(
                vector_id=vector_id,
                er_out=er_rec.observed_output,
                re_out=re_rec.observed_output,
                failure_type="DETERMINISM_BREACH",
            )
            all_mismatches.extend(mismatches)

        # Pass 2: Backward compatibility pairs (BIC-BC-01).
        bc_mismatches = []
        for first_id, second_id in BC_PAIRS:
            if first_id not in er_by_id or second_id not in er_by_id:
                bc_mismatches.append(FieldMismatch(
                    vector_id=f"{first_id}/{second_id}",
                    field_name="(BC pair missing)",
                    er_value_hex="present",
                    re_value_hex="missing",
                    failure_type="BACKWARD_COMPAT_VIOLATION",
                ))
                continue
            first_out  = er_by_id[first_id].observed_output
            second_out = er_by_id[second_id].observed_output
            mismatches = _compare_outputs(
                vector_id=f"{first_id}/{second_id}",
                er_out=first_out,
                re_out=second_out,
                failure_type="BACKWARD_COMPAT_VIOLATION",
            )
            bc_mismatches.extend(mismatches)

        all_mismatches.extend(bc_mismatches)

        passed = len(all_mismatches) == 0
        return ComparisonReport(
            passed=passed,
            total_vectors=len(er_records),
            mismatches=tuple(all_mismatches),
            clip_violations=(),   # Populated by ClipVerifier.
            notes=(),
        )
