# jarvis/verification/data_models/comparison_report.py
# ComparisonReport data class for BIC and CCV output.

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FieldMismatch:
    """
    Record of a single field mismatch detected by the bit comparator.
    """
    vector_id:    str
    field_name:   str
    er_value_hex: str    # hex representation of ER-stage value
    re_value_hex: str    # hex representation of RE-stage value
    failure_type: str    # e.g., DETERMINISM_BREACH, BACKWARD_COMPAT_VIOLATION


@dataclass(frozen=True)
class ComparisonReport:
    """
    Full comparison report produced after BIC and CCV complete.

    Fields:
      passed          -- True iff all comparisons passed with zero failures.
      total_vectors   -- Number of vector pairs compared.
      mismatches      -- List of FieldMismatch records. Empty on pass.
      clip_violations -- List of clip-chain violation description strings.
      notes           -- Any non-failure informational notes.
    """
    passed:          bool
    total_vectors:   int
    mismatches:      tuple    # tuple of FieldMismatch, immutable
    clip_violations: tuple    # tuple of str, immutable
    notes:           tuple    # tuple of str, immutable
