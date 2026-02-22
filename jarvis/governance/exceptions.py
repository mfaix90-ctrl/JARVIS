# jarvis/governance/exceptions.py
# Version: 1.0.0
# Authority: Master FAS v6.1.0-G -- Governance Integration Layer

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.governance.policy_validator import PolicyValidationResult


class GovernanceViolationError(Exception):
    """
    Raised when validate_pipeline_config() returns blocking violations.

    Attributes
    ----------
    result : PolicyValidationResult
    blocking_violations : tuple
    """

    def __init__(self, result: "PolicyValidationResult") -> None:
        self.result = result
        self.blocking_violations = result.blocking_violations
        lines = [
            f"Governance policy violated: "
            f"{len(result.blocking_violations)} blocking violation(s) detected.",
        ]
        for v in result.blocking_violations:
            lines.append(f"  [{v.rule_id}] {v.field_name}: {v.message}")
        super().__init__("\n".join(lines))
