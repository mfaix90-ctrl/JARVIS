# jarvis/governance/policy_validator.py
# Version: 1.0.0
# Authority: Master FAS v6.1.0-G -- Governance Integration Layer

from __future__ import annotations
from dataclasses import dataclass
from typing import List
from jarvis.core.regime import GlobalRegimeState
from jarvis.utils.constants import QUALITY_SCORE_CAP_UNDER_UNCERTAINTY

_WINDOW_MINIMUM: int                  = 20
_STEP_MINIMUM: int                    = 1
_PERIODS_PER_YEAR_MIN: int            = 1
_PERIODS_PER_YEAR_MAX: int            = 1000
_CRISIS_MIN_META_UNCERTAINTY: float   = 0.5
_STANDARD_PERIODS_PER_YEAR: frozenset = frozenset({12, 52, 252})


@dataclass(frozen=True)
class PolicyViolation:
    rule_id:        str
    field_name:     str
    observed_value: object
    message:        str
    is_blocking:    bool


@dataclass(frozen=True)
class PolicyValidationResult:
    is_compliant:        bool
    violations:          tuple
    warnings:            tuple
    blocking_violations: tuple
    validated_fields:    tuple


def validate_pipeline_config(
    meta_uncertainty: float,
    initial_capital: float,
    window: int,
    step: int,
    regime: GlobalRegimeState,
    periods_per_year: int = 252,
) -> PolicyValidationResult:
    violations: List[PolicyViolation] = []
    validated_fields: List[str] = []

    validated_fields.append("meta_uncertainty")
    if not isinstance(meta_uncertainty, (int, float)):
        violations.append(PolicyViolation("GOV-01", "meta_uncertainty", meta_uncertainty,
            f"meta_uncertainty muss numerisch sein; erhalten: {type(meta_uncertainty).__name__}.", True))
    elif not (0.0 <= float(meta_uncertainty) <= 1.0):
        violations.append(PolicyViolation("GOV-01", "meta_uncertainty", meta_uncertainty,
            f"meta_uncertainty muss in [0.0, 1.0] liegen; erhalten: {meta_uncertainty}.", True))

    validated_fields.append("initial_capital")
    if not isinstance(initial_capital, (int, float)):
        violations.append(PolicyViolation("GOV-02", "initial_capital", initial_capital,
            f"initial_capital muss numerisch sein; erhalten: {type(initial_capital).__name__}.", True))
    elif float(initial_capital) <= 0.0:
        violations.append(PolicyViolation("GOV-02", "initial_capital", initial_capital,
            f"initial_capital muss strikt positiv sein (> 0.0); erhalten: {initial_capital}.", True))

    validated_fields.append("window")
    if not isinstance(window, int):
        violations.append(PolicyViolation("GOV-03", "window", window,
            f"window muss ein Integer sein; erhalten: {type(window).__name__}.", True))
    elif window < _WINDOW_MINIMUM:
        violations.append(PolicyViolation("GOV-03", "window", window,
            f"window muss >= {_WINDOW_MINIMUM} sein (spiegelt RiskEngine-Minimum); erhalten: {window}.", True))

    validated_fields.append("step")
    if not isinstance(step, int):
        violations.append(PolicyViolation("GOV-04", "step", step,
            f"step muss ein Integer sein; erhalten: {type(step).__name__}.", True))
    elif step < _STEP_MINIMUM:
        violations.append(PolicyViolation("GOV-04", "step", step,
            f"step muss >= {_STEP_MINIMUM} sein; erhalten: {step}.", True))
    elif isinstance(window, int) and step > window:
        violations.append(PolicyViolation("GOV-04", "step", step,
            f"step muss <= window ({window}) sein; erhalten: {step}.", True))

    validated_fields.append("regime")
    if not isinstance(regime, GlobalRegimeState):
        violations.append(PolicyViolation("GOV-05", "regime", regime,
            f"regime muss eine GlobalRegimeState-Enum-Instanz sein; erhalten: {type(regime).__name__}.", True))

    validated_fields.append("regime_meta_uncertainty_coherence")
    if (isinstance(regime, GlobalRegimeState)
            and regime == GlobalRegimeState.CRISIS
            and isinstance(meta_uncertainty, (int, float))
            and 0.0 <= float(meta_uncertainty) <= 1.0
            and float(meta_uncertainty) < _CRISIS_MIN_META_UNCERTAINTY):
        violations.append(PolicyViolation("GOV-06", "meta_uncertainty", meta_uncertainty,
            f"Bei regime=CRISIS muss meta_uncertainty >= {_CRISIS_MIN_META_UNCERTAINTY} sein; erhalten: {meta_uncertainty}.", True))

    validated_fields.append("high_uncertainty_capital_floor")
    if (isinstance(meta_uncertainty, (int, float))
            and 0.0 <= float(meta_uncertainty) <= 1.0
            and float(meta_uncertainty) >= QUALITY_SCORE_CAP_UNDER_UNCERTAINTY
            and isinstance(initial_capital, (int, float))
            and float(initial_capital) <= 0.0):
        violations.append(PolicyViolation("GOV-07", "initial_capital", initial_capital,
            f"Bei hoher meta_uncertainty (>= {QUALITY_SCORE_CAP_UNDER_UNCERTAINTY}) "
            f"muss initial_capital > 0.0 sein; erhalten: {initial_capital}. "
            f"(Advisory: GOV-02 flaggt dies bereits als blocking.)", False))

    validated_fields.append("periods_per_year")
    if not isinstance(periods_per_year, int):
        violations.append(PolicyViolation("GOV-08", "periods_per_year", periods_per_year,
            f"periods_per_year muss ein Integer sein; erhalten: {type(periods_per_year).__name__}.", True))
    elif not (_PERIODS_PER_YEAR_MIN <= periods_per_year <= _PERIODS_PER_YEAR_MAX):
        violations.append(PolicyViolation("GOV-08", "periods_per_year", periods_per_year,
            f"periods_per_year muss in [{_PERIODS_PER_YEAR_MIN}, {_PERIODS_PER_YEAR_MAX}] liegen; erhalten: {periods_per_year}.", True))
    elif periods_per_year not in _STANDARD_PERIODS_PER_YEAR:
        violations.append(PolicyViolation("GOV-08", "periods_per_year", periods_per_year,
            f"periods_per_year={periods_per_year} ist nicht-standard. "
            f"Standard-Werte: {sorted(_STANDARD_PERIODS_PER_YEAR)}. Wert akzeptiert, aber zur Audit-Pruefung markiert.", False))

    blocking = tuple(v for v in violations if v.is_blocking)
    advisory = tuple(v for v in violations if not v.is_blocking)
    return PolicyValidationResult(
        is_compliant=len(blocking) == 0,
        violations=tuple(violations),
        warnings=advisory,
        blocking_violations=blocking,
        validated_fields=tuple(validated_fields),
    )
