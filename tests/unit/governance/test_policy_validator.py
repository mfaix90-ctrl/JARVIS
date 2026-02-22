# =============================================================================
# JARVIS v6.1.0 -- GOVERNANCE TESTS
# File:   tests/unit/governance/test_policy_validator.py
# Coverage target: >= 90% line, 100% branch on policy_validator.py
# =============================================================================

import pytest
from jarvis.core.regime import GlobalRegimeState
from jarvis.governance.policy_validator import (
    PolicyViolation,
    PolicyValidationResult,
    validate_pipeline_config,
)
from jarvis.governance.exceptions import GovernanceViolationError


# =============================================================================
# SECTION 1 -- Helpers
# =============================================================================

def _valid_call(**overrides):
    """Return a valid validate_pipeline_config() call, with optional overrides."""
    defaults = dict(
        meta_uncertainty=0.3,
        initial_capital=100_000.0,
        window=60,
        step=1,
        regime=GlobalRegimeState.RISK_ON,
        periods_per_year=252,
    )
    defaults.update(overrides)
    return validate_pipeline_config(**defaults)


# =============================================================================
# SECTION 2 -- Happy path: fully compliant call
# =============================================================================

class TestCompliantCall:

    def test_valid_call_is_compliant(self):
        result = _valid_call()
        assert result.is_compliant is True

    def test_valid_call_no_violations(self):
        result = _valid_call()
        assert result.blocking_violations == ()

    def test_valid_call_validated_fields_non_empty(self):
        result = _valid_call()
        assert len(result.validated_fields) > 0

    def test_valid_call_returns_policy_validation_result(self):
        result = _valid_call()
        assert isinstance(result, PolicyValidationResult)

    def test_all_regime_states_accepted(self):
        for regime in GlobalRegimeState:
            unc = 0.6 if regime == GlobalRegimeState.CRISIS else 0.3
            result = _valid_call(regime=regime, meta_uncertainty=unc)
            assert result.is_compliant is True, \
                f"Expected compliant for regime={regime}"

    def test_standard_periods_per_year_252(self):
        assert _valid_call(periods_per_year=252).is_compliant is True

    def test_standard_periods_per_year_52(self):
        assert _valid_call(periods_per_year=52).is_compliant is True

    def test_standard_periods_per_year_12(self):
        assert _valid_call(periods_per_year=12).is_compliant is True


# =============================================================================
# SECTION 3 -- GOV-01: meta_uncertainty
# =============================================================================

class TestGov01MetaUncertainty:

    def test_meta_uncertainty_zero_boundary(self):
        assert _valid_call(meta_uncertainty=0.0).is_compliant is True

    def test_meta_uncertainty_one_boundary(self):
        assert _valid_call(meta_uncertainty=1.0).is_compliant is True

    def test_meta_uncertainty_negative_blocking(self):
        result = _valid_call(meta_uncertainty=-0.01)
        assert result.is_compliant is False
        rule_ids = [v.rule_id for v in result.blocking_violations]
        assert "GOV-01" in rule_ids

    def test_meta_uncertainty_above_one_blocking(self):
        result = _valid_call(meta_uncertainty=1.01)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-01" for v in result.blocking_violations)

    def test_meta_uncertainty_string_blocking(self):
        result = _valid_call(meta_uncertainty="high")
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-01" for v in result.blocking_violations)

    def test_meta_uncertainty_none_blocking(self):
        result = _valid_call(meta_uncertainty=None)
        assert result.is_compliant is False

    def test_meta_uncertainty_int_zero_accepted(self):
        """int 0 is numerically valid (subclass of numeric)."""
        assert _valid_call(meta_uncertainty=0).is_compliant is True


# =============================================================================
# SECTION 4 -- GOV-02: initial_capital
# =============================================================================

class TestGov02InitialCapital:

    def test_capital_positive_accepted(self):
        assert _valid_call(initial_capital=1.0).is_compliant is True

    def test_capital_zero_blocking(self):
        result = _valid_call(initial_capital=0.0)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-02" for v in result.blocking_violations)

    def test_capital_negative_blocking(self):
        result = _valid_call(initial_capital=-100.0)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-02" for v in result.blocking_violations)

    def test_capital_string_blocking(self):
        result = _valid_call(initial_capital="lots")
        assert result.is_compliant is False

    def test_capital_large_value_accepted(self):
        assert _valid_call(initial_capital=1e12).is_compliant is True


# =============================================================================
# SECTION 5 -- GOV-03: window
# =============================================================================

class TestGov03Window:

    def test_window_20_boundary(self):
        assert _valid_call(window=20, step=1).is_compliant is True

    def test_window_19_blocking(self):
        result = _valid_call(window=19, step=1)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-03" for v in result.blocking_violations)

    def test_window_float_blocking(self):
        result = _valid_call(window=60.0, step=1)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-03" for v in result.blocking_violations)

    def test_window_large_accepted(self):
        assert _valid_call(window=1000, step=1).is_compliant is True


# =============================================================================
# SECTION 6 -- GOV-04: step
# =============================================================================

class TestGov04Step:

    def test_step_1_boundary(self):
        assert _valid_call(window=60, step=1).is_compliant is True

    def test_step_0_blocking(self):
        result = _valid_call(window=60, step=0)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-04" for v in result.blocking_violations)

    def test_step_negative_blocking(self):
        result = _valid_call(window=60, step=-1)
        assert result.is_compliant is False

    def test_step_greater_than_window_blocking(self):
        result = _valid_call(window=60, step=61)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-04" for v in result.blocking_violations)

    def test_step_equal_window_accepted(self):
        assert _valid_call(window=60, step=60).is_compliant is True

    def test_step_float_blocking(self):
        result = _valid_call(window=60, step=1.0)
        assert result.is_compliant is False


# =============================================================================
# SECTION 7 -- GOV-05: regime
# =============================================================================

class TestGov05Regime:

    def test_string_regime_blocking(self):
        result = _valid_call(regime="RISK_ON")
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-05" for v in result.blocking_violations)

    def test_none_regime_blocking(self):
        result = _valid_call(regime=None)
        assert result.is_compliant is False

    def test_int_regime_blocking(self):
        result = _valid_call(regime=0)
        assert result.is_compliant is False


# =============================================================================
# SECTION 8 -- GOV-06: CRISIS meta_uncertainty coherence
# =============================================================================

class TestGov06CrisisCoherence:

    def test_crisis_with_unc_0_5_boundary(self):
        """meta_uncertainty=0.5 is the exact boundary: must be compliant."""
        result = _valid_call(
            regime=GlobalRegimeState.CRISIS,
            meta_uncertainty=0.5,
        )
        assert result.is_compliant is True

    def test_crisis_with_unc_above_0_5(self):
        result = _valid_call(regime=GlobalRegimeState.CRISIS, meta_uncertainty=0.7)
        assert result.is_compliant is True

    def test_crisis_with_unc_below_0_5_blocking(self):
        result = _valid_call(regime=GlobalRegimeState.CRISIS, meta_uncertainty=0.3)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-06" for v in result.blocking_violations)

    def test_crisis_unc_0_49_blocking(self):
        """0.49 is just below boundary."""
        result = _valid_call(regime=GlobalRegimeState.CRISIS, meta_uncertainty=0.49)
        assert result.is_compliant is False

    def test_non_crisis_low_unc_accepted(self):
        """GOV-06 is ONLY for CRISIS regime; RISK_ON with unc=0.1 is fine."""
        for regime in (GlobalRegimeState.RISK_ON, GlobalRegimeState.RISK_OFF,
                       GlobalRegimeState.TRANSITION):
            result = _valid_call(regime=regime, meta_uncertainty=0.1)
            assert result.is_compliant is True, \
                f"Expected compliant for non-CRISIS regime={regime} with low unc"

    def test_gov06_field_is_meta_uncertainty(self):
        result = _valid_call(regime=GlobalRegimeState.CRISIS, meta_uncertainty=0.1)
        gov06_violations = [v for v in result.violations if v.rule_id == "GOV-06"]
        assert len(gov06_violations) == 1
        assert gov06_violations[0].field_name == "meta_uncertainty"

    def test_gov06_is_blocking(self):
        result = _valid_call(regime=GlobalRegimeState.CRISIS, meta_uncertainty=0.1)
        gov06 = [v for v in result.violations if v.rule_id == "GOV-06"]
        assert gov06[0].is_blocking is True


# =============================================================================
# SECTION 9 -- GOV-07: high uncertainty capital floor (advisory)
# =============================================================================

class TestGov07Advisory:

    def test_gov07_is_non_blocking(self):
        """GOV-07 fires only when capital <= 0 AND uncertainty >= threshold.
        But GOV-02 already fires as blocking for capital <= 0.
        Verify GOV-07 is non-blocking (advisory) when it fires."""
        from jarvis.utils.constants import QUALITY_SCORE_CAP_UNDER_UNCERTAINTY
        # To trigger GOV-07: meta_uncertainty >= 0.60 and capital <= 0
        # GOV-02 also fires. GOV-07 should be advisory (is_blocking=False).
        result = validate_pipeline_config(
            meta_uncertainty=QUALITY_SCORE_CAP_UNDER_UNCERTAINTY,
            initial_capital=-1.0,   # triggers GOV-02 (blocking) and GOV-07 (advisory)
            window=60,
            step=1,
            regime=GlobalRegimeState.RISK_ON,
        )
        advisory = [v for v in result.violations if v.rule_id == "GOV-07"]
        if advisory:
            assert advisory[0].is_blocking is False


# =============================================================================
# SECTION 10 -- GOV-08: periods_per_year
# =============================================================================

class TestGov08PeriodsPerYear:

    def test_non_standard_period_is_advisory(self):
        """periods_per_year=100 is in [1,1000] but non-standard -> advisory."""
        result = _valid_call(periods_per_year=100)
        assert result.is_compliant is True   # non-blocking: compliant but warned
        gov08 = [v for v in result.violations if v.rule_id == "GOV-08"]
        assert len(gov08) == 1
        assert gov08[0].is_blocking is False

    def test_period_zero_blocking(self):
        result = _valid_call(periods_per_year=0)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-08" for v in result.blocking_violations)

    def test_period_1001_blocking(self):
        result = _valid_call(periods_per_year=1001)
        assert result.is_compliant is False

    def test_period_1_boundary_accepted(self):
        assert _valid_call(periods_per_year=1).is_compliant is True

    def test_period_1000_boundary_accepted(self):
        assert _valid_call(periods_per_year=1000).is_compliant is True

    def test_period_float_blocking(self):
        result = _valid_call(periods_per_year=252.0)
        assert result.is_compliant is False
        assert any(v.rule_id == "GOV-08" for v in result.blocking_violations)


# =============================================================================
# SECTION 11 -- Multiple simultaneous violations
# =============================================================================

class TestMultipleViolations:

    def test_two_violations_both_reported(self):
        """window=5 (GOV-03) and step=0 (GOV-04) both blocking."""
        result = _valid_call(window=5, step=0)
        rule_ids = {v.rule_id for v in result.violations}
        assert "GOV-03" in rule_ids
        assert "GOV-04" in rule_ids

    def test_all_fields_invalid(self):
        result = validate_pipeline_config(
            meta_uncertainty=-0.1,
            initial_capital=-1.0,
            window=5,
            step=-1,
            regime="BAD",
        )
        assert result.is_compliant is False
        rule_ids = {v.rule_id for v in result.blocking_violations}
        assert len(rule_ids) >= 4

    def test_result_violations_includes_all(self):
        result = _valid_call(meta_uncertainty=-0.1, initial_capital=-1.0)
        assert len(result.violations) >= 2


# =============================================================================
# SECTION 12 -- PolicyViolation structure
# =============================================================================

class TestPolicyViolationStructure:

    def test_policy_violation_fields(self):
        v = PolicyViolation(
            rule_id="GOV-01",
            field_name="meta_uncertainty",
            observed_value=-0.1,
            message="test message",
            is_blocking=True,
        )
        assert v.rule_id == "GOV-01"
        assert v.field_name == "meta_uncertainty"
        assert v.observed_value == -0.1
        assert v.is_blocking is True

    def test_policy_violation_frozen(self):
        import dataclasses
        v = PolicyViolation(
            rule_id="GOV-01", field_name="f", observed_value=0, message="m", is_blocking=True
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.rule_id = "X"  # type: ignore


# =============================================================================
# SECTION 13 -- Determinism
# =============================================================================

class TestDeterminism:

    def test_valid_call_deterministic(self):
        results = [_valid_call() for _ in range(10)]
        assert all(r.is_compliant == results[0].is_compliant for r in results)

    def test_invalid_call_deterministic(self):
        results = [_valid_call(meta_uncertainty=-0.1) for _ in range(10)]
        assert all(r.is_compliant is False for r in results)
        assert all(len(r.violations) == len(results[0].violations) for r in results)
