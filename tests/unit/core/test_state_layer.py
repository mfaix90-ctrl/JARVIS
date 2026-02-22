# =============================================================================
# JARVIS v6.0.1 — SESSION 05, PHASE 5.1: STATE LAYER — Unit Tests
# File:   tests/unit/core/test_state_layer.py
# Authority: JARVIS FAS v6.0.1 — 02-05_CORE.md, S05 section
# Phase:  5.1 — LatentState invariant enforcement only
# =============================================================================
#
# Coverage:
#   INV-S05-01  dimension = 12 (hard assert)
#   INV-S05-02  NaN / Inf rejected with StateError
#   INV-S05-03  regime_confidence clipped to [0.0, 1.0]
#   INV-S05-04  stability clipped to [0.0, 1.0]
#   INV-S05-05  prediction_uncertainty floored at 0.0
#   INV-S05-06  regime clamped to [0, 4]
#   INV-S05-07  frozen -- FrozenInstanceError on mutation
#   INV-S05-08  field names match FAS contract
# =============================================================================

from __future__ import annotations

import dataclasses
import math

import pytest

from jarvis.core.state_layer import (
    LATENT_STATE_DIMS,
    REGIME_INT_MAX,
    REGIME_INT_MIN,
    LatentState,
    StateError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_state(**overrides) -> LatentState:
    """Return a valid LatentState, optionally overriding specific fields."""
    defaults = dict(
        regime=0,
        volatility=0.02,
        trend_strength=0.1,
        mean_reversion=0.05,
        liquidity=0.8,
        stress=0.1,
        momentum=0.2,
        drift=0.001,
        noise=0.01,
        regime_confidence=0.75,
        stability=0.9,
        prediction_uncertainty=0.3,
    )
    defaults.update(overrides)
    return LatentState(**defaults)


# ---------------------------------------------------------------------------
# INV-S05-01: dimension = exactly 12
# ---------------------------------------------------------------------------

class TestDimension:
    def test_field_count_is_12(self) -> None:
        state = LatentState.default()
        actual = len(dataclasses.fields(state))
        assert actual == LATENT_STATE_DIMS, (
            f"Expected {LATENT_STATE_DIMS} fields, found {actual}"
        )

    def test_constant_value(self) -> None:
        assert LATENT_STATE_DIMS == 12

    def test_as_tuple_length_is_12(self) -> None:
        state = LatentState.default()
        assert len(state.as_tuple()) == LATENT_STATE_DIMS

    def test_field_names_match_fas_contract(self) -> None:
        """All 12 FAS-specified field names must be present in declared order."""
        expected = [
            "regime",
            "volatility",
            "trend_strength",
            "mean_reversion",
            "liquidity",
            "stress",
            "momentum",
            "drift",
            "noise",
            "regime_confidence",
            "stability",
            "prediction_uncertainty",
        ]
        actual = [f.name for f in dataclasses.fields(LatentState)]
        assert actual == expected, (
            f"Field names or order mismatch.\nExpected: {expected}\nGot:      {actual}"
        )


# ---------------------------------------------------------------------------
# INV-S05-02: NaN and Inf rejected on every float field
# ---------------------------------------------------------------------------

class TestFiniteEnforcement:
    _FLOAT_FIELDS = [
        "volatility",
        "trend_strength",
        "mean_reversion",
        "liquidity",
        "stress",
        "momentum",
        "drift",
        "noise",
        "regime_confidence",
        "stability",
        "prediction_uncertainty",
    ]

    @pytest.mark.parametrize("field_name", _FLOAT_FIELDS)
    def test_nan_raises_state_error(self, field_name: str) -> None:
        with pytest.raises(StateError, match=field_name):
            _valid_state(**{field_name: float("nan")})

    @pytest.mark.parametrize("field_name", _FLOAT_FIELDS)
    def test_pos_inf_raises_state_error(self, field_name: str) -> None:
        with pytest.raises(StateError, match=field_name):
            _valid_state(**{field_name: float("inf")})

    @pytest.mark.parametrize("field_name", _FLOAT_FIELDS)
    def test_neg_inf_raises_state_error(self, field_name: str) -> None:
        with pytest.raises(StateError, match=field_name):
            _valid_state(**{field_name: float("-inf")})

    def test_state_error_is_not_caught_silently(self) -> None:
        """StateError must be a subclass of Exception, not BaseException only."""
        assert issubclass(StateError, Exception)

    def test_valid_state_constructs_without_error(self) -> None:
        state = _valid_state()
        assert state is not None


# ---------------------------------------------------------------------------
# INV-S05-03: regime_confidence clipped to [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestRegimeConfidenceClipping:
    def test_above_one_clipped_to_one(self) -> None:
        state = _valid_state(regime_confidence=1.5)
        assert state.regime_confidence == 1.0

    def test_below_zero_clipped_to_zero(self) -> None:
        state = _valid_state(regime_confidence=-0.3)
        assert state.regime_confidence == 0.0

    def test_exact_zero_preserved(self) -> None:
        state = _valid_state(regime_confidence=0.0)
        assert state.regime_confidence == 0.0

    def test_exact_one_preserved(self) -> None:
        state = _valid_state(regime_confidence=1.0)
        assert state.regime_confidence == 1.0

    def test_interior_value_unchanged(self) -> None:
        state = _valid_state(regime_confidence=0.65)
        assert state.regime_confidence == 0.65

    def test_nan_still_rejected_before_clipping(self) -> None:
        """NaN must be caught by the finiteness check, not by clipping."""
        with pytest.raises(StateError):
            _valid_state(regime_confidence=float("nan"))


# ---------------------------------------------------------------------------
# INV-S05-04: stability clipped to [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestStabilityClipping:
    def test_above_one_clipped_to_one(self) -> None:
        state = _valid_state(stability=2.0)
        assert state.stability == 1.0

    def test_below_zero_clipped_to_zero(self) -> None:
        state = _valid_state(stability=-1.0)
        assert state.stability == 0.0

    def test_exact_zero_preserved(self) -> None:
        state = _valid_state(stability=0.0)
        assert state.stability == 0.0

    def test_exact_one_preserved(self) -> None:
        state = _valid_state(stability=1.0)
        assert state.stability == 1.0

    def test_interior_value_unchanged(self) -> None:
        state = _valid_state(stability=0.42)
        assert state.stability == 0.42


# ---------------------------------------------------------------------------
# INV-S05-05: prediction_uncertainty floored at 0.0
# ---------------------------------------------------------------------------

class TestPredictionUncertaintyFloor:
    def test_negative_floored_to_zero(self) -> None:
        state = _valid_state(prediction_uncertainty=-0.5)
        assert state.prediction_uncertainty == 0.0

    def test_zero_preserved(self) -> None:
        state = _valid_state(prediction_uncertainty=0.0)
        assert state.prediction_uncertainty == 0.0

    def test_positive_value_unchanged(self) -> None:
        state = _valid_state(prediction_uncertainty=0.8)
        assert state.prediction_uncertainty == 0.8

    def test_large_value_unchanged(self) -> None:
        state = _valid_state(prediction_uncertainty=99.9)
        assert state.prediction_uncertainty == 99.9


# ---------------------------------------------------------------------------
# INV-S05-06: regime clamped to [0, 4]
# ---------------------------------------------------------------------------

class TestRegimeClamping:
    def test_above_max_clamped(self) -> None:
        state = _valid_state(regime=10)
        assert state.regime == REGIME_INT_MAX

    def test_below_min_clamped(self) -> None:
        state = _valid_state(regime=-3)
        assert state.regime == REGIME_INT_MIN

    def test_valid_values_preserved(self) -> None:
        for r in range(REGIME_INT_MIN, REGIME_INT_MAX + 1):
            state = _valid_state(regime=r)
            assert state.regime == r

    def test_regime_is_int_after_construction(self) -> None:
        state = _valid_state(regime=2)
        assert isinstance(state.regime, int)


# ---------------------------------------------------------------------------
# INV-S05-07: frozen — no mutation allowed
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_cannot_set_field_after_construction(self) -> None:
        state = _valid_state()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            state.regime = 3  # type: ignore[misc]

    def test_cannot_set_float_field_after_construction(self) -> None:
        state = _valid_state()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            state.volatility = 0.99  # type: ignore[misc]

    def test_cannot_delete_field(self) -> None:
        state = _valid_state()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            del state.regime  # type: ignore[misc]

    def test_two_equal_states_are_equal(self) -> None:
        a = _valid_state()
        b = _valid_state()
        assert a == b

    def test_states_with_different_fields_not_equal(self) -> None:
        a = _valid_state(regime=0)
        b = _valid_state(regime=1)
        assert a != b

    def test_state_is_hashable(self) -> None:
        """frozen=True implies __hash__ is generated; instance must be hashable."""
        state = _valid_state()
        h = hash(state)
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# INV-S05-08: D(t) field correspondence
# ---------------------------------------------------------------------------

class TestSystemContractCorrespondence:
    def test_regime_confidence_maps_to_R(self) -> None:
        """R in D(t) — always in [0, 1]."""
        state = _valid_state(regime_confidence=0.88)
        assert 0.0 <= state.regime_confidence <= 1.0

    def test_stability_maps_to_S(self) -> None:
        """S in D(t) — always in [0, 1]."""
        state = _valid_state(stability=0.72)
        assert 0.0 <= state.stability <= 1.0

    def test_prediction_uncertainty_non_negative(self) -> None:
        """Contribution to sigma^2 — always >= 0."""
        state = _valid_state(prediction_uncertainty=0.4)
        assert state.prediction_uncertainty >= 0.0


# ---------------------------------------------------------------------------
# default() factory
# ---------------------------------------------------------------------------

class TestDefaultFactory:
    def test_default_has_correct_dimensions(self) -> None:
        state = LatentState.default()
        assert len(dataclasses.fields(state)) == LATENT_STATE_DIMS

    def test_default_regime_confidence_is_zero(self) -> None:
        """Forces FM-01 on first cycle — downstream must treat as uncertain."""
        state = LatentState.default()
        assert state.regime_confidence == 0.0

    def test_default_regime_is_max_index(self) -> None:
        """regime=4 (SHOCK/UNKNOWN index) for conservative initialisation."""
        state = LatentState.default()
        assert state.regime == REGIME_INT_MAX

    def test_default_all_floats_finite(self) -> None:
        state = LatentState.default()
        for f in dataclasses.fields(state):
            if f.type in ("float",) or f.name != "regime":
                val = getattr(state, f.name)
                if isinstance(val, float):
                    assert math.isfinite(val), (
                        f"default().{f.name} is non-finite: {val}"
                    )

    def test_default_is_deterministic(self) -> None:
        """Two calls to default() must return equal instances."""
        assert LatentState.default() == LatentState.default()


# ---------------------------------------------------------------------------
# No forbidden dependencies
# ---------------------------------------------------------------------------

class TestNoDependencies:
    def test_no_numpy_import(self) -> None:
        import importlib
        import sys
        mod = sys.modules.get("jarvis.core.state_layer")
        if mod is None:
            mod = importlib.import_module("jarvis.core.state_layer")
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            src = f.read()
        assert "import numpy" not in src, "numpy must not be imported in Phase 5.1"
        assert "from numpy" not in src, "numpy must not be imported in Phase 5.1"

    def test_no_logging_import(self) -> None:
        import importlib
        import sys
        mod = sys.modules.get("jarvis.core.state_layer")
        if mod is None:
            mod = importlib.import_module("jarvis.core.state_layer")
        source_file = mod.__file__
        with open(source_file) as f:
            src = f.read()
        # Allow "logging" only in comments/docstrings
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    assert "logging" not in (name or ""), (
                        f"logging must not be imported in Phase 5.1, found: {name}"
                    )

    def test_no_datetime_now_in_code(self) -> None:
        import importlib
        import sys
        import ast
        mod = sys.modules.get("jarvis.core.state_layer")
        if mod is None:
            mod = importlib.import_module("jarvis.core.state_layer")
        source_file = mod.__file__
        with open(source_file) as f:
            src = f.read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == "now" and isinstance(node.value, ast.Attribute):
                    if node.value.attr == "datetime":
                        pytest.fail("datetime.now() found in executable code")
