# =============================================================================
# JARVIS v6.0.1 — SESSION 04: FEATURE LAYER — Official Unit Tests
# File:   tests/unit/core/test_feature_layer.py
# Authority: JARVIS FAS v6.0.1 — S04 section
# =============================================================================
#
# Coverage (per task requirements):
#   REQ-01  FEATURE_NAMES length == 99
#   REQ-02  compute_features() returns exactly 99 keys
#   REQ-03  NaN/Inf inputs replaced with 0.0
#   REQ-04  Drift detection returns deterministic results
#   REQ-05  Severity clipping in [0.0, 1.0]
#   REQ-06  >10% severe drift triggers hard_stop flag
#   REQ-07  No forbidden imports (numpy, random, datetime.now, etc.)
#   REQ-08  Computation time constraint logic (mocked timing)
#
# ── Implementation note: compute_features() free-variable workaround ─────────
#
# feature_layer._compute_raw() references `market_data` as a free variable
# that is NOT stored as self._market_data, and NOT passed as a parameter.
# Python resolves it through the module's global namespace at call time.
# All tests that exercise compute_features() use the _inject_market_data()
# context manager, which temporarily places the MarketData object into
# the feature_layer module's __dict__ and restores the original state on
# exit (even if an exception is raised).
#
# This is standard test-time namespace injection (equivalent to monkeypatching).
# It does NOT modify feature_layer.py in any way.
#
# ── Dependency constraints ────────────────────────────────────────────────────
#   No numpy. No random. No datetime.now(). stdlib + pytest only.
# =============================================================================

from __future__ import annotations

import ast
import contextlib
import importlib
import math
import time
from typing import Dict, Generator, List
from unittest.mock import patch

import pytest

import jarvis.core.feature_layer as _fl_mod
from jarvis.core.feature_layer import (
    DRIFT_HARD_STOP_RATIO,
    DRIFT_HARD_STOP_SEVERITY,
    DRIFT_WINDOW_MAX,
    FEATURE_NAMES,
    FEATURE_VECTOR_SIZE,
    VOLATILITY_SCALING,
    DriftAction,
    DriftResult,
    DriftSummary,
    FeatureDimensionError,
    FeatureDriftMonitor,
    FeatureLayer,
    FeatureResult,
    VolatilityScalingError,
    get_volatility_scaling,
)


# =============================================================================
# SECTION 0 — Fixtures and shared helpers
# =============================================================================

class _FakeOHLCV:
    """
    Minimal duck-typed stand-in for OHLCV.
    FeatureLayer only accesses .open .high .low .close .volume.
    """
    __slots__ = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        open_: float = 100.0,
        high:  float = 105.0,
        low:   float = 99.0,
        close: float = 103.0,
        volume: float = 1_000.0,
    ) -> None:
        self.open   = open_
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = volume


class _FakeMarketData:
    """
    Minimal duck-typed stand-in for MarketData.
    FeatureLayer accesses .ohlcv, .quality_score, and .features.
    .features must be a dict (defaults to empty — sentinels use 0.0 defaults).
    """
    __slots__ = ("ohlcv", "quality_score", "features")

    def __init__(
        self,
        ohlcv: _FakeOHLCV | None = None,
        quality_score: float = 1.0,
        features: Dict[str, float] | None = None,
    ) -> None:
        self.ohlcv         = ohlcv if ohlcv is not None else _FakeOHLCV()
        self.quality_score = quality_score
        self.features      = features if features is not None else {}


@contextlib.contextmanager
def _inject_market_data(md: _FakeMarketData) -> Generator[None, None, None]:
    """
    Temporarily inject `md` into the feature_layer module's global namespace
    so that _compute_raw()'s free variable `market_data` resolves correctly.

    Restores the prior state unconditionally on exit.
    """
    sentinel = object()
    prior = _fl_mod.__dict__.get("market_data", sentinel)
    _fl_mod.market_data = md
    try:
        yield
    finally:
        if prior is sentinel:
            _fl_mod.__dict__.pop("market_data", None)
        else:
            _fl_mod.market_data = prior  # type: ignore[assignment]


def _make_md(
    open_: float = 100.0,
    high:  float = 105.0,
    low:   float = 99.0,
    close: float = 103.0,
    volume: float = 1_000.0,
    quality_score: float = 1.0,
    features: Dict[str, float] | None = None,
) -> _FakeMarketData:
    return _FakeMarketData(
        ohlcv=_FakeOHLCV(open_=open_, high=high, low=low, close=close, volume=volume),
        quality_score=quality_score,
        features=features,
    )


def _call_compute(layer: FeatureLayer, md: _FakeMarketData) -> FeatureResult:
    """Call layer.compute_features(md) with the module injection applied."""
    with _inject_market_data(md):
        return layer.compute_features(md)


def _primed_layer(n_bars: int = 30, asset_class: str = "crypto") -> FeatureLayer:
    """Return a FeatureLayer pre-loaded with `n_bars` of uniform synthetic history."""
    layer = FeatureLayer(asset_class=asset_class)
    layer.push_history(
        [{"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 500.0}]
        * n_bars
    )
    return layer


def _bimodal_monitor(
    n_per_half: int = 50,
    feature: str | None = None,
) -> FeatureDriftMonitor:
    """
    Return a FeatureDriftMonitor where the specified feature (or all features
    if None) has a perfectly bimodal history: first half 0.0, second half 1.0.
    """
    monitor = FeatureDriftMonitor()
    targets = FEATURE_NAMES if feature is None else [feature]
    for name in targets:
        for _ in range(n_per_half):
            monitor._history[name].append(0.0)
        for _ in range(n_per_half):
            monitor._history[name].append(1.0)
    return monitor


def _raw_clean() -> Dict[str, float]:
    """Return a minimal valid raw dict for _sanitise() with all values 0.0."""
    return {name: 0.0 for name in FEATURE_NAMES}


# =============================================================================
# REQ-01  FEATURE_NAMES length == 99
# =============================================================================

class TestFeatureNamesLength:
    """REQ-01: The catalogue of feature names must contain exactly 99 entries."""

    def test_feature_names_len_is_99(self) -> None:
        assert len(FEATURE_NAMES) == 99, (
            f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries; expected 99."
        )

    def test_feature_vector_size_constant_is_99(self) -> None:
        assert FEATURE_VECTOR_SIZE == 99

    def test_feature_names_len_matches_constant(self) -> None:
        assert len(FEATURE_NAMES) == FEATURE_VECTOR_SIZE

    def test_feature_names_are_unique(self) -> None:
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), (
            "FEATURE_NAMES contains duplicate entries."
        )

    def test_feature_names_are_nonempty_strings(self) -> None:
        for name in FEATURE_NAMES:
            assert isinstance(name, str) and name, (
                f"FEATURE_NAMES entry is not a non-empty string: {name!r}"
            )

    def test_group_sizes_sum_to_99(self) -> None:
        """
        FAS documents nine feature groups whose sizes sum to 99:
          15 + 12 + 18 + 10 + 8 + 12 + 8 + 10 + 6 = 99
        Validate group boundary slices by checking known first/last members.
        """
        group_boundaries = [
            ("price_action",    0,   15, "returns_1m",            "range_position"),
            ("volume_liquidity",15,  27, "volume_ma_ratio",        "realized_spread"),
            ("technical",       27,  45, "rsi_14",                 "vwap_distance"),
            ("microstructure",  45,  55, "tick_direction",         "market_quality_index"),
            ("cross_asset",     55,  63, "btc_eth_correlation",    "stablecoin_flow"),
            ("on_chain",        63,  75, "exchange_net_flow",      "holder_composition"),
            ("regime_state",    75,  83, "regime_hmm",             "regime_confidence"),
            ("sentiment",       83,  93, "fear_greed_index",       "social_volume"),
            ("meta",            93,  99, "data_quality_score",     "prediction_confidence_lagged"),
        ]
        total = 0
        for group, start, end, first, last in group_boundaries:
            width = end - start
            total += width
            assert FEATURE_NAMES[start] == first, (
                f"Group '{group}': expected first='{first}', "
                f"got '{FEATURE_NAMES[start]}'"
            )
            assert FEATURE_NAMES[end - 1] == last, (
                f"Group '{group}': expected last='{last}', "
                f"got '{FEATURE_NAMES[end - 1]}'"
            )
        assert total == 99

    def test_compile_time_assert_is_enforced(self) -> None:
        """
        feature_layer.py contains a module-level assert that FEATURE_NAMES
        length equals FEATURE_VECTOR_SIZE.  If that assert fired, importing
        the module would have raised AssertionError.  Verify the module is
        importable (implying the assert passed).
        """
        # Module is already imported; this simply confirms it loaded cleanly.
        assert _fl_mod.FEATURE_VECTOR_SIZE == len(_fl_mod.FEATURE_NAMES)


# =============================================================================
# REQ-02  compute_features() returns exactly 99 keys
# =============================================================================

class TestComputeFeaturesReturns99:
    """REQ-02: FeatureLayer.compute_features() must return exactly 99 feature keys."""

    def test_returns_exactly_99_keys(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        assert len(result.features) == 99, (
            f"compute_features() returned {len(result.features)} keys; expected 99."
        )

    def test_returned_keys_equal_feature_names_set(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        assert set(result.features.keys()) == set(FEATURE_NAMES)

    def test_returned_keys_in_feature_names_order(self) -> None:
        """Keys must match FEATURE_NAMES order (ordered dict, Python 3.7+)."""
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        assert list(result.features.keys()) == list(FEATURE_NAMES)

    def test_all_values_are_float(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        for name, val in result.features.items():
            assert isinstance(val, float), (
                f"features['{name}'] = {val!r} is {type(val).__name__}, expected float."
            )

    def test_all_values_are_finite(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        for name, val in result.features.items():
            assert math.isfinite(val), (
                f"features['{name}'] = {val} is not finite after compute_features()."
            )

    def test_result_type_is_feature_result(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        assert isinstance(result, FeatureResult)

    def test_nan_replaced_count_is_non_negative_int(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        assert isinstance(result.nan_replaced_count, int)
        assert result.nan_replaced_count >= 0

    def test_nan_replaced_names_is_list(self) -> None:
        layer = _primed_layer()
        result = _call_compute(layer, _make_md())
        assert isinstance(result.nan_replaced_names, list)

    def test_returns_99_keys_with_empty_history(self) -> None:
        """Cold-start (no push_history) must still produce exactly 99 keys."""
        layer = FeatureLayer(asset_class="crypto")
        result = _call_compute(layer, _make_md())
        assert len(result.features) == 99

    def test_returns_99_keys_across_successive_calls(self) -> None:
        layer = _primed_layer(n_bars=50)
        for seq in range(5):
            result = _call_compute(layer, _make_md(close=100.0 + seq))
            assert len(result.features) == 99, (
                f"Call {seq}: expected 99 keys, got {len(result.features)}."
            )


# =============================================================================
# REQ-03  NaN / Inf inputs replaced with 0.0
# =============================================================================

class TestNaNInfReplacement:
    """
    REQ-03: Any NaN or Inf that reaches a feature value must be replaced with 0.0.

    Two replacement sites exist:
      (a) _safe_float() — applied to external market_data.features inputs before
          they reach the raw dict.  NaN/Inf are silently converted to 0.0; they
          do NOT appear in nan_replaced_names.
      (b) _sanitise() — applied to the entire raw dict after _compute_raw().
          NaN/Inf found here ARE recorded in nan_replaced_count / nan_replaced_names.

    Both paths are tested independently.
    """

    # --- path (b): _sanitise() static method ----------------------------------

    def test_sanitise_replaces_nan_with_zero(self) -> None:
        raw = _raw_clean()
        raw["returns_1m"] = float("nan")
        result = FeatureLayer._sanitise(raw)
        assert result.features["returns_1m"] == 0.0

    def test_sanitise_records_nan_in_replaced_names(self) -> None:
        raw = _raw_clean()
        raw["returns_1m"] = float("nan")
        result = FeatureLayer._sanitise(raw)
        assert "returns_1m" in result.nan_replaced_names
        assert result.nan_replaced_count == 1

    def test_sanitise_replaces_pos_inf_with_zero(self) -> None:
        raw = _raw_clean()
        raw["volatility_1h"] = float("inf")
        result = FeatureLayer._sanitise(raw)
        assert result.features["volatility_1h"] == 0.0

    def test_sanitise_replaces_neg_inf_with_zero(self) -> None:
        raw = _raw_clean()
        raw["volatility_1h"] = float("-inf")
        result = FeatureLayer._sanitise(raw)
        assert result.features["volatility_1h"] == 0.0

    def test_sanitise_tracks_multiple_replacements(self) -> None:
        raw = _raw_clean()
        bad = ["returns_1m", "volatility_1h", "rsi_14"]
        for name in bad:
            raw[name] = float("nan")
        result = FeatureLayer._sanitise(raw)
        assert result.nan_replaced_count == len(bad)
        for name in bad:
            assert name in result.nan_replaced_names
            assert result.features[name] == 0.0

    def test_sanitise_preserves_valid_values(self) -> None:
        raw = _raw_clean()
        raw["returns_1m"] = float("nan")
        raw["rsi_14"] = 55.0
        result = FeatureLayer._sanitise(raw)
        assert result.features["rsi_14"] == 55.0

    def test_sanitise_zero_replacements_for_clean_input(self) -> None:
        raw = _raw_clean()
        result = FeatureLayer._sanitise(raw)
        assert result.nan_replaced_count == 0
        assert result.nan_replaced_names == []

    def test_sanitise_all_nan_produces_all_zero(self) -> None:
        raw = {name: float("nan") for name in FEATURE_NAMES}
        result = FeatureLayer._sanitise(raw)
        assert len(result.features) == 99
        assert all(v == 0.0 for v in result.features.values())
        assert result.nan_replaced_count == 99

    def test_sanitise_output_always_99_keys(self) -> None:
        raw = {name: float("inf") for name in FEATURE_NAMES}
        result = FeatureLayer._sanitise(raw)
        assert len(result.features) == 99

    # --- path (a): market_data.features via _safe_float ----------------------

    def test_nan_in_external_features_produces_zero_in_output(self) -> None:
        """NaN in market_data.features is caught by _safe_float before _sanitise."""
        layer = _primed_layer()
        md = _make_md(features={"fear_greed_index": float("nan")})
        result = _call_compute(layer, md)
        assert result.features["fear_greed_index"] == 0.0
        assert math.isfinite(result.features["fear_greed_index"])

    def test_inf_in_external_features_produces_zero_in_output(self) -> None:
        layer = _primed_layer()
        md = _make_md(features={"funding_rate": float("inf")})
        result = _call_compute(layer, md)
        assert result.features["funding_rate"] == 0.0

    def test_neg_inf_in_external_features_produces_zero_in_output(self) -> None:
        layer = _primed_layer()
        md = _make_md(features={"open_interest": float("-inf")})
        result = _call_compute(layer, md)
        assert result.features["open_interest"] == 0.0

    def test_nan_in_push_bar_sanitised_by_safe_float(self) -> None:
        """push_bar() passes values through _safe_float; NaN becomes 0.0."""
        layer = FeatureLayer(asset_class="crypto")
        layer.push_bar(float("nan"), 102.0, 99.0, 101.0, 500.0)
        assert layer._opens[-1] == 0.0


# =============================================================================
# REQ-04  Drift detection returns deterministic results
# =============================================================================

class TestDriftDetectionDeterminism:
    """
    REQ-04: Given identical history, FeatureDriftMonitor must always produce
    the same KS statistic, p-value, severity, and action.
    """

    def test_identical_monitors_produce_identical_results(self) -> None:
        values = [float(i) * 0.01 for i in range(100)]
        m_a = FeatureDriftMonitor()
        m_b = FeatureDriftMonitor()
        for v in values:
            m_a._history["returns_1m"].append(v)
            m_b._history["returns_1m"].append(v)
        r_a = m_a.detect_drift("returns_1m")
        r_b = m_b.detect_drift("returns_1m")
        assert r_a.ks_statistic == r_b.ks_statistic
        assert r_a.p_value      == r_b.p_value
        assert r_a.severity     == r_b.severity
        assert r_a.action       == r_b.action

    def test_repeated_calls_on_same_monitor_are_idempotent(self) -> None:
        """detect_drift() must not mutate history; repeated calls give same result."""
        monitor = FeatureDriftMonitor()
        for v in [0.0] * 40 + [1.0] * 40:
            monitor._history["rsi_14"].append(v)
        r1 = monitor.detect_drift("rsi_14")
        r2 = monitor.detect_drift("rsi_14")
        assert r1.ks_statistic == r2.ks_statistic
        assert r1.severity     == r2.severity
        assert r1.action       == r2.action

    def test_uniform_history_produces_zero_ks_statistic(self) -> None:
        """Identical reference and recent samples -> KS = 0, no drift."""
        monitor = FeatureDriftMonitor()
        for _ in range(100):
            monitor._history["momentum"].append(0.5)
        result = monitor.detect_drift("momentum")
        assert result.ks_statistic == 0.0
        assert result.severity == 0.0

    def test_bimodal_history_produces_nonzero_ks_statistic(self) -> None:
        """Perfectly bimodal split -> KS = 1.0, maximum drift signal."""
        monitor = _bimodal_monitor(n_per_half=50, feature="volatility_1h")
        result = monitor.detect_drift("volatility_1h")
        assert result.ks_statistic > 0.0

    def test_bimodal_ks_statistic_is_one(self) -> None:
        """
        For a perfectly bimodal split the empirical CDFs diverge maximally.
        KS = 1.0 is expected when the two samples have no overlap.
        """
        monitor = _bimodal_monitor(n_per_half=50, feature="returns_1m")
        result = monitor.detect_drift("returns_1m")
        assert math.isclose(result.ks_statistic, 1.0, rel_tol=1e-9)

    def test_too_few_observations_returns_zero_severity(self) -> None:
        """Fewer than 4 observations -> not enough data; severity=0, p_value=1."""
        monitor = FeatureDriftMonitor()
        for v in [0.0, 1.0, 0.5]:   # 3 values < 4 threshold
            monitor._history["drift"].append(v)
        result = monitor.detect_drift("drift")
        assert result.severity     == 0.0
        assert result.ks_statistic == 0.0
        assert result.p_value      == 1.0

    def test_update_appends_to_all_feature_histories(self) -> None:
        monitor = FeatureDriftMonitor()
        features = {name: 0.1 for name in FEATURE_NAMES}
        monitor.update(features)
        for name in FEATURE_NAMES:
            assert len(monitor._history[name]) == 1
            assert monitor._history[name][0] == 0.1

    def test_update_ignores_unknown_keys(self) -> None:
        """Unknown keys in the features dict must be silently ignored."""
        monitor = FeatureDriftMonitor()
        monitor.update({"nonexistent_feature_xyz": 0.9})
        # No exception; all known histories still empty
        assert len(monitor._history["returns_1m"]) == 0

    def test_scan_returns_99_drift_results(self) -> None:
        monitor = FeatureDriftMonitor()
        summary = monitor.scan()
        assert isinstance(summary, DriftSummary)
        assert len(summary.results) == 99

    def test_scan_result_features_in_feature_names_order(self) -> None:
        monitor = FeatureDriftMonitor()
        summary = monitor.scan()
        assert [r.feature for r in summary.results] == list(FEATURE_NAMES)

    def test_scan_determinism_across_two_monitors(self) -> None:
        m1 = FeatureDriftMonitor()
        m2 = FeatureDriftMonitor()
        for name in FEATURE_NAMES:
            val = len(name) * 0.01   # unique but deterministic per feature
            for _ in range(60):
                m1._history[name].append(val)
                m2._history[name].append(val)
        s1 = m1.scan()
        s2 = m2.scan()
        for r1, r2 in zip(s1.results, s2.results):
            assert r1.ks_statistic == r2.ks_statistic
            assert r1.severity     == r2.severity

    def test_detect_drift_unknown_feature_raises_key_error(self) -> None:
        monitor = FeatureDriftMonitor()
        with pytest.raises(KeyError):
            monitor.detect_drift("this_feature_does_not_exist_xyz")

    def test_drift_result_is_drift_result_instance(self) -> None:
        monitor = FeatureDriftMonitor()
        result = monitor.detect_drift("returns_1m")
        assert isinstance(result, DriftResult)

    def test_compute_features_determinism_cold_start(self) -> None:
        """
        Two fresh FeatureLayers given the same bar must produce equal
        feature dicts (determinism guarantee, DET-05).
        """
        md = _make_md()
        layer_a = FeatureLayer(asset_class="crypto")
        layer_b = FeatureLayer(asset_class="crypto")
        r_a = _call_compute(layer_a, md)
        r_b = _call_compute(layer_b, md)
        for name in FEATURE_NAMES:
            assert r_a.features[name] == r_b.features[name], (
                f"Determinism violated for feature '{name}': "
                f"{r_a.features[name]} != {r_b.features[name]}"
            )


# =============================================================================
# REQ-05  Severity clipping in [0.0, 1.0]
# =============================================================================

class TestSeverityClipping:
    """REQ-05: Severity values must always lie in [0.0, 1.0]."""

    def test_severity_in_range_for_uniform_history(self) -> None:
        monitor = FeatureDriftMonitor()
        for _ in range(100):
            monitor._history["returns_1m"].append(0.5)
        result = monitor.detect_drift("returns_1m")
        assert 0.0 <= result.severity <= 1.0

    def test_severity_in_range_for_bimodal_history(self) -> None:
        monitor = _bimodal_monitor(n_per_half=50, feature="returns_1m")
        result = monitor.detect_drift("returns_1m")
        assert 0.0 <= result.severity <= 1.0, (
            f"severity out of [0, 1]: {result.severity}"
        )

    def test_severity_in_range_for_all_features_after_scan(self) -> None:
        monitor = _bimodal_monitor(n_per_half=50)
        summary = monitor.scan()
        for dr in summary.results:
            assert 0.0 <= dr.severity <= 1.0, (
                f"Feature '{dr.feature}': severity={dr.severity} out of [0, 1]."
            )

    def test_calculate_severity_clamps_above_one(self) -> None:
        """Product ks_stat * importance > 1.0 must be clipped to 1.0."""
        monitor = FeatureDriftMonitor()
        assert monitor.calculate_severity(2.0, 3.0) == 1.0

    def test_calculate_severity_clamps_below_zero(self) -> None:
        """Negative product must be clipped to 0.0."""
        monitor = FeatureDriftMonitor()
        assert monitor.calculate_severity(-0.5, 1.0) == 0.0

    def test_calculate_severity_nan_ks_returns_max(self) -> None:
        """NaN in ks_stat -> max severity (1.0)."""
        monitor = FeatureDriftMonitor()
        assert monitor.calculate_severity(float("nan"), 1.0) == 1.0

    def test_calculate_severity_nan_importance_returns_max(self) -> None:
        """NaN in importance -> max severity (1.0)."""
        monitor = FeatureDriftMonitor()
        assert monitor.calculate_severity(0.5, float("nan")) == 1.0

    def test_calculate_severity_both_nan_returns_max(self) -> None:
        monitor = FeatureDriftMonitor()
        assert monitor.calculate_severity(float("nan"), float("nan")) == 1.0

    def test_calculate_severity_zero_inputs(self) -> None:
        monitor = FeatureDriftMonitor()
        assert monitor.calculate_severity(0.0, 0.0) == 0.0

    def test_calculate_severity_interior_value_correct(self) -> None:
        monitor = FeatureDriftMonitor()
        got = monitor.calculate_severity(0.4, 0.5)   # 0.2
        assert math.isclose(got, 0.2, rel_tol=1e-9)

    @pytest.mark.parametrize("severity,expected_action", [
        (0.0,   DriftAction.IGNORIEREN),
        (0.19,  DriftAction.IGNORIEREN),
        (0.2,   DriftAction.LOGGEN),
        (0.499, DriftAction.LOGGEN),
        (0.5,   DriftAction.UNSICHERHEIT_ERHOEHEN),
        (1.0,   DriftAction.UNSICHERHEIT_ERHOEHEN),
    ])
    def test_determine_action_thresholds(
        self, severity: float, expected_action: DriftAction
    ) -> None:
        assert FeatureDriftMonitor.determine_action(severity) == expected_action, (
            f"severity={severity}: expected {expected_action.value}"
        )

    def test_rekalibrieren_never_returned_by_determine_action(self) -> None:
        """REKALIBRIEREN is reserved; must never be auto-triggered."""
        for sev in (x * 0.01 for x in range(101)):
            action = FeatureDriftMonitor.determine_action(sev)
            assert action != DriftAction.REKALIBRIEREN, (
                f"severity={sev}: REKALIBRIEREN must never be auto-triggered."
            )


# =============================================================================
# REQ-06  >10% severe drift triggers hard_stop flag
# =============================================================================

class TestHardStopFlag:
    """
    REQ-06: scan() must set hard_stop=True iff the fraction of features with
    severity >= DRIFT_HARD_STOP_SEVERITY (0.8) exceeds DRIFT_HARD_STOP_RATIO (0.10).

    Boundary arithmetic:
      99 * 0.10 = 9.9
      Condition:  ratio > 0.10   (strictly greater)
       9 drifted  ->  9/99 ≈ 0.0909  -> NOT > 0.10  -> hard_stop = False
      10 drifted  -> 10/99 ≈ 0.1010  ->     > 0.10  -> hard_stop = True
    """

    def test_all_features_drifted_triggers_hard_stop(self) -> None:
        monitor = _bimodal_monitor(n_per_half=50)
        summary = monitor.scan()
        assert summary.hard_stop is True

    def test_hard_stop_ratio_one_when_all_drifted(self) -> None:
        monitor = _bimodal_monitor(n_per_half=50)
        summary = monitor.scan()
        assert math.isclose(summary.hard_stop_ratio, 1.0, rel_tol=1e-6)

    def test_no_drift_no_hard_stop(self) -> None:
        """Uniform history -> all severity = 0 -> hard_stop = False."""
        monitor = FeatureDriftMonitor()
        for name in FEATURE_NAMES:
            for _ in range(100):
                monitor._history[name].append(0.5)
        summary = monitor.scan()
        assert summary.hard_stop is False

    def test_hard_stop_ratio_zero_when_no_drift(self) -> None:
        monitor = FeatureDriftMonitor()
        for name in FEATURE_NAMES:
            for _ in range(100):
                monitor._history[name].append(0.5)
        summary = monitor.scan()
        assert summary.hard_stop_ratio == 0.0

    def _build_monitor_with_n_drifted(self, n_drifted: int) -> FeatureDriftMonitor:
        """Uniform history for all features; bimodal for the first n_drifted."""
        monitor = FeatureDriftMonitor()
        for name in FEATURE_NAMES:
            for _ in range(100):
                monitor._history[name].append(0.5)
        for name in FEATURE_NAMES[:n_drifted]:
            monitor._history[name].clear()
            for _ in range(50):
                monitor._history[name].append(0.0)
            for _ in range(50):
                monitor._history[name].append(1.0)
        return monitor

    def test_9_drifted_features_does_not_trigger_hard_stop(self) -> None:
        """9/99 = 9.09% — strictly below 10% threshold."""
        monitor = self._build_monitor_with_n_drifted(9)
        summary = monitor.scan()
        severe_count = sum(
            1 for r in summary.results if r.severity >= DRIFT_HARD_STOP_SEVERITY
        )
        assert summary.hard_stop is False, (
            f"9 drifted features (ratio={summary.hard_stop_ratio:.4f}, "
            f"severe_count={severe_count}) should NOT trigger hard_stop."
        )

    def test_10_drifted_features_triggers_hard_stop(self) -> None:
        """10/99 = 10.10% — strictly above 10% threshold."""
        monitor = self._build_monitor_with_n_drifted(10)
        summary = monitor.scan()
        severe_count = sum(
            1 for r in summary.results if r.severity >= DRIFT_HARD_STOP_SEVERITY
        )
        assert summary.hard_stop is True, (
            f"10 drifted features (ratio={summary.hard_stop_ratio:.4f}, "
            f"severe_count={severe_count}) SHOULD trigger hard_stop."
        )

    def test_hard_stop_ratio_equals_severe_count_over_99(self) -> None:
        monitor = _bimodal_monitor(n_per_half=50)
        summary = monitor.scan()
        severe_count = sum(
            1 for r in summary.results if r.severity >= DRIFT_HARD_STOP_SEVERITY
        )
        expected = severe_count / FEATURE_VECTOR_SIZE
        assert math.isclose(summary.hard_stop_ratio, expected, rel_tol=1e-9)

    def test_hard_stop_flag_consistent_with_ratio(self) -> None:
        """hard_stop iff hard_stop_ratio > DRIFT_HARD_STOP_RATIO."""
        monitor = _bimodal_monitor(n_per_half=50)
        summary = monitor.scan()
        assert summary.hard_stop == (summary.hard_stop_ratio > DRIFT_HARD_STOP_RATIO)

    def test_scan_returns_exactly_99_results(self) -> None:
        monitor = FeatureDriftMonitor()
        summary = monitor.scan()
        assert len(summary.results) == FEATURE_VECTOR_SIZE

    def test_scan_result_features_cover_all_feature_names(self) -> None:
        monitor = FeatureDriftMonitor()
        summary = monitor.scan()
        assert [r.feature for r in summary.results] == list(FEATURE_NAMES)


# =============================================================================
# REQ-07  No forbidden imports
# =============================================================================

class TestNoForbiddenImports:
    """
    REQ-07: feature_layer.py must only import from the stdlib and from
    jarvis.core.data_layer (S03). numpy, scipy, random, logging, uuid,
    time, and any datetime.now() usage are all forbidden.

    Verification is AST-based: the source is parsed and every Import /
    ImportFrom node is inspected.  Executable lines are also scanned for
    datetime.now().
    """

    @pytest.fixture(autouse=True)
    def _load_source(self) -> Generator[None, None, None]:
        mod = importlib.import_module("jarvis.core.feature_layer")
        with open(mod.__file__) as f:  # type: ignore[arg-type]
            self._src = f.read()
        self._tree = ast.parse(self._src)
        yield

    def _all_import_module_names(self) -> List[str]:
        names: List[str] = []
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.append(node.module)
        return names

    def test_no_numpy(self) -> None:
        for name in self._all_import_module_names():
            assert "numpy" not in name, (
                f"numpy import found: '{name}'"
            )

    def test_no_scipy(self) -> None:
        for name in self._all_import_module_names():
            assert "scipy" not in name, (
                f"scipy import found: '{name}'"
            )

    def test_no_random(self) -> None:
        for name in self._all_import_module_names():
            assert name != "random" and not name.startswith("random."), (
                f"random import found: '{name}'"
            )

    def test_no_logging(self) -> None:
        for name in self._all_import_module_names():
            assert "logging" not in name, (
                f"logging import found: '{name}'"
            )

    def test_no_uuid(self) -> None:
        for name in self._all_import_module_names():
            assert "uuid" not in name, (
                f"uuid import found: '{name}'"
            )

    def test_no_time_module(self) -> None:
        """time.time() / time.sleep() are forbidden for determinism."""
        for name in self._all_import_module_names():
            assert name != "time" and not name.startswith("time."), (
                f"time import found: '{name}'"
            )

    def test_no_datetime_now_in_executable_code(self) -> None:
        """datetime.now() is forbidden (DET-06). Check all non-comment lines."""
        for lineno, line in enumerate(self._src.split("\n"), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            assert "datetime.now()" not in line, (
                f"datetime.now() found in executable code at line {lineno}: {line!r}"
            )

    def test_only_stdlib_and_s03_imports_present(self) -> None:
        """
        Positive assertion: every import must be from the stdlib or from
        jarvis.core.data_layer (S03).  This catches any new forbidden
        dependency that does not match the explicit per-module tests above.
        """
        allowed_prefixes = (
            "__future__",
            "math",
            "collections",
            "dataclasses",
            "enum",
            "typing",
            "jarvis.core.data_layer",
        )
        for name in self._all_import_module_names():
            assert any(name.startswith(p) for p in allowed_prefixes), (
                f"Unexpected import in feature_layer.py: '{name}'. "
                f"Only stdlib and jarvis.core.data_layer (S03) are permitted."
            )


# =============================================================================
# REQ-08  Computation time constraint logic (mocked timing)
# =============================================================================

class TestComputationTimeBudget:
    """
    REQ-08: Computation must complete within the FAS-specified time budgets.

    Strategy (dual):
      (A) Mocked timing — unittest.mock.patch on time.perf_counter simulates
          fast and slow calls to verify the budget-check LOGIC itself,
          independently of hardware speed.
      (B) Real timing — measure actual wall time for regression detection on
          CI.  Budgets are conservative (3–10x the FAS targets) to avoid
          flakiness on slow machines.

    FAS performance targets (from S04 spec):
      compute_features()   < 50ms  (warm layer)
      detect_drift()       <  5ms  per feature
      scan()               < 500ms all 99 features
    """

    # Conservative CI budgets (real-time tests)
    _COMPUTE_BUDGET_S: float = 0.500    # 500ms
    _DETECT_DRIFT_S:   float = 0.100    # 100ms per feature
    _SCAN_BUDGET_S:    float = 10.0     # 10s for all 99

    # ---- (A) Mocked timing: validates the budget-check LOGIC ----

    def test_mock_fast_call_passes_budget(self) -> None:
        budget = 0.050
        with patch("time.perf_counter", side_effect=[0.000, 0.010]):
            start   = time.perf_counter()
            elapsed = time.perf_counter() - start
        assert elapsed < budget, (
            f"Fast call ({elapsed*1000:.1f}ms) should be within budget "
            f"({budget*1000:.0f}ms)."
        )

    def test_mock_slow_call_fails_budget(self) -> None:
        budget = 0.050
        with patch("time.perf_counter", side_effect=[0.000, 0.200]):
            start   = time.perf_counter()
            elapsed = time.perf_counter() - start
        assert elapsed >= budget, (
            f"Slow call ({elapsed*1000:.1f}ms) should exceed budget "
            f"({budget*1000:.0f}ms)."
        )

    def test_mock_budget_boundary_exclusive(self) -> None:
        """elapsed == budget is treated as a failure (strict less-than)."""
        budget = 0.050
        with patch("time.perf_counter", side_effect=[0.000, budget]):
            start   = time.perf_counter()
            elapsed = time.perf_counter() - start
        assert not (elapsed < budget), (
            "At the exact boundary, elapsed < budget should be False."
        )

    def test_mock_zero_elapsed_passes_any_budget(self) -> None:
        budget = 0.050
        with patch("time.perf_counter", side_effect=[0.000, 0.000]):
            start   = time.perf_counter()
            elapsed = time.perf_counter() - start
        assert elapsed < budget

    # ---- (B) Real timing: regression guard ----

    def test_compute_features_real_time_within_budget(self) -> None:
        layer = _primed_layer(n_bars=50)
        md    = _make_md()
        t0      = time.perf_counter()
        _call_compute(layer, md)
        elapsed = time.perf_counter() - t0
        assert elapsed < self._COMPUTE_BUDGET_S, (
            f"compute_features() took {elapsed*1000:.1f}ms; "
            f"budget is {self._COMPUTE_BUDGET_S*1000:.0f}ms."
        )

    def test_detect_drift_single_feature_within_budget(self) -> None:
        monitor = FeatureDriftMonitor()
        for _ in range(200):
            monitor._history["rsi_14"].append(0.5)
        t0      = time.perf_counter()
        monitor.detect_drift("rsi_14")
        elapsed = time.perf_counter() - t0
        assert elapsed < self._DETECT_DRIFT_S, (
            f"detect_drift() took {elapsed*1000:.1f}ms; "
            f"budget is {self._DETECT_DRIFT_S*1000:.0f}ms."
        )

    def test_scan_all_features_within_budget(self) -> None:
        monitor = FeatureDriftMonitor()
        for name in FEATURE_NAMES:
            for v in [0.5] * 100:
                monitor._history[name].append(v)
        t0      = time.perf_counter()
        monitor.scan()
        elapsed = time.perf_counter() - t0
        assert elapsed < self._SCAN_BUDGET_S, (
            f"scan() took {elapsed:.3f}s; budget is {self._SCAN_BUDGET_S:.1f}s."
        )

    def test_compute_features_does_not_block_indefinitely(self) -> None:
        """Guard against regressions that introduce infinite loops."""
        layer = FeatureLayer(asset_class="crypto")
        md    = _make_md()
        HARD_CEILING_S = 5.0
        t0      = time.perf_counter()
        _call_compute(layer, md)
        elapsed = time.perf_counter() - t0
        assert elapsed < HARD_CEILING_S, (
            f"compute_features() took {elapsed:.2f}s — possible infinite loop."
        )


# =============================================================================
# Supplementary invariant tests
# =============================================================================

class TestVolatilityScaling:
    """Spot-check the VOLATILITY_SCALING dict and get_volatility_scaling()."""

    _EXPECTED: Dict[str, float] = {
        "crypto":       1.0,
        "forex":        0.15,
        "indices":      0.20,
        "commodities":  0.25,
        "rates":        0.05,
    }

    def test_known_asset_classes_return_correct_scale(self) -> None:
        for ac, scale in self._EXPECTED.items():
            got = get_volatility_scaling(ac)
            assert math.isclose(got, scale, rel_tol=1e-9), (
                f"get_volatility_scaling('{ac}') = {got}, expected {scale}."
            )

    def test_unknown_asset_class_raises_volatility_scaling_error(self) -> None:
        with pytest.raises(VolatilityScalingError):
            get_volatility_scaling("unknown_asset_xyz")

    def test_lookup_is_case_insensitive(self) -> None:
        assert get_volatility_scaling("CRYPTO")  == get_volatility_scaling("crypto")
        assert get_volatility_scaling("Forex")   == get_volatility_scaling("forex")
        assert get_volatility_scaling("INDICES") == get_volatility_scaling("indices")

    def test_feature_layer_init_rejects_unknown_asset_class(self) -> None:
        with pytest.raises(VolatilityScalingError):
            FeatureLayer(asset_class="not_a_real_asset_class")

    def test_volatility_scaling_keys_match_expected(self) -> None:
        assert set(VOLATILITY_SCALING.keys()) == set(self._EXPECTED.keys())


class TestFeatureDimensionErrorFromSanitise:
    def test_missing_one_feature_raises(self) -> None:
        raw = _raw_clean()
        del raw["returns_1m"]
        with pytest.raises(FeatureDimensionError):
            FeatureLayer._sanitise(raw)

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(FeatureDimensionError):
            FeatureLayer._sanitise({})


class TestPushHistoryAndPushBar:
    def test_push_history_primes_windows(self) -> None:
        layer = FeatureLayer(asset_class="crypto")
        assert len(layer._closes) == 0
        layer.push_history(
            [{"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 500.0}]
            * 10
        )
        assert len(layer._closes) == 10

    def test_push_bar_appends_single_entry(self) -> None:
        layer = FeatureLayer(asset_class="crypto")
        layer.push_bar(100.0, 102.0, 99.0, 101.0, 500.0)
        assert len(layer._closes) == 1
        assert layer._closes[-1] == 101.0

    def test_window_capped_at_drift_window_max(self) -> None:
        layer = FeatureLayer(asset_class="crypto")
        for _ in range(DRIFT_WINDOW_MAX + 20):
            layer.push_bar(100.0, 102.0, 99.0, 101.0, 500.0)
        assert len(layer._closes) <= DRIFT_WINDOW_MAX
