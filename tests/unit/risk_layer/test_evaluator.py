import dataclasses

import pytest

from jarvis.core.risk_layer import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    RiskValidationError,
    RiskVerdict,
    Side,
)
from jarvis.core.risk_layer.evaluator import (
    RiskDecision,
    evaluate_portfolio_risk,
    evaluate_position_risk,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

def _params(
    soft_warn: float = 0.05,
    hard_stop: float = 0.10,
) -> RiskParameters:
    """Return a valid RiskParameters with configurable drawdown thresholds."""
    return RiskParameters(
        max_position_pct_nav=0.05,
        max_gross_exposure_pct=1.5,
        max_drawdown_hard_stop=hard_stop,
        max_drawdown_soft_warn=soft_warn,
        volatility_target_ann=0.15,
        liquidity_haircut_floor=0.2,
        max_open_positions=10,
        kelly_fraction=0.25,
    )


def _portfolio(nav: float, peak_nav: float) -> PortfolioState:
    """
    Return a PortfolioState with the given nav and peak_nav.
    realized_drawdown_pct is derived from the two values.
    """
    drawdown = max(0.0, 1.0 - (nav / peak_nav))
    return PortfolioState(
        nav=nav,
        gross_exposure_usd=0.0,
        net_exposure_usd=0.0,
        open_positions=0,
        peak_nav=peak_nav,
        realized_drawdown_pct=drawdown,
        current_step=0,
    )


def _position(asset_class: str = "crypto") -> PositionSpec:
    """Return a valid PositionSpec for the given asset class."""
    return PositionSpec(
        symbol="BTC-USD",
        asset_class=asset_class,
        side=Side.LONG,
        entry_price=50_000.0,
        current_price=50_000.0,
        quantity=1.0,
        max_position_usd=50_000.0,
    )


# =============================================================================
# SECTION 1 -- RiskDecision: structure and immutability
# =============================================================================

class TestRiskDecisionStructure:
    """RiskDecision is a frozen dataclass with exactly four fields."""

    def test_construction_succeeds(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        assert rd.verdict is RiskVerdict.APPROVE

    def test_messages_is_tuple(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=("msg1", "msg2"),
            max_position_size=None,
            requires_rebalance=False,
        )
        assert isinstance(rd.messages, tuple)

    def test_empty_messages_tuple(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        assert rd.messages == ()

    def test_max_position_size_none(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        assert rd.max_position_size is None

    def test_max_position_size_float(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=10_000.0,
            requires_rebalance=False,
        )
        assert rd.max_position_size == 10_000.0

    def test_requires_rebalance_false(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        assert rd.requires_rebalance is False

    def test_requires_rebalance_true(self):
        rd = RiskDecision(
            verdict=RiskVerdict.HALT,
            messages=(),
            max_position_size=None,
            requires_rebalance=True,
        )
        assert rd.requires_rebalance is True

    def test_frozen_verdict_immutable(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rd.verdict = RiskVerdict.HALT  # type: ignore

    def test_frozen_messages_immutable(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rd.messages = ("new",)  # type: ignore

    def test_frozen_max_position_size_immutable(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rd.max_position_size = 999.0  # type: ignore

    def test_frozen_requires_rebalance_immutable(self):
        rd = RiskDecision(
            verdict=RiskVerdict.APPROVE,
            messages=(),
            max_position_size=None,
            requires_rebalance=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rd.requires_rebalance = True  # type: ignore

    def test_all_verdicts_accepted(self):
        for v in RiskVerdict:
            rd = RiskDecision(
                verdict=v,
                messages=(),
                max_position_size=None,
                requires_rebalance=False,
            )
            assert rd.verdict is v


# =============================================================================
# SECTION 2 -- evaluate_position_risk: verdict branches
# =============================================================================

class TestEvaluatePositionRiskVerdicts:
    """
    Drawdown threshold derivation:
        hard_stop_nav = peak_nav * (1 - hard_stop)
        soft_warn_nav = peak_nav * (1 - soft_warn)

    With peak_nav=1_000_000, soft_warn=0.05, hard_stop=0.10:
        hard_stop_nav = 900_000
        soft_warn_nav = 950_000

    Boundary table:
        nav = 1_000_000  -> APPROVE  (no drawdown)
        nav = 950_001    -> APPROVE  (just above soft warn threshold)
        nav = 950_000    -> REDUCE   (exactly at soft warn threshold)
        nav = 900_001    -> REDUCE   (just above hard stop threshold)
        nav = 900_000    -> HALT     (exactly at hard stop threshold)
        nav = 800_000    -> HALT     (well below hard stop)
    """

    PEAK = 1_000_000.0
    PARAMS = _params(soft_warn=0.05, hard_stop=0.10)
    POS = _position()

    def test_ok_no_drawdown(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.APPROVE

    def test_ok_just_above_soft_warn(self):
        pf = _portfolio(nav=950_001.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.APPROVE

    def test_soft_warn_exactly_at_threshold(self):
        # nav == soft_warn_nav (950_000): <= condition fires -> REDUCE
        pf = _portfolio(nav=950_000.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.REDUCE

    def test_soft_warn_between_thresholds(self):
        pf = _portfolio(nav=925_000.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.REDUCE

    def test_soft_warn_just_above_hard_stop(self):
        pf = _portfolio(nav=900_001.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.REDUCE

    def test_hard_stop_exactly_at_threshold(self):
        # nav == hard_stop_nav (900_000): hard stop fires first -> HALT
        pf = _portfolio(nav=900_000.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.HALT

    def test_hard_stop_well_below_threshold(self):
        pf = _portfolio(nav=500_000.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.HALT

    def test_hard_stop_takes_priority_over_soft_warn(self):
        """HALT must be returned even though nav also breaches soft_warn_nav."""
        pf = _portfolio(nav=800_000.0, peak_nav=self.PEAK)
        rd = evaluate_position_risk(self.POS, pf, self.PARAMS)
        # Both thresholds are breached; HALT must win.
        assert rd.verdict is RiskVerdict.HALT


class TestEvaluatePositionRiskOutputShape:
    """RiskDecision fields returned by evaluate_position_risk in Phase 7B."""

    def test_messages_is_empty_tuple(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        rd = evaluate_position_risk(_position(), pf, _params())
        assert rd.messages == ()
        assert isinstance(rd.messages, tuple)

    def test_max_position_size_is_none(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        rd = evaluate_position_risk(_position(), pf, _params())
        assert rd.max_position_size is None

    def test_requires_rebalance_is_false(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        rd = evaluate_position_risk(_position(), pf, _params())
        assert rd.requires_rebalance is False


# =============================================================================
# SECTION 3 -- evaluate_position_risk: invalid asset class
# =============================================================================

class TestEvaluatePositionRiskAssetClass:
    """
    evaluate_position_risk raises RiskValidationError for unknown asset classes.

    Note: PositionSpec.__post_init__ already rejects unknown asset classes,
    so the only way to hit the evaluator's asset class check with a bad value
    is via a crafted PositionSpec (using object.__setattr__ to bypass
    frozen protection). We test the evaluator's guard directly by
    constructing a position through the normal API with each valid class,
    and separately verifying the guard fires via the domain validation path.
    """

    def test_all_valid_asset_classes_pass(self):
        from jarvis.core.data_layer import VALID_ASSET_CLASSES
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        params = _params()
        for ac in VALID_ASSET_CLASSES:
            pos = _position(asset_class=ac)
            rd = evaluate_position_risk(pos, pf, params)
            assert rd.verdict is RiskVerdict.APPROVE

    def test_invalid_asset_class_raises_at_domain_boundary(self):
        """
        PositionSpec raises RiskValidationError for unknown asset classes.
        This confirms the domain boundary enforces the constraint before
        the evaluator is ever reached.
        """
        with pytest.raises(RiskValidationError) as exc_info:
            _position(asset_class="real_estate")
        assert exc_info.value.field_name == "asset_class"


# =============================================================================
# SECTION 4 -- evaluate_portfolio_risk: verdict branches
# =============================================================================

class TestEvaluatePortfolioRiskVerdicts:
    """
    Same threshold arithmetic as position risk, but without a position input.

    Boundary table (peak=1_000_000, soft=0.05, hard=0.10):
        nav = 1_000_000  -> APPROVE
        nav = 950_001    -> APPROVE
        nav = 950_000    -> REDUCE
        nav = 900_001    -> REDUCE
        nav = 900_000    -> HALT
        nav = 100_000    -> HALT
    """

    PEAK = 1_000_000.0
    PARAMS = _params(soft_warn=0.05, hard_stop=0.10)

    def test_ok_no_drawdown(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=self.PEAK)
        rd = evaluate_portfolio_risk(pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.APPROVE

    def test_ok_just_above_soft_warn(self):
        pf = _portfolio(nav=950_001.0, peak_nav=self.PEAK)
        rd = evaluate_portfolio_risk(pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.APPROVE

    def test_soft_warn_exactly_at_threshold(self):
        pf = _portfolio(nav=950_000.0, peak_nav=self.PEAK)
        rd = evaluate_portfolio_risk(pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.REDUCE

    def test_soft_warn_midpoint(self):
        pf = _portfolio(nav=925_000.0, peak_nav=self.PEAK)
        rd = evaluate_portfolio_risk(pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.REDUCE

    def test_hard_stop_exactly_at_threshold(self):
        pf = _portfolio(nav=900_000.0, peak_nav=self.PEAK)
        rd = evaluate_portfolio_risk(pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.HALT

    def test_hard_stop_well_below(self):
        pf = _portfolio(nav=100_000.0, peak_nav=self.PEAK)
        rd = evaluate_portfolio_risk(pf, self.PARAMS)
        assert rd.verdict is RiskVerdict.HALT


class TestEvaluatePortfolioRiskOutputShape:
    """RiskDecision fields returned by evaluate_portfolio_risk in Phase 7B."""

    def test_messages_is_empty_tuple(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        rd = evaluate_portfolio_risk(pf, _params())
        assert rd.messages == ()
        assert isinstance(rd.messages, tuple)

    def test_max_position_size_is_none(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        rd = evaluate_portfolio_risk(pf, _params())
        assert rd.max_position_size is None

    def test_requires_rebalance_is_false(self):
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        rd = evaluate_portfolio_risk(pf, _params())
        assert rd.requires_rebalance is False


# =============================================================================
# SECTION 5 -- Determinism
# =============================================================================

class TestDeterminism:
    """
    Identical inputs must always produce identical outputs.
    No hidden state, no randomness, no clock dependence.
    """

    def test_evaluate_position_risk_is_deterministic(self):
        pf = _portfolio(nav=950_000.0, peak_nav=1_000_000.0)
        params = _params()
        pos = _position()
        results = [evaluate_position_risk(pos, pf, params) for _ in range(10)]
        assert all(r.verdict is results[0].verdict for r in results)

    def test_evaluate_portfolio_risk_is_deterministic(self):
        pf = _portfolio(nav=950_000.0, peak_nav=1_000_000.0)
        params = _params()
        results = [evaluate_portfolio_risk(pf, params) for _ in range(10)]
        assert all(r.verdict is results[0].verdict for r in results)

    def test_position_and_portfolio_agree_on_verdict(self):
        """
        evaluate_position_risk and evaluate_portfolio_risk share the same
        drawdown logic. For any given (portfolio, params) pair, both functions
        must return the same verdict (modulo asset class errors, which produce
        exceptions rather than verdicts).
        """
        for nav in (1_000_000.0, 950_000.0, 900_000.0, 800_000.0):
            pf = _portfolio(nav=nav, peak_nav=1_000_000.0)
            params = _params()
            pos = _position()
            rd_pos = evaluate_position_risk(pos, pf, params)
            rd_pf = evaluate_portfolio_risk(pf, params)
            assert rd_pos.verdict is rd_pf.verdict, (
                f"Verdict mismatch at nav={nav}: "
                f"position={rd_pos.verdict}, portfolio={rd_pf.verdict}"
            )

    def test_different_params_produce_different_verdicts(self):
        """Changing params changes the outcome -- no stale caching."""
        pf = _portfolio(nav=960_000.0, peak_nav=1_000_000.0)
        # With soft_warn=0.05 (warn at 950k): nav=960k -> APPROVE
        params_loose = _params(soft_warn=0.05, hard_stop=0.10)
        # With soft_warn=0.02 (warn at 980k): nav=960k -> REDUCE
        params_tight = _params(soft_warn=0.02, hard_stop=0.10)
        assert evaluate_portfolio_risk(pf, params_loose).verdict is RiskVerdict.APPROVE
        assert evaluate_portfolio_risk(pf, params_tight).verdict is RiskVerdict.REDUCE

    def test_inputs_are_not_mutated_by_position_eval(self):
        """evaluate_position_risk must not modify any of its inputs."""
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        params = _params()
        pos = _position()
        nav_before = pf.nav
        peak_before = pf.peak_nav
        drawdown_before = pf.realized_drawdown_pct
        evaluate_position_risk(pos, pf, params)
        assert pf.nav == nav_before
        assert pf.peak_nav == peak_before
        assert pf.realized_drawdown_pct == drawdown_before

    def test_inputs_are_not_mutated_by_portfolio_eval(self):
        """evaluate_portfolio_risk must not modify any of its inputs."""
        pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
        params = _params()
        nav_before = pf.nav
        evaluate_portfolio_risk(pf, params)
        assert pf.nav == nav_before


# =============================================================================
# SECTION 6 -- Boundary arithmetic precision
# =============================================================================

class TestBoundaryArithmetic:
    """
    Test the exact threshold boundary: nav == threshold is a BREACH (<=).
    Test nav == threshold + epsilon is NOT a breach (> threshold).
    """

    def test_soft_warn_boundary_inclusive(self):
        """nav exactly equal to soft_warn_nav is a breach (REDUCE or HALT)."""
        peak = 1_000_000.0
        # soft_warn=0.05 -> soft_warn_nav=950_000
        pf = _portfolio(nav=950_000.0, peak_nav=peak)
        rd = evaluate_portfolio_risk(pf, _params(soft_warn=0.05, hard_stop=0.10))
        assert rd.verdict in (RiskVerdict.REDUCE, RiskVerdict.HALT)

    def test_one_above_soft_warn_is_approve(self):
        """nav = soft_warn_nav + 1 is not a breach."""
        peak = 1_000_000.0
        pf = _portfolio(nav=950_001.0, peak_nav=peak)
        rd = evaluate_portfolio_risk(pf, _params(soft_warn=0.05, hard_stop=0.10))
        assert rd.verdict is RiskVerdict.APPROVE

    def test_hard_stop_boundary_inclusive(self):
        """nav exactly equal to hard_stop_nav is a HALT."""
        peak = 1_000_000.0
        pf = _portfolio(nav=900_000.0, peak_nav=peak)
        rd = evaluate_portfolio_risk(pf, _params(soft_warn=0.05, hard_stop=0.10))
        assert rd.verdict is RiskVerdict.HALT

    def test_one_above_hard_stop_is_reduce(self):
        """nav = hard_stop_nav + 1 falls in the soft warn band."""
        peak = 1_000_000.0
        pf = _portfolio(nav=900_001.0, peak_nav=peak)
        rd = evaluate_portfolio_risk(pf, _params(soft_warn=0.05, hard_stop=0.10))
        assert rd.verdict is RiskVerdict.REDUCE

    def test_nav_equal_peak_nav_is_always_approve(self):
        """Zero drawdown must always be APPROVE regardless of params."""
        for soft, hard in [(0.01, 0.02), (0.05, 0.10), (0.20, 0.30)]:
            pf = _portfolio(nav=1_000_000.0, peak_nav=1_000_000.0)
            rd = evaluate_portfolio_risk(pf, _params(soft_warn=soft, hard_stop=hard))
            assert rd.verdict is RiskVerdict.APPROVE, (
                f"Expected APPROVE at zero drawdown with soft={soft}, hard={hard}"
            )
