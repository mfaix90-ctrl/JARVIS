from jarvis.core.risk_layer import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    Side,
)
from jarvis.core.risk_layer.engine import assess_trade


# =============================================================================
# SHARED HELPERS
# =============================================================================
#
# Nav=1_000_000, max_position_pct_nav=0.05 -> max_allowed=50_000.
# Soft warn threshold: peak * (1 - 0.05) = 950_000.
# Hard stop threshold: peak * (1 - 0.10) = 900_000.
#
# APPROVE:  nav >= 950_001  (above soft warn)
# REDUCE:   950_000 >= nav > 900_000  (between soft warn and hard stop)
# HALT:     nav <= 900_000  (at or below hard stop)
# =============================================================================

_PEAK_NAV     = 1_000_000.0
_SOFT_WARN    = 0.05
_HARD_STOP    = 0.10
_POS_PCT_NAV  = 0.05          # max_allowed = 50_000


def _params(max_position_pct_nav: float = _POS_PCT_NAV) -> RiskParameters:
    # Phase 7D integration: kelly_fraction=1.0 and liquidity_haircut_floor=0.4625
    # ensure cap-only sizing semantics for these integration tests.
    # REDUCE clamped: floor = 100_000 * 0.4625 = 46_250 == cap (floor wins).
    # REDUCE/APPROVE not-clamped: kelly*target > floor -> target unchanged.
    return RiskParameters(
        max_position_pct_nav=max_position_pct_nav,
        max_gross_exposure_pct=1.5,
        max_drawdown_hard_stop=_HARD_STOP,
        max_drawdown_soft_warn=_SOFT_WARN,
        volatility_target_ann=0.15,
        liquidity_haircut_floor=0.4625,
        max_open_positions=10,
        kelly_fraction=1.0,
    )


def _portfolio(nav: float) -> PortfolioState:
    """PortfolioState with peak_nav fixed at _PEAK_NAV."""
    drawdown = max(0.0, 1.0 - (nav / _PEAK_NAV))
    return PortfolioState(
        nav=nav,
        gross_exposure_usd=0.0,
        net_exposure_usd=0.0,
        open_positions=0,
        peak_nav=_PEAK_NAV,
        realized_drawdown_pct=drawdown,
        current_step=0,
    )


def _position(quantity: float = 1.0, current_price: float = 10_000.0) -> PositionSpec:
    """requested_notional = quantity * current_price."""
    return PositionSpec(
        symbol="BTC-USD",
        asset_class="crypto",
        side=Side.LONG,
        entry_price=current_price,
        current_price=current_price,
        quantity=quantity,
        max_position_usd=current_price * quantity * 2,
    )


# =============================================================================
# SECTION 1 -- APPROVE flow
# nav=1_000_000 -> above soft warn -> APPROVE
# requested=10_000 < max_allowed=50_000 -> no clamping
# =============================================================================

class TestApproveFlow:

    def test_allowed_is_true(self):
        result = assess_trade(
            _position(quantity=1.0, current_price=10_000.0),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result.allowed is True

    def test_target_notional_equals_requested_when_below_cap(self):
        # requested = 1.0 * 10_000 = 10_000; max_allowed = 50_000
        result = assess_trade(
            _position(quantity=1.0, current_price=10_000.0),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result.target_notional == 10_000.0

    def test_target_notional_clamped_when_above_cap(self):
        # requested = 10.0 * 10_000 = 100_000; max_allowed = 50_000
        result = assess_trade(
            _position(quantity=10.0, current_price=10_000.0),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result.target_notional == 50_000.0

    def test_returns_position_sizing_result(self):
        from jarvis.core.risk_layer import PositionSizingResult
        result = assess_trade(
            _position(),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert isinstance(result, PositionSizingResult)


# =============================================================================
# SECTION 2 -- REDUCE flow
# nav=925_000 -> between soft warn (950_000) and hard stop (900_000) -> REDUCE
# Phase 7C: REDUCE cap arithmetic identical to APPROVE
# =============================================================================

class TestReduceFlow:

    def test_allowed_is_true(self):
        result = assess_trade(
            _position(quantity=1.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert result.allowed is True

    def test_target_notional_clamped_to_cap(self):
        # requested = 10.0 * 10_000 = 100_000
        # max_allowed = 925_000 * 0.05 = 46_250
        result = assess_trade(
            _position(quantity=10.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert result.target_notional == 925_000.0 * _POS_PCT_NAV

    def test_target_notional_equals_requested_when_below_cap(self):
        # requested = 1.0 * 10_000 = 10_000
        # max_allowed = 925_000 * 0.05 = 46_250 -> no clamp
        result = assess_trade(
            _position(quantity=1.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert result.target_notional == 10_000.0


# =============================================================================
# SECTION 3 -- HALT flow
# nav=900_000 -> exactly at hard stop threshold -> HALT
# =============================================================================

class TestHaltFlow:

    def test_allowed_is_false(self):
        result = assess_trade(
            _position(),
            _portfolio(nav=900_000.0),
            _params(),
        )
        assert result.allowed is False

    def test_target_notional_is_none(self):
        result = assess_trade(
            _position(),
            _portfolio(nav=900_000.0),
            _params(),
        )
        assert result.target_notional is None

    def test_allowed_is_false_well_below_hard_stop(self):
        result = assess_trade(
            _position(),
            _portfolio(nav=500_000.0),
            _params(),
        )
        assert result.allowed is False
        assert result.target_notional is None


# =============================================================================
# SECTION 4 -- Determinism
# =============================================================================

class TestDeterminism:

    def test_approve_twenty_identical_calls(self):
        pos = _position(quantity=1.0, current_price=10_000.0)
        pf  = _portfolio(nav=1_000_000.0)
        p   = _params()
        results = [assess_trade(pos, pf, p) for _ in range(20)]
        assert all(r.allowed          == results[0].allowed          for r in results)
        assert all(r.target_notional  == results[0].target_notional  for r in results)
        assert all(r.reason           is results[0].reason           for r in results)

    def test_reduce_twenty_identical_calls(self):
        pos = _position(quantity=1.0, current_price=10_000.0)
        pf  = _portfolio(nav=925_000.0)
        p   = _params()
        results = [assess_trade(pos, pf, p) for _ in range(20)]
        assert all(r.allowed          == results[0].allowed          for r in results)
        assert all(r.target_notional  == results[0].target_notional  for r in results)

    def test_halt_twenty_identical_calls(self):
        pos = _position()
        pf  = _portfolio(nav=900_000.0)
        p   = _params()
        results = [assess_trade(pos, pf, p) for _ in range(20)]
        assert all(r.allowed         == results[0].allowed         for r in results)
        assert all(r.target_notional == results[0].target_notional for r in results)

    def test_different_nav_produces_different_result(self):
        pos = _position(quantity=10.0, current_price=10_000.0)
        p   = _params()
        result_approve = assess_trade(pos, _portfolio(nav=1_000_000.0), p)
        result_reduce  = assess_trade(pos, _portfolio(nav=925_000.0),   p)
        # Both allowed, but different caps -> different target_notional
        assert result_approve.target_notional != result_reduce.target_notional


# =============================================================================
# SECTION 5 -- Inputs not mutated
# =============================================================================

class TestInputsNotMutated:

    def test_position_spec_unchanged_after_approve(self):
        pos = _position(quantity=2.0, current_price=15_000.0)
        qty_before   = pos.quantity
        price_before = pos.current_price
        symbol_before = pos.symbol
        assess_trade(pos, _portfolio(nav=1_000_000.0), _params())
        assert pos.quantity      == qty_before
        assert pos.current_price == price_before
        assert pos.symbol        == symbol_before

    def test_portfolio_state_unchanged_after_approve(self):
        pf = _portfolio(nav=1_000_000.0)
        nav_before      = pf.nav
        peak_before     = pf.peak_nav
        drawdown_before = pf.realized_drawdown_pct
        assess_trade(_position(), pf, _params())
        assert pf.nav                   == nav_before
        assert pf.peak_nav              == peak_before
        assert pf.realized_drawdown_pct == drawdown_before

    def test_risk_parameters_unchanged_after_approve(self):
        p = _params(max_position_pct_nav=0.05)
        pct_before  = p.max_position_pct_nav
        hard_before = p.max_drawdown_hard_stop
        soft_before = p.max_drawdown_soft_warn
        assess_trade(_position(), _portfolio(nav=1_000_000.0), p)
        assert p.max_position_pct_nav   == pct_before
        assert p.max_drawdown_hard_stop == hard_before
        assert p.max_drawdown_soft_warn == soft_before

    def test_position_spec_unchanged_after_halt(self):
        pos = _position(quantity=3.0, current_price=20_000.0)
        qty_before = pos.quantity
        assess_trade(pos, _portfolio(nav=900_000.0), _params())
        assert pos.quantity == qty_before

    def test_portfolio_state_unchanged_after_halt(self):
        pf = _portfolio(nav=900_000.0)
        nav_before = pf.nav
        assess_trade(_position(), pf, _params())
        assert pf.nav == nav_before
