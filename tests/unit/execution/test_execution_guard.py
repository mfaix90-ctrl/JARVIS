import dataclasses

from jarvis.core.execution_guard import ExecutionOrder, build_execution_order
from jarvis.core.risk_layer import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    Side,
)


# =============================================================================
# SHARED HELPERS
# =============================================================================
#
# Nav=1_000_000, max_position_pct_nav=0.05 -> max_allowed=50_000.
# Soft warn threshold: peak * (1 - 0.05) = 950_000.
# Hard stop threshold: peak * (1 - 0.10) = 900_000.
#
# APPROVE:  nav=1_000_000  (above soft warn)
# REDUCE:   nav=925_000    (between soft warn and hard stop)
# HALT:     nav=900_000    (exactly at hard stop)
# =============================================================================

_PEAK_NAV    = 1_000_000.0
_SOFT_WARN   = 0.05
_HARD_STOP   = 0.10
_POS_PCT_NAV = 0.05          # max_allowed = nav * 0.05


def _params() -> RiskParameters:
    # Phase 7D integration: kelly_fraction=1.0 and liquidity_haircut_floor=0.4625
    # ensure cap-only sizing semantics for these integration tests.
    # REDUCE clamped: floor = requested * 0.4625 wins when requested > cap.
    # Not-clamped: kelly*target dominates floor.
    return RiskParameters(
        max_position_pct_nav=_POS_PCT_NAV,
        max_gross_exposure_pct=1.5,
        max_drawdown_hard_stop=_HARD_STOP,
        max_drawdown_soft_warn=_SOFT_WARN,
        volatility_target_ann=0.15,
        liquidity_haircut_floor=0.4625,
        max_open_positions=10,
        kelly_fraction=1.0,
    )


def _portfolio(nav: float) -> PortfolioState:
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


def _position(
    quantity: float = 1.0,
    current_price: float = 10_000.0,
    symbol: str = "BTC-USD",
    side: Side = Side.LONG,
) -> PositionSpec:
    """requested_notional = quantity * current_price."""
    return PositionSpec(
        symbol=symbol,
        asset_class="crypto",
        side=side,
        entry_price=current_price,
        current_price=current_price,
        quantity=quantity,
        max_position_usd=current_price * quantity * 2,
    )


# =============================================================================
# SECTION 1 -- APPROVE flow
# nav=1_000_000 -> APPROVE
# requested=10_000 < max_allowed=50_000 -> no clamping
# =============================================================================

class TestApproveFlow:

    def test_returns_execution_order(self):
        result = build_execution_order(
            _position(quantity=1.0, current_price=10_000.0),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert isinstance(result, ExecutionOrder)

    def test_result_is_not_none(self):
        result = build_execution_order(
            _position(),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result is not None

    def test_symbol_echoed_from_position(self):
        result = build_execution_order(
            _position(symbol="ETH-USD"),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result.symbol == "ETH-USD"

    def test_side_echoed_from_position(self):
        result = build_execution_order(
            _position(side=Side.SHORT),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result.side is Side.SHORT

    def test_target_notional_equals_requested_when_below_cap(self):
        # requested = 1.0 * 10_000 = 10_000; max_allowed = 50_000
        result = build_execution_order(
            _position(quantity=1.0, current_price=10_000.0),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert result.target_notional == 10_000.0

    def test_execution_order_is_frozen(self):
        result = build_execution_order(
            _position(),
            _portfolio(nav=1_000_000.0),
            _params(),
        )
        assert isinstance(result, ExecutionOrder)
        try:
            result.symbol = "TAMPERED"  # type: ignore
            assert False, "FrozenInstanceError expected"
        except dataclasses.FrozenInstanceError:
            pass


# =============================================================================
# SECTION 2 -- REDUCE flow
# nav=925_000 -> between soft warn (950_000) and hard stop (900_000) -> REDUCE
# requested=100_000 > max_allowed=925_000*0.05=46_250 -> clamped
# =============================================================================

class TestReduceFlow:

    def test_returns_execution_order(self):
        result = build_execution_order(
            _position(quantity=10.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert isinstance(result, ExecutionOrder)

    def test_result_is_not_none(self):
        result = build_execution_order(
            _position(quantity=10.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert result is not None

    def test_target_notional_clamped_to_cap(self):
        # max_allowed = 925_000 * 0.05 = 46_250
        result = build_execution_order(
            _position(quantity=10.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert result.target_notional == 925_000.0 * _POS_PCT_NAV

    def test_symbol_and_side_echoed(self):
        result = build_execution_order(
            _position(symbol="SOL-USD", side=Side.SHORT,
                      quantity=10.0, current_price=10_000.0),
            _portfolio(nav=925_000.0),
            _params(),
        )
        assert result.symbol == "SOL-USD"
        assert result.side is Side.SHORT


# =============================================================================
# SECTION 3 -- HALT flow
# nav=900_000 -> exactly at hard stop -> HALT -> None
# =============================================================================

class TestHaltFlow:

    def test_returns_none(self):
        result = build_execution_order(
            _position(),
            _portfolio(nav=900_000.0),
            _params(),
        )
        assert result is None

    def test_returns_none_well_below_hard_stop(self):
        result = build_execution_order(
            _position(),
            _portfolio(nav=500_000.0),
            _params(),
        )
        assert result is None

    def test_no_execution_order_constructed_on_halt(self):
        result = build_execution_order(
            _position(),
            _portfolio(nav=900_000.0),
            _params(),
        )
        assert not isinstance(result, ExecutionOrder)


# =============================================================================
# SECTION 4 -- Determinism
# =============================================================================

class TestDeterminism:

    def test_approve_twenty_identical_calls(self):
        pos = _position(quantity=1.0, current_price=10_000.0)
        pf  = _portfolio(nav=1_000_000.0)
        p   = _params()
        results = [build_execution_order(pos, pf, p) for _ in range(20)]
        assert all(r is not None for r in results)
        assert all(r.symbol          == results[0].symbol          for r in results)
        assert all(r.side            is results[0].side            for r in results)
        assert all(r.target_notional == results[0].target_notional for r in results)

    def test_reduce_twenty_identical_calls(self):
        pos = _position(quantity=10.0, current_price=10_000.0)
        pf  = _portfolio(nav=925_000.0)
        p   = _params()
        results = [build_execution_order(pos, pf, p) for _ in range(20)]
        assert all(r is not None for r in results)
        assert all(r.target_notional == results[0].target_notional for r in results)

    def test_halt_twenty_identical_calls(self):
        pos = _position()
        pf  = _portfolio(nav=900_000.0)
        p   = _params()
        results = [build_execution_order(pos, pf, p) for _ in range(20)]
        assert all(r is None for r in results)

    def test_changing_nav_changes_outcome(self):
        pos = _position(quantity=10.0, current_price=10_000.0)
        p   = _params()
        approve = build_execution_order(pos, _portfolio(nav=1_000_000.0), p)
        halt    = build_execution_order(pos, _portfolio(nav=900_000.0),   p)
        assert approve is not None
        assert halt    is None


# =============================================================================
# SECTION 5 -- Inputs not mutated
# =============================================================================

class TestInputsNotMutated:

    def test_position_spec_unchanged_after_approve(self):
        pos = _position(quantity=2.0, current_price=15_000.0)
        qty_before    = pos.quantity
        price_before  = pos.current_price
        symbol_before = pos.symbol
        build_execution_order(pos, _portfolio(nav=1_000_000.0), _params())
        assert pos.quantity      == qty_before
        assert pos.current_price == price_before
        assert pos.symbol        == symbol_before

    def test_portfolio_state_unchanged_after_approve(self):
        pf = _portfolio(nav=1_000_000.0)
        nav_before      = pf.nav
        peak_before     = pf.peak_nav
        drawdown_before = pf.realized_drawdown_pct
        build_execution_order(_position(), pf, _params())
        assert pf.nav                   == nav_before
        assert pf.peak_nav              == peak_before
        assert pf.realized_drawdown_pct == drawdown_before

    def test_risk_parameters_unchanged_after_approve(self):
        p = _params()
        pct_before  = p.max_position_pct_nav
        hard_before = p.max_drawdown_hard_stop
        soft_before = p.max_drawdown_soft_warn
        build_execution_order(_position(), _portfolio(nav=1_000_000.0), p)
        assert p.max_position_pct_nav   == pct_before
        assert p.max_drawdown_hard_stop == hard_before
        assert p.max_drawdown_soft_warn == soft_before

    def test_position_spec_unchanged_after_halt(self):
        pos = _position(quantity=5.0, current_price=20_000.0)
        qty_before = pos.quantity
        build_execution_order(pos, _portfolio(nav=900_000.0), _params())
        assert pos.quantity == qty_before

    def test_portfolio_state_unchanged_after_halt(self):
        pf = _portfolio(nav=900_000.0)
        nav_before = pf.nav
        build_execution_order(_position(), pf, _params())
        assert pf.nav == nav_before
