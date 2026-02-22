import pytest
from jarvis.core.risk_layer import (
    PortfolioState,
    PositionSpec,
    RiskParameters,
    Side,
)


@pytest.fixture
def valid_position_spec() -> PositionSpec:
    """Minimal valid PositionSpec. All fields at their smallest valid values."""
    return PositionSpec(
        symbol="BTC-USD",
        asset_class="crypto",
        side=Side.LONG,
        entry_price=50_000.0,
        current_price=51_000.0,
        quantity=0.1,
        max_position_usd=10_000.0,
    )


@pytest.fixture
def valid_portfolio_state() -> PortfolioState:
    """Minimal valid PortfolioState with no drawdown (nav == peak_nav)."""
    return PortfolioState(
        nav=1_000_000.0,
        gross_exposure_usd=200_000.0,
        net_exposure_usd=150_000.0,
        open_positions=3,
        peak_nav=1_000_000.0,
        realized_drawdown_pct=0.0,
        current_step=100,
    )


@pytest.fixture
def valid_portfolio_state_with_drawdown() -> PortfolioState:
    """PortfolioState with a 5% drawdown from peak."""
    return PortfolioState(
        nav=950_000.0,
        gross_exposure_usd=100_000.0,
        net_exposure_usd=100_000.0,
        open_positions=1,
        peak_nav=1_000_000.0,
        realized_drawdown_pct=0.05,
        current_step=500,
    )


@pytest.fixture
def valid_risk_parameters() -> RiskParameters:
    """
    Conservative but valid RiskParameters.
    soft_warn=0.05, hard_stop=0.10 (5% gap -- clear ordering).
    """
    return RiskParameters(
        max_position_pct_nav=0.05,
        max_gross_exposure_pct=1.5,
        max_drawdown_hard_stop=0.10,
        max_drawdown_soft_warn=0.05,
        volatility_target_ann=0.15,
        liquidity_haircut_floor=0.2,
        max_open_positions=10,
        kelly_fraction=0.25,
    )
