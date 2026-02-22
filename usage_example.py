# usage_example.py
# Minimal usage example for jarvis/portfolio/portfolio_allocator.py.
# This file is not part of the jarvis package. For reference only.

from jarvis.portfolio.portfolio_allocator import allocate_positions

# Inputs
total_capital: float = 100_000.0
exposure_fraction: float = 0.80
asset_prices: dict[str, float] = {
    "BTC":  65_000.00,
    "ETH":   3_200.00,
    "SPY":     520.00,
    "GLD":     185.00,
}

# Compute
positions: dict[str, float] = allocate_positions(
    total_capital=total_capital,
    exposure_fraction=exposure_fraction,
    asset_prices=asset_prices,
)

# Inspect
# allocated_capital = 100_000 * 0.80 = 80_000
# equal_weight_capital = 80_000 / 4 = 20_000
# BTC: 20_000 / 65_000 = 0.30769...
# ETH: 20_000 /  3_200 = 6.25
# SPY: 20_000 /    520 = 38.46153...
# GLD: 20_000 /    185 = 108.10810...

for symbol, size in positions.items():
    print(f"{symbol}: {size:.6f} units")

# Expected output:
# BTC: 0.307692 units
# ETH: 6.250000 units
# SPY: 38.461538 units
# GLD: 108.108108 units

# ValueError examples:
# allocate_positions(-1.0, 0.5, {"BTC": 65000.0})   # total_capital <= 0
# allocate_positions(100.0, 1.5, {"BTC": 65000.0})  # exposure_fraction > 1
# allocate_positions(100.0, 0.5, {})                # empty asset_prices
# allocate_positions(100.0, 0.5, {"BTC": -1.0})     # non-positive price
