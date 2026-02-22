import dataclasses

import pytest

from jarvis.core.data_layer import VALID_ASSET_CLASSES
from jarvis.core.risk_layer import (
    PositionSpec,
    RiskNumericalError,
    RiskValidationError,
    Side,
)


def _valid_kwargs():
    return dict(
        symbol="ETH-USD",
        asset_class="crypto",
        side=Side.SHORT,
        entry_price=2000.0,
        current_price=1900.0,
        quantity=5.0,
        max_position_usd=20_000.0,
    )


class TestPositionSpecHappyPath:
    def test_construction_succeeds(self):
        spec = PositionSpec(**_valid_kwargs())
        assert spec.symbol == "ETH-USD"
        assert spec.side is Side.SHORT

    def test_frozen(self):
        spec = PositionSpec(**_valid_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.quantity = 99.0  # type: ignore

    def test_all_asset_classes_accepted(self):
        for ac in VALID_ASSET_CLASSES:
            kwargs = _valid_kwargs()
            kwargs["asset_class"] = ac
            spec = PositionSpec(**kwargs)
            assert spec.asset_class == ac

    def test_both_sides_accepted(self):
        for side in Side:
            kwargs = _valid_kwargs()
            kwargs["side"] = side
            spec = PositionSpec(**kwargs)
            assert spec.side is side


class TestPositionSpecSymbolValidation:
    def test_empty_symbol_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), "symbol": ""})
        assert exc_info.value.field_name == "symbol"

    def test_non_string_symbol_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "symbol": 123})  # type: ignore

    def test_non_ascii_symbol_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "symbol": "BTC\u20ac"})


class TestPositionSpecAssetClassValidation:
    def test_unknown_asset_class_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), "asset_class": "real_estate"})
        assert exc_info.value.field_name == "asset_class"

    def test_empty_asset_class_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "asset_class": ""})

    def test_uppercase_asset_class_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "asset_class": "CRYPTO"})


class TestPositionSpecSideValidation:
    def test_string_side_raises(self):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), "side": "LONG"})  # type: ignore
        assert exc_info.value.field_name == "side"

    def test_none_side_raises(self):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), "side": None})  # type: ignore


class TestPositionSpecFloatValidation:
    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd",
    ])
    def test_nan_raises_numerical_error(self, field):
        with pytest.raises(RiskNumericalError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), field: float("nan")})
        assert exc_info.value.field_name == field

    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd",
    ])
    def test_inf_raises_numerical_error(self, field):
        with pytest.raises(RiskNumericalError):
            PositionSpec(**{**_valid_kwargs(), field: float("inf")})

    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd",
    ])
    def test_zero_raises_validation_error(self, field):
        with pytest.raises(RiskValidationError) as exc_info:
            PositionSpec(**{**_valid_kwargs(), field: 0.0})
        assert exc_info.value.field_name == field

    @pytest.mark.parametrize("field", [
        "entry_price", "current_price", "quantity", "max_position_usd",
    ])
    def test_negative_raises_validation_error(self, field):
        with pytest.raises(RiskValidationError):
            PositionSpec(**{**_valid_kwargs(), field: -0.001})

    def test_very_small_positive_quantity_is_valid(self):
        spec = PositionSpec(**{**_valid_kwargs(), "quantity": 1e-10})
        assert spec.quantity == 1e-10

    def test_very_large_price_is_valid(self):
        spec = PositionSpec(**{**_valid_kwargs(), "entry_price": 1e15})
        assert spec.entry_price == 1e15
