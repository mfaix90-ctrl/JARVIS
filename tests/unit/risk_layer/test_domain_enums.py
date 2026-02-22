import pytest

from jarvis.core.risk_layer import RiskVerdict, Side


class TestSide:
    def test_long_value(self):
        assert Side.LONG == "LONG"

    def test_short_value(self):
        assert Side.SHORT == "SHORT"

    def test_exactly_two_members(self):
        assert set(Side) == {Side.LONG, Side.SHORT}

    def test_str_subclass(self):
        assert isinstance(Side.LONG, str)

    def test_from_string(self):
        assert Side("LONG") is Side.LONG
        assert Side("SHORT") is Side.SHORT

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            Side("INVALID")


class TestRiskVerdict:
    def test_all_five_members_present(self):
        members = {v.value for v in RiskVerdict}
        assert members == {"APPROVE", "REDUCE", "HOLD", "HALT", "REJECT"}

    def test_str_subclass(self):
        assert isinstance(RiskVerdict.APPROVE, str)
