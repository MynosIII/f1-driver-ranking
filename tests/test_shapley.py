from itertools import combinations

import pytest

from historical_xw.domain import Component
from historical_xw.shapley import grouped_shapley


def test_shapley_efficiency_with_interactions() -> None:
    players = frozenset(Component)
    values = {}
    for size in range(4):
        for coalition in combinations(players, size):
            group = frozenset(coalition)
            base = sum({Component.DRIVER: 0.1, Component.CAR: 0.2, Component.TEAM: -0.03}[p] for p in group)
            interaction = 0.06 if {Component.DRIVER, Component.CAR}.issubset(group) else 0.0
            values[group] = 0.05 + base + interaction
    result = grouped_shapley(values)
    assert values[players] == pytest.approx(values[frozenset()] + sum(result.values()))
    assert result[Component.DRIVER] == pytest.approx(0.13)
    assert result[Component.CAR] == pytest.approx(0.23)

