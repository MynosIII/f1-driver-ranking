from __future__ import annotations

import math
from typing import Mapping

from .domain import Component
from .simulation import Coalition


def grouped_shapley(values: Mapping[Coalition, float]) -> dict[Component, float]:
    """Exact three-player grouped Shapley allocation in probability points."""
    players = frozenset(Component)
    if set(values) != {frozenset(s) for r in range(4) for s in __import__("itertools").combinations(players, r)}:
        raise ValueError("All eight component coalitions are required")
    n = len(players)
    result: dict[Component, float] = {}
    for player in players:
        contribution = 0.0
        others = players - {player}
        for coalition in values:
            if player in coalition or not coalition.issubset(others):
                continue
            weight = math.factorial(len(coalition)) * math.factorial(n - len(coalition) - 1) / math.factorial(n)
            contribution += weight * (values[coalition | {player}] - values[coalition])
        result[player] = contribution
    return result

