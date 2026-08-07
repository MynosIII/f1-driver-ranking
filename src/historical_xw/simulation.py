from __future__ import annotations

from dataclasses import replace
from itertools import product
import numpy as np

from .domain import Component, Entry, HistoricalContext, SimulationConfig

Coalition = frozenset[Component]


def _strength(entry: Entry, coalition: Coalition, context: HistoricalContext) -> float:
    """Latent race strength. Context is fixed across all counterfactual coalitions."""
    driver = entry.driver_score if Component.DRIVER in coalition else 0.0
    car = entry.car_score if Component.CAR in coalition else 0.0
    team = entry.team_score if Component.TEAM in coalition else 0.0
    overtake_index = float(np.clip(context.circuit_features.get("overtake_index", 0.5), 0, 1))
    # Grid matters more when prior editions show little passing and less at passable tracks.
    grid_coefficient = 0.125 - 0.085 * overtake_index
    grid_effect = -grid_coefficient * (entry.grid - 1)
    wet_driver = context.wet_fraction * 0.16 * driver
    heat_car = max(context.track_temperature - 30.0, 0.0) * 0.006 * car
    distance_factor = np.log1p(max(context.race_distance_laps, 1)) / np.log1p(60)
    operations = (
        (context.safety_car_laps / max(context.race_distance_laps, 1)) * 0.35
        + 0.08 * distance_factor
    ) * team
    interactions = 0.10 * driver * car + 0.07 * car * team + 0.05 * driver * team
    return grid_effect + 0.72 * driver + 0.90 * car + 0.52 * team + wet_driver + heat_car + operations + interactions


def coalition_probabilities(
    entries: list[Entry], context: HistoricalContext, coalition: Coalition
) -> dict[str, float]:
    """Field-normalized Plackett-Luce win probabilities for one coalition."""
    logits = np.asarray([_strength(e, coalition, context) for e in entries], dtype=float)
    exp = np.exp(logits - logits.max())
    probabilities = exp / exp.sum()
    return {entry.identity: float(p) for entry, p in zip(entries, probabilities, strict=True)}


def coalition_expected_performance(
    entries: list[Entry], context: HistoricalContext, coalition: Coalition
) -> dict[str, float]:
    """Expected normalized finish (xP): 1 is first, 0 is last, .5 is field average."""

    logits = np.asarray([_strength(e, coalition, context) for e in entries], dtype=float)
    if len(entries) == 1:
        return {entries[0].identity: 1.0}
    differences = logits[:, None] - logits[None, :]
    pairwise = 1.0 / (1.0 + np.exp(-np.clip(differences, -700, 700)))
    np.fill_diagonal(pairwise, np.nan)
    expected = np.nanmean(pairwise, axis=1)
    return {entry.identity: float(value) for entry, value in zip(entries, expected, strict=True)}


def all_coalitions(entries: list[Entry], context: HistoricalContext) -> dict[Coalition, dict[str, float]]:
    components = tuple(Component)
    result: dict[Coalition, dict[str, float]] = {}
    for mask in product((False, True), repeat=3):
        coalition = frozenset(c for c, active in zip(components, mask, strict=True) if active)
        result[coalition] = coalition_probabilities(entries, context, coalition)
    return result


def all_performance_coalitions(
    entries: list[Entry], context: HistoricalContext
) -> dict[Coalition, dict[str, float]]:
    components = tuple(Component)
    result: dict[Coalition, dict[str, float]] = {}
    for mask in product((False, True), repeat=3):
        coalition = frozenset(c for c, active in zip(components, mask, strict=True) if active)
        result[coalition] = coalition_expected_performance(entries, context, coalition)
    return result


def monte_carlo_wins(
    entries: list[Entry], context: HistoricalContext, config: SimulationConfig
) -> dict[str, tuple[float, float, float]]:
    """Sample latent capabilities and return probability plus a 95% simulation interval."""
    rng = np.random.default_rng(config.seed)
    wins = np.zeros(len(entries), dtype=int)
    for _ in range(config.simulations):
        sampled = [
            replace(
                e,
                driver_score=float(rng.normal(e.driver_score, e.driver_uncertainty)),
                car_score=float(rng.normal(e.car_score, e.car_uncertainty)),
                team_score=float(rng.normal(e.team_score, e.team_uncertainty)),
            )
            for e in entries
        ]
        probabilities = np.asarray(
            list(coalition_probabilities(sampled, context, frozenset(Component)).values())
        )
        wins[rng.choice(len(entries), p=probabilities)] += 1
    p = wins / config.simulations
    se = np.sqrt(p * (1 - p) / config.simulations)
    return {
        entry.identity: (float(prob), float(max(0, prob - 1.96 * err)), float(min(1, prob + 1.96 * err)))
        for entry, prob, err in zip(entries, p, se, strict=True)
    }
