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
    grid_effect = -0.085 * (entry.grid - 1)
    wet_driver = context.wet_fraction * 0.16 * driver
    heat_car = max(context.track_temperature - 30.0, 0.0) * 0.006 * car
    operations = (context.safety_car_laps / max(context.race_distance_laps, 1)) * 0.35 * team
    interactions = 0.10 * driver * car + 0.07 * car * team + 0.05 * driver * team
    return grid_effect + 0.72 * driver + 0.90 * car + 0.52 * team + wet_driver + heat_car + operations + interactions


def coalition_probabilities(
    entries: list[Entry], context: HistoricalContext, coalition: Coalition
) -> dict[str, float]:
    """Field-normalized Plackett-Luce win probabilities for one coalition."""
    logits = np.asarray([_strength(e, coalition, context) for e in entries], dtype=float)
    exp = np.exp(logits - logits.max())
    probabilities = exp / exp.sum()
    return {entry.driver: float(p) for entry, p in zip(entries, probabilities, strict=True)}


def all_coalitions(entries: list[Entry], context: HistoricalContext) -> dict[Coalition, dict[str, float]]:
    components = tuple(Component)
    result: dict[Coalition, dict[str, float]] = {}
    for mask in product((False, True), repeat=3):
        coalition = frozenset(c for c, active in zip(components, mask, strict=True) if active)
        result[coalition] = coalition_probabilities(entries, context, coalition)
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
        entry.driver: (float(prob), float(max(0, prob - 1.96 * err)), float(min(1, prob + 1.96 * err)))
        for entry, prob, err in zip(entries, p, se, strict=True)
    }


def _sample_full_order(entries: list[Entry], context: HistoricalContext, coalition: Coalition, rng: np.random.Generator) -> list[int]:
    """One Plackett-Luce draw of a complete finishing order: repeatedly pick
    a winner among whoever's left (same softmax as ``coalition_probabilities``),
    remove them, repeat. Returns indices into ``entries``, best-to-worst.
    """
    remaining = list(range(len(entries)))
    order: list[int] = []
    while remaining:
        subset = [entries[i] for i in remaining]
        probabilities = np.asarray(list(coalition_probabilities(subset, context, coalition).values()))
        pick = rng.choice(len(remaining), p=probabilities)
        order.append(remaining.pop(pick))
    return order


def monte_carlo_full_ranking(
    entries: list[Entry],
    context: HistoricalContext,
    config: SimulationConfig,
    coalition: Coalition | None = None,
    resample_capabilities: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Expected finishing percentile (1.0 = always wins, 0.0 = always last)
    for every entry, from full Plackett-Luce order sampling rather than
    just a winner draw -- the position-based analogue of ``monte_carlo_wins``.

    ``resample_capabilities=False`` skips the capability-uncertainty
    resampling (used for the seven reduced coalitions in a Shapley
    decomposition, where a point estimate is wanted, not a confidence
    interval); ``True`` (the default) matches ``monte_carlo_wins``'s
    behavior for the full coalition.
    """
    coalition = coalition if coalition is not None else frozenset(Component)
    rng = np.random.default_rng(config.seed)
    n = len(entries)
    total = np.zeros(n)
    total_sq = np.zeros(n)
    for _ in range(config.simulations):
        sampled = (
            [
                replace(
                    e,
                    driver_score=float(rng.normal(e.driver_score, e.driver_uncertainty)),
                    car_score=float(rng.normal(e.car_score, e.car_uncertainty)),
                    team_score=float(rng.normal(e.team_score, e.team_uncertainty)),
                )
                for e in entries
            ]
            if resample_capabilities
            else entries
        )
        order = _sample_full_order(sampled, context, coalition, rng)
        for rank, idx in enumerate(order):
            percentile = 1.0 - rank / (n - 1) if n > 1 else 1.0
            total[idx] += percentile
            total_sq[idx] += percentile * percentile
    mean = total / config.simulations
    variance = np.maximum(total_sq / config.simulations - mean**2, 0.0)
    se = np.sqrt(variance / config.simulations)
    return {
        entry.driver: (float(m), float(max(0.0, m - 1.96 * s)), float(min(1.0, m + 1.96 * s)))
        for entry, m, s in zip(entries, mean, se, strict=True)
    }


def all_coalitions_expected_percentile(
    entries: list[Entry], context: HistoricalContext, config: SimulationConfig
) -> dict[Coalition, dict[str, float]]:
    """Point-estimate expected finishing percentile for all eight coalitions
    (no capability resampling -- these feed the Shapley decomposition, which
    needs point values, not intervals; the full coalition's own uncertainty
    is separately quantified by ``monte_carlo_full_ranking``)."""
    components = tuple(Component)
    result: dict[Coalition, dict[str, float]] = {}
    for mask in product((False, True), repeat=3):
        coalition = frozenset(c for c, active in zip(components, mask, strict=True) if active)
        percentiles = monte_carlo_full_ranking(entries, context, config, coalition, resample_capabilities=False)
        result[coalition] = {driver: mean for driver, (mean, _, _) in percentiles.items()}
    return result
