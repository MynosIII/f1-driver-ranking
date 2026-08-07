from __future__ import annotations

from typing import Any

from .domain import Component, Entry, HistoricalContext, SimulationConfig
from .shapley import grouped_shapley
from .simulation import all_coalitions, monte_carlo_wins


def decompose_event(entries: list[Entry], context: HistoricalContext, config: SimulationConfig) -> list[dict[str, Any]]:
    coalition_values = all_coalitions(entries, context)
    full = frozenset(Component)
    mc = monte_carlo_wins(entries, context, config)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        values = {coalition: field[entry.driver] for coalition, field in coalition_values.items()}
        shapley = grouped_shapley(values)
        baseline = values[frozenset()]
        xw = values[full]
        neutral_driver = values[full - {Component.DRIVER}]
        neutral_car = values[full - {Component.CAR}]
        neutral_team = values[full - {Component.TEAM}]
        estimate, lower, upper = mc[entry.driver]
        row = {
            "season": context.season,
            "event": context.event,
            "driver": entry.driver,
            "constructor": entry.constructor,
            "team": entry.team,
            "grid": entry.grid,
            "XW": xw,
            "XW_simulated": estimate,
            "XW_lower_95": lower,
            "XW_upper_95": upper,
            "XdW_score": entry.driver_score,
            "XcW_score": entry.car_score,
            "XtW_score": entry.team_score,
            "XdW_probability_contribution": shapley[Component.DRIVER],
            "XcW_probability_contribution": shapley[Component.CAR],
            "XtW_probability_contribution": shapley[Component.TEAM],
            "XW_neutral_baseline": baseline,
            "XW_with_neutral_driver": neutral_driver,
            "XW_with_neutral_car": neutral_car,
            "XW_with_neutral_team": neutral_team,
            "driver_direct_uplift": xw - neutral_driver,
            "car_direct_uplift": xw - neutral_car,
            "team_direct_uplift": xw - neutral_team,
            "driver_above_car_expectation": xw - neutral_driver,
            "analysis_mode": config.mode.value,
            "neutral_reference": config.neutral_reference.value,
            "data_quality": "example_precomputed_capabilities",
            "model_confidence": max(0.0, 1.0 - (entry.driver_uncertainty + entry.car_uncertainty + entry.team_uncertainty) / 3),
        }
        error = xw - baseline - sum(shapley.values())
        row["decomposition_error"] = error
        rows.append(row)
    return rows
