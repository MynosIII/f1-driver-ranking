from __future__ import annotations

from typing import Any

from .domain import Component, Entry, HistoricalContext, SimulationConfig
from .shapley import grouped_shapley
from .simulation import all_coalitions, all_performance_coalitions, monte_carlo_wins


def decompose_event(entries: list[Entry], context: HistoricalContext, config: SimulationConfig) -> list[dict[str, Any]]:
    coalition_values = all_coalitions(entries, context)
    performance_values = all_performance_coalitions(entries, context)
    full = frozenset(Component)
    mc = monte_carlo_wins(entries, context, config)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        values = {coalition: field[entry.identity] for coalition, field in coalition_values.items()}
        shapley = grouped_shapley(values)
        xp_values = {coalition: field[entry.identity] for coalition, field in performance_values.items()}
        xp_shapley = grouped_shapley(xp_values)
        baseline = values[frozenset()]
        xw = values[full]
        xp = xp_values[full]
        neutral_driver = values[full - {Component.DRIVER}]
        neutral_car = values[full - {Component.CAR}]
        neutral_team = values[full - {Component.TEAM}]
        estimate, lower, upper = mc[entry.identity]
        row = {
            "season": context.season,
            "event": context.event,
            "driver": entry.driver,
            "driver_id": entry.identity,
            "constructor": entry.constructor,
            "team": entry.team,
            "grid": entry.grid,
            "XW": xw,
            "XP": xp,
            "expected_position": 1.0 + (len(entries) - 1.0) * (1.0 - xp),
            "XW_simulated": estimate,
            "XW_lower_95": lower,
            "XW_upper_95": upper,
            "XdW_score": entry.driver_score,
            "XcW_score": entry.car_score,
            "XtW_score": entry.team_score,
            "XdW_probability_contribution": shapley[Component.DRIVER],
            "XcW_probability_contribution": shapley[Component.CAR],
            "XtW_probability_contribution": shapley[Component.TEAM],
            "XdP_performance_contribution": xp_shapley[Component.DRIVER],
            "XcP_performance_contribution": xp_shapley[Component.CAR],
            "XtP_performance_contribution": xp_shapley[Component.TEAM],
            "XP_neutral_baseline": xp_values[frozenset()],
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
            "regulation_era": context.regulation_era,
            "race_distance_laps": context.race_distance_laps,
            "wet_fraction": context.wet_fraction,
            "track_temperature": context.track_temperature,
            "air_temperature": context.air_temperature,
            "wind_speed": context.wind_speed,
            "circuit_id": context.circuit_id,
            "overtake_index": context.circuit_features.get("overtake_index", 0.5),
            "weather_source": context.weather_source,
            "data_quality": "example_precomputed_capabilities",
            "model_confidence": max(0.0, 1.0 - (entry.driver_uncertainty + entry.car_uncertainty + entry.team_uncertainty) / 3),
        }
        error = xw - baseline - sum(shapley.values())
        row["decomposition_error"] = error
        row["xp_decomposition_error"] = xp - xp_values[frozenset()] - sum(xp_shapley.values())
        rows.append(row)
    return rows
