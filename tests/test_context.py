import pandas as pd

from historical_xw.context import circuit_passability, regulation_era


def test_regulation_eras_are_historical_not_one_hardcoded_label() -> None:
    assert regulation_era(1950) == "1950-1960_front_engine"
    assert regulation_era(2010) == "2006-2013_v8"
    assert regulation_era(2024) == "2022-2025_ground_effect"
    assert regulation_era(2026) == "2026_present_active_aero"


def test_circuit_passability_uses_only_prior_same_circuit_races() -> None:
    events = pd.DataFrame(
        [
            {"season": 2020, "round": 1, "circuit_id": "x"},
            {"season": 2021, "round": 1, "circuit_id": "x"},
        ]
    )
    results = pd.DataFrame(
        [
            {"season": 2020, "round": 1, "driver_id": "a", "grid": 3, "position": 1, "status": "Finished"},
            {"season": 2020, "round": 1, "driver_id": "b", "grid": 1, "position": 2, "status": "Finished"},
            {"season": 2021, "round": 1, "driver_id": "a", "grid": 1, "position": 1, "status": "Finished"},
            {"season": 2021, "round": 1, "driver_id": "b", "grid": 2, "position": 2, "status": "Finished"},
        ]
    )
    contexts = circuit_passability(events, results)
    first, second = contexts.iloc[0], contexts.iloc[1]
    assert first["circuit_history_races"] == 0
    assert second["circuit_history_races"] == 1
    assert second["overtake_source"] == "prior_same_circuit_grid_movement"
