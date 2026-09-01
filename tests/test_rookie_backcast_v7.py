from __future__ import annotations

import numpy as np
import pandas as pd

from historical_xw.rookie_backcast_v7 import (
    RookieBackcastConfigV7,
    build_rookie_backcast_v7,
    cross_fitted_car_expectations_v7,
    identify_rookie_windows_v7,
)


def _history() -> pd.DataFrame:
    rows = []
    for round_number in range(1, 9):
        for position, driver_id in enumerate(("veteran_a", "veteran_b", "rookie"), start=1):
            mechanical = driver_id == "rookie" and round_number in {1, 2}
            crash = driver_id == "rookie" and round_number == 3
            eligible = not mechanical
            observed = np.nan if mechanical else (0.0 if crash else 1.0 - (position - 1) / 2)
            rows.append(
                {
                    "season": 2000,
                    "round": round_number,
                    "position": position,
                    "event": f"Race {round_number}",
                    "driver_id": driver_id,
                    "driver": driver_id,
                    "constructor": "team_r" if driver_id == "rookie" else f"team_{driver_id[-1]}",
                    "constructor_id_f1db": "team_r"
                    if driver_id == "rookie"
                    else f"team_{driver_id[-1]}",
                    "model_entity_id": "car_r"
                    if driver_id == "rookie"
                    else f"car_{driver_id[-1]}",
                    "observed_performance": observed,
                    "rating_eligible": eligible,
                    "status_class": "mechanical" if mechanical else ("driver_fault" if crash else "finished"),
                    "pre_rating": 1500.0,
                    "post_rating": 1500.0,
                    "driver_expected_from_rating": 0.5,
                    "car_suitability_score_v6": 0.0,
                    "layout_key": f"layout-{round_number % 2}",
                    "straight_demand": float(round_number % 2),
                    "altitude_demand": 0.0,
                    "wet_fraction": 1.0 if round_number == 4 else 0.0,
                    "wind_speed": 5.0,
                    "qualifying_position": float(position),
                    "qualifying_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_mechanical_results_do_not_consume_rookie_slots_but_crash_does() -> None:
    windows = identify_rookie_windows_v7(_history())
    rookie = windows[windows["driver_id"].eq("rookie")].iloc[0]
    assert rookie["rookie_eligible_races"] == 5
    selected = _history().loc[rookie["rookie_window_indices"]]
    assert selected["round"].tolist() == [3, 4, 5, 6, 7]
    assert selected.iloc[0]["status_class"] == "driver_fault"


def test_rookie_window_rows_never_enter_its_crossfit_car_signal() -> None:
    history = _history()
    windows = identify_rookie_windows_v7(history)
    crossfit = cross_fitted_car_expectations_v7(history, windows)
    assert not crossfit["rookie_window_row_used_in_car_fit"].any()
    assert len(crossfit[crossfit["driver_id"].eq("rookie")]) == 5


def test_two_pass_run_centres_1950_and_freezes_debut_ratings() -> None:
    history = _history()
    history["season"] = 1950
    late_driver = history.iloc[-1].copy()
    late_driver["driver_id"] = "late_1950_driver"
    late_driver["driver"] = "late_1950_driver"
    late_driver["pre_rating"] = 1650.0
    late_driver["post_rating"] = 1650.0
    history = pd.concat([history, late_driver.to_frame().T], ignore_index=True)
    result = build_rookie_backcast_v7(
        history,
        RookieBackcastConfigV7(maximum_window_seasons=1),
    )
    assert np.isclose(result.debut_estimates["rookie_offset_shrunk"].mean(), 0.0)
    assert result.debut_estimates["debut_field_anchor"].eq(1500.0).all()
    assert result.audit["rookie_window_rows_used_in_car_fit"] == 0
    assert result.audit["frozen_debut_rating_mismatches"] == 0
    assert result.audit["qualifying_zero_sum_max_error"] < 1e-10
    assert result.audit["race_zero_sum_max_error"] < 1e-10


def test_returning_driver_is_not_created_as_a_second_rookie() -> None:
    history = _history()
    return_row = history[history["driver_id"].eq("veteran_a")].iloc[-1].copy()
    return_row["season"] = 2003
    return_row["round"] = 1
    return_row["event"] = "Return"
    history = pd.concat([history, return_row.to_frame().T], ignore_index=True)
    windows = identify_rookie_windows_v7(history)
    assert windows[windows["driver_id"].eq("veteran_a")].shape[0] == 1


def test_one_off_driver_uses_population_prior_without_individual_weight() -> None:
    history = _history()
    one_off = history.iloc[-1].copy()
    one_off["season"] = 2001
    one_off["round"] = 1
    one_off["event"] = "One off"
    one_off["driver_id"] = "one_off"
    one_off["driver"] = "one_off"
    history = pd.concat([history, one_off.to_frame().T], ignore_index=True)
    result = build_rookie_backcast_v7(history)
    estimate = result.debut_estimates[result.debut_estimates["driver_id"].eq("one_off")].iloc[0]
    assert estimate["rookie_reliability"] == 0.0
    assert estimate["debut_rating_method"] == "one_off_population_prior"
