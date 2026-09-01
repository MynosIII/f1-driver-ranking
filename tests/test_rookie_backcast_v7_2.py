from __future__ import annotations

import numpy as np
import pandas as pd

from historical_xw.rookie_backcast_v7_2 import (
    RookieBackcastConfigV72,
    build_rookie_backcast_v7_2,
    calendar_k_schedule_v7_2,
)


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    calendars = {2000: 4, 2001: 8}
    for season, rounds in calendars.items():
        for round_number in range(1, rounds + 1):
            drivers = ["veteran_a", "veteran_b"]
            if season == 2001:
                drivers.append("rookie")
            for position, driver_id in enumerate(drivers, start=1):
                rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "position": position,
                        "event": f"{season} Race {round_number}",
                        "driver_id": driver_id,
                        "driver": driver_id,
                        "constructor": f"team_{driver_id}",
                        "constructor_id_f1db": f"team_{driver_id}",
                        "model_entity_id": f"car_{driver_id}",
                        "observed_performance": 1.0 - (position - 1) / (len(drivers) - 1),
                        "rating_eligible": True,
                        "status_class": "finished",
                        "pre_rating": 9000.0,
                        "post_rating": 9000.0,
                        "driver_expected_from_rating": 0.5,
                        "car_suitability_score_v6": 0.0,
                        "layout_key": f"layout-{round_number % 2}",
                        "straight_demand": float(round_number % 2),
                        "altitude_demand": 0.0,
                        "wet_fraction": 0.0,
                        "wind_speed": 5.0,
                        "qualifying_position": float(position),
                        "qualifying_eligible": True,
                    }
                )
    return pd.DataFrame(rows)


def test_calendar_normalization_equalizes_scheduled_season_budgets() -> None:
    schedule = calendar_k_schedule_v7_2(_history())
    assert schedule["calendar_reference_rounds"].eq(6.0).all()
    assert schedule["scheduled_race_k_budget"].nunique() == 1
    assert schedule["scheduled_qualifying_k_budget"].nunique() == 1
    assert np.isclose(schedule["scheduled_race_k_budget"].iloc[0], 24.0 * 6.0)
    assert np.isclose(schedule["scheduled_qualifying_k_budget"].iloc[0], 6.0 * 6.0)
    assert schedule.set_index("season").loc[2000, "calendar_k_multiplier"] == 1.5
    assert schedule.set_index("season").loc[2001, "calendar_k_multiplier"] == 0.75


def test_rookie_anchor_and_opponents_come_from_provisional_same_scale() -> None:
    result = build_rookie_backcast_v7_2(_history())
    rookie = result.debut_estimates.set_index("driver_id").loc["rookie"]
    debut_event = result.provisional_history[
        (result.provisional_history["season"] == 2001)
        & (result.provisional_history["round"] == 1)
    ]
    veteran_median = debut_event.loc[
        debut_event["driver_id"].isin(["veteran_a", "veteran_b"]),
        "provisional_pre_qualifying_rating",
    ].median()
    assert np.isclose(rookie["debut_field_anchor"], veteran_median)
    assert rookie["debut_field_anchor"] != 9000.0
    assert rookie["debut_anchor_source"] == (
        "provisional_active_field_median_same_scale_v7_2"
    )
    assert rookie["rookie_opponent_rating_source"] == "provisional_pre_race_rating_v7_2"
    assert result.car_crossfit["car_signal_basis"].eq(
        "provisional_driver_expected_from_rating_v7_2"
    ).all()


def test_final_run_is_frozen_zero_sum_and_crossfit_safe() -> None:
    result = build_rookie_backcast_v7_2(
        _history(), RookieBackcastConfigV72(maximum_window_seasons=2)
    )
    assert result.audit["race_k_budget_max_error"] < 1e-12
    assert result.audit["qualifying_k_budget_max_error"] < 1e-12
    assert result.audit["rookie_window_rows_used_in_car_fit"] == 0
    assert result.audit["frozen_debut_rating_mismatches"] == 0
    assert result.audit["final_replay_count"] == 1
    assert result.audit["provisional_qualifying_zero_sum_max_error"] < 1e-10
    assert result.audit["provisional_race_zero_sum_max_error"] < 1e-10
    assert result.audit["qualifying_zero_sum_max_error"] < 1e-10
    assert result.audit["race_zero_sum_max_error"] < 1e-10
    assert result.retrospective_history["calendar_k_multiplier"].notna().all()


def test_every_1950_debut_uses_the_common_inaugural_anchor() -> None:
    history = _history().query("season == 2001").copy()
    history = history[~(history["driver_id"].eq("rookie") & history["round"].lt(3))]
    history["season"] = 1950
    result = build_rookie_backcast_v7_2(history)
    assert result.debut_estimates["debut_field_anchor"].eq(1500.0).all()
    assert np.isclose(result.debut_estimates["rookie_offset_shrunk"].mean(), 0.0)
