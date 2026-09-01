from __future__ import annotations

import numpy as np
import pandas as pd

from historical_xw.rookie_backcast_v7_5 import build_annual_anchor_rating_v7_5
from historical_xw.season_scope_v7_5 import filter_historical_indianapolis_v7_5


def _rating_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for season in (2000, 2001):
        for round_number in (1, 2, 3):
            drivers = ["a", "b", "c"] if season == 2000 else ["a", "b", "rookie"]
            if season == 2001 and round_number >= 2:
                drivers.append("midseason_rookie")
            for position, driver_id in enumerate(drivers, start=1):
                rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "position": position,
                        "event": f"Race {season}-{round_number}",
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
                        "layout_key": f"layout-{round_number}",
                        "straight_demand": 0.5,
                        "altitude_demand": 0.0,
                        "wet_fraction": 0.0,
                        "wind_speed": 5.0,
                        "qualifying_position": float(position),
                        "qualifying_eligible": True,
                    }
                )
    return pd.DataFrame(rows)


def test_historical_indy_filter_keeps_modern_indianapolis_and_non_indy_careers() -> None:
    history = pd.DataFrame(
        [
            {"season": 1950, "round": 3, "event": "Indianapolis 500", "driver_id": "only"},
            {"season": 1950, "round": 3, "event": "Indianapolis 500", "driver_id": "both"},
            {"season": 1950, "round": 4, "event": "Swiss Grand Prix", "driver_id": "both"},
            {
                "season": 2005,
                "round": 9,
                "event": "United States Grand Prix",
                "driver_id": "modern",
            },
        ]
    )
    result = filter_historical_indianapolis_v7_5(history)
    assert set(result.history["driver_id"]) == {"both", "modern"}
    assert result.excluded_driver_ids == ("only",)
    assert result.audit["historical_indianapolis_events_removed"] == 1
    assert result.audit["modern_indianapolis_events_removed"] == 0


def test_all_rookies_in_a_season_use_previous_full_field_terminal_median() -> None:
    result = build_annual_anchor_rating_v7_5(_rating_history())
    provisional = result.provisional_history
    prior_terminal = (
        provisional.query("season == 2000")
        .sort_values("event_index")
        .groupby("driver_id")
        .tail(1)
    )
    expected_anchor = float(prior_terminal["provisional_post_rating"].median())
    debut = result.debut_estimates.set_index("driver_id")
    assert np.isclose(debut.loc["rookie", "debut_field_anchor"], expected_anchor)
    assert np.isclose(
        debut.loc["midseason_rookie", "debut_field_anchor"], expected_anchor
    )
    first_2001 = provisional.query("season == 2001 and round == 1")
    active_survivor_median = float(
        first_2001.loc[
            first_2001["driver_id"].isin(["a", "b"]),
            "provisional_pre_qualifying_rating",
        ].median()
    )
    assert not np.isclose(expected_anchor, active_survivor_median)
    assert result.audit["annual_anchor_blend_coefficient"] is None
    assert result.audit["annual_anchor_target_level"] is None
    assert result.audit["annual_anchor_max_validation_error"] < 1e-12
    assert result.audit["seasons_with_multiple_rookie_anchors"] == 0


def test_v75_preserves_calendar_and_zero_sum_invariants() -> None:
    result = build_annual_anchor_rating_v7_5(_rating_history())
    assert result.audit["race_k_budget_max_error"] < 1e-12
    assert result.audit["qualifying_k_budget_max_error"] < 1e-12
    assert result.audit["rookie_window_rows_used_in_car_fit"] == 0
    assert result.audit["frozen_debut_rating_mismatches"] == 0
    assert result.audit["provisional_qualifying_zero_sum_max_error"] < 1e-10
    assert result.audit["provisional_race_zero_sum_max_error"] < 1e-10
    assert result.audit["qualifying_zero_sum_max_error"] < 1e-10
    assert result.audit["race_zero_sum_max_error"] < 1e-10
