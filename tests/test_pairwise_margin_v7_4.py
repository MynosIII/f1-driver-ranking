from __future__ import annotations

import numpy as np
import pandas as pd

from historical_xw.pairwise_margin_v7_4 import (
    PairwiseMarginConfigV74,
    build_pairwise_margin_adjustments_v7_4,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    history_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    specs: list[tuple[int, int, tuple[float, float, float]]] = []
    for season in range(2000, 2004):
        specs.extend(
            [
                (season, 1, (0.0, 8.0, 24.0)),
                (season, 2, (0.0, 12.0, 30.0)),
            ]
        )
    specs.extend(
        [
            (2004, 1, (0.0, 1.0, 30.0)),
            (2004, 2, (0.0, 20.0, 30.0)),
            (2004, 3, (0.0, 1.0, 50.0)),
        ]
    )
    drivers = ("a", "b", "c")
    strengths = {"a": 1650.0, "b": 1550.0, "c": 1450.0}
    for season, round_number, gaps in specs:
        for position, (driver_id, gap) in enumerate(zip(drivers, gaps, strict=True), start=1):
            history_rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "position": position,
                    "driver_id": driver_id,
                    "driver_id_f1db": driver_id,
                    "driver": driver_id.upper(),
                    "status_class": "finished",
                    "observed_performance": 1.0 - (position - 1) / 2.0,
                    "model_entity_id": "shared" if driver_id in {"a", "b"} else "other",
                    "layout_key": "layout",
                    "straight_demand": 0.5,
                    "altitude_demand": 0.0,
                    "car_suitability_score_v6": 0.0,
                }
            )
            baseline_rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "driver_id": driver_id,
                    "retrospective_pre_race_rating": strengths[driver_id],
                    "driver_context_adjustment": 0.0,
                }
            )
            timing_rows.append(
                {
                    "season": season,
                    "round": round_number,
                    "result_order": position,
                    "driver_id_f1db": driver_id,
                    "finish_position_text": str(position),
                    "laps": 60.0,
                    "race_time_seconds": 6000.0 + gap if position > 1 else 6000.0,
                    "gap_seconds": gap if position > 1 else np.nan,
                    "gap_laps": np.nan,
                }
            )
    # A legacy "+many laps" row that F1DB identifies as NC must never form a pair.
    history_rows.append(
        {
            "season": 2004,
            "round": 3,
            "position": 4,
            "driver_id": "nc",
            "driver_id_f1db": "nc",
            "driver": "NC",
            "status_class": "finished",
            "observed_performance": 0.0,
            "model_entity_id": "other",
            "layout_key": "layout",
            "straight_demand": 0.5,
            "altitude_demand": 0.0,
            "car_suitability_score_v6": 0.0,
        }
    )
    baseline_rows.append(
        {
            "season": 2004,
            "round": 3,
            "driver_id": "nc",
            "retrospective_pre_race_rating": 1500.0,
            "driver_context_adjustment": 0.0,
        }
    )
    timing_rows.append(
        {
            "season": 2004,
            "round": 3,
            "result_order": 4,
            "driver_id_f1db": "nc",
            "finish_position_text": "NC",
            "laps": 30.0,
            "race_time_seconds": np.nan,
            "gap_seconds": np.nan,
            "gap_laps": 30.0,
        }
    )
    return pd.DataFrame(history_rows), pd.DataFrame(timing_rows), pd.DataFrame(baseline_rows)


def test_all_pairs_include_second_third_and_weight_same_car_more() -> None:
    history, timing, baseline = _inputs()
    result = build_pairwise_margin_adjustments_v7_4(history, timing, baseline)
    target = result.pairwise_features.query("season == 2004 and round == 1")
    assert len(target) == 3
    assert {tuple(sorted(pair)) for pair in zip(target.driver_a_id, target.driver_b_id)} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }
    teammate = target[target["same_car_model"]].iloc[0]
    assert teammate["pair_attribution_weight"] == 3.0
    assert result.audit["mechanical_crash_dsq_pairs"] == 0


def test_closer_than_expected_runner_up_gets_more_credit() -> None:
    history, timing, baseline = _inputs()
    result = build_pairwise_margin_adjustments_v7_4(history, timing, baseline)
    features = result.driver_race_features.query("season == 2004").set_index(
        ["round", "driver_id"]
    )
    assert features.loc[(1, "b"), "pairwise_margin_performance_adjustment"] > features.loc[
        (2, "b"), "pairwise_margin_performance_adjustment"
    ]
    assert features.loc[(2, "a"), "raw_pairwise_margin_performance_adjustment"] > features.loc[
        (1, "a"), "raw_pairwise_margin_performance_adjustment"
    ]


def test_nc_is_excluded_and_final_adjustment_is_bounded_zero_sum() -> None:
    history, timing, baseline = _inputs()
    result = build_pairwise_margin_adjustments_v7_4(
        history,
        timing,
        baseline,
        config=PairwiseMarginConfigV74(maximum_performance_adjustment=0.20),
    )
    assert "nc" not in set(result.pairwise_features["driver_a_id"])
    assert "nc" not in set(result.pairwise_features["driver_b_id"])
    assert result.audit["margin_adjustment_zero_sum_max_error"] < 1e-12
    assert result.audit["maximum_absolute_performance_adjustment"] <= 0.20
    assert result.audit["all_gap_slopes_positive"]
    field_xw_sums = result.adjusted_history.groupby(["season", "round"])[
        "field_xw_v7_4"
    ].sum()
    assert np.allclose(field_xw_sums, 1.0)


def test_severely_shortened_race_has_negligible_margin_weight() -> None:
    history, timing, baseline = _inputs()
    race_context = history[["season", "round"]].drop_duplicates().copy()
    race_context["race_completion_fraction_v7_4"] = 1.0
    race_context["race_completion_reliability_v7_4"] = 1.0
    race_context["race_completion_quality_v7_4"] = "test_full"
    shortened = race_context["season"].eq(2004) & race_context["round"].eq(1)
    race_context.loc[shortened, "race_completion_fraction_v7_4"] = 1.0 / 44.0
    race_context.loc[shortened, "race_completion_reliability_v7_4"] = (
        (1.0 / 44.0) / 0.75
    ) ** 2
    result = build_pairwise_margin_adjustments_v7_4(
        history,
        timing,
        baseline,
        race_context=race_context,
    )
    target = result.driver_race_features.query("season == 2004 and round == 1")
    assert target["pairwise_margin_performance_adjustment"].abs().max() < 0.001
