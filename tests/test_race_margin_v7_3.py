from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from historical_xw.race_margin_v7_3 import (
    attach_winner_margin_adjustments_v7_3,
    build_winner_margin_features_v7_3,
    load_f1db_race_timing_v7_3,
    parse_duration_seconds_v7_3,
    parse_lap_gap_v7_3,
)


def test_duration_and_lap_gap_parsing() -> None:
    assert np.isclose(parse_duration_seconds_v7_3("+13.643"), 13.643)
    assert np.isclose(parse_duration_seconds_v7_3("+1:16.996"), 76.996)
    assert np.isclose(parse_duration_seconds_v7_3("1:20:43.273"), 4843.273)
    assert np.isnan(parse_duration_seconds_v7_3("+1 lap"))
    assert parse_lap_gap_v7_3("+1 lap") == 1.0
    assert parse_lap_gap_v7_3("+12 laps") == 12.0


def test_f1db_timing_loader_preserves_result_entry_grain(tmp_path: Path) -> None:
    race_root = tmp_path / "src" / "data" / "seasons" / "2000" / "races" / "01-test"
    race_root.mkdir(parents=True)
    (race_root / "race.yml").write_text("id: 1\nround: 1\n", encoding="utf-8")
    (race_root / "race-results.yml").write_text(
        """- position: 1
  driverId: shared-driver
  constructorId: a
  laps: 50
  time: "1:20:00.000"
  gap:
  interval:
- position: 2
  driverId: shared-driver
  constructorId: b
  laps: 50
  time: "1:20:10.000"
  gap: "+10.000"
  interval: "+10.000"
""",
        encoding="utf-8",
    )
    timing = load_f1db_race_timing_v7_3(tmp_path, from_season=2000, to_season=2000)
    assert len(timing) == 2
    assert timing["result_order"].tolist() == [1, 2]
    assert timing.iloc[1]["gap_seconds"] == 10.0


def _margin_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    history_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    race_specs: list[tuple[int, int, float, float]] = []
    for season in range(2000, 2005):
        race_specs.extend(
            [
                (season, 1, 0.40, 4.0),
                (season, 2, 0.60, 12.0),
                (season, 3, 0.80, 28.0),
            ]
        )
    race_specs.extend(
        [
            (2005, 1, 0.60, 1.0),
            (2005, 2, 0.60, 20.0),
            (2005, 3, 0.40, 8.0),
            (2005, 4, 0.80, 8.0),
        ]
    )
    for season, round_number, winner_probability, gap_seconds in race_specs:
        winner_id = f"winner-{season}-{round_number}"
        runner_up_id = f"runner-{season}-{round_number}"
        history_rows.extend(
            [
                {
                    "season": season,
                    "round": round_number,
                    "driver_id_f1db": winner_id,
                    "XW": winner_probability,
                    "status_class": "finished",
                    "observed_performance": 1.0,
                },
                {
                    "season": season,
                    "round": round_number,
                    "driver_id_f1db": runner_up_id,
                    "XW": 1.0 - winner_probability,
                    "status_class": "finished",
                    "observed_performance": 0.0,
                },
            ]
        )
        timing_rows.extend(
            [
                {
                    "season": season,
                    "round": round_number,
                    "result_order": 1,
                    "driver_id_f1db": winner_id,
                    "laps": 60.0,
                    "race_time_seconds": 6000.0,
                    "gap_seconds": np.nan,
                    "gap_laps": np.nan,
                },
                {
                    "season": season,
                    "round": round_number,
                    "result_order": 2,
                    "driver_id_f1db": runner_up_id,
                    "laps": 60.0,
                    "race_time_seconds": 6000.0 + gap_seconds,
                    "gap_seconds": gap_seconds,
                    "gap_laps": np.nan,
                },
            ]
        )
    return pd.DataFrame(history_rows), pd.DataFrame(timing_rows)


def test_expected_gap_is_monotonic_in_pairwise_xw_and_actual_gap_changes_credit() -> None:
    history, timing = _margin_inputs()
    margins = build_winner_margin_features_v7_3(history, timing)
    target = margins[margins["season"].eq(2005)].set_index("round")
    assert target.loc[4, "expected_winner_gap_share_from_xw"] > target.loc[
        3, "expected_winner_gap_share_from_xw"
    ]
    assert np.isclose(
        target.loc[1, "expected_winner_gap_share_from_xw"],
        target.loc[2, "expected_winner_gap_share_from_xw"],
    )
    assert target.loc[2, "winner_margin_performance_adjustment"] > target.loc[
        1, "winner_margin_performance_adjustment"
    ]
    adjusted = attach_winner_margin_adjustments_v7_3(history, margins)
    event_sums = adjusted.groupby(["season", "round"])[
        "winner_margin_performance_adjustment"
    ].sum()
    assert event_sums.abs().max() < 1e-12
    assert "observed_performance_position_only" in adjusted
