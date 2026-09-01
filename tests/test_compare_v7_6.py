from __future__ import annotations

import pandas as pd
import pytest

from historical_xw.compare_v7_6 import compare_histories_v7_6


def _history(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["a", "b", "a", "b"],
            "driver": ["A", "B", "A", "B"],
            "season": [2000, 2000, 2000, 2000],
            "round": [1, 1, 2, 2],
            "event_index": [1, 1, 2, 2],
            "retrospective_rating": [1500.0, 1510.0, 1520.0, 1490.0],
        }
    ).assign(
        retrospective_rating=lambda frame: frame["retrospective_rating"] + offset
    )


def test_compare_histories_reports_exact_uniform_shift() -> None:
    drivers, trend, summary = compare_histories_v7_6(_history(0.0), _history(7.0))

    assert summary["matched_driver_race_rows"] == 4
    assert summary["rating_correlation"] == pytest.approx(1.0)
    assert summary["mean_absolute_rating_change"] == pytest.approx(7.0)
    assert drivers["final_rating_change"].tolist() == pytest.approx([7.0, 7.0])
    assert (
        trend["median_rating_v7_6"] - trend["median_rating_v7_5"]
    ).tolist() == pytest.approx([7.0, 7.0])


def test_compare_histories_rejects_duplicate_driver_race_rows() -> None:
    duplicated = pd.concat([_history(0.0), _history(0.0).iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="not unique"):
        compare_histories_v7_6(duplicated, _history(0.0))
