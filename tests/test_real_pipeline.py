import pandas as pd

from historical_xw.real_pipeline import cross_fitted_scores


def test_cross_fit_does_not_use_target_finish() -> None:
    base = pd.DataFrame([
        {"round": 1, "driver_id": "a", "constructor": "A", "grid": 1, "position": 1, "status": "Finished"},
        {"round": 1, "driver_id": "b", "constructor": "B", "grid": 2, "position": 2, "status": "Finished"},
        {"round": 2, "driver_id": "a", "constructor": "A", "grid": 2, "position": 2, "status": "Finished"},
        {"round": 2, "driver_id": "b", "constructor": "B", "grid": 1, "position": 1, "status": "Finished"},
    ])
    changed = base.copy()
    changed.loc[changed["round"] == 2, "position"] = [20, 19]
    left = cross_fitted_scores(base).query("round == 2")[["XdW_score", "XcW_score", "XtW_score"]]
    right = cross_fitted_scores(changed).query("round == 2")[["XdW_score", "XcW_score", "XtW_score"]]
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
