import pandas as pd

from historical_xw.real_pipeline import JolpicaClient, cross_fitted_scores


def _driver(driver_id, given, family):
    return {"driverId": driver_id, "givenName": given, "familyName": family}


def test_season_results_deduplicates_a_driver_listed_twice_in_one_race(tmp_path, monkeypatch) -> None:
    """Regression test for a real crash on real 1950s data: a shared/relay-
    drive entry lists the same driver twice in one race's classification,
    which the simulation's coalition_probabilities dict (keyed by driver
    name) silently collapses to one key -- surfacing later as a numpy
    'a and p must have same size' crash deep in Plackett-Luce sampling,
    not here. Deduplicating at the source is what actually fixes it.
    """
    schedule = {"MRData": {"RaceTable": {"Races": [{"round": "1"}]}}}
    results = {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "raceName": "Test GP",
                        "date": "1951-01-01",
                        "Circuit": {"circuitName": "Test Circuit"},
                        "Results": [
                            {"Driver": _driver("fangio", "Juan", "Fangio"), "Constructor": {"name": "Alfa"}, "grid": "1", "position": "1", "points": "9", "status": "Finished"},
                            {"Driver": _driver("farina", "Nino", "Farina"), "Constructor": {"name": "Alfa"}, "grid": "2", "position": "2", "points": "6", "status": "Finished"},
                            # Relay drive: Fangio appears again, later in the field.
                            {"Driver": _driver("fangio", "Juan", "Fangio"), "Constructor": {"name": "Alfa"}, "grid": "2", "position": "3", "points": "0", "status": "Finished"},
                        ],
                    }
                ]
            }
        }
    }

    client = JolpicaClient(tmp_path)

    def fake_get(self, path, limit=2000):
        return schedule if path == "1951.json" else results

    monkeypatch.setattr(JolpicaClient, "get", fake_get)
    _, result_rows = client.season_results(1951)

    assert len(result_rows) == 2  # the repeated Fangio row is dropped
    assert (result_rows["driver"] == "Juan Fangio").sum() == 1
    assert result_rows.set_index("driver").loc["Juan Fangio", "position"] == 1  # first (best) occurrence kept


def test_season_results_unclassified_position_becomes_none(tmp_path, monkeypatch) -> None:
    schedule = {"MRData": {"RaceTable": {"Races": [{"round": "1"}]}}}
    results = {
        "MRData": {
            "RaceTable": {
                "Races": [
                    {
                        "raceName": "Test GP",
                        "date": "1951-01-01",
                        "Circuit": {"circuitName": "Test Circuit"},
                        "Results": [
                            {"Driver": _driver("a", "A", "A"), "Constructor": {"name": "X"}, "grid": "1", "position": "NC", "points": "0", "status": "Not classified"},
                        ],
                    }
                ]
            }
        }
    }

    client = JolpicaClient(tmp_path)
    monkeypatch.setattr(JolpicaClient, "get", lambda self, path, limit=2000: schedule if path == "1951.json" else results)
    _, result_rows = client.season_results(1951)

    assert result_rows.iloc[0]["position"] is None


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
