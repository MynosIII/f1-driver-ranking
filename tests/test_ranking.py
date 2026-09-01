import pandas as pd
import pytest
import numpy as np

from historical_xw.ranking import RankingConfig, build_driver_ranking, calculate_rating_history


def _race(round_no: int, winner: str, probabilities: dict[str, float]) -> list[dict]:
    return [
        {
            "season": 2024,
            "round": round_no,
            "event": f"Race {round_no}",
            "driver": driver,
            "XW": probability,
            "observed_winner": int(driver == winner),
        }
        for driver, probability in probabilities.items()
    ]


def test_rating_update_is_zero_sum_and_rewards_upsets_more() -> None:
    data = pd.DataFrame(
        _race(1, "favorite", {"favorite": 0.8, "outsider": 0.2})
        + _race(2, "outsider", {"favorite": 0.8, "outsider": 0.2})
    )
    history = calculate_rating_history(data, RankingConfig(k_factor=32.0))

    per_race = history.groupby("round")["rating_delta"].sum()
    assert np.allclose(per_race.to_numpy(), 0.0, atol=1e-12)
    favorite_win = history.query("round == 1 and observed_winner == 1").iloc[0]
    outsider_win = history.query("round == 2 and observed_winner == 1").iloc[0]
    assert favorite_win["rating_delta"] == pytest.approx(6.4)
    assert outsider_win["rating_delta"] == pytest.approx(25.6)


def test_ranking_exposes_expected_win_surplus_and_stabilization() -> None:
    data = pd.DataFrame(
        _race(1, "a", {"a": 0.6, "b": 0.4})
        + _race(2, "a", {"a": 0.6, "b": 0.4})
    )
    ranking, history = build_driver_ranking(
        data,
        RankingConfig(prime_races=2, stabilization_races=2),
    )

    leader = ranking.iloc[0]
    assert leader["driver"] == "a"
    assert leader["wins_above_expected"] == pytest.approx(0.8)
    assert leader["experience_weight"] == pytest.approx(0.5)
    assert len(history) == 4


def test_rejects_probabilities_that_are_not_field_normalized() -> None:
    bad = pd.DataFrame(_race(1, "a", {"a": 0.7, "b": 0.4}))
    with pytest.raises(ValueError, match="sum to one"):
        calculate_rating_history(bad)


def test_xp_rewards_position_overperformance_and_excludes_mechanical_failure() -> None:
    data = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "event": "R", "driver": "a", "constructor": "A", "position": 1, "XW": 0.4, "XP": 0.6, "observed_winner": 1, "status": "Finished"},
            {"season": 2024, "round": 1, "event": "R", "driver": "b", "constructor": "B", "position": 2, "XW": 0.3, "XP": 0.7, "observed_winner": 0, "status": "Engine"},
            {"season": 2024, "round": 1, "event": "R", "driver": "c", "constructor": "C", "position": 3, "XW": 0.3, "XP": 0.2, "observed_winner": 0, "status": "Finished"},
        ]
    )
    history = calculate_rating_history(data, RankingConfig(teammate_k_factor=0))
    winner = history.query("driver == 'a'").iloc[0]
    mechanical = history.query("driver == 'b'").iloc[0]

    assert winner["observed_performance"] == pytest.approx(1.0)
    assert winner["performance_surprise"] == pytest.approx(0.4)
    assert mechanical["rating_delta"] == pytest.approx(0.0)
    assert not mechanical["rating_eligible"]


def test_crash_and_disqualification_are_driver_attributable() -> None:
    data = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "event": "R", "driver": "a", "position": 1, "XW": 0.5, "XP": 0.6, "observed_winner": 1, "status": "Finished"},
            {"season": 2024, "round": 1, "event": "R", "driver": "b", "position": 2, "XW": 0.5, "XP": 0.4, "observed_winner": 0, "status": "Collision"},
        ]
    )
    crashed = calculate_rating_history(data).query("driver == 'b'").iloc[0]
    assert crashed["status_class"] == "driver_fault"
    assert crashed["observed_performance"] == 0.0
    assert crashed["rating_delta"] < 0.0


def test_ambiguous_retirement_is_not_silently_blames_driver() -> None:
    data = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "event": "R", "driver": "a", "position": 1, "XW": 0.6, "XP": 0.7, "observed_winner": 1, "status": "Finished"},
            {"season": 2024, "round": 1, "event": "R", "driver": "b", "position": 2, "XW": 0.4, "XP": 0.3, "observed_winner": 0, "status": "Retired"},
        ]
    )
    retired = calculate_rating_history(data).query("driver == 'b'").iloc[0]
    assert retired["status_class"] == "other"
    assert retired["rating_delta"] == 0.0


def test_lapped_driver_is_a_classified_finisher() -> None:
    data = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "event": "R", "driver": "a", "position": 1, "XW": 0.8, "XP": 0.8, "observed_winner": 1, "status": "Finished"},
            {"season": 2024, "round": 1, "event": "R", "driver": "b", "position": 2, "XW": 0.2, "XP": 0.2, "observed_winner": 0, "status": "Lapped"},
        ]
    )
    lapped = calculate_rating_history(data).query("driver == 'b'").iloc[0]
    assert lapped["status_class"] == "finished"
    assert lapped["rating_eligible"]


def test_shared_historical_win_accepts_fractional_winner_credit() -> None:
    data = pd.DataFrame(
        [
            {"season": 1951, "round": 4, "event": "R", "driver": "a", "position": 1, "XW": 0.4, "XP": 0.7, "observed_winner": 0.5, "status": "Finished"},
            {"season": 1951, "round": 4, "event": "R", "driver": "b", "position": 1, "XW": 0.3, "XP": 0.6, "observed_winner": 0.5, "status": "Finished"},
            {"season": 1951, "round": 4, "event": "R", "driver": "c", "position": 3, "XW": 0.3, "XP": 0.2, "observed_winner": 0.0, "status": "Finished"},
        ]
    )
    history = calculate_rating_history(data)
    assert history["observed_winner"].sum() == pytest.approx(1.0)
