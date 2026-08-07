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
