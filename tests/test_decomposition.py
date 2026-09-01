import pytest

from historical_xw.decomposition import decompose_event
from historical_xw.domain import SimulationConfig
from historical_xw.example import hungary_2021


@pytest.fixture(scope="module")
def rows():
    entries, context = hungary_2021()
    return decompose_event(entries, context, SimulationConfig(simulations=500, seed=7))


def test_field_probabilities_sum_to_one(rows) -> None:
    assert sum(row["XW"] for row in rows) == pytest.approx(1.0)
    assert sum(row["XW_simulated"] for row in rows) == pytest.approx(1.0)
    assert sum(row["XP"] for row in rows) == pytest.approx(len(rows) / 2)


def test_each_decomposition_is_additive(rows) -> None:
    for row in rows:
        reconstructed = row["XW_neutral_baseline"] + row["XdW_probability_contribution"] + row["XcW_probability_contribution"] + row["XtW_probability_contribution"]
        assert row["XW"] == pytest.approx(reconstructed, abs=1e-12)
        assert abs(row["decomposition_error"]) < 1e-12
        assert abs(row["xp_decomposition_error"]) < 1e-12


def test_mandatory_columns_are_independent(rows) -> None:
    required = {
        "XW", "XdW_score", "XcW_score", "XtW_score",
        "XdW_probability_contribution", "XcW_probability_contribution", "XtW_probability_contribution",
        "XW_neutral_baseline", "XW_with_neutral_driver", "XW_with_neutral_car",
        "XW_with_neutral_team", "driver_direct_uplift", "car_direct_uplift", "team_direct_uplift",
        "XP", "expected_position", "XdP_performance_contribution",
        "XcP_performance_contribution", "XtP_performance_contribution",
    }
    assert required.issubset(rows[0])
