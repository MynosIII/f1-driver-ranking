import pytest

from historical_xw.decomposition import decompose_event_v8
from historical_xw.domain import SimulationConfig
from historical_xw.example import hungary_2021


@pytest.fixture(scope="module")
def rows():
    entries, context = hungary_2021()
    return decompose_event_v8(entries, context, SimulationConfig(simulations=300, seed=7))


def test_field_expected_percentiles_sum_to_n_over_two(rows) -> None:
    # Every complete finishing order's percentiles sum to n/2 exactly (a
    # combinatorial identity, 1-rank/(n-1) summed over rank 0..n-1); since
    # each simulation is a complete order, the *average* over simulations
    # preserves that sum exactly too, not just asymptotically.
    n = len(rows)
    assert sum(row["XP"] for row in rows) == pytest.approx(n / 2, abs=1e-9)


def test_each_decomposition_is_additive(rows) -> None:
    for row in rows:
        reconstructed = (
            row["XP_neutral_baseline"]
            + row["XdP_probability_contribution"]
            + row["XcP_probability_contribution"]
            + row["XtP_probability_contribution"]
        )
        assert row["XP"] == pytest.approx(reconstructed, abs=1e-9)
        assert abs(row["decomposition_error_p"]) < 1e-9


def test_stronger_entries_get_higher_expected_percentile(rows) -> None:
    by_driver = {row["driver"]: row["XP"] for row in rows}
    # Hamilton/Verstappen (2021 Hungary: strong driver+car+team scores) should
    # clearly outrank Mazepin/Schumacher (weakest across all three).
    assert by_driver["Lewis Hamilton"] > by_driver["Nikita Mazepin"]
    assert by_driver["Max Verstappen"] > by_driver["Mick Schumacher"]


def test_mandatory_columns_present(rows) -> None:
    required = {
        "XP", "XP_simulated", "XP_lower_95", "XP_upper_95",
        "XdP_probability_contribution", "XcP_probability_contribution", "XtP_probability_contribution",
        "XP_neutral_baseline", "XP_with_neutral_driver", "XP_with_neutral_car", "XP_with_neutral_team",
    }
    assert required.issubset(rows[0])


def test_xp_differs_from_win_only_xw_for_a_strong_non_winner() -> None:
    """The entire point of v8: a driver who's clearly the 2nd-best but never
    wins gets meaningfully more credit from XP than from XW.
    """
    from historical_xw.decomposition import decompose_event

    entries, context = hungary_2021()
    config = SimulationConfig(simulations=300, seed=7)
    xw_rows = {r["driver"]: r["XW"] for r in decompose_event(entries, context, config)}
    xp_rows = {r["driver"]: r["XP"] for r in decompose_event_v8(entries, context, config)}

    # Bottas: strong car/team, unremarkable in this field for outright wins,
    # but should clearly beat a back-marker on the position-based measure by
    # more than the win-probability measure would ever distinguish two
    # midfield-or-lower non-winners.
    assert xp_rows["Valtteri Bottas"] - xp_rows["Mick Schumacher"] > xw_rows["Valtteri Bottas"] - xw_rows["Mick Schumacher"]
