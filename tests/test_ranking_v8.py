import numpy as np
import pandas as pd
import pytest

from historical_xw.ranking_v8 import (
    RankingConfigV8,
    build_driver_ranking_v8,
    calculate_rating_history_v8,
    estimate_rookie_bootstrap_ratings,
    percentile_from_rank,
)


def _race(round_no: int, results: list[tuple[str, str, int, str, float]]) -> list[dict]:
    return [
        {
            "season": 2024,
            "round": round_no,
            "event": f"Race {round_no}",
            "driver": driver,
            "constructor": constructor,
            "position": position,
            "status": status,
            "XP": xp,
        }
        for driver, constructor, position, status, xp in results
    ]


def test_percentile_from_rank_endpoints() -> None:
    assert percentile_from_rank(0, 20) == 1.0
    assert percentile_from_rank(19, 20) == 0.0
    assert percentile_from_rank(0, 1) == 1.0


def test_rating_update_is_zero_sum_when_nobody_is_excluded() -> None:
    data = pd.DataFrame(
        _race(1, [("a", "t1", 1, "Finished", 0.7), ("b", "t2", 2, "Finished", 0.3)])
    )
    history = calculate_rating_history_v8(data, RankingConfigV8(k_factor=32.0))
    assert np.isclose(history["rating_delta"].sum(), 0.0, atol=1e-10)


def test_rating_update_is_zero_sum_for_a_larger_field_too() -> None:
    """A 2-driver race is zero-sum by log-odds symmetry alone even without
    re-centering (logit(p) + logit(1-p) == 0). This regression test uses a
    5-driver field, where that symmetry does not hold and only the explicit
    re-centering step keeps it zero-sum -- exactly the field size at which
    this project found real, substantial rating inflation (~300 points
    accumulated over 75 years) in the version of this function that didn't
    re-center.
    """
    data = pd.DataFrame(
        _race(
            1,
            [
                ("a", "t1", 1, "Finished", 0.5),
                ("b", "t2", 2, "Finished", 0.3),
                ("c", "t3", 3, "Finished", 0.35),
                ("d", "t4", 4, "Finished", 0.1),
                ("e", "t5", 5, "Finished", 0.05),
            ],
        )
    )
    history = calculate_rating_history_v8(data, RankingConfigV8(k_factor=32.0))
    assert np.isclose(history["rating_delta"].sum(), 0.0, atol=1e-9)


def test_no_systematic_inflation_across_many_races() -> None:
    """Direct regression test for the real bug: without re-centering, mean
    rating_delta across many races was persistently positive (not just
    occasionally, by chance), producing ~300 points of cross-era inflation
    on the real 1950-2025 dataset. Simulates enough races with varied,
    somewhat-miscalibrated XP (deliberately not matching the field's actual
    order well, the realistic case) that a systematic bias would show up
    clearly if re-centering weren't happening.
    """
    rng = np.random.default_rng(0)
    rows = []
    for r in range(200):
        n = 5
        xp = rng.dirichlet(np.ones(n) * 2) * n * 0.5  # unbiased-ish but noisy per-race XP
        finish_order = rng.permutation(n)
        for i, driver_idx in enumerate(finish_order):
            rows.append(
                {
                    "season": 2000, "round": r, "event": f"Race {r}", "driver": f"d{driver_idx}",
                    "constructor": f"t{driver_idx}", "position": i + 1, "status": "Finished",
                    "XP": float(np.clip(xp[driver_idx], 0.02, 0.98)),
                }
            )
    history = calculate_rating_history_v8(pd.DataFrame(rows), RankingConfigV8(rookie_bootstrap_races=0))
    assert abs(history["rating_delta"].mean()) < 0.5, (
        f"mean rating_delta {history['rating_delta'].mean():.3f} suggests systematic drift, not noise"
    )


def test_mechanical_dnf_is_excluded_from_the_update() -> None:
    data = pd.DataFrame(
        _race(1, [("a", "t1", 1, "Finished", 0.9), ("b", "t2", 2, "Engine", 0.5), ("c", "t3", 3, "Finished", 0.1)])
    )
    history = calculate_rating_history_v8(data, RankingConfigV8(k_factor=32.0))
    b = history.query("driver == 'b'").iloc[0]
    assert b["excluded_from_update"]
    assert b["rating_delta"] == 0.0
    assert b["post_rating"] == b["pre_rating"] == 1500.0


def test_accident_dnf_is_not_excluded() -> None:
    # b was expected to run mid-pack (XP=0.5) but crashed out unclassified
    # (position=None) -- falls to last in finish order, a real penalty.
    data = pd.DataFrame(
        _race(1, [("a", "t1", 1, "Finished", 0.9), ("b", "t2", None, "Accident", 0.5), ("c", "t3", 2, "Finished", 0.1)])
    )
    history = calculate_rating_history_v8(data, RankingConfigV8(k_factor=32.0))
    b = history.query("driver == 'b'").iloc[0]
    assert not b["excluded_from_update"]
    assert b["rating_delta"] < 0.0


def test_unclassified_position_sorts_after_every_classified_finisher() -> None:
    data = pd.DataFrame(
        _race(1, [("a", "t1", 1, "Finished", 0.6), ("b", "t2", None, "Accident", 0.6), ("c", "t3", 2, "Finished", 0.4)])
    )
    history = calculate_rating_history_v8(data, RankingConfigV8())
    ranks = history.set_index("driver")["finish_rank"]
    assert ranks["a"] == 0
    assert ranks["c"] == 1
    assert ranks["b"] == 2


def test_rookie_bootstrap_matches_neighbor_interpolation_example() -> None:
    """The worked example this project was designed around: a rookie
    finishing between a 2000-rated driver and an 1800-rated one, with equal
    cars, gets an implied starting rating of ~1900.
    """
    pass1 = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "driver": "ahead", "finish_rank": 0, "pre_rating": 2000.0, "XcW_score": 0.0},
            {"season": 2024, "round": 1, "driver": "rookie", "finish_rank": 1, "pre_rating": 1500.0, "XcW_score": 0.0},
            {"season": 2024, "round": 1, "driver": "behind", "finish_rank": 2, "pre_rating": 1800.0, "XcW_score": 0.0},
        ]
    )
    bootstrapped = estimate_rookie_bootstrap_ratings(pass1, RankingConfigV8(rookie_bootstrap_races=5))
    assert bootstrapped["rookie"] == pytest.approx(1900.0)


def test_rookie_bootstrap_stronger_car_reduces_implied_rating() -> None:
    pass1 = pd.DataFrame(
        [
            {"season": 2024, "round": 1, "driver": "ahead", "finish_rank": 0, "pre_rating": 2000.0, "XcW_score": 0.0},
            {"season": 2024, "round": 1, "driver": "rookie", "finish_rank": 1, "pre_rating": 1500.0, "XcW_score": 1.0},
            {"season": 2024, "round": 1, "driver": "behind", "finish_rank": 2, "pre_rating": 1800.0, "XcW_score": 0.0},
        ]
    )
    bootstrapped = estimate_rookie_bootstrap_ratings(pass1, RankingConfigV8(rookie_bootstrap_races=5))
    assert bootstrapped["rookie"] < 1900.0


def test_rookie_bootstrap_is_a_single_pass_not_iterative() -> None:
    """estimate_rookie_bootstrap_ratings has no loop/convergence check -- it
    reads pass1 once and returns. This test exists to catch a future
    refactor accidentally reintroducing iteration.
    """
    import inspect

    source = inspect.getsource(estimate_rookie_bootstrap_ratings)
    assert "while" not in source
    for keyword in ("for _ in range", "converge", "iterate"):
        assert keyword not in source


def test_build_driver_ranking_v8_end_to_end_smoke() -> None:
    data = pd.DataFrame(
        _race(1, [("a", "t1", 1, "Finished", 0.6), ("b", "t2", 2, "Finished", 0.4)])
        + _race(2, [("a", "t1", 1, "Finished", 0.6), ("b", "t2", 2, "Engine", 0.4)])
    )
    ranking, history = build_driver_ranking_v8(data, RankingConfigV8(prime_races=2, stabilization_races=2))
    assert list(ranking["driver"]) == ["a", "b"]  # a won both, unambiguous leader
    assert ranking.iloc[0]["rank"] == 1
    assert len(history) == 4
    assert history.query("driver == 'b' and round == 2")["excluded_from_update"].iloc[0]


def test_dominant_car_does_not_bury_a_dominant_driver_below_the_field() -> None:
    """Regression test for a real bug found against actual 2018-2024 data:
    Hamilton (dominant driver, dominant Mercedes) came out ranked dead last
    of 40 real drivers, because the rookie-bootstrap's car adjustment was
    miscalibrated large enough to crush his implied starting rating, and
    that low starting rating then anchored his whole career-rating shrinkage
    forever, even though the ongoing race-by-race signal correctly favored
    him. A driver who wins every single race must still end up ranked above
    one who finishes last every single race, even in a much stronger car,
    given enough races for the sustained real signal to dominate a single
    early-race bootstrap adjustment.

    XP is deliberately flat/neutral (0.25 for all four, i.e. "no prior
    information") rather than pre-weighted toward the strong car: that
    isolates the car's influence to exactly the bootstrap's car adjustment
    being tested here, instead of also (unrealistically) being baked into
    the expected-percentile target itself.
    """
    rows = []
    for round_no in range(1, 13):
        event = f"Race {round_no}"
        rows.extend(
            [
                {"season": 2024, "round": round_no, "event": event, "driver": "star", "constructor": "top_team",
                 "position": 1, "status": "Finished", "XP": 0.25, "XcW_score": 1.8},
                {"season": 2024, "round": round_no, "event": event, "driver": "mid1", "constructor": "mid_team_a",
                 "position": 2, "status": "Finished", "XP": 0.25, "XcW_score": 0.0},
                {"season": 2024, "round": round_no, "event": event, "driver": "mid2", "constructor": "mid_team_b",
                 "position": 3, "status": "Finished", "XP": 0.25, "XcW_score": -0.1},
                {"season": 2024, "round": round_no, "event": event, "driver": "backmarker", "constructor": "bottom_team",
                 "position": 4, "status": "Finished", "XP": 0.25, "XcW_score": -1.8},
            ]
        )
    ranking, _ = build_driver_ranking_v8(pd.DataFrame(rows), RankingConfigV8(stabilization_races=40))
    ranks = ranking.set_index("driver")["rank"]
    assert ranks["star"] < ranks["backmarker"], (
        f"star (rank {ranks['star']}) must outrank backmarker (rank {ranks['backmarker']})"
    )
    assert ranks["star"] == 1


def test_saturated_expectation_still_allows_positive_surprise() -> None:
    """Regression test for a real bug found against actual 2018-2024 data:
    with a *raw* percentile-vs-percentile comparison, a driver whose XP is
    already near 1.0 (correctly reflecting a dominant car) has almost no
    room to register positive surprise even by winning every single race,
    while having unlimited room to register negative surprise on any race
    they don't win outright -- it rated a driver who won 29% of 148 races
    (and podiumed in most of the rest) barely above the flat 1500 default.
    Comparing in log-odds space (what this function actually does) fixes
    that: winning every race against a saturated 0.9+ expectation must
    still produce positive rating movement, not roughly zero.

    A realistic multi-driver field, not a single-entrant "race": with only
    one entrant, per-race re-centering (see the zero-sum test below) always
    zeroes the result out, since there is no field to be relatively better
    than -- that's correct, not a bug, but it means this test needs three
    backmarkers finishing exactly as (unremarkably) expected, so the only
    real signal in the race is "dominant" beating a saturated expectation.
    """
    rows = []
    for r in range(1, 11):
        rows.extend(
            [
                {"season": 2024, "round": r, "event": f"Race {r}", "driver": "dominant", "constructor": "top_team",
                 "position": 1, "status": "Finished", "XP": 0.95},
                {"season": 2024, "round": r, "event": f"Race {r}", "driver": "mid1", "constructor": "mid_team",
                 "position": 2, "status": "Finished", "XP": 0.2},
                {"season": 2024, "round": r, "event": f"Race {r}", "driver": "mid2", "constructor": "mid_team_b",
                 "position": 3, "status": "Finished", "XP": 0.15},
                {"season": 2024, "round": r, "event": f"Race {r}", "driver": "backmarker", "constructor": "bottom_team",
                 "position": 4, "status": "Finished", "XP": 0.1},
            ]
        )
    history = calculate_rating_history_v8(pd.DataFrame(rows), RankingConfigV8(rookie_bootstrap_races=0))
    dominant = history.query("driver == 'dominant'")
    assert (dominant["rating_delta"] > 0).all()
    assert dominant.iloc[-1]["post_rating"] > dominant.iloc[0]["pre_rating"] + 5.0


def test_missing_required_column_raises() -> None:
    bad = pd.DataFrame(_race(1, [("a", "t1", 1, "Finished", 0.6)])).drop(columns=["XP"])
    with pytest.raises(ValueError, match="Missing v8 ranking columns"):
        calculate_rating_history_v8(bad)
