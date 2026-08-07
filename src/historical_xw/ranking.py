from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Controls the transparent first-stage historical driver ranking."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    prime_races: int = 60
    stabilization_races: float = 40.0
    prime_mean_weight: float = 0.75
    prime_peak_weight: float = 0.25
    probability_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive")
        if self.prime_races < 1:
            raise ValueError("prime_races must be at least one")
        if self.stabilization_races < 0:
            raise ValueError("stabilization_races cannot be negative")
        if not np.isclose(self.prime_mean_weight + self.prime_peak_weight, 1.0):
            raise ValueError("prime weights must sum to one")


REQUIRED_COLUMNS = {"season", "event", "driver", "XW", "observed_winner"}


def _event_columns(frame: pd.DataFrame) -> list[str]:
    return ["season", "round"] if "round" in frame.columns else ["season", "event"]


def _validate_races(frame: pd.DataFrame, tolerance: float) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("At least one race decomposition is required")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Ranking inputs cannot contain null required values")
    if not frame["XW"].between(0.0, 1.0).all():
        raise ValueError("XW values must be probabilities in [0, 1]")
    if not frame["observed_winner"].isin([0, 1, False, True]).all():
        raise ValueError("observed_winner must be binary")

    keys = _event_columns(frame)
    grouped = frame.groupby(keys, sort=False, dropna=False)
    probability_sums = grouped["XW"].sum()
    winner_sums = grouped["observed_winner"].sum()
    if not np.allclose(probability_sums.to_numpy(), 1.0, atol=tolerance, rtol=0.0):
        bad = probability_sums[~np.isclose(probability_sums, 1.0, atol=tolerance, rtol=0.0)]
        raise ValueError(f"XW must sum to one within every race; invalid groups: {bad.to_dict()}")
    if not winner_sums.eq(1).all():
        raise ValueError(
            f"Every race must have exactly one observed winner; invalid groups: "
            f"{winner_sums[winner_sums.ne(1)].to_dict()}"
        )
    if frame.duplicated(keys + ["driver"]).any():
        raise ValueError("A driver can appear only once in each race")


def calculate_rating_history(
    decompositions: pd.DataFrame,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """Update ratings from cross-fitted win probabilities and observed winners.

    The update is ``K * (observed_win - XW)``. Because both observed wins and
    field-normalized XW sum to one, every race is zero-sum. The winner always
    gains unless its pre-race probability was exactly one, while unlikely wins
    receive a larger reward.
    """

    cfg = config or RankingConfig()
    data = decompositions.copy()
    _validate_races(data, cfg.probability_tolerance)
    keys = _event_columns(data)
    sort_columns = keys + (["position"] if "position" in data.columns else ["driver"])
    data = data.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    ratings: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for _, race in data.groupby(keys, sort=False, dropna=False):
        race_rows: list[dict[str, object]] = []
        for item in race.to_dict(orient="records"):
            driver = str(item["driver"])
            pre_rating = ratings.get(driver, cfg.initial_rating)
            observed = int(item["observed_winner"])
            expected = float(item["XW"])
            surprise = observed - expected
            delta = cfg.k_factor * surprise
            post_rating = pre_rating + delta
            race_rows.append(
                {
                    **item,
                    "pre_rating": pre_rating,
                    "win_surprise": surprise,
                    "rating_delta": delta,
                    "post_rating": post_rating,
                    "brier_component": (observed - expected) ** 2,
                }
            )
            ratings[driver] = post_rating
        if not np.isclose(sum(float(row["rating_delta"]) for row in race_rows), 0.0, atol=1e-10):
            raise AssertionError("A field-normalized race must produce a zero-sum rating update")
        rows.extend(race_rows)
    return pd.DataFrame(rows)


def build_driver_ranking(
    decompositions: pd.DataFrame,
    config: RankingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a career ranking and its auditable race-by-race rating history."""

    cfg = config or RankingConfig()
    history = calculate_rating_history(decompositions, cfg)
    ranked_rows: list[dict[str, object]] = []
    for driver, races in history.groupby("driver", sort=False):
        races = races.sort_values(_event_columns(races), kind="stable")
        prime = races.nlargest(min(cfg.prime_races, len(races)), "post_rating")
        prime_mean = float(prime["post_rating"].mean())
        peak_rating = float(races["post_rating"].max())
        career_raw = (
            cfg.prime_mean_weight * prime_mean
            + cfg.prime_peak_weight * peak_rating
        )
        race_count = len(races)
        experience = race_count / (race_count + cfg.stabilization_races)
        career_rating = cfg.initial_rating + experience * (career_raw - cfg.initial_rating)
        row: dict[str, object] = {
            "driver": driver,
            "races": race_count,
            "wins": int(races["observed_winner"].sum()),
            "expected_wins": float(races["XW"].sum()),
            "wins_above_expected": float(races["win_surprise"].sum()),
            "current_rating": float(races.iloc[-1]["post_rating"]),
            "prime_mean_rating": prime_mean,
            "peak_rating": peak_rating,
            "experience_weight": experience,
            "career_rating": career_rating,
            "mean_brier_component": float(races["brier_component"].mean()),
        }
        for source, target in (
            ("XdW_probability_contribution", "mean_driver_xw_contribution"),
            ("XcW_probability_contribution", "mean_car_xw_contribution"),
            ("XtW_probability_contribution", "mean_team_xw_contribution"),
        ):
            if source in races:
                row[target] = float(races[source].mean())
        ranked_rows.append(row)

    ranking = pd.DataFrame(ranked_rows).sort_values(
        ["career_rating", "peak_rating", "driver"],
        ascending=[False, False, True],
        kind="stable",
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    low, high = ranking["career_rating"].min(), ranking["career_rating"].max()
    ranking["score_0_100"] = (
        50.0 if np.isclose(low, high) else 100.0 * (ranking["career_rating"] - low) / (high - low)
    )
    return ranking.reset_index(drop=True), history
