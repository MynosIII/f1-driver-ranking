from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .outcomes import OutcomeClass, classify_outcome, observed_driver_performance


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Controls the transparent first-stage historical driver ranking."""

    initial_rating: float = 1500.0
    k_factor: float = 32.0
    performance_k_factor: float = 24.0
    teammate_k_factor: float = 8.0
    prime_races: int = 60
    stabilization_races: float = 40.0
    prime_mean_weight: float = 0.75
    prime_peak_weight: float = 0.25
    probability_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive")
        if self.performance_k_factor < 0 or self.teammate_k_factor < 0:
            raise ValueError("performance and teammate factors cannot be negative")
        if self.prime_races < 1:
            raise ValueError("prime_races must be at least one")
        if self.stabilization_races < 0:
            raise ValueError("stabilization_races cannot be negative")
        if not np.isclose(self.prime_mean_weight + self.prime_peak_weight, 1.0):
            raise ValueError("prime weights must sum to one")


REQUIRED_COLUMNS = {"season", "event", "driver", "XW", "observed_winner"}


def _event_columns(frame: pd.DataFrame) -> list[str]:
    return ["season", "round"] if "round" in frame.columns else ["season", "event"]


def _identity_column(frame: pd.DataFrame) -> str:
    return "driver_id" if "driver_id" in frame.columns else "driver"


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
    if "XP" in frame and not frame["XP"].between(0.0, 1.0).all():
        raise ValueError("XP values must be normalized performance in [0, 1]")
    if not frame["observed_winner"].between(0.0, 1.0).all():
        raise ValueError("observed_winner must be a win share in [0, 1]")

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
    if frame.duplicated(keys + [_identity_column(frame)]).any():
        raise ValueError("A driver can appear only once in each race")


def calculate_rating_history(
    decompositions: pd.DataFrame,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """Update ratings from cross-fitted win probabilities and observed winners.

    The update combines win conversion, normalized finishing performance, and a
    teammate-controlled residual. Mechanical failures and DNS results do not update
    the driver; crashes and disqualifications are driver-attributable zero performance.
    """

    cfg = config or RankingConfig()
    data = decompositions.copy()
    if "driver_id" in data:
        data["driver_id"] = data["driver_id"].where(data["driver_id"].notna(), data["driver"])
    _validate_races(data, cfg.probability_tolerance)
    keys = _event_columns(data)
    identity_column = _identity_column(data)
    if "status_class" not in data:
        data["status_class"] = (
            data["status"].map(classify_outcome).astype(str)
            if "status" in data
            else OutcomeClass.FINISHED.value
        )
    if "observed_performance" not in data and "position" in data:
        field_sizes = data.groupby(keys)["driver"].transform("size")
        data["observed_performance"] = [
            observed_driver_performance(position, int(field_size), status_class)
            for position, field_size, status_class in zip(
                data["position"], field_sizes, data["status_class"], strict=True
            )
        ]
    data["rating_eligible"] = ~data["status_class"].isin(
        [OutcomeClass.MECHANICAL.value, OutcomeClass.DNS.value, OutcomeClass.OTHER.value]
    )
    data["teammate_surprise"] = 0.0
    if {"constructor", "XP", "observed_performance"}.issubset(data.columns):
        for _, teammates in data.groupby(keys + ["constructor"], sort=False, dropna=False):
            eligible = teammates[teammates["rating_eligible"] & teammates["observed_performance"].notna()]
            if len(eligible) < 2:
                continue
            for index, row in eligible.iterrows():
                others = eligible.drop(index)
                actual_margin = float(row["observed_performance"] - others["observed_performance"].mean())
                expected_margin = float(row["XP"] - others["XP"].mean())
                data.loc[index, "teammate_surprise"] = actual_margin - expected_margin
    sort_columns = keys + (["position"] if "position" in data.columns else ["driver"])
    data = data.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    ratings: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for _, race in data.groupby(keys, sort=False, dropna=False):
        race_rows: list[dict[str, object]] = []
        for item in race.to_dict(orient="records"):
            identity = str(item[identity_column])
            pre_rating = ratings.get(identity, cfg.initial_rating)
            observed = float(item["observed_winner"])
            expected = float(item["XW"])
            eligible = bool(item["rating_eligible"])
            surprise = observed - expected
            win_delta = cfg.k_factor * surprise if eligible else 0.0
            if "XP" in item and "observed_performance" in item and pd.notna(item["observed_performance"]):
                performance_surprise = float(item["observed_performance"]) - float(item["XP"])
                performance_delta = cfg.performance_k_factor * performance_surprise if eligible else 0.0
            else:
                performance_surprise = 0.0
                performance_delta = 0.0
            teammate_delta = cfg.teammate_k_factor * float(item["teammate_surprise"]) if eligible else 0.0
            delta = win_delta + performance_delta + teammate_delta
            post_rating = pre_rating + delta
            race_rows.append(
                {
                    **item,
                    "pre_rating": pre_rating,
                    "win_surprise": surprise,
                    "performance_surprise": performance_surprise,
                    "win_rating_delta": win_delta,
                    "performance_rating_delta": performance_delta,
                    "teammate_rating_delta": teammate_delta,
                    "rating_delta": delta,
                    "post_rating": post_rating,
                    "brier_component": (observed - expected) ** 2,
                }
            )
            ratings[identity] = post_rating
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
    identity_column = _identity_column(history)
    for identity, races in history.groupby(identity_column, sort=False):
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
            "driver_id": identity,
            "driver": str(races.iloc[-1]["driver"]),
            "races": race_count,
            "wins": float(races["observed_winner"].sum()),
            "expected_wins": float(races["XW"].sum()),
            "wins_above_expected": float(races["win_surprise"].sum()),
            "performance_above_expected": float(races["performance_surprise"].sum()),
            "teammate_performance_above_expected": float(races["teammate_surprise"].sum()),
            "mechanical_exclusions": int(races["status_class"].eq("mechanical").sum()),
            "dns_exclusions": int(races["status_class"].eq("dns").sum()),
            "ambiguous_outcome_exclusions": int(races["status_class"].eq("other").sum()),
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
