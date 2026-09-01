from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .decomposition import decompose_event
from .domain import Entry, HistoricalContext, SimulationConfig
from .outcomes import OutcomeClass, classify_outcome, observed_driver_performance


@dataclass(frozen=True, slots=True)
class RevisedEngineConfig:
    """Configuration for the additive v3 experiment; v0.2 remains unchanged."""

    initial_rating: float = 1500.0
    win_k_factor: float = 24.0
    performance_k_factor: float = 18.0
    season_retention: float = 0.95
    max_race_delta: float = 28.0
    calibration_default_shrinkage: float = 0.45
    calibration_minimum_shrinkage: float = 0.20
    calibration_min_observations: int = 100
    calibration_lookback_seasons: int = 5
    car_prior_races: float = 3.0
    driver_score_scale: float = 0.45
    car_score_scale: float = 1.25
    team_score_scale: float = 0.15
    prime_window: int = 20
    prime_minimum: int = 8
    simulations: int = 200

    def __post_init__(self) -> None:
        if not 0.0 <= self.season_retention <= 1.0:
            raise ValueError("season_retention must be in [0, 1]")
        if not 0.0 <= self.calibration_default_shrinkage <= 1.0:
            raise ValueError("calibration_default_shrinkage must be in [0, 1]")
        if not 0.0 <= self.calibration_minimum_shrinkage <= 1.0:
            raise ValueError("calibration_minimum_shrinkage must be in [0, 1]")
        if self.max_race_delta <= 0 or self.car_prior_races < 0:
            raise ValueError("delta cap must be positive and car prior cannot be negative")


_CIRCUIT_TAGS: dict[str, frozenset[str]] = {
    "monaco": frozenset({"street", "high_downforce", "low_speed"}),
    "marina_bay": frozenset({"street", "high_downforce", "low_speed"}),
    "baku": frozenset({"street", "low_drag", "stop_go"}),
    "jeddah": frozenset({"street", "low_drag", "high_speed"}),
    "vegas": frozenset({"street", "low_drag", "stop_go"}),
    "detroit": frozenset({"street", "stop_go", "low_speed"}),
    "long_beach": frozenset({"street", "stop_go", "low_speed"}),
    "phoenix": frozenset({"street", "stop_go", "low_speed"}),
    "adelaide": frozenset({"street", "stop_go"}),
    "monza": frozenset({"permanent", "low_drag", "high_speed"}),
    "spa": frozenset({"permanent", "low_drag", "high_speed"}),
    "reims": frozenset({"road", "low_drag", "high_speed"}),
    "avus": frozenset({"road", "low_drag", "high_speed"}),
    "hockenheimring": frozenset({"permanent", "low_drag", "high_speed"}),
    "hungaroring": frozenset({"permanent", "high_downforce", "low_speed"}),
    "zandvoort": frozenset({"permanent", "high_downforce", "technical"}),
    "catalunya": frozenset({"permanent", "high_downforce", "technical"}),
    "suzuka": frozenset({"permanent", "high_speed", "technical"}),
    "silverstone": frozenset({"permanent", "high_speed", "technical"}),
    "americas": frozenset({"permanent", "technical", "stop_go"}),
    "interlagos": frozenset({"permanent", "technical", "medium_speed"}),
    "rodriguez": frozenset({"permanent", "high_downforce", "stop_go"}),
    "villeneuve": frozenset({"semi_street", "low_drag", "stop_go"}),
    "albert_park": frozenset({"semi_street", "medium_speed", "technical"}),
    "red_bull_ring": frozenset({"permanent", "stop_go", "high_speed"}),
    "bahrain": frozenset({"permanent", "stop_go", "medium_speed"}),
    "yas_marina": frozenset({"permanent", "stop_go", "medium_speed"}),
    "losail": frozenset({"permanent", "high_speed", "technical"}),
    "imola": frozenset({"permanent", "technical", "medium_speed"}),
    "mugello": frozenset({"permanent", "high_speed", "technical"}),
    "paul_ricard": frozenset({"permanent", "high_speed", "technical"}),
}


def circuit_tags(circuit_id: object) -> frozenset[str]:
    """Return stable pre-race circuit attributes, with a conservative fallback."""

    key = str(circuit_id).strip().lower()
    return _CIRCUIT_TAGS.get(key, frozenset({"permanent", "balanced"}))


def circuit_similarity(current: pd.Series | dict[str, Any], prior: pd.Series | dict[str, Any]) -> float:
    """Similarity based only on known circuit context, never the target race result."""

    current_tags = circuit_tags(current["circuit_id"])
    prior_tags = circuit_tags(prior["circuit_id"])
    tag_score = len(current_tags & prior_tags) / max(len(current_tags | prior_tags), 1)
    current_passability = float(current.get("overtake_index", 0.5))
    prior_passability = float(prior.get("overtake_index", 0.5))
    passability_score = float(np.exp(-abs(current_passability - prior_passability) / 0.22))
    same_circuit_bonus = 0.35 if str(current["circuit_id"]) == str(prior["circuit_id"]) else 0.0
    return float(np.clip(0.15 + 0.50 * tag_score + 0.35 * passability_score + same_circuit_bonus, 0.05, 1.35))


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    standard_deviation = float(numeric.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation < 1e-12:
        return numeric * 0.0
    return (numeric - float(numeric.mean())) / standard_deviation


def _pairwise_rating_performance(ratings: np.ndarray, scale: float = 260.0) -> np.ndarray:
    if len(ratings) <= 1:
        return np.ones(len(ratings), dtype=float)
    differences = (ratings[:, None] - ratings[None, :]) / scale
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(differences, -30.0, 30.0)))
    np.fill_diagonal(probabilities, np.nan)
    return np.nanmean(probabilities, axis=1)


def _bounded_zero_sum(values: np.ndarray, limit: float) -> np.ndarray:
    """Project race updates onto a zero-sum box without changing their order."""

    if len(values) == 0:
        return values.astype(float)
    values = np.asarray(values, dtype=float)
    lower = float(values.min() - limit)
    upper = float(values.max() + limit)
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        total = float(np.clip(values - midpoint, -limit, limit).sum())
        if total > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return np.clip(values - (lower + upper) / 2.0, -limit, limit)


def _optimal_shrinkage(
    history: list[dict[str, float]],
    target: str,
    current_season: int,
    cfg: RevisedEngineConfig,
) -> float:
    """Fit calibration on prior observations only; current race is never included."""

    recent = [
        row
        for row in history
        if row["season"] >= current_season - cfg.calibration_lookback_seasons
        and np.isfinite(row[target])
    ]
    if len(recent) < cfg.calibration_min_observations:
        return cfg.calibration_default_shrinkage
    raw = np.asarray([row["raw"] for row in recent], dtype=float)
    baseline = np.asarray([row["baseline"] for row in recent], dtype=float)
    observed = np.asarray([row[target] for row in recent], dtype=float)
    candidates = np.linspace(0.0, 0.95, 20)
    losses = [np.mean((observed - ((1.0 - value) * raw + value * baseline)) ** 2) for value in candidates]
    fitted = float(candidates[int(np.argmin(losses))])
    return max(fitted, cfg.calibration_minimum_shrinkage)


def _context_from_race(race: pd.DataFrame) -> HistoricalContext:
    row = race.iloc[0]
    return HistoricalContext(
        season=int(row["season"]),
        event=str(row["event"]),
        regulation_era=str(row.get("regulation_era", "unknown")),
        race_distance_laps=int(row.get("race_distance_laps", 60) or 60),
        wet_fraction=float(row.get("wet_fraction", 0.0) or 0.0),
        track_temperature=float(row.get("track_temperature", 25.0) or 25.0),
        air_temperature=float(row.get("air_temperature", 20.0) or 20.0),
        wind_speed=float(row.get("wind_speed", 0.0) or 0.0),
        circuit_features={"overtake_index": float(row.get("overtake_index", 0.5) or 0.5)},
        circuit_id=str(row.get("circuit_id", "unknown")),
        weather_source=str(row.get("weather_source", "source_export")),
    )


def _same_season_car_scores(
    race: pd.DataFrame,
    season_car_history: list[dict[str, Any]],
    cfg: RevisedEngineConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Estimate car and reliability from prior, similar races in this season only."""

    event_context = race.iloc[0]
    car_values: dict[str, float] = {}
    reliability_values: dict[str, float] = {}
    effective_samples: dict[str, float] = {}
    for constructor in race["constructor"].astype(str).unique():
        candidates = [row for row in season_car_history if row["constructor"] == constructor]
        if not candidates:
            car_values[constructor] = 0.0
            reliability_values[constructor] = 0.5
            effective_samples[constructor] = 0.0
            continue
        weights = np.asarray([circuit_similarity(event_context, row) for row in candidates], dtype=float)
        car_signals = np.asarray([row["car_signal"] for row in candidates], dtype=float)
        reliability = np.asarray([row["reliability"] for row in candidates], dtype=float)
        effective = float(weights.sum())
        shrink = effective / (effective + cfg.car_prior_races)
        car_values[constructor] = shrink * float(np.average(car_signals, weights=weights))
        reliability_values[constructor] = float(np.average(reliability, weights=weights))
        effective_samples[constructor] = effective
    constructors = race["constructor"].astype(str)
    car_raw = constructors.map(car_values).astype(float)
    team_raw = constructors.map(reliability_values).astype(float)
    samples = constructors.map(effective_samples).astype(float)
    return _zscore(car_raw), _zscore(team_raw), samples


def _validate_source(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "round",
        "event",
        "driver",
        "driver_id",
        "constructor",
        "grid",
        "position",
        "status",
        "observed_winner",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Revised engine source is missing columns: {sorted(missing)}")
    data = source.copy()
    legacy_comparison = {
        "XW": "v02_XW",
        "XP": "v02_XP",
        "pre_rating": "v02_pre_rating",
        "post_rating": "v02_post_rating",
        "rating_delta": "v02_rating_delta",
    }
    data = data.rename(
        columns={old: new for old, new in legacy_comparison.items() if old in data.columns}
    )
    fact_columns = {
        "season",
        "round",
        "event",
        "driver",
        "driver_id",
        "constructor",
        "team",
        "grid",
        "position",
        "status",
        "observed_winner",
        "date",
        "source",
        "regulation_era",
        "race_distance_laps",
        "wet_fraction",
        "track_temperature",
        "air_temperature",
        "wind_speed",
        "circuit_id",
        "overtake_index",
        "weather_source",
        "positions_gained",
        "overtake_proxy_raw",
        "circuit_history_races",
        "overtake_source",
        *legacy_comparison.values(),
    }
    data = data[[column for column in data.columns if column in fact_columns]].copy()
    data = data.sort_values(["season", "round", "position"], kind="stable").reset_index(drop=True)
    if data.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("Driver/race rows must be unique")
    return data


def calculate_revised_history(
    source: pd.DataFrame,
    config: RevisedEngineConfig | None = None,
) -> pd.DataFrame:
    """Run the separate v3 causal expectation and rating experiment."""

    cfg = config or RevisedEngineConfig()
    data = _validate_source(source)
    ratings: dict[str, float] = {}
    season_seen: set[tuple[int, str]] = set()
    calibration_history: list[dict[str, float]] = []
    output_rows: list[dict[str, Any]] = []
    active_season: int | None = None
    season_car_history: list[dict[str, Any]] = []

    for (season, round_number), race in data.groupby(["season", "round"], sort=True):
        season = int(season)
        round_number = int(round_number)
        if active_season != season:
            active_season = season
            season_car_history = []
        race = race.copy().reset_index(drop=True)
        field_size = len(race)
        race["status_class"] = race["status"].map(classify_outcome).astype(str)
        race["observed_performance"] = [
            observed_driver_performance(position, field_size, status)
            for position, status in zip(race["position"], race["status_class"], strict=True)
        ]
        race["rating_eligible"] = ~race["status_class"].isin(
            [OutcomeClass.MECHANICAL.value, OutcomeClass.DNS.value, OutcomeClass.OTHER.value]
        )

        pre_ratings: list[float] = []
        season_reversion: list[float] = []
        for row in race.itertuples():
            identity = str(row.driver_id)
            previous = ratings.get(identity, cfg.initial_rating)
            key = (season, identity)
            if key not in season_seen:
                retained = cfg.initial_rating + cfg.season_retention * (previous - cfg.initial_rating)
                season_seen.add(key)
            else:
                retained = previous
            pre_ratings.append(retained)
            season_reversion.append(retained - previous)
            ratings[identity] = retained
        race["pre_rating"] = pre_ratings
        race["season_reversion_delta"] = season_reversion
        race["driver_expected_from_rating"] = _pairwise_rating_performance(
            race["pre_rating"].to_numpy(dtype=float)
        )
        race["driver_score_raw_revised"] = _zscore(race["pre_rating"])
        car_scores, team_scores, effective_samples = _same_season_car_scores(
            race, season_car_history, cfg
        )
        race["car_score_raw_revised"] = car_scores
        race["team_score_raw_revised"] = team_scores
        race["XdW_score_revised"] = cfg.driver_score_scale * race["driver_score_raw_revised"]
        race["XcW_score_revised"] = cfg.car_score_scale * race["car_score_raw_revised"]
        race["XtW_score_revised"] = cfg.team_score_scale * race["team_score_raw_revised"]
        race["car_effective_prior_races"] = effective_samples

        entries = [
            Entry(
                driver=str(row.driver),
                constructor=str(row.constructor),
                team=str(row.constructor),
                grid=max(int(row.grid), 1),
                driver_score=float(row.XdW_score_revised),
                car_score=float(row.XcW_score_revised),
                team_score=float(row.XtW_score_revised),
                driver_uncertainty=0.30,
                car_uncertainty=float(0.45 / np.sqrt(1.0 + row.car_effective_prior_races)),
                team_uncertainty=0.30,
                driver_id=str(row.driver_id),
            )
            for row in race.itertuples()
        ]
        context = _context_from_race(race)
        decomposed = pd.DataFrame(
            decompose_event(
                entries,
                context,
                SimulationConfig(
                    simulations=cfg.simulations,
                    seed=season * 100 + round_number,
                ),
            )
        )
        decomposed = decomposed.set_index("driver_id")
        race = race.set_index("driver_id")
        for column in decomposed.columns:
            if column not in race.columns or column in {
                "XW",
                "XP",
                "expected_position",
                "XdW_score",
                "XcW_score",
                "XtW_score",
            }:
                race[column] = decomposed[column]
        race["XW_raw_revised"] = decomposed["XW"]
        race["XP_raw_revised"] = decomposed["XP"]
        race["XW_neutral_revised"] = decomposed["XW_neutral_baseline"]
        race["XP_neutral_revised"] = decomposed["XP_neutral_baseline"]

        xw_lambda = _optimal_shrinkage(calibration_history, "winner", season, cfg)
        xp_lambda = _optimal_shrinkage(calibration_history, "performance", season, cfg)
        race["xw_calibration_shrinkage"] = xw_lambda
        race["xp_calibration_shrinkage"] = xp_lambda
        race["XW"] = (
            (1.0 - xw_lambda) * race["XW_raw_revised"]
            + xw_lambda * race["XW_neutral_revised"]
        )
        race["XW"] = race["XW"] / race["XW"].sum()
        race["XP"] = (
            (1.0 - xp_lambda) * race["XP_raw_revised"]
            + xp_lambda * race["XP_neutral_revised"]
        ).clip(0.0, 1.0)
        race["xw_calibration_adjustment"] = race["XW"] - race["XW_raw_revised"]
        race["xp_calibration_adjustment"] = race["XP"] - race["XP_raw_revised"]
        race["expected_position"] = 1.0 + (field_size - 1.0) * (1.0 - race["XP"])

        race["win_surprise"] = race["observed_winner"].astype(float) - race["XW"]
        race["performance_surprise"] = race["observed_performance"] - race["XP"]
        race["raw_rating_delta"] = 0.0
        eligible = race["rating_eligible"] & race["observed_performance"].notna()
        race.loc[eligible, "raw_rating_delta"] = (
            cfg.win_k_factor * race.loc[eligible, "win_surprise"]
            + cfg.performance_k_factor * race.loc[eligible, "performance_surprise"]
        )
        centered = race.loc[eligible, "raw_rating_delta"]
        race["field_centering_adjustment"] = 0.0
        if not centered.empty:
            projected = _bounded_zero_sum(centered.to_numpy(dtype=float), cfg.max_race_delta)
            race.loc[eligible, "rating_delta"] = projected
            race.loc[eligible, "field_centering_adjustment"] = (
                projected - centered.to_numpy(dtype=float)
            )
        race.loc[~eligible, "rating_delta"] = 0.0
        race["teammate_rating_delta"] = 0.0
        race["post_rating"] = race["pre_rating"] + race["rating_delta"]
        for identity, value in race["post_rating"].items():
            ratings[str(identity)] = float(value)

        for row in race.reset_index().itertuples():
            calibration_history.append(
                {
                    "season": float(season),
                    "raw": float(row.XW_raw_revised),
                    "baseline": float(row.XW_neutral_revised),
                    "winner": float(row.observed_winner),
                    "performance": float("nan"),
                }
            )
            if row.rating_eligible and np.isfinite(row.observed_performance):
                calibration_history.append(
                    {
                        "season": float(season),
                        "raw": float(row.XP_raw_revised),
                        "baseline": float(row.XP_neutral_revised),
                        "winner": float("nan"),
                        "performance": float(row.observed_performance),
                    }
                )

        race_for_car = race.reset_index()
        for constructor, teammates in race_for_car.groupby("constructor", sort=False):
            usable = teammates[
                teammates["rating_eligible"] & teammates["observed_performance"].notna()
            ]
            if usable.empty:
                car_signal = 0.0
            else:
                car_signal = float(
                    (usable["observed_performance"] - usable["driver_expected_from_rating"]).mean()
                )
            reliability = float(
                (~teammates["status_class"].eq(OutcomeClass.MECHANICAL.value)).mean()
            )
            season_car_history.append(
                {
                    "season": season,
                    "round": round_number,
                    "constructor": str(constructor),
                    "circuit_id": str(race_for_car.iloc[0].get("circuit_id", "unknown")),
                    "overtake_index": float(race_for_car.iloc[0].get("overtake_index", 0.5)),
                    "car_signal": car_signal,
                    "reliability": reliability,
                }
            )

        race["model_version"] = "revised_v3_parallel"
        race["car_prior_scope"] = "prior_similar_races_current_season_only"
        race["rating_update_method"] = "field_centered_bounded_no_teammate_term"
        output_rows.extend(race.reset_index().to_dict(orient="records"))

    history = pd.DataFrame(output_rows)
    return history.sort_values(["season", "round", "position"], kind="stable").reset_index(drop=True)


def build_revised_ranking(
    history: pd.DataFrame,
    config: RevisedEngineConfig | None = None,
) -> pd.DataFrame:
    """Sustained-prime ranking with sample size shown as uncertainty, not a penalty."""

    cfg = config or RevisedEngineConfig()
    rows: list[dict[str, Any]] = []
    for identity, races in history.groupby("driver_id", sort=False):
        races = races.sort_values(["season", "round"], kind="stable")
        window = min(cfg.prime_window, len(races))
        minimum = min(cfg.prime_minimum, window)
        rolling = races["post_rating"].rolling(window=window, min_periods=minimum).mean()
        sustained_prime = float(rolling.max()) if rolling.notna().any() else float(races["post_rating"].mean())
        peak = float(races["post_rating"].max())
        career_rating = 0.8 * sustained_prime + 0.2 * peak
        reliability = len(races) / (len(races) + 40.0)
        rows.append(
            {
                "driver_id": identity,
                "driver": str(races.iloc[-1]["driver"]),
                "races": len(races),
                "eligible_races": int(races["rating_eligible"].sum()),
                "wins": float(races["observed_winner"].sum()),
                "expected_wins": float(races["XW"].sum()),
                "wins_above_expected": float(races["win_surprise"].sum()),
                "performance_above_expected": float(races["performance_surprise"].sum(skipna=True)),
                "current_rating": float(races.iloc[-1]["post_rating"]),
                "sustained_prime_rating": sustained_prime,
                "peak_rating": peak,
                "career_rating": career_rating,
                "experience_reliability": reliability,
                "uncertainty_indicator": 1.0 - reliability,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["career_rating", "sustained_prime_rating", "driver"],
        ascending=[False, False, True],
        kind="stable",
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["percentile_score"] = 100.0 * (
        1.0 - (ranking["rank"] - 1.0) / max(len(ranking) - 1.0, 1.0)
    )
    return ranking.reset_index(drop=True)


def load_exported_races(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one v0.2 race export is required")
    return pd.concat(frames, ignore_index=True)


def write_revised_outputs(history: pd.DataFrame, output: Path) -> dict[str, Path]:
    """Write only revised_v3 files; legacy output names are never touched."""

    output.mkdir(parents=True, exist_ok=True)
    ranking = build_revised_ranking(history)
    targets = {
        "history_parquet": output / "driver_rating_history_revised_v3.parquet",
        "history_csv": output / "driver_rating_history_revised_v3.csv",
        "ranking_parquet": output / "driver_ranking_revised_v3.parquet",
        "ranking_csv": output / "driver_ranking_revised_v3.csv",
    }
    history.to_parquet(targets["history_parquet"], index=False)
    history.to_csv(targets["history_csv"], index=False)
    ranking.to_parquet(targets["ranking_parquet"], index=False)
    ranking.to_csv(targets["ranking_csv"], index=False)
    for season, frame in history.groupby("season", sort=True):
        frame.to_parquet(output / f"race_xw_revised_v3_{int(season)}.parquet", index=False)
        frame.to_csv(output / f"race_xw_revised_v3_{int(season)}.csv", index=False)
    return targets
