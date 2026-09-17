from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .reliability import MECHANICAL, classify_status

REQUIRED_COLUMNS_V8 = {"season", "event", "driver", "constructor", "XP", "position", "status"}


@dataclass(frozen=True, slots=True)
class RankingConfigV8:
    """v8 extends v7's win-only rating (``ranking.RankingConfig``) with a
    full-field position signal, DNF-fault awareness, and a bootstrapped
    (not flat) rookie starting rating.
    """

    initial_rating: float = 1500.0
    # Lower than v7's 32.0 deliberately: that value was calibrated for a
    # raw [0,1] probability difference, but v8's surprise is a *log-odds*
    # difference (see logit_clip_epsilon below), which has a much wider
    # typical range. 8.0 was picked empirically against real 2018-2024
    # data to produce a comparable overall rating spread to v7's (top
    # career ratings around 2000-2100, most of the field in 1400-1700).
    k_factor: float = 8.0
    prime_races: int = 60
    stabilization_races: float = 40.0
    prime_mean_weight: float = 0.75
    prime_peak_weight: float = 0.25
    probability_tolerance: float = 1e-8
    # How many of a driver's earliest races anchor their bootstrapped
    # starting rating -- see ``estimate_rookie_bootstrap_ratings``.
    rookie_bootstrap_races: int = 5
    # Elo-equivalent points attributed to one standard deviation of car
    # strength (the same z-scored ``car_score`` fed into the simulation),
    # used only to car-adjust the rookie bootstrap, never the ongoing
    # race-by-race rating update. Calibrated empirically against real
    # 2018-2024 data: with car_score z-scored (std=1) and pass-1 neighbor
    # ratings for a driver's first few races typically spread only
    # ~50-150 points around the 1500 default (they haven't had time to
    # diverge yet), a scale anywhere near 200 lets one dominant car (a
    # ~2 SD outlier, e.g. 2018-2020 Mercedes) swing the bootstrap by
    # ~400 points -- enough to bury a genuinely great driver in a
    # genuinely great car below the entire field, which real-data testing
    # against Hamilton/Verstappen's actual 2018-2024 records caught
    # directly (see the regression test asserting this).
    car_rating_scale: float = 40.0
    car_adjustment_weight: float = 1.0
    # A percentile is bounded in [0, 1], so comparing raw percentiles has a
    # ceiling-saturation problem win probability doesn't: a driver whose XP
    # is already 0.95+ (correctly reflecting a dominant car) has almost no
    # room to generate *positive* surprise even at their absolute best, but
    # unbounded room to generate negative surprise on an off day -- found
    # against real 2018-2024 data, where it rated a driver who won 29% of
    # 148 races (and podiumed in most of the rest) barely above the flat
    # 1500 default, because his XP was so consistently high that even
    # genuinely excellent finishes read as slightly "disappointing". The
    # fix is the same one Elo itself uses: compare in log-odds space, not
    # raw probability space, so distance-to-the-ceiling is stretched out
    # instead of compressed. ``logit_clip_epsilon`` keeps percentiles of
    # exactly 0 or 1 (a literal win or literal last) away from +-infinity.
    logit_clip_epsilon: float = 0.02

    def __post_init__(self) -> None:
        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive")
        if self.prime_races < 1:
            raise ValueError("prime_races must be at least one")
        if self.stabilization_races < 0:
            raise ValueError("stabilization_races cannot be negative")
        if not np.isclose(self.prime_mean_weight + self.prime_peak_weight, 1.0):
            raise ValueError("prime weights must sum to one")
        if self.rookie_bootstrap_races < 0:
            raise ValueError("rookie_bootstrap_races cannot be negative")
        if not 0 < self.logit_clip_epsilon < 0.5:
            raise ValueError("logit_clip_epsilon must be in (0, 0.5)")


def percentile_from_rank(rank_index: int, field_size: int) -> float:
    """0-based rank index -> percentile, 1.0 = winner, 0.0 = last."""
    if field_size <= 1:
        return 1.0
    return 1.0 - rank_index / (field_size - 1)


def _clipped_logit(p: float, epsilon: float) -> float:
    p = min(max(p, epsilon), 1.0 - epsilon)
    return math.log(p / (1.0 - p))


def _event_columns(frame: pd.DataFrame) -> list[str]:
    return ["season", "round"] if "round" in frame.columns else ["season", "event"]


def _validate_v8(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS_V8 - set(frame.columns)
    if missing:
        raise ValueError(f"Missing v8 ranking columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("At least one race decomposition is required")
    if frame[["season", "driver", "constructor", "XP"]].isna().any().any():
        raise ValueError("Ranking inputs cannot contain null required values (position/status may be null for non-finishers)")
    keys = _event_columns(frame)
    if frame.duplicated(keys + ["driver"]).any():
        raise ValueError("A driver can appear only once in each race")


def _finish_rank(race: pd.DataFrame) -> pd.Series:
    """0-based finishing rank for one race's rows, from Ergast-style
    ``position`` (numeric for classified finishers, null/non-numeric for
    unclassified/DNF). Unclassified entries sort after every classified
    finisher, in their original row order (Ergast already lists them in
    fell-out order), never ahead of a driver who was actually running.
    """
    position = pd.to_numeric(race["position"], errors="coerce")
    fallback = (position.max() + 1) if position.notna().any() else len(race)
    filled = position.fillna(fallback)
    return filled.rank(method="first").astype(int) - 1


def calculate_rating_history_v8(
    decompositions: pd.DataFrame,
    config: RankingConfigV8 | None = None,
    initial_ratings: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Update ratings from field-normalized expected finishing percentile
    (``XP``) vs. observed finishing percentile, compared in log-odds space.

    ``K * (logit(observed_percentile) - logit(XP))`` replaces v7's win-only
    ``K * (observed_win - XW)`` -- a driver who was expected to finish 3rd
    and actually finished 3rd sees no surprise; one expected to win who
    finished 10th sees a large negative one, not just "didn't win". The
    log-odds comparison (not a raw percentile difference) is what keeps a
    driver whose XP is already near 1.0 able to still register genuine
    surprise at the top end, rather than having almost no room above an
    already-sky-high expectation while having unlimited room below it --
    see ``RankingConfigV8.logit_clip_epsilon``.

    A DNF classified as ``MECHANICAL`` (car/team fault, via
    ``reliability.classify_status``) is excluded from that race's update
    entirely -- the driver's rating carries over unchanged, since the
    car/team component of the decomposition already attributes that
    failure elsewhere. An ``ACCIDENT`` DNF is not excluded: it counts
    against the driver at their actual (fell-out) finishing position, same
    as any other result.

    ``initial_ratings`` overrides ``config.initial_rating`` for specific
    drivers -- this is how the rookie bootstrap in
    ``estimate_rookie_bootstrap_ratings`` feeds a second pass.
    """
    cfg = config or RankingConfigV8()
    _validate_v8(decompositions)
    initial_ratings = initial_ratings or {}
    keys = _event_columns(decompositions)

    ratings: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for _, race in decompositions.groupby(keys, sort=False, dropna=False):
        field_size = len(race)
        race = race.assign(finish_rank=_finish_rank(race)).sort_values("finish_rank", kind="stable")
        for item in race.to_dict(orient="records"):
            driver = str(item["driver"])
            pre_rating = ratings.get(driver, initial_ratings.get(driver, cfg.initial_rating))
            category = classify_status(item.get("status"))
            observed_percentile = percentile_from_rank(int(item["finish_rank"]), field_size)
            expected = float(item["XP"])
            excluded = category == MECHANICAL
            raw_percentile_diff = observed_percentile - expected
            surprise = 0.0 if excluded else (
                _clipped_logit(observed_percentile, cfg.logit_clip_epsilon)
                - _clipped_logit(expected, cfg.logit_clip_epsilon)
            )
            delta = 0.0 if excluded else cfg.k_factor * surprise
            post_rating = pre_rating + delta
            rows.append(
                {
                    **item,
                    "field_size": field_size,
                    "status_category": category,
                    "observed_percentile": observed_percentile,
                    "pre_rating": pre_rating,
                    "raw_percentile_diff": raw_percentile_diff,
                    "position_surprise": surprise,
                    "rating_delta": delta,
                    "post_rating": post_rating,
                    "excluded_from_update": excluded,
                }
            )
            ratings[driver] = post_rating
    return pd.DataFrame(rows)


def estimate_rookie_bootstrap_ratings(
    history_pass1: pd.DataFrame,
    config: RankingConfigV8 | None = None,
    car_score_column: str = "XcW_score",
) -> dict[str, float]:
    """A rookie's starting rating, estimated (not assumed flat) from their
    first ``rookie_bootstrap_races``: for each of those races, average the
    *contemporaneous* pre-race ratings of whoever finished immediately
    ahead and behind them (a 2000-rated driver ahead, 1800 behind ->
    implied ~1900), then correct for the car they were in -- a strong car
    inflates that implied rating, a weak one deflates it, relative to that
    race's field-average car strength.

    This is explicitly a single bootstrap pass, not an iterative solve:
    it reads pre-race ratings from ``history_pass1`` (which itself used
    everyone's flat/default starting rating) and is meant to be fed back
    into exactly one second call to ``calculate_rating_history_v8`` via
    ``initial_ratings`` -- see ``build_driver_ranking_v8``. Re-running the
    bootstrap against its own output would be circular; this project
    deliberately does not do that.

    IMPORTANT PRECONDITION, found by testing against real data: this
    treats a driver's *first appearance in the supplied dataset* as their
    rookie debut. Run this on a truncated window (e.g. 2018-2024 only) and
    an already-established champion whose first *dataset* race happens to
    fall in that window (Hamilton, debut 2007, tested against a 2018-2024
    slice) gets bootstrapped as if 2018 were his rookie season -- there is
    no way for this function to know he already had 11 prior seasons of
    real form. The fix is not in the code: run this against a dataset that
    actually spans real driver debuts (the full historical record from the
    sport's first season), not a topical slice. ``rookie_bootstrap_races=0``
    disables the bootstrap entirely (flat ``initial_rating`` for everyone,
    same as v7) if that's ever a more honest choice than a wrong bootstrap.
    """
    cfg = config or RankingConfigV8()
    if cfg.rookie_bootstrap_races == 0:
        return {}
    keys = _event_columns(history_pass1)
    data = history_pass1.sort_values(keys, kind="stable").reset_index(drop=True)
    data["driver_race_index"] = data.groupby("driver").cumcount()

    implied: dict[str, list[float]] = {}
    for _, race in data.groupby(keys, sort=False, dropna=False):
        race = race.sort_values("finish_rank").reset_index(drop=True)
        n = len(race)
        has_car_score = car_score_column in race.columns
        field_mean_car = float(race[car_score_column].mean()) if has_car_score else 0.0
        for i in range(n):
            row = race.iloc[i]
            if row["driver_race_index"] >= cfg.rookie_bootstrap_races:
                continue
            neighbor_ratings = []
            if i > 0:
                neighbor_ratings.append(float(race.iloc[i - 1]["pre_rating"]))
            if i < n - 1:
                neighbor_ratings.append(float(race.iloc[i + 1]["pre_rating"]))
            if not neighbor_ratings:
                continue  # sole entrant that race, no anchor available
            local_rating = float(np.mean(neighbor_ratings))
            car_score = float(row[car_score_column]) if has_car_score else 0.0
            car_adjustment = cfg.car_adjustment_weight * cfg.car_rating_scale * (car_score - field_mean_car)
            implied.setdefault(str(row["driver"]), []).append(local_rating - car_adjustment)

    return {driver: float(np.mean(values)) for driver, values in implied.items()}


def build_driver_ranking_v8(
    decompositions: pd.DataFrame,
    config: RankingConfigV8 | None = None,
    car_score_column: str = "XcW_score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Career ranking + full rating history, using the position-based,
    fault-aware, bootstrapped-rookie-starting-rating v8 model.
    """
    cfg = config or RankingConfigV8()
    pass1 = calculate_rating_history_v8(decompositions, cfg)
    bootstrap_ratings = estimate_rookie_bootstrap_ratings(pass1, cfg, car_score_column)
    history = calculate_rating_history_v8(decompositions, cfg, initial_ratings=bootstrap_ratings)

    ranked_rows: list[dict[str, object]] = []
    for driver, races in history.groupby("driver", sort=False):
        races = races.sort_values(_event_columns(races), kind="stable")
        prime = races.nlargest(min(cfg.prime_races, len(races)), "post_rating")
        prime_mean = float(prime["post_rating"].mean())
        peak_rating = float(races["post_rating"].max())
        career_raw = cfg.prime_mean_weight * prime_mean + cfg.prime_peak_weight * peak_rating
        race_count = len(races)
        starting_rating = bootstrap_ratings.get(driver, cfg.initial_rating)
        experience = race_count / (race_count + cfg.stabilization_races)
        # Shrinks toward the shared flat reference, not toward this driver's
        # own bootstrap -- same reason v7 shrinks toward one fixed constant
        # for everyone. The bootstrap already does its job by seeding the
        # race-by-race walk (and so, correctly, prime/peak/current above);
        # anchoring the *summary* stat's regularization on a value that's
        # itself an estimate would let a miscalibrated bootstrap corrupt the
        # career number permanently, even once experience is high enough
        # that it should be dominated by real accumulated results instead.
        career_rating = cfg.initial_rating + experience * (career_raw - cfg.initial_rating)
        ranked_rows.append(
            {
                "driver": driver,
                "races": race_count,
                "races_excluded_mechanical": int(races["excluded_from_update"].sum()),
                "wins": int((races["finish_rank"] == 0).sum()),
                "accidents": int((races["status_category"] == "accident").sum()),
                "starting_rating": starting_rating,
                "expected_percentile_sum": float(races["XP"].sum()),
                "position_above_expected": float(races["position_surprise"].sum()),
                "current_rating": float(races.iloc[-1]["post_rating"]),
                "prime_mean_rating": prime_mean,
                "peak_rating": peak_rating,
                "experience_weight": experience,
                "career_rating": career_rating,
            }
        )

    ranking = pd.DataFrame(ranked_rows).sort_values(
        ["career_rating", "peak_rating", "driver"], ascending=[False, False, True], kind="stable"
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    low, high = ranking["career_rating"].min(), ranking["career_rating"].max()
    ranking["score_0_100"] = 50.0 if np.isclose(low, high) else 100.0 * (ranking["career_rating"] - low) / (high - low)
    return ranking.reset_index(drop=True), history
