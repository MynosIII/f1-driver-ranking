from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model_rules_v6 import SessionEloConfigV6, qualifying_elo_deltas_v6
from .pairwise_margin_v7_4 import (
    PairwiseMarginConfigV74,
    PairwiseMarginResultV74,
    build_pairwise_margin_adjustments_v7_4,
)
from .rookie_backcast_v7 import _bounded_zero_sum, _pairwise_expected, _validate
from .rookie_backcast_v7_2 import (
    RookieBackcastConfigV72,
    RookieBackcastResultV72,
    _context_adjustment,
    _update_context,
    calendar_k_schedule_v7_2,
    estimate_rookie_debut_ratings_v7_2,
    run_frozen_retrospective_history_v7_2,
)
from .season_scope_v7_5 import (
    SeasonScopeResultV75,
    exclude_event_keys_v7_5,
    filter_historical_indianapolis_v7_5,
)


@dataclass(slots=True)
class AnnualAnchorRatingResultV75:
    debut_estimates: pd.DataFrame
    provisional_history: pd.DataFrame
    causal_history: pd.DataFrame
    retrospective_history: pd.DataFrame
    car_crossfit: pd.DataFrame
    season_anchors: pd.DataFrame
    audit: dict[str, Any]


@dataclass(slots=True)
class RookieBackcastResultV75:
    debut_estimates: pd.DataFrame
    provisional_history: pd.DataFrame
    causal_history: pd.DataFrame
    retrospective_history: pd.DataFrame
    car_crossfit: pd.DataFrame
    season_anchors: pd.DataFrame
    position_only_baseline_history: pd.DataFrame
    position_only_season_anchors: pd.DataFrame
    driver_race_margin_features: pd.DataFrame
    pairwise_margin_features: pd.DataFrame
    pairwise_calibration: pd.DataFrame
    excluded_events: pd.DataFrame
    excluded_driver_ids: tuple[str, ...]
    audit: dict[str, Any]


def _replay_provisional_history_v7_5(
    history: pd.DataFrame,
    config: RookieBackcastConfigV72,
) -> pd.DataFrame:
    """Run a causal replay with one frozen, observed anchor per season.

    A season's rookie anchor is the median of one terminal rating per driver from
    the immediately preceding available season. Mid-season rookies therefore use
    exactly the same anchor as opening-round rookies. No blending coefficient or
    era target is imposed.
    """

    data = _validate(history)
    schedule = calendar_k_schedule_v7_2(data, config)
    data = data.merge(schedule, on="season", how="left", validate="many_to_one")
    ratings: dict[str, float] = {}
    debut_anchors: dict[str, float] = {}
    debut_reference_seasons: dict[str, float] = {}
    debut_reference_counts: dict[str, int] = {}
    last_season: dict[str, int] = {}
    context_history: dict[str, dict[str, object]] = {}
    terminal_by_season: dict[int, dict[str, float]] = {}
    season_anchors: dict[int, tuple[float, float, int, str]] = {}
    outputs: list[pd.DataFrame] = []

    for _, race in data.groupby("event_index", sort=True):
        race = race.copy()
        season = int(race["season"].iloc[0])
        if season not in season_anchors:
            prior_seasons = sorted(value for value in terminal_by_season if value < season)
            if prior_seasons:
                reference_season = prior_seasons[-1]
                terminal_ratings = list(terminal_by_season[reference_season].values())
                season_anchor = float(np.median(terminal_ratings))
                reference_count = len(terminal_ratings)
                method = "previous_season_terminal_driver_median"
            else:
                reference_season = np.nan
                season_anchor = float(config.default_field_anchor)
                reference_count = 0
                method = "inaugural_common_field_anchor"
            season_anchors[season] = (
                season_anchor,
                float(reference_season),
                reference_count,
                method,
            )
        season_anchor, reference_season, reference_count, method = season_anchors[season]
        current_driver_ids = race["driver_id"].astype(str).tolist()
        for driver_id in current_driver_ids:
            if driver_id not in debut_anchors:
                debut_anchors[driver_id] = season_anchor
                debut_reference_seasons[driver_id] = reference_season
                debut_reference_counts[driver_id] = reference_count
            if driver_id not in ratings:
                ratings[driver_id] = season_anchor

        pre_values: list[float] = []
        inactivity_gaps: list[int] = []
        return_multipliers: list[float] = []
        context_adjustments: list[float] = []
        context_samples: list[int] = []
        for _, row in race.iterrows():
            driver_id = str(row["driver_id"])
            pre_values.append(ratings[driver_id])
            gap = max(season - last_season.get(driver_id, season) - 1, 0)
            inactivity_gaps.append(gap)
            return_multipliers.append(
                min(
                    config.maximum_return_multiplier,
                    1.0 + config.return_multiplier_per_season * gap,
                )
            )
            adjustment, samples = _context_adjustment(context_history.get(driver_id), row)
            context_adjustments.append(adjustment)
            context_samples.append(samples)

        race["provisional_pre_qualifying_rating"] = pre_values
        race["provisional_debut_anchor"] = race["driver_id"].astype(str).map(debut_anchors)
        race["provisional_anchor_reference_season"] = (
            race["driver_id"].astype(str).map(debut_reference_seasons)
        )
        race["provisional_anchor_reference_driver_count"] = (
            race["driver_id"].astype(str).map(debut_reference_counts)
        )
        race["provisional_season_anchor"] = season_anchor
        race["provisional_season_anchor_reference_season"] = reference_season
        race["provisional_season_anchor_reference_driver_count"] = reference_count
        race["provisional_season_anchor_method"] = method
        race["inactivity_seasons"] = inactivity_gaps
        race["return_uncertainty_multiplier"] = return_multipliers
        race["driver_context_adjustment"] = context_adjustments
        race["driver_context_prior_samples"] = context_samples
        race["pre_session_strength"] = (
            race["provisional_pre_qualifying_rating"]
            + config.car_rating_points_per_z
            * pd.to_numeric(race["car_suitability_score_v6"], errors="coerce").fillna(0.0)
        )
        multiplier = float(race["calendar_k_multiplier"].iloc[0])
        qualifying_config = SessionEloConfigV6(
            race_k_factor=config.qualifying_base_race_k_factor * multiplier,
            qualifying_fraction=config.qualifying_fraction,
            elo_scale=config.elo_scale,
            maximum_qualifying_delta=config.maximum_qualifying_delta * multiplier,
        )
        race["provisional_qualifying_delta"] = qualifying_elo_deltas_v6(
            race, qualifying_config
        ).fillna(0.0)
        race["provisional_pre_race_rating"] = (
            race["provisional_pre_qualifying_rating"]
            + race["provisional_qualifying_delta"]
        )
        race["provisional_driver_expected_from_rating"] = _pairwise_expected(
            race["provisional_pre_race_rating"].to_numpy(dtype=float), config.elo_scale
        )
        strengths = (
            race["provisional_pre_race_rating"]
            + config.car_rating_points_per_z
            * pd.to_numeric(race["car_suitability_score_v6"], errors="coerce").fillna(0.0)
        )
        race["provisional_expected_performance"] = np.clip(
            _pairwise_expected(strengths.to_numpy(dtype=float), config.elo_scale)
            + race["driver_context_adjustment"].to_numpy(dtype=float),
            0.0,
            1.0,
        )
        eligible = race["rating_eligible"] & race["observed_performance"].notna()
        race["provisional_performance_residual"] = (
            race["observed_performance"] - race["provisional_expected_performance"]
        )
        raw = (
            race.loc[eligible, "effective_race_k"].to_numpy(dtype=float)
            * race.loc[eligible, "provisional_performance_residual"].to_numpy(dtype=float)
            * race.loc[eligible, "return_uncertainty_multiplier"].to_numpy(dtype=float)
        )
        race["provisional_race_delta"] = 0.0
        race.loc[eligible, "provisional_race_delta"] = _bounded_zero_sum(
            raw, float(race["effective_max_race_delta"].iloc[0])
        )
        race["provisional_post_rating"] = (
            race["provisional_pre_race_rating"] + race["provisional_race_delta"]
        )

        for _, row in race.iterrows():
            driver_id = str(row["driver_id"])
            rating = float(row["provisional_post_rating"])
            ratings[driver_id] = rating
            last_season[driver_id] = season
            terminal_by_season.setdefault(season, {})[driver_id] = rating
            if bool(row["rating_eligible"]) and np.isfinite(row["observed_performance"]):
                context_history[driver_id] = _update_context(
                    context_history.get(driver_id),
                    row,
                    float(row["provisional_performance_residual"]),
                )
        outputs.append(race)

    result = pd.concat(outputs, ignore_index=True).sort_values(
        ["season", "round", "position"], kind="stable"
    )
    result["causal_rating"] = result["provisional_post_rating"]
    result["provisional_rating"] = result["provisional_post_rating"]
    result["rating_history_scope"] = (
        "causal_previous_season_terminal_median_anchor_calendar_normalized_v7_5"
    )
    return result


def _annual_anchor_table_v7_5(provisional: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "provisional_season_anchor",
        "provisional_season_anchor_reference_season",
        "provisional_season_anchor_reference_driver_count",
        "provisional_season_anchor_method",
    ]
    return (
        provisional[columns]
        .drop_duplicates()
        .sort_values("season", kind="stable")
        .reset_index(drop=True)
        .rename(
            columns={
                "provisional_season_anchor": "season_rookie_anchor",
                "provisional_season_anchor_reference_season": "anchor_reference_season",
                "provisional_season_anchor_reference_driver_count": (
                    "anchor_reference_driver_count"
                ),
                "provisional_season_anchor_method": "season_anchor_method",
            }
        )
    )


def build_annual_anchor_rating_v7_5(
    history: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> AnnualAnchorRatingResultV75:
    cfg = config or RookieBackcastConfigV72()
    data = _validate(history)
    provisional = _replay_provisional_history_v7_5(data, cfg)
    debut, car_crossfit = estimate_rookie_debut_ratings_v7_2(provisional, cfg)
    debut["debut_anchor_source"] = (
        "previous_season_full_field_terminal_driver_median_v7_5"
    )
    debut["rookie_opponent_rating_source"] = "provisional_pre_race_rating_v7_5"
    car_crossfit["car_signal_basis"] = "provisional_driver_expected_from_rating_v7_5"
    retrospective = run_frozen_retrospective_history_v7_2(data, debut, cfg)
    retrospective["rating_history_scope"] = (
        "retrospective_first_five_backcast_previous_season_anchor_frozen_v7_5"
    )
    causal = provisional.copy()
    anchors = _annual_anchor_table_v7_5(provisional)
    schedule = calendar_k_schedule_v7_2(data, cfg)
    target_race_budget = cfg.race_k_factor * float(
        schedule["calendar_reference_rounds"].iloc[0]
    )
    target_qualifying_budget = (
        cfg.qualifying_base_race_k_factor
        * cfg.qualifying_fraction
        * float(schedule["calendar_reference_rounds"].iloc[0])
    )
    terminal = (
        provisional.sort_values("event_index", kind="stable")
        .groupby(["season", "driver_id"], as_index=False)
        .tail(1)
    )
    terminal_medians = terminal.groupby("season")["provisional_post_rating"].median()
    anchor_errors: list[float] = []
    for row in anchors.itertuples(index=False):
        if pd.isna(row.anchor_reference_season):
            continue
        expected = float(terminal_medians.loc[int(row.anchor_reference_season)])
        anchor_errors.append(abs(float(row.season_rookie_anchor) - expected))
    debut_season_anchor_counts = (
        debut.groupby("debut_season")["debut_field_anchor"].nunique(dropna=False)
    )
    audit = {
        "version": "7.5",
        "drivers": int(debut["driver_id"].nunique()),
        "history_rows": int(len(retrospective)),
        "provisional_history_rows": int(len(provisional)),
        "calendar_reference_rounds": float(schedule["calendar_reference_rounds"].iloc[0]),
        "minimum_season_rounds": int(schedule["season_rounds"].min()),
        "maximum_season_rounds": int(schedule["season_rounds"].max()),
        "target_season_race_k_budget": target_race_budget,
        "target_season_qualifying_k_budget": target_qualifying_budget,
        "race_k_budget_max_error": float(
            (schedule["scheduled_race_k_budget"] - target_race_budget).abs().max()
        ),
        "qualifying_k_budget_max_error": float(
            (schedule["scheduled_qualifying_k_budget"] - target_qualifying_budget)
            .abs()
            .max()
        ),
        "debut_anchor_source": (
            "previous_season_full_field_terminal_driver_median_v7_5"
        ),
        "annual_anchor_blend_coefficient": None,
        "annual_anchor_target_level": None,
        "annual_anchor_max_validation_error": max(anchor_errors, default=0.0),
        "seasons_with_multiple_rookie_anchors": int(
            debut_season_anchor_counts.gt(1).sum()
        ),
        "rookie_window_rows_used_in_car_fit": int(
            car_crossfit["rookie_window_row_used_in_car_fit"].sum()
        ),
        "complete_first_five_windows": int(debut["rookie_window_complete"].sum()),
        "partial_windows": int(
            debut["rookie_eligible_races"].between(1, cfg.rookie_eligible_races - 1).sum()
        ),
        "population_prior_only": int(debut["rookie_eligible_races"].eq(0).sum()),
        "rookie_population_offset": float(debut["rookie_population_offset"].iloc[0]),
        "rookie_population_scale": float(debut["rookie_population_scale"].iloc[0]),
        "mean_rookie_reliability": float(debut["rookie_reliability"].mean()),
        "minimum_debut_rating": float(debut["debut_rating_final"].min()),
        "maximum_debut_rating": float(debut["debut_rating_final"].max()),
        "inaugural_offset_mean": float(
            debut.loc[debut["debut_season"].eq(1950), "rookie_offset_shrunk"].mean()
        ),
        "provisional_qualifying_zero_sum_max_error": float(
            provisional.groupby(["season", "round"])["provisional_qualifying_delta"]
            .sum()
            .abs()
            .max()
        ),
        "provisional_race_zero_sum_max_error": float(
            provisional.groupby(["season", "round"])["provisional_race_delta"]
            .sum()
            .abs()
            .max()
        ),
        "qualifying_zero_sum_max_error": float(
            retrospective.groupby(["season", "round"])["retrospective_qualifying_delta"]
            .sum()
            .abs()
            .max()
        ),
        "race_zero_sum_max_error": float(
            retrospective.groupby(["season", "round"])["retrospective_race_delta"]
            .sum()
            .abs()
            .max()
        ),
        "frozen_debut_rating_mismatches": int(
            retrospective.groupby("driver_id")["debut_rating_frozen"].nunique().ne(1).sum()
        ),
        "final_replay_count": 1,
    }
    return AnnualAnchorRatingResultV75(
        debut,
        provisional,
        causal,
        retrospective,
        car_crossfit,
        anchors,
        audit,
    )


def build_rookie_backcast_v7_5(
    history: pd.DataFrame,
    timing: pd.DataFrame,
    race_context: pd.DataFrame | None = None,
    rating_config: RookieBackcastConfigV72 | None = None,
    margin_config: PairwiseMarginConfigV74 | None = None,
) -> RookieBackcastResultV75:
    rating_cfg = rating_config or RookieBackcastConfigV72()
    margin_cfg = margin_config or PairwiseMarginConfigV74()
    scope: SeasonScopeResultV75 = filter_historical_indianapolis_v7_5(history)
    scoped_timing = exclude_event_keys_v7_5(timing, scope.excluded_events)
    scoped_context = exclude_event_keys_v7_5(race_context, scope.excluded_events)
    baseline: RookieBackcastResultV72 | AnnualAnchorRatingResultV75
    baseline = build_annual_anchor_rating_v7_5(scope.history, rating_cfg)
    margin: PairwiseMarginResultV74 = build_pairwise_margin_adjustments_v7_4(
        scope.history,
        scoped_timing,
        baseline.retrospective_history,
        scoped_context,
        margin_cfg,
    )
    rating = build_annual_anchor_rating_v7_5(margin.adjusted_history, rating_cfg)
    audit = {
        **rating.audit,
        **margin.audit,
        **scope.audit,
        "version": "7.5",
        "baseline_expected_strength_source": (
            "v7_5_position_only_retrospective_rating_plus_exact_car_suitability_"
            "and_causal_driver_context"
        ),
        "expected_gap_driver_component": "retrospective_pre_race_rating",
        "expected_gap_car_component": "80_rating_points_times_car_suitability_score_v6",
        "expected_gap_driver_circuit_component": (
            "260_rating_points_times_causal_driver_context_adjustment"
        ),
        "expected_gap_distance_scaling": (
            "calibrated_gap_share_times_observed_winner_race_time_seconds"
        ),
        "expected_gap_shorter_race_property": (
            "same_expected_share_produces_fewer_expected_seconds"
        ),
        "safety_car_adjustment": "none_intentionally_v7_5",
        "era_normalization": "none",
        "position_performance_weight_role": "primary_order_signal",
        "gap_performance_weight_role": "second_primary_signal",
        "all_classified_pairs_used": True,
        "same_car_crossfit_enabled": True,
        "position_only_history_preserved": True,
        "final_replay_count": 1,
    }
    return RookieBackcastResultV75(
        rating.debut_estimates,
        rating.provisional_history,
        rating.causal_history,
        rating.retrospective_history,
        rating.car_crossfit,
        rating.season_anchors,
        baseline.retrospective_history,
        baseline.season_anchors,
        margin.driver_race_features,
        margin.pairwise_features,
        margin.calibration,
        scope.excluded_events,
        scope.excluded_driver_ids,
        audit,
    )
