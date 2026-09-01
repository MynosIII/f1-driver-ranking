from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .rookie_backcast_v7 import _bounded_zero_sum, _layout_similarity


@dataclass(frozen=True, slots=True)
class PairwiseMarginConfigV74:
    elo_scale: float = 260.0
    car_rating_points_per_z: float = 80.0
    gap_performance_weight: float = 0.40
    maximum_performance_adjustment: float = 0.20
    same_car_comparison_weight: float = 3.0
    lap_equivalent_reliability: float = 0.70
    season_decay_years: float = 12.0
    minimum_gap_slope: float = 1e-4
    maximum_gap_slope: float = 0.02
    minimum_residual_scale: float = 2e-4
    maximum_residual_scale: float = 0.03
    residual_tanh_scale: float = 2.0
    reference_opponents: float = 19.0
    minimum_field_depth_factor: float = 0.60
    maximum_field_depth_factor: float = 1.10
    car_crossfit_prior_rows: float = 8.0
    maximum_car_margin_component: float = 0.08


@dataclass(slots=True)
class PairwiseMarginResultV74:
    adjusted_history: pd.DataFrame
    driver_race_features: pd.DataFrame
    pairwise_features: pd.DataFrame
    calibration: pd.DataFrame
    audit: dict[str, Any]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = 0.5 * float(ordered_weights.sum())
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), threshold)])


def _timing_state_v7_4(
    timing: pd.DataFrame,
    config: PairwiseMarginConfigV74,
    race_context: pd.DataFrame | None,
) -> pd.DataFrame:
    required = {
        "season",
        "round",
        "result_order",
        "driver_id_f1db",
        "laps",
        "race_time_seconds",
        "gap_seconds",
        "gap_laps",
    }
    if missing := required - set(timing.columns):
        raise ValueError(f"Timing is missing columns: {sorted(missing)}")
    source = timing.copy()
    if race_context is None:
        source["race_completion_fraction_v7_4"] = 1.0
        source["race_completion_reliability_v7_4"] = 1.0
        source["race_completion_quality_v7_4"] = "context_not_supplied_assume_full"
    else:
        context_columns = [
            "season",
            "round",
            "race_completion_fraction_v7_4",
            "race_completion_reliability_v7_4",
            "race_completion_quality_v7_4",
        ]
        source = source.merge(
            race_context[context_columns],
            on=["season", "round"],
            how="left",
            validate="many_to_one",
        )
        source["race_completion_reliability_v7_4"] = source[
            "race_completion_reliability_v7_4"
        ].fillna(1.0)
        source["race_completion_quality_v7_4"] = source[
            "race_completion_quality_v7_4"
        ].fillna("context_row_missing_assume_full")
    pieces: list[pd.DataFrame] = []
    for _, race in source.groupby(["season", "round"], sort=True):
        ordered = race.sort_values("result_order", kind="stable").copy()
        winner = ordered.iloc[0]
        winner_time = float(winner["race_time_seconds"])
        winner_laps = float(winner["laps"])
        ordered["winner_race_time_seconds"] = winner_time
        ordered["winner_race_laps"] = winner_laps
        ordered["cumulative_gap_share"] = np.nan
        ordered["cumulative_gap_reliability"] = 0.0
        ordered["cumulative_gap_quality"] = "unavailable"
        if np.isfinite(winner_time) and winner_time > 0.0:
            winner_mask = ordered["result_order"].eq(int(winner["result_order"]))
            exact_mask = ordered["gap_seconds"].notna()
            ordered.loc[winner_mask, "cumulative_gap_share"] = 0.0
            ordered.loc[winner_mask, "cumulative_gap_reliability"] = 1.0
            ordered.loc[winner_mask, "cumulative_gap_quality"] = "winner_origin"
            ordered.loc[exact_mask, "cumulative_gap_share"] = (
                ordered.loc[exact_mask, "gap_seconds"].astype(float) / winner_time
            )
            ordered.loc[exact_mask, "cumulative_gap_reliability"] = 1.0
            ordered.loc[exact_mask, "cumulative_gap_quality"] = "exact_seconds"
            if np.isfinite(winner_laps) and winner_laps > 0.0:
                lap_mask = ordered["gap_laps"].notna() & ~exact_mask
                ordered.loc[lap_mask, "cumulative_gap_share"] = (
                    ordered.loc[lap_mask, "gap_laps"].astype(float) / winner_laps
                )
                ordered.loc[lap_mask, "cumulative_gap_reliability"] = (
                    config.lap_equivalent_reliability
                )
                ordered.loc[lap_mask, "cumulative_gap_quality"] = "lap_equivalent"
        # Shared drives and multiple entries are valid in F1DB, but the rating
        # history has one driver row per race. Match the driver's best classified entry.
        ordered = ordered.drop_duplicates("driver_id_f1db", keep="first")
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True)


def build_pairwise_gap_features_v7_4(
    history: pd.DataFrame,
    timing: pd.DataFrame,
    baseline_history: pd.DataFrame,
    race_context: pd.DataFrame | None = None,
    config: PairwiseMarginConfigV74 | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create every usable classified-driver pair and its actual timing gap."""

    cfg = config or PairwiseMarginConfigV74()
    required_history = {
        "season",
        "round",
        "driver_id",
        "driver_id_f1db",
        "driver",
        "status_class",
        "observed_performance",
        "model_entity_id",
        "layout_key",
        "straight_demand",
        "altitude_demand",
        "car_suitability_score_v6",
    }
    if missing := required_history - set(history.columns):
        raise ValueError(f"History is missing pairwise columns: {sorted(missing)}")
    required_baseline = {
        "season",
        "round",
        "driver_id",
        "retrospective_pre_race_rating",
        "driver_context_adjustment",
    }
    if missing := required_baseline - set(baseline_history.columns):
        raise ValueError(f"Baseline is missing pairwise columns: {sorted(missing)}")

    data = history.copy()
    data["driver_id"] = data["driver_id"].astype(str)
    data["driver_id_f1db"] = data["driver_id_f1db"].astype(str)
    baseline = baseline_history[
        [
            "season",
            "round",
            "driver_id",
            "retrospective_pre_race_rating",
            "driver_context_adjustment",
        ]
    ].copy()
    baseline["driver_id"] = baseline["driver_id"].astype(str)
    if baseline.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("Baseline must be unique at driver-race grain")
    data = data.merge(
        baseline,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    timing_state = _timing_state_v7_4(timing, cfg, race_context)
    timing_columns = [
        "season",
        "round",
        "driver_id_f1db",
        "result_order",
        "finish_position_text",
        "gap_laps",
        "winner_race_time_seconds",
        "winner_race_laps",
        "cumulative_gap_share",
        "cumulative_gap_reliability",
        "cumulative_gap_quality",
        "race_completion_fraction_v7_4",
        "race_completion_reliability_v7_4",
        "race_completion_quality_v7_4",
    ]
    data = data.merge(
        timing_state[timing_columns],
        on=["season", "round", "driver_id_f1db"],
        how="left",
        validate="one_to_one",
    )
    data["pairwise_pre_race_strength_v7_4"] = (
        pd.to_numeric(data["retrospective_pre_race_rating"], errors="coerce")
        + cfg.car_rating_points_per_z
        * pd.to_numeric(data["car_suitability_score_v6"], errors="coerce").fillna(0.0)
        + cfg.elo_scale
        * pd.to_numeric(data["driver_context_adjustment"], errors="coerce").fillna(0.0)
    )
    data["field_xw_v7_4"] = np.nan
    pair_rows: list[dict[str, Any]] = []
    for (season, round_number), race in data.groupby(["season", "round"], sort=True):
        race = race.copy()
        strengths = race["pairwise_pre_race_strength_v7_4"].to_numpy(dtype=float)
        finite = np.isfinite(strengths)
        if finite.any():
            centered = np.clip((strengths[finite] - np.max(strengths[finite])) / cfg.elo_scale, -30, 0)
            exponentials = np.exp(centered)
            data.loc[race.index[finite], "field_xw_v7_4"] = exponentials / exponentials.sum()
        eligible = race[
            race["status_class"].astype(str).eq("finished")
            & race["finish_position_text"].astype(str).str.fullmatch(r"\d+")
            & race["cumulative_gap_share"].notna()
            & race["pairwise_pre_race_strength_v7_4"].notna()
        ].copy()
        eligible = eligible.sort_values("driver_id", kind="stable")
        field_size = int(len(eligible))
        if field_size <= 1:
            continue
        event_pairs: list[dict[str, Any]] = []
        for left_index, right_index in combinations(eligible.index, 2):
            left = eligible.loc[left_index]
            right = eligible.loc[right_index]
            same_unresolved_lap = (
                left["cumulative_gap_quality"] == "lap_equivalent"
                and right["cumulative_gap_quality"] == "lap_equivalent"
                and float(left["gap_laps"]) == float(right["gap_laps"])
            )
            reliability = min(
                float(left["cumulative_gap_reliability"]),
                float(right["cumulative_gap_reliability"]),
            )
            if same_unresolved_lap:
                reliability = 0.0
            completion_reliability = float(
                left["race_completion_reliability_v7_4"]
            )
            reliability *= completion_reliability
            strength_difference = float(
                left["pairwise_pre_race_strength_v7_4"]
                - right["pairwise_pre_race_strength_v7_4"]
            )
            xw_logit = strength_difference / cfg.elo_scale
            pairwise_xw = 1.0 / (1.0 + np.exp(-np.clip(xw_logit, -30.0, 30.0)))
            same_car = str(left["model_entity_id"]) == str(right["model_entity_id"])
            event_pairs.append(
                {
                    "season": int(season),
                    "round": int(round_number),
                    "driver_a_id": str(left["driver_id"]),
                    "driver_a": left["driver"],
                    "driver_b_id": str(right["driver_id"]),
                    "driver_b": right["driver"],
                    "driver_a_finish_position": left.get("position"),
                    "driver_b_finish_position": right.get("position"),
                    "driver_a_field_xw": data.loc[left_index, "field_xw_v7_4"],
                    "driver_b_field_xw": data.loc[right_index, "field_xw_v7_4"],
                    "driver_a_pairwise_xw": pairwise_xw,
                    "pairwise_xw_logit_a_minus_b": xw_logit,
                    "observed_gap_share_a_minus_b": float(
                        right["cumulative_gap_share"] - left["cumulative_gap_share"]
                    ),
                    "winner_race_time_seconds": float(
                        left["winner_race_time_seconds"]
                    ),
                    "pair_timing_reliability": reliability,
                    "race_completion_fraction_v7_4": left[
                        "race_completion_fraction_v7_4"
                    ],
                    "race_completion_reliability_v7_4": completion_reliability,
                    "same_car_model": same_car,
                    "pair_attribution_weight": reliability
                    * (cfg.same_car_comparison_weight if same_car else 1.0),
                    "eligible_timed_field_size": field_size,
                }
            )
        usable_count = sum(row["pair_timing_reliability"] > 0.0 for row in event_pairs)
        for row in event_pairs:
            row["usable_pair_count_in_race"] = usable_count
        pair_rows.extend(event_pairs)
    return data, pd.DataFrame(pair_rows)


def calibrate_expected_pairwise_gaps_v7_4(
    pairs: pd.DataFrame,
    config: PairwiseMarginConfigV74 | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a positive leave-season-out xW-to-seconds relationship for every era."""

    cfg = config or PairwiseMarginConfigV74()
    output = pairs.copy()
    usable = output[
        output["pair_timing_reliability"].gt(0.0)
        & output["observed_gap_share_a_minus_b"].notna()
        & output["pairwise_xw_logit_a_minus_b"].notna()
    ].copy()
    calibration_rows: list[dict[str, Any]] = []
    for target_season in sorted(output["season"].unique()):
        candidates = usable[usable["season"].ne(int(target_season))]
        season_distance = np.abs(
            candidates["season"].to_numpy(dtype=float) - float(target_season)
        )
        race_normalizer = candidates["usable_pair_count_in_race"].clip(lower=1).to_numpy(
            dtype=float
        )
        weights = (
            np.exp(-season_distance / cfg.season_decay_years)
            * candidates["pair_timing_reliability"].to_numpy(dtype=float)
            / race_normalizer
        )
        x_values = candidates["pairwise_xw_logit_a_minus_b"].to_numpy(dtype=float)
        y_values = candidates["observed_gap_share_a_minus_b"].to_numpy(dtype=float)
        raw_slope = float(
            np.sum(weights * x_values * y_values)
            / max(float(np.sum(weights * np.square(x_values))), 1e-12)
        )
        slope = float(np.clip(raw_slope, cfg.minimum_gap_slope, cfg.maximum_gap_slope))
        residuals = y_values - slope * x_values
        residual_center = _weighted_median(residuals, weights)
        mad = _weighted_median(np.abs(residuals - residual_center), weights)
        residual_scale = float(
            np.clip(1.4826 * mad, cfg.minimum_residual_scale, cfg.maximum_residual_scale)
        )
        effective_races = float(
            weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12)
        )
        calibration_rows.append(
            {
                "season": int(target_season),
                "pairwise_gap_slope": slope,
                "pairwise_gap_raw_slope": raw_slope,
                "pairwise_gap_residual_center": residual_center,
                "pairwise_gap_residual_scale": residual_scale,
                "pairwise_gap_effective_training_races": effective_races,
                "target_season_excluded": True,
            }
        )
    calibration = pd.DataFrame(calibration_rows)
    output = output.merge(calibration, on="season", how="left", validate="many_to_one")
    output["expected_gap_share_a_minus_b"] = (
        output["pairwise_gap_slope"] * output["pairwise_xw_logit_a_minus_b"]
    ).clip(-0.15, 0.15)
    output["observed_gap_seconds_a_minus_b"] = (
        output["observed_gap_share_a_minus_b"]
        * output["winner_race_time_seconds"]
    )
    output["expected_gap_seconds_a_minus_b"] = (
        output["expected_gap_share_a_minus_b"]
        * output["winner_race_time_seconds"]
    )
    output["pair_gap_residual_share"] = (
        output["observed_gap_share_a_minus_b"]
        - output["expected_gap_share_a_minus_b"]
    )
    output["pair_gap_residual_score_a"] = 0.5 * np.tanh(
        (
            output["pair_gap_residual_share"]
            - output["pairwise_gap_residual_center"]
        )
        / (cfg.residual_tanh_scale * output["pairwise_gap_residual_scale"])
    )
    output.loc[
        output["pair_timing_reliability"].le(0.0), "pair_gap_residual_score_a"
    ] = np.nan
    output["pair_gap_model"] = (
        "all_classified_pairs_leave_season_out_era_calibrated_xw_seconds_v7_4"
    )
    return output, calibration


def aggregate_driver_margin_features_v7_4(
    history_with_timing: pd.DataFrame,
    calibrated_pairs: pd.DataFrame,
    config: PairwiseMarginConfigV74 | None = None,
) -> pd.DataFrame:
    cfg = config or PairwiseMarginConfigV74()
    usable = calibrated_pairs[calibrated_pairs["pair_gap_residual_score_a"].notna()].copy()
    left = usable[
        [
            "season",
            "round",
            "driver_a_id",
            "pair_gap_residual_score_a",
            "pair_attribution_weight",
            "same_car_model",
            "eligible_timed_field_size",
            "race_completion_reliability_v7_4",
        ]
    ].rename(
        columns={
            "driver_a_id": "driver_id",
            "pair_gap_residual_score_a": "pair_gap_residual_score",
        }
    )
    right = usable[
        [
            "season",
            "round",
            "driver_b_id",
            "pair_gap_residual_score_a",
            "pair_attribution_weight",
            "same_car_model",
            "eligible_timed_field_size",
            "race_completion_reliability_v7_4",
        ]
    ].rename(
        columns={
            "driver_b_id": "driver_id",
            "pair_gap_residual_score_a": "pair_gap_residual_score",
        }
    )
    right["pair_gap_residual_score"] *= -1.0
    long = pd.concat([left, right], ignore_index=True)
    long["weighted_pair_gap_residual"] = (
        long["pair_gap_residual_score"] * long["pair_attribution_weight"]
    )
    grouped = long.groupby(["season", "round", "driver_id"], as_index=False).agg(
        margin_comparison_count=("pair_gap_residual_score", "count"),
        margin_comparison_weight=("pair_attribution_weight", "sum"),
        weighted_pair_gap_residual=("weighted_pair_gap_residual", "sum"),
        same_car_margin_comparisons=("same_car_model", "sum"),
        eligible_timed_field_size=("eligible_timed_field_size", "max"),
        margin_race_completion_reliability=(
            "race_completion_reliability_v7_4",
            "max",
        ),
    )
    grouped["mean_pair_gap_residual_score"] = (
        grouped["weighted_pair_gap_residual"] / grouped["margin_comparison_weight"]
    )
    possible_opponents = (grouped["eligible_timed_field_size"] - 1).clip(lower=1)
    grouped["margin_pair_coverage"] = (
        grouped["margin_comparison_count"] / possible_opponents
    ).clip(0.0, 1.0)
    grouped["margin_field_depth_factor"] = np.sqrt(
        possible_opponents / cfg.reference_opponents
    ).clip(cfg.minimum_field_depth_factor, cfg.maximum_field_depth_factor)
    grouped["raw_pairwise_margin_performance_adjustment"] = (
        cfg.gap_performance_weight
        * grouped["mean_pair_gap_residual_score"]
        * np.sqrt(grouped["margin_pair_coverage"])
        * grouped["margin_field_depth_factor"]
    ).clip(-cfg.maximum_performance_adjustment, cfg.maximum_performance_adjustment)

    base = history_with_timing.copy()
    base["driver_id"] = base["driver_id"].astype(str)
    features = base.merge(
        grouped,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )
    numeric_defaults = {
        "margin_comparison_count": 0,
        "margin_comparison_weight": 0.0,
        "weighted_pair_gap_residual": 0.0,
        "same_car_margin_comparisons": 0,
        "margin_pair_coverage": 0.0,
        "margin_field_depth_factor": 0.0,
        "raw_pairwise_margin_performance_adjustment": 0.0,
        "margin_race_completion_reliability": 0.0,
    }
    for column, default in numeric_defaults.items():
        features[column] = features[column].fillna(default)

    eligible_pool = features[
        features["margin_comparison_count"].gt(0)
        & features["raw_pairwise_margin_performance_adjustment"].notna()
    ]
    groups = {
        key: group
        for key, group in eligible_pool.groupby(["season", "model_entity_id"], sort=False)
    }
    car_components: list[float] = []
    car_effective_rows: list[float] = []
    car_other_drivers: list[int] = []
    for _, target in features.iterrows():
        candidates = groups.get((int(target["season"]), target["model_entity_id"]))
        if candidates is None:
            candidates = features.iloc[0:0]
        else:
            candidates = candidates[
                candidates["driver_id"].astype(str).ne(str(target["driver_id"]))
            ]
        if candidates.empty:
            car_components.append(0.0)
            car_effective_rows.append(0.0)
            car_other_drivers.append(0)
            continue
        weights = _layout_similarity(target, candidates)
        same_layout = candidates["layout_key"].astype(str).eq(str(target["layout_key"]))
        weights *= np.where(same_layout, 1.5, 1.0)
        effective = float(weights.sum())
        raw_component = float(
            np.average(
                candidates["raw_pairwise_margin_performance_adjustment"].to_numpy(
                    dtype=float
                ),
                weights=weights,
            )
        )
        component = raw_component * effective / (effective + cfg.car_crossfit_prior_rows)
        car_components.append(
            float(
                np.clip(
                    component,
                    -cfg.maximum_car_margin_component,
                    cfg.maximum_car_margin_component,
                )
            )
        )
        car_effective_rows.append(effective)
        car_other_drivers.append(int(candidates["driver_id"].nunique()))
    features["crossfit_car_margin_component"] = car_components
    features["crossfit_car_margin_effective_rows"] = car_effective_rows
    features["crossfit_car_margin_other_drivers"] = car_other_drivers
    features["driver_margin_adjustment_before_centering"] = (
        features["raw_pairwise_margin_performance_adjustment"]
        - features["crossfit_car_margin_component"]
    ) * features["race_completion_reliability_v7_4"].fillna(1.0)
    features["pairwise_margin_performance_adjustment"] = 0.0
    for _, race in features.groupby(["season", "round"], sort=True):
        active = race["margin_comparison_count"].gt(0)
        values = race.loc[active, "driver_margin_adjustment_before_centering"].to_numpy(
            dtype=float
        )
        features.loc[race.index[active], "pairwise_margin_performance_adjustment"] = (
            _bounded_zero_sum(values, cfg.maximum_performance_adjustment)
        )
    features["observed_performance_position_only"] = features["observed_performance"]
    features["gap_adjusted_observed_performance"] = (
        features["observed_performance_position_only"]
        + features["pairwise_margin_performance_adjustment"]
    )
    features["observed_performance"] = features["gap_adjusted_observed_performance"]
    features["pairwise_margin_model"] = (
        "all_field_expected_seconds_gap_residual_same_car_crossfit_v7_4"
    )
    return features


def build_pairwise_margin_adjustments_v7_4(
    history: pd.DataFrame,
    timing: pd.DataFrame,
    baseline_history: pd.DataFrame,
    race_context: pd.DataFrame | None = None,
    config: PairwiseMarginConfigV74 | None = None,
) -> PairwiseMarginResultV74:
    cfg = config or PairwiseMarginConfigV74()
    state, pairs = build_pairwise_gap_features_v7_4(
        history, timing, baseline_history, race_context, cfg
    )
    calibrated_pairs, calibration = calibrate_expected_pairwise_gaps_v7_4(pairs, cfg)
    adjusted = aggregate_driver_margin_features_v7_4(state, calibrated_pairs, cfg)
    active = adjusted["margin_comparison_count"].gt(0)
    pair_usable = calibrated_pairs["pair_timing_reliability"].gt(0.0)
    audit = {
        "pairwise_margin_version": "7.4",
        "classified_pair_rows": int(len(calibrated_pairs)),
        "usable_timed_pair_rows": int(pair_usable.sum()),
        "usable_timed_pair_rate": float(pair_usable.mean()),
        "driver_race_rows_with_margin_comparisons": int(active.sum()),
        "driver_race_margin_coverage_rate": float(active.mean()),
        "same_car_pair_rows": int(calibrated_pairs["same_car_model"].sum()),
        "same_car_comparison_weight": cfg.same_car_comparison_weight,
        "gap_performance_weight": cfg.gap_performance_weight,
        "maximum_performance_adjustment": cfg.maximum_performance_adjustment,
        "mean_absolute_performance_adjustment": float(
            adjusted.loc[active, "pairwise_margin_performance_adjustment"].abs().mean()
        ),
        "maximum_absolute_performance_adjustment": float(
            adjusted["pairwise_margin_performance_adjustment"].abs().max()
        ),
        "margin_adjustment_zero_sum_max_error": float(
            adjusted.groupby(["season", "round"])[
                "pairwise_margin_performance_adjustment"
            ]
            .sum()
            .abs()
            .max()
        ),
        "minimum_calibrated_gap_slope": float(calibration["pairwise_gap_slope"].min()),
        "maximum_calibrated_gap_slope": float(calibration["pairwise_gap_slope"].max()),
        "all_gap_slopes_positive": bool(calibration["pairwise_gap_slope"].gt(0.0).all()),
        "target_season_used_in_gap_calibration": bool(
            (~calibration["target_season_excluded"]).any()
        ),
        "mechanical_crash_dsq_pairs": 0,
        "shortened_races_downweighted": int(
            calibrated_pairs.loc[
                calibrated_pairs["race_completion_reliability_v7_4"].lt(1.0),
                ["season", "round"],
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "minimum_race_completion_reliability": float(
            calibrated_pairs["race_completion_reliability_v7_4"].min()
        ),
        "car_margin_rows_with_other_driver_evidence": int(
            adjusted["crossfit_car_margin_other_drivers"].gt(0).sum()
        ),
    }
    driver_columns = [
        "season",
        "round",
        "driver_id",
        "driver",
        "observed_performance_position_only",
        "gap_adjusted_observed_performance",
        "margin_comparison_count",
        "same_car_margin_comparisons",
        "margin_pair_coverage",
        "margin_field_depth_factor",
        "race_completion_reliability_v7_4",
        "mean_pair_gap_residual_score",
        "raw_pairwise_margin_performance_adjustment",
        "crossfit_car_margin_component",
        "crossfit_car_margin_effective_rows",
        "crossfit_car_margin_other_drivers",
        "pairwise_margin_performance_adjustment",
    ]
    return PairwiseMarginResultV74(
        adjusted,
        adjusted[driver_columns].copy(),
        calibrated_pairs,
        calibration,
        audit,
    )
