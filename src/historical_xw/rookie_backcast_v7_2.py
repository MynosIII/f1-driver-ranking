from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model_rules_v6 import SessionEloConfigV6, qualifying_elo_deltas_v6
from .rookie_backcast_v7 import (
    _bounded_zero_sum,
    _condition_weight,
    _layout_similarity,
    _pairwise_expected,
    _validate,
    identify_rookie_windows_v7,
)


@dataclass(frozen=True, slots=True)
class RookieBackcastConfigV72:
    """v7 formulas with same-scale rookie fitting and calendar-normalized K."""

    rookie_eligible_races: int = 5
    maximum_window_seasons: int = 3
    maximum_window_entries: int = 30
    rookie_prior_equivalent_races: float = 5.0
    default_field_anchor: float = 1500.0
    elo_scale: float = 260.0
    offset_limit: float = 500.0
    offset_grid_step: float = 2.0
    car_signal_prior_races: float = 3.0
    minimum_population_scale: float = 75.0
    race_k_factor: float = 24.0
    maximum_race_delta: float = 28.0
    qualifying_base_race_k_factor: float = 18.0
    qualifying_fraction: float = 1.0 / 3.0
    maximum_qualifying_delta: float = 9.0
    car_rating_points_per_z: float = 80.0
    maximum_return_multiplier: float = 1.50
    return_multiplier_per_season: float = 0.08
    reference_season_rounds: float | None = None


@dataclass(slots=True)
class RookieBackcastResultV72:
    debut_estimates: pd.DataFrame
    provisional_history: pd.DataFrame
    causal_history: pd.DataFrame
    retrospective_history: pd.DataFrame
    car_crossfit: pd.DataFrame
    audit: dict[str, Any]


def calendar_k_schedule_v7_2(
    history: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> pd.DataFrame:
    """Return per-season K factors whose scheduled seasonal totals are constant."""

    cfg = config or RookieBackcastConfigV72()
    data = _validate(history)
    rounds = (
        data.groupby("season", as_index=False)["event_index"]
        .nunique()
        .rename(columns={"event_index": "season_rounds"})
    )
    reference = (
        float(cfg.reference_season_rounds)
        if cfg.reference_season_rounds is not None
        else float(rounds["season_rounds"].median())
    )
    if reference <= 0.0 or (rounds["season_rounds"] <= 0).any():
        raise ValueError("Calendar round counts and reference must be positive")
    rounds["calendar_reference_rounds"] = reference
    rounds["calendar_k_multiplier"] = reference / rounds["season_rounds"]
    rounds["effective_race_k"] = cfg.race_k_factor * rounds["calendar_k_multiplier"]
    rounds["effective_qualifying_k"] = (
        cfg.qualifying_base_race_k_factor
        * cfg.qualifying_fraction
        * rounds["calendar_k_multiplier"]
    )
    rounds["effective_max_race_delta"] = (
        cfg.maximum_race_delta * rounds["calendar_k_multiplier"]
    )
    rounds["effective_max_qualifying_delta"] = (
        cfg.maximum_qualifying_delta * rounds["calendar_k_multiplier"]
    )
    rounds["scheduled_race_k_budget"] = (
        rounds["season_rounds"] * rounds["effective_race_k"]
    )
    rounds["scheduled_qualifying_k_budget"] = (
        rounds["season_rounds"] * rounds["effective_qualifying_k"]
    )
    return rounds


def _new_context_state() -> dict[str, object]:
    return {
        "samples": 0,
        "xtx": np.zeros((3, 3), dtype=float),
        "xty": np.zeros(3, dtype=float),
        "wet_samples": 0,
        "wet_residual_sum": 0.0,
    }


def _context_adjustment(
    state: dict[str, object] | None, row: pd.Series
) -> tuple[float, int]:
    if state is None:
        return 0.0, 0
    samples = int(state["samples"])
    adjustments: list[float] = []
    if samples >= 5:
        xtx = np.asarray(state["xtx"], dtype=float)
        xty = np.asarray(state["xty"], dtype=float)
        coefficients = np.linalg.solve(xtx + np.diag([0.0, 8.0, 8.0]), xty)
        target = np.asarray(
            [
                1.0,
                float(row.get("straight_demand", 0.5)),
                float(row.get("altitude_demand", 0.0)),
            ]
        )
        adjustments.append(float(target @ coefficients))
    wet_fraction = float(row.get("wet_fraction", 0.0) or 0.0)
    wet_samples = int(state["wet_samples"])
    if wet_fraction > 0.20 and wet_samples >= 3:
        shrink = wet_samples / (wet_samples + 5.0)
        adjustments.append(shrink * float(state["wet_residual_sum"]) / wet_samples)
    return (float(np.mean(adjustments)) if adjustments else 0.0, samples)


def _update_context(
    state: dict[str, object] | None,
    row: pd.Series,
    residual: float,
) -> dict[str, object]:
    state = _new_context_state() if state is None else state
    features = np.asarray(
        [
            1.0,
            float(row.get("straight_demand", 0.5)),
            float(row.get("altitude_demand", 0.0)),
        ]
    )
    state["samples"] = int(state["samples"]) + 1
    state["xtx"] = np.asarray(state["xtx"]) + np.outer(features, features)
    state["xty"] = np.asarray(state["xty"]) + features * residual
    if float(row.get("wet_fraction", 0.0) or 0.0) > 0.20:
        state["wet_samples"] = int(state["wet_samples"]) + 1
        state["wet_residual_sum"] = float(state["wet_residual_sum"]) + residual
    return state


def _replay_history_v7_2(
    history: pd.DataFrame,
    config: RookieBackcastConfigV72,
    *,
    prefix: str,
    frozen_debut_ratings: dict[str, float] | None,
    frozen_debut_anchors: dict[str, float] | None = None,
) -> pd.DataFrame:
    data = _validate(history)
    schedule = calendar_k_schedule_v7_2(data, config)
    data = data.merge(schedule, on="season", how="left", validate="many_to_one")
    ratings: dict[str, float] = {}
    debut_anchors: dict[str, float] = {}
    last_season: dict[str, int] = {}
    context_history: dict[str, dict[str, object]] = {}
    outputs: list[pd.DataFrame] = []

    pre_qualifying_column = f"{prefix}_pre_qualifying_rating"
    qualifying_delta_column = f"{prefix}_qualifying_delta"
    pre_race_column = f"{prefix}_pre_race_rating"
    driver_expected_column = f"{prefix}_driver_expected_from_rating"
    expected_column = f"{prefix}_expected_performance"
    residual_column = f"{prefix}_performance_residual"
    race_delta_column = f"{prefix}_race_delta"
    post_column = f"{prefix}_post_rating"
    debut_anchor_column = f"{prefix}_debut_anchor"

    for _, race in data.groupby("event_index", sort=True):
        race = race.copy()
        current_driver_ids = race["driver_id"].astype(str).tolist()
        returning_ratings = [ratings[driver_id] for driver_id in current_driver_ids if driver_id in ratings]
        active_field_anchor = (
            float(np.median(returning_ratings))
            if returning_ratings
            else config.default_field_anchor
        )
        # Every member of the inaugural 1950 cohort shares the common field
        # anchor, including drivers whose first appearance came after round one.
        new_driver_anchor = (
            config.default_field_anchor
            if int(race["season"].iloc[0]) == 1950
            else active_field_anchor
        )
        for driver_id in current_driver_ids:
            if driver_id not in debut_anchors:
                if frozen_debut_anchors is not None:
                    debut_anchors[driver_id] = float(
                        frozen_debut_anchors.get(driver_id, new_driver_anchor)
                    )
                else:
                    debut_anchors[driver_id] = new_driver_anchor
            if driver_id not in ratings:
                ratings[driver_id] = (
                    float(frozen_debut_ratings[driver_id])
                    if frozen_debut_ratings is not None
                    else new_driver_anchor
                )

        pre_values: list[float] = []
        inactivity_gaps: list[int] = []
        return_multipliers: list[float] = []
        context_adjustments: list[float] = []
        context_samples: list[int] = []
        for _, row in race.iterrows():
            driver_id = str(row["driver_id"])
            pre_values.append(ratings[driver_id])
            gap = max(
                int(row["season"]) - last_season.get(driver_id, int(row["season"])) - 1,
                0,
            )
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

        race[pre_qualifying_column] = pre_values
        race[debut_anchor_column] = race["driver_id"].astype(str).map(debut_anchors)
        race["inactivity_seasons"] = inactivity_gaps
        race["return_uncertainty_multiplier"] = return_multipliers
        race["driver_context_adjustment"] = context_adjustments
        race["driver_context_prior_samples"] = context_samples
        race["pre_session_strength"] = (
            race[pre_qualifying_column]
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
        race[qualifying_delta_column] = qualifying_elo_deltas_v6(
            race, qualifying_config
        ).fillna(0.0)
        race[pre_race_column] = race[pre_qualifying_column] + race[qualifying_delta_column]
        race[driver_expected_column] = _pairwise_expected(
            race[pre_race_column].to_numpy(dtype=float), config.elo_scale
        )
        strengths = (
            race[pre_race_column]
            + config.car_rating_points_per_z
            * pd.to_numeric(race["car_suitability_score_v6"], errors="coerce").fillna(0.0)
        )
        race[expected_column] = np.clip(
            _pairwise_expected(strengths.to_numpy(dtype=float), config.elo_scale)
            + race["driver_context_adjustment"].to_numpy(dtype=float),
            0.0,
            1.0,
        )
        eligible = race["rating_eligible"] & race["observed_performance"].notna()
        race[residual_column] = race["observed_performance"] - race[expected_column]
        raw = (
            race.loc[eligible, "effective_race_k"].to_numpy(dtype=float)
            * race.loc[eligible, residual_column].to_numpy(dtype=float)
            * race.loc[eligible, "return_uncertainty_multiplier"].to_numpy(dtype=float)
        )
        race[race_delta_column] = 0.0
        race.loc[eligible, race_delta_column] = _bounded_zero_sum(
            raw, float(race["effective_max_race_delta"].iloc[0])
        )
        race[post_column] = race[pre_race_column] + race[race_delta_column]

        for _, row in race.iterrows():
            driver_id = str(row["driver_id"])
            ratings[driver_id] = float(row[post_column])
            last_season[driver_id] = int(row["season"])
            if bool(row["rating_eligible"]) and np.isfinite(row["observed_performance"]):
                context_history[driver_id] = _update_context(
                    context_history.get(driver_id), row, float(row[residual_column])
                )
        outputs.append(race)

    return pd.concat(outputs, ignore_index=True).sort_values(
        ["season", "round", "position"], kind="stable"
    )


def run_provisional_history_v7_2(
    history: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> pd.DataFrame:
    cfg = config or RookieBackcastConfigV72()
    result = _replay_history_v7_2(
        history,
        cfg,
        prefix="provisional",
        frozen_debut_ratings=None,
    )
    result["causal_rating"] = result["provisional_post_rating"]
    result["provisional_rating"] = result["provisional_post_rating"]
    result["rating_history_scope"] = (
        "causal_provisional_active_field_anchor_calendar_normalized_v7_2"
    )
    return result


def cross_fitted_car_expectations_v7_2(
    provisional_history: pd.DataFrame,
    windows: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> pd.DataFrame:
    """Cross-fit car expectation against the provisional same-scale ratings."""

    cfg = config or RookieBackcastConfigV72()
    data = _validate(provisional_history)
    data["car_signal_observed"] = (
        pd.to_numeric(data["observed_performance"], errors="coerce")
        - pd.to_numeric(
            data["provisional_driver_expected_from_rating"], errors="coerce"
        )
    )
    eligible_pool = data[data["rating_eligible"] & data["car_signal_observed"].notna()]
    output: list[dict[str, Any]] = []
    for rookie in windows.itertuples(index=False):
        window_indices = set(int(value) for value in rookie.rookie_window_indices)
        for index in window_indices:
            target = data.loc[index]
            same_season = eligible_pool[eligible_pool["season"].eq(int(target["season"]))]
            candidates = same_season[
                same_season["model_entity_id"].astype(str).eq(str(target["model_entity_id"]))
            ]
            identity_scope = "exact_model"
            if len(candidates) < 2:
                candidates = same_season[
                    same_season["constructor_id_f1db"]
                    .astype(str)
                    .eq(str(target["constructor_id_f1db"]))
                ]
                identity_scope = "constructor_fallback"
            leakage_mask = candidates.index.isin(window_indices) & candidates[
                "driver_id"
            ].astype(str).eq(str(rookie.driver_id))
            candidates = candidates[~leakage_mask].copy()
            if candidates.empty:
                signal = 0.0
                effective = 0.0
                teammate_rows = 0
            else:
                weights = _layout_similarity(target, candidates)
                same_race = candidates["event_index"].eq(int(target["event_index"])).to_numpy()
                weights *= np.where(same_race, 3.0, 1.0)
                teammate_rows = int(same_race.sum())
                effective = float(weights.sum())
                raw_signal = float(
                    np.average(
                        candidates["car_signal_observed"].to_numpy(dtype=float),
                        weights=weights,
                    )
                )
                signal = raw_signal * effective / (effective + cfg.car_signal_prior_races)
            output.append(
                {
                    "driver_id": str(rookie.driver_id),
                    "history_index": index,
                    "season": int(target["season"]),
                    "round": int(target["round"]),
                    "rookie_car_crossfit_signal": signal,
                    "rookie_car_crossfit_effective_rows": effective,
                    "rookie_car_crossfit_teammate_rows": teammate_rows,
                    "rookie_car_crossfit_scope": identity_scope,
                    "rookie_window_row_used_in_car_fit": False,
                    "car_signal_basis": "provisional_driver_expected_from_rating_v7_2",
                }
            )
    return pd.DataFrame(output)


def _candidate_expected_grid_v7_2(
    candidate_ratings: np.ndarray,
    target: pd.Series,
    race: pd.DataFrame,
    car_signal: float,
    scale: float,
) -> np.ndarray:
    opponents = race[race["driver_id"].astype(str).ne(str(target["driver_id"]))]
    if opponents.empty:
        driver_expected = np.ones(len(candidate_ratings), dtype=float)
    else:
        opponent_ratings = pd.to_numeric(
            opponents["provisional_pre_race_rating"], errors="coerce"
        ).to_numpy(dtype=float)
        differences = (
            candidate_ratings[:, np.newaxis] - opponent_ratings[np.newaxis, :]
        ) / scale
        driver_expected = np.mean(
            1.0 / (1.0 + np.exp(-np.clip(differences, -30.0, 30.0))), axis=1
        )
    return np.clip(driver_expected + car_signal, 0.0, 1.0)


def estimate_rookie_debut_ratings_v7_2(
    provisional_history: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or RookieBackcastConfigV72()
    data = _validate(provisional_history)
    windows = identify_rookie_windows_v7(data, cfg)
    first_rows = data.sort_values("event_index").drop_duplicates("driver_id", keep="first")
    anchors = first_rows.set_index(first_rows["driver_id"].astype(str))[
        "provisional_debut_anchor"
    ].to_dict()
    car_crossfit = cross_fitted_car_expectations_v7_2(data, windows, cfg)
    crossfit_lookup = car_crossfit.set_index("history_index")
    grids = np.arange(
        -cfg.offset_limit,
        cfg.offset_limit + cfg.offset_grid_step / 2.0,
        cfg.offset_grid_step,
    )
    races_by_event = {
        int(event_index): race for event_index, race in data.groupby("event_index", sort=False)
    }
    estimates: list[dict[str, Any]] = []
    for rookie in windows.itertuples(index=False):
        indices = [int(value) for value in rookie.rookie_window_indices]
        anchor = float(anchors.get(str(rookie.driver_id), cfg.default_field_anchor))
        base = {
            **rookie._asdict(),
            "debut_field_anchor": anchor,
            "debut_anchor_source": "provisional_active_field_median_same_scale_v7_2",
            "rookie_opponent_rating_source": "provisional_pre_race_rating_v7_2",
        }
        if not indices:
            estimates.append(
                {
                    **base,
                    "rookie_raw_offset": 0.0,
                    "rookie_offset_standard_error": float("inf"),
                    "rookie_first_five_rmse": float("nan"),
                }
            )
            continue
        observations = data.loc[indices]
        weights = np.asarray([_condition_weight(row) for _, row in observations.iterrows()])
        observed = observations["observed_performance"].to_numpy(dtype=float)
        candidate_ratings = anchor + grids
        predictions = []
        for history_index, target in observations.iterrows():
            car_signal = float(
                crossfit_lookup.loc[history_index, "rookie_car_crossfit_signal"]
            )
            predictions.append(
                _candidate_expected_grid_v7_2(
                    candidate_ratings,
                    target,
                    races_by_event[int(target["event_index"])],
                    car_signal,
                    cfg.elo_scale,
                )
            )
        prediction_grid = np.column_stack(predictions)
        losses = np.average(
            (observed[np.newaxis, :] - prediction_grid) ** 2,
            axis=1,
            weights=weights,
        )
        best_index = int(np.argmin(losses))
        raw_offset = float(grids[best_index])
        local_results: list[np.ndarray] = []
        for history_index, target in observations.iterrows():
            car_signal = float(
                crossfit_lookup.loc[history_index, "rookie_car_crossfit_signal"]
            )
            local_results.append(
                _candidate_expected_grid_v7_2(
                    np.asarray(
                        [
                            anchor + raw_offset,
                            anchor + raw_offset + 5.0,
                            anchor + raw_offset - 5.0,
                        ]
                    ),
                    target,
                    races_by_event[int(target["event_index"])],
                    car_signal,
                    cfg.elo_scale,
                )
            )
        local = np.asarray(local_results)
        residuals = observed - local[:, 0]
        rmse = float(np.sqrt(np.average(residuals**2, weights=weights)))
        derivative = (local[:, 1] - local[:, 2]) / 10.0
        residual_floor = max(rmse, 0.08 if len(indices) >= 3 else 0.15)
        information = float(np.sum(weights * derivative**2))
        standard_error = residual_floor / np.sqrt(max(information, 1e-12))
        estimates.append(
            {
                **base,
                "rookie_raw_offset": raw_offset,
                "rookie_offset_standard_error": standard_error,
                "rookie_first_five_rmse": rmse,
            }
        )

    result = pd.DataFrame(estimates)
    population_sample = result[
        result["rookie_eligible_races"].ge(3) & result["debut_season"].gt(1950)
    ]
    population_offset = (
        float(population_sample["rookie_raw_offset"].median())
        if len(population_sample)
        else 0.0
    )
    deviations = np.abs(population_sample["rookie_raw_offset"] - population_offset)
    robust_scale = float(1.4826 * np.median(deviations)) if len(deviations) else 0.0
    typical_measurement_variance = (
        float(np.nanmedian(population_sample["rookie_offset_standard_error"] ** 2))
        if len(population_sample)
        else 0.0
    )
    if not np.isfinite(typical_measurement_variance):
        typical_measurement_variance = 0.0
    population_variance = max(
        robust_scale**2 - typical_measurement_variance,
        cfg.minimum_population_scale**2,
    )
    standard_error_squared = result["rookie_offset_standard_error"] ** 2
    reliability = population_variance / (population_variance + standard_error_squared)
    eligible_races = result["rookie_eligible_races"].astype(float)
    reliability *= eligible_races / (eligible_races + cfg.rookie_prior_equivalent_races)
    reliability = reliability.mask(result["rookie_window_races"].eq(1), 0.0)
    reliability = reliability.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 1.0)
    result["rookie_population_offset"] = population_offset
    result["rookie_population_scale"] = float(np.sqrt(population_variance))
    result["rookie_reliability"] = reliability
    result["rookie_offset_shrunk"] = (
        reliability * result["rookie_raw_offset"]
        + (1.0 - reliability) * population_offset
    )
    inaugural = result["debut_season"].eq(1950)
    if inaugural.any():
        result.loc[inaugural, "rookie_offset_shrunk"] -= float(
            result.loc[inaugural, "rookie_offset_shrunk"].mean()
        )
    result["debut_rating_final"] = (
        result["debut_field_anchor"] + result["rookie_offset_shrunk"]
    )
    result["debut_rating_method"] = np.select(
        [
            result["debut_season"].eq(1950),
            result["rookie_window_races"].eq(1),
            result["rookie_eligible_races"].eq(0),
            result["rookie_eligible_races"].lt(cfg.rookie_eligible_races),
        ],
        [
            "inaugural_cohort_centered_backcast",
            "one_off_population_prior",
            "rookie_population_prior_only",
            "partial_window_empirical_bayes",
        ],
        default="first_five_crossfit_empirical_bayes",
    )
    return result, car_crossfit


def run_frozen_retrospective_history_v7_2(
    history: pd.DataFrame,
    debut_estimates: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> pd.DataFrame:
    cfg = config or RookieBackcastConfigV72()
    debut_ratings = debut_estimates.set_index("driver_id")["debut_rating_final"].to_dict()
    debut_anchors = debut_estimates.set_index("driver_id")["debut_field_anchor"].to_dict()
    result = _replay_history_v7_2(
        history,
        cfg,
        prefix="retrospective",
        frozen_debut_ratings={str(key): float(value) for key, value in debut_ratings.items()},
        frozen_debut_anchors={str(key): float(value) for key, value in debut_anchors.items()},
    )
    result["retrospective_rating"] = result["retrospective_post_rating"]
    result["debut_rating_frozen"] = result["driver_id"].astype(str).map(debut_ratings)
    result["rating_history_scope"] = (
        "retrospective_first_five_backcast_frozen_once_calendar_normalized_v7_2"
    )
    return result


def build_rookie_backcast_v7_2(
    history: pd.DataFrame,
    config: RookieBackcastConfigV72 | None = None,
) -> RookieBackcastResultV72:
    cfg = config or RookieBackcastConfigV72()
    data = _validate(history)
    provisional = run_provisional_history_v7_2(data, cfg)
    debut, car_crossfit = estimate_rookie_debut_ratings_v7_2(provisional, cfg)
    retrospective = run_frozen_retrospective_history_v7_2(data, debut, cfg)
    causal = provisional.copy()
    schedule = calendar_k_schedule_v7_2(data, cfg)
    target_race_budget = cfg.race_k_factor * float(
        schedule["calendar_reference_rounds"].iloc[0]
    )
    target_qualifying_budget = (
        cfg.qualifying_base_race_k_factor
        * cfg.qualifying_fraction
        * float(schedule["calendar_reference_rounds"].iloc[0])
    )
    audit = {
        "version": "7.2",
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
            (
                schedule["scheduled_qualifying_k_budget"]
                - target_qualifying_budget
            ).abs().max()
        ),
        "debut_anchor_source": "provisional_active_field_median_same_scale_v7_2",
        "rookie_opponent_rating_source": "provisional_pre_race_rating_v7_2",
        "car_crossfit_signal_basis": "provisional_driver_expected_from_rating_v7_2",
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
    return RookieBackcastResultV72(
        debut,
        provisional,
        causal,
        retrospective,
        car_crossfit,
        audit,
    )
