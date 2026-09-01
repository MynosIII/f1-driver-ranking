from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model_rules_v6 import SessionEloConfigV6, qualifying_elo_deltas_v6


@dataclass(frozen=True, slots=True)
class RookieBackcastConfigV7:
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
    car_rating_points_per_z: float = 80.0
    maximum_return_multiplier: float = 1.50
    return_multiplier_per_season: float = 0.08


@dataclass(slots=True)
class RookieBackcastResultV7:
    debut_estimates: pd.DataFrame
    causal_history: pd.DataFrame
    retrospective_history: pd.DataFrame
    audit: dict[str, Any]


def _validate(history: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "round",
        "driver_id",
        "driver",
        "constructor",
        "observed_performance",
        "rating_eligible",
        "pre_rating",
        "post_rating",
        "driver_expected_from_rating",
        "car_suitability_score_v6",
        "model_entity_id",
        "constructor_id_f1db",
        "layout_key",
        "straight_demand",
        "altitude_demand",
        "wet_fraction",
        "wind_speed",
        "qualifying_position",
        "qualifying_eligible",
    }
    if missing := required - set(history.columns):
        raise ValueError(f"v6 history is missing columns: {sorted(missing)}")
    data = history.copy()
    data["season"] = pd.to_numeric(data["season"], errors="raise").astype(int)
    data["round"] = pd.to_numeric(data["round"], errors="raise").astype(int)
    data["rating_eligible"] = data["rating_eligible"].astype("boolean").fillna(False).astype(bool)
    data = data.sort_values(["season", "round", "position"], kind="stable").reset_index(
        drop=True
    )
    if data.duplicated(["season", "round", "driver_id"]).any():
        raise ValueError("History must be unique at driver-race grain")
    data["event_index"] = data.groupby(["season", "round"], sort=True).ngroup()
    return data


def _bounded_zero_sum(values: np.ndarray, limit: float) -> np.ndarray:
    if len(values) == 0:
        return np.asarray(values, dtype=float)
    raw = np.asarray(values, dtype=float)
    lower = float(raw.min() - limit)
    upper = float(raw.max() + limit)
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        total = float(np.clip(raw - midpoint, -limit, limit).sum())
        if total > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return np.clip(raw - (lower + upper) / 2.0, -limit, limit)


def _pairwise_expected(strengths: np.ndarray, scale: float) -> np.ndarray:
    if len(strengths) <= 1:
        return np.ones(len(strengths), dtype=float)
    differences = (strengths[:, None] - strengths[None, :]) / scale
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(differences, -30.0, 30.0)))
    np.fill_diagonal(probabilities, np.nan)
    return np.nanmean(probabilities, axis=1)


def identify_rookie_windows_v7(
    history: pd.DataFrame,
    config: RookieBackcastConfigV7 | None = None,
) -> pd.DataFrame:
    cfg = config or RookieBackcastConfigV7()
    data = _validate(history)
    rows: list[dict[str, Any]] = []
    for driver_id, career in data.groupby("driver_id", sort=False):
        career = career.sort_values("event_index", kind="stable")
        debut = career.iloc[0]
        within_seasons = career["season"].le(
            int(debut["season"]) + cfg.maximum_window_seasons - 1
        )
        candidates = career.loc[within_seasons].head(cfg.maximum_window_entries)
        eligible = candidates[
            candidates["rating_eligible"] & candidates["observed_performance"].notna()
        ].head(cfg.rookie_eligible_races)
        eligible_indices = list(eligible.index.astype(int))
        rows.append(
            {
                "driver_id": str(driver_id),
                "driver": debut["driver"],
                "debut_season": int(debut["season"]),
                "debut_round": int(debut["round"]),
                "debut_event_index": int(debut["event_index"]),
                "rookie_window_races": int(len(candidates)),
                "rookie_eligible_races": int(len(eligible)),
                "rookie_window_indices": eligible_indices,
                "rookie_window_last_season": int(eligible["season"].max())
                if len(eligible)
                else int(debut["season"]),
                "rookie_window_complete": len(eligible) >= cfg.rookie_eligible_races,
            }
        )
    return pd.DataFrame(rows)


def _field_anchors(data: pd.DataFrame, windows: pd.DataFrame, default: float) -> dict[str, float]:
    prior_starts: dict[str, int] = {}
    anchors: dict[str, float] = {}
    debut_lookup = windows.set_index("driver_id")["debut_event_index"].to_dict()
    for event_index, race in data.groupby("event_index", sort=True):
        active_experienced = race[
            race["driver_id"].astype(str).map(prior_starts).fillna(0).gt(0)
        ]
        field_anchor = (
            float(active_experienced["pre_rating"].median())
            if len(active_experienced)
            else default
        )
        for driver_id in race["driver_id"].astype(str):
            if debut_lookup.get(driver_id) == event_index:
                anchors[driver_id] = field_anchor
            prior_starts[driver_id] = prior_starts.get(driver_id, 0) + 1
    for rookie in windows[windows["debut_season"].eq(1950)].itertuples(index=False):
        anchors[str(rookie.driver_id)] = default
    return anchors


def _layout_similarity(target: pd.Series, candidates: pd.DataFrame) -> np.ndarray:
    straight = pd.to_numeric(candidates["straight_demand"], errors="coerce").fillna(0.5)
    altitude = pd.to_numeric(candidates["altitude_demand"], errors="coerce").fillna(0.0)
    distance = np.sqrt(
        (straight.to_numpy(dtype=float) - float(target.get("straight_demand", 0.5))) ** 2
        + (altitude.to_numpy(dtype=float) - float(target.get("altitude_demand", 0.0))) ** 2
    )
    return np.exp(-2.0 * distance)


def cross_fitted_car_expectations_v7(
    history: pd.DataFrame,
    windows: pd.DataFrame,
    config: RookieBackcastConfigV7 | None = None,
) -> pd.DataFrame:
    """Estimate rookie-window car signals without using those rookie observations."""

    cfg = config or RookieBackcastConfigV7()
    data = _validate(history)
    data["car_signal_observed"] = (
        pd.to_numeric(data["observed_performance"], errors="coerce")
        - pd.to_numeric(data["driver_expected_from_rating"], errors="coerce")
    )
    eligible_pool = data[
        data["rating_eligible"] & data["car_signal_observed"].notna()
    ]
    output: list[dict[str, Any]] = []
    for rookie in windows.itertuples(index=False):
        window_indices = set(int(value) for value in rookie.rookie_window_indices)
        for index in window_indices:
            target = data.loc[index]
            same_season = eligible_pool[eligible_pool["season"].eq(int(target["season"]))]
            exact = same_season[
                same_season["model_entity_id"].astype(str).eq(str(target["model_entity_id"]))
            ]
            candidates = exact
            identity_scope = "exact_model"
            if len(candidates) < 2:
                candidates = same_season[
                    same_season["constructor_id_f1db"].astype(str).eq(
                        str(target["constructor_id_f1db"])
                    )
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
                    np.average(candidates["car_signal_observed"].to_numpy(dtype=float), weights=weights)
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
                }
            )
    return pd.DataFrame(output)


def _condition_weight(row: pd.Series) -> float:
    wet = float(np.clip(pd.to_numeric(row.get("wet_fraction", 0.0), errors="coerce"), 0.0, 1.0))
    wind = float(max(pd.to_numeric(row.get("wind_speed", 0.0), errors="coerce"), 0.0))
    return float(1.0 / (1.0 + 0.75 * wet + 0.015 * wind))


def _candidate_expected_grid(
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
            opponents["pre_rating"], errors="coerce"
        ).to_numpy(dtype=float)
        differences = (
            candidate_ratings[:, np.newaxis] - opponent_ratings[np.newaxis, :]
        ) / scale
        driver_expected = np.mean(
            1.0 / (1.0 + np.exp(-np.clip(differences, -30, 30))), axis=1
        )
    return np.clip(driver_expected + car_signal, 0.0, 1.0)


def estimate_rookie_debut_ratings_v7(
    history: pd.DataFrame,
    config: RookieBackcastConfigV7 | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or RookieBackcastConfigV7()
    data = _validate(history)
    windows = identify_rookie_windows_v7(data, cfg)
    anchors = _field_anchors(data, windows, cfg.default_field_anchor)
    car_crossfit = cross_fitted_car_expectations_v7(data, windows, cfg)
    crossfit_lookup = car_crossfit.set_index("history_index")
    grids = np.arange(
        -cfg.offset_limit,
        cfg.offset_limit + cfg.offset_grid_step / 2.0,
        cfg.offset_grid_step,
    )
    races_by_event = {
        int(event_index): race
        for event_index, race in data.groupby("event_index", sort=False)
    }
    estimates: list[dict[str, Any]] = []
    for rookie in windows.itertuples(index=False):
        indices = [int(value) for value in rookie.rookie_window_indices]
        anchor = float(anchors.get(str(rookie.driver_id), cfg.default_field_anchor))
        if not indices:
            estimates.append(
                {
                    **rookie._asdict(),
                    "debut_field_anchor": anchor,
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
            race = races_by_event[int(target["event_index"])]
            car_signal = float(crossfit_lookup.loc[history_index, "rookie_car_crossfit_signal"])
            predictions.append(
                _candidate_expected_grid(
                    candidate_ratings, target, race, car_signal, cfg.elo_scale
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
        best_predictions = []
        plus_predictions = []
        minus_predictions = []
        for history_index, target in observations.iterrows():
            race = races_by_event[int(target["event_index"])]
            car_signal = float(crossfit_lookup.loc[history_index, "rookie_car_crossfit_signal"])
            local_predictions = _candidate_expected_grid(
                np.asarray(
                    [
                        anchor + raw_offset,
                        anchor + raw_offset + 5.0,
                        anchor + raw_offset - 5.0,
                    ]
                ),
                target,
                race,
                car_signal,
                cfg.elo_scale,
            )
            best_predictions.append(local_predictions[0])
            plus_predictions.append(local_predictions[1])
            minus_predictions.append(local_predictions[2])
        residuals = observed - np.asarray(best_predictions)
        rmse = float(np.sqrt(np.average(residuals**2, weights=weights)))
        derivative = (np.asarray(plus_predictions) - np.asarray(minus_predictions)) / 10.0
        residual_floor = max(rmse, 0.08 if len(indices) >= 3 else 0.15)
        information = float(np.sum(weights * derivative**2))
        standard_error = residual_floor / np.sqrt(max(information, 1e-12))
        estimates.append(
            {
                **rookie._asdict(),
                "debut_field_anchor": anchor,
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
    inaugural_center = float(result.loc[inaugural, "rookie_offset_shrunk"].mean())
    result.loc[inaugural, "rookie_offset_shrunk"] -= inaugural_center
    result["debut_rating_final"] = result["debut_field_anchor"] + result[
        "rookie_offset_shrunk"
    ]
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


def _driver_context_adjustment(
    state: dict[str, object] | None, row: pd.Series
) -> tuple[float, int]:
    if state is None:
        return 0.0, 0
    samples = int(state["samples"])
    adjustments = []
    if samples >= 5:
        xtx = np.asarray(state["xtx"], dtype=float)
        xty = np.asarray(state["xty"], dtype=float)
        # Closed-form Ridge(alpha=8) with an unpenalized intercept. Retaining
        # only X'X and X'y preserves causal ordering without a career refit.
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
        wet_mean = float(state["wet_residual_sum"]) / wet_samples
        adjustments.append(shrink * wet_mean)
    return (float(np.mean(adjustments)) if adjustments else 0.0, samples)


def _update_driver_context(
    state: dict[str, object] | None,
    row: pd.Series,
) -> dict[str, object]:
    if state is None:
        state = {
            "samples": 0,
            "xtx": np.zeros((3, 3), dtype=float),
            "xty": np.zeros(3, dtype=float),
            "wet_samples": 0,
            "wet_residual_sum": 0.0,
        }
    features = np.asarray(
        [
            1.0,
            float(row.get("straight_demand", 0.5)),
            float(row.get("altitude_demand", 0.0)),
        ]
    )
    residual = float(row["retrospective_performance_residual"])
    state["samples"] = int(state["samples"]) + 1
    state["xtx"] = np.asarray(state["xtx"]) + np.outer(features, features)
    state["xty"] = np.asarray(state["xty"]) + features * residual
    if float(row.get("wet_fraction", 0.0) or 0.0) > 0.20:
        state["wet_samples"] = int(state["wet_samples"]) + 1
        state["wet_residual_sum"] = float(state["wet_residual_sum"]) + residual
    return state


def run_frozen_retrospective_history_v7(
    history: pd.DataFrame,
    debut_estimates: pd.DataFrame,
    config: RookieBackcastConfigV7 | None = None,
) -> pd.DataFrame:
    cfg = config or RookieBackcastConfigV7()
    data = _validate(history)
    debut_ratings = debut_estimates.set_index("driver_id")["debut_rating_final"].to_dict()
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    context_history: dict[str, dict[str, object]] = {}
    outputs: list[pd.DataFrame] = []
    qualifying_config = SessionEloConfigV6(
        race_k_factor=18.0,
        qualifying_fraction=1.0 / 3.0,
        elo_scale=cfg.elo_scale,
    )
    for _, race in data.groupby("event_index", sort=True):
        race = race.copy()
        pre_values = []
        gaps = []
        multipliers = []
        context_adjustments = []
        context_samples = []
        for _, row in race.iterrows():
            driver_id = str(row["driver_id"])
            pre_values.append(float(ratings.get(driver_id, debut_ratings[driver_id])))
            gap = max(int(row["season"]) - last_season.get(driver_id, int(row["season"])) - 1, 0)
            gaps.append(gap)
            multipliers.append(
                min(
                    cfg.maximum_return_multiplier,
                    1.0 + cfg.return_multiplier_per_season * gap,
                )
            )
            adjustment, samples = _driver_context_adjustment(
                context_history.get(driver_id), row
            )
            context_adjustments.append(adjustment)
            context_samples.append(samples)
        race["retrospective_pre_qualifying_rating"] = pre_values
        race["inactivity_seasons"] = gaps
        race["return_uncertainty_multiplier"] = multipliers
        race["driver_context_adjustment"] = context_adjustments
        race["driver_context_prior_samples"] = context_samples
        race["pre_session_strength"] = (
            race["retrospective_pre_qualifying_rating"]
            + cfg.car_rating_points_per_z
            * pd.to_numeric(race["car_suitability_score_v6"], errors="coerce").fillna(0.0)
        )
        race["retrospective_qualifying_delta"] = qualifying_elo_deltas_v6(
            race, qualifying_config
        ).fillna(0.0)
        race["retrospective_pre_race_rating"] = (
            race["retrospective_pre_qualifying_rating"]
            + race["retrospective_qualifying_delta"]
        )
        strengths = (
            race["retrospective_pre_race_rating"]
            + cfg.car_rating_points_per_z
            * pd.to_numeric(race["car_suitability_score_v6"], errors="coerce").fillna(0.0)
        )
        race["retrospective_expected_performance"] = np.clip(
            _pairwise_expected(strengths.to_numpy(dtype=float), cfg.elo_scale)
            + race["driver_context_adjustment"].to_numpy(dtype=float),
            0.0,
            1.0,
        )
        eligible = race["rating_eligible"] & race["observed_performance"].notna()
        race["retrospective_performance_residual"] = (
            race["observed_performance"] - race["retrospective_expected_performance"]
        )
        raw = (
            cfg.race_k_factor
            * race.loc[eligible, "retrospective_performance_residual"].to_numpy(dtype=float)
            * race.loc[eligible, "return_uncertainty_multiplier"].to_numpy(dtype=float)
        )
        race["retrospective_race_delta"] = 0.0
        race.loc[eligible, "retrospective_race_delta"] = _bounded_zero_sum(
            raw, cfg.maximum_race_delta
        )
        race["retrospective_post_rating"] = (
            race["retrospective_pre_race_rating"] + race["retrospective_race_delta"]
        )
        race["causal_rating"] = race["post_rating"]
        race["retrospective_rating"] = race["retrospective_post_rating"]
        race["debut_rating_frozen"] = race["driver_id"].astype(str).map(debut_ratings)
        race["rating_history_scope"] = "retrospective_first_five_backcast_frozen_once"
        for _, row in race.iterrows():
            driver_id = str(row["driver_id"])
            ratings[driver_id] = float(row["retrospective_post_rating"])
            last_season[driver_id] = int(row["season"])
            if bool(row["rating_eligible"]) and np.isfinite(row["observed_performance"]):
                context_history[driver_id] = _update_driver_context(
                    context_history.get(driver_id), row
                )
        outputs.append(race)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["season", "round", "position"], kind="stable"
    )


def build_rookie_backcast_v7(
    history: pd.DataFrame,
    config: RookieBackcastConfigV7 | None = None,
) -> RookieBackcastResultV7:
    cfg = config or RookieBackcastConfigV7()
    data = _validate(history)
    debut, car_crossfit = estimate_rookie_debut_ratings_v7(data, cfg)
    retrospective = run_frozen_retrospective_history_v7(data, debut, cfg)
    causal = data.copy()
    causal["causal_rating"] = causal["post_rating"]
    causal["rating_history_scope"] = "causal_prior_information_v3"
    all_ineligible_unselected = True
    for rookie in debut.itertuples(index=False):
        career = data[data["driver_id"].astype(str).eq(str(rookie.driver_id))]
        selected = set(int(value) for value in rookie.rookie_window_indices)
        last_window_season = int(rookie.debut_season) + cfg.maximum_window_seasons - 1
        within_window = career[career["season"].le(last_window_season)].head(
            cfg.maximum_window_entries
        )
        excluded = within_window[~within_window.index.isin(selected)]
        if not excluded.empty and excluded["rating_eligible"].any() and len(selected) < 5:
            all_ineligible_unselected = False
            break
    audit = {
        "drivers": int(debut["driver_id"].nunique()),
        "complete_first_five_windows": int(debut["rookie_window_complete"].sum()),
        "partial_windows": int(
            debut["rookie_eligible_races"].between(1, cfg.rookie_eligible_races - 1).sum()
        ),
        "population_prior_only": int(debut["rookie_eligible_races"].eq(0).sum()),
        "one_off_population_prior": int(
            debut["debut_rating_method"].eq("one_off_population_prior").sum()
        ),
        "inaugural_1950_drivers": int(debut["debut_season"].eq(1950).sum()),
        "inaugural_offset_mean": float(
            debut.loc[debut["debut_season"].eq(1950), "rookie_offset_shrunk"].mean()
        ),
        "rookie_population_offset": float(debut["rookie_population_offset"].iloc[0]),
        "rookie_population_scale": float(debut["rookie_population_scale"].iloc[0]),
        "rookie_prior_equivalent_races": cfg.rookie_prior_equivalent_races,
        "mean_rookie_reliability": float(debut["rookie_reliability"].mean()),
        "minimum_debut_rating": float(debut["debut_rating_final"].min()),
        "maximum_debut_rating": float(debut["debut_rating_final"].max()),
        "rookie_window_rows_used_in_car_fit": int(
            car_crossfit["rookie_window_row_used_in_car_fit"].sum()
        ),
        "unselected_early_rows_are_ineligible_or_window_capped": all_ineligible_unselected,
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
    }
    return RookieBackcastResultV7(debut, causal, retrospective, audit)
