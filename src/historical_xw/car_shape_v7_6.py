from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .car_geometry_v6 import CarGeometryResultV6, _constructor_scores, _softmax
from .circuit_shape_v9 import SHAPE_MODEL_FEATURES_V9
from .model_rules_v6 import CarryoverConfigV6, calculate_carryover_weights_v6


BASELINE_FEATURES_V76: tuple[str, ...] = ("straight_demand", "altitude_demand")
COMPOSITE_FEATURES_V76: tuple[str, ...] = (
    "geometry_only_straight_exposure_v9",
    "geometry_only_slow_rotation_v9",
    "geometry_only_flowing_rotation_v9",
    "geometry_only_direction_change_v9",
    "geometry_only_complexity_v9",
    "altitude_demand",
)
DISTRIBUTIONAL_FEATURES_V76: tuple[str, ...] = (
    *SHAPE_MODEL_FEATURES_V9,
    "altitude_demand",
)
AUGMENTED_FEATURES_V76: tuple[str, ...] = (
    *BASELINE_FEATURES_V76,
    *SHAPE_MODEL_FEATURES_V9,
)
FEATURE_SETS_V76: dict[str, tuple[str, ...]] = {
    "baseline_v6": BASELINE_FEATURES_V76,
    "shape_composites_v9": COMPOSITE_FEATURES_V76,
    "shape_distributions_v9": DISTRIBUTIONAL_FEATURES_V76,
    "baseline_plus_shape_distributions_v9": AUGMENTED_FEATURES_V76,
}


@dataclass(frozen=True, slots=True)
class CarShapeConfigV76:
    minimum_races: int = 4
    alpha_candidates: tuple[float, ...] = (
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    )
    car_win_temperature: float = 0.35
    carryover: CarryoverConfigV6 = CarryoverConfigV6()


@dataclass(slots=True)
class CarShapeValidationResultV76:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    alpha_selection: pd.DataFrame
    paired_season_deltas: pd.DataFrame
    audit: dict[str, Any]


def read_car_shape_validation_v7_6(path: Path) -> CarShapeValidationResultV76:
    return CarShapeValidationResultV76(
        pd.read_parquet(path / "nested_loo_predictions_v7_6.parquet"),
        pd.read_csv(path / "model_comparison_v7_6.csv"),
        pd.read_csv(path / "nested_alpha_selection_v7_6.csv"),
        pd.read_csv(path / "paired_season_deltas_v7_6.csv"),
        json.loads((path / "validation_audit_v7_6.json").read_text(encoding="utf-8")),
    )


def _prepare_profiles_v7_6(profiles: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "round", "circuit_id", "layout_key"}
    required.update(column for values in FEATURE_SETS_V76.values() for column in values)
    if missing := required - set(profiles.columns):
        raise ValueError(f"Shape profiles are missing v7.6 columns: {sorted(missing)}")
    result = profiles.copy()
    if result.duplicated(["season", "round", "circuit_id"]).any():
        raise ValueError("Shape profiles must be unique at event-circuit grain")
    return result


def car_race_targets_v7_6(history: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season",
        "round",
        "circuit_id",
        "model_entity_id",
        "constructor_id_f1db",
        "driver_id",
        "observed_performance",
    }
    if missing := required - set(history.columns):
        raise ValueError(f"History is missing v7.6 car-target columns: {sorted(missing)}")
    data = history.copy()
    if "car_fit_eligible" in data:
        eligible = data["car_fit_eligible"].fillna(False).astype(bool)
    else:
        eligible = data["rating_eligible"].fillna(False).astype(bool)
    data = data.loc[eligible & data["observed_performance"].notna()].copy()
    data["field_performance"] = data.groupby(["season", "round"])[
        "observed_performance"
    ].transform("mean")
    data["relative_performance"] = data["observed_performance"] - data["field_performance"]
    targets = data.groupby(
        [
            "season",
            "round",
            "circuit_id",
            "model_entity_id",
            "constructor_id_f1db",
        ],
        as_index=False,
    ).agg(
        car_relative_performance=("relative_performance", "mean"),
        contributing_drivers=("driver_id", "nunique"),
    )
    profile_data = _prepare_profiles_v7_6(profiles)
    profile_columns = [
        "season",
        "round",
        "circuit_id",
        "layout_key",
        "shape_topology_quality",
        "trace_method_v9",
        *sorted({column for values in FEATURE_SETS_V76.values() for column in values}),
    ]
    return targets.merge(
        profile_data[profile_columns],
        on=["season", "round", "circuit_id"],
        how="inner",
        validate="many_to_one",
    )


def _loo_predictions_all_alphas(
    targets: pd.DataFrame,
    features: tuple[str, ...],
    alphas: tuple[float, ...],
    minimum_races: int,
) -> pd.DataFrame:
    predictions: list[dict[str, Any]] = []
    for (season, entity), sample in targets.groupby(
        ["season", "model_entity_id"], sort=False
    ):
        sample = sample.sort_values("round", kind="stable")
        if len(sample) < minimum_races or sample["layout_key"].nunique() < 3:
            continue
        season_profiles = targets[targets["season"].eq(int(season))].drop_duplicates(
            "layout_key"
        )
        scaler = StandardScaler().fit(season_profiles[list(features)].to_numpy(dtype=float))
        sample_x = scaler.transform(sample[list(features)].to_numpy(dtype=float))
        sample_y = sample["car_relative_performance"].to_numpy(dtype=float)
        for target_index, target in sample.iterrows():
            target_position = int(sample.index.get_loc(target_index))
            train_mask = np.arange(len(sample)) != target_position
            train = sample.iloc[train_mask]
            if len(train) < minimum_races - 1 or train["layout_key"].nunique() < 2:
                continue
            train_x = sample_x[train_mask]
            train_y = sample_y[train_mask]
            x_mean = train_x.mean(axis=0)
            y_mean = float(train_y.mean())
            centered_x = train_x - x_mean
            centered_y = train_y - y_mean
            gram = centered_x.T @ centered_x
            cross = centered_x.T @ centered_y
            target_x = sample_x[target_position] - x_mean
            for alpha in alphas:
                coefficients = np.linalg.solve(
                    gram + float(alpha) * np.eye(len(features)), cross
                )
                predicted = float(y_mean + target_x @ coefficients)
                observed = float(target["car_relative_performance"])
                predictions.append(
                    {
                        "history_index": int(target_index),
                        "season": int(season),
                        "round": int(target["round"]),
                        "model_entity_id": str(entity),
                        "layout_key": str(target["layout_key"]),
                        "observed": observed,
                        "prediction": predicted,
                        "absolute_error": abs(observed - predicted),
                        "squared_error": (observed - predicted) ** 2,
                        "shape_topology_quality": str(target["shape_topology_quality"]),
                        "trace_method_v9": str(target["trace_method_v9"]),
                        "training_races": int(len(train)),
                        "alpha": float(alpha),
                    }
                )
    return pd.DataFrame(predictions)


def validate_car_shape_v7_6(
    history: pd.DataFrame,
    profiles: pd.DataFrame,
    config: CarShapeConfigV76 | None = None,
) -> CarShapeValidationResultV76:
    """Nested comparison: leave race out for prediction and season out for alpha."""

    cfg = config or CarShapeConfigV76()
    targets = car_race_targets_v7_6(history, profiles)
    candidate_frames: list[pd.DataFrame] = []
    for model_name, features in FEATURE_SETS_V76.items():
        frame = _loo_predictions_all_alphas(
            targets, features, cfg.alpha_candidates, cfg.minimum_races
        )
        frame["model"] = model_name
        candidate_frames.append(frame)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    selection_rows: list[dict[str, Any]] = []
    selected_predictions: list[pd.DataFrame] = []
    for model_name in FEATURE_SETS_V76:
        model_candidates = candidates[candidates["model"].eq(model_name)]
        for target_season in sorted(model_candidates["season"].unique()):
            training = model_candidates[model_candidates["season"].ne(int(target_season))]
            scores = training.groupby("alpha", as_index=False).agg(
                selection_mae=("absolute_error", "mean"),
                selection_rows=("absolute_error", "size"),
            )
            best = scores.sort_values(["selection_mae", "alpha"], kind="stable").iloc[0]
            chosen_alpha = float(best["alpha"])
            selection_rows.append(
                {
                    "model": model_name,
                    "target_season": int(target_season),
                    "selected_alpha": chosen_alpha,
                    "selection_mae": float(best["selection_mae"]),
                    "selection_rows": int(best["selection_rows"]),
                    "target_season_excluded": True,
                }
            )
            selected = model_candidates[
                model_candidates["season"].eq(int(target_season))
                & model_candidates["alpha"].eq(chosen_alpha)
            ].copy()
            selected["selected_outside_target_season"] = True
            selected_predictions.append(selected)
    predictions = pd.concat(selected_predictions, ignore_index=True)
    baseline_predictions = predictions[predictions["model"].eq("baseline_v6")]
    augmented_predictions = predictions[
        predictions["model"].eq("baseline_plus_shape_distributions_v9")
    ]
    hybrid = baseline_predictions.merge(
        augmented_predictions[
            ["history_index", "prediction", "absolute_error", "squared_error", "alpha"]
        ],
        on="history_index",
        how="inner",
        suffixes=("_baseline", "_shape"),
        validate="one_to_one",
    )
    use_shape = hybrid["shape_topology_quality"].eq("high")
    hybrid["prediction"] = hybrid["prediction_baseline"].where(
        ~use_shape, hybrid["prediction_shape"]
    )
    hybrid["absolute_error"] = hybrid["absolute_error_baseline"].where(
        ~use_shape, hybrid["absolute_error_shape"]
    )
    hybrid["squared_error"] = hybrid["squared_error_baseline"].where(
        ~use_shape, hybrid["squared_error_shape"]
    )
    hybrid["alpha"] = hybrid["alpha_baseline"].where(~use_shape, hybrid["alpha_shape"])
    hybrid["model"] = "topology_gated_baseline_plus_shape_v7_6"
    hybrid["geometry_branch"] = np.where(use_shape, "shape_augmented", "baseline")
    hybrid["selected_outside_target_season"] = True
    hybrid = hybrid[predictions.columns.tolist() + ["geometry_branch"]]
    predictions["geometry_branch"] = predictions["model"]
    predictions = pd.concat([predictions, hybrid], ignore_index=True)
    summary = predictions.groupby("model", as_index=False).agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
        mean_training_races=("training_races", "mean"),
    )
    correlations = pd.DataFrame(
        [
            {
                "model": model_name,
                "correlation": float(frame[["observed", "prediction"]].corr().iloc[0, 1]),
            }
            for model_name, frame in predictions.groupby("model", sort=False)
        ]
    )
    summary = summary.merge(correlations, on="model", validate="one_to_one")
    season_errors = predictions.groupby(["model", "season"], as_index=False).agg(
        season_mae=("absolute_error", "mean"),
        season_rows=("absolute_error", "size"),
    )
    baseline = season_errors[season_errors["model"].eq("baseline_v6")][
        ["season", "season_mae"]
    ].rename(columns={"season_mae": "baseline_season_mae"})
    paired = season_errors.merge(baseline, on="season", how="inner", validate="many_to_one")
    paired["mae_improvement_vs_baseline"] = (
        paired["baseline_season_mae"] - paired["season_mae"]
    )
    summary = summary.merge(
        paired.groupby("model", as_index=False).agg(
            seasons_better_than_baseline=(
                "mae_improvement_vs_baseline",
                lambda values: int(values.gt(0.0).sum()),
            ),
            median_season_mae_improvement=("mae_improvement_vs_baseline", "median"),
        ),
        on="model",
        validate="one_to_one",
    )
    baseline_mae = float(summary.set_index("model").loc["baseline_v6", "mae"])
    summary["mae_improvement_vs_baseline"] = baseline_mae - summary["mae"]
    summary["relative_mae_improvement_vs_baseline"] = (
        summary["mae_improvement_vs_baseline"] / baseline_mae
    )
    winner = summary.sort_values(["mae", "model"], kind="stable").iloc[0]
    winning_season_deltas = paired[
        paired["model"].eq(str(winner["model"]))
    ]["mae_improvement_vs_baseline"].to_numpy(dtype=float)
    random = np.random.default_rng(760)
    bootstrap = np.mean(
        random.choice(
            winning_season_deltas,
            size=(50000, len(winning_season_deltas)),
            replace=True,
        ),
        axis=1,
    )
    audit = {
        "version": "7.6",
        "validation_design": "leave_race_out_prediction_nested_leave_season_out_alpha",
        "target_rows": int(len(targets)),
        "prediction_rows": int(predictions["model"].eq("baseline_v6").sum()),
        "seasons": [int(predictions["season"].min()), int(predictions["season"].max())],
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS_V76.items()},
        "winning_feature_set": str(winner["model"]),
        "winning_mae": float(winner["mae"]),
        "baseline_mae": baseline_mae,
        "winning_relative_mae_improvement": float(
            winner["relative_mae_improvement_vs_baseline"]
        ),
        "winning_season_block_mean_improvement_ci_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "topology_gate": "use_augmented_shape_only_when_shape_topology_quality_is_high",
        "topology_gate_was_predefined_geometry_quality_category": True,
        "target_season_used_for_alpha_selection": False,
    }
    return CarShapeValidationResultV76(
        predictions,
        summary,
        pd.DataFrame(selection_rows),
        paired,
        audit,
    )


def fit_car_shape_v7_6(
    history: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    feature_set: str,
    alpha: float | dict[int, float],
    config: CarShapeConfigV76 | None = None,
) -> CarGeometryResultV6:
    """Fit the selected full-season static chassis curve for retrospective v7.6."""

    cfg = config or CarShapeConfigV76()
    if feature_set not in FEATURE_SETS_V76:
        raise ValueError(f"Unknown v7.6 feature set: {feature_set}")
    features = FEATURE_SETS_V76[feature_set]
    data = history.copy()
    data["car_fit_eligible"] = data["rating_eligible"].fillna(False).astype(bool)
    profile_data = _prepare_profiles_v7_6(profiles)
    constructor_scores = _constructor_scores(data)
    carryover = calculate_carryover_weights_v6(
        constructor_scores[["season", "constructor_id", "score"]], cfg.carryover
    )
    prior_lookup = constructor_scores.set_index(["season", "constructor_id"])["score_z"]
    race_outputs: list[pd.DataFrame] = []
    car_outputs: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for season, season_history in data.groupby("season", sort=True):
        season_alpha = float(alpha.get(int(season), 1000.0)) if isinstance(alpha, dict) else float(alpha)
        season_profiles = profile_data[profile_data["season"].eq(int(season))].drop_duplicates(
            ["season", "round", "circuit_id"]
        )
        if season_profiles.empty:
            continue
        observations = season_history.drop(
            columns=["layout_key", *features], errors="ignore"
        ).merge(
            season_profiles[["season", "round", "circuit_id", "layout_key", *features]],
            on=["season", "round", "circuit_id"],
            how="inner",
            validate="many_to_one",
        )
        eligible = observations["car_fit_eligible"] & observations[
            "observed_performance"
        ].notna()
        observations["field_performance"] = observations.groupby(["season", "round"])[
            "observed_performance"
        ].transform(lambda values: values[eligible.loc[values.index]].mean())
        observations["relative_performance"] = (
            observations["observed_performance"] - observations["field_performance"]
        )
        observations.loc[~eligible, "relative_performance"] = np.nan
        entity_constructor = (
            observations.sort_values("round")
            .drop_duplicates("model_entity_id")
            .set_index("model_entity_id")["constructor_id_f1db"]
            .astype(str)
        )
        unique_layouts = season_profiles.drop_duplicates("layout_key")
        scaler = StandardScaler().fit(unique_layouts[list(features)].to_numpy(dtype=float))
        grids: list[pd.DataFrame] = []
        for entity in sorted(observations["model_entity_id"].dropna().astype(str).unique()):
            sample = observations[
                observations["model_entity_id"].astype(str).eq(entity)
            ].dropna(subset=["relative_performance"])
            intercept = float(sample["relative_performance"].mean()) if len(sample) else 0.0
            slopes = np.zeros(len(features), dtype=float)
            fitted = len(sample) >= cfg.minimum_races and sample["layout_key"].nunique() >= 3
            if fitted:
                model = Ridge(alpha=season_alpha).fit(
                    scaler.transform(sample[list(features)].to_numpy(dtype=float)),
                    sample["relative_performance"].to_numpy(dtype=float),
                )
                intercept = float(model.intercept_)
                slopes = np.asarray(model.coef_, dtype=float)
            grid = unique_layouts[["season", "layout_key", "circuit_id", *features]].copy()
            grid["model_entity_id"] = entity
            grid["constructor_id_f1db"] = entity_constructor.get(entity)
            grid["current_car_score_raw_v7_6"] = intercept + scaler.transform(
                grid[list(features)].to_numpy(dtype=float)
            ) @ slopes
            grid["car_model_observations_v7_6"] = len(sample)
            grid["car_model_fitted_v7_6"] = fitted
            grids.append(grid)
            coefficient_rows.append(
                {
                    "season": int(season),
                    "model_entity_id": entity,
                    "feature": "baseline",
                    "coefficient": intercept,
                    "observations": len(sample),
                    "model_fitted": fitted,
                    "feature_set": feature_set,
                    "ridge_alpha": season_alpha,
                }
            )
            coefficient_rows.extend(
                {
                    "season": int(season),
                    "model_entity_id": entity,
                    "feature": feature,
                    "coefficient": float(value),
                    "observations": len(sample),
                    "model_fitted": fitted,
                    "feature_set": feature_set,
                    "ridge_alpha": season_alpha,
                }
                for feature, value in zip(features, slopes, strict=True)
            )
        cars = pd.concat(grids, ignore_index=True)
        season_target_scale = float(
            observations.loc[eligible, "relative_performance"].std(ddof=0)
        )
        if not np.isfinite(season_target_scale) or season_target_scale <= 1e-12:
            cars["current_car_score_z_v7_6"] = 0.0
        else:
            # Preserve Ridge shrinkage. Per-layout z-scoring would inflate even
            # near-zero geometry effects back to unit variance.
            cars["current_car_score_z_v7_6"] = (
                cars["current_car_score_raw_v7_6"] / season_target_scale
            )
        cars["car_performance_scale_v7_6"] = season_target_scale
        season_weights = carryover[carryover["season"].eq(int(season))]

        def prior_for_row(row: pd.Series) -> tuple[float, float]:
            total = 0.0
            weighted = 0.0
            constructor = str(row["constructor_id_f1db"])
            for weight in season_weights.itertuples(index=False):
                key = (int(weight.prior_season), constructor)
                if key not in prior_lookup.index or float(weight.effective_weight) <= 0.0:
                    continue
                total += float(weight.effective_weight)
                weighted += float(weight.effective_weight) * float(prior_lookup.loc[key])
            return (weighted / total if total else 0.0, total)

        priors = cars.apply(prior_for_row, axis=1, result_type="expand")
        cars[["constructor_prior_z", "prior_weight_total"]] = priors
        cars["car_suitability_score_v7_6"] = (
            cars["current_car_score_z_v7_6"]
            + cars["prior_weight_total"] * cars["constructor_prior_z"]
        ) / (1.0 + cars["prior_weight_total"])
        event_entities = observations[
            ["season", "round", "layout_key", "model_entity_id"]
        ].drop_duplicates()
        scores = event_entities.merge(
            cars[
                [
                    "season",
                    "layout_key",
                    "model_entity_id",
                    "current_car_score_raw_v7_6",
                    "constructor_prior_z",
                    "prior_weight_total",
                    "car_suitability_score_v7_6",
                ]
            ],
            on=["season", "layout_key", "model_entity_id"],
            how="left",
            validate="many_to_one",
        )
        scores["expected_car_win_v7_6"] = scores.groupby(["season", "round"])[
            "car_suitability_score_v7_6"
        ].transform(lambda values: _softmax(values, cfg.car_win_temperature))
        race = observations.merge(
            scores,
            on=["season", "round", "layout_key", "model_entity_id"],
            how="left",
            validate="many_to_one",
        )
        race["car_model_scope_v7_6"] = (
            f"full_same_season_static_retrospective_{feature_set}"
        )
        race["car_prior_scope_v7_6"] = (
            "constructor_lineage_regulation_and_stability_shrunk"
        )
        race_outputs.append(race)
        car_outputs.append(cars)
    return CarGeometryResultV6(
        pd.concat(race_outputs, ignore_index=True),
        pd.concat(car_outputs, ignore_index=True),
        carryover,
        pd.DataFrame(coefficient_rows),
    )


def fit_topology_gated_car_shape_v7_6(
    history: pd.DataFrame,
    profiles: pd.DataFrame,
    *,
    baseline_alpha_by_season: dict[int, float],
    augmented_alpha_by_season: dict[int, float],
    config: CarShapeConfigV76 | None = None,
) -> CarGeometryResultV6:
    """Use rich shape only on traces that passed the pre-defined topology audit."""

    cfg = config or CarShapeConfigV76()
    baseline = fit_car_shape_v7_6(
        history,
        profiles,
        feature_set="baseline_v6",
        alpha=baseline_alpha_by_season,
        config=cfg,
    )
    augmented = fit_car_shape_v7_6(
        history,
        profiles,
        feature_set="baseline_plus_shape_distributions_v9",
        alpha=augmented_alpha_by_season,
        config=cfg,
    )
    event_quality = profiles[
        ["season", "round", "circuit_id", "shape_topology_quality"]
    ].drop_duplicates(["season", "round", "circuit_id"])
    layout_quality = profiles[
        ["season", "layout_key", "shape_topology_quality"]
    ].drop_duplicates(["season", "layout_key"])

    def select_branch(
        baseline_frame: pd.DataFrame,
        augmented_frame: pd.DataFrame,
        keys: list[str],
        quality: pd.DataFrame,
    ) -> pd.DataFrame:
        base = baseline_frame.merge(quality, on=keys, how="left", validate="many_to_one")
        rich = augmented_frame.merge(quality, on=keys, how="left", validate="many_to_one")
        selected = pd.concat(
            [
                rich[rich["shape_topology_quality"].eq("high")],
                base[~base["shape_topology_quality"].eq("high")],
            ],
            ignore_index=True,
        )
        selected["car_suitability_model_branch_v7_6"] = np.where(
            selected["shape_topology_quality"].eq("high"),
            "baseline_plus_shape_distributions_v9",
            "baseline_v6_topology_fallback",
        )
        return selected

    race = select_branch(
        baseline.race_attribution,
        augmented.race_attribution,
        ["season", "round", "circuit_id"],
        event_quality,
    )
    cars = select_branch(
        baseline.car_layout_scores,
        augmented.car_layout_scores,
        ["season", "layout_key"],
        layout_quality,
    )
    coefficients = pd.concat(
        [
            baseline.coefficients.assign(model_branch="baseline_v6"),
            augmented.coefficients.assign(
                model_branch="baseline_plus_shape_distributions_v9"
            ),
        ],
        ignore_index=True,
    )
    return CarGeometryResultV6(
        race,
        cars,
        baseline.carryover_weights,
        coefficients,
    )
