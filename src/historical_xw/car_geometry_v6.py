from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .model_rules_v6 import CarryoverConfigV6, calculate_carryover_weights_v6
from .wikipedia_geometry_v5 import GEOMETRY_MODEL_FEATURES


@dataclass(frozen=True, slots=True)
class CarGeometryConfigV6:
    ridge_alpha: float = 100.0
    minimum_races: int = 4
    car_win_temperature: float = 0.35
    carryover: CarryoverConfigV6 = CarryoverConfigV6()


@dataclass(slots=True)
class CarGeometryResultV6:
    race_attribution: pd.DataFrame
    car_layout_scores: pd.DataFrame
    carryover_weights: pd.DataFrame
    coefficients: pd.DataFrame


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    deviation = float(numeric.std(ddof=0))
    if deviation < 1e-12 or not np.isfinite(deviation):
        return numeric * 0.0
    return (numeric - float(numeric.mean())) / deviation


def _softmax(values: pd.Series, temperature: float) -> np.ndarray:
    logits = values.to_numpy(dtype=float) / temperature
    logits -= float(np.max(logits))
    weights = np.exp(np.clip(logits, -50.0, 50.0))
    return weights / weights.sum()


def _constructor_scores(history: pd.DataFrame) -> pd.DataFrame:
    source = history[
        history["car_fit_eligible"] & history["observed_performance"].notna()
    ].copy()
    values = (
        source.groupby(["season", "constructor_id_f1db"], as_index=False)[
            "observed_performance"
        ]
        .mean()
        .rename(
            columns={
                "constructor_id_f1db": "constructor_id",
                "observed_performance": "raw_score",
            }
        )
    )
    values["positive"] = values["raw_score"] - values.groupby("season")[
        "raw_score"
    ].transform("min") + 1e-6
    values["score"] = values["positive"] / values.groupby("season")["positive"].transform(
        "sum"
    )
    values["score_z"] = values.groupby("season")["score"].transform(_zscore)
    return values


def fit_car_geometry_v6(
    history: pd.DataFrame,
    profiles: pd.DataFrame,
    config: CarGeometryConfigV6 | None = None,
) -> CarGeometryResultV6:
    """Fit one static circuit-suitability curve per chassis-season.

    Exact F1DB chassis IDs are used where available. Ambiguous historical rows
    retain their ``constructor:...`` fallback identity. All races in a season
    share one fitted curve, so the same chassis at the same layout receives the
    same car expectation even when the layout appears twice.
    """

    cfg = config or CarGeometryConfigV6()
    required_history = {
        "season",
        "round",
        "circuit_id",
        "model_entity_id",
        "constructor_id_f1db",
        "observed_performance",
    }
    required_profiles = {
        "season",
        "round",
        "circuit_id",
        "layout_key",
        *GEOMETRY_MODEL_FEATURES,
    }
    if missing := required_history - set(history.columns):
        raise ValueError(f"History is missing columns: {sorted(missing)}")
    if missing := required_profiles - set(profiles.columns):
        raise ValueError(f"Profiles are missing columns: {sorted(missing)}")
    data = history.copy()
    if "rating_eligible" in data:
        data["car_fit_eligible"] = data["rating_eligible"].fillna(False).astype(bool)
    else:
        data["car_fit_eligible"] = ~data.get(
            "status_class_f1db", pd.Series("finished", index=data.index)
        ).isin(["mechanical", "dns", "other"])
    constructor_scores = _constructor_scores(data)
    carryover = calculate_carryover_weights_v6(
        constructor_scores[["season", "constructor_id", "score"]], cfg.carryover
    )
    prior_lookup = constructor_scores.set_index(["season", "constructor_id"])["score_z"]

    features = list(GEOMETRY_MODEL_FEATURES)
    race_outputs: list[pd.DataFrame] = []
    car_outputs: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, object]] = []
    for season, season_history in data.groupby("season", sort=True):
        season_profiles = profiles[profiles["season"].eq(season)].drop_duplicates(
            ["season", "round", "circuit_id"]
        )
        if season_profiles.empty:
            continue
        observations = season_history.merge(
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
            observations.sort_values(["round"])
            .drop_duplicates("model_entity_id")
            .set_index("model_entity_id")["constructor_id_f1db"]
            .astype(str)
        )
        unique_layouts = season_profiles.drop_duplicates("layout_key")
        scaler = StandardScaler().fit(unique_layouts[features].to_numpy(dtype=float))
        entity_grids: list[pd.DataFrame] = []
        for entity in sorted(observations["model_entity_id"].dropna().astype(str).unique()):
            sample = observations[
                observations["model_entity_id"].astype(str).eq(entity)
            ].dropna(subset=["relative_performance"])
            intercept = float(sample["relative_performance"].mean()) if len(sample) else 0.0
            slopes = np.zeros(len(features), dtype=float)
            fitted = len(sample) >= cfg.minimum_races and sample["layout_key"].nunique() >= 3
            if fitted:
                model = Ridge(alpha=cfg.ridge_alpha)
                model.fit(
                    scaler.transform(sample[features].to_numpy(dtype=float)),
                    sample["relative_performance"].to_numpy(dtype=float),
                )
                intercept = float(model.intercept_)
                slopes = np.asarray(model.coef_, dtype=float)
            grid = unique_layouts[
                ["season", "layout_key", "circuit_id", *features]
            ].copy()
            grid["model_entity_id"] = entity
            grid["constructor_id_f1db"] = entity_constructor.get(entity)
            grid["current_car_score_raw"] = intercept + scaler.transform(
                grid[features].to_numpy(dtype=float)
            ) @ slopes
            grid["car_model_observations"] = len(sample)
            grid["car_model_fitted"] = fitted
            entity_grids.append(grid)
            coefficient_rows.append(
                {
                    "season": int(season),
                    "model_entity_id": entity,
                    "feature": "baseline",
                    "coefficient": intercept,
                    "observations": len(sample),
                    "model_fitted": fitted,
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
                }
                for feature, value in zip(features, slopes, strict=True)
            )
        cars = pd.concat(entity_grids, ignore_index=True)
        cars["current_car_score_z"] = cars.groupby("layout_key")[
            "current_car_score_raw"
        ].transform(_zscore)
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
        cars["car_suitability_score_v6"] = (
            cars["current_car_score_z"]
            + cars["prior_weight_total"] * cars["constructor_prior_z"]
        ) / (1.0 + cars["prior_weight_total"])

        event_entities = observations[
            ["season", "round", "layout_key", "model_entity_id"]
        ].drop_duplicates()
        event_scores = event_entities.merge(
            cars[
                [
                    "season",
                    "layout_key",
                    "model_entity_id",
                    "current_car_score_raw",
                    "constructor_prior_z",
                    "prior_weight_total",
                    "car_suitability_score_v6",
                ]
            ],
            on=["season", "layout_key", "model_entity_id"],
            how="left",
            validate="many_to_one",
        )
        event_scores["expected_car_win_v6"] = event_scores.groupby(["season", "round"])[
            "car_suitability_score_v6"
        ].transform(lambda values: _softmax(values, cfg.car_win_temperature))
        race_output = observations.merge(
            event_scores,
            on=["season", "round", "layout_key", "model_entity_id"],
            how="left",
            validate="many_to_one",
        )
        race_output["car_model_scope_v6"] = "full_same_season_static_retrospective"
        race_output["car_prior_scope_v6"] = "constructor_lineage_regulation_and_stability_shrunk"
        race_outputs.append(race_output)
        car_outputs.append(cars)

    if not race_outputs:
        raise ValueError("No seasons have both F1DB car identities and geometry profiles")
    return CarGeometryResultV6(
        race_attribution=pd.concat(race_outputs, ignore_index=True),
        car_layout_scores=pd.concat(car_outputs, ignore_index=True),
        carryover_weights=carryover,
        coefficients=pd.DataFrame(coefficient_rows),
    )
