from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .car_shape_v7_6 import (
    CarShapeConfigV76,
    CarShapeValidationResultV76,
    fit_topology_gated_car_shape_v7_6,
    validate_car_shape_v7_6,
)
from .pairwise_margin_v7_4 import PairwiseMarginConfigV74
from .rookie_backcast_v7_2 import RookieBackcastConfigV72
from .rookie_backcast_v7_5 import RookieBackcastResultV75, build_rookie_backcast_v7_5
from .season_scope_v7_5 import filter_historical_indianapolis_v7_5


@dataclass(slots=True)
class RookieBackcastResultV76:
    rating: RookieBackcastResultV75
    enhanced_history: pd.DataFrame
    car_layout_scores: pd.DataFrame
    car_coefficients: pd.DataFrame
    validation: CarShapeValidationResultV76
    audit: dict[str, Any]


def _alpha_map(
    validation: CarShapeValidationResultV76,
    model: str,
) -> dict[int, float]:
    rows = validation.alpha_selection[
        validation.alpha_selection["model"].eq(model)
    ]
    return {
        int(row.target_season): float(row.selected_alpha)
        for row in rows.itertuples(index=False)
    }


def build_rookie_backcast_v7_6(
    history: pd.DataFrame,
    timing: pd.DataFrame,
    shape_profiles: pd.DataFrame,
    race_context: pd.DataFrame | None = None,
    rating_config: RookieBackcastConfigV72 | None = None,
    margin_config: PairwiseMarginConfigV74 | None = None,
    car_config: CarShapeConfigV76 | None = None,
    validation: CarShapeValidationResultV76 | None = None,
) -> RookieBackcastResultV76:
    """Build parallel v7.6 with topology-gated shape suitability and v7.5 ratings."""

    rating_cfg = rating_config or RookieBackcastConfigV72()
    margin_cfg = margin_config or PairwiseMarginConfigV74()
    car_cfg = car_config or CarShapeConfigV76()
    scope = filter_historical_indianapolis_v7_5(history)
    scoped = scope.history
    validated = validation or validate_car_shape_v7_6(scoped, shape_profiles, car_cfg)
    baseline_alpha = _alpha_map(validated, "baseline_v6")
    augmented_alpha = _alpha_map(
        validated, "baseline_plus_shape_distributions_v9"
    )
    car = fit_topology_gated_car_shape_v7_6(
        scoped,
        shape_profiles,
        baseline_alpha_by_season=baseline_alpha,
        augmented_alpha_by_season=augmented_alpha,
        config=car_cfg,
    )
    enhanced = car.race_attribution.copy()
    enhanced["car_suitability_score_v6_baseline"] = enhanced[
        "car_suitability_score_v6"
    ]
    enhanced["expected_car_win_v6_baseline"] = enhanced["expected_car_win_v6"]
    enhanced["car_suitability_score_v6"] = enhanced["car_suitability_score_v7_6"]
    enhanced["expected_car_win_v6"] = enhanced["expected_car_win_v7_6"]
    enhanced["car_model_scope_v6_baseline"] = enhanced["car_model_scope_v6"]
    enhanced["car_model_scope_v6"] = enhanced["car_model_scope_v7_6"]
    enhanced = enhanced.sort_values(
        ["season", "round", "position"], kind="stable"
    ).reset_index(drop=True)
    rating = build_rookie_backcast_v7_5(
        enhanced,
        timing,
        race_context,
        rating_cfg,
        margin_cfg,
    )
    distinct_scores = car.car_layout_scores.groupby(
        ["season", "layout_key", "model_entity_id"]
    )["car_suitability_score_v7_6"].nunique()
    expected_sum_error = (
        enhanced.drop_duplicates(["season", "round", "model_entity_id"])
        .groupby(["season", "round"])["expected_car_win_v7_6"]
        .sum()
        .sub(1.0)
        .abs()
        .max()
    )
    audit = {
        **rating.audit,
        **scope.audit,
        "version": "7.6",
        "circuit_shape_version": "9",
        "car_geometry_validation_design": validated.audit["validation_design"],
        "car_geometry_selected_model": validated.audit["winning_feature_set"],
        "car_geometry_baseline_mae": validated.audit["baseline_mae"],
        "car_geometry_v7_6_mae": validated.audit["winning_mae"],
        "car_geometry_relative_mae_improvement": validated.audit[
            "winning_relative_mae_improvement"
        ],
        "car_geometry_season_block_improvement_ci_95": validated.audit[
            "winning_season_block_mean_improvement_ci_95"
        ],
        "car_geometry_target_season_used_for_alpha_selection": validated.audit[
            "target_season_used_for_alpha_selection"
        ],
        "car_geometry_topology_gate": validated.audit["topology_gate"],
        "car_geometry_score_scaling": (
            "raw_ridge_prediction_divided_by_same_season_observed_car_performance_sd"
        ),
        "per_layout_zscore_amplification_removed": True,
        "high_quality_shape_driver_rows": int(
            enhanced["car_suitability_model_branch_v7_6"]
            .eq("baseline_plus_shape_distributions_v9")
            .sum()
        ),
        "baseline_topology_fallback_driver_rows": int(
            enhanced["car_suitability_model_branch_v7_6"]
            .eq("baseline_v6_topology_fallback")
            .sum()
        ),
        "static_car_layout_max_unique_scores_v7_6": int(distinct_scores.max()),
        "expected_car_win_sum_max_error_v7_6": float(expected_sum_error),
        "v7_5_code_and_outputs_preserved": True,
    }
    return RookieBackcastResultV76(
        rating,
        enhanced,
        car.car_layout_scores,
        car.coefficients,
        validated,
        audit,
    )
