from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .car_shape_v7_6 import read_car_shape_validation_v7_6
from .f1db_race_context_v7_4 import load_f1db_race_context_v7_4
from .pairwise_margin_v7_4 import PairwiseMarginConfigV74
from .plotting import plot_rating_history
from .race_margin_v7_3 import load_f1db_race_timing_v7_3
from .revised_engine import build_revised_ranking
from .rookie_backcast_v7_2 import RookieBackcastConfigV72
from .rookie_backcast_v7_6 import build_rookie_backcast_v7_6
from .rookie_cli_v7_1 import plot_coverage_v7_1, plotted_driver_ids_v7_1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rookie-backcast-v7.6",
        description="Topology-gated circuit-shape suitability and v7.5 rating architecture.",
    )
    parser.add_argument(
        "--history",
        default="data/outputs/f1db_v6/driver_event_history_v6.parquet",
    )
    parser.add_argument(
        "--shape-profiles",
        default="data/outputs/circuit_shape_v9/historical_event_shape_profiles_v9.parquet",
    )
    parser.add_argument(
        "--validation",
        default="data/outputs/car_shape_validation_v7_6",
    )
    parser.add_argument(
        "--timing",
        default="data/outputs/rookie_backcast_v7_3/f1db_race_timing_v7_3.parquet",
    )
    parser.add_argument("--f1db-root", default="data/cache/f1db")
    parser.add_argument("--output", default="data/outputs/rookie_backcast_v7_6")
    return parser


def _report(audit: dict[str, object]) -> str:
    lower, upper = audit["car_geometry_season_block_improvement_ci_95"]
    return f"""# Circuit-shape rating v7.6 audit

Version 7.6 is parallel to v7.5. It changes only the car/circuit suitability
block and preserves Indianapolis filtering, qualifying, timing gaps, calendar K,
rookie backcasting and annual anchoring.

## Out-of-sample geometry validation

- Design: {audit['car_geometry_validation_design']}
- Selected model: {audit['car_geometry_selected_model']}
- Baseline MAE: {audit['car_geometry_baseline_mae']:.9f}
- V7.6 MAE: {audit['car_geometry_v7_6_mae']:.9f}
- Relative MAE improvement: {audit['car_geometry_relative_mae_improvement']:.4%}
- Season-block mean-improvement 95% interval: [{lower:.9f}, {upper:.9f}]
- Target season used to select Ridge alpha: {audit['car_geometry_target_season_used_for_alpha_selection']}

The gain is positive but very small. V7.6 should be interpreted as a more
specific circuit attribution model, not as a dramatic predictive breakthrough.

## Reliability gate

- Rule: {audit['car_geometry_topology_gate']}
- Driver-race rows using rich shape: {audit['high_quality_shape_driver_rows']}
- Driver-race rows using baseline fallback: {audit['baseline_topology_fallback_driver_rows']}

High-topology-quality traces use straight/altitude plus ordered turn distributions.
Review/low traces fall back to the prior straight/altitude feature block.

## Rating invariants

- Static car-layout maximum unique scores: {audit['static_car_layout_max_unique_scores_v7_6']}
- Expected-car-win sum error: {audit['expected_car_win_sum_max_error_v7_6']:.3e}
- Race K budget error: {audit['race_k_budget_max_error']:.3e}
- Qualifying K budget error: {audit['qualifying_k_budget_max_error']:.3e}
- Race zero-sum error: {audit['race_zero_sum_max_error']:.3e}
- Gap zero-sum error: {audit['margin_adjustment_zero_sum_max_error']:.3e}
- Drivers plotted: {audit['plot_driver_count']}
- Antonelli plotted: {audit['antonelli_plotted']}
- V7.5 preserved: {audit['v7_5_code_and_outputs_preserved']}
"""


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.history)
    profiles = pd.read_parquet(args.shape_profiles)
    validation = read_car_shape_validation_v7_6(Path(args.validation))
    timing_path = Path(args.timing)
    timing = (
        pd.read_parquet(timing_path)
        if timing_path.exists()
        else load_f1db_race_timing_v7_3(args.f1db_root)
    )
    context = load_f1db_race_context_v7_4(args.f1db_root)
    result = build_rookie_backcast_v7_6(
        source,
        timing,
        profiles,
        context,
        RookieBackcastConfigV72(),
        PairwiseMarginConfigV74(),
        validation=validation,
    )
    rating = result.rating
    result.enhanced_history.to_parquet(
        output / "driver_event_history_shape_v7_6.parquet", index=False
    )
    result.car_layout_scores.to_parquet(
        output / "car_layout_scores_v7_6.parquet", index=False
    )
    result.car_coefficients.to_parquet(
        output / "car_shape_coefficients_v7_6.parquet", index=False
    )
    rating.pairwise_margin_features.to_parquet(
        output / "all_pair_expected_gap_features_v7_6.parquet", index=False
    )
    rating.driver_race_margin_features.to_parquet(
        output / "driver_race_margin_features_v7_6.parquet", index=False
    )
    rating.pairwise_calibration.to_csv(
        output / "pairwise_gap_calibration_v7_6.csv", index=False
    )
    rating.debut_estimates.to_parquet(
        output / "rookie_debut_estimates_v7_6.parquet", index=False
    )
    rating.debut_estimates.drop(columns="rookie_window_indices").to_csv(
        output / "rookie_debut_estimates_v7_6.csv", index=False
    )
    rating.season_anchors.to_csv(output / "season_rookie_anchors_v7_6.csv", index=False)
    rating.provisional_history.to_parquet(
        output / "driver_rating_history_provisional_v7_6.parquet", index=False
    )
    rating.causal_history.to_parquet(
        output / "driver_rating_history_causal_v7_6.parquet", index=False
    )
    rating.retrospective_history.to_parquet(
        output / "driver_rating_history_retrospective_v7_6.parquet", index=False
    )

    plot_frame = rating.retrospective_history.copy()
    plot_frame["post_rating"] = plot_frame["retrospective_rating"]
    plot_frame["rating_delta"] = (
        plot_frame["retrospective_qualifying_delta"]
        + plot_frame["retrospective_race_delta"]
    )
    plotted_ids = plotted_driver_ids_v7_1(plot_frame)
    plotted_names = sorted(
        plot_frame.loc[plot_frame["driver_id"].astype(str).isin(plotted_ids), "driver"]
        .dropna()
        .astype(str)
        .unique()
    )
    plot_rating_history(
        plot_frame,
        output / "driver_rating_history_retrospective_v7_6.html",
        minimum_races=1,
        drivers=plotted_names,
    )
    ranking = build_revised_ranking(plot_frame)
    ranking.to_parquet(output / "driver_ranking_retrospective_v7_6.parquet", index=False)
    ranking.to_csv(output / "driver_ranking_retrospective_v7_6.csv", index=False)
    audit = {**result.audit, **plot_coverage_v7_1(plot_frame)}
    (output / "rookie_backcast_v7_6_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output / "ROOKIE_BACKCAST_V7_6_AUDIT.md").write_text(
        _report(audit), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
