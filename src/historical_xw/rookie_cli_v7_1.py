from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .plotting import plot_rating_history
from .revised_engine import build_revised_ranking
from .rookie_backcast_v7 import RookieBackcastConfigV7, build_rookie_backcast_v7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rookie-backcast-v7.1",
        description="v7 formulas with complete all-driver plot coverage.",
    )
    parser.add_argument(
        "--history",
        default="data/outputs/f1db_v6/driver_event_history_v6.parquet",
    )
    parser.add_argument("--output", default="data/outputs/rookie_backcast_v7_1")
    return parser


def plotted_driver_ids_v7_1(history: pd.DataFrame) -> set[str]:
    finished = history[history["status_class"].astype(str).eq("finished")]
    finished_counts = finished.groupby("driver_id")["event_index"].nunique()
    return set(finished_counts[finished_counts.ge(1)].index.astype(str))


def plot_coverage_v7_1(history: pd.DataFrame) -> dict[str, object]:
    plotted_ids = plotted_driver_ids_v7_1(history)
    all_ids = set(history["driver_id"].astype(str))
    antonelli = history[history["driver_id"].astype(str).eq("antonelli")]
    return {
        "plot_rule": "at_least_one_finished_race",
        "history_driver_count": len(all_ids),
        "plot_driver_count": len(plotted_ids),
        "plot_missing_driver_ids": sorted(all_ids - plotted_ids),
        "all_eligible_drivers_plotted": all(
            driver_id in plotted_ids
            for driver_id in all_ids
            if history.loc[
                history["driver_id"].astype(str).eq(driver_id), "status_class"
            ].astype(str).eq("finished").any()
        ),
        "antonelli_history_rows": int(len(antonelli)),
        "antonelli_finished_races": int(
            antonelli.loc[antonelli["status_class"].astype(str).eq("finished"), "event_index"]
            .nunique()
        ),
        "antonelli_plotted": "antonelli" in plotted_ids,
    }


def _report(audit: dict[str, object]) -> str:
    return f"""# Rookie backcast v7.1 audit

Version 7.1 is a visualization-coverage release. It reruns the unchanged v7
two-pass formulas and plots every driver with at least one race classified as
finished.

## Plot coverage

- Drivers in history: {audit['history_driver_count']}
- Drivers included in plot: {audit['plot_driver_count']}
- All eligible drivers plotted: {audit['all_eligible_drivers_plotted']}
- Missing driver IDs: {audit['plot_missing_driver_ids']}
- Antonelli history rows: {audit['antonelli_history_rows']}
- Antonelli finished races: {audit['antonelli_finished_races']}
- Antonelli plotted: {audit['antonelli_plotted']}

## Rating invariants

- Rookie evaluation rows used in their car fit: {audit['rookie_window_rows_used_in_car_fit']}
- Frozen debut-rating mismatches: {audit['frozen_debut_rating_mismatches']}
- Qualifying zero-sum maximum error: {audit['qualifying_zero_sum_max_error']:.3e}
- Race zero-sum maximum error: {audit['race_zero_sum_max_error']:.3e}

The rating engine, empirical-Bayes rookie calibration, car cross-fit, qualifying
update, xP residual, and driver/circuit adjustments are identical to v7.
"""


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(args.history)
    result = build_rookie_backcast_v7(source, RookieBackcastConfigV7())

    result.debut_estimates.to_parquet(
        output / "rookie_debut_estimates_v7_1.parquet", index=False
    )
    result.debut_estimates.drop(columns="rookie_window_indices").to_csv(
        output / "rookie_debut_estimates_v7_1.csv", index=False
    )
    result.causal_history.to_parquet(
        output / "driver_rating_history_causal_v7_1.parquet", index=False
    )
    result.retrospective_history.to_parquet(
        output / "driver_rating_history_retrospective_v7_1.parquet", index=False
    )

    plot_frame = result.retrospective_history.copy()
    plot_frame["post_rating"] = plot_frame["retrospective_rating"]
    plot_frame["rating_delta"] = (
        plot_frame["retrospective_qualifying_delta"]
        + plot_frame["retrospective_race_delta"]
    )
    plotted_ids = plotted_driver_ids_v7_1(plot_frame)
    plotted_names = sorted(
        plot_frame.loc[
            plot_frame["driver_id"].astype(str).isin(plotted_ids), "driver"
        ].dropna().astype(str).unique()
    )
    plot_rating_history(
        plot_frame,
        output / "driver_rating_history_retrospective_v7_1.html",
        minimum_races=1,
        drivers=plotted_names,
    )

    ranking = build_revised_ranking(plot_frame)
    ranking.to_parquet(output / "driver_ranking_retrospective_v7_1.parquet", index=False)
    ranking.to_csv(output / "driver_ranking_retrospective_v7_1.csv", index=False)

    audit = {**result.audit, **plot_coverage_v7_1(plot_frame)}
    (output / "rookie_backcast_v7_1_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (output / "ROOKIE_BACKCAST_V7_1_AUDIT.md").write_text(
        _report(audit), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
