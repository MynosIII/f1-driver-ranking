from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


JOIN_KEYS = ["driver_id", "season", "round"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare-v7.6",
        description="Create an inspectable v7.5 versus v7.6 rating comparison.",
    )
    parser.add_argument(
        "--v7-5",
        default=(
            "data/outputs/rookie_backcast_v7_5/"
            "driver_rating_history_retrospective_v7_5.parquet"
        ),
    )
    parser.add_argument(
        "--v7-6",
        default=(
            "data/outputs/rookie_backcast_v7_6/"
            "driver_rating_history_retrospective_v7_6.parquet"
        ),
    )
    parser.add_argument(
        "--validation",
        default="data/outputs/car_shape_validation_v7_6/model_comparison_v7_6.csv",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/car_shape_comparison_v7_6",
    )
    return parser


def _linear_slope(frame: pd.DataFrame, value_column: str) -> float:
    x = frame["event_index"].to_numpy(dtype=float)
    y = frame[value_column].to_numpy(dtype=float)
    centered_x = x - x.mean()
    return float(np.sum(centered_x * (y - y.mean())) / np.sum(centered_x**2))


def compare_histories_v7_6(
    history_v7_5: pd.DataFrame,
    history_v7_6: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "driver_id",
        "driver",
        "season",
        "round",
        "event_index",
        "retrospective_rating",
    }
    for label, frame in (("v7.5", history_v7_5), ("v7.6", history_v7_6)):
        if missing := required - set(frame.columns):
            raise ValueError(f"{label} history is missing columns: {sorted(missing)}")
        if frame.duplicated(JOIN_KEYS).any():
            raise ValueError(f"{label} history is not unique at driver-race grain")

    columns = [*JOIN_KEYS, "driver", "event_index", "retrospective_rating"]
    comparison = history_v7_5[columns].merge(
        history_v7_6[columns],
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_v7_5", "_v7_6"),
        validate="one_to_one",
    )
    comparison["rating_change_v7_6_minus_v7_5"] = (
        comparison["retrospective_rating_v7_6"]
        - comparison["retrospective_rating_v7_5"]
    )
    comparison["absolute_rating_change"] = comparison[
        "rating_change_v7_6_minus_v7_5"
    ].abs()

    trend_v7_5 = (
        comparison.groupby("event_index_v7_5", as_index=False)
        .agg(
            median_rating_v7_5=("retrospective_rating_v7_5", "median"),
            mean_rating_v7_5=("retrospective_rating_v7_5", "mean"),
        )
        .rename(columns={"event_index_v7_5": "event_index"})
    )
    trend_v7_6 = (
        comparison.groupby("event_index_v7_6", as_index=False)
        .agg(
            median_rating_v7_6=("retrospective_rating_v7_6", "median"),
            mean_rating_v7_6=("retrospective_rating_v7_6", "mean"),
        )
        .rename(columns={"event_index_v7_6": "event_index"})
    )
    trend = trend_v7_5.merge(trend_v7_6, on="event_index", validate="one_to_one")

    ordered = comparison.sort_values(
        ["driver_id", "season", "round"], kind="stable"
    )
    driver_changes = ordered.groupby("driver_id", as_index=False).agg(
        driver=("driver_v7_6", "last"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        race_rows=("round", "size"),
        final_rating_v7_5=("retrospective_rating_v7_5", "last"),
        final_rating_v7_6=("retrospective_rating_v7_6", "last"),
        peak_rating_v7_5=("retrospective_rating_v7_5", "max"),
        peak_rating_v7_6=("retrospective_rating_v7_6", "max"),
        mean_absolute_race_change=("absolute_rating_change", "mean"),
    )
    driver_changes["final_rating_change"] = (
        driver_changes["final_rating_v7_6"] - driver_changes["final_rating_v7_5"]
    )
    driver_changes["peak_rating_change"] = (
        driver_changes["peak_rating_v7_6"] - driver_changes["peak_rating_v7_5"]
    )

    summary = {
        "matched_driver_race_rows": int(len(comparison)),
        "drivers": int(driver_changes["driver_id"].nunique()),
        "rating_correlation": float(
            comparison["retrospective_rating_v7_5"].corr(
                comparison["retrospective_rating_v7_6"]
            )
        ),
        "mean_absolute_rating_change": float(comparison["absolute_rating_change"].mean()),
        "p95_absolute_rating_change": float(
            comparison["absolute_rating_change"].quantile(0.95)
        ),
        "maximum_absolute_rating_change": float(comparison["absolute_rating_change"].max()),
        "first_event_median_v7_5": float(trend.iloc[0]["median_rating_v7_5"]),
        "last_event_median_v7_5": float(trend.iloc[-1]["median_rating_v7_5"]),
        "first_event_median_v7_6": float(trend.iloc[0]["median_rating_v7_6"]),
        "last_event_median_v7_6": float(trend.iloc[-1]["median_rating_v7_6"]),
        "median_linear_slope_per_race_v7_5": _linear_slope(
            trend, "median_rating_v7_5"
        ),
        "median_linear_slope_per_race_v7_6": _linear_slope(
            trend, "median_rating_v7_6"
        ),
    }
    summary["first_to_last_median_change_v7_5"] = (
        summary["last_event_median_v7_5"] - summary["first_event_median_v7_5"]
    )
    summary["first_to_last_median_change_v7_6"] = (
        summary["last_event_median_v7_6"] - summary["first_event_median_v7_6"]
    )
    return driver_changes, trend, summary


def _comparison_figure(
    driver_changes: pd.DataFrame,
    trend: pd.DataFrame,
    validation: pd.DataFrame,
    summary: dict[str, Any],
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Field median through history",
            "Driver peak ratings",
            "Largest final-rating changes",
            "Out-of-sample car-performance MAE",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    figure.add_trace(
        go.Scatter(
            x=trend["event_index"],
            y=trend["median_rating_v7_5"],
            name="v7.5 median",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=trend["event_index"],
            y=trend["median_rating_v7_6"],
            name="v7.6 median",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=driver_changes["peak_rating_v7_5"],
            y=driver_changes["peak_rating_v7_6"],
            text=driver_changes["driver"],
            customdata=driver_changes[["peak_rating_change"]],
            hovertemplate=(
                "%{text}<br>v7.5: %{x:.1f}<br>v7.6: %{y:.1f}"
                "<br>change: %{customdata[0]:+.1f}<extra></extra>"
            ),
            name="driver peaks",
            mode="markers",
            marker={"size": 6, "opacity": 0.65},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    lower = float(
        min(driver_changes["peak_rating_v7_5"].min(), driver_changes["peak_rating_v7_6"].min())
    )
    upper = float(
        max(driver_changes["peak_rating_v7_5"].max(), driver_changes["peak_rating_v7_6"].max())
    )
    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            name="no change",
            mode="lines",
            line={"dash": "dash", "color": "#888"},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    largest = driver_changes.loc[
        driver_changes["final_rating_change"].abs().nlargest(20).index
    ].sort_values("final_rating_change")
    figure.add_trace(
        go.Bar(
            x=largest["final_rating_change"],
            y=largest["driver"],
            orientation="h",
            marker_color=np.where(
                largest["final_rating_change"].ge(0), "#2ca02c", "#d62728"
            ),
            name="final change",
            showlegend=False,
            hovertemplate="%{y}: %{x:+.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    labels = {
        "baseline_v6": "v7.5 baseline",
        "shape_composites_v9": "shape composites",
        "shape_distributions_v9": "shape distributions",
        "baseline_plus_shape_distributions_v9": "baseline + shape",
        "topology_gated_baseline_plus_shape_v7_6": "v7.6 topology gate",
    }
    validation = validation.copy()
    validation["display_model"] = validation["model"].map(labels).fillna(validation["model"])
    figure.add_trace(
        go.Bar(
            x=validation["display_model"],
            y=validation["mae"],
            text=validation["mae"].map(lambda value: f"{value:.6f}"),
            textposition="outside",
            marker_color=[
                "#ff7f0e" if model == "topology_gated_baseline_plus_shape_v7_6" else "#4c78a8"
                for model in validation["model"]
            ],
            name="validation MAE",
            showlegend=False,
            hovertemplate="%{x}<br>MAE: %{y:.9f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Championship race number", row=1, col=1)
    figure.update_yaxes(title_text="Median retrospective rating", row=1, col=1)
    figure.update_xaxes(title_text="v7.5 peak", row=1, col=2)
    figure.update_yaxes(title_text="v7.6 peak", row=1, col=2)
    figure.update_xaxes(title_text="v7.6 minus v7.5 rating points", row=2, col=1)
    figure.update_yaxes(title_text="MAE (lower is better)", row=2, col=2)
    figure.update_xaxes(tickangle=-22, row=2, col=2)
    figure.update_layout(
        title=(
            "Circuit-shape v7.6 validation and rating effect"
            f"<br><sup>Out-of-sample gain: 0.043%; rating correlation: "
            f"{summary['rating_correlation']:.4f}; mean absolute rating change: "
            f"{summary['mean_absolute_rating_change']:.1f}</sup>"
        ),
        template="plotly_dark",
        height=900,
        margin={"l": 90, "r": 40, "t": 110, "b": 90},
    )
    return figure


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history_v7_5 = pd.read_parquet(args.v7_5)
    history_v7_6 = pd.read_parquet(args.v7_6)
    validation = pd.read_csv(args.validation)
    driver_changes, trend, summary = compare_histories_v7_6(
        history_v7_5, history_v7_6
    )
    driver_changes.to_csv(output / "driver_changes_v7_5_vs_v7_6.csv", index=False)
    trend.to_csv(output / "event_trend_v7_5_vs_v7_6.csv", index=False)
    (output / "comparison_summary_v7_6.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    figure = _comparison_figure(driver_changes, trend, validation, summary)
    figure.write_html(
        output / "car_shape_v7_5_vs_v7_6.html",
        include_plotlyjs=True,
        full_html=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
