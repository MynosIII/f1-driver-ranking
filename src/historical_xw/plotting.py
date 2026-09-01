from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def plot_rating_history(
    history: pd.DataFrame,
    output: Path,
    minimum_races: int = 25,
    drivers: list[str] | None = None,
) -> Path:
    """Write the legacy interactive Elo-over-races chart from maintained outputs."""

    required = {"season", "round", "driver", "post_rating"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing rating-history columns: {sorted(missing)}")
    data = history.sort_values(["season", "round"]).copy()
    events = data[["season", "round"]].drop_duplicates().reset_index(drop=True)
    events["race_index"] = events.index + 1
    data = data.merge(events, on=["season", "round"], how="left")
    counts = data.groupby("driver")["race_index"].nunique()
    data = data[data["driver"].isin(counts[counts >= minimum_races].index)]
    if drivers:
        data = data[data["driver"].isin(drivers)]
    figure = px.line(
        data,
        x="race_index",
        y="post_rating",
        color="driver",
        hover_data=["season", "round", "event", "rating_delta"],
        title="Historical xW/xP-adjusted Formula 1 driver rating",
        labels={"race_index": "Championship race index", "post_rating": "Driver rating"},
    )
    figure.update_layout(template="plotly_white", height=750, width=1250)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs="cdn")
    return output
