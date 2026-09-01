from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SeasonScopeResultV75:
    history: pd.DataFrame
    excluded_events: pd.DataFrame
    excluded_driver_ids: tuple[str, ...]
    audit: dict[str, Any]


def historical_indianapolis_mask_v7_5(history: pd.DataFrame) -> pd.Series:
    """Identify the 1950-1960 Indianapolis 500 WDC rounds, not modern US GPs."""

    required = {"season", "event"}
    if missing := required - set(history.columns):
        raise ValueError(f"Indianapolis scope filter is missing columns: {sorted(missing)}")
    seasons = pd.to_numeric(history["season"], errors="coerce")
    event_names = (
        history["event"]
        .astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    return seasons.between(1950, 1960) & event_names.eq("indianapolis 500")


def filter_historical_indianapolis_v7_5(
    history: pd.DataFrame,
) -> SeasonScopeResultV75:
    """Remove only the historical Indy 500 championship population.

    Drivers who raced elsewhere remain in the model. Drivers whose complete source
    history consists of the excluded Indy rounds disappear naturally with those rows.
    """

    required = {"season", "round", "event", "driver_id"}
    if missing := required - set(history.columns):
        raise ValueError(f"Season scope is missing columns: {sorted(missing)}")
    data = history.copy()
    data["driver_id"] = data["driver_id"].astype(str)
    excluded_mask = historical_indianapolis_mask_v7_5(data)
    excluded_rows = data.loc[excluded_mask].copy()
    retained = data.loc[~excluded_mask].copy()
    excluded_events = (
        excluded_rows[["season", "round", "event"]]
        .drop_duplicates()
        .sort_values(["season", "round"], kind="stable")
        .reset_index(drop=True)
    )
    retained_ids = set(retained["driver_id"])
    excluded_driver_ids = tuple(
        sorted(set(excluded_rows["driver_id"]) - retained_ids)
    )
    audit = {
        "season_scope_version": "7.5",
        "historical_indianapolis_rule": (
            "season_1950_to_1960_and_event_name_exactly_indianapolis_500"
        ),
        "historical_indianapolis_events_removed": int(len(excluded_events)),
        "historical_indianapolis_rows_removed": int(excluded_mask.sum()),
        "historical_indianapolis_only_drivers_removed": int(len(excluded_driver_ids)),
        "history_rows_before_scope_filter": int(len(data)),
        "history_rows_after_scope_filter": int(len(retained)),
        "modern_indianapolis_events_removed": 0,
    }
    return SeasonScopeResultV75(
        retained.reset_index(drop=True),
        excluded_events,
        excluded_driver_ids,
        audit,
    )


def exclude_event_keys_v7_5(
    frame: pd.DataFrame | None,
    excluded_events: pd.DataFrame,
) -> pd.DataFrame | None:
    """Apply the same event-key exclusion to timing or race-context inputs."""

    if frame is None:
        return None
    if frame.empty or excluded_events.empty:
        return frame.copy()
    keys = ["season", "round"]
    if missing := set(keys) - set(frame.columns):
        raise ValueError(f"Event-key filter is missing columns: {sorted(missing)}")
    excluded = excluded_events[keys].drop_duplicates().assign(_excluded_event=True)
    merged = frame.merge(excluded, on=keys, how="left", validate="many_to_one")
    return merged.loc[merged["_excluded_event"].isna()].drop(
        columns="_excluded_event"
    ).reset_index(drop=True)
