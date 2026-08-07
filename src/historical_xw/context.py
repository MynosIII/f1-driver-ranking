from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from .outcomes import OutcomeClass, classify_outcome


def regulation_era(season: int) -> str:
    boundaries = (
        (1960, "1950-1960_front_engine"),
        (1965, "1961-1965_1.5l"),
        (1982, "1966-1982_3l_and_ground_effect"),
        (1988, "1983-1988_turbo"),
        (1994, "1989-1994_3.5l"),
        (2005, "1995-2005_3l_v10"),
        (2013, "2006-2013_v8"),
        (2016, "2014-2016_hybrid"),
        (2021, "2017-2021_wide_hybrid"),
        (2025, "2022-2025_ground_effect"),
    )
    for last_season, label in boundaries:
        if season <= last_season:
            return label
    return "2026_present_active_aero"


def circuit_passability(events: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-safe overtaking proxy from earlier editions of each circuit.

    Exact overtake counts are unavailable for the complete championship history. The
    proxy is positive grid-to-finish movement among classified finishers. The last
    eight earlier races at the circuit are exponentially weighted toward recency.
    """

    event_keys = ["season", "round", "circuit_id"]
    joined = (
        results.copy()
        if "circuit_id" in results
        else results.merge(events[event_keys], on=["season", "round"], how="left")
    )
    joined["outcome_class"] = joined["status"].map(classify_outcome).astype(str)
    clean = joined[
        joined["outcome_class"].eq(OutcomeClass.FINISHED.value)
        & joined["grid"].gt(0)
    ].copy()
    clean["positive_grid_gain"] = (clean["grid"] - clean["position"]).clip(lower=0)
    race_proxy = (
        clean.groupby(event_keys, as_index=False)
        .agg(
            mean_positive_grid_gain=("positive_grid_gain", "mean"),
            classified_finishers=("driver_id", "size"),
        )
        .sort_values(["season", "round"])
    )

    output: list[dict[str, Any]] = []
    ordered_events = events.sort_values(["season", "round"])
    prior_all: list[float] = []
    prior_by_circuit: dict[str, list[float]] = {}
    proxy_lookup = race_proxy.set_index(["season", "round"])["mean_positive_grid_gain"]
    for event in ordered_events.itertuples():
        circuit_id = str(event.circuit_id)
        circuit_history = prior_by_circuit.get(circuit_id, [])[-8:]
        if circuit_history:
            ages = np.arange(len(circuit_history) - 1, -1, -1)
            weights = np.power(0.65, ages)
            raw = float(np.average(circuit_history, weights=weights))
            source = "prior_same_circuit_grid_movement"
        elif prior_all:
            raw = float(np.median(prior_all[-50:]))
            source = "prior_championship_grid_movement"
        else:
            raw = 1.0
            source = "neutral_no_prior_history"
        # Bounded transform: one gained position -> .33, two -> .50, four -> .67.
        index = raw / (raw + 2.0)
        output.append(
            {
                "season": int(event.season),
                "round": int(event.round),
                "circuit_id": circuit_id,
                "overtake_index": float(np.clip(index, 0.0, 1.0)),
                "overtake_proxy_raw": raw,
                "circuit_history_races": len(circuit_history),
                "overtake_source": source,
            }
        )
        current = proxy_lookup.get((event.season, event.round), np.nan)
        if np.isfinite(current):
            value = float(current)
            prior_all.append(value)
            prior_by_circuit.setdefault(circuit_id, []).append(value)
    return pd.DataFrame(output)


@dataclass(frozen=True, slots=True)
class WeatherObservation:
    wet_fraction: float
    air_temperature: float
    track_temperature: float
    wind_speed: float
    source: str


class HistoricalWeatherClient:
    """Cached Open-Meteo ERA5 reanalysis for the race-location time window."""

    endpoint = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=15), reraise=True)
    def _download(self, params: dict[str, Any]) -> dict[str, Any]:
        response = requests.get(self.endpoint, params=params, timeout=90)
        response.raise_for_status()
        return response.json()

    def event_weather(
        self,
        latitude: float,
        longitude: float,
        date: str,
        start_time: str | None,
    ) -> WeatherObservation:
        params: dict[str, Any] = {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
            "start_date": date,
            "end_date": date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]
        target = self.cache / f"{date}_{key}.json"
        if not target.exists():
            target.write_text(json.dumps(self._download(params)), encoding="utf-8")
        payload = json.loads(target.read_text(encoding="utf-8"))
        hourly = pd.DataFrame(payload["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"])
        if start_time:
            cleaned = start_time.removesuffix("Z")
            start = pd.Timestamp(datetime.fromisoformat(f"{date}T{cleaned}"))
            window = hourly[hourly["time"].between(start - pd.Timedelta(hours=1), start + pd.Timedelta(hours=3))]
        else:
            window = hourly.iloc[12:17]
        if window.empty:
            window = hourly
        precipitation = pd.to_numeric(window["precipitation"], errors="coerce").fillna(0.0)
        air = float(pd.to_numeric(window["temperature_2m"], errors="coerce").mean())
        wind = float(pd.to_numeric(window["wind_speed_10m"], errors="coerce").mean())
        return WeatherObservation(
            wet_fraction=float((precipitation > 0.1).mean()),
            air_temperature=air,
            track_temperature=air,
            wind_speed=wind,
            source="open_meteo_era5_air_temperature_track_proxy",
        )
