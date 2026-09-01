from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests
from jinja2 import Template
from tenacity import retry, stop_after_attempt, wait_exponential

from .decomposition import decompose_event
from .domain import Entry, HistoricalContext, SimulationConfig
from .context import (
    HistoricalWeatherClient,
    WeatherObservation,
    circuit_passability,
    regulation_era,
)
from .outcomes import classify_outcome, observed_driver_performance


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20), reraise=True)
def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


class JolpicaClient:
    base = "https://api.jolpi.ca/ergast/f1"

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, path: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}?limit={limit}&offset={offset}"
        key = hashlib.sha256(url.encode()).hexdigest()[:20]
        target = self.cache / f"{path.replace('/', '_')}_{key}.json"
        if not target.exists():
            target.write_text(json.dumps(_get_json(url)), encoding="utf-8")
        return json.loads(target.read_text(encoding="utf-8"))

    def season_results(self, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        result_rows, event_rows = [], []
        races_by_round: dict[int, dict[str, Any]] = {}
        offset, total = 0, 1
        while offset < total:
            payload = self.get(f"{season}/results.json", limit=100, offset=offset)["MRData"]
            total = int(payload["total"])
            page_size = int(payload["limit"])
            for page_race in payload["RaceTable"]["Races"]:
                round_no = int(page_race["round"])
                if round_no not in races_by_round:
                    races_by_round[round_no] = {**page_race, "Results": []}
                races_by_round[round_no]["Results"].extend(page_race["Results"])
            offset += page_size
        races = [races_by_round[round_no] for round_no in sorted(races_by_round)]
        for race in races:
            round_no = int(race["round"])
            circuit = race["Circuit"]
            location = circuit.get("Location", {})
            event_rows.append(
                {
                    "season": season,
                    "round": round_no,
                    "event": race["raceName"],
                    "date": race["date"],
                    "race_time": race.get("time"),
                    "circuit": circuit["circuitName"],
                    "circuit_id": circuit["circuitId"],
                    "latitude": float(location["lat"]),
                    "longitude": float(location["long"]),
                    "source": "jolpica",
                    "source_record_id": f"{season}-{round_no}",
                    "ingested_at": datetime.now(UTC),
                    "available_at": race["date"],
                    "data_quality": "official_structure",
                }
            )
            for item in race["Results"]:
                result_rows.append(
                    {
                        "season": season,
                        "round": round_no,
                        "event": race["raceName"],
                        "circuit_id": circuit["circuitId"],
                        "driver_id": item["Driver"]["driverId"],
                        "driver": f'{item["Driver"]["givenName"]} {item["Driver"]["familyName"]}',
                        "constructor": item["Constructor"]["name"],
                        "grid": int(item["grid"]),
                        "position": int(item["position"]),
                        "position_text": item.get("positionText"),
                        "points": float(item["points"]),
                        "laps": int(item.get("laps", 0)),
                        "status": item["status"],
                        "winner": int(item["position"] == "1"),
                        "source": "jolpica",
                        "source_record_id": f'{season}-{round_no}-{item["Driver"]["driverId"]}',
                        "ingested_at": datetime.now(UTC),
                        "available_at": race["date"],
                        "data_quality": "official_classification",
                    }
                )
        result_frame = pd.DataFrame(result_rows).sort_values(
            ["season", "round", "driver_id", "position"]
        )
        result_frame["shared_drive_entries"] = result_frame.groupby(
            ["season", "round", "driver_id"]
        )["driver_id"].transform("size")
        result_frame = result_frame.drop_duplicates(
            ["season", "round", "driver_id"], keep="first"
        )
        winner_counts = result_frame.groupby(["season", "round"])["winner"].transform("sum")
        result_frame["winner"] = np.where(
            result_frame["winner"].eq(1),
            1.0 / winner_counts.clip(lower=1),
            0.0,
        )
        return pd.DataFrame(event_rows), result_frame.reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std and np.isfinite(std) else series * 0


def cross_fitted_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Expanding all-season priors; a race never enters its own capability features."""
    data = results.copy()
    if "season" not in data:
        data["season"] = 0
    data = data.sort_values(["season", "round", "position"])
    race_keys = data[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    race_keys["race_index"] = np.arange(len(race_keys))
    data = data.merge(race_keys, on=["season", "round"], how="left")
    data["regulation_era"] = data["season"].map(lambda value: regulation_era(int(value)))
    data["status_class"] = data["status"].map(classify_outcome).astype(str)
    identity = "driver" if "driver" in data else "driver_id"
    field_sizes = data.groupby(["season", "round"])[identity].transform("size")
    data["performance"] = [
        observed_driver_performance(position, int(size), status_class)
        for position, size, status_class in zip(
            data["position"], field_sizes, data["status_class"], strict=True
        )
    ]
    scored = []
    for race_index in sorted(data["race_index"].unique()):
        current = data[data["race_index"] == race_index].copy()
        history = data[data["race_index"] < race_index]
        if history.empty:
            current["car_raw"] = -current["grid"].astype(float)
            current["driver_raw"] = 0.0
            current["team_raw"] = 0.0
        else:
            current_era = str(current["regulation_era"].iloc[0])
            era_history = history[history["regulation_era"].eq(current_era)]
            model_history = era_history if not era_history.empty else history
            global_prior = float(model_history["performance"].mean())
            car_means = model_history.groupby("constructor")["performance"].agg(["mean", "count"])
            car_shrunk = (car_means["mean"] * car_means["count"] + global_prior * 4) / (car_means["count"] + 4)
            current_season = int(current["season"].iloc[0])
            recent_history = model_history[model_history["season"].ge(current_season - 5)]
            driver_means = recent_history.assign(car_mean=recent_history["constructor"].map(car_shrunk).fillna(global_prior), residual=lambda x: x.performance - x.car_mean).groupby("driver_id")["residual"].agg(["mean", "count"])
            driver_shrunk = driver_means["mean"] * driver_means["count"] / (driver_means["count"] + 5)
            reliability = model_history.assign(finished=model_history["status_class"].eq("finished").astype(float)).groupby("constructor")["finished"].mean()
            current["car_raw"] = current["constructor"].map(car_shrunk).fillna(global_prior)
            current["driver_raw"] = current["driver_id"].map(driver_shrunk).fillna(0.0)
            reliability_prior = float(reliability.mean()) if not reliability.empty else 0.5
            current["team_raw"] = current["constructor"].map(reliability).fillna(reliability_prior)
        current["XdW_score"] = _zscore(current["driver_raw"])
        current["XcW_score"] = _zscore(current["car_raw"])
        current["XtW_score"] = _zscore(current["team_raw"])
        scored.append(current)
    return pd.concat(scored, ignore_index=True)


def _context(row: pd.Series, weather: WeatherObservation) -> HistoricalContext:
    return HistoricalContext(
        season=int(row.season),
        event=str(row.event),
        regulation_era=regulation_era(int(row.season)),
        race_distance_laps=int(row.race_distance_laps),
        wet_fraction=weather.wet_fraction,
        track_temperature=weather.track_temperature,
        air_temperature=weather.air_temperature,
        wind_speed=weather.wind_speed,
        circuit_features={"overtake_index": float(row.overtake_index)},
        circuit_id=str(row.circuit_id),
        weather_source=weather.source,
    )


def _neutral_weather(source: str = "weather_disabled_neutral") -> WeatherObservation:
    return WeatherObservation(0.0, 20.0, 20.0, 0.0, source)


def _event_weather(
    client: HistoricalWeatherClient,
    event: Any,
) -> WeatherObservation:
    try:
        return client.event_weather(
            float(event.latitude),
            float(event.longitude),
            str(event.date),
            event.race_time if pd.notna(event.race_time) else None,
        )
    except (requests.RequestException, KeyError, ValueError):
        return _neutral_weather("weather_reanalysis_unavailable_neutral")


def analyze_history(
    from_season: int,
    to_season: int,
    root: Path,
    simulations: int = 500,
    resume: bool = True,
    include_weather: bool = True,
) -> pd.DataFrame:
    """Run one cross-fitted timeline across any completed season range."""

    if from_season > to_season:
        raise ValueError("from_season cannot be after to_season")
    for directory in (root, root / "outputs", root / "processed"):
        directory.mkdir(parents=True, exist_ok=True)
    client = JolpicaClient(root / "cache" / "jolpica")
    event_frames, result_frames = [], []
    for season in range(from_season, to_season + 1):
        season_events, season_results = client.season_results(season)
        event_frames.append(season_events)
        result_frames.append(season_results)
    events = pd.concat(event_frames, ignore_index=True)
    results = pd.concat(result_frames, ignore_index=True)
    distances = (
        results.groupby(["season", "round"], as_index=False)["laps"]
        .max()
        .rename(columns={"laps": "race_distance_laps"})
    )
    events = events.merge(distances, on=["season", "round"], how="left")
    events = events.merge(
        circuit_passability(events, results),
        on=["season", "round", "circuit_id"],
        how="left",
    )
    scored = cross_fitted_scores(results)
    weather_client = HistoricalWeatherClient(root / "cache" / "weather")
    weather_by_event: dict[tuple[int, int], WeatherObservation] = {}
    if include_weather:
        event_rows = list(events.itertuples(index=False))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_event_weather, weather_client, event): (
                    int(event.season),
                    int(event.round),
                )
                for event in event_rows
            }
            for future in as_completed(futures):
                weather_by_event[futures[future]] = future.result()
    all_outputs: list[pd.DataFrame] = []

    for season in range(from_season, to_season + 1):
        output_path = root / "outputs" / f"race_xw_{season}.parquet"
        previous = pd.DataFrame()
        completed: set[int] = set()
        if resume and output_path.exists():
            candidate = pd.read_parquet(output_path)
            if {"XP", "observed_performance", "status_class"}.issubset(candidate.columns):
                previous = candidate
                completed = set(previous["round"].unique())
        rows: list[dict[str, Any]] = []
        season_scores = scored[scored["season"].eq(season)]
        for rnd, race in season_scores.groupby("round", sort=True):
            if rnd in completed:
                continue
            event = events.loc[
                events["season"].eq(season) & events["round"].eq(rnd)
            ].iloc[0]
            weather = _neutral_weather()
            if include_weather:
                weather = weather_by_event[(season, int(rnd))]
            context = _context(event, weather)
            entries = [
                Entry(
                    r.driver,
                    r.constructor,
                    r.constructor,
                    int(r.grid or 20),
                    float(r.XdW_score),
                    float(r.XcW_score),
                    float(r.XtW_score),
                    0.35,
                    0.30,
                    0.30,
                    driver_id=str(r.driver_id),
                )
                for r in race.itertuples()
            ]
            decomposed = decompose_event(
                entries,
                context,
                SimulationConfig(simulations=simulations, seed=season * 100 + int(rnd)),
            )
            race_results = results.loc[
                results["season"].eq(season) & results["round"].eq(rnd)
            ].set_index("driver_id")
            field_size = len(race_results)
            for item in decomposed:
                result = race_results.loc[item["driver_id"]]
                status_class = classify_outcome(result.status)
                item.update(
                    {
                        "round": int(rnd),
                        "date": event.date,
                        "source": "jolpica_cross_fitted_xw_xp",
                        "observed_winner": float(result.winner),
                        "position": int(result.position),
                        "status": result.status,
                        "status_class": status_class.value,
                        "observed_performance": observed_driver_performance(
                            int(result.position), field_size, status_class
                        ),
                        "positions_gained": (
                            float(result.grid - result.position) if result.grid > 0 else np.nan
                        ),
                        "overtake_proxy_raw": float(event.overtake_proxy_raw),
                        "circuit_history_races": int(event.circuit_history_races),
                        "overtake_source": event.overtake_source,
                    }
                )
            rows.extend(decomposed)
        new = pd.DataFrame(rows)
        final = pd.concat([previous, new], ignore_index=True) if not previous.empty else new
        final = final.sort_values(["season", "round", "position"]).reset_index(drop=True)
        final.to_parquet(output_path, index=False)
        final.to_csv(output_path.with_suffix(".csv"), index=False)
        build_report(root, season, final)
        all_outputs.append(final)

    combined = pd.concat(all_outputs, ignore_index=True)
    label = f"{from_season}_{to_season}"
    scored.to_parquet(root / "processed" / f"component_scores_{label}.parquet", index=False)
    events.to_parquet(root / "processed" / f"events_{label}.parquet", index=False)
    results.to_parquet(root / "processed" / f"results_{label}.parquet", index=False)
    build_database(root, events, results, scored, combined)
    return combined


def analyze_season(
    season: int,
    root: Path,
    simulations: int = 2000,
    resume: bool = True,
    include_weather: bool = True,
) -> pd.DataFrame:
    return analyze_history(season, season, root, simulations, resume, include_weather)


def ingest_fastf1_pilot(root: Path, season: int = 2024, event: str = "British Grand Prix") -> dict[str, int]:
    """Download and persist real pilot timing, weather, race control and sampled telemetry."""
    import fastf1

    cache = root / "cache" / "fastf1"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    session = fastf1.get_session(season, event, "R")
    session.load(telemetry=True, weather=True, messages=True)
    raw = root / "raw" / "fastf1" / str(season) / event.replace(" ", "_")
    raw.mkdir(parents=True, exist_ok=True)
    fastf1_laps = session.laps
    laps = pd.DataFrame(fastf1_laps)
    weather = pd.DataFrame(session.weather_data)
    race_control = pd.DataFrame(session.race_control_messages)
    session_results = pd.DataFrame(session.results)
    laps.to_parquet(raw / "laps.parquet", index=False)
    weather.to_parquet(raw / "weather.parquet", index=False)
    race_control.to_parquet(raw / "race_control.parquet", index=False)
    session_results.to_parquet(raw / "session_results.parquet", index=False)
    telemetry_parts = []
    for driver in fastf1_laps["Driver"].dropna().unique():
        driver_laps = fastf1_laps.pick_drivers(driver).pick_wo_box()
        lap = driver_laps.pick_fastest()
        if lap is None or pd.isna(lap.get("LapTime")):
            continue
        telemetry = lap.get_telemetry().copy()
        telemetry["Driver"] = driver
        telemetry["LapNumber"] = lap["LapNumber"]
        telemetry_parts.append(telemetry)
    telemetry = pd.concat(telemetry_parts, ignore_index=True)
    telemetry.to_parquet(raw / "telemetry_clean_reference_laps.parquet", index=False)
    return {"laps": len(laps), "weather": len(weather), "race_control": len(race_control), "telemetry": len(telemetry)}


TABLES = ["events", "sessions", "drivers", "constructors", "results", "qualifying_results", "laps", "telemetry", "weather", "race_control", "stints", "pit_stops", "track_status", "circuit_points", "microsegments", "corners", "straights", "driver_segment_features", "car_segment_features", "team_operational_features", "driver_race_features", "car_race_features", "team_race_features", "component_scores", "counterfactual_probabilities", "race_xw", "season_summaries", "data_quality", "source_reconciliation"]


def build_database(root: Path, events: pd.DataFrame, results: pd.DataFrame, scores: pd.DataFrame, xw: pd.DataFrame) -> None:
    db = root / "processed" / "historical_xw.duckdb"
    with duckdb.connect(str(db)) as con:
        for name, frame in (("events", events), ("results", results), ("component_scores", scores), ("race_xw", xw)):
            con.register("frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")
        for name in TABLES:
            if name not in {"events", "results", "component_scores", "race_xw"}:
                con.execute(f"CREATE TABLE IF NOT EXISTS {name} (source VARCHAR, source_record_id VARCHAR, ingested_at TIMESTAMP, available_at TIMESTAMP, data_quality VARCHAR)")


def build_report(root: Path, season: int, frame: pd.DataFrame) -> None:
    summary = frame.groupby("driver", as_index=False).agg(XW=("XW", "sum"), XP=("XP", "mean"), observed_performance=("observed_performance", "mean"), XdW=("XdW_probability_contribution", "sum"), XcW=("XcW_probability_contribution", "sum"), XtW=("XtW_probability_contribution", "sum")).sort_values("XW", ascending=False)
    template = Template("""<!doctype html><meta charset=utf-8><title>Historical XW {{season}}</title><style>body{font:15px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}</style><h1>Historical XW — {{season}}</h1><p>Real Jolpica classifications; cross-fitted lagged capability priors. Target-race outcomes are excluded from their own features.</p>{{table}}""")
    (root / "outputs" / f"season_report_{season}.html").write_text(template.render(season=season, table=summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")), encoding="utf-8")
