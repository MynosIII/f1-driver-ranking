from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import requests
from jinja2 import Template
from tenacity import retry, stop_after_attempt, wait_exponential

from .decomposition import decompose_event, decompose_event_v8
from .domain import Entry, HistoricalContext, SimulationConfig

# A full 1950-2025 run makes several hundred requests to the same host in
# one long-running process. A companion project (F1Predictor) hit a
# Cloudflare-level rate limit under exactly that load that did not clear
# for several minutes -- not the seconds a simple burst limit would
# suggest -- so this is deliberately more patient (stop_after_attempt=20,
# up to 180s backoff) than the 5-attempt/20s-cap version this project
# shipped with before that was known. A (connect, read) timeout tuple
# replaces the bare 90, since a single float only reliably bounds the read
# phase on some platforms -- the same fix applied to F1Predictor after it
# separately lost a run to a connection-phase stall no timeout tripped on.
_JOLPICA_MIN_INTERVAL_S = 0.4
_last_request_at = 0.0


@retry(stop=stop_after_attempt(20), wait=wait_exponential(min=2, max=180), reraise=True)
def _get_json(url: str) -> dict[str, Any]:
    global _last_request_at
    wait = _JOLPICA_MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()
    response = requests.get(url, timeout=(10, 90))
    response.raise_for_status()
    return response.json()


class JolpicaClient:
    base = "https://api.jolpi.ca/ergast/f1"

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, path: str, limit: int = 2000) -> dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}?limit={limit}"
        key = hashlib.sha256(url.encode()).hexdigest()[:20]
        target = self.cache / f"{path.replace('/', '_')}_{key}.json"
        if not target.exists():
            target.write_text(json.dumps(_get_json(url)), encoding="utf-8")
        return json.loads(target.read_text(encoding="utf-8"))

    def season_results(self, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        schedule = self.get(f"{season}.json")["MRData"]["RaceTable"]["Races"]
        result_rows, event_rows = [], []
        for event in schedule:
            round_no = int(event["round"])
            races = self.get(f"{season}/{round_no}/results.json")["MRData"]["RaceTable"]["Races"]
            if not races:
                continue
            race = races[0]
            event_rows.append({"season": season, "round": round_no, "event": race["raceName"], "date": race["date"], "circuit": race["Circuit"]["circuitName"], "source": "jolpica", "source_record_id": f"{season}-{round_no}", "ingested_at": datetime.now(UTC), "available_at": race["date"], "data_quality": "official_structure"})
            for item in race["Results"]:
                result_rows.append({"season": season, "round": round_no, "event": race["raceName"], "driver_id": item["Driver"]["driverId"], "driver": f'{item["Driver"]["givenName"]} {item["Driver"]["familyName"]}', "constructor": item["Constructor"]["name"], "grid": int(item["grid"]), "position": int(item["position"]), "points": float(item["points"]), "status": item["status"], "winner": int(item["position"] == "1"), "source": "jolpica", "source_record_id": f'{season}-{round_no}-{item["Driver"]["driverId"]}', "ingested_at": datetime.now(UTC), "available_at": race["date"], "data_quality": "official_classification"})
        return pd.DataFrame(event_rows), pd.DataFrame(result_rows)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std and np.isfinite(std) else series * 0


def cross_fitted_scores(results: pd.DataFrame) -> pd.DataFrame:
    """Lagged expanding priors: target-race finish/winner never enters its own features."""
    data = results.sort_values(["round", "position"]).copy()
    data["performance"] = -data["position"].astype(float)
    global_prior = data["performance"].mean()
    scored = []
    for rnd in sorted(data["round"].unique()):
        current = data[data["round"] == rnd].copy()
        history = data[data["round"] < rnd]
        if history.empty:
            current["car_raw"] = -current["grid"].astype(float)
            current["driver_raw"] = 0.0
            current["team_raw"] = 0.0
        else:
            car_means = history.groupby("constructor")["performance"].agg(["mean", "count"])
            car_shrunk = (car_means["mean"] * car_means["count"] + global_prior * 4) / (car_means["count"] + 4)
            driver_means = history.assign(car_mean=history["constructor"].map(car_shrunk).fillna(global_prior), residual=lambda x: x.performance - x.car_mean).groupby("driver_id")["residual"].agg(["mean", "count"])
            driver_shrunk = driver_means["mean"] * driver_means["count"] / (driver_means["count"] + 5)
            reliability = history.assign(finished=history["status"].str.contains("Finished|Lap", regex=True).astype(float)).groupby("constructor")["finished"].mean()
            current["car_raw"] = current["constructor"].map(car_shrunk).fillna(global_prior)
            current["driver_raw"] = current["driver_id"].map(driver_shrunk).fillna(0.0)
            current["team_raw"] = current["constructor"].map(reliability).fillna(reliability.mean())
        current["XdW_score"] = _zscore(current["driver_raw"])
        current["XcW_score"] = _zscore(current["car_raw"])
        current["XtW_score"] = _zscore(current["team_raw"])
        scored.append(current)
    return pd.concat(scored, ignore_index=True)


def _context(row: pd.Series) -> HistoricalContext:
    return HistoricalContext(int(row.season), str(row.event), "2022-2025", 52 if "British" in row.event else 60, 0.0, 30.0, 20.0, 3.0)


def analyze_season(season: int, root: Path, simulations: int = 2000, resume: bool = True) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    client = JolpicaClient(root / "cache" / "jolpica")
    events, results = client.season_results(season)
    scored = cross_fitted_scores(results)
    output_path = root / "outputs" / f"race_xw_{season}.parquet"
    completed: set[int] = set()
    previous = pd.DataFrame()
    if resume and output_path.exists():
        previous = pd.read_parquet(output_path)
        completed = set(previous["round"].unique())
    rows = []
    for rnd, race in scored.groupby("round", sort=True):
        if rnd in completed:
            continue
        entries = [Entry(r.driver, r.constructor, r.constructor, int(r.grid or 20), float(r.XdW_score), float(r.XcW_score), float(r.XtW_score), .35, .30, .30) for r in race.itertuples()]
        decomposed = decompose_event(entries, _context(race.iloc[0]), SimulationConfig(simulations=simulations, seed=season * 100 + int(rnd)))
        winner_by_driver = results.loc[results["round"] == rnd].set_index("driver")["winner"].to_dict()
        for item in decomposed:
            item["round"] = int(rnd)
            item["source"] = "jolpica_cross_fitted"
            item["observed_winner"] = int(winner_by_driver[item["driver"]])
        rows.extend(decomposed)
    final = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True) if not previous.empty else pd.DataFrame(rows)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False)
    final.to_csv(output_path.with_suffix(".csv"), index=False)
    scored.to_parquet(root / "processed" / f"component_scores_{season}.parquet", index=False)
    events.to_parquet(root / "processed" / f"events_{season}.parquet", index=False)
    results.to_parquet(root / "processed" / f"results_{season}.parquet", index=False)
    build_database(root, events, results, scored, final)
    build_report(root, season, final)
    return final


def analyze_season_v8(season: int, root: Path, simulations: int = 2000, resume: bool = True) -> pd.DataFrame:
    """Position-based (XP), fault-aware analogue of ``analyze_season``.
    Uses the same real Jolpica results and the same leakage-safe cross-fitted
    driver/car/team priors -- only the decomposition target changes (XP
    instead of XW), and ``status``/``position`` are carried through so
    ``ranking_v8.calculate_rating_history_v8`` can classify DNF fault.
    """
    root.mkdir(parents=True, exist_ok=True)
    client = JolpicaClient(root / "cache" / "jolpica")
    events, results = client.season_results(season)
    scored = cross_fitted_scores(results)
    output_path = root / "outputs" / f"race_xp_{season}.parquet"
    completed: set[int] = set()
    previous = pd.DataFrame()
    if resume and output_path.exists():
        previous = pd.read_parquet(output_path)
        completed = set(previous["round"].unique())
    rows = []
    for rnd, race in scored.groupby("round", sort=True):
        if rnd in completed:
            continue
        entries = [Entry(r.driver, r.constructor, r.constructor, int(r.grid or 20), float(r.XdW_score), float(r.XcW_score), float(r.XtW_score), .35, .30, .30) for r in race.itertuples()]
        decomposed = decompose_event_v8(entries, _context(race.iloc[0]), SimulationConfig(simulations=simulations, seed=season * 100 + int(rnd)))
        result_by_driver = results.loc[results["round"] == rnd].set_index("driver")
        car_score_by_driver = race.set_index("driver")["XcW_score"].to_dict()
        for item in decomposed:
            item["round"] = int(rnd)
            item["source"] = "jolpica_cross_fitted"
            item["position"] = int(result_by_driver.loc[item["driver"], "position"])
            item["status"] = str(result_by_driver.loc[item["driver"], "status"])
            item["XcW_score"] = car_score_by_driver[item["driver"]]
        rows.extend(decomposed)
    final = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True) if not previous.empty else pd.DataFrame(rows)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False)
    final.to_csv(output_path.with_suffix(".csv"), index=False)
    return final


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
    summary = frame.groupby("driver", as_index=False).agg(XW=("XW", "sum"), XdW=("XdW_probability_contribution", "sum"), XcW=("XcW_probability_contribution", "sum"), XtW=("XtW_probability_contribution", "sum")).sort_values("XW", ascending=False)
    template = Template("""<!doctype html><meta charset=utf-8><title>Historical XW {{season}}</title><style>body{font:15px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}</style><h1>Historical XW — {{season}}</h1><p>Real Jolpica classifications; cross-fitted lagged capability priors. Target-race outcomes are excluded from their own features.</p>{{table}}""")
    (root / "outputs" / f"season_report_{season}.html").write_text(template.render(season=season, table=summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")), encoding="utf-8")
