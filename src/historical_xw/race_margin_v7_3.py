from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml


_LAP_GAP = re.compile(r"^\+?\s*(\d+)\s+laps?\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RaceMarginConfigV73:
    maximum_performance_adjustment: float = 0.04
    gap_share_floor: float = 1e-4
    log_ratio_scale: float = float(np.log(4.0))
    season_decay_years: float = 12.0
    minimum_xw_gap_slope: float = 0.05
    maximum_xw_gap_slope: float = 0.50
    minimum_effective_training_races: float = 25.0
    lap_equivalent_reliability: float = 0.70
    minimum_expected_gap_share: float = 1e-4
    maximum_expected_gap_share: float = 0.10


def parse_duration_seconds_v7_3(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip().lower().lstrip("+").strip()
    if not text or "lap" in text:
        return float("nan")
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return float("nan")
    if not 1 <= len(parts) <= 3:
        return float("nan")
    return float(sum(part * 60**power for power, part in enumerate(reversed(parts))))


def parse_lap_gap_v7_3(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    match = _LAP_GAP.match(str(value).strip())
    return float(match.group(1)) if match else float("nan")


def load_f1db_race_timing_v7_3(
    root: str | Path,
    *,
    from_season: int = 1950,
    to_season: int = 2025,
) -> pd.DataFrame:
    """Load F1DB race time, cumulative gap, and interval fields without changing v6."""

    source_root = Path(root)
    seasons_root = source_root / "src" / "data" / "seasons"
    if not seasons_root.exists():
        raise FileNotFoundError(f"Not an F1DB source checkout: {source_root}")
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    rows: list[dict[str, Any]] = []
    for season in range(from_season, to_season + 1):
        races_root = seasons_root / str(season) / "races"
        if not races_root.exists():
            continue
        for race_root in sorted(races_root.iterdir()):
            race_path = race_root / "race.yml"
            results_path = race_root / "race-results.yml"
            if not race_path.exists() or not results_path.exists():
                continue
            race = yaml.load(race_path.read_text(encoding="utf-8"), Loader=loader) or {}
            results = (
                yaml.load(results_path.read_text(encoding="utf-8"), Loader=loader) or []
            )
            round_number = int(race["round"])
            for result_order, raw in enumerate(results, start=1):
                raw_time = raw.get("time")
                raw_gap = raw.get("gap")
                raw_interval = raw.get("interval")
                gap_seconds = parse_duration_seconds_v7_3(raw_gap)
                gap_laps = parse_lap_gap_v7_3(raw_gap)
                interval_seconds = parse_duration_seconds_v7_3(raw_interval)
                interval_laps = parse_lap_gap_v7_3(raw_interval)
                if result_order == 1 and np.isfinite(
                    parse_duration_seconds_v7_3(raw_time)
                ):
                    timing_class = "winner_absolute_time"
                elif np.isfinite(gap_seconds):
                    timing_class = "same_lap_seconds"
                elif np.isfinite(gap_laps):
                    timing_class = "laps_down"
                else:
                    timing_class = "unavailable"
                rows.append(
                    {
                        "season": season,
                        "round": round_number,
                        "f1db_race_id": int(race["id"]),
                        "result_order": result_order,
                        "driver_id_f1db": str(raw["driverId"]),
                        "constructor_id_f1db": str(raw["constructorId"]),
                        "finish_position_text": str(raw.get("position", "")),
                        "laps": pd.to_numeric(raw.get("laps"), errors="coerce"),
                        "race_time": raw_time,
                        "race_time_seconds": parse_duration_seconds_v7_3(raw_time),
                        "gap": raw_gap,
                        "gap_seconds": gap_seconds,
                        "gap_laps": gap_laps,
                        "interval": raw_interval,
                        "interval_seconds": interval_seconds,
                        "interval_laps": interval_laps,
                        "timing_class": timing_class,
                    }
                )
    timing = pd.DataFrame(rows).sort_values(
        ["season", "round", "result_order"], kind="stable"
    )
    if timing.duplicated(["season", "round", "result_order"]).any():
        raise ValueError("F1DB timing must be unique at classified result-entry grain")
    return timing.reset_index(drop=True)


def _logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-5, 1.0 - 1e-5))
    return float(np.log(clipped / (1.0 - clipped)))


def build_winner_margin_features_v7_3(
    history: pd.DataFrame,
    timing: pd.DataFrame,
    config: RaceMarginConfigV73 | None = None,
) -> pd.DataFrame:
    """Estimate expected winner-to-second gap from their pre-race xW difference.

    Expected gaps use a leave-season-out, locally era-weighted regression of
    normalized historical winning margins on pairwise xW. The positive bounded
    slope guarantees that a larger xW advantage implies a larger expected gap.
    Exact second gaps and lap-equivalent gaps are kept distinct.
    """

    cfg = config or RaceMarginConfigV73()
    required_history = {
        "season",
        "round",
        "driver_id_f1db",
        "XW",
        "status_class",
    }
    if missing := required_history - set(history.columns):
        raise ValueError(f"History is missing margin columns: {sorted(missing)}")
    required_timing = {
        "season",
        "round",
        "result_order",
        "driver_id_f1db",
        "laps",
        "race_time_seconds",
        "gap_seconds",
        "gap_laps",
    }
    if missing := required_timing - set(timing.columns):
        raise ValueError(f"Timing is missing columns: {sorted(missing)}")

    history_lookup = history.copy()
    history_lookup["driver_id_f1db"] = history_lookup["driver_id_f1db"].astype(str)
    history_lookup["XW"] = pd.to_numeric(history_lookup["XW"], errors="coerce")
    lookup = history_lookup.set_index(["season", "round", "driver_id_f1db"])
    race_rows: list[dict[str, Any]] = []
    for (season, round_number), race_timing in timing.groupby(
        ["season", "round"], sort=True
    ):
        ordered = race_timing.sort_values("result_order", kind="stable")
        if len(ordered) < 2:
            continue
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1]
        winner_id = str(winner["driver_id_f1db"])
        runner_up_id = str(runner_up["driver_id_f1db"])
        winner_key = (int(season), int(round_number), winner_id)
        runner_up_key = (int(season), int(round_number), runner_up_id)
        winner_xw = float(lookup.loc[winner_key, "XW"]) if winner_key in lookup.index else np.nan
        runner_up_xw = (
            float(lookup.loc[runner_up_key, "XW"]) if runner_up_key in lookup.index else np.nan
        )
        denominator = winner_xw + runner_up_xw
        pairwise_xw = (
            winner_xw / denominator
            if np.isfinite(denominator) and denominator > 0.0
            else np.nan
        )
        winner_time = float(winner["race_time_seconds"])
        winner_laps = float(winner["laps"])
        exact_gap = float(runner_up["gap_seconds"])
        lap_gap = float(runner_up["gap_laps"])
        if np.isfinite(exact_gap) and np.isfinite(winner_time) and winner_time > 0.0:
            observed_gap_seconds = exact_gap
            observed_gap_share = exact_gap / winner_time
            timing_quality = "exact_seconds"
            timing_reliability = 1.0
        elif (
            np.isfinite(lap_gap)
            and np.isfinite(winner_time)
            and winner_time > 0.0
            and np.isfinite(winner_laps)
            and winner_laps > 0.0
        ):
            observed_gap_share = lap_gap / winner_laps
            observed_gap_seconds = observed_gap_share * winner_time
            timing_quality = "lap_equivalent"
            timing_reliability = cfg.lap_equivalent_reliability
        else:
            observed_gap_seconds = np.nan
            observed_gap_share = np.nan
            timing_quality = "unavailable"
            timing_reliability = 0.0
        winner_finished = (
            winner_key in lookup.index
            and str(lookup.loc[winner_key, "status_class"]) == "finished"
        )
        runner_up_finished = (
            runner_up_key in lookup.index
            and str(lookup.loc[runner_up_key, "status_class"]) == "finished"
        )
        race_rows.append(
            {
                "season": int(season),
                "round": int(round_number),
                "winner_driver_id_f1db": winner_id,
                "runner_up_driver_id_f1db": runner_up_id,
                "winner_xw": winner_xw,
                "runner_up_xw": runner_up_xw,
                "winner_xw_advantage": winner_xw - runner_up_xw,
                "winner_pairwise_xw": pairwise_xw,
                "winner_xw_logit_advantage": _logit(pairwise_xw)
                if np.isfinite(pairwise_xw)
                else np.nan,
                "winner_race_time_seconds": winner_time,
                "observed_winner_gap_seconds": observed_gap_seconds,
                "observed_winner_gap_share": observed_gap_share,
                "winner_gap_timing_quality": timing_quality,
                "winner_gap_timing_reliability": timing_reliability,
                "winner_and_runner_up_finished": winner_finished and runner_up_finished,
            }
        )
    races = pd.DataFrame(race_rows)
    training = races[
        races["winner_and_runner_up_finished"]
        & races["observed_winner_gap_share"].notna()
        & races["winner_xw_logit_advantage"].notna()
        & races["observed_winner_gap_share"].ge(0.0)
    ].copy()

    expected_shares: list[float] = []
    effective_samples: list[float] = []
    expected_slopes: list[float] = []
    for target in races.itertuples(index=False):
        if not np.isfinite(target.winner_xw_logit_advantage):
            expected_shares.append(np.nan)
            effective_samples.append(0.0)
            expected_slopes.append(np.nan)
            continue
        candidates = training[training["season"].ne(int(target.season))]
        if candidates.empty:
            expected_shares.append(np.nan)
            effective_samples.append(0.0)
            expected_slopes.append(np.nan)
            continue
        season_distance = np.abs(candidates["season"].to_numpy(dtype=float) - target.season)
        weights = np.exp(-season_distance / cfg.season_decay_years)
        weights *= candidates["winner_gap_timing_reliability"].to_numpy(dtype=float)
        x_values = candidates["winner_xw_logit_advantage"].to_numpy(dtype=float)
        y_values = np.log(
            candidates["observed_winner_gap_share"].to_numpy(dtype=float)
            + cfg.gap_share_floor
        )
        x_mean = float(np.average(x_values, weights=weights))
        y_mean = float(np.average(y_values, weights=weights))
        covariance = float(np.sum(weights * (x_values - x_mean) * (y_values - y_mean)))
        variance = float(np.sum(weights * np.square(x_values - x_mean)))
        raw_slope = covariance / max(variance, 1e-12)
        slope = float(
            np.clip(raw_slope, cfg.minimum_xw_gap_slope, cfg.maximum_xw_gap_slope)
        )
        expected = float(
            np.exp(y_mean + slope * (target.winner_xw_logit_advantage - x_mean))
            - cfg.gap_share_floor
        )
        expected_shares.append(
            float(
                np.clip(
                    expected,
                    cfg.minimum_expected_gap_share,
                    cfg.maximum_expected_gap_share,
                )
            )
        )
        effective_samples.append(
            float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12))
        )
        expected_slopes.append(slope)
    races["expected_winner_gap_share_from_xw"] = expected_shares
    races["expected_gap_effective_training_races"] = effective_samples
    races["expected_gap_xw_logit_slope"] = expected_slopes
    races["expected_winner_gap_seconds_from_xw"] = (
        races["expected_winner_gap_share_from_xw"]
        * races["winner_race_time_seconds"]
    )
    actual = races["observed_winner_gap_share"]
    expected = races["expected_winner_gap_share_from_xw"]
    races["winner_gap_log_ratio"] = np.log(
        (actual + cfg.gap_share_floor) / (expected + cfg.gap_share_floor)
    )
    support_reliability = (
        races["expected_gap_effective_training_races"]
        / cfg.minimum_effective_training_races
    ).clip(0.0, 1.0)
    eligible = (
        races["winner_and_runner_up_finished"]
        & actual.notna()
        & expected.notna()
    )
    races["winner_margin_performance_adjustment"] = 0.0
    races.loc[eligible, "winner_margin_performance_adjustment"] = (
        cfg.maximum_performance_adjustment
        * np.tanh(races.loc[eligible, "winner_gap_log_ratio"] / cfg.log_ratio_scale)
        * races.loc[eligible, "winner_gap_timing_reliability"]
        * support_reliability.loc[eligible]
    )
    races["winner_margin_model"] = (
        "leave_season_out_local_era_log_gap_regression_on_pairwise_xw_v7_3"
    )
    return races


def attach_winner_margin_adjustments_v7_3(
    history: pd.DataFrame,
    race_margins: pd.DataFrame,
) -> pd.DataFrame:
    """Add a bounded, race-zero-sum winner/runner-up margin residual."""

    output = history.copy()
    output["observed_performance_position_only"] = output["observed_performance"]
    output["winner_margin_performance_adjustment"] = 0.0
    output["margin_role"] = "none"
    feature_columns = [
        "season",
        "round",
        "winner_xw",
        "runner_up_xw",
        "winner_xw_advantage",
        "winner_pairwise_xw",
        "observed_winner_gap_seconds",
        "observed_winner_gap_share",
        "expected_winner_gap_seconds_from_xw",
        "expected_winner_gap_share_from_xw",
        "expected_gap_effective_training_races",
        "expected_gap_xw_logit_slope",
        "winner_gap_log_ratio",
        "winner_gap_timing_quality",
        "winner_margin_model",
    ]
    output = output.merge(
        race_margins[feature_columns],
        on=["season", "round"],
        how="left",
        validate="many_to_one",
    )
    driver_ids = output["driver_id_f1db"].astype(str)
    winner_lookup = race_margins.set_index(["season", "round"])[
        "winner_driver_id_f1db"
    ]
    runner_up_lookup = race_margins.set_index(["season", "round"])[
        "runner_up_driver_id_f1db"
    ]
    adjustments = race_margins.set_index(["season", "round"])[
        "winner_margin_performance_adjustment"
    ]
    event_keys = pd.MultiIndex.from_frame(output[["season", "round"]])
    winner_ids = winner_lookup.reindex(event_keys).to_numpy(dtype=str)
    runner_up_ids = runner_up_lookup.reindex(event_keys).to_numpy(dtype=str)
    event_adjustments = adjustments.reindex(event_keys).fillna(0.0).to_numpy(dtype=float)
    winner_mask = driver_ids.to_numpy() == winner_ids
    runner_up_mask = driver_ids.to_numpy() == runner_up_ids
    output.loc[winner_mask, "winner_margin_performance_adjustment"] = event_adjustments[
        winner_mask
    ]
    output.loc[runner_up_mask, "winner_margin_performance_adjustment"] = -event_adjustments[
        runner_up_mask
    ]
    output.loc[winner_mask, "margin_role"] = "winner"
    output.loc[runner_up_mask, "margin_role"] = "runner_up"
    output["gap_adjusted_observed_performance"] = (
        output["observed_performance_position_only"]
        + output["winner_margin_performance_adjustment"]
    )
    output["observed_performance"] = output["gap_adjusted_observed_performance"]
    return output
