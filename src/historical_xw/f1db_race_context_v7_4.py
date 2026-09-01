from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_f1db_race_context_v7_4(
    root: str | Path,
    *,
    from_season: int = 1950,
    to_season: int = 2025,
    full_weight_completion_fraction: float = 0.75,
) -> pd.DataFrame:
    """Load scheduled/actual race distance and a shortened-race reliability."""

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
            if not race_path.exists():
                continue
            race = yaml.load(race_path.read_text(encoding="utf-8"), Loader=loader) or {}
            actual_laps = pd.to_numeric(race.get("laps"), errors="coerce")
            scheduled_laps = pd.to_numeric(race.get("scheduledLaps"), errors="coerce")
            if (
                np.isfinite(actual_laps)
                and np.isfinite(scheduled_laps)
                and float(scheduled_laps) > 0.0
            ):
                completion = float(np.clip(actual_laps / scheduled_laps, 0.0, 1.0))
                reliability = float(
                    np.clip(completion / full_weight_completion_fraction, 0.0, 1.0) ** 2
                )
                quality = "exact_scheduled_and_actual_laps"
            else:
                completion = np.nan
                reliability = 1.0
                quality = "scheduled_distance_unavailable_assume_full"
            rows.append(
                {
                    "season": season,
                    "round": int(race["round"]),
                    "actual_race_laps_f1db": actual_laps,
                    "scheduled_race_laps_f1db": scheduled_laps,
                    "race_completion_fraction_v7_4": completion,
                    "race_completion_reliability_v7_4": reliability,
                    "race_completion_quality_v7_4": quality,
                }
            )
    result = pd.DataFrame(rows).sort_values(["season", "round"], kind="stable")
    if result.duplicated(["season", "round"]).any():
        raise ValueError("Race context must be unique by season and round")
    return result.reset_index(drop=True)
