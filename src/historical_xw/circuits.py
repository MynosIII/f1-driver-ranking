from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruct_microsegments(telemetry: pd.DataFrame, step_metres: float = 5.0) -> pd.DataFrame:
    """Interpolate multiple clean laps to a canonical distance-based trajectory."""
    required = {"lap", "distance", "x", "y", "z", "speed", "throttle", "brake"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(f"Missing telemetry columns: {sorted(missing)}")
    max_distance = telemetry.groupby("lap")["distance"].max().median()
    distance = np.arange(0.0, max_distance, step_metres)
    laps = []
    for _, lap in telemetry.sort_values("distance").groupby("lap"):
        laps.append({col: np.interp(distance, lap["distance"], lap[col]) for col in ("x", "y", "z", "speed", "throttle", "brake")})
    canonical = {col: np.median([lap[col] for lap in laps], axis=0) for col in laps[0]}
    x, y, z = canonical["x"], canonical["y"], canonical["z"]
    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    curvature = np.gradient(heading, distance, edge_order=1)
    radius = np.divide(1.0, np.abs(curvature), out=np.full_like(curvature, np.inf), where=np.abs(curvature) > 1e-6)
    result = pd.DataFrame({"distance": distance, **canonical})
    result["heading"] = heading
    result["gradient"] = np.gradient(z, distance, edge_order=1)
    result["curvature"] = curvature
    result["radius"] = radius
    result["direction"] = np.where(curvature > 0.001, "left", np.where(curvature < -0.001, "right", "straight"))
    result["segment_class"] = np.where(radius > 500, "straight", np.where(canonical["speed"] < 140, "slow", np.where(canonical["speed"] < 220, "medium", "fast")))
    result["braking_probability"] = canonical["brake"]
    result["throttle_probability"] = canonical["throttle"] / 100.0
    result["data_quality"] = "canonical_median"
    return result

