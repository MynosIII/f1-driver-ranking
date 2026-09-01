from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.signal import find_peaks


ANGLE_BANDS_V9: tuple[tuple[str, float, float], ...] = (
    ("kink_8_30", 8.0, 30.0),
    ("turn_30_60", 30.0, 60.0),
    ("turn_60_90", 60.0, 90.0),
    ("turn_90_135", 90.0, 135.0),
    ("turn_135_180", 135.0, 180.0),
    ("hairpin_180_225", 180.0, 225.0),
    ("extended_over_225", 225.0, float("inf")),
)

RADIUS_BANDS_V9: tuple[tuple[str, float, float], ...] = (
    ("radius_under_50", 0.0, 50.0),
    ("radius_50_90", 50.0, 90.0),
    ("radius_90_150", 90.0, 150.0),
    ("radius_150_250", 150.0, 250.0),
    ("radius_250_450", 250.0, 450.0),
    ("radius_over_450", 450.0, float("inf")),
)

STRAIGHT_BANDS_V9: tuple[tuple[str, float, float], ...] = (
    ("straight_40_150", 40.0, 150.0),
    ("straight_150_300", 150.0, 300.0),
    ("straight_300_600", 300.0, 600.0),
    ("straight_600_1000", 600.0, 1000.0),
    ("straight_over_1000", 1000.0, float("inf")),
)

SHAPE_MODEL_FEATURES_V9: tuple[str, ...] = (
    "top_three_straight_share",
    "straight_600_1000_track_share",
    "straight_over_1000_track_share",
    "radius_under_50_distance_share",
    "radius_50_90_distance_share",
    "radius_90_150_distance_share",
    "radius_150_250_distance_share",
    "radius_250_450_distance_share",
    "steady_long_turn_distance_share",
    "opposite_direction_links_per_km",
    "same_direction_links_per_km",
    "multi_turn_complexes_per_km",
    "multi_apex_turns_per_km",
    "tightening_turns_per_km",
    "approach_severity_per_km",
    "top_three_turning_share",
)


@dataclass(frozen=True, slots=True)
class CircuitShapeConfigV9:
    curvature_smoothing_metres: float = 25.0
    straight_radius_metres: float = 1200.0
    turn_entry_radius_metres: float = 800.0
    minimum_turn_angle_deg: float = 8.0
    minimum_turn_length_metres: float = 12.0
    same_direction_merge_gap_metres: float = 45.0
    minimum_straight_metres: float = 40.0
    minimum_complex_link_metres: float = 60.0
    maximum_complex_link_metres: float = 180.0
    apex_minimum_spacing_metres: float = 25.0
    steady_turn_minimum_angle_deg: float = 60.0
    steady_turn_minimum_length_metres: float = 100.0
    steady_turn_minimum_consistency: float = 0.58


@dataclass(slots=True)
class CircuitShapeResultV9:
    profile: pd.DataFrame
    turns: pd.DataFrame
    complexes: pd.DataFrame
    canonical_microsegments: pd.DataFrame


def _cyclic_runs(mask: np.ndarray) -> list[np.ndarray]:
    values = np.asarray(mask, dtype=bool)
    count = len(values)
    if count == 0 or not values.any():
        return []
    if values.all():
        return [np.arange(count)]
    starts = np.flatnonzero(values & ~np.roll(values, 1))
    output: list[np.ndarray] = []
    for start in starts:
        indices: list[int] = []
        cursor = int(start)
        while values[cursor]:
            indices.append(cursor)
            cursor = (cursor + 1) % count
        output.append(np.asarray(indices, dtype=int))
    return output


def _linear_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _angle_band(angle: float) -> str:
    angle = round(float(angle), 3)
    for name, lower, upper in ANGLE_BANDS_V9:
        if lower <= angle < upper:
            return name
    return "below_detection"


def _radius_band(radius: float) -> str:
    for name, lower, upper in RADIUS_BANDS_V9:
        if lower <= radius < upper:
            return name
    return "radius_unavailable"


def canonicalize_circuit_v9(
    microsegments: pd.DataFrame,
    config: CircuitShapeConfigV9 | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Place the lap seam in its longest low-curvature region and denoise curvature."""

    cfg = config or CircuitShapeConfigV9()
    required = {"distance", "x", "y", "curvature"}
    if missing := required - set(microsegments.columns):
        raise ValueError(f"Shape microsegments are missing columns: {sorted(missing)}")
    data = microsegments.sort_values("distance", kind="stable").reset_index(drop=True).copy()
    if len(data) < 40:
        raise ValueError("At least 40 microsegments are required for shape extraction")
    distances = pd.to_numeric(data["distance"], errors="raise").to_numpy(dtype=float)
    step = float(np.median(np.diff(distances)))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Microsegment spacing must be positive")
    raw = pd.to_numeric(data["curvature"], errors="coerce").to_numpy(dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = max(cfg.curvature_smoothing_metres / step, 0.5)
    smooth = gaussian_filter1d(raw, sigma=sigma, mode="wrap")
    residual = raw - smooth
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    straight_threshold = max(1.0 / cfg.straight_radius_metres, 2.0 * noise)
    low_curvature = np.abs(smooth) < straight_threshold
    runs = _cyclic_runs(low_curvature)
    if runs:
        longest = max(runs, key=len)
        seam = int(longest[len(longest) // 2])
    else:
        seam = int(np.argmin(np.abs(smooth)))
    order = np.r_[np.arange(seam, len(data)), np.arange(0, seam)]
    data = data.iloc[order].reset_index(drop=True)
    smooth = smooth[order]
    raw = raw[order]
    data["source_distance_v9"] = data["distance"].to_numpy(dtype=float)
    data["distance"] = np.arange(len(data), dtype=float) * step
    data["curvature_raw_v9"] = raw
    data["curvature_shape_v9"] = smooth
    data["radius_shape_v9"] = np.divide(
        1.0,
        np.abs(smooth),
        out=np.full_like(smooth, np.inf),
        where=np.abs(smooth) > 1e-9,
    )
    data["direction_shape_v9"] = np.where(smooth >= 0.0, "left", "right")
    return data, {
        "microsegment_step_m": step,
        "curvature_noise_floor": noise,
        "straight_curvature_threshold": straight_threshold,
        "canonical_seam_source_distance_m": float(distances[seam]),
    }


def _candidate_turn_intervals(
    segments: pd.DataFrame,
    metadata: dict[str, float],
    cfg: CircuitShapeConfigV9,
) -> list[tuple[int, int, int]]:
    curvature = segments["curvature_shape_v9"].to_numpy(dtype=float)
    step = metadata["microsegment_step_m"]
    noise = metadata["curvature_noise_floor"]
    entry_threshold = max(1.0 / cfg.turn_entry_radius_metres, 3.0 * noise)
    sustain_threshold = max(metadata["straight_curvature_threshold"], 2.0 * noise)
    gap_points = max(1, int(round(cfg.same_direction_merge_gap_metres / step)))
    intervals: list[tuple[int, int, int]] = []
    for direction in (1, -1):
        signed = direction * curvature
        active = signed >= sustain_threshold
        active = binary_closing(
            active,
            structure=np.ones(gap_points + 1, dtype=bool),
            border_value=0,
        )
        for start, end in _linear_runs(active):
            piece = signed[start:end]
            angle = float(np.degrees(np.clip(piece, 0.0, None).sum() * step))
            length = float((end - start) * step)
            if (
                len(piece)
                and float(piece.max()) >= entry_threshold
                and angle >= cfg.minimum_turn_angle_deg
                and length >= cfg.minimum_turn_length_metres
            ):
                intervals.append((int(start), int(end), direction))
    intervals.sort(key=lambda item: item[0])
    # Resolve rare smoothing overlaps by assigning the shared part to the stronger sign.
    resolved: list[tuple[int, int, int]] = []
    for interval in intervals:
        if resolved and interval[0] < resolved[-1][1]:
            previous = resolved[-1]
            boundary = (previous[1] + interval[0]) // 2
            resolved[-1] = (previous[0], max(previous[0] + 1, boundary), previous[2])
            interval = (max(boundary, interval[0]), interval[1], interval[2])
        if interval[1] > interval[0]:
            resolved.append(interval)
    return resolved


def extract_turns_v9(
    segments: pd.DataFrame,
    metadata: dict[str, float],
    config: CircuitShapeConfigV9 | None = None,
) -> pd.DataFrame:
    cfg = config or CircuitShapeConfigV9()
    step = metadata["microsegment_step_m"]
    intervals = _candidate_turn_intervals(segments, metadata, cfg)
    rows: list[dict[str, Any]] = []
    curvature = segments["curvature_shape_v9"].to_numpy(dtype=float)
    for turn_id, (start, end, direction_sign) in enumerate(intervals, start=1):
        piece = segments.iloc[start:end]
        signed = curvature[start:end]
        absolute = np.abs(signed)
        angle_signed = float(np.degrees(signed.sum() * step))
        angle_absolute = float(np.degrees(absolute.sum() * step))
        length = float((end - start) * step)
        equivalent_radius = length / max(np.radians(angle_absolute), 1e-9)
        peak_index_local = int(np.argmax(absolute))
        peak_index = start + peak_index_local
        peak_curvature = float(absolute[peak_index_local])
        prominence = max(3.0 * metadata["curvature_noise_floor"], 0.12 * peak_curvature)
        peaks, _ = find_peaks(
            absolute,
            prominence=prominence,
            distance=max(1, int(round(cfg.apex_minimum_spacing_metres / step))),
        )
        if not len(peaks):
            peaks = np.asarray([peak_index_local])
        halfway = max(1, len(absolute) // 2)
        first_half = float(absolute[:halfway].mean())
        second_half = float(absolute[halfway:].mean()) if halfway < len(absolute) else first_half
        mean_curvature = float(absolute.mean())
        tightening_index = (second_half - first_half) / max(mean_curvature, 1e-9)
        chord = float(
            np.hypot(
                float(piece["x"].iloc[-1] - piece["x"].iloc[0]),
                float(piece["y"].iloc[-1] - piece["y"].iloc[0]),
            )
        )
        rows.append(
            {
                "turn_id": turn_id,
                "start_index": start,
                "end_index": end,
                "entry_distance_m": float(start * step),
                "exit_distance_m": float(end * step),
                "apex_distance_m": float(peak_index * step),
                "direction": "left" if direction_sign > 0 else "right",
                "direction_sign": direction_sign,
                "turn_length_m": length,
                "chord_length_m": chord,
                "arc_to_chord_ratio": length / max(chord, 1e-9),
                "signed_angle_deg": angle_signed,
                "absolute_angle_deg": angle_absolute,
                "angle_band": _angle_band(angle_absolute),
                "equivalent_radius_m": equivalent_radius,
                "radius_band": _radius_band(equivalent_radius),
                "peak_curvature_per_m": peak_curvature,
                "peak_radius_m": 1.0 / max(peak_curvature, 1e-9),
                "mean_abs_curvature_per_m": mean_curvature,
                "curvature_consistency": mean_curvature / max(peak_curvature, 1e-9),
                "curvature_variation_coefficient": float(absolute.std())
                / max(mean_curvature, 1e-9),
                "apex_count": int(len(peaks)),
                "tightening_index": tightening_index,
                "turn_shape": (
                    "tightening"
                    if tightening_index > 0.20
                    else "opening"
                    if tightening_index < -0.20
                    else "balanced"
                ),
            }
        )
    turns = pd.DataFrame(rows)
    if turns.empty:
        return turns
    track_length = float(len(segments) * step)
    transition: list[float] = []
    straight_between: list[float] = []
    next_similarity: list[float] = []
    next_relation: list[str] = []
    straight_threshold = metadata["straight_curvature_threshold"]
    for index, row in turns.iterrows():
        next_index = (index + 1) % len(turns)
        next_row = turns.iloc[next_index]
        start_gap = int(row["end_index"])
        end_gap = int(next_row["start_index"])
        if next_index == 0:
            gap_indices = np.r_[np.arange(start_gap, len(segments)), np.arange(0, end_gap)]
            gap = track_length - float(row["exit_distance_m"]) + float(next_row["entry_distance_m"])
        else:
            gap_indices = np.arange(start_gap, end_gap)
            gap = float(next_row["entry_distance_m"] - row["exit_distance_m"])
        gap_curvature = np.abs(curvature[gap_indices]) if len(gap_indices) else np.asarray([])
        straight = float((gap_curvature < straight_threshold).sum() * step)
        angle_similarity = float(
            np.exp(
                -abs(
                    np.log(
                        max(float(row["absolute_angle_deg"]), 1e-9)
                        / max(float(next_row["absolute_angle_deg"]), 1e-9)
                    )
                )
            )
        )
        gap_similarity = float(np.exp(-gap / 120.0))
        transition.append(gap)
        straight_between.append(straight)
        next_similarity.append(angle_similarity * gap_similarity)
        next_relation.append(
            "same_direction"
            if row["direction_sign"] == next_row["direction_sign"]
            else "opposite_direction"
        )
    turns["transition_to_next_m"] = transition
    turns["straight_to_next_m"] = straight_between
    turns["next_turn_similarity"] = next_similarity
    turns["next_direction_relation"] = next_relation
    turns["preceding_transition_m"] = np.roll(turns["transition_to_next_m"], 1)
    turns["preceding_straight_m"] = np.roll(turns["straight_to_next_m"], 1)
    turns["approach_severity"] = (
        np.log1p(turns["preceding_straight_m"])
        * np.radians(turns["absolute_angle_deg"])
        / np.sqrt(turns["equivalent_radius_m"].clip(lower=1.0))
    )
    turns["steady_long_turn"] = (
        turns["absolute_angle_deg"].ge(cfg.steady_turn_minimum_angle_deg)
        & turns["turn_length_m"].ge(cfg.steady_turn_minimum_length_metres)
        & turns["curvature_consistency"].ge(cfg.steady_turn_minimum_consistency)
    )
    return turns


def build_turn_complexes_v9(
    turns: pd.DataFrame,
    config: CircuitShapeConfigV9 | None = None,
) -> pd.DataFrame:
    cfg = config or CircuitShapeConfigV9()
    if turns.empty:
        return pd.DataFrame()
    complex_ids = [1]
    for index in range(len(turns) - 1):
        row = turns.iloc[index]
        following = turns.iloc[index + 1]
        dynamic_limit = float(
            np.clip(
                0.5 * min(row["turn_length_m"], following["turn_length_m"]),
                cfg.minimum_complex_link_metres,
                cfg.maximum_complex_link_metres,
            )
        )
        linked = (
            float(row["transition_to_next_m"]) <= dynamic_limit
            and float(row["straight_to_next_m"]) < cfg.minimum_straight_metres
        )
        complex_ids.append(complex_ids[-1] if linked else complex_ids[-1] + 1)
    tagged = turns.copy()
    tagged["complex_id"] = complex_ids
    rows: list[dict[str, Any]] = []
    for complex_id, group in tagged.groupby("complex_id", sort=True):
        signs = group["direction_sign"].astype(int).tolist()
        alternating = all(left != right for left, right in zip(signs, signs[1:]))
        if len(group) == 1:
            kind = "standalone"
        elif len(set(signs)) == 1:
            kind = "same_direction_compound"
        elif alternating and len(group) == 2:
            kind = "chicane"
        elif alternating:
            kind = "esses"
        else:
            kind = "mixed_complex"
        gaps = group["transition_to_next_m"].iloc[:-1]
        span = float(group["turn_length_m"].sum() + gaps.sum())
        rows.append(
            {
                "complex_id": int(complex_id),
                "first_turn_id": int(group["turn_id"].iloc[0]),
                "last_turn_id": int(group["turn_id"].iloc[-1]),
                "turn_count": int(len(group)),
                "complex_type": kind,
                "complex_span_m": span,
                "complex_absolute_angle_deg": float(group["absolute_angle_deg"].sum()),
                "complex_net_angle_deg": float(group["signed_angle_deg"].sum()),
                "direction_change_count": int(
                    sum(left != right for left, right in zip(signs, signs[1:]))
                ),
                "mean_internal_gap_m": float(gaps.mean()) if len(gaps) else 0.0,
                "turn_sequence": "-".join(
                    f"{'L' if row.direction_sign > 0 else 'R'}{row.absolute_angle_deg:.0f}"
                    for row in group.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(rows)


def _straight_lengths_v9(
    segments: pd.DataFrame,
    metadata: dict[str, float],
    cfg: CircuitShapeConfigV9,
) -> np.ndarray:
    mask = (
        np.abs(segments["curvature_shape_v9"].to_numpy(dtype=float))
        < metadata["straight_curvature_threshold"]
    )
    step = metadata["microsegment_step_m"]
    return np.asarray(
        [len(run) * step for run in _cyclic_runs(mask) if len(run) * step >= cfg.minimum_straight_metres],
        dtype=float,
    )


def shape_profile_v9(
    segments: pd.DataFrame,
    turns: pd.DataFrame,
    complexes: pd.DataFrame,
    metadata: dict[str, float],
    config: CircuitShapeConfigV9 | None = None,
) -> dict[str, Any]:
    cfg = config or CircuitShapeConfigV9()
    step = metadata["microsegment_step_m"]
    track_length = float(len(segments) * step)
    length_km = track_length / 1000.0
    straight_lengths = _straight_lengths_v9(segments, metadata, cfg)
    curvature = np.abs(segments["curvature_shape_v9"].to_numpy(dtype=float))
    total_angle = float(np.degrees(curvature.sum() * step))
    profile: dict[str, Any] = {
        "shape_version": "9",
        "track_length_m": track_length,
        "microsegment_step_m": step,
        "curvature_noise_floor": metadata["curvature_noise_floor"],
        "canonical_seam_source_distance_m": metadata["canonical_seam_source_distance_m"],
        "turn_count": int(len(turns)),
        "turns_per_km": float(len(turns) / max(length_km, 1e-9)),
        "total_absolute_turning_deg": total_angle,
        "total_absolute_turning_deg_per_km": total_angle / max(length_km, 1e-9),
        "net_signed_turning_deg": float(
            np.degrees(segments["curvature_shape_v9"].sum() * step)
        ),
        "straight_count": int(len(straight_lengths)),
        "longest_straight_m": float(straight_lengths.max()) if len(straight_lengths) else 0.0,
        "top_three_straight_share": float(
            np.sort(straight_lengths)[-3:].sum() / max(track_length, 1e-9)
        )
        if len(straight_lengths)
        else 0.0,
        "complex_count": int(len(complexes)),
        "multi_turn_complex_count": int(complexes["turn_count"].gt(1).sum())
        if not complexes.empty
        else 0,
        "chicane_count": int(complexes["complex_type"].eq("chicane").sum())
        if not complexes.empty
        else 0,
        "esses_count": int(complexes["complex_type"].eq("esses").sum())
        if not complexes.empty
        else 0,
        "same_direction_compound_count": int(
            complexes["complex_type"].eq("same_direction_compound").sum()
        )
        if not complexes.empty
        else 0,
        "maximum_turns_in_complex": int(complexes["turn_count"].max())
        if not complexes.empty
        else 0,
        "steady_long_turn_count": int(turns["steady_long_turn"].sum())
        if not turns.empty
        else 0,
        "multi_apex_turn_count": int(turns["apex_count"].ge(2).sum())
        if not turns.empty
        else 0,
        "tightening_turn_count": int(turns["turn_shape"].eq("tightening").sum())
        if not turns.empty
        else 0,
        "opening_turn_count": int(turns["turn_shape"].eq("opening").sum())
        if not turns.empty
        else 0,
        "opposite_direction_link_count": int(
            (
                turns["next_direction_relation"].eq("opposite_direction")
                & turns["straight_to_next_m"].lt(cfg.minimum_straight_metres)
            ).sum()
        )
        if not turns.empty
        else 0,
        "same_direction_link_count": int(
            (
                turns["next_direction_relation"].eq("same_direction")
                & turns["straight_to_next_m"].lt(cfg.minimum_straight_metres)
            ).sum()
        )
        if not turns.empty
        else 0,
        "high_similarity_link_count": int(turns["next_turn_similarity"].ge(0.55).sum())
        if not turns.empty
        else 0,
        "maximum_approach_severity": float(turns["approach_severity"].max())
        if not turns.empty
        else 0.0,
        "approach_severity_per_km": float(turns["approach_severity"].sum())
        / max(length_km, 1e-9)
        if not turns.empty
        else 0.0,
    }
    winding_value = float(profile["net_signed_turning_deg"]) / 360.0
    winding_error = abs(winding_value - round(winding_value))
    profile["topological_winding_estimate"] = int(round(winding_value))
    profile["topological_winding_error"] = winding_error
    profile["shape_topology_quality"] = (
        "high"
        if winding_error <= 0.08
        else "review"
        if winding_error <= 0.15
        else "low"
    )
    profile["opposite_direction_links_per_km"] = profile[
        "opposite_direction_link_count"
    ] / max(length_km, 1e-9)
    profile["same_direction_links_per_km"] = profile[
        "same_direction_link_count"
    ] / max(length_km, 1e-9)
    profile["multi_turn_complexes_per_km"] = profile[
        "multi_turn_complex_count"
    ] / max(length_km, 1e-9)
    profile["multi_apex_turns_per_km"] = profile["multi_apex_turn_count"] / max(
        length_km, 1e-9
    )
    profile["tightening_turns_per_km"] = profile["tightening_turn_count"] / max(
        length_km, 1e-9
    )
    for name, lower, upper in ANGLE_BANDS_V9:
        selected = turns["absolute_angle_deg"].between(lower, upper, inclusive="left") if not turns.empty else pd.Series(dtype=bool)
        profile[f"{name}_count"] = int(selected.sum())
        profile[f"{name}_turning_share"] = (
            float(turns.loc[selected, "absolute_angle_deg"].sum()) / max(total_angle, 1e-9)
            if not turns.empty
            else 0.0
        )
    turn_distance = float(turns["turn_length_m"].sum()) if not turns.empty else 0.0
    for name, lower, upper in RADIUS_BANDS_V9:
        selected = turns["equivalent_radius_m"].between(lower, upper, inclusive="left") if not turns.empty else pd.Series(dtype=bool)
        profile[f"{name}_count"] = int(selected.sum())
        profile[f"{name}_distance_share"] = (
            float(turns.loc[selected, "turn_length_m"].sum()) / max(turn_distance, 1e-9)
            if not turns.empty
            else 0.0
        )
    for name, lower, upper in STRAIGHT_BANDS_V9:
        selected = (straight_lengths >= lower) & (straight_lengths < upper)
        profile[f"{name}_count"] = int(selected.sum())
        profile[f"{name}_track_share"] = float(straight_lengths[selected].sum()) / max(
            track_length, 1e-9
        )
    if not turns.empty:
        ordered_angles = np.sort(turns["absolute_angle_deg"].to_numpy(dtype=float))
        profile["top_three_turning_share"] = float(ordered_angles[-3:].sum()) / max(
            ordered_angles.sum(), 1e-9
        )
        direction_angle = turns.groupby("direction")["absolute_angle_deg"].sum()
        left = float(direction_angle.get("left", 0.0))
        right = float(direction_angle.get("right", 0.0))
        profile["directional_turning_imbalance"] = abs(left - right) / max(left + right, 1e-9)
        profile["complex_turn_share"] = float(
            complexes.loc[complexes["turn_count"].gt(1), "turn_count"].sum()
        ) / max(len(turns), 1)
        profile["steady_long_turn_distance_share"] = float(
            turns.loc[turns["steady_long_turn"], "turn_length_m"].sum()
        ) / max(track_length, 1e-9)
    else:
        profile.update(
            {
                "top_three_turning_share": 0.0,
                "directional_turning_imbalance": 0.0,
                "complex_turn_share": 0.0,
                "steady_long_turn_distance_share": 0.0,
            }
        )
    profile["geometry_only_straight_exposure_v9"] = float(
        np.clip(
            0.55 * profile["top_three_straight_share"]
            + 0.45 * profile["longest_straight_m"] / 1800.0,
            0.0,
            1.0,
        )
    )
    profile["geometry_only_slow_rotation_v9"] = float(
        np.clip(
            profile["radius_under_50_distance_share"]
            + 0.6 * profile["radius_50_90_distance_share"],
            0.0,
            1.0,
        )
    )
    profile["geometry_only_flowing_rotation_v9"] = float(
        np.clip(
            profile["radius_150_250_distance_share"]
            + profile["radius_250_450_distance_share"]
            + 0.5 * profile["steady_long_turn_distance_share"],
            0.0,
            1.0,
        )
    )
    profile["geometry_only_direction_change_v9"] = float(
        np.clip(profile["opposite_direction_link_count"] / max(length_km * 3.0, 1.0), 0.0, 1.0)
    )
    profile["geometry_only_complexity_v9"] = float(
        np.clip(
            0.55 * profile["complex_turn_share"]
            + 0.25 * profile["multi_apex_turn_count"] / max(len(turns), 1)
            + 0.20 * profile["tightening_turn_count"] / max(len(turns), 1),
            0.0,
            1.0,
        )
    )
    return profile


def analyze_circuit_shape_v9(
    microsegments: pd.DataFrame,
    config: CircuitShapeConfigV9 | None = None,
    **identity: Any,
) -> CircuitShapeResultV9:
    cfg = config or CircuitShapeConfigV9()
    segments, metadata = canonicalize_circuit_v9(microsegments, cfg)
    turns = extract_turns_v9(segments, metadata, cfg)
    complexes = build_turn_complexes_v9(turns, cfg)
    profile = shape_profile_v9(segments, turns, complexes, metadata, cfg)
    profile.update(identity)
    for frame in (turns, complexes, segments):
        for key, value in identity.items():
            frame[key] = value
    return CircuitShapeResultV9(pd.DataFrame([profile]), turns, complexes, segments)
