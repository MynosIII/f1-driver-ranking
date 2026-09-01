from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage
from scipy.signal import savgol_filter


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
USER_AGENT = "historical-xw-geometry-v5/0.1 (historical research; local application)"

GEOMETRY_MODEL_FEATURES = (
    "straight_demand",
    "altitude_demand",
)

RACE_SOURCE_OVERRIDES: dict[tuple[int, str], tuple[str, float, str]] = {
    (1957, "French Grand Prix"): (
        "File:Rouen-Les-Essarts.svg",
        6.542,
        "Wikimedia Rouen-Les-Essarts layout; race-page course length",
    ),
    (1962, "French Grand Prix"): (
        "File:Rouen-Les-Essarts.svg",
        6.542,
        "Wikimedia Rouen-Les-Essarts layout; race-page course length",
    ),
    (1964, "French Grand Prix"): (
        "File:Rouen-Les-Essarts.svg",
        6.542,
        "Wikimedia Rouen-Les-Essarts layout; matching 1957/1962/1968 course",
    ),
    (1968, "French Grand Prix"): (
        "File:Rouen-Les-Essarts.svg",
        6.542,
        "Wikimedia Rouen-Les-Essarts layout; race-page course length",
    ),
    (1964, "Austrian Grand Prix"): (
        "File:Circuit Zeltweg.svg",
        3.186,
        "Wikimedia 1964 Zeltweg layout; race-page course length",
    ),
}


@dataclass(frozen=True, slots=True)
class RaceLayoutSource:
    season: int
    round: int
    event: str
    circuit_id: str
    wikipedia_title: str
    image_title: str | None
    course_length_km: float | None
    latitude: float
    longitude: float


def _normalise_title(value: str) -> str:
    return value.replace("_", " ").strip()


def _field_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _template_fields(wikitext: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in wikitext.splitlines():
        match = re.match(r"^\|\s*([^=]+?)\s*=\s*(.*?)\s*$", line)
        if match:
            fields[_field_key(match.group(1))] = match.group(2).strip()
    return fields


def _first_number(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = re.sub(r"<!--.*?-->", "", value)
    match = re.search(r"(?<!\d)(\d+(?:[.,]\d+)?)", cleaned)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _image_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = unquote(value).strip()
    cleaned = re.sub(r"^\[\[(?:File|Image):", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.split("|")[0].replace("]]", "").strip()
    cleaned = re.sub(r"^(?:File|Image):", "", cleaned, flags=re.IGNORECASE)
    if not re.search(r"\.(?:svg|png|jpe?g)$", cleaned, flags=re.IGNORECASE):
        return None
    return _normalise_title(f"File:{cleaned}")


class WikipediaGeometryClient:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.media_cache = cache / "media"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.media_cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_media_request = 0.0
        self._last_api_request = 0.0

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(8):
            delay = max(0.0, 1.0 - (time.monotonic() - self._last_api_request))
            if delay:
                time.sleep(delay)
            response = self.session.get(url, params=params, timeout=60)
            self._last_api_request = time.monotonic()
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            time.sleep(min(wait, 60.0))
        response.raise_for_status()
        raise RuntimeError("unreachable")

    def race_layout_sources(self, events: pd.DataFrame) -> pd.DataFrame:
        required = {
            "season",
            "round",
            "event",
            "circuit_id",
            "latitude",
            "longitude",
        }
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"Events are missing columns: {sorted(missing)}")
        cache_path = self.cache / "race_layout_sources.json"
        records: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            records = json.loads(cache_path.read_text(encoding="utf-8"))
        requested = {
            _normalise_title(f"{int(row.season)} {row.event}"): row
            for row in events.itertuples(index=False)
        }
        missing_titles = [title for title in requested if title not in records]
        for start in range(0, len(missing_titles), 40):
            batch = missing_titles[start : start + 40]
            payload = self._get_json(
                WIKIPEDIA_API,
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "titles": "|".join(batch),
                    "redirects": 1,
                    "format": "json",
                    "formatversion": 2,
                },
            )
            query = payload.get("query", {})
            aliases = {
                _normalise_title(item["from"]): _normalise_title(item["to"])
                for group in ("normalized", "redirects")
                for item in query.get(group, [])
            }
            pages = {_normalise_title(page["title"]): page for page in query.get("pages", [])}
            for title in batch:
                canonical = aliases.get(title, title)
                canonical = aliases.get(canonical, canonical)
                page = pages.get(canonical, {})
                revisions = page.get("revisions", [])
                wikitext = ""
                if revisions:
                    wikitext = revisions[0].get("slots", {}).get("main", {}).get("content", "")
                fields = _template_fields(wikitext)
                course_length = _first_number(fields.get("coursekm"))
                if course_length is None:
                    distance = _first_number(fields.get("distancekm"))
                    laps = _first_number(fields.get("distancelaps"))
                    if distance is not None and laps and laps > 0:
                        course_length = distance / laps
                records[title] = {
                    "wikipedia_title": canonical,
                    "image_title": _image_title(fields.get("image")),
                    "course_length_km": course_length,
                    "page_found": bool(wikitext),
                }
            cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

        rows: list[dict[str, Any]] = []
        for title, event in requested.items():
            source = records[title]
            rows.append(
                {
                    "season": int(event.season),
                    "round": int(event.round),
                    "event": str(event.event),
                    "circuit_id": str(event.circuit_id),
                    "latitude": float(event.latitude),
                    "longitude": float(event.longitude),
                    **source,
                }
            )
        frame = pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True)
        frame["source_override"] = False
        frame["source_override_note"] = None
        for (season, event), (image, length, note) in RACE_SOURCE_OVERRIDES.items():
            selected = frame["season"].eq(season) & frame["event"].eq(event)
            frame.loc[selected, "image_title"] = image
            frame.loc[selected, "course_length_km"] = length
            frame.loc[selected, "source_override"] = True
            frame.loc[selected, "source_override_note"] = note
        known_lengths = frame.dropna(subset=["image_title", "course_length_km"])
        length_by_image = known_lengths.groupby("image_title")["course_length_km"].median()
        missing_length = frame["course_length_km"].isna() & frame["image_title"].notna()
        frame.loc[missing_length, "course_length_km"] = frame.loc[
            missing_length, "image_title"
        ].map(length_by_image)
        known_images = frame.dropna(subset=["image_title", "course_length_km"]).copy()
        known_images["length_key"] = known_images["course_length_km"].round(3)
        image_by_circuit_length = (
            known_images.groupby(["circuit_id", "length_key"])["image_title"]
            .agg(lambda values: values.mode().iloc[0])
            .to_dict()
        )
        for index in frame.index[frame["image_title"].isna() & frame["course_length_km"].notna()]:
            key = (str(frame.at[index, "circuit_id"]), round(float(frame.at[index, "course_length_km"]), 3))
            frame.at[index, "image_title"] = image_by_circuit_length.get(key)
        return frame

    def image_information(self, image_titles: list[str]) -> pd.DataFrame:
        cache_path = self.cache / "image_information.json"
        records: dict[str, dict[str, Any]] = {}
        if cache_path.exists():
            records = json.loads(cache_path.read_text(encoding="utf-8"))
        requested = sorted({_normalise_title(title) for title in image_titles})
        missing = [
            title
            for title in requested
            if title not in records or "thumbnail_url" not in records[title]
        ]
        for start in range(0, len(missing), 40):
            batch = missing[start : start + 40]
            payload = self._get_json(
                WIKIPEDIA_API,
                {
                    "action": "query",
                    "prop": "imageinfo",
                    "iiprop": "url|mime|sha1|extmetadata",
                    "iiurlwidth": 1200,
                    "titles": "|".join(batch),
                    "format": "json",
                    "formatversion": 2,
                },
            )
            for page in payload.get("query", {}).get("pages", []):
                title = _normalise_title(page.get("title", ""))
                info = (page.get("imageinfo") or [{}])[0]
                metadata = info.get("extmetadata", {})
                records[title] = {
                    "image_title": title,
                    "media_url": info.get("url"),
                    "thumbnail_url": info.get("thumburl"),
                    "mime_type": info.get("mime"),
                    "sha1": info.get("sha1"),
                    "license": metadata.get("LicenseShortName", {}).get("value"),
                    "artist": metadata.get("Artist", {}).get("value"),
                    "description_url": info.get("descriptionurl"),
                }
            for title in batch:
                records.setdefault(title, {"image_title": title, "media_url": None})
            cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return pd.DataFrame([records[title] for title in requested])

    def download_media(self, information: pd.Series | dict[str, Any]) -> Path:
        title = str(information["image_title"])
        url = information.get("media_url")
        if not url:
            raise ValueError(f"No downloadable media URL for {title}")
        suffix = Path(title).suffix.lower() or ".bin"
        sha1 = str(information.get("sha1") or abs(hash(title)))
        target = self.media_cache / f"{sha1}{suffix}"
        if target.exists():
            return target
        cached_thumbnails = sorted(self.media_cache.glob(f"{sha1}_*px.png"))
        if cached_thumbnails:
            return cached_thumbnails[0]
        download_url = "https://commons.wikimedia.org/w/thumb.php"
        download_params = {"f": re.sub(r"^File:", "", title), "width": 800}
        target = self.media_cache / f"{sha1}_800px.png"
        if not target.exists():
            for attempt in range(6):
                delay = max(0.0, 0.75 - (time.monotonic() - self._last_media_request))
                if delay:
                    time.sleep(delay)
                response = self.session.get(
                    str(download_url), params=download_params, timeout=90
                )
                self._last_media_request = time.monotonic()
                if response.status_code == 400:
                    response = self.session.get(
                        str(download_url),
                        params={"f": download_params["f"], "width": 500},
                        timeout=90,
                    )
                if response.status_code == 404:
                    response = self.session.get(
                        str(information.get("thumbnail_url") or url), timeout=90
                    )
                if response.status_code != 429:
                    response.raise_for_status()
                    target.write_bytes(response.content)
                    break
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                time.sleep(min(wait, 30.0))
            else:
                response.raise_for_status()
        return target

    def circuit_altitudes(self, events: pd.DataFrame) -> pd.DataFrame:
        cache_path = self.cache / "circuit_altitudes.json"
        records: dict[str, float | None] = {}
        if cache_path.exists():
            records = json.loads(cache_path.read_text(encoding="utf-8"))
        circuits = events.drop_duplicates("circuit_id")
        missing = [row for row in circuits.itertuples() if str(row.circuit_id) not in records]
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            response = self._get_json(
                ELEVATION_API,
                {
                    "latitude": ",".join(str(float(row.latitude)) for row in batch),
                    "longitude": ",".join(str(float(row.longitude)) for row in batch),
                },
            )
            values = response["elevation"] if isinstance(response, dict) else [item["elevation"] for item in response]
            if not isinstance(values, list):
                values = [values]
            for row, value in zip(batch, values, strict=True):
                records[str(row.circuit_id)] = None if value is None else float(value)
            cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return circuits[["circuit_id", "latitude", "longitude"]].assign(
            altitude_m=lambda frame: frame["circuit_id"].map(records)
        )


def _smooth(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < 9:
        return values
    size = min(window, len(values) if len(values) % 2 else len(values) - 1)
    return savgol_filter(values, max(size, 7), 3, mode="wrap")


def _resample_closed_outline(points: np.ndarray, length_m: float, step_m: float) -> pd.DataFrame:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 20:
        raise ValueError("Layout outline contains fewer than 20 valid points")
    if np.linalg.norm(finite[0] - finite[-1]) > 1e-6:
        finite = np.vstack([finite, finite[0]])
    differences = np.diff(finite, axis=0)
    cumulative = np.r_[0.0, np.cumsum(np.hypot(differences[:, 0], differences[:, 1]))]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    finite = finite[keep]
    cumulative = cumulative[keep]
    if cumulative[-1] <= 0:
        raise ValueError("Layout outline has zero length")
    target = np.arange(0.0, length_m, step_m)
    source_distance = cumulative / cumulative[-1] * length_m
    x = np.interp(target, source_distance, finite[:, 0])
    y = np.interp(target, source_distance, finite[:, 1])
    coordinate_scale = length_m / cumulative[-1]
    x = _smooth((x - x[0]) * coordinate_scale)
    y = _smooth((y - y[0]) * coordinate_scale)
    dx = np.gradient(x, step_m)
    dy = np.gradient(y, step_m)
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = _smooth(np.gradient(heading, step_m), 21)
    radius = np.divide(
        1.0,
        np.abs(curvature),
        out=np.full_like(curvature, np.inf),
        where=np.abs(curvature) > 1e-7,
    )
    segment_class = np.select(
        [radius >= 450.0, radius < 100.0, radius < 250.0],
        ["straight", "tight", "medium"],
        default="sweeping",
    )
    return pd.DataFrame(
        {
            "distance": target,
            "x": x,
            "y": y,
            "heading": heading,
            "curvature": curvature,
            "radius": radius,
            "direction": np.where(
                segment_class == "straight",
                "straight",
                np.where(curvature >= 0, "left", "right"),
            ),
            "segment_class": segment_class,
        }
    )


def _svg_outline(path: Path) -> np.ndarray:
    try:
        from svgelements import Path as SVGPath
        from svgelements import SVG
    except ImportError as error:
        raise RuntimeError("Install requirements-geometry-v5.txt to parse SVG layouts") from error
    document = SVG.parse(path)
    candidates: list[tuple[float, Any]] = []
    for element in document.elements():
        if not isinstance(element, SVGPath):
            continue
        try:
            length = float(element.length(error=1e-4))
            start = element.point(0.0)
            end = element.point(1.0)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if length < 20.0 or start is None or end is None:
            continue
        closure = float(np.hypot(float(start.x - end.x), float(start.y - end.y)))
        closure_bonus = 2.0 if closure <= max(length * 0.03, 3.0) else 1.0
        stroke = str(element.values.get("stroke", "")).lower()
        fill = str(element.values.get("fill", "")).lower()
        style_bonus = 1.5 if stroke not in {"", "none"} and fill in {"", "none"} else 1.0
        candidates.append((length * closure_bonus * style_bonus, element))
    if not candidates:
        raise ValueError("No usable vector path exists in the SVG")
    selected = max(candidates, key=lambda item: item[0])[1]
    parameters = np.linspace(0.0, 1.0, 12000)
    points = []
    for parameter in parameters:
        point = selected.point(float(parameter))
        points.append((float(point.x), float(point.y)))
    return np.asarray(points, dtype=float)


def _raster_outline(path: Path) -> np.ndarray:
    from skimage.measure import find_contours

    image = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    rgb = image[:, :, :3].astype(float)
    alpha = image[:, :, 3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    colour_distance = np.linalg.norm(rgb - background, axis=2)
    gray = rgb.mean(axis=2)
    if np.quantile(alpha, 0.05) < 32:
        mask = (alpha > 64) & ((gray < 245) | (colour_distance > 20))
    else:
        mask = (colour_distance > 35) & (gray < 245)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("No foreground circuit line detected in raster layout")
    best_label = None
    best_score = -np.inf
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        if len(xx) < 30:
            continue
        bbox_area = (xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1)
        score = bbox_area * np.sqrt(len(xx))
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is None:
        raise ValueError("No sufficiently large circuit component detected")
    component = labels == best_label
    contours = find_contours(component.astype(float), 0.5, fully_connected="high")
    if not contours:
        raise ValueError("Circuit component has no traceable outline")
    contour = max(contours, key=len)
    return np.column_stack([contour[:, 1], -contour[:, 0]])


def media_to_microsegments(path: Path, length_km: float, step_m: float = 5.0) -> pd.DataFrame:
    if not np.isfinite(length_km) or length_km <= 0:
        raise ValueError("A positive published course length is required")
    suffix = path.suffix.lower()
    if suffix == ".svg":
        outline = _svg_outline(path)
        source_type = "wikipedia_svg_scaled"
    elif suffix in {".png", ".jpg", ".jpeg"}:
        outline = _raster_outline(path)
        source_type = "wikipedia_raster_traced_scaled"
    else:
        raise ValueError(f"Unsupported circuit-layout format: {suffix}")
    microsegments = _resample_closed_outline(outline, float(length_km) * 1000.0, step_m)
    microsegments["geometry_source_type"] = source_type
    return microsegments


def _runs(mask: np.ndarray, step: float) -> list[tuple[int, int, float]]:
    changes = np.diff(np.r_[False, mask.astype(bool), False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end), float((end - start) * step)) for start, end in zip(starts, ends)]


def geometry_profile(
    microsegments: pd.DataFrame,
    *,
    source: RaceLayoutSource,
    altitude_m: float | None,
    media_information: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    segments = microsegments.sort_values("distance").reset_index(drop=True)
    step = float(np.median(np.diff(segments["distance"])))
    length_m = float(segments["distance"].max() + step)
    straight_mask = segments["segment_class"].eq("straight").to_numpy()
    corner_mask = ~straight_mask
    straight_runs = [run for run in _runs(straight_mask, step) if run[2] >= 25.0]
    corner_runs = [run for run in _runs(corner_mask, step) if run[2] >= 15.0]
    straight_lengths = np.asarray([run[2] for run in straight_runs], dtype=float)
    centres = np.asarray([(start + end) * step / 2 for start, end, _ in corner_runs])
    spacing = (
        np.diff(np.r_[centres, centres[0] + length_m])
        if len(centres) > 1
        else np.asarray([], dtype=float)
    )
    angles = []
    radii = []
    directions = []
    for start, end, _ in corner_runs:
        piece = segments.iloc[start:end]
        angles.append(float(np.degrees(np.abs(piece["curvature"]).sum() * step)))
        finite_radius = piece.loc[np.isfinite(piece["radius"]), "radius"]
        radii.append(float(finite_radius.median()) if len(finite_radius) else np.nan)
        directions.append(str(piece["direction"].mode().iloc[0]))
    length_km = length_m / 1000.0
    straight_share = float(straight_mask.mean())
    tight_share = float(segments["segment_class"].eq("tight").mean())
    medium_share = float(segments["segment_class"].eq("medium").mean())
    sweeping_share = float(segments["segment_class"].eq("sweeping").mean())
    corner_count = len(corner_runs)
    longest_straight = float(straight_lengths.max()) if len(straight_lengths) else 0.0
    altitude = float(altitude_m) if altitude_m is not None and np.isfinite(altitude_m) else np.nan
    return {
        "season": source.season,
        "round": source.round,
        "event": source.event,
        "circuit_id": source.circuit_id,
        "wikipedia_title": source.wikipedia_title,
        "image_title": source.image_title,
        "media_url": media_information.get("media_url"),
        "description_url": media_information.get("description_url"),
        "license": media_information.get("license"),
        "geometry_source_type": str(segments["geometry_source_type"].iloc[0]),
        "profile_quality": "historical_layout_scaled_to_published_length",
        "geometry_available": True,
        "track_length_m": length_m,
        "altitude_m": altitude,
        "straight_share": straight_share,
        "corner_share": 1.0 - straight_share,
        "longest_straight_m": longest_straight,
        "median_straight_m": float(np.median(straight_lengths)) if len(straight_lengths) else np.nan,
        "corner_count": corner_count,
        "corner_density_per_km": corner_count / max(length_km, 1e-9),
        "median_corner_spacing_m": float(np.median(spacing)) if len(spacing) else np.nan,
        "mean_corner_spacing_m": float(np.mean(spacing)) if len(spacing) else np.nan,
        "median_corner_angle_deg": float(np.median(angles)) if angles else np.nan,
        "mean_corner_angle_deg": float(np.mean(angles)) if angles else np.nan,
        "p90_corner_angle_deg": float(np.quantile(angles, 0.9)) if angles else np.nan,
        "median_corner_radius_m": float(np.nanmedian(radii)) if radii else np.nan,
        "tight_corner_share": tight_share,
        "medium_corner_share": medium_share,
        "sweeping_corner_share": sweeping_share,
        "left_corner_share": directions.count("left") / max(corner_count, 1),
        "right_corner_share": directions.count("right") / max(corner_count, 1),
        "total_turning_deg_per_km": float(
            np.degrees(np.abs(segments["curvature"]).sum() * step) / max(length_km, 1e-9)
        ),
        "straight_demand": float(np.clip(0.6 * straight_share + 0.4 * longest_straight / 2200.0, 0, 1)),
        "tight_corner_demand": float(np.clip(tight_share * 2.5, 0, 1)),
        "medium_corner_demand": float(np.clip(medium_share * 2.2, 0, 1)),
        "sweeping_corner_demand": float(np.clip(sweeping_share * 2.2, 0, 1)),
        "corner_density_demand": float(np.clip(corner_count / max(length_km, 1e-9) / 5.0, 0, 1)),
        "direction_change_demand": float(
            np.clip(np.degrees(np.abs(segments["curvature"]).sum() * step) / length_km / 500.0, 0, 1)
        ),
        "altitude_demand": float(np.clip(max(altitude, 0.0) / 2500.0, 0, 1)) if np.isfinite(altitude) else 0.0,
    }
