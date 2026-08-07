from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class Component(StrEnum):
    DRIVER = "driver"
    CAR = "car"
    TEAM = "team"


class AnalysisMode(StrEnum):
    CROSS_FITTED = "cross_fitted"
    DESCRIPTIVE = "retrospective_descriptive"


class NeutralReference(StrEnum):
    SEASON_GRID_AVERAGE = "season_grid_average"
    REGULATION_ERA_AVERAGE = "regulation_era_average"
    HISTORICAL_GRID_AVERAGE = "historical_grid_average"
    MEDIAN_COMPONENT = "median_component"
    REPLACEMENT_LEVEL = "replacement_level"


@dataclass(frozen=True, slots=True)
class Entry:
    driver: str
    constructor: str
    team: str
    grid: int
    driver_score: float
    car_score: float
    team_score: float
    driver_uncertainty: float = 0.25
    car_uncertainty: float = 0.20
    team_uncertainty: float = 0.18


@dataclass(frozen=True, slots=True)
class HistoricalContext:
    season: int
    event: str
    regulation_era: str
    race_distance_laps: int
    wet_fraction: float
    track_temperature: float
    air_temperature: float
    wind_speed: float
    safety_car_laps: int = 0
    red_flags: int = 0
    circuit_features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    simulations: int = 20_000
    seed: int = 20210731
    neutral_reference: NeutralReference = NeutralReference.SEASON_GRID_AVERAGE
    mode: AnalysisMode = AnalysisMode.CROSS_FITTED

