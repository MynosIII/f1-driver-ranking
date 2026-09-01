from __future__ import annotations

from enum import StrEnum

import numpy as np


class OutcomeClass(StrEnum):
    FINISHED = "finished"
    MECHANICAL = "mechanical"
    DRIVER_FAULT = "driver_fault"
    DNS = "dns"
    OTHER = "other"


MECHANICAL_TERMS = (
    "engine", "gearbox", "transmission", "clutch", "hydraulic", "mechanical",
    "suspension", "electrical", "brake", "fuel", "power unit", "radiator",
    "oil", "overheating", "turbo", "driveshaft", "wheel bearing", "water leak",
    "pneumatic", "spark plug", "halfshaft", "cv joint",
)
DRIVER_FAULT_TERMS = (
    "accident", "collision", "spun", "spin", "disqualified", "excluded",
    "black flag", "illegal", "underweight",
)
DNS_TERMS = (
    "did not start", "did not qualify", "did not prequalify", "withdrawn", "withdrew",
)


def classify_outcome(status: object) -> OutcomeClass:
    """Classify an official result without assigning ambiguous retirements to a driver."""

    value = str(status).strip().lower()
    if (
        value.startswith("finished")
        or value.startswith("+")
        or value in {"classified", "lapped"}
    ):
        return OutcomeClass.FINISHED
    if any(term in value for term in DNS_TERMS):
        return OutcomeClass.DNS
    if any(term in value for term in DRIVER_FAULT_TERMS):
        return OutcomeClass.DRIVER_FAULT
    if any(term in value for term in MECHANICAL_TERMS):
        return OutcomeClass.MECHANICAL
    return OutcomeClass.OTHER


def normalized_finish(position: int | float, field_size: int) -> float:
    """Map winner to 1, last to 0, and intermediate positions linearly."""

    if field_size <= 1:
        return 1.0
    return float(np.clip(1.0 - (float(position) - 1.0) / (field_size - 1.0), 0.0, 1.0))


def observed_driver_performance(
    position: int | float,
    field_size: int,
    outcome: OutcomeClass | str,
) -> float:
    """Return driver-attributable race performance; NaN means no driver update."""

    outcome_class = OutcomeClass(outcome)
    if outcome_class in {OutcomeClass.MECHANICAL, OutcomeClass.DNS}:
        return float("nan")
    if outcome_class is OutcomeClass.DRIVER_FAULT:
        return 0.0
    return normalized_finish(position, field_size)
