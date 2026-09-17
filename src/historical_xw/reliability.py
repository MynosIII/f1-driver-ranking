from __future__ import annotations

FINISHED = "finished"
ACCIDENT = "accident"
MECHANICAL = "mechanical"
OTHER = "other"

_ACCIDENT_STATUSES = {
    "Accident", "Collision", "Collision damage", "Spun off", "Damage",
    "Debris", "Injury", "Fatal accident", "Off track",
}

_MECHANICAL_STATUSES = {
    "Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics", "Electrical",
    "Suspension", "Brakes", "Puncture", "Tyre", "Wheel", "Steering",
    "Fuel system", "Fuel pressure", "Fuel pump", "Fuel leak", "Overheating",
    "Water leak", "Oil leak", "Oil pressure", "Oil pump", "Power Unit",
    "Turbo", "Battery", "Water pressure", "Vibrations", "Driveshaft",
    "Exhaust", "Undertray", "Wing", "Front wing", "Rear wing", "Radiator",
    "Alternator", "Differential", "Throttle", "Handling", "Track rod",
    "Wheel rim", "Wheel nut", "Wheel bearing", "Wheel bolt", "Rear wing minor",
    "ERS", "Power loss", "Technical", "Mechanical", "Cooling system",
    "Water pump", "Physical", "Halfshaft", "CV joint", "Chassis",
    "Electronics", "Hydraulic leak", "Launch control", "Pneumatics",
    "Seat", "Spark plugs", "Crankshaft", "Camshaft", "Fire", "Broken wing",
}

_OTHER_DNF_STATUSES = {
    "Disqualified", "Withdrew", "Not classified", "107% Rule", "Illness",
    "Safety concerns", "Excluded", "Did not qualify", "Did not prequalify",
    "Retired",
}


def classify_status(status: str | None) -> str:
    """Map an Ergast/Jolpica ``status`` string to who's responsible for a
    non-finish: the driver (accident), the car/team (mechanical), or
    neither cleanly (other). An unrecognized string fails safe to
    ``OTHER`` rather than guessing blame. Same vocabulary as
    ``f1predictor.features.reliability.classify_status`` in the companion
    F1Predictor project, kept in sync deliberately.
    """
    if not status:
        return OTHER
    if status == "Finished" or status.startswith("+"):
        return FINISHED
    if status in _ACCIDENT_STATUSES:
        return ACCIDENT
    if status in _MECHANICAL_STATUSES:
        return MECHANICAL
    if status in _OTHER_DNF_STATUSES:
        return OTHER
    return OTHER
