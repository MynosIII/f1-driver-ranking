from __future__ import annotations

from .domain import Entry, HistoricalContext


def hungary_2021() -> tuple[list[Entry], HistoricalContext]:
    """Reproducible completed-race example; scores are illustrative precomputed priors."""
    entries = [
        Entry("Lewis Hamilton", "Mercedes", "Mercedes", 1, 1.35, 1.55, 0.45),
        Entry("Valtteri Bottas", "Mercedes", "Mercedes", 2, 0.55, 1.55, 0.45),
        Entry("Max Verstappen", "Red Bull", "Red Bull", 3, 1.65, 1.45, 0.55),
        Entry("Sergio Perez", "Red Bull", "Red Bull", 4, 0.45, 1.45, 0.55),
        Entry("Pierre Gasly", "AlphaTauri", "AlphaTauri", 5, 0.65, 0.10, 0.05),
        Entry("Lando Norris", "McLaren", "McLaren", 6, 1.00, 0.40, 0.20),
        Entry("Charles Leclerc", "Ferrari", "Ferrari", 7, 1.05, 0.65, 0.10),
        Entry("Esteban Ocon", "Alpine", "Alpine", 8, 0.25, -0.10, 0.05),
        Entry("Fernando Alonso", "Alpine", "Alpine", 9, 0.85, -0.10, 0.05),
        Entry("Sebastian Vettel", "Aston Martin", "Aston Martin", 10, 0.30, -0.20, -0.05),
        Entry("Daniel Ricciardo", "McLaren", "McLaren", 11, 0.00, 0.40, 0.20),
        Entry("Lance Stroll", "Aston Martin", "Aston Martin", 12, -0.20, -0.20, -0.05),
        Entry("Kimi Raikkonen", "Alfa Romeo", "Alfa Romeo", 13, 0.00, -0.95, -0.30),
        Entry("Antonio Giovinazzi", "Alfa Romeo", "Alfa Romeo", 14, -0.35, -0.95, -0.30),
        Entry("Carlos Sainz", "Ferrari", "Ferrari", 15, 0.70, 0.65, 0.10),
        Entry("Yuki Tsunoda", "AlphaTauri", "AlphaTauri", 16, -0.45, 0.10, 0.05),
        Entry("Nicholas Latifi", "Williams", "Williams", 18, -0.55, -1.10, -0.20),
        Entry("George Russell", "Williams", "Williams", 17, 0.75, -1.10, -0.20),
        Entry("Nikita Mazepin", "Haas", "Haas", 19, -1.15, -1.55, -0.60),
        Entry("Mick Schumacher", "Haas", "Haas", 20, -0.35, -1.55, -0.60),
    ]
    context = HistoricalContext(
        season=2021, event="Hungarian Grand Prix", regulation_era="2017-2021",
        race_distance_laps=70, wet_fraction=0.08, track_temperature=27.0,
        air_temperature=25.0, wind_speed=2.2, safety_car_laps=3, red_flags=1,
        circuit_features={"straight_share": 0.31, "traction_demand": 0.82, "aero_demand": 0.74},
    )
    return entries, context

