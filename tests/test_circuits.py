import numpy as np
import pandas as pd

from historical_xw.circuits import reconstruct_microsegments


def test_distance_alignment_and_geometry() -> None:
    rows = []
    for lap in (1, 2):
        for angle in np.linspace(0, 2 * np.pi, 101):
            rows.append({"lap": lap, "distance": angle * 100, "x": 100 * np.cos(angle), "y": 100 * np.sin(angle), "z": angle, "speed": 180, "throttle": 70, "brake": 0.1})
    result = reconstruct_microsegments(pd.DataFrame(rows), step_metres=5)
    assert result["distance"].diff().dropna().eq(5).all()
    assert {"heading", "curvature", "radius", "segment_class", "data_quality"}.issubset(result.columns)

