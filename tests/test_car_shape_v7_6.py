from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from historical_xw.car_shape_v7_6 import _loo_predictions_all_alphas


def test_fast_all_alpha_loo_matches_independent_sklearn_ridge() -> None:
    rows = []
    for round_number, (first, second, target) in enumerate(
        [
            (0.0, 1.0, -0.2),
            (0.2, 0.7, -0.1),
            (0.5, 0.5, 0.1),
            (0.8, 0.3, 0.25),
            (1.0, 0.0, 0.4),
        ],
        start=1,
    ):
        rows.append(
            {
                "season": 2000,
                "round": round_number,
                "model_entity_id": "car",
                "layout_key": f"layout-{round_number}",
                "car_relative_performance": target,
                "shape_topology_quality": "high",
                "trace_method_v9": "test",
                "first": first,
                "second": second,
            }
        )
    targets = pd.DataFrame(rows)
    predictions = _loo_predictions_all_alphas(
        targets, ("first", "second"), (0.1, 10.0), minimum_races=4
    )
    scaler = StandardScaler().fit(targets[["first", "second"]])
    for result in predictions.itertuples(index=False):
        train = targets[targets["round"].ne(result.round)]
        test = targets[targets["round"].eq(result.round)]
        model = Ridge(alpha=result.alpha).fit(
            scaler.transform(train[["first", "second"]]),
            train["car_relative_performance"],
        )
        expected = float(model.predict(scaler.transform(test[["first", "second"]]))[0])
        assert np.isclose(result.prediction, expected)
