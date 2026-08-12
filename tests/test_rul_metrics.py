from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rul_audit.metrics.rul import endpoint_metrics, nasa_score, rmse


def test_registered_metric_formulas_and_sign_convention() -> None:
    truth = np.array([100.0, 100.0])
    prediction = np.array([87.0, 110.0])

    assert rmse(truth, prediction) == math.sqrt((13.0**2 + 10.0**2) / 2)
    assert nasa_score(truth, prediction) == (math.e - 1.0) * 2


def test_endpoint_metrics_use_one_latest_row_per_engine() -> None:
    frame = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [30, 31, 30, 31],
            "true_rul": [6.0, 5.0, 8.0, 7.0],
            "pred_rul": [100.0, 4.0, 100.0, 9.0],
        }
    )

    metrics = endpoint_metrics(frame)

    assert metrics["engine_count"] == 2
    assert metrics["rmse"] == math.sqrt(2.5)
    expected_score = math.exp(1.0 / 13.0) - 1.0 + math.exp(2.0 / 10.0) - 1.0
    assert math.isclose(float(metrics["nasa_score"]), expected_score)
