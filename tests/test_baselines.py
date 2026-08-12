from __future__ import annotations

import numpy as np
import pytest
import torch

from rul_audit.models.baselines import (
    build_neural_model,
    fit_lightgbm,
    predict_lightgbm,
    predict_neural,
)


@pytest.mark.parametrize("model_id", ["lstm", "cnn_1d", "transformer"])
def test_neural_baseline_shapes(model_id: str) -> None:
    features = np.zeros((4, 30, 14), dtype=np.float32)
    point_model = build_neural_model(model_id, input_size=14, output_mode="point")
    quantile_model = build_neural_model(model_id, input_size=14, output_mode="quantile")

    assert predict_neural(point_model, features).shape == (4,)
    quantiles = predict_neural(quantile_model, features, output_mode="quantile")
    assert quantiles.shape == (4, 3)
    assert bool(np.all(quantiles[:, 0] <= quantiles[:, 1]))
    assert bool(np.all(quantiles[:, 1] <= quantiles[:, 2]))
    assert isinstance(point_model, torch.nn.Module)


def test_lightgbm_uses_flattened_windows_for_point_and_quantiles() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(24, 6, 3)).astype(np.float32)
    labels = rng.normal(size=24).astype(np.float32)

    point = fit_lightgbm(features, labels, seed=11, n_estimators=3)
    quantile = fit_lightgbm(
        features, labels, seed=11, output_mode="quantile", n_estimators=3
    )

    assert predict_lightgbm(point, features).shape == (24,)
    quantile_predictions = predict_lightgbm(
        quantile, features, output_mode="quantile"
    )
    assert quantile_predictions.shape == (24, 3)
    assert bool(np.all(quantile_predictions[:, 0] <= quantile_predictions[:, 1]))
    assert bool(np.all(quantile_predictions[:, 1] <= quantile_predictions[:, 2]))
