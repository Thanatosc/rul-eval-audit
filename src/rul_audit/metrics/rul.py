"""Registered point metrics for C-MAPSS RUL evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _paired_arrays(true_rul: np.ndarray, pred_rul: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(true_rul, dtype=float).reshape(-1)
    prediction = np.asarray(pred_rul, dtype=float).reshape(-1)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("true_rul and pred_rul must be non-empty arrays of equal shape")
    if not bool(np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("metrics require finite values")
    return truth, prediction


def rmse(true_rul: np.ndarray, pred_rul: np.ndarray) -> float:
    truth, prediction = _paired_arrays(true_rul, pred_rul)
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def nasa_score(true_rul: np.ndarray, pred_rul: np.ndarray) -> float:
    """PHM08 asymmetric score with d = prediction - truth."""

    truth, prediction = _paired_arrays(true_rul, pred_rul)
    errors = prediction - truth
    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(penalties.sum())


def endpoint_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the latest evaluated cycle for every engine."""

    required = {"unit_id", "cycle", "true_rul", "pred_rul"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    ordered = predictions.sort_values(["unit_id", "cycle"])
    result = ordered.groupby("unit_id", sort=True, as_index=False).tail(1)
    if result["unit_id"].duplicated().any():
        raise AssertionError("endpoint selection did not produce one row per engine")
    return result.reset_index(drop=True)


def endpoint_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    endpoints = endpoint_rows(predictions)
    return {
        "rmse": rmse(endpoints["true_rul"].to_numpy(), endpoints["pred_rul"].to_numpy()),
        "nasa_score": nasa_score(
            endpoints["true_rul"].to_numpy(), endpoints["pred_rul"].to_numpy()
        ),
        "engine_count": len(endpoints),
    }
