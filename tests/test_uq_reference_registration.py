from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.uq_reference_arm import (
    build_intervals,
    conformal_quantile,
    expand_cells,
    load_config,
    metric_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [30, 31, 30, 31],
            "true_rul": [10.0, 9.0, 8.0, 7.0],
            "pred_rul": [9.0, 8.0, 8.0, 8.0],
            "pred_q10": [7.0, 6.0, 6.0, 5.0],
            "pred_q50": [9.0, 8.0, 8.0, 8.0],
            "pred_q90": [11.0, 10.0, 10.0, 9.0],
        }
    )


def test_finite_sample_conformal_quantile_uses_registered_higher_rank() -> None:
    qhat, rank = conformal_quantile(np.arange(1.0, 20.0), alpha=0.10)

    assert rank == 18
    assert qhat == 18.0


def test_residual_and_cqr_paths_emit_ordered_closed_intervals() -> None:
    frame = _frame()
    for method in ("residual_split_cp", "cqr"):
        calibration, intervals, qhat, rank = build_intervals(
            frame, frame, method=method, alpha=0.25
        )
        metrics = metric_bundle(intervals)

        assert len(calibration) == 4
        assert np.isfinite(qhat)
        assert rank == 4
        assert (intervals["lower"] <= intervals["upper"]).all()
        assert intervals.groupby("unit_id")["is_endpoint"].sum().eq(1).all()
        assert metrics["test_engines"] == 2
        if method == "cqr":
            assert "base_covered" in intervals
            assert metric_bundle(intervals, base=True)["test_windows"] == 4


def test_frozen_registration_expands_to_exactly_280_bounded_cells() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "uq_reference_arm.yaml")
    cells = expand_cells(PROJECT_ROOT, config)

    assert config["status"] == "frozen_authorized_for_execution"
    assert config["registration"]["test_interval_results_seen_at_freeze"] is False
    assert len(cells) == 280
    assert len({cell.uq_run_id for cell in cells}) == 280
    assert sum(cell.panel == "kill_protocol_sensitivity" for cell in cells) == 120
    assert sum(
        cell.panel == "unified_reference" and cell.method == "residual_split_cp"
        for cell in cells
    ) == 80
    assert sum(cell.panel == "unified_reference" and cell.method == "cqr" for cell in cells) == 80
    assert not any("mondrian" in cell.method or "hcp" in cell.method for cell in cells)
