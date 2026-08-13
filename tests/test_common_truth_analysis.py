from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "common_truth"


def test_common_truth_summary_preserves_secondary_analysis_boundary() -> None:
    summary = json.loads((RESULTS / "COMMON_TRUTH_SUMMARY.json").read_text(encoding="utf-8"))

    assert summary["analysis_id"] == "common_truth_v1"
    assert summary["status"] == "completed_validated"
    assert summary["primary_kill_test_member"] is False
    assert summary["changes_kill_v1_decision"] is False
    assert summary["registered_input_runs"] == 120
    assert summary["run_truth_metric_rows"] == 240
    assert summary["paired_label_contrasts"] == 24
    assert summary["prediction_clipping"] == "none"
    assert summary["validation"]["endpoint_metrics_recomputed"] == 240
    assert summary["validation"]["artifact_errors"] == 0


def test_common_truth_tables_close_and_are_finite() -> None:
    metrics = pd.read_csv(RESULTS / "COMMON_TRUTH_RUN_METRICS.csv")
    contrasts = pd.read_csv(RESULTS / "COMMON_TRUTH_LABEL_CONTRASTS.csv")
    reversals = pd.read_csv(RESULTS / "COMMON_TRUTH_RANKING_REVERSALS.csv")

    assert len(metrics) == 240
    assert set(metrics["common_truth"]) == {"raw_rul", "piecewise_125"}
    assert set(metrics["prediction_clipping"]) == {"none"}
    assert metrics["rmse"].notna().all()
    assert set(metrics.loc[metrics["dataset"] == "FD001", "engine_count"]) == {100}
    assert set(metrics.loc[metrics["dataset"] == "FD004", "engine_count"]) == {248}
    assert len(contrasts) == 24
    assert set(contrasts["seed_count"]) == {5}
    assert contrasts["mean_rmse_difference"].notna().all()
    assert set(reversals.columns) == {
        "common_truth",
        "dataset",
        "sensor_set",
        "model_a",
        "model_b",
        "capped_trained_mean_rmse_delta",
        "linear_trained_mean_rmse_delta",
    }


def test_common_truth_validation_report_states_nonreplacement_boundary() -> None:
    report = (RESULTS / "COMMON_TRUTH_VALIDATION.md").read_text(encoding="utf-8")

    assert "COMPLETED AND VALIDATED" in report
    assert "Predictions were not clipped" in report
    assert "do not replace the" in report
    assert "change the frozen `kill_v1` PASS decision" in report
