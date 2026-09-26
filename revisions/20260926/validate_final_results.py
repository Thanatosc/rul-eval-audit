"""Final dataset/artifact completeness and independently recomputed endpoint scores."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from train_repair import cells, REPO

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"


def main():
    expected = {c.run_id for c in cells("grid")}
    metas = {f.parent.name: json.loads(f.read_text(encoding="utf-8")) for f in (HERE / "runs").glob("*/meta.json")}
    assert set(metas) == expected and len(metas) == 96
    spec_hash = hashlib.sha256((HERE / "ANALYSIS_SPEC.md").read_bytes()).hexdigest()
    script_hash = hashlib.sha256((HERE / "train_repair.py").read_bytes()).hexdigest()
    model_hash = hashlib.sha256((REPO / "src/rul_audit/models/baselines.py").read_bytes()).hexdigest()
    scores = pd.read_csv(ANALYSIS / "all_run_scores.csv")
    assert len(scores) == 576
    mismatches, data_hashes, largest_error = [], {}, 0.
    rows = 0
    for run_id, meta in metas.items():
        assert meta["status"] == "completed"
        assert meta["script_sha256"] == script_hash
        assert meta["specification_sha256"] == spec_hash
        assert meta["source_model_sha256"] == model_hash
        for file, sha in meta["data_sha256"].items():
            if file not in data_hashes:
                data_hashes[file] = hashlib.sha256((REPO / "data/interim/cmapss" / file).read_bytes()).hexdigest()
            assert sha == data_hashes[file]
        path = HERE / "runs" / run_id
        for name in ("val", "calib", "test"):
            frame = pd.read_parquet(path / f"preds_{name}.parquet")
            assert len(frame) == meta["prediction_rows"][name]
            assert frame.unit_id.nunique() == meta["units"][name]
            assert not frame.duplicated(["unit_id", "cycle"]).any()
            assert np.isfinite(frame[["raw_rul", "pred_rul"]].to_numpy()).all()
            rows += len(frame)
        ep = frame.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)
        for truth in ("raw", "capped125"):
            actual = ep.raw_rul.to_numpy(float)
            if truth == "capped125": actual = np.minimum(actual, 125)
            error = ep.pred_rul.to_numpy(float) - actual
            rmse = float(np.linalg.norm(error)/np.sqrt(len(error)))
            cost = np.asarray([np.exp(-e/13)-1 if e<0 else np.exp(e/10)-1 for e in error])
            found = scores[(scores.run_id == run_id) & (scores.truth == truth) & (scores.population == "endpoint")]
            assert len(found) == 1
            record = found.iloc[0]
            delta = max(abs(record.rmse-rmse), abs(record.nasa_mean-float(cost.mean()))/max(1, float(cost.mean())))
            largest_error = max(largest_error, delta)
            assert np.isclose(record.rmse, rmse, rtol=1e-12, atol=1e-10)
            assert np.isclose(record.nasa_mean, cost.mean(), rtol=1e-12, atol=1e-10)
            # Independently aggregate all test windows with NumPy engine bins.
            window_truth = frame.raw_rul.to_numpy(float)
            if truth == "capped125": window_truth = np.minimum(window_truth, 125)
            window_error = frame.pred_rul.to_numpy(float)-window_truth
            window_cost = np.asarray([np.exp(-e/13)-1 if e<0 else np.exp(e/10)-1 for e in window_error])
            _, inverse, count = np.unique(frame.unit_id.to_numpy(), return_inverse=True, return_counts=True)
            engine_squared = np.bincount(inverse, weights=window_error**2)/count
            engine_cost = np.bincount(inverse, weights=window_cost)/count
            for pop, measured_rmse, measured_cost in (
                ("equal_engine_windows", np.sqrt(engine_squared.mean()), engine_cost.mean()),
                ("pooled_windows", np.sqrt(np.mean(window_error**2)), window_cost.mean())):
                item = scores[(scores.run_id==run_id)&(scores.truth==truth)&(scores.population==pop)].iloc[0]
                assert np.isclose(item.rmse, measured_rmse, rtol=1e-12, atol=1e-10)
                assert np.isclose(item.nasa_mean, measured_cost, rtol=1e-12, atol=1e-10)
                largest_error = max(largest_error, abs(item.rmse-measured_rmse),
                                    abs(item.nasa_mean-measured_cost)/max(1, measured_cost))
        curve = pd.read_csv(path / "training_curve.csv")
        assert int(curve.selected.sum()) == 1
    assert len(pd.read_csv(ANALYSIS / "core_engine_bootstrap.csv")) == 216
    assert len(pd.read_csv(ANALYSIS / "paired_factor_contrasts.csv")) == 96
    assert len(pd.read_csv(ANALYSIS / "metric_winner_contexts.csv")) == 144
    assert len(pd.read_csv(ANALYSIS / "uq_stage_engine_bootstrap.csv")) == 108
    archived_path = REPO / "results/runs/unified_v1__fd002__lstm__seed11__point/preds_test.parquet"
    diagnostic_path = HERE / "diagnostics/diagnostic__fd002__s42__lstm__i11__piecewise_125__common_14__scale1/preds_test.parquet"
    original = pd.read_parquet(archived_path)
    reproduced = pd.read_parquet(diagnostic_path)
    paired = original[["unit_id", "cycle", "pred_rul"]].merge(
        reproduced[["unit_id", "cycle", "pred_rul"]], on=["unit_id", "cycle"],
        suffixes=("_archived", "_diagnostic"), validate="one_to_one")
    assert len(paired) == len(original) == len(reproduced)
    diagnostic_difference = float(np.max(np.abs(paired.pred_rul_archived-paired.pred_rul_diagnostic)))
    assert diagnostic_difference == 0.
    result = {"status": "passed", "grid_cells": len(metas), "prediction_tables_checked": 3*len(metas),
              "prediction_rows_checked": rows, "independent_endpoint_scores": 192,
              "independent_window_scores": 384,
              "maximum_rmse_absolute_or_nasa_relative_error": largest_error,
              "data_files_verified": len(data_hashes), "single_training_script_hash": script_hash,
              "frozen_specification_hash": spec_hash, "complete_planned_grid": True,
              "diagnostic_maximum_absolute_prediction_difference": diagnostic_difference,
              "diagnostic_matched_prediction_rows": len(paired),
              "historical_prediction_sha256": hashlib.sha256(archived_path.read_bytes()).hexdigest(),
              "diagnostic_prediction_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
              "scope": "Artifact/data identity, complete grid and endpoint recomputation; not a certificate of scientific novelty or generalization."}
    (HERE / "FINAL_RESULT_VERIFICATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
