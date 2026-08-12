from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rul_audit.protocols.assets import (
    AssetValidationError,
    validate_run_artifacts,
    validate_unified_grid,
    validate_unit_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unified_grid_has_160_unique_cells() -> None:
    summary = validate_unified_grid(
        PROJECT_ROOT / "configs" / "unified_grid.yaml",
        PROJECT_ROOT / "results" / "UNIFIED_GRID_STATUS.csv",
    )

    assert summary == {"expected_cells": 160, "status_rows": 160}
    with (PROJECT_ROOT / "results" / "UNIFIED_GRID_STATUS.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["status"] for row in rows} == {"completed"}
    assert {row["notes"] for row in rows} == {"artifact_validation_pass"}


def test_unit_split_accepts_disjoint_engine_units(tmp_path: Path) -> None:
    split_path = tmp_path / "FD001_seed42.json"
    split_path.write_text(
        json.dumps(
            {
                "dataset": "FD001",
                "seed": 42,
                "unit_split": {"train": [1, 2, 3], "val": [4], "calib": [5]},
                "fractions": {"train": 0.70, "val": 0.15, "calib": 0.15},
                "allocation_unit": "engine_unit",
                "calib_isolation": "never_used_for_training_or_tuning",
                "status": "ready",
            }
        ),
        encoding="utf-8",
    )

    assert validate_unit_split(split_path) == {"train": 3, "val": 1, "calib": 1}


def test_unit_split_rejects_calibration_overlap(tmp_path: Path) -> None:
    split_path = tmp_path / "FD001_seed42.json"
    split_path.write_text(
        json.dumps(
            {
                "dataset": "FD001",
                "seed": 42,
                "unit_split": {"train": [1, 2], "val": [3], "calib": [2, 4]},
                "fractions": {"train": 0.70, "val": 0.15, "calib": 0.15},
                "allocation_unit": "engine_unit",
                "calib_isolation": "never_used_for_training_or_tuning",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="overlap between train and calib"):
        validate_unit_split(split_path)


def _write_run(tmp_path: Path, output_mode: str, *, omit_q90: bool = False) -> Path:
    run_id = f"test__seed11__{output_mode}"
    run_dir = tmp_path / run_id
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.bin").write_bytes(b"checkpoint")

    meta = {
        "run_id": run_id,
        "dataset": "C-MAPSS",
        "subset": "FD001",
        "model": "lstm",
        "output_mode": output_mode,
        "seed": 11,
        "protocol_version": "unified_v1",
        "protocol_sha256": "a" * 64,
        "split_file": "configs/splits/FD001_seed42.json",
        "split_sha256": "b" * 64,
        "code_revision": "test-revision",
        "started_at": "2026-08-11T00:00:00+00:00",
        "finished_at": "2026-08-11T00:01:00+00:00",
        "training_seconds": 60.0,
        "status": "completed",
        "metrics": {"rmse": 10.0},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    columns: dict[str, pa.Array] = {
        "unit_id": pa.array([1, 1], type=pa.int64()),
        "cycle": pa.array([30, 31], type=pa.int64()),
        "true_rul": pa.array([70.0, 69.0], type=pa.float64()),
        "pred_rul": pa.array([69.0, 68.0], type=pa.float64()),
    }
    if output_mode == "quantile":
        columns.update(
            {
                "pred_q10": pa.array([65.0, 64.0], type=pa.float64()),
                "pred_q50": pa.array([69.0, 68.0], type=pa.float64()),
            }
        )
        if not omit_q90:
            columns["pred_q90"] = pa.array([73.0, 72.0], type=pa.float64())

    table = pa.table(columns)
    for split in ("val", "calib", "test"):
        pq.write_table(table, run_dir / f"preds_{split}.parquet")
    return run_dir


@pytest.mark.parametrize("output_mode", ["point", "quantile"])
def test_run_artifact_schema_accepts_complete_runs(tmp_path: Path, output_mode: str) -> None:
    run_dir = _write_run(tmp_path, output_mode)

    result = validate_run_artifacts(run_dir)

    assert result["run_id"] == run_dir.name
    assert result["prediction_tables"] == 3


def test_quantile_run_rejects_missing_quantile_column(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "quantile", omit_q90=True)

    with pytest.raises(AssetValidationError, match="pred_q90"):
        validate_run_artifacts(run_dir)
