"""Validate and summarize all 160 registered unified-grid run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from rul_audit.data.cmapss import sha256_file
from rul_audit.metrics.rul import endpoint_metrics
from rul_audit.protocols.assets import (
    AssetValidationError,
    validate_run_artifacts,
    validate_unified_grid,
)


def _equal_metric(observed: Any, expected: Any) -> bool:
    if isinstance(expected, int):
        return observed == expected
    return math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-9)


def analyze(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / "configs" / "unified_grid.yaml"
    status_path = root / "results" / "UNIFIED_GRID_STATUS.csv"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_unified_grid(config_path, status_path)
    with status_path.open(encoding="utf-8", newline="") as handle:
        status_rows = list(csv.DictReader(handle))

    errors: list[str] = []
    if len(status_rows) != 160:
        errors.append(f"status register has {len(status_rows)} rows, expected 160")
    statuses = pd.Series([row["status"] for row in status_rows]).value_counts().to_dict()
    if statuses != {"completed": 160}:
        errors.append(f"status register is not 160 completed: {statuses}")

    protocol_hash = sha256_file(root / config["protocol"]["document"])
    expected_code_revision = config["registration"]["implementation_sha256"]
    expected_engine_counts = config["controls"]["official_test"]["unit_counts"]
    metrics_rows: list[dict[str, Any]] = []
    calib_checks = 0
    quantile_checks = 0
    for row in status_rows:
        run_id = row["run_id"]
        run_dir = root / row["run_path"]
        try:
            validate_run_artifacts(run_dir, project_root=root)
        except (AssetValidationError, FileNotFoundError, json.JSONDecodeError) as exc:
            errors.append(f"{run_id}: artifact validation failed: {exc}")
            continue
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        expected_fields = {
            "run_id": run_id,
            "dataset": row["dataset"],
            "subset": row["subset"],
            "model": row["model"],
            "seed": int(row["seed"]),
            "output_mode": row["output_mode"],
            "protocol_version": row["protocol_version"],
            "protocol_sha256": protocol_hash,
            "code_revision": expected_code_revision,
            "data_class": "NASA_C-MAPSS_real_unified_registered_cell",
        }
        for field, expected in expected_fields.items():
            if meta.get(field) != expected:
                errors.append(
                    f"{run_id}: meta.{field}={meta.get(field)!r}, expected {expected!r}"
                )

        split_path = root / meta["split_file"]
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        calib_units = {int(value) for value in split_payload["unit_split"]["calib"]}
        calib_table = pq.read_table(run_dir / "preds_calib.parquet", columns=["unit_id"])
        predicted_calib_units = {int(value) for value in calib_table["unit_id"].to_pylist()}
        if predicted_calib_units != calib_units or not calib_units:
            errors.append(
                f"{run_id}: calib unit mismatch expected={sorted(calib_units)} "
                f"observed={sorted(predicted_calib_units)}"
            )
        else:
            calib_checks += 1

        split_frames = {
            split: pq.read_table(run_dir / f"preds_{split}.parquet").to_pandas()
            for split in ("val", "calib", "test")
        }
        if any(float(frame["true_rul"].max()) > 125.0 for frame in split_frames.values()):
            errors.append(f"{run_id}: piecewise_125 truth exceeds 125")
        if meta["output_mode"] == "quantile":
            valid_quantiles = all(
                bool(
                    (
                        (frame["pred_q10"] <= frame["pred_q50"])
                        & (frame["pred_q50"] <= frame["pred_q90"])
                        & ((frame["pred_rul"] - frame["pred_q50"]).abs() <= 1e-8)
                    ).all()
                )
                for frame in split_frames.values()
            )
            if not valid_quantiles:
                errors.append(f"{run_id}: quantile order or q50 alias check failed")
            else:
                quantile_checks += 1

        recomputed = endpoint_metrics(split_frames["test"])
        for name, value in recomputed.items():
            if not _equal_metric(meta["metrics"].get(name), value):
                errors.append(
                    f"{run_id}: metric {name} mismatch meta={meta['metrics'].get(name)} "
                    f"recomputed={value}"
                )
        expected_engines = int(expected_engine_counts[meta["subset"]])
        if recomputed["engine_count"] != expected_engines:
            errors.append(
                f"{run_id}: endpoint count {recomputed['engine_count']} != {expected_engines}"
            )
        metrics_rows.append(
            {
                "run_id": run_id,
                "dataset": meta["dataset"],
                "subset": meta["subset"],
                "model": meta["model"],
                "seed": meta["seed"],
                "output_mode": meta["output_mode"],
                "rmse": recomputed["rmse"],
                "nasa_score": recomputed["nasa_score"],
                "engine_count": recomputed["engine_count"],
                "training_seconds": float(meta["training_seconds"]),
                "val_rows": len(split_frames["val"]),
                "calib_rows": len(split_frames["calib"]),
                "test_rows": len(split_frames["test"]),
            }
        )

    if errors:
        raise RuntimeError("\n".join(errors[:50]))

    metrics = pd.DataFrame(metrics_rows).sort_values(
        ["subset", "model", "seed", "output_mode"]
    )
    means = (
        metrics.groupby(["subset", "model", "output_mode"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_nasa_score=("nasa_score", "mean"),
            sd_nasa_score=("nasa_score", "std"),
            mean_training_seconds=("training_seconds", "mean"),
        )
        .sort_values(["subset", "output_mode", "mean_rmse", "model"])
    )
    paired = metrics.pivot(
        index=["subset", "model", "seed"], columns="output_mode", values=["rmse", "nasa_score"]
    )
    paired.columns = [f"{metric}_{mode}" for metric, mode in paired.columns]
    paired = paired.reset_index()
    paired["quantile_minus_point_rmse"] = paired["rmse_quantile"] - paired["rmse_point"]
    paired["quantile_minus_point_nasa_score"] = (
        paired["nasa_score_quantile"] - paired["nasa_score_point"]
    )

    results_dir = root / "results"
    metrics.to_parquet(results_dir / "UNIFIED_GRID_RUN_METRICS.parquet", index=False)
    means.to_csv(results_dir / "UNIFIED_GRID_MEAN_METRICS.csv", index=False)
    paired.to_csv(results_dir / "UNIFIED_GRID_POINT_QUANTILE_PAIRS.csv", index=False)
    summary = {
        "schema_version": 1,
        "status": "completed_validated",
        "registered_cells": 160,
        "completed_cells": 160,
        "failed_cells": 0,
        "artifact_validation_errors": 0,
        "validated_run_directories": len(metrics),
        "validated_prediction_tables": len(metrics) * 3,
        "recomputed_endpoint_metrics": len(metrics),
        "calib_isolation_checks": calib_checks,
        "quantile_contract_checks": quantile_checks,
        "point_cells": int((metrics["output_mode"] == "point").sum()),
        "quantile_cells": int((metrics["output_mode"] == "quantile").sum()),
        "protocol_sha256": protocol_hash,
        "implementation_sha256": expected_code_revision,
        "total_training_seconds": float(metrics["training_seconds"].sum()),
        "outputs": [
            "results/UNIFIED_GRID_RUN_METRICS.parquet",
            "results/UNIFIED_GRID_MEAN_METRICS.csv",
            "results/UNIFIED_GRID_POINT_QUANTILE_PAIRS.csv",
        ],
    }
    (results_dir / "UNIFIED_GRID_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
