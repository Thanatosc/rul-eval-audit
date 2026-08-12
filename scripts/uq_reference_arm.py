"""Register, execute, validate, and summarize the frozen ``uq_ref_v1`` arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REGISTRATION_ID = "uq_ref_v1"
MASTER_SEED = 20260812
STATUS_COLUMNS = (
    "uq_run_id",
    "panel",
    "dataset",
    "subset",
    "model",
    "seed",
    "method",
    "rul_label",
    "sensor_set",
    "source_run_id",
    "source_output_mode",
    "status",
    "output_path",
    "notes",
)
FROZEN_COLUMNS = STATUS_COLUMNS[:-3] + ("output_path",)


@dataclass(frozen=True)
class UQCell:
    uq_run_id: str
    panel: str
    dataset: str
    subset: str
    model: str
    seed: int
    method: str
    rul_label: str
    sensor_set: str
    source_run_id: str
    source_output_mode: str
    output_path: str

    def frozen_row(self) -> dict[str, str]:
        row = asdict(self)
        return {column: str(row[column]) for column in FROZEN_COLUMNS}

    def status_row(self) -> dict[str, str]:
        row = self.frozen_row()
        row.update({"status": "pending", "notes": "registered_pending"})
        return {column: row[column] for column in STATUS_COLUMNS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("UQ reference config must be a YAML mapping")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _kill_cell(row: dict[str, str]) -> UQCell:
    source_run_id = row["run_id"]
    suffix = source_run_id.removeprefix("kill_v1__")
    uq_run_id = f"{REGISTRATION_ID}__kill__{suffix}__residual_split_cp"
    return UQCell(
        uq_run_id=uq_run_id,
        panel="kill_protocol_sensitivity",
        dataset=row["dataset"],
        subset=row["dataset"],
        model=row["model"],
        seed=int(row["seed"]),
        method="residual_split_cp",
        rul_label=row["rul_label"],
        sensor_set=row["sensor_set"],
        source_run_id=source_run_id,
        source_output_mode="point",
        output_path=f"results/uq_reference/{uq_run_id}",
    )


def _unified_cell(row: dict[str, str]) -> UQCell:
    source_run_id = row["run_id"]
    method = "residual_split_cp" if row["output_mode"] == "point" else "cqr"
    suffix = source_run_id.removeprefix("unified_v1__")
    suffix = suffix.removesuffix(f"__{row['output_mode']}")
    uq_run_id = f"{REGISTRATION_ID}__unified__{suffix}__{method}"
    return UQCell(
        uq_run_id=uq_run_id,
        panel="unified_reference",
        dataset=row["dataset"],
        subset=row["subset"],
        model=row["model"],
        seed=int(row["seed"]),
        method=method,
        rul_label="piecewise_125",
        sensor_set="common_14",
        source_run_id=source_run_id,
        source_output_mode=row["output_mode"],
        output_path=f"results/uq_reference/{uq_run_id}",
    )


def expand_cells(project_root: Path, config: dict[str, Any]) -> list[UQCell]:
    root = project_root.resolve()
    kill_rows = _read_csv(root / config["panels"]["kill_protocol_sensitivity"]["source_register"])
    unified_rows = _read_csv(root / config["panels"]["unified_reference"]["source_register"])
    if len(kill_rows) != 120 or {row["status"] for row in kill_rows} != {"completed"}:
        raise RuntimeError("Kill Test source register must contain 120 completed rows")
    if len(unified_rows) != 160 or {row["status"] for row in unified_rows} != {"completed"}:
        raise RuntimeError("unified source register must contain 160 completed rows")
    cells = [_kill_cell(row) for row in kill_rows] + [_unified_cell(row) for row in unified_rows]
    expected = int(config["registration"]["expected_cells"])
    if len(cells) != expected or len({cell.uq_run_id for cell in cells}) != expected:
        raise RuntimeError(f"UQ register expansion is not {expected} unique cells")
    counts = pd.Series([(cell.panel, cell.method) for cell in cells]).value_counts().to_dict()
    required = {
        ("kill_protocol_sensitivity", "residual_split_cp"): 120,
        ("unified_reference", "residual_split_cp"): 80,
        ("unified_reference", "cqr"): 80,
    }
    if counts != required:
        raise RuntimeError(f"unexpected UQ panel expansion: {counts}")
    return cells


def write_registers(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_config(root / "configs" / "uq_reference_arm.yaml")
    cells = expand_cells(root, config)
    frozen_path = root / config["registration"]["frozen_cell_register"]
    status_path = root / config["registration"]["mutable_status_register"]
    if frozen_path.exists() or status_path.exists():
        raise FileExistsError("refusing to overwrite an existing UQ register")
    _write_csv(frozen_path, [cell.frozen_row() for cell in cells], FROZEN_COLUMNS)
    _write_csv(status_path, [cell.status_row() for cell in cells], STATUS_COLUMNS)
    return {
        "registered_cells": len(cells),
        "frozen_cell_register_sha256": sha256_file(frozen_path),
        "status": "pending",
    }


def _validate_frozen_hashes(root: Path, config: dict[str, Any]) -> None:
    registration = config["registration"]
    if config.get("status") != "frozen_authorized_for_execution":
        raise RuntimeError("uq_ref_v1 is not in the frozen authorized state")
    expected_script = registration["implementation_sha256"]
    observed_script = sha256_file(root / registration["implementation_file"])
    if observed_script != expected_script:
        raise RuntimeError("UQ implementation hash differs from the frozen registration")
    expected_schema = registration["output_schema_sha256"]
    observed_schema = sha256_file(root / registration["output_schema"])
    if observed_schema != expected_schema:
        raise RuntimeError("UQ output schema hash differs from the frozen registration")
    frozen_path = root / registration["frozen_cell_register"]
    if sha256_file(frozen_path) != registration["frozen_cell_register_sha256"]:
        raise RuntimeError("UQ frozen cell-register hash mismatch")
    for relative, expected in registration["upstream_hashes"].items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise RuntimeError(f"frozen upstream hash mismatch for {relative}")


def validate_registration(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_config(root / "configs" / "uq_reference_arm.yaml")
    _validate_frozen_hashes(root, config)
    expected_cells = expand_cells(root, config)
    frozen_rows = _read_csv(root / config["registration"]["frozen_cell_register"])
    status_rows = _read_csv(root / config["registration"]["mutable_status_register"])
    expected_rows = [cell.frozen_row() for cell in expected_cells]
    if frozen_rows != expected_rows:
        raise RuntimeError("frozen UQ cell register does not match deterministic expansion")
    if len(status_rows) != len(expected_rows):
        raise RuntimeError("mutable UQ status register has the wrong row count")
    for frozen, status in zip(frozen_rows, status_rows, strict=True):
        if {column: status[column] for column in FROZEN_COLUMNS} != frozen:
            raise RuntimeError("mutable UQ status register drifted from frozen cells")
    allowed = {"pending", "running", "completed", "failed"}
    if not {row["status"] for row in status_rows} <= allowed:
        raise RuntimeError("mutable UQ status register contains an invalid state")
    return {
        "registered_cells": len(expected_cells),
        "status_counts": pd.Series([row["status"] for row in status_rows]).value_counts().to_dict(),
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a non-empty finite vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha))
    if rank > len(values):
        raise ValueError("finite-sample conformal rank exceeds the calibration sample size")
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def _validate_source_frame(frame: pd.DataFrame, method: str, split: str) -> None:
    required = {"unit_id", "cycle", "true_rul", "pred_rul"}
    if method == "cqr":
        required.update({"pred_q10", "pred_q50", "pred_q90"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split} source prediction is missing {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f"{split} source prediction is empty or contains nulls")
    numeric = [column for column in required if column != "unit_id"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{split} source prediction contains non-finite values")
    if frame.duplicated(["unit_id", "cycle"]).any():
        raise ValueError(f"{split} source prediction has duplicate unit/cycle rows")
    if method == "cqr":
        ordered = (frame["pred_q10"] <= frame["pred_q50"]) & (
            frame["pred_q50"] <= frame["pred_q90"]
        )
        if not bool(ordered.all()):
            raise ValueError(f"{split} source prediction violates quantile order")


def build_intervals(
    calib: pd.DataFrame,
    test: pd.DataFrame,
    *,
    method: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, int]:
    _validate_source_frame(calib, method, "calib")
    _validate_source_frame(test, method, "test")
    if method == "residual_split_cp":
        scores = (calib["true_rul"] - calib["pred_rul"]).abs().to_numpy(dtype=float)
    elif method == "cqr":
        scores = np.maximum(
            calib["pred_q10"].to_numpy(dtype=float) - calib["true_rul"].to_numpy(dtype=float),
            calib["true_rul"].to_numpy(dtype=float) - calib["pred_q90"].to_numpy(dtype=float),
        )
    else:
        raise ValueError(f"unknown UQ method: {method}")
    qhat, rank = conformal_quantile(scores, alpha)
    calibration = calib[["unit_id", "cycle", "true_rul"]].copy()
    calibration["score"] = scores
    intervals = test[["unit_id", "cycle", "true_rul", "pred_rul"]].copy()
    if method == "residual_split_cp":
        intervals["lower"] = intervals["pred_rul"] - qhat
        intervals["upper"] = intervals["pred_rul"] + qhat
    else:
        intervals["base_lower"] = test["pred_q10"].to_numpy(dtype=float)
        intervals["base_upper"] = test["pred_q90"].to_numpy(dtype=float)
        intervals["base_covered"] = (
            (intervals["true_rul"] >= intervals["base_lower"])
            & (intervals["true_rul"] <= intervals["base_upper"])
        )
        intervals["base_interval_width"] = intervals["base_upper"] - intervals["base_lower"]
        intervals["lower"] = intervals["base_lower"] - qhat
        intervals["upper"] = intervals["base_upper"] + qhat
    if not np.isfinite(intervals[["lower", "upper"]].to_numpy(dtype=float)).all():
        raise ValueError("computed interval bounds are non-finite")
    if not bool((intervals["lower"] <= intervals["upper"]).all()):
        raise ValueError("computed interval has lower bound above upper bound")
    intervals["covered"] = (
        (intervals["true_rul"] >= intervals["lower"])
        & (intervals["true_rul"] <= intervals["upper"])
    )
    intervals["interval_width"] = intervals["upper"] - intervals["lower"]
    intervals = intervals.sort_values(["unit_id", "cycle"], kind="stable").reset_index(drop=True)
    intervals["is_endpoint"] = False
    endpoint_index = intervals.groupby("unit_id", sort=False).tail(1).index
    intervals.loc[endpoint_index, "is_endpoint"] = True
    return calibration, intervals, qhat, rank


def metric_bundle(intervals: pd.DataFrame, *, base: bool = False) -> dict[str, float | int]:
    prefix = "base_" if base else ""
    covered = f"{prefix}covered"
    width = f"{prefix}interval_width"
    if covered not in intervals or width not in intervals:
        raise ValueError(f"interval table does not contain {prefix or 'conformal '}metrics")
    engines = intervals.groupby("unit_id", sort=False).agg(
        coverage=(covered, "mean"), mean_width=(width, "mean")
    )
    endpoints = intervals.loc[intervals["is_endpoint"]]
    return {
        "test_windows": len(intervals),
        "test_engines": int(intervals["unit_id"].nunique()),
        "pooled_window_coverage": float(intervals[covered].mean()),
        "pooled_window_mean_interval_width": float(intervals[width].mean()),
        "engine_balanced_coverage": float(engines["coverage"].mean()),
        "engine_balanced_mean_interval_width": float(engines["mean_width"].mean()),
        "endpoint_coverage": float(endpoints[covered].mean()),
        "endpoint_mean_interval_width": float(endpoints[width].mean()),
    }


def _status_path(root: Path, config: dict[str, Any]) -> Path:
    return root / config["registration"]["mutable_status_register"]


def _registered_status(root: Path, config: dict[str, Any], uq_run_id: str) -> dict[str, str]:
    matches = [row for row in _read_csv(_status_path(root, config)) if row["uq_run_id"] == uq_run_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one UQ status row for {uq_run_id}")
    return matches[0]


def _update_status(root: Path, config: dict[str, Any], uq_run_id: str, status: str, notes: str) -> None:
    path = _status_path(root, config)
    rows = _read_csv(path)
    matched = False
    for row in rows:
        if row["uq_run_id"] == uq_run_id:
            row["status"] = status
            row["notes"] = notes
            matched = True
    if not matched:
        raise RuntimeError(f"UQ status row not found: {uq_run_id}")
    _write_csv(path, rows, STATUS_COLUMNS)


def _source_run_dir(root: Path, cell: UQCell) -> Path:
    return root / "results" / "runs" / cell.source_run_id


def _validate_source_meta(cell: UQCell, source_dir: Path) -> dict[str, Any]:
    meta = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
    expected = {
        "run_id": cell.source_run_id,
        "model": cell.model,
        "seed": cell.seed,
        "output_mode": cell.source_output_mode,
        "status": "completed",
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"source meta {field}={meta.get(field)!r}, expected {value!r}")
    observed_subset = meta.get("subset", meta.get("dataset"))
    if observed_subset != cell.subset:
        raise ValueError("source meta subset does not match the frozen UQ cell")
    return meta


def run_cell(project_root: Path, cell: UQCell, *, authorize_real_execution: bool = False) -> dict[str, Any]:
    if not authorize_real_execution:
        raise PermissionError("real UQ post-processing is gated")
    root = project_root.resolve()
    config = load_config(root / "configs" / "uq_reference_arm.yaml")
    validate_registration(root)
    registered = {candidate.uq_run_id: candidate for candidate in expand_cells(root, config)}
    if registered.get(cell.uq_run_id) != cell:
        raise ValueError("UQ cell does not match the frozen register")
    status_row = _registered_status(root, config, cell.uq_run_id)
    if status_row["status"] != "pending":
        raise RuntimeError("registered UQ cell is not pending; retry refused")
    output_dir = root / cell.output_path
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing UQ output: {output_dir}")
    source_dir = _source_run_dir(root, cell)
    source_meta = _validate_source_meta(cell, source_dir)
    calib_path = source_dir / "preds_calib.parquet"
    test_path = source_dir / "preds_test.parquet"
    calib = pd.read_parquet(calib_path)
    test = pd.read_parquet(test_path)
    alpha = float(config["calibration"]["alpha"])
    calibration, intervals, qhat, rank = build_intervals(
        calib, test, method=cell.method, alpha=alpha
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "registration_id": REGISTRATION_ID,
        "uq_run_id": cell.uq_run_id,
        "panel": cell.panel,
        "dataset": cell.dataset,
        "subset": cell.subset,
        "model": cell.model,
        "seed": cell.seed,
        "method": cell.method,
        "rul_label": cell.rul_label,
        "sensor_set": cell.sensor_set,
        "source_run_id": cell.source_run_id,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "calibration_windows": len(calibration),
        "calibration_engines": int(calibration["unit_id"].nunique()),
        "conformal_rank": rank,
        "qhat": qhat,
        "metrics": metric_bundle(intervals),
        "validity_label": config["calibration"]["validity_label"],
    }
    if cell.method == "cqr":
        summary["base_quantile_metrics"] = metric_bundle(intervals, base=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration.to_parquet(output_dir / "calibration_scores.parquet", index=False)
    intervals.to_parquet(output_dir / "intervals_test.parquet", index=False)
    registration_path = root / "configs" / "uq_reference_arm.yaml"
    meta = {
        "schema_version": 1,
        **cell.frozen_row(),
        "status": "completed",
        "registration_sha256": sha256_file(registration_path),
        "implementation_sha256": config["registration"]["implementation_sha256"],
        "output_schema_sha256": config["registration"]["output_schema_sha256"],
        "frozen_cell_register_sha256": config["registration"]["frozen_cell_register_sha256"],
        "source_meta_sha256": sha256_file(source_dir / "meta.json"),
        "source_calib_sha256": sha256_file(calib_path),
        "source_test_sha256": sha256_file(test_path),
        "source_protocol_sha256": source_meta["protocol_sha256"],
        "source_split_sha256": source_meta["split_sha256"],
        "source_code_revision": source_meta["code_revision"],
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_completed_cell(root, cell)
    _update_status(root, config, cell.uq_run_id, "completed", "artifact_validation_pass")
    return summary


def validate_completed_cell(project_root: Path, cell: UQCell) -> dict[str, Any]:
    root = project_root.resolve()
    output_dir = root / cell.output_path
    config = load_config(root / "configs" / "uq_reference_arm.yaml")
    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    calibration = pd.read_parquet(output_dir / "calibration_scores.parquet")
    intervals = pd.read_parquet(output_dir / "intervals_test.parquet")
    if meta.get("uq_run_id") != cell.uq_run_id or summary.get("uq_run_id") != cell.uq_run_id:
        raise ValueError("UQ output identity mismatch")
    required = {
        "unit_id",
        "cycle",
        "true_rul",
        "pred_rul",
        "lower",
        "upper",
        "covered",
        "interval_width",
        "is_endpoint",
    }
    if cell.method == "cqr":
        required.update({"base_lower", "base_upper", "base_covered", "base_interval_width"})
    if required - set(intervals.columns):
        raise ValueError("UQ interval artifact is missing required columns")
    if intervals.empty or intervals[list(required)].isna().any().any():
        raise ValueError("UQ interval artifact is empty or contains nulls")
    if not bool((intervals["lower"] <= intervals["upper"]).all()):
        raise ValueError("UQ interval ordering validation failed")
    expected_covered = (intervals["true_rul"] >= intervals["lower"]) & (
        intervals["true_rul"] <= intervals["upper"]
    )
    if not bool((intervals["covered"] == expected_covered).all()):
        raise ValueError("UQ coverage indicator validation failed")
    if not np.allclose(
        intervals["interval_width"], intervals["upper"] - intervals["lower"], atol=1e-10
    ):
        raise ValueError("UQ interval width validation failed")
    if not bool((intervals.groupby("unit_id")["is_endpoint"].sum() == 1).all()):
        raise ValueError("UQ endpoint marker validation failed")
    if cell.method == "cqr":
        if not bool((intervals["base_lower"] <= intervals["base_upper"]).all()):
            raise ValueError("base quantile interval ordering validation failed")
        expected_base_covered = (
            (intervals["true_rul"] >= intervals["base_lower"])
            & (intervals["true_rul"] <= intervals["base_upper"])
        )
        if not bool((intervals["base_covered"] == expected_base_covered).all()):
            raise ValueError("base quantile coverage indicator validation failed")
        if not np.allclose(
            intervals["base_interval_width"],
            intervals["base_upper"] - intervals["base_lower"],
            atol=1e-10,
        ):
            raise ValueError("base quantile interval width validation failed")
    source_dir = _source_run_dir(root, cell)
    source_calib = pd.read_parquet(source_dir / "preds_calib.parquet")
    source_test = pd.read_parquet(source_dir / "preds_test.parquet")
    expected_calibration, expected_intervals, qhat, rank = build_intervals(
        source_calib,
        source_test,
        method=cell.method,
        alpha=float(config["calibration"]["alpha"]),
    )
    pd.testing.assert_frame_equal(
        calibration.reset_index(drop=True),
        expected_calibration.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        intervals.reset_index(drop=True),
        expected_intervals.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if not math.isclose(qhat, float(summary["qhat"]), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("UQ conformal quantile validation failed")
    if rank != int(summary["conformal_rank"]):
        raise ValueError("UQ conformal rank validation failed")
    observed = metric_bundle(intervals)
    for name, value in observed.items():
        expected = summary["metrics"][name]
        if isinstance(value, int):
            if value != expected:
                raise ValueError(f"UQ metric mismatch: {name}")
        elif not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"UQ metric mismatch: {name}")
    return {"uq_run_id": cell.uq_run_id, "test_windows": len(intervals)}


def run_pending(project_root: Path, *, max_cells: int | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_config(root / "configs" / "uq_reference_arm.yaml")
    validate_registration(root)
    cells = {cell.uq_run_id: cell for cell in expand_cells(root, config)}
    status_rows = _read_csv(_status_path(root, config))
    pending = [row["uq_run_id"] for row in status_rows if row["status"] == "pending"]
    if max_cells is not None:
        if max_cells <= 0:
            raise ValueError("max_cells must be positive")
        pending = pending[:max_cells]
    completed = 0
    for index, uq_run_id in enumerate(pending, start=1):
        print(f"START {index}/{len(pending)} {uq_run_id}", flush=True)
        try:
            run_cell(root, cells[uq_run_id], authorize_real_execution=True)
        except Exception as exc:
            _update_status(root, config, uq_run_id, "failed", f"{type(exc).__name__}: {exc}")
            print(f"FAILED {uq_run_id}: {type(exc).__name__}: {exc}", flush=True)
            raise
        completed += 1
        if index == 1 or index % 20 == 0 or index == len(pending):
            print(f"COMPLETED {index}/{len(pending)} {uq_run_id}", flush=True)
    return {"completed_this_invocation": completed}


def _engine_metrics(frame: pd.DataFrame, *, base: bool = False) -> pd.DataFrame:
    prefix = "base_" if base else ""
    return (
        frame.groupby("unit_id", as_index=False)
        .agg(
            coverage=(f"{prefix}covered", "mean"),
            mean_width=(f"{prefix}interval_width", "mean"),
        )
        .sort_values("unit_id")
    )


def bootstrap_mean_ci(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("engine bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(resamples, len(data)))
    means = data[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _load_interval(root: Path, cell: UQCell) -> pd.DataFrame:
    return pd.read_parquet(root / cell.output_path / "intervals_test.parquet")


def _paired_comparison(
    root: Path,
    pairs: list[tuple[UQCell, UQCell]],
    *,
    comparison_id: str,
    comparison_type: str,
    level_a: str,
    level_b: str,
    base_a: bool = False,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    engine_rows: list[pd.DataFrame] = []
    seed_coverage_deltas: list[float] = []
    seed_width_deltas: list[float] = []
    for left, right in pairs:
        left_frame = _load_interval(root, left)
        right_frame = left_frame if left.uq_run_id == right.uq_run_id else _load_interval(root, right)
        left_engines = _engine_metrics(left_frame, base=base_a).rename(
            columns={"coverage": "coverage_a", "mean_width": "mean_width_a"}
        )
        right_engines = _engine_metrics(right_frame).rename(
            columns={"coverage": "coverage_b", "mean_width": "mean_width_b"}
        )
        merged = left_engines.merge(right_engines, on="unit_id", validate="one_to_one")
        if len(merged) != int(left_frame["unit_id"].nunique()):
            raise RuntimeError("paired UQ comparison has mismatched test engines")
        merged["seed"] = left.seed
        merged["coverage_delta"] = merged["coverage_b"] - merged["coverage_a"]
        merged["width_delta"] = merged["mean_width_b"] - merged["mean_width_a"]
        engine_rows.append(merged)
        seed_coverage_deltas.append(float(merged["coverage_delta"].mean()))
        seed_width_deltas.append(float(merged["width_delta"].mean()))
    all_engines = pd.concat(engine_rows, ignore_index=True)
    if all_engines["seed"].nunique() != 5:
        raise RuntimeError("registered UQ comparison requires five seeds")
    averaged = (
        all_engines.groupby("unit_id", as_index=False)
        .agg(
            coverage_a=("coverage_a", "mean"),
            coverage_b=("coverage_b", "mean"),
            mean_width_a=("mean_width_a", "mean"),
            mean_width_b=("mean_width_b", "mean"),
            coverage_delta=("coverage_delta", "mean"),
            width_delta=("width_delta", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    if set(averaged["seed_count"]) != {5}:
        raise RuntimeError("not every test engine is represented in all five seeds")
    coverage_ci = bootstrap_mean_ci(
        averaged["coverage_delta"].to_numpy(),
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    width_ci = bootstrap_mean_ci(
        averaged["width_delta"].to_numpy(),
        resamples=bootstrap_resamples,
        seed=bootstrap_seed + 1,
    )
    coverage_a = float(averaged["coverage_a"].mean())
    coverage_b = float(averaged["coverage_b"].mean())
    width_a = float(averaged["mean_width_a"].mean())
    width_b = float(averaged["mean_width_b"].mean())
    return {
        "comparison_id": comparison_id,
        "comparison_type": comparison_type,
        "level_a": level_a,
        "level_b": level_b,
        "test_engines": len(averaged),
        "seed_count": 5,
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "coverage_delta_b_minus_a": coverage_b - coverage_a,
        "coverage_delta_ci95_low": coverage_ci[0],
        "coverage_delta_ci95_high": coverage_ci[1],
        "absolute_coverage_error_change": abs(coverage_b - 0.90) - abs(coverage_a - 0.90),
        "mean_width_a": width_a,
        "mean_width_b": width_b,
        "mean_width_delta_b_minus_a": width_b - width_a,
        "mean_width_delta_ci95_low": width_ci[0],
        "mean_width_delta_ci95_high": width_ci[1],
        "positive_coverage_delta_seeds": int(np.sum(np.asarray(seed_coverage_deltas) > 0)),
        "positive_width_delta_seeds": int(np.sum(np.asarray(seed_width_deltas) > 0)),
    }


def _cell_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: summary[key]
        for key in (
            "uq_run_id",
            "panel",
            "dataset",
            "subset",
            "model",
            "seed",
            "method",
            "rul_label",
            "sensor_set",
            "source_run_id",
            "calibration_windows",
            "calibration_engines",
            "conformal_rank",
            "qhat",
        )
    }
    row.update(summary["metrics"])
    base = summary.get("base_quantile_metrics", {})
    row.update({f"base_{key}": value for key, value in base.items()})
    return row


def _plot_results(
    root: Path,
    cells: pd.DataFrame,
    protocol: pd.DataFrame,
    output_path: Path,
) -> None:
    base = cells.loc[cells["method"] == "cqr", "base_engine_balanced_coverage"].dropna()
    residual = cells.loc[
        (cells["panel"] == "unified_reference") & (cells["method"] == "residual_split_cp"),
        "engine_balanced_coverage",
    ]
    cqr = cells.loc[cells["method"] == "cqr", "engine_balanced_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].boxplot(
        [base.to_numpy(), residual.to_numpy(), cqr.to_numpy()],
        tick_labels=["Base q10-q90", "Residual CP", "CQR"],
        showmeans=True,
    )
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.2, label="Nominal 90%")
    axes[0].set_ylabel("Engine-balanced empirical coverage")
    axes[0].set_title("Unified reference panel (80 cells per method)")
    axes[0].legend(frameon=False)
    colors = protocol["effect_type"].map({"rul_label": "#d55e00", "sensor_set": "#0072b2"})
    axes[1].scatter(
        protocol["coverage_delta_b_minus_a"] * 100.0,
        protocol["mean_width_delta_b_minus_a"],
        c=colors,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].axvline(0.0, color="grey", linewidth=0.8)
    axes[1].axhline(0.0, color="grey", linewidth=0.8)
    axes[1].set_xlabel("Coverage shift, B - A (percentage points)")
    axes[1].set_ylabel("Mean-width shift, B - A")
    axes[1].set_title("Registered protocol contrasts")
    for effect_type, color, label in (
        ("rul_label", "#d55e00", "RUL label"),
        ("sensor_set", "#0072b2", "Sensor set"),
    ):
        axes[1].scatter([], [], color=color, label=label)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = load_config(root / "configs" / "uq_reference_arm.yaml")
    registration = validate_registration(root)
    if registration["status_counts"] != {"completed": 280}:
        raise RuntimeError(f"UQ analysis requires 280 completed cells: {registration['status_counts']}")
    cells = expand_cells(root, config)
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        validate_completed_cell(root, cell)
        summary = json.loads((root / cell.output_path / "summary.json").read_text(encoding="utf-8"))
        summaries.append(_cell_summary_row(summary))
    cell_frame = pd.DataFrame(summaries).sort_values(
        ["panel", "subset", "model", "seed", "method", "rul_label", "sensor_set"]
    )
    lookup = {
        (
            cell.panel,
            cell.subset,
            cell.model,
            cell.seed,
            cell.method,
            cell.rul_label,
            cell.sensor_set,
        ): cell
        for cell in cells
    }
    bootstrap = config["evaluation"]["bootstrap"]
    resamples = int(bootstrap["resamples"])
    protocol_rows: list[dict[str, Any]] = []
    counter = 0
    for dataset in ("FD001", "FD004"):
        for model in ("lstm", "cnn_1d", "lightgbm"):
            for sensor_set in ("all_21", "common_14"):
                pairs = [
                    (
                        lookup[("kill_protocol_sensitivity", dataset, model, seed, "residual_split_cp", "piecewise_125", sensor_set)],
                        lookup[("kill_protocol_sensitivity", dataset, model, seed, "residual_split_cp", "linear_uncapped", sensor_set)],
                    )
                    for seed in (11, 23, 37, 53, 71)
                ]
                row = _paired_comparison(
                    root,
                    pairs,
                    comparison_id=f"label__{dataset}__{model}__{sensor_set}",
                    comparison_type="protocol",
                    level_a="piecewise_125",
                    level_b="linear_uncapped",
                    bootstrap_resamples=resamples,
                    bootstrap_seed=MASTER_SEED + counter * 10,
                )
                row.update({"effect_type": "rul_label", "dataset": dataset, "model": model, "held_level": sensor_set})
                protocol_rows.append(row)
                counter += 1
            for rul_label in ("piecewise_125", "linear_uncapped"):
                pairs = [
                    (
                        lookup[("kill_protocol_sensitivity", dataset, model, seed, "residual_split_cp", rul_label, "all_21")],
                        lookup[("kill_protocol_sensitivity", dataset, model, seed, "residual_split_cp", rul_label, "common_14")],
                    )
                    for seed in (11, 23, 37, 53, 71)
                ]
                row = _paired_comparison(
                    root,
                    pairs,
                    comparison_id=f"sensor__{dataset}__{model}__{rul_label}",
                    comparison_type="protocol",
                    level_a="all_21",
                    level_b="common_14",
                    bootstrap_resamples=resamples,
                    bootstrap_seed=MASTER_SEED + counter * 10,
                )
                row.update({"effect_type": "sensor_set", "dataset": dataset, "model": model, "held_level": rul_label})
                protocol_rows.append(row)
                counter += 1
    protocol_frame = pd.DataFrame(protocol_rows).sort_values(
        ["effect_type", "dataset", "model", "held_level"]
    )
    method_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        for model in ("lstm", "cnn_1d", "transformer", "lightgbm"):
            pairs = [
                (
                    lookup[("unified_reference", subset, model, seed, "residual_split_cp", "piecewise_125", "common_14")],
                    lookup[("unified_reference", subset, model, seed, "cqr", "piecewise_125", "common_14")],
                )
                for seed in (11, 23, 37, 53, 71)
            ]
            row = _paired_comparison(
                root,
                pairs,
                comparison_id=f"method__{subset}__{model}",
                comparison_type="method_bundle",
                level_a="residual_split_cp_point_model",
                level_b="cqr_quantile_model",
                bootstrap_resamples=resamples,
                bootstrap_seed=MASTER_SEED + counter * 10,
            )
            row.update({"subset": subset, "model": model})
            method_rows.append(row)
            counter += 1
            same_pairs = [(right, right) for _, right in pairs]
            base_row = _paired_comparison(
                root,
                same_pairs,
                comparison_id=f"cqr_calibration__{subset}__{model}",
                comparison_type="within_quantile_model_calibration",
                level_a="uncalibrated_q10_q90",
                level_b="cqr",
                base_a=True,
                bootstrap_resamples=resamples,
                bootstrap_seed=MASTER_SEED + counter * 10,
            )
            base_row.update({"subset": subset, "model": model})
            base_rows.append(base_row)
            counter += 1
    method_frame = pd.DataFrame(method_rows).sort_values(["subset", "model"])
    base_frame = pd.DataFrame(base_rows).sort_values(["subset", "model"])
    output_root = root / "results" / "uq_reference"
    output_root.mkdir(parents=True, exist_ok=True)
    cell_path = output_root / "UQ_REFERENCE_CELL_SUMMARY.csv"
    protocol_path = output_root / "UQ_PROTOCOL_CONTRASTS.csv"
    method_path = output_root / "UQ_METHOD_COMPARISONS.csv"
    base_path = output_root / "UQ_CQR_CALIBRATION_EFFECTS.csv"
    cell_frame.to_csv(cell_path, index=False)
    protocol_frame.to_csv(protocol_path, index=False)
    method_frame.to_csv(method_path, index=False)
    base_frame.to_csv(base_path, index=False)
    figure_path = root / config["outputs"]["figure"]
    _plot_results(root, cell_frame, protocol_frame, figure_path)
    summary = {
        "schema_version": 1,
        "registration_id": REGISTRATION_ID,
        "status": "completed_validated",
        "registered_cells": 280,
        "completed_cells": 280,
        "failed_cells": 0,
        "validated_cells": 280,
        "panel_counts": {
            "kill_protocol_sensitivity_residual_split_cp": 120,
            "unified_reference_residual_split_cp": 80,
            "unified_reference_cqr": 80,
        },
        "protocol_contrasts": 24,
        "method_comparisons": 16,
        "cqr_calibration_comparisons": 16,
        "protocol_contrasts_abs_coverage_shift_ge_0_03": int(
            (protocol_frame["coverage_delta_b_minus_a"].abs() >= 0.03).sum()
        ),
        "maximum_absolute_protocol_coverage_shift": float(
            protocol_frame["coverage_delta_b_minus_a"].abs().max()
        ),
        "cqr_closer_to_nominal_than_base_count": int(
            (base_frame["absolute_coverage_error_change"] < 0).sum()
        ),
        "cqr_closer_to_nominal_than_residual_count": int(
            (method_frame["absolute_coverage_error_change"] < 0).sum()
        ),
        "validity_label": config["calibration"]["validity_label"],
        "bootstrap": bootstrap,
        "output_hashes": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in (cell_path, protocol_path, method_path, base_path, figure_path)
        },
    }
    summary_path = output_root / "UQ_REFERENCE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-registers", action="store_true")
    parser.add_argument("--validate-registration", action="store_true")
    parser.add_argument("--run-pending", action="store_true")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--authorize-real-execution", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    output: dict[str, Any] = {}
    if args.write_registers:
        output["registers"] = write_registers(args.root)
    if args.validate_registration:
        output["registration"] = validate_registration(args.root)
    if args.run_pending:
        if not args.authorize_real_execution:
            parser.error("--run-pending requires --authorize-real-execution")
        output["execution"] = run_pending(args.root, max_cells=args.max_cells)
    if args.analyze:
        output["analysis"] = analyze(args.root)
    if not output:
        parser.error("select --write-registers, --validate-registration, --run-pending, or --analyze")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
